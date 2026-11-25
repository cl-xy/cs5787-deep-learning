"""Imagenette reference, FID/TPIPS stages and evaluate_saved_split with stubbed packages."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from team_pipeline import extended_metrics, imagenette
from team_pipeline.data import IMAGENETTE_CLASSES

NAMES = ["original", "member1"]
PAIRS = [[1, 10], [2, 20]]
PROTOCOL = {"splits": {"validation": PAIRS}}
LABELS = [0, 217]


# ---- Imagenette reference ------------------------------------------------------------
def make_source(tmp_path, per_class=4):
    source = tmp_path / "source"
    for label, synset, _ in IMAGENETTE_CLASSES:
        if label in LABELS:
            folder = source / "imagenette2" / "train" / synset
            folder.mkdir(parents=True)
            for i in range(per_class):
                Image.new("RGB", (40, 30), (label % 255, i * 40, 0)).save(folder / f"{i}.JPEG")
    return source


@pytest.fixture
def no_download(monkeypatch):
    fake = types.SimpleNamespace(Imagenette=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", fake)


def imagenette_pairs(per_class=2):
    return [(label, 100 * label + i) for label in LABELS for i in range(per_class)]


def test_imagenette_reference_is_built_balanced_and_reused(tmp_path, no_download, capsys):
    source = make_source(tmp_path)
    folder, description = imagenette.prepare_imagenette_reference(
        source, tmp_path / "prepared", imagenette_pairs(), images_per_class=3, image_size=16)
    assert sorted(p.name for p in folder.glob("*.png")) == [
        f"class_{label:04d}_{i:04d}.png" for label in LABELS for i in range(3)]
    assert "labels=[0, 217]" in description
    with Image.open(next(folder.glob("*.png"))) as image:
        assert image.size == (16, 16) and image.mode == "RGB"
    again, same = imagenette.prepare_imagenette_reference(
        source, tmp_path / "prepared", imagenette_pairs(), images_per_class=3, image_size=16)
    assert (again, same) == (folder, description)
    assert "Reusing" in capsys.readouterr().out


def test_imagenette_reference_rejects_bad_inputs(tmp_path, no_download):
    source = make_source(tmp_path)
    prepare = imagenette.prepare_imagenette_reference
    with pytest.raises(ValueError, match="outside Imagenette"):
        prepare(source, tmp_path / "p", [(5, 1), (5, 2)], images_per_class=2, image_size=16)
    with pytest.raises(ValueError, match="balanced"):
        prepare(source, tmp_path / "p", [(0, 1), (0, 2), (217, 3)], images_per_class=2,
                image_size=16)
    with pytest.raises(ValueError, match="real images"):
        prepare(source, tmp_path / "p", imagenette_pairs(), images_per_class=9, image_size=16)
    with pytest.raises(ValueError, match="images_per_class"):
        prepare(source, tmp_path / "p", imagenette_pairs(), images_per_class=1, image_size=16)


def test_incomplete_cached_reference_is_refused(tmp_path, no_download):
    source = make_source(tmp_path)
    folder, _ = imagenette.prepare_imagenette_reference(
        source, tmp_path / "prepared", imagenette_pairs(), images_per_class=2, image_size=16)
    next(folder.glob("*.png")).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        imagenette.prepare_imagenette_reference(
            source, tmp_path / "prepared", imagenette_pairs(), images_per_class=2, image_size=16)


# ---- FID / TPIPS stages with stubbed packages ----------------------------------------
@pytest.fixture
def stub_metric_packages(monkeypatch):
    class Fid:
        @staticmethod
        def get_files_features(files, model, **kwargs):
            return np.array([[float(len(f)), 1.0] for f in files]) + len(files)

        @staticmethod
        def fid_from_feats(generated, reference):
            return float(abs(generated.mean() - reference.mean()))

    features = types.SimpleNamespace(build_feature_extractor=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "cleanfid", types.SimpleNamespace(fid=Fid, features=features))
    monkeypatch.setitem(sys.modules, "cleanfid.fid", Fid)
    monkeypatch.setitem(sys.modules, "cleanfid.features", features)
    monkeypatch.setattr(extended_metrics.warnings, "warn", lambda *a, **k: None)
    monkeypatch.setattr(extended_metrics, "package_version", lambda name: "stub")

    class Metric:
        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

        def distance(self, ref, image, factor, value_range):
            return torch.tensor(float((ref - image).abs().mean()) + len(factor))

    tpips = types.SimpleNamespace(load_model=lambda *a, **k: Metric())
    monkeypatch.setitem(sys.modules, "tpips", tpips)


def make_reference(tmp_path, count=3):
    folder = tmp_path / "reference"
    folder.mkdir()
    for i in range(count):
        Image.new("RGB", (8, 8), (i, i, i)).save(folder / f"r{i}.png")
    return folder


def evaluated_split(tiny_run):
    from team_pipeline import evaluation
    evaluation.evaluate_split(tiny_run, PROTOCOL, "validation", NAMES, lambda x, y: (x - y).square().mean())
    return tiny_run / "validation"


def test_lpips_results_pass_through_when_extras_are_disabled(tiny_run):
    split = evaluated_split(tiny_run)
    base = pd.read_csv(split / "results.csv")
    summary = extended_metrics.evaluate_fid_tpips(split, NAMES, tpips_factors=())
    pd.testing.assert_frame_equal(summary, base)
    pd.testing.assert_frame_equal(pd.read_csv(split / "results_extended.csv"), base)
    metadata = json.loads((split / "evaluation_extended.json").read_text())
    assert metadata["fid"] is None and metadata["tpips"] is None


def test_fid_and_tpips_columns_and_metadata(tiny_run, stub_metric_packages):
    split = evaluated_split(tiny_run)
    reference = make_reference(tiny_run)
    summary = extended_metrics.evaluate_fid_tpips(
        split, NAMES, fid_reference_dir=reference, fid_reference_description="synthetic real set",
        tpips_factors=("overall", "Shape"), tpips_device="cpu")
    assert summary["fid"].notna().all()
    assert {"tpips_overall_mean", "tpips_shape_std"} <= set(summary.columns)
    paired = pd.read_csv(split / "paired_metrics_extended.csv")
    assert {"tpips_overall", "tpips_shape"} <= set(paired.columns) and len(paired) == 4
    metadata = json.loads((split / "evaluation_extended.json").read_text())
    assert metadata["fid"]["small_sample_pilot"] is True
    assert metadata["fid"]["reference_images"] == 3
    assert metadata["tpips"]["factors"] == {"tpips_overall": "overall", "tpips_shape": "Shape"}
    assert (split / "fid_reference_inventory.json").exists()


@pytest.mark.parametrize("factors", [("a", "a"), ("", "x"), ("A b", "a_b"), ("!!",)])
def test_invalid_tpips_factors_are_rejected(tiny_run, factors):
    split = evaluated_split(tiny_run)
    with pytest.raises(ValueError, match="TPIPS"):
        extended_metrics.evaluate_fid_tpips(split, NAMES, tpips_factors=factors)


def test_fid_reference_must_be_described_and_outside_the_split(tiny_run, stub_metric_packages):
    split = evaluated_split(tiny_run)
    reference = make_reference(tiny_run)
    with pytest.raises(ValueError, match="Describe"):
        extended_metrics.evaluate_fid_tpips(split, NAMES, fid_reference_dir=reference,
                                            tpips_factors=())
    inside = split / "original"
    with pytest.raises(ValueError, match="outside"):
        extended_metrics.evaluate_fid_tpips(split, NAMES, fid_reference_dir=inside,
                                            fid_reference_description="x", tpips_factors=())


def test_artifact_check_uses_shared_consistency_rules(tiny_run):
    split = evaluated_split(tiny_run)
    path = split / "member1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["runtime"] = {"device": "elsewhere"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="environment differs"):
        extended_metrics.evaluate_fid_tpips(split, NAMES, tpips_factors=())


# ---- evaluate_saved_split -----------------------------------------------------------
def test_evaluate_saved_split_runs_lpips_then_extras_without_globals(tiny_run, stub_metric_packages):
    reference = make_reference(tiny_run)
    outcome = extended_metrics.evaluate_saved_split(
        tiny_run, PROTOCOL, "validation", NAMES, fid_reference_dir=reference,
        fid_reference_description="synthetic", tpips_factors=(),
        metric=lambda x, y: (x - y).square().mean())
    assert outcome["results"]["fid"].notna().all()
    assert all(path.exists() for path in outcome["paths"].values())
    assert outcome["paths"]["results"] == tiny_run / "validation/results_extended.csv"


def test_evaluate_saved_split_never_installs_unless_given_an_installer(tiny_run, monkeypatch):
    evaluated_split(tiny_run)
    monkeypatch.setattr(extended_metrics, "package_version", lambda name: "not-installed")
    with pytest.raises(RuntimeError, match="pip install clean-fid"):
        extended_metrics.evaluate_saved_split(
            tiny_run, PROTOCOL, "validation", NAMES, fid_reference_dir=tiny_run,
            fid_reference_description="x")
    installed = []
    monkeypatch.setattr(extended_metrics, "evaluate_fid_tpips", lambda *a, **k: "done")
    outcome = extended_metrics.evaluate_saved_split(
        tiny_run, PROTOCOL, "validation", NAMES, fid_reference_dir=tiny_run,
        fid_reference_description="x", tpips_factors=("overall",), installer=installed.append)
    assert installed == [["clean-fid", "tpips"]] and outcome["results"] == "done"


def test_evaluate_saved_split_rejects_other_splits_and_skips_current_lpips(tiny_run, monkeypatch):
    with pytest.raises(ValueError, match="validation or test"):
        extended_metrics.evaluate_saved_split(tiny_run, PROTOCOL, "calibration", NAMES)
    evaluated_split(tiny_run)
    monkeypatch.setattr(extended_metrics, "evaluate_split",
                        lambda *a, **k: pytest.fail("LPIPS outputs are current"))
    extended_metrics.evaluate_saved_split(tiny_run, PROTOCOL, "validation", NAMES)
