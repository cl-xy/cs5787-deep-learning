"""Offline regression checks for the code actually executed by the notebook.

Run with: venv/bin/python -m pytest -q
"""
import json
import linecache
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "DiT_Layer_Looping_GCP.ipynb").read_text())


def execute_cell(namespace, index):
    source = "".join(NOTEBOOK["cells"][index]["source"])
    filename = f"<dit-notebook-cell-{index}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)


@pytest.fixture
def notebook():
    namespace = {}
    for index in [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24, 45]:
        execute_cell(namespace, index)
    return namespace


def test_existing_notebook_contracts(notebook):
    notebook["run_notebook_tests"]()


def test_inverse_k_entry_point_and_checkpoint_compatibility(notebook, tmp_path, monkeypatch):
    """Dynamic damping must survive save/load; old numeric alpha stays fixed."""
    class SmallModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(6)])

    monkeypatch.setitem(sys.modules, "models", SimpleNamespace(DiT_XL_2=SmallModel))
    for alpha in ["inverse_k", 0.5]:
        model = SmallModel()
        model.blocks = notebook["ScheduledBlocks"](model.blocks, [1, 2, 4], "damped", alpha)
        path = tmp_path / f"checkpoint_{alpha}.pt"
        notebook["checkpoint_stats"](model, path, keep=True)
        loaded = notebook["load_compact_checkpoint"](path)
        assert loaded.blocks.metadata() == model.blocks.metadata()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
        assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
        if alpha == 0.5:
            payload = torch.load(path, weights_only=True)
            # Emulate an existing checkpoint before the damping policy was added.
            for key in ["damping_policy", "k_definition"]:
                payload["schedule"].pop(key)
            for entry in payload["schedule"]["execution_schedule"]:
                entry.pop("iteration")
            torch.save(payload, path)
            legacy = notebook["load_compact_checkpoint"](path)
            assert legacy.blocks.metadata()["damping_policy"] == "fixed"
            assert [entry["scale"] for entry in legacy.blocks.metadata()["execution_schedule"]] == [1, .5, .5, 1, .5, 1]

    notebook["write_fixed_selection"](tmp_path, [1, 2, 4])
    calls = []
    monkeypatch.setitem(notebook, "RUN_ARGS", object())
    monkeypatch.setitem(notebook, "KEEP_CHECKPOINTS", False)
    monkeypatch.setitem(notebook, "load_run", lambda args: (tmp_path, {"splits": {"validation": [[0, 10]]}}, {}))
    monkeypatch.setitem(notebook, "run_variant", lambda *args, **kwargs: calls.append(kwargs))
    notebook["run_method"]("validation", "damped", "new_damping")
    assert calls[-1]["alpha"] == "inverse_k"
    assert calls[-1]["method_metadata"]["damping_policy"] == "inverse_k"
    assert calls[-1]["method_metadata"]["replacement_residual_scale"] is None
    notebook["run_method"]("validation", "damped", "legacy_damping", alpha=0.5)
    assert calls[-1]["method_metadata"]["damping_policy"] == "fixed"


def test_imagenette_class_array_and_balanced_protocol(notebook):
    from torchvision.models import ResNet50_Weights

    expected = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
    assert notebook["CLASS_LABELS"] == expected
    categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    for label, synset, name in notebook["IMAGENETTE_CLASSES"]:
        assert categories[label] == name
        assert synset.startswith("n") and len(synset) == 9
    protocol = notebook["load_protocol"](notebook["PROTOCOL"])
    seeds = set()
    for split, count in [("calibration", 2), ("validation", 2), ("test", 500)]:
        pairs = protocol["splits"][split]
        assert Counter(label for label, seed in pairs) == dict.fromkeys(expected, count)
        split_seeds = {seed for label, seed in pairs}
        assert not seeds & split_seeds
        seeds.update(split_seeds)
    assert {label for label, seed in notebook["make_protocol"]()["splits"]["test"]} == set(expected)


def test_imagenette_reference_selection_and_cache(notebook, tmp_path, monkeypatch):
    from PIL import Image
    import torchvision.datasets

    downloads = []

    def local_imagenette(root, split, size, download):
        downloads.append((split, size, download))
        # Different native aspect ratios and grayscale exercise RGB/crop handling.
        for label, synset, name in notebook["IMAGENETTE_CLASSES"]:
            folder = Path(root) / "imagenette2" / "train" / synset
            folder.mkdir(parents=True)
            for i in range(3):
                Image.new("L", (12 + i, 20 - i), color=i * 60).save(folder / f"{i}.JPEG")

    monkeypatch.setattr(torchvision.datasets, "Imagenette", local_imagenette)
    pairs = [(label, 100 + i) for i, label in enumerate(notebook["CLASS_LABELS"])]
    prepare = notebook["prepare_imagenette_reference"]
    output, description = prepare(tmp_path / "source", tmp_path / "prepared", pairs,
                                  images_per_class=2, image_size=8, seed=42)
    assert downloads == [("train", "full", True)]
    assert len(list(output.glob("*.png"))) == 20
    for path in output.glob("*.png"):
        with Image.open(path) as image:
            assert image.mode == "RGB" and image.size == (8, 8)
    manifest = json.loads((output / "reference_manifest.json").read_text())
    assert Counter(row["label"] for row in manifest["inventory"]) == dict.fromkeys(notebook["CLASS_LABELS"], 2)
    assert len({row["source"] for row in manifest["inventory"]}) == 20
    assert manifest["settings"]["source_split"] == "train"
    assert "seed=42" in description and "8x8" in description
    assert not (output / "imagenette2").exists()
    # A completed reference avoids downloading and processing again.
    assert prepare(tmp_path / "source", tmp_path / "prepared", pairs,
                   images_per_class=2, image_size=8, seed=42) == (output, description)
    assert len(downloads) == 1
    # Check input labels/balance before any network or image access.
    with pytest.raises(ValueError, match="outside Imagenette"):
        prepare(tmp_path / "unused", tmp_path / "prepared", [(207, 100)])
    with pytest.raises(ValueError, match="balanced"):
        prepare(tmp_path / "unused", tmp_path / "prepared", [*pairs, (0, 200)])
    with pytest.raises(ValueError, match="integer >= 2"):
        prepare(tmp_path / "unused", tmp_path / "prepared", pairs, images_per_class=1)
    next(output.glob("*.png")).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        prepare(tmp_path / "source", tmp_path / "prepared", pairs,
                images_per_class=2, image_size=8, seed=42)
    assert len(downloads) == 1


def test_large_run_preview_is_bounded(notebook, tmp_path, monkeypatch):
    """Every sample is saved; only 16 decoded images enter the preview grid."""
    import torchvision.utils

    vae = nn.Linear(1, 1)
    vae.config = SimpleNamespace()
    monkeypatch.setitem(notebook, "_prepare", lambda *a: (notebook["Tiny"](), vae, None))
    monkeypatch.setitem(notebook, "warmup", lambda *a: None)
    monkeypatch.setitem(notebook, "generate", lambda *a: torch.zeros(1, 3, 8, 8))
    saved_batches = []
    save_image = torchvision.utils.save_image

    def capture_grid(images, path, **kwargs):
        if Path(path).name == "grid.png":
            saved_batches.append(len(images))
        return save_image(images, path, **kwargs)

    monkeypatch.setattr(torchvision.utils, "save_image", capture_grid)
    pairs = [(0, 100 + i) for i in range(19)]
    result = notebook["run_variant"](
        dict(device="cpu", image_size=8, cfg_scale=4), pairs, tmp_path / "original")
    assert len(result) == 19
    assert len(list((tmp_path / "original").glob("image_*.pt"))) == 19
    assert len(list((tmp_path / "original").glob("image_*.png"))) == 19
    assert saved_batches == [16]


def test_cpu_configuration_and_resume(notebook, tmp_path, monkeypatch):
    """No CUDA calls, downloads or checkout changes are needed to configure CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: pytest.fail("CUDA called"))
    subprocess = SimpleNamespace(run=lambda *a, **kw: None)
    notebook.update(WORKSPACE=ROOT, OUTPUT_BASE=tmp_path, subprocess=subprocess)
    execute_cell(notebook, 28)
    assert notebook["RUN_ARGS"].device == "cpu"
    assert notebook["RUN_ARGS"].output == notebook["RUN_ROOT"]
    assert notebook["NAMES"][-1] == "member4_xsa_reuse"

    notebook.update(load_run=lambda args: (args.output, notebook["PROTOCOL"], {}),
                    calibrate=lambda *a: pytest.fail("Unexpected calibration inference"))
    execute_cell(notebook, 32)
    assert notebook["selection"]["removed"] == [4, 7]
    # Re-entering selection reuses the recorded choice without profiling.
    execute_cell(notebook, 32)

    # Resume uses the recorded device/checkpoint and keeps both run paths aligned.
    root = tmp_path / "saved"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"device": "cpu", "checkpoint": "saved.pt"}))
    source = "".join(NOTEBOOK["cells"][28]["source"])
    source = source.replace("RESUME_RUN = None", f"RESUME_RUN = {str(root)!r}")
    exec(compile(source, "<resume-configuration>", "exec"), notebook)
    assert notebook["RUN_ROOT"] == notebook["RUN_ARGS"].output == root
    assert notebook["RUN_ARGS"].checkpoint == Path("saved.pt")


def test_real_sampler_and_lpips_only_evaluation(notebook, tmp_path, monkeypatch):
    """Exercise all five schedules with real DiT/diffusion, without pretrained weights."""
    # Keep the pinned third-party checkout clean during the smoke test.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(ROOT / "third_party/DiT"))
    import models

    def small_dit(**kwargs):
        return models.DiT(hidden_size=24, depth=8, num_heads=3, **kwargs)

    monkeypatch.setattr(models, "DiT_XL_2", small_dit)
    with torch.random.fork_rng():
        torch.manual_seed(24)
        model = small_dit(input_size=32, num_classes=1000)
        # Nonzero adaLN gates/output make this a real attention/sampler exercise.
        with torch.no_grad():
            for block in model.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.01)
            nn.init.normal_(model.final_layer.linear.weight, std=0.01)
        checkpoint = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), checkpoint)
    del model

    class Decoder(nn.Module):
        config = SimpleNamespace()

        def decode(self, latent):
            return SimpleNamespace(sample=F.interpolate(latent[:, :3], size=256).tanh())

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoencoderKL=SimpleNamespace(from_pretrained=lambda *a, **kw: Decoder())))
    config = dict(device="cpu", image_size=256, cfg_scale=4.0,
                  checkpoint=str(checkpoint), vae_id="offline-decoder", sampling_steps=2)
    pairs = [(0, 10016), (217, 10017)]
    names = ["original", "member1_removal", "member2_loop", "member3_damped", "member4_xsa_reuse"]
    run_root = tmp_path / "run"
    split_root = run_root / "validation"
    for name, mode in zip(names, ["original", "removal", "loop", "damped", "xsa_reuse"]):
        notebook["run_variant"](config, pairs, split_root / name, mode=mode,
                                removed=() if mode == "original" else (4, 7),
                                alpha=0.5 if mode == "damped" else 1.0)
    # Inject a cheap paired distance; LPIPS weights are deliberately not downloaded.
    summary = notebook["compare_results"](
        split_root, names, metric=lambda x, y: (x - y).square().mean())
    assert summary.images.eq(2).all()
    assert summary.loc[0, "lpips_mean"] == 0
    notebook["comparison_grid"](split_root, names, pairs)

    # Disabled extras must never import/install TPIPS or Clean-FID.
    monkeypatch.setitem(sys.modules, "tpips", None)
    monkeypatch.setitem(sys.modules, "cleanfid", None)
    notebook.update(RUN_ROOT=run_root, NAMES=names, WORKSPACE=tmp_path,
                    evaluate_split=lambda *a: pytest.fail("Completed LPIPS rerun"),
                    subprocess=SimpleNamespace(run=lambda *a, **kw: pytest.fail("Optional pip install")),
                    display=lambda *a: None,
                    Image=lambda **kw: kw)
    execute_cell(notebook, 46)
    notebook["USE_IMAGENETTE_REFERENCE"] = False
    execute_cell(notebook, 47)
    import pandas as pd
    pd.testing.assert_frame_equal(summary, pd.read_csv(split_root / "results_extended.csv"))
    metadata = json.loads((split_root / "evaluation_extended.json").read_text())
    assert metadata["fid"] is None and metadata["tpips"] is None

    # Route the same completed generation through Imagenette FID, still offline.
    # Stub the external Inception implementation while using actual image export,
    # manifests, class selection, feature-file lists and result serialization.
    from PIL import Image as PILImage
    import numpy as np
    import torchvision.datasets

    def local_reference(root, **kwargs):
        mapping = {label: synset for label, synset, name in notebook["IMAGENETTE_CLASSES"]}
        for label in [0, 217]:
            folder = Path(root) / "imagenette2/train" / mapping[label]
            folder.mkdir(parents=True)
            for i in range(3):
                PILImage.new("RGB", (12, 16), color=(i * 60, 20, 30)).save(folder / f"{i}.JPEG")

    monkeypatch.setattr(torchvision.datasets, "Imagenette", local_reference)
    feature_calls = []

    def features(files, **kwargs):
        feature_calls.append(files)
        return np.array([[float(i), float(i + 1)] for i in range(len(files))])

    fake_fid = SimpleNamespace(
        get_files_features=features,
        fid_from_feats=lambda a, b: float(np.square(a.mean(0) - b.mean(0)).sum()))
    monkeypatch.setitem(sys.modules, "cleanfid", SimpleNamespace(fid=fake_fid))
    monkeypatch.setitem(sys.modules, "cleanfid.features", SimpleNamespace(
        build_feature_extractor=lambda *a, **kw: object()))
    monkeypatch.setitem(notebook, "_extra_version", lambda name: "test-installed")
    notebook.update(USE_IMAGENETTE_REFERENCE=True, FID_REAL_IMAGES_PER_CLASS=2)
    with pytest.warns(UserWarning, match="Small-sample FID"):
        result = notebook["evaluate_saved_split"]("validation")
    assert result["fid"].notna().all()
    assert len(feature_calls) == 6  # One real reference extraction, five methods.
    assert len(feature_calls[0]) == 4
    assert all(Path(p).name.startswith("class_") for p in feature_calls[0])
    assert all(Path(files[0]).name == "image_0000.png" for files in feature_calls[1:])
    metadata = json.loads((split_root / "evaluation_extended.json").read_text())
    assert metadata["fid"]["reference_images"] == 4
    assert "Imagenette" in metadata["fid"]["reference_description"]
    assert metadata["tpips"] is None

    # Same validation must still reject incomplete artifacts when extras are off.
    samples = split_root / "member4_xsa_reuse/samples.csv"
    pd.read_csv(samples).iloc[:1].to_csv(samples, index=False)
    with pytest.raises(ValueError, match="sample order"):
        notebook["evaluate_fid_tpips"](split_root, names, tpips_factors=())
