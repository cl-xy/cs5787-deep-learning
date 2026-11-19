"""Report tables and figures from a saved synthetic run."""
import json
import os

import matplotlib
import pandas as pd
import pytest

from team_pipeline import evaluation, report

NAMES = ["original", "member1"]
PROTOCOL = {"splits": {"validation": [[1, 10], [2, 20]]}}


@pytest.fixture
def evaluated(tiny_run):
    evaluation.evaluate_split(tiny_run, PROTOCOL, "validation", NAMES,
                              lambda x, y: (x - y).square().mean())
    return tiny_run


def add_fid(run, **overrides):
    split = run / "validation"
    results = pd.read_csv(split / "results.csv")
    results["fid"] = [1.5, 2.5]
    results.to_csv(split / "results_extended.csv", index=False)
    manifest = json.loads((split / "original/manifest.json").read_text())
    metadata = dict(generation_code_sha256=manifest["shared_code_sha256"],
                    generation_runtime=manifest["runtime"], images_per_variant=2,
                    variants=NAMES, fid=dict(reference_images=3))
    metadata.update(overrides)
    (split / "evaluation_extended.json").write_text(json.dumps(metadata))


def test_report_tables_are_derived_from_saved_artifacts(evaluated):
    data = report.load_report_data(evaluated, "validation")
    summary = data["summary"].set_index("variant")
    assert summary.loc["original", "method"] == "Original DiT"
    assert summary.loc["member1", "method"] == "Removal"
    assert summary.loc["original", "speedup_vs_original"] == pytest.approx(1.0)
    assert summary.loc["member1", "parameter_reduction_pct"] > 0
    assert summary.loc["member1", "damping_policy"] == "none"
    output = evaluated / "report/validation"
    assert {"report_summary.csv", "per_class_lpips.csv", "report_metadata.json"} <= {
        p.name for p in output.iterdir()}
    metadata = json.loads((output / "report_metadata.json").read_text())
    assert metadata["images_per_variant"] == 2 and metadata["labels"] == [1, 2]
    assert set(metadata["input_sha256"]) >= {"validation/results.csv",
                                             "validation/original/manifest.json"}
    assert set(data["per_class"]["label"]) == {1, 2}


def test_every_figure_is_written_without_changing_the_matplotlib_backend(evaluated):
    backend = matplotlib.get_backend()
    data = report.load_report_data(evaluated, "validation")
    for name, plot in report.REPORT_PLOTS.items():
        paths = plot(data)
        assert paths is not None, name
        assert paths["png"].read_bytes()[:4] == b"\x89PNG", name
        assert paths["pdf"].read_bytes()[:4] == b"%PDF", name
    assert matplotlib.get_backend() == backend


def test_fid_is_used_only_with_matching_provenance(evaluated):
    add_fid(evaluated)
    data = report.load_report_data(evaluated, "validation")
    assert data["summary"]["fid"].tolist() == [1.5, 2.5]
    assert report.plot_report_quality(data)["png"].exists()
    add_fid(evaluated, generation_code_sha256="0" * 64)
    with pytest.raises(ValueError, match="provenance"):
        report.load_report_data(evaluated, "validation")


def test_missing_fid_is_reported_as_unavailable(evaluated):
    data = report.load_report_data(evaluated, "validation")
    assert "fid" not in data["summary"]
    assert report.plot_report_quality(data)["png"].exists()


def test_invalid_inputs_are_rejected(evaluated):
    with pytest.raises(ValueError, match="validation or test"):
        report.load_report_data(evaluated, "calibration")
    with pytest.raises(ValueError, match="including original"):
        report.load_report_data(evaluated, "validation", names=["member1"])
    with pytest.raises(ValueError, match="single folder"):
        report.load_report_data(evaluated, "validation", names=["original", "../x"])
    data = report.load_report_data(evaluated, "validation")
    with pytest.raises(ValueError, match="positive integer"):
        report.plot_report_samples(data, 0)


def test_manifest_mismatch_and_tampered_measurements_are_rejected(evaluated):
    split = evaluated / "validation"
    path = split / "member1/manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(dict(manifest, runtime={"device": "elsewhere"})))
    with pytest.raises(ValueError, match="environment differs"):
        report.load_report_data(evaluated, "validation")
    path.write_text(json.dumps(manifest))
    summary = pd.read_csv(split / "results.csv")
    summary.loc[1, "lpips_mean"] += 1
    summary.to_csv(split / "results.csv", index=False)
    with pytest.raises(ValueError, match="Summary means"):
        report.load_report_data(evaluated, "validation")


def test_find_report_run_prefers_current_then_newest(tmp_path):
    def completed(name, mtime):
        results = tmp_path / name / "validation" / "results.csv"
        results.parent.mkdir(parents=True)
        results.write_text("variant\n")
        os.utime(results, ns=(mtime, mtime))
        return tmp_path / name

    old, new = completed("old", 1_000_000_000), completed("new", 2_000_000_000)
    assert report.find_report_run(tmp_path) == new
    assert report.find_report_run(tmp_path, current=old) == old
    assert report.find_report_run(tmp_path, current=tmp_path / "missing") == new
    with pytest.raises(FileNotFoundError):
        report.find_report_run(tmp_path / "empty")
