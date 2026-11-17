"""Report exports must use saved data and preserve historical method labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reports():
    notebook = json.loads((ROOT / "DiT_Layer_Looping_GCP.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == "report-helpers")["source"])
    namespace = {}
    exec(compile(source, "<notebook-report-helpers>", "exec"), namespace)
    return namespace


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / "historical_fixed_alpha"
    split = root / "validation"
    pairs = [[0, 10], [217, 11], [0, 12], [217, 13]]
    config = dict(image_size=8, device="cpu", sampling_steps=50, cfg_scale=4.0, dtype="float32")
    summary, paired = [], []
    methods = [("original", "original"), ("member1_removal", "removal"),
               ("member2_loop", "loop"), ("member3_damped", "damped"),
               ("member4_xsa_reuse", "xsa_reuse")]
    for method_index, (name, mode) in enumerate(methods):
        folder = split / name
        folder.mkdir(parents=True)
        samples = pd.DataFrame([(i, *pair) for i, pair in enumerate(pairs)], columns=["index", "label", "seed"])
        samples["latency_s"] = np.array([1, 1.1, .9, 1.05]) * (0.8 if mode == "removal" else 1)
        samples.to_csv(folder / "samples.csv", index=False)
        rows = samples.copy()
        rows["paired_lpips"] = 0 if name == "original" else np.array([.3, .4, .5, .4]) + method_index * .01
        rows["variant"] = name
        paired.append(rows)
        entries = []
        for slot in range(4):
            if slot == 1 and mode == "removal":
                continue
            replacement = slot == 1 and mode != "original"
            entries.append(dict(slot=slot, source=0 if replacement else slot,
                                scale=.5 if replacement and mode == "damped" else 1,
                                replacement=replacement, xsa=replacement and mode == "xsa_reuse"))
        manifest = dict(config=config, pairs=pairs, shared_code_sha256="old-generation-code",
                        runtime={"device": "cpu"}, schedule=dict(
                            mode=mode, alpha=.5 if mode == "damped" else 1,
                            removed=[] if mode == "original" else [1],
                            depth=4, execution_schedule=entries))
        (folder / "manifest.json").write_text(json.dumps(manifest))
        for index in range(4):
            Image.new("RGB", (8, 8), color=(method_index * 35, index * 45, 40)).save(folder / f"image_{index:04d}.png")
        summary.append(dict(variant=name, images=4, lpips_mean=rows.paired_lpips.mean(),
                            lpips_std=rows.paired_lpips.std(ddof=1), unique_parameters=100 if name == "original" else 80,
                            latency_mean_s=rows.latency_s.mean(), latency_std_s=rows.latency_s.std(ddof=0),
                            checkpoint_MiB=1 if name == "original" else .8,
                            block_evaluations_per_forward=len(entries)))
    summary = pd.DataFrame(summary)
    summary.to_csv(split / "results.csv", index=False)
    pd.concat(paired).to_csv(split / "paired_metrics.csv", index=False)
    summary["fid"] = [100, 105, 120, 108, 125]
    summary.to_csv(split / "results_extended.csv", index=False)
    (split / "evaluation_extended.json").write_text(json.dumps(dict(
        generation_code_sha256="old-generation-code", generation_runtime={"device": "cpu"},
        images_per_variant=4, variants=[name for name, mode in methods],
        fid={"reference_images": 5000, "generated_images": 4, "reference_description": "test real reference"})))
    return root


def test_report_exports_historical_data_without_recomputation(reports, saved_run):
    inputs = {path: path.read_bytes() for path in (saved_run / "validation").rglob("*") if path.is_file()}
    # A new kernel policy must never relabel old fixed-alpha results.
    reports["ALPHA"] = "inverse_k"
    report = reports["load_report_data"](saved_run)
    summary = report["summary"].set_index("variant")
    assert summary.loc["member3_damped", "method"] == "Damped (fixed α=0.5)"
    assert summary.loc["member1_removal", "speedup_vs_original"] == pytest.approx(1.25)
    assert summary.loc["member1_removal", "parameter_reduction_pct"] == pytest.approx(20)
    assert summary["fid"].tolist() == [100, 105, 120, 108, 125]
    for function in ["plot_report_quality", "plot_report_efficiency", "plot_report_per_class",
                     "plot_report_samples", "plot_report_schedules"]:
        paths = reports[function](report)
        assert paths["pdf"].read_bytes().startswith(b"%PDF")
        with Image.open(paths["png"]) as image:
            assert min(image.size) > 300
    assert len(list(report["output"].glob("*.png"))) == 5
    assert len(list(report["output"].glob("*.pdf"))) == 5
    metadata = json.loads((report["output"] / "report_metadata.json").read_text())
    assert metadata["class_counts"] == {"0": 2, "217": 2}
    assert metadata["generation_code_sha256"] == "old-generation-code"
    assert "fixed α=0.5" in metadata["method_labels"]["member3_damped"]
    assert all(path.read_bytes() == data for path, data in inputs.items())


def test_report_handles_missing_fid_and_rejects_bad_provenance(reports, saved_run):
    path = saved_run / "validation/evaluation_extended.json"
    metadata = json.loads(path.read_text())
    metadata["generation_code_sha256"] = "different-generation"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        reports["load_report_data"](saved_run)
    (saved_run / "validation/results_extended.csv").unlink()
    report = reports["load_report_data"](saved_run)
    assert "fid" not in report["summary"]
    assert reports["plot_report_quality"](report)["png"].is_file()
    samples = saved_run / "validation/member2_loop/samples.csv"
    pd.read_csv(samples).iloc[:2].to_csv(samples, index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        reports["load_report_data"](saved_run)
