"""Offline regression checks for the code actually executed by the notebook.

Run with: venv/bin/python -m pytest -q
"""
import json
import linecache
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
    pairs = [(207, 10016), (360, 10017)]
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
    notebook.update(RUN_ROOT=run_root, NAMES=names,
                    evaluate_split=lambda *a: pytest.fail("Completed LPIPS rerun"),
                    subprocess=SimpleNamespace(run=lambda *a, **kw: pytest.fail("Optional pip install")),
                    display=lambda *a: None,
                    Image=lambda **kw: kw)
    execute_cell(notebook, 46)
    execute_cell(notebook, 47)
    import pandas as pd
    pd.testing.assert_frame_equal(summary, pd.read_csv(split_root / "results_extended.csv"))
    metadata = json.loads((split_root / "evaluation_extended.json").read_text())
    assert metadata["fid"] is None and metadata["tpips"] is None

    # Same validation must still reject incomplete artifacts when extras are off.
    samples = split_root / "member4_xsa_reuse/samples.csv"
    pd.read_csv(samples).iloc[:1].to_csv(samples, index=False)
    with pytest.raises(ValueError, match="sample order"):
        notebook["evaluate_fid_tpips"](split_root, names, tpips_factors=())
