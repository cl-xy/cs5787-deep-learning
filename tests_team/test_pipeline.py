"""Shared protocol and end-to-end artifact checks; no pretrained downloads."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from team_pipeline import experiment as exp
from team_pipeline.data import (initial_latent, load_protocol, make_protocol,
                                preprocess_image, protocol_hash, validate_pairs)


def test_protocol_has_balanced_disjoint_splits_and_stable_hash(tmp_path):
    protocol = make_protocol(labels=[1, 2], calibration_per_class=2,
                             validation_per_class=1, test_per_class=3)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    assert load_protocol(path) == protocol
    assert [len(v) for v in protocol["splits"].values()] == [4, 2, 6]
    reordered = dict(reversed(list(protocol.items())))
    assert protocol_hash(protocol) == protocol_hash(reordered)
    changed = copy.deepcopy(protocol)
    changed["splits"]["test"][0][1] = changed["splits"]["calibration"][0][1]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="leakage"):
        load_protocol(path)


@pytest.mark.parametrize("pairs", [[], [(1, 3), (2, 3)], [(1000, 3)], [(1.5, 3)], [(True, 3)]])
def test_input_validation_rejects_ambiguous_samples(pairs):
    with pytest.raises(ValueError):
        validate_pairs(pairs)


def test_preprocessing_and_initial_noise_are_fixed():
    before = torch.get_rng_state().clone()
    a = initial_latent(20, "cpu")
    torch.randn(5)
    assert torch.equal(a, initial_latent(20, "cpu"))
    torch.set_rng_state(before)
    x = torch.zeros(1, 3, 8, 8, dtype=torch.float64, requires_grad=True)
    y = preprocess_image(x, 8)
    assert y.dtype == torch.float32 and y.device.type == "cpu" and not y.requires_grad
    for invalid in [torch.full_like(x, float("nan")), x + 2, x.to(torch.uint8), x.squeeze(0)]:
        with pytest.raises(ValueError):
            preprocess_image(invalid, 8)
    with pytest.raises(ValueError, match="size"):
        preprocess_image(x, 16)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = exp.ScheduledBlocks([nn.Linear(1, 1), nn.Linear(1, 1)])


def test_shared_runner_and_evaluator_reject_incomplete_or_mismatched_runs(tmp_path, monkeypatch):
    # The actual save/load, manifest, timing and evaluation code runs unchanged.
    vae = nn.Linear(1, 1)
    vae.config = SimpleNamespace()
    monkeypatch.setattr(exp, "_prepare", lambda *a: (Tiny(), vae, None))
    monkeypatch.setattr(exp, "warmup", lambda *a: None)
    monkeypatch.setattr(exp, "generate", lambda model, vae, diffusion, label, seed, cfg:
                        torch.full((1, 3, 8, 8), seed / 100))
    config = dict(device="cpu", image_size=8, cfg_scale=4)
    pairs = [(1, 10), (2, 20)]
    for name in ["original", "member1"]:
        exp.run_variant(config, pairs, tmp_path / name)
    metric = lambda x, y: (x - y).square().mean()
    result = exp.compare_results(tmp_path, ["original", "member1"], metric=metric)
    assert result.lpips_mean.eq(0).all()
    assert result.images.eq(2).all()
    assert result.peak_GPU_allocated_GiB.isna().all()
    assert result.parameter_reduction_pct.eq(0).all()
    assert result.checkpoint_reduction_pct.eq(0).all()
    assert result.latency_speedup_x.gt(0).all()
    assert result.throughput_images_per_s.gt(0).all()
    assert (tmp_path / "paired_metrics.csv").exists()
    with pytest.raises(FileExistsError):
        exp.run_variant(config, pairs, tmp_path / "member1")
    samples_path = tmp_path / "member1/samples.csv"
    samples = pd.read_csv(samples_path)
    samples.iloc[:1].to_csv(samples_path, index=False)
    with pytest.raises(ValueError, match="sample order"):
        exp.compare_results(tmp_path, ["member1"], metric=metric)
    samples.to_csv(samples_path, index=False)
    path = tmp_path / "member1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["cfg_scale"] = 5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration"):
        exp.compare_results(tmp_path, ["member1"], metric=metric)


def test_custom_builder_keeps_shared_preparation_and_freezes_model(monkeypatch):
    import sys
    import types
    monkeypatch.setattr(exp, "load_model", lambda *a: Tiny())
    vae = nn.Linear(1, 1)
    factory = types.SimpleNamespace(from_pretrained=lambda *a, **k: vae)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(AutoencoderKL=factory))
    monkeypatch.setitem(sys.modules, "diffusion", types.SimpleNamespace(create_diffusion=lambda *a: "sampler"))
    seen = []
    def builder(model):
        seen.append(model)
        model.blocks = exp.ScheduledBlocks(model.blocks.retained.values(), [1], "damped", .25)
        return model
    model, loaded_vae, diffusion = exp._prepare(dict(device="cpu", checkpoint="unused",
        vae_id="unused", sampling_steps=50), "original", (), .5, builder)
    assert seen == [model] and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert len(model.blocks.retained) == 1
    assert loaded_vae is vae and diffusion == "sampler"


def test_dit_flop_estimate_tracks_effective_schedule():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = SimpleNamespace(fc1=SimpleNamespace(out_features=16))

        def forward(self, x, c):
            return x

    class Model(nn.Module):
        def __init__(self, mode="original", alpha=.5):
            super().__init__()
            self.in_channels, self.out_channels, self.num_heads = 4, 8, 2
            self.pos_embed = nn.Parameter(torch.zeros(1, 4, 8))
            self.x_embedder = SimpleNamespace(num_patches=4, patch_size=(2, 2))
            removed = () if mode == "original" else (1,)
            self.blocks = exp.ScheduledBlocks([Block(), Block(), Block()], removed, mode, alpha)

    original = Model()
    removal = Model("removal")
    damped_zero = Model("damped", 0)
    full = exp.estimate_dit_flops(original, sampling_steps=5)
    reduced = exp.estimate_dit_flops(removal, sampling_steps=5)
    no_op = exp.estimate_dit_flops(damped_zero, sampling_steps=5)
    assert full["dit_sampling_flops"] == 5 * full["dit_forward_flops"]
    assert reduced["dit_sampling_flops"] < full["dit_sampling_flops"]
    assert no_op["dit_sampling_flops"] == reduced["dit_sampling_flops"]
    with pytest.raises(ValueError, match="positive"):
        exp.estimate_dit_flops(original, sampling_steps=0)


def test_mac_notebook_bootstrap_ships_identical_shared_files():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ["DiT_Layer_Looping_Mac.ipynb"]:
        notebook = json.loads((root / name).read_text())
        found = False
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            # Compile every cell, including the updated protocol settings.
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FILES" for t in node.targets):
                    files = ast.literal_eval(node.value)
                    for path in (root / "team_pipeline").glob("*.py"):
                        assert files[str(path.relative_to(root))] == path.read_text()
                    assert files["team_protocol.json"] == (root / "team_protocol.json").read_text()
                    found = True
        assert found
