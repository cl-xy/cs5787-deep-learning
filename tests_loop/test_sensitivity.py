"""Sensitivity screening on a tiny REAL DiT (official source, random weights, CPU)."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from loop_compression import sensitivity as sens
from loop_compression.blocks import LoopedBlocks
from team_pipeline import experiment as exp
from team_pipeline.methods import ScheduledBlocks

ROOT = Path(__file__).resolve().parents[1]
DEPTH = 6


@pytest.fixture(scope="module")
def dit_module():
    repo = Path(os.environ.get("DIT_TEST_REPO", ROOT / "third_party" / "DiT"))
    if not (repo / "models.py").exists():
        pytest.skip("Official DiT source not available")
    exp.configure_dit(repo)
    import models
    return models


def tiny_model(models, seed=0):
    torch.manual_seed(seed)
    model = models.DiT(input_size=32, patch_size=2, hidden_size=32, depth=DEPTH, num_heads=2).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.05)
        # Block 3 becomes an exact identity (zero gates); block 2 gets a large residual.
        for index, scale in [(3, 0.0), (2, 20.0)]:
            layer = model.blocks[index].adaLN_modulation[-1]
            layer.weight.mul_(scale)
            layer.bias.mul_(scale)
    model.blocks = ScheduledBlocks(model.blocks)
    return model.requires_grad_(False)


def fake_vae():
    return SimpleNamespace(decode=lambda z: SimpleNamespace(sample=torch.zeros(1, 3, 16, 16)))


def make_cache(models, pairs=((207, 1), (360, 2))):
    from diffusion import create_diffusion
    model = tiny_model(models)
    return model, sens.build_cache(model, fake_vae(), create_diffusion("4"), list(pairs), 4.0)


def test_cache_is_deterministic_and_covers_every_sampler_step(dit_module):
    _, first = make_cache(dit_module)
    _, second = make_cache(dit_module)
    assert first["x"].shape == (8, 4, 32, 32)
    assert first["step_index"].tolist() == [0, 1, 2, 3] * 2
    assert first["image_index"].tolist() == [0] * 4 + [1] * 4
    assert first["y"].tolist() == [207] * 4 + [360] * 4
    assert first["t"][:4].tolist() == sorted(first["t"][:4].tolist(), reverse=True)
    for key in ("x", "t", "y"):
        assert torch.equal(first[key], second[key])
    # Step 0 of image 0 is exactly the seeded initial noise.
    assert torch.equal(first["x"][0:1], exp.initial_latent(1, "cpu"))


def test_recording_hook_is_removed_and_does_not_change_outputs(dit_module):
    model = tiny_model(dit_module)
    x = torch.randn(2, 4, 32, 32)
    t, y = torch.tensor([500, 500]), torch.tensor([3, 1000])
    expected = model.forward_with_cfg(x, t, y, 4.0)
    store = {"x": [], "t": [], "y": []}
    with sens.record_states(model, store):
        assert torch.equal(model.forward_with_cfg(x, t, y, 4.0), expected)
    assert "forward_with_cfg" not in model.__dict__
    assert len(store["x"]) == 1 and store["x"][0].shape == (1, 4, 32, 32)


def test_state_selection_and_buckets(dit_module):
    _, cache = make_cache(dit_module)
    chosen = sens.select_states(cache, images=1, stride=2, offset=1)
    assert chosen["step_index"].tolist() == [1, 3] and chosen["image_index"].eq(0).all()
    with pytest.raises(ValueError):
        sens.select_states(cache, images=0, stride=2)
    with pytest.raises(ValueError):
        sens.select_states(cache, stride=0)
    buckets, count = sens.assign_buckets(np.array([999, 999, 799, 599, 399, 199]), 3)
    # Distinct timesteps 999,799,599,399,199 split 2/2/1 in sampling order.
    assert count == 3 and buckets.tolist() == [0, 0, 0, 1, 1, 2]
    assert sens.assign_buckets(np.array([5, 5, 5]), 4)[1] == 1  # fewer timesteps than buckets


def make_screen(models, batch=3):
    model, cache = make_cache(models)
    states = sens.select_states(cache, stride=1, offset=0)
    return sens.EpsilonScreen(model, states, "cpu", batch_states=batch)


def test_identity_has_zero_error_and_screen_restores_blocks(dit_module):
    screen = make_screen(dit_module)
    blocks = screen.base_blocks
    original = screen.model.blocks
    identity = LoopedBlocks(blocks, [1, 4], K=1, strategy="naive")
    errors = screen.errors(identity)
    assert errors.eps_err.abs().max() < 1e-10 and errors.var_err.abs().max() < 1e-10
    assert screen.model.blocks is original
    # Batching does not change results.
    other = sens.EpsilonScreen(screen.model, screen.states, "cpu", batch_states=8)
    assert torch.allclose(other.reference, screen.reference, atol=1e-4, rtol=1e-4)


def test_removal_error_ranks_a_large_residual_block_above_an_identity_block(dit_module):
    screen = make_screen(dit_module)
    removal = lambda i: screen.errors(LoopedBlocks(screen.base_blocks, [i, i], kept=[i], K=1,
                                                   strategy="damped", h=0.0)).eps_err.mean()
    assert removal(3) < 1e-10               # zero-gate block is an exact identity
    assert removal(2) > 100 * max(removal(3), 1e-9)
    assert removal(2) > removal(4)


def test_feature_changes_flag_the_identity_and_the_large_block(dit_module):
    screen = make_screen(dit_module)
    l1, bi = screen.feature_changes()
    assert l1.shape == bi.shape == (screen.count, DEPTH)
    assert np.abs(l1[:, 3]).max() < 1e-7 and np.abs(bi[:, 3]).max() < 1e-7
    assert l1[:, 2].mean() > l1[:, 4].mean() and l1[:, 2].mean() > 10 * l1[:, 3].mean() + 1e-3
    assert np.isfinite(l1).all() and (bi >= -1e-6).all()


def test_profile_blocks_table(dit_module, tmp_path):
    screen = make_screen(dit_module)
    buckets, count = sens.assign_buckets(screen.states["t"].numpy(), 2)
    overall, per_bucket = sens.profile_blocks(screen, buckets, count, log=lambda *_: None)
    assert list(overall.index) == list(range(DEPTH))
    for column in ["l1_ratio", "cosine_bi", "removal_eps_err", "loop_h1_eps_err", "loop_h0.5_eps_err"]:
        assert column in overall.columns
    assert np.isnan(overall.loc[0, "loop_h1_eps_err"])
    assert overall.loc[3, "removal_eps_err"] < 1e-10
    assert overall.loc[2, "removal_eps_err"] == overall["removal_eps_err"].max()
    assert set(per_bucket["bucket"]) == {0, 1} and len(per_bucket) == DEPTH * 2
    # Loop sensitivity: the previous (identity) block 3 in place of block 4 only skips block 4.
    assert overall.loc[4, "loop_h1_eps_err"] == pytest.approx(overall.loc[4, "removal_eps_err"], rel=1e-5)
    sens.plot_block_heatmaps(per_bucket, tmp_path / "heatmap.png", 2)
    assert (tmp_path / "heatmap.png").stat().st_size > 0
