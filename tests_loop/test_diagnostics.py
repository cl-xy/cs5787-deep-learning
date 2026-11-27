"""Loop diagnostics on a tiny real DiT (official source, random weights, CPU)."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from loop_compression import diagnostics as diag
from loop_compression import sensitivity as sens
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


def tiny_model(models):
    torch.manual_seed(0)
    model = models.DiT(input_size=32, patch_size=2, hidden_size=32, depth=DEPTH, num_heads=2).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        layer = model.blocks[3].adaLN_modulation[-1]       # zero gates: block 3 is an exact identity
        layer.weight.zero_()
        layer.bias.zero_()
    model.blocks = ScheduledBlocks(model.blocks)
    return model.requires_grad_(False)


def make_screen(models):
    from diffusion import create_diffusion
    model = tiny_model(models)
    fake_vae = SimpleNamespace(decode=lambda z: SimpleNamespace(sample=torch.zeros(1, 3, 16, 16)))
    cache = sens.build_cache(model, fake_vae, create_diffusion("4"), [(207, 1), (360, 2)], 4.0)
    states = sens.select_states(cache, stride=1, offset=0)
    return sens.EpsilonScreen(model, states, "cpu", batch_states=3)


def test_identity_unit_does_not_move(dit_module):
    screen = make_screen(dit_module)
    x, t, y = screen._batch(0)
    z, c = diag.window_inputs(screen.model, screen.base_blocks, 3, x, t, y)
    result = diag.unit_trajectory([screen.base_blocks[3]], z, c, 1.0, 3)
    assert result["rel_step"].abs().max() < 1e-6
    assert torch.allclose(result["cos_prev"], torch.ones_like(result["cos_prev"]), atol=1e-6)
    assert torch.allclose(result["norm_ratio"], torch.ones_like(result["norm_ratio"]), atol=1e-6)


def test_damped_first_step_is_h_times_the_naive_step(dit_module):
    screen = make_screen(dit_module)
    x, t, y = screen._batch(0)
    z, c = diag.window_inputs(screen.model, screen.base_blocks, 2, x, t, y)
    unit = [screen.base_blocks[2], screen.base_blocks[4]]
    naive = diag.unit_trajectory(unit, z, c, 1.0, 2)
    damped = diag.unit_trajectory(unit, z, c, 0.5, 2)
    assert naive["rel_step"][0].min() > 0
    assert torch.allclose(damped["rel_step"][0], 0.5 * naive["rel_step"][0], rtol=1e-4)
    zero = diag.unit_trajectory(unit, z, c, 0.0, 2)
    assert zero["rel_step"].abs().max() == 0 and torch.allclose(zero["cos_in"], torch.ones_like(zero["cos_in"]))


def test_window_inputs_compose_with_the_block_chain(dit_module):
    screen = make_screen(dit_module)
    x, t, y = screen._batch(0)
    z0, c0 = diag.window_inputs(screen.model, screen.base_blocks, 0, x, t, y)
    z2, c2 = diag.window_inputs(screen.model, screen.base_blocks, 2, x, t, y)
    for block in screen.base_blocks[:2]:
        z0 = block(z0, c0)
    assert torch.allclose(z0, z2, atol=1e-6) and torch.equal(c0, c2)
    assert len(z2) == 2 * len(x)  # conditional and unconditional halves


def test_trajectories_shapes_and_tables(dit_module):
    screen = make_screen(dit_module)
    traj = diag.trajectories(screen, [1, 4], [2, 3], kmax=3, h_values=(0.5,))
    assert set(traj) == {("naive", 1.0), ("damped", 0.5)}
    assert traj[("naive", 1.0)]["rel_step"].shape == (3, screen.count)
    buckets, count = sens.assign_buckets(screen.states["t"].numpy(), 2)
    eps = diag.eps_by_k(screen, [1, 4], [2, 3], 2, [("naive", 1.0)], buckets, count)
    assert eps[("naive", 1.0)].shape == (2, count) and np.isfinite(eps[("naive", 1.0)]).all()
    from loop_compression.blocks import LoopedBlocks
    for strategy, h in [("naive", 1.0), ("damped", 0.5)]:
        fast = diag.eps_by_k(screen, [1, 4], [2, 3], 2, [(strategy, h)], buckets, count)[(strategy, h)]
        for k in (1, 2):
            full = screen.errors(LoopedBlocks(screen.base_blocks, [1, 4], kept=[2, 3], K=k,
                                              strategy=strategy, h=h)).eps_err.to_numpy()
            expected = [full[buckets == b].mean() for b in range(count)]
            assert np.allclose(fast[k - 1], expected, rtol=1e-3, atol=1e-8), (strategy, k)
    table = diag.bucket_table("w", traj, buckets, count)
    assert len(table) == 2 * count * 3 and {"norm_ratio", "cos_prev", "cos_in", "rel_step"} <= set(table.columns)
    ref = diag.block_norm_reference(screen)
    assert ref.shape == (screen.count, DEPTH + 1) and np.allclose(ref[:, 0], 1.0)
    assert np.allclose(ref[:, 4], ref[:, 3], rtol=1e-5)  # block 3 is an identity


def test_onsets_detect_saturation_drift_and_degradation():
    ks = np.arange(1, 7)
    table = pd.DataFrame(dict(
        window="w", strategy="naive", h=1.0, bucket=0, k=ks,
        rel_step=[0.5, 0.1, 0.01, 0.005, 0.001, 0.001],
        norm_ratio=[1.0, 1.1, 1.2, 1.4, 1.8, 2.5],
        cos_in=[0.99, 0.95, 0.9, 0.85, 0.7, 0.5],
        eps_err=[3.0, 1.0, 1.2, 2.5, 4.0, 8.0]))
    row = diag.onsets(table).iloc[0]
    assert row.saturation_k == 3          # first k >= 2 with rel_step < 0.02
    assert row.drift_k == 4               # |norm - 1| = 0.4 > 0.25 at k = 4
    assert row.best_k == 2 and row.degrades_k == 4  # 2.5 > 2 x 1.0
    never = table.assign(rel_step=0.5, norm_ratio=1.0, cos_in=1.0, eps_err=[1, 1, 1, 1, 1, 1.5])
    row = diag.onsets(never).iloc[0]
    assert pd.isna(row.saturation_k) and pd.isna(row.drift_k) and pd.isna(row.degrades_k)


def test_windows_from_frozen(tmp_path):
    import json
    entry = lambda w, k: dict(loop=dict(settings=dict(window=w, kept=k)))
    (tmp_path / "frozen_configs.json").write_text(json.dumps(
        {"2": entry([16, 21], [17, 18]), "4": entry([16, 21], [17, 18]), "6": entry([7, 14], [10, 11])}))
    assert len(diag.windows_from_frozen(tmp_path)) == 2
    assert list(diag.windows_from_frozen(tmp_path, [6])) == ["7-14_kept10-11"]


def test_onsets_report_the_euler_time_of_the_best_loop_count():
    ks = np.arange(1, 5)
    table = pd.DataFrame(dict(window="w", strategy="damped", h=0.5, bucket=0, k=ks, euler_time=0.5 * ks,
                              rel_step=0.5, norm_ratio=1.0, cos_in=1.0, eps_err=[3.0, 2.0, 1.0, 2.0]))
    row = diag.onsets(table).iloc[0]
    assert row.best_k == 3 and row.best_euler_time == 1.5
