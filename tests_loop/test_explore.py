"""Screening helpers on a tiny real DiT (official source, random weights, CPU)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from loop_compression import explore
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


def make_screen(models):
    from diffusion import create_diffusion
    torch.manual_seed(0)
    model = models.DiT(input_size=32, patch_size=2, hidden_size=32, depth=DEPTH, num_heads=2).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.05)
    model.blocks = ScheduledBlocks(model.blocks)
    model.requires_grad_(False)
    vae = SimpleNamespace(decode=lambda z: SimpleNamespace(sample=torch.zeros(1, 3, 16, 16)))
    cache = sens.build_cache(model, vae, create_diffusion("4"), [(207, 1), (360, 2)], 4.0)
    states = sens.select_states(cache, stride=1, offset=0)
    return sens.EpsilonScreen(model, states, "cpu", batch_states=4), states


def test_gate_grid_contains_the_baselines_and_the_random_control():
    configs = explore.gate_configs({"16-21": ([16, 21], [18, 19])}, loop_counts=(2,), fractions=(0.5,))
    groups = [c["group"] for c in configs]
    assert groups.count("removal") == 1 and groups.count("ungated") == 1
    gated = [c for c in configs if c["group"] == "gated"]
    assert {c["extra"]["signal"] for c in gated} == {"update", "grad", "random"}
    assert {c["extra"]["select"] for c in gated if c["extra"]["signal"] == "random"} == {"high"}
    assert len({sens.settings_id(c["settings"]) for c in configs}) == len(configs)


def test_quality_configs_remove_nothing_and_leave_the_last_block():
    configs = explore.quality_configs(28)
    assert configs
    for c in configs:
        s = c["settings"]
        assert s["kept"] == list(range(s["window"][0], s["window"][1] + 1)) and s["window"][1] < 27
        assert s["strategy"] == "damped" and 0 < s["h"] < 1
        assert c["extra"]["euler_time"] == pytest.approx(s["K"] * s["h"])
    assert {c["extra"]["euler_time"] for c in configs} >= {1.0, 1.5}


def test_score_configs_is_resumable_and_reports_ratios(dit_module, tmp_path):
    screen, states = make_screen(dit_module)
    buckets, count = sens.assign_buckets(states["t"].numpy(), 2)
    configs = explore.gate_configs({"1-4": ([1, 4], [2, 3])}, loop_counts=(2,), fractions=(0.5,),
                                   signals=("update",))
    lines = []
    frame = explore.score_configs(screen, configs, buckets, count, tmp_path / "gate.csv", log=lines.append)
    assert len(frame) == len(configs) and np.isfinite(frame.eps_err).all()
    ungated = frame[frame.group == "ungated"].iloc[0]
    gated = frame[frame.group == "gated"]
    assert (gated.flops_ratio == ungated.flops_ratio).all()           # masked loop: FLOPs not reduced
    assert (gated.est_pruned_block_calls < ungated.est_pruned_block_calls).all()
    assert frame[frame.group == "removal"].params_ratio.iloc[0] == pytest.approx(ungated.params_ratio)
    again = explore.score_configs(screen, configs, buckets, count, tmp_path / "gate.csv", log=lines.append)
    assert len(again) == len(configs) and "0 to run" in lines[-1]
    explore.plot_gate(frame, tmp_path / "gate.png")
    assert (tmp_path / "gate.png").exists()


def test_policy_run_matches_the_per_rule_screening(dit_module, tmp_path):
    screen, states = make_screen(dit_module)
    buckets, count = sens.assign_buckets(states["t"].numpy(), 2)
    t_values = sorted({int(t) for t in states["t"]}, reverse=True)
    split = (t_values[len(t_values) // 2 - 1] + t_values[len(t_values) // 2]) // 2
    policy = dict(window=[1, 4], kept=[2, 3], iteration="block", expected=dict(eps_err=0.0),
                  schedule_by_t=[dict(t_min=split + 1, t_max=999, K=1, strategy="naive"),
                                 dict(t_min=0, t_max=split, K=2, strategy="damped", h=0.5)])
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    configs = explore.policy_configs([path])
    frame = explore.score_configs(screen, configs, buckets, count, tmp_path / "policy.csv", log=lambda *_: None)
    fixed = {}
    for name, extra in [("removal", dict(K=1, strategy="naive")), ("loop", dict(K=2, strategy="damped", h=0.5))]:
        _, err = explore.container_errors(screen, dict(window=[1, 4], kept=[2, 3], iteration="block", **extra))
        fixed[name] = err.eps_err.to_numpy()
    t = states["t"].numpy()
    expected = np.where(t > split, fixed["removal"], fixed["loop"])
    expected_buckets = sens.bucket_means(expected, buckets, count)
    measured = frame.iloc[0][[f"eps_err_b{b}" for b in range(count)]].to_numpy(dtype=float)
    assert np.allclose(measured, expected_buckets, rtol=1e-4, atol=1e-10)
    assert screen.model.blocks is screen.original_blocks
