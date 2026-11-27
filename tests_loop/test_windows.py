"""Window sweep, resumable scoring and shortlist selection."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from loop_compression import sensitivity as sens
from loop_compression.blocks import LoopedBlocks
from team_pipeline import experiment as exp
from team_pipeline.methods import ScheduledBlocks

ROOT = Path(__file__).resolve().parents[1]


def test_window_geometry_for_the_28_block_model():
    windows = list(sens.window_candidates(28))
    assert windows and len({(w["window"][0], w["window"][1], w["kept_size"]) for w in windows}) == len(windows)
    for w in windows:
        first, last = w["window"]
        discarded = [i for i in range(first, last + 1) if i not in w["kept"]]
        assert 4 <= first and last <= 26                       # block 0-3 and the last block are untouched
        assert len(discarded) == w["discard"] and len(w["kept"]) == w["kept_size"]
        assert w["kept"] == list(range(w["kept"][0], w["kept"][0] + w["kept_size"]))
        assert first + w["discard"] // 2 == w["kept"][0]       # kept segment sits in the middle
    assert {w["discard"] for w in windows} == {2, 4, 6}


def test_variants_cover_the_planned_strategies_and_are_valid():
    kinds = [(k, s) for k, s in sens.strategy_variants((2,))]
    assert [k for k, _ in kinds].count("removal") == 1 and [k for k, _ in kinds].count("loop") == 9
    assert {s.get("strategy") for k, s in kinds if k == "loop"} == {"naive", "damped", "anchored"}
    assert {s["iteration"] for _, s in kinds} == {"block", "layer"}
    assert {s.get("h") for k, s in kinds if k == "loop" and s["strategy"] != "naive"} >= {0.25, 0.5, 0.75}
    assert {s.get("beta") for _, s in kinds if s["strategy"] == "anchored"} == {0.5}
    plan = sens.sweep_plan(28, (2, 4, 6), (2, 4), (2,), a_stride=2)
    assert len({p["config_id"] for p in plan}) == len(plan)
    blocks = [torch.nn.Identity() for _ in range(28)]

    class Pass(torch.nn.Module):
        def forward(self, x, c):
            return x
    blocks = [Pass() for _ in range(28)]
    for item in plan[::37]:
        LoopedBlocks(blocks, **item["settings"])


# ------------------------------------------------------------- scoring on a tiny DiT

DEPTH = 10


@pytest.fixture(scope="module")
def screen_factory():
    repo = Path(os.environ.get("DIT_TEST_REPO", ROOT / "third_party" / "DiT"))
    if not (repo / "models.py").exists():
        pytest.skip("Official DiT source not available")
    exp.configure_dit(repo)
    import models
    from diffusion import create_diffusion
    from types import SimpleNamespace

    def make():
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
        return sens.EpsilonScreen(model, states, "cpu", batch_states=4)
    return make


def test_equal_parameter_baseline_matches_the_shared_removal_mode(screen_factory):
    screen = screen_factory()
    looped = LoopedBlocks(screen.base_blocks, [3, 6], kept=[4, 5], K=1, strategy="naive")
    shared = ScheduledBlocks(screen.base_blocks, removed=[3, 6], mode="removal")
    a, b = screen.errors(looped), screen.errors(shared)
    assert np.allclose(a.eps_err, b.eps_err, rtol=1e-6, atol=1e-12) and a.eps_err.min() > 0


def test_sweep_scores_every_config_and_resumes(screen_factory, tmp_path):
    screen = screen_factory()
    buckets, count = sens.assign_buckets(screen.states["t"].numpy(), 2)
    plan = sens.sweep_plan(DEPTH, (2,), (2,), (2,), a_stride=2, a_min=1)
    assert len(plan) == 3 * 11
    csv = tmp_path / "sweep.csv"
    logs = []
    frame = sens.evaluate_plan(screen, plan[:11], buckets, count, csv, log=logs.append)
    assert len(frame) == 11 and frame.config_id.is_unique
    again = sens.evaluate_plan(screen, plan, buckets, count, csv, log=logs.append)
    assert len(again) == 33 and "11 already scored" in " ".join(logs)
    assert pd.read_csv(csv).iloc[:11].equals(frame)           # earlier rows untouched on resume
    assert screen.model.blocks is screen.original_blocks
    by = again.set_index("config_id")
    removal = again[again.kind == "removal"].iloc[0]
    assert removal.params_ratio == pytest.approx(again[again.kind == "loop"].params_ratio.iloc[0])
    assert removal.flops_ratio < 1
    loop = again[(again.kind == "loop") & (again.strategy == "naive") & (again.iteration == "block")].iloc[0]
    assert loop.flops_ratio > removal.flops_ratio             # looping costs extra compute
    window_removal = again[again.kind == "window_removal"].iloc[0]
    assert window_removal.params_ratio == pytest.approx(removal.params_ratio)  # kept blocks stay registered
    assert window_removal.flops_ratio < removal.flops_ratio
    assert np.isfinite(again[["eps_err", "var_err", "eps_err_b0", "eps_err_b1"]].to_numpy()).all()
    assert json.loads(removal.settings)["K"] == 1


# --------------------------------------------------------------------- shortlist

def synthetic_frame():
    rows = []
    rng = np.random.default_rng(0)
    for discard in (2, 4):
        for first in (4, 8, 12):
            size = discard + 2
            settings_base = dict(window=[first, first + size - 1], kept=[first + discard // 2, first + discard // 2 + 1])
            for kind, extra in sens.strategy_variants((2,)):
                s = sens.full_settings(settings_base["window"], settings_base["kept"], extra)
                rows.append(dict(config_id=sens.settings_id(s), kind=kind, discard=discard, kept_size=2,
                                 first=first, last=first + size - 1, size=size,
                                 center_depth_fraction=(2 * first + size) / 56, K=s["K"],
                                 iteration=s["iteration"], strategy=s["strategy"], h=s["h"], beta=s["beta"],
                                 eps_err=float(rng.uniform(0.01, 1)), settings=json.dumps(s, sort_keys=True)))
    return pd.DataFrame(rows)


def test_shortlist_roles_names_and_files(tmp_path):
    frame = synthetic_frame()
    block_table = pd.DataFrame({"removal_eps_err": np.linspace(0.9, 0.0, 28)}, index=pd.Index(range(28), name="block"))
    block_table.loc[27, "removal_eps_err"] = -1.0   # protected last block must never be chosen
    entries = sens.build_shortlist(frame, per_budget=2, block_table=block_table)
    names = [e["name"] for e in entries]
    assert len(set(names)) == len(names)
    for discard in (2, 4):
        roles = [e["role"] for e in entries if e["budget"] == discard]
        assert roles.count("loop") == 2 and roles.count("naive_same_window") == 2
        assert roles.count("removal_same_window") == 2 and roles.count("removal_best_window") == 1
        assert roles.count("removal_lowest_blocks") == 1
    loops = [e for e in entries if e["role"] == "loop" and e["budget"] == 2]
    sub = frame[(frame.discard == 2) & (frame.kind == "loop")].sort_values("eps_err")
    assert [e["eps_err"] for e in loops] == pytest.approx(sub.eps_err.head(2).tolist())
    for entry in entries:
        if entry["role"] == "naive_same_window":
            assert entry["settings"]["strategy"] == "naive" and entry["settings"]["iteration"] == "block"
        if entry["role"] == "removal_same_window":
            assert entry["settings"]["K"] == 1
    lowest = next(e for e in entries if e["role"] == "removal_lowest_blocks" and e["budget"] == 4)
    assert len(lowest["removed"]) == 4 and 0 not in lowest["removed"] and 27 not in lowest["removed"]
    folder = tmp_path / "shortlist"
    sens.write_shortlist(entries, folder)
    assert (tmp_path / "shortlist.json").exists()
    written = {p.stem for p in folder.glob("*.json")}
    assert written == {e["name"] for e in entries if e["settings"] is not None}
    # Config files are directly usable by the shared runner's --method-config.
    settings = json.loads((folder / f"{loops[0]['name']}.json").read_text())
    assert settings == loops[0]["settings"]


def test_plots_are_written(tmp_path):
    sens.plot_window_results(synthetic_frame(), tmp_path)
    for name in ("window_depth_curve.png", "window_position_size_heatmap.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_early_windows_reduced_variants_and_tagged_outputs(tmp_path):
    default_starts = {p["settings"]["window"][0] for p in sens.sweep_plan(28, (2, 4, 6), (2, 4), (2,), a_stride=4)}
    early = sens.sweep_plan(28, (2, 4, 6), (2, 4), (2,), a_stride=4, a_min=0)
    assert min(default_starts) == 4 and min(p["settings"]["window"][0] for p in early) == 0
    assert sens.sweep_plan(28, (2,), (2,), (2,), a_stride=4, protect_last=0)[-1]["settings"]["window"][1] <= 27
    full = list(sens.strategy_variants((2,)))
    reduced = list(sens.strategy_variants((2,), reduced=True))
    assert len(reduced) < len(full) and {s["iteration"] for _, s in reduced} == {"block"}
    assert {k for k, _ in reduced} == {"removal", "window_removal", "loop"}
    assert len({p["config_id"] for p in sens.sweep_plan(28, (2, 4), (2,), (2,), 4, 0, reduced=True)}) == len(
        sens.sweep_plan(28, (2, 4), (2,), (2,), 4, 0, reduced=True))
    entries = [dict(name="d2_r1_loop", role="loop", budget=2, settings=dict(window=[0, 3]))]
    sens.write_shortlist(entries, tmp_path / "shortlist_early")
    assert (tmp_path / "shortlist_early.json").exists() and not (tmp_path / "shortlist.json").exists()
    assert (tmp_path / "shortlist_early" / "d2_r1_loop.json").exists()
    sens.plot_window_results(synthetic_frame(), tmp_path, "_early")
    assert (tmp_path / "window_depth_curve_early.png").exists()
