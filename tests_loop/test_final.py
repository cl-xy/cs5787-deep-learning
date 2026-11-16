"""Final benchmark: frozen-only inputs, command planning, timing and aggregation."""
import json

import numpy as np
import pandas as pd
import pytest

from loop_compression import final


def frozen_payload():
    def item(name, role, settings=True):
        return dict(name=name, role=role, settings=dict(window=[8, 11], kept=[9, 10], K=2) if settings else None,
                    removed=None if settings else [5, 9], validation_lpips=0.4,
                    unique_parameters=1, dit_sampling_GFLOPs=1.0)
    return {"2": dict(loop=item("d2_r1_loop", "loop"), strongest_removal=item("d2_removal_lowest_blocks", "x", False),
                      naive_same_window=item("d2_r1_naive_same_window", "n"), removal_same_window=None),
            "4": dict(loop=item("d4_r1_loop", "loop"), strongest_removal=item("d4_removal_best_window", "r"),
                      naive_same_window=None, removal_same_window=None)}


def write_frozen(tmp_path):
    (tmp_path / "frozen_configs.json").write_text(json.dumps(frozen_payload()))
    return tmp_path


def test_final_requires_frozen_configs(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze"):
        final.load_frozen(tmp_path)


def test_load_frozen_selects_budgets_and_roles(tmp_path):
    out = write_frozen(tmp_path)
    names = [e["name"] for e in final.load_frozen(out)]
    assert names == ["d2_r1_loop", "d2_removal_lowest_blocks", "d4_r1_loop", "d4_removal_best_window"]
    assert [e["name"] for e in final.load_frozen(out, [4])] == ["d4_r1_loop", "d4_removal_best_window"]
    extras = [e["name"] for e in final.load_frozen(out, [2], extras=True)]
    assert "d2_r1_naive_same_window" in extras and len(extras) == 3
    assert [e["role"] for e in final.load_frozen(out, [2])] == ["loop", "strongest_removal"]
    with pytest.raises(ValueError):
        final.load_frozen(out, [9])


def test_frozen_config_copies_are_write_once(tmp_path):
    entries = final.load_frozen(write_frozen(tmp_path))
    folder = final.write_frozen_configs(tmp_path, entries)
    assert json.loads((folder / "d2_r1_loop.json").read_text())["K"] == 2
    assert not (folder / "d2_removal_lowest_blocks.json").exists()   # shared removal has no config file
    final.write_frozen_configs(tmp_path, entries)                       # identical rewrite is fine
    entries[0]["settings"]["K"] = 3
    with pytest.raises(ValueError, match="changed"):
        final.write_frozen_configs(tmp_path, entries)


def test_plan_commands_use_the_requested_split_only(tmp_path):
    entries = final.load_frozen(write_frozen(tmp_path), [2])
    jobs = final.plan_commands("out", entries, "test", tmp_path, limit=8)
    assert [n for n, _ in jobs] == ["original", "d2_r1_loop", "d2_removal_lowest_blocks"]
    for name, command in jobs:
        assert command[command.index("--split") + 1] == "test" and command[command.index("--limit") + 1] == "8"
    assert jobs[0][1][jobs[0][1].index("--method") + 1] == "original"
    with pytest.raises(ValueError):
        final.plan_commands("out", entries, "validation", tmp_path)
    with pytest.raises(ValueError):
        final.plan_commands("out", entries, "calibration", tmp_path)


def test_dry_run_and_existing_results_skip_work(tmp_path, monkeypatch):
    monkeypatch.setattr(final.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    jobs = [("original", ["x"]), ("a", ["y"])]
    final.run_jobs(tmp_path, "test", jobs, dry_run=True)
    done = tmp_path / "test" / "a"
    done.mkdir(parents=True)
    pd.DataFrame(dict(index=[0])).to_csv(done / "samples.csv", index=False)
    (done / "image_0000.pt").write_bytes(b"x")
    final.run_jobs(tmp_path, "test", [("a", ["y"])])               # skipped, no subprocess


def test_interleaving_alternates_direction_and_covers_everything():
    order = final.interleaved_order(["original", "a", "b"], 3)
    assert [n for r, n in order if r == 0] == ["original", "a", "b"]
    assert [n for r, n in order if r == 1] == ["b", "a", "original"]
    assert len(order) == 9 and {n for _, n in order} == {"original", "a", "b"}


def test_timing_summary_reports_spread_and_speedup():
    rows = []
    for rep, offset in enumerate([0.0, 0.2, -0.2]):
        rows.append(pd.DataFrame(dict(variant="original", rep=rep, latency_s=[10 + offset] * 4)))
        rows.append(pd.DataFrame(dict(variant="fast", rep=rep, latency_s=[5 + offset] * 4)))
    table = final.summarize_timing(pd.concat(rows)).set_index("variant")
    assert table.loc["original", "latency_mean_s"] == pytest.approx(10)
    assert table.loc["fast", "speedup_vs_original"] == pytest.approx(2.0, rel=0.05)
    assert table.loc["original", "latency_rep_mean_std_s"] == pytest.approx(0.2)
    assert (table.device == "mps").all() and table.loc["fast", "repetitions"] == 3


def synthetic_results():
    return pd.DataFrame(dict(
        variant=["original", "d2_r1_loop", "d2_removal_lowest_blocks"], images=64,
        lpips_mean=[0.0, 0.45, 0.50], unique_parameters=[675e6, 627e6, 627e6],
        parameter_reduction_pct=[0, 7.1, 7.1], dit_sampling_GFLOPs=[23724.0, 23724.0, 22030.0],
        flops_reduction_pct=[0, 0.0, 7.1], effective_block_evaluations_per_forward=[28, 28, 26],
        latency_mean_s=[9.6, 11.8, 9.7]))


def test_summary_and_savings_table_show_looping_saves_parameters_not_flops(tmp_path):
    entries = final.load_frozen(write_frozen(tmp_path), [2])
    fid = pd.DataFrame(dict(variant=["original", "d2_r1_loop", "d2_removal_lowest_blocks"],
                            fid_vs_original=[0.0, 30.0, 35.0], images=1000))
    clip = pd.DataFrame(dict(variant=["original", "d2_r1_loop", "d2_removal_lowest_blocks"],
                             clip_score=[28.0, 27.5, 27.0], images=1000))
    summary = final.build_summary(synthetic_results(), entries, fid, clip)
    assert summary.set_index("variant").loc["d2_r1_loop", "role"] == "loop"
    assert summary.set_index("variant").loc["original", "role"] == "baseline"
    assert summary.images.eq(64).all()                       # per-table columns are not overwritten
    assert {"fid_vs_original", "clip_score"} <= set(summary.columns)
    savings = final.savings_table(summary).set_index("variant")
    assert savings.loc["d2_r1_loop", "parameter_reduction_pct"] > 7
    assert savings.loc["d2_r1_loop", "flops_reduction_pct"] == 0
    assert savings.loc["d2_removal_lowest_blocks", "flops_reduction_pct"] > 7
    for fid_plot in (False, True):
        final.plot_pareto(summary, tmp_path / "plots", fid=fid_plot)
    assert (tmp_path / "plots" / "pareto_lpips.png").stat().st_size > 0
    assert (tmp_path / "plots" / "pareto_fid.png").stat().st_size > 0


def test_report_end_to_end_on_synthetic_results(tmp_path):
    out = write_frozen(tmp_path)
    (out / "test").mkdir()
    synthetic_results().to_csv(out / "test" / "results.csv", index=False)
    final.main(["report", "--output", str(out), "--budgets", "2"])
    report = out / "report"
    summary = pd.read_csv(report / "summary.csv")
    assert list(summary.variant) == ["original", "d2_r1_loop", "d2_removal_lowest_blocks"]
    assert (report / "parameter_vs_flop_savings.csv").exists() and (report / "pareto_lpips.png").exists()
    assert not (report / "pareto_fid.png").exists()           # no FID results yet


def test_one_run_filling_two_roles_appears_once(tmp_path):
    payload = frozen_payload()
    payload["4"]["removal_same_window"] = dict(payload["4"]["strongest_removal"])   # same run, second role
    (tmp_path / "frozen_configs.json").write_text(json.dumps(payload))
    names = [e["name"] for e in final.load_frozen(tmp_path, [4], extras=True)]
    assert names == ["d4_r1_loop", "d4_removal_best_window"]
    roles = {e["name"]: e["role"] for e in final.load_frozen(tmp_path, [4], extras=True)}
    assert roles["d4_removal_best_window"] == "strongest_removal"
