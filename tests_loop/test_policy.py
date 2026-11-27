import json

import numpy as np
import pandas as pd
import pytest

from loop_compression import policy

TIMESTEPS = [958, 856, 754, 652, 550, 449, 347, 245, 143, 41]


def row(kind, K, strategy, h, errs, flops, window=(4, 9), kept=(6, 7), discard=2):
    out = dict(kind=kind, discard=discard, first=window[0], last=window[1], kept=json.dumps(list(kept)),
               K=K, iteration="block", strategy=strategy, h=h, beta=0.0, flops_ratio=flops)
    out.update({f"eps_err_b{i}": e for i, e in enumerate(errs)})
    return out


def frame():
    return pd.DataFrame([
        row("removal", 1, "naive", 1.0, [1, 1, 1, 9, 9], 0.9),
        row("loop", 2, "naive", 1.0, [3, 3, 3, 3, 3], 1.0),
        row("loop", 2, "damped", 0.5, [2, 2, 2, 5, 5], 1.0),
        row("window_removal", 1, "damped", 0.0, [0, 0, 0, 0, 0], 0.8),  # different retained set: ignored
    ])


def groups_without_skip():
    return policy.candidate_groups(frame(), allow_skip=False)


def test_skip_is_a_candidate_only_when_allowed():
    without = groups_without_skip()
    assert len(without) == 1
    assert "window_removal" not in set(next(iter(without.values())).kind)
    with_skip = next(iter(policy.candidate_groups(frame()).values()))
    assert "window_removal" in set(with_skip.kind)


def test_per_bucket_policy_beats_best_fixed_and_picks_per_bucket_minimum():
    rows = next(iter(groups_without_skip().values()))
    columns = policy.bucket_columns(frame())
    _, fixed_error = policy.best_fixed(rows, columns)
    picks, error, flops = policy.best_policy(rows, columns)
    assert fixed_error == pytest.approx(3.0)
    assert [rows.iloc[p].kind for p in picks[:3]] == ["removal"] * 3
    assert rows.iloc[picks[3]].strategy == "naive" and rows.iloc[picks[3]].K == 2
    assert error == pytest.approx((1 + 1 + 1 + 3 + 3) / 5)
    assert error < fixed_error
    assert flops == pytest.approx((0.9 * 3 + 1.0 * 2) / 5)


def test_flops_cap_is_respected_and_infeasible_cap_raises():
    rows = next(iter(groups_without_skip().values()))
    columns = policy.bucket_columns(frame())
    _, _, flops = policy.best_policy(rows, columns, max_flops_ratio=0.95)
    assert flops <= 0.95 + 1e-9
    with pytest.raises(ValueError):
        policy.best_policy(rows, columns, max_flops_ratio=0.5)


def test_schedule_covers_every_timestep_without_gaps():
    rows = next(iter(groups_without_skip().values()))
    ranges = policy.bucket_ranges(TIMESTEPS, 5)
    schedule = policy.schedule_by_t(rows, [0, 1, 2, 1, 0], ranges)
    assert schedule[0]["t_max"] == 999 and schedule[-1]["t_min"] == 0
    for upper, lower in zip(schedule, schedule[1:]):
        assert upper["t_min"] == lower["t_max"] + 1
    covered = {t for entry in schedule for t in range(entry["t_min"], entry["t_max"] + 1)}
    assert covered == set(range(1000))


def test_tables_and_main_write_outputs(tmp_path):
    sens = tmp_path / "profile" / "sensitivity"
    sens.mkdir(parents=True)
    frame().to_csv(sens / "window_sweep.csv", index=False)
    (sens / "block_sensitivity.json").write_text(json.dumps(dict(timesteps=TIMESTEPS)))
    policy.main(["--output", str(tmp_path)])
    out = sens / "policy"
    for name in ["policy_table.csv", "policy_summary.csv", "policy_2.json", "policy_buckets.png"]:
        assert (out / name).exists()
    saved = json.loads((out / "policy_2.json").read_text())
    assert saved["window"] == [4, 9] and saved["kept"] == [6, 7]
    assert len(saved["schedule_by_t"]) == 5


def skip_frame():
    """Bucket 0 is free to bypass; buckets 3-4 profit from looping; looping costs FLOPs."""
    return pd.DataFrame([
        row("removal", 1, "naive", 1.0, [1, 1, 1, 9, 9], 0.9),
        row("loop", 2, "naive", 1.0, [3, 3, 3, 3, 3], 1.0),
        row("window_removal", 1, "damped", 0.0, [1.1, 40, 40, 40, 40], 0.8)])


def test_flops_matched_policy_spends_saved_compute_on_the_hard_buckets():
    rows = next(iter(policy.candidate_groups(skip_frame()).values()))
    columns = policy.bucket_columns(skip_frame())
    removal_flops = 0.9
    picks, error, flops = policy.best_policy(rows, columns, removal_flops)
    kinds = [rows.iloc[p].kind for p in picks]
    assert kinds[0] == "window_removal"            # nearly free to bypass at the noisiest bucket
    assert kinds[3:].count("loop") == 1            # the saved compute buys one loop in a hard bucket
    assert kinds[1] == kinds[2] == "removal"
    assert flops <= removal_flops + 1e-9
    _, fixed = policy.best_fixed(rows, columns, removal_flops)
    assert fixed == pytest.approx(np.mean([1, 1, 1, 9, 9]))  # only plain removal fits the cap; skip never counts
    assert error < fixed


def test_step_weights_follow_the_respaced_sampler_steps():
    edges = policy.bucket_edges(policy.bucket_ranges(TIMESTEPS, 5))
    weights = policy.step_weights(edges, steps=50)
    assert weights.sum() == pytest.approx(1.0) and (weights > 0).all()
    assert policy.step_weights([(0, 999)], 50)[0] == 1.0
    rows = next(iter(policy.candidate_groups(skip_frame()).values()))
    columns = policy.bucket_columns(skip_frame())
    heavy = np.array([0.0, 0.0, 0.0, 0.0, 1.0])     # all steps in bucket 4
    _, _, flops = policy.best_policy(rows, columns, None, heavy)
    assert flops == pytest.approx(rows.iloc[int(rows[columns[4]].to_numpy().argmin())].flops_ratio)
