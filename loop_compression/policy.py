"""Per-timestep-bucket loop policy from an existing window sweep (no new model runs).

``sensitivity windows`` stores ``eps_err_b0..bN`` (bucket 0 = noisiest sampler
steps) for every config. A timestep policy keeps the retained blocks fixed and
only changes HOW the kept unit is applied at each step, so it may only choose
among configs sharing the same ``window`` and ``kept``: plain removal (K=1), the
looped variants and, with skip allowed, ``window_removal`` (the unit bypassed at
that step, h=0). Parameters are identical for every choice; only FLOPs vary.

    python -m loop_compression.policy --output team_runs/main
    python -m loop_compression.policy --output team_runs/main --match-removal-flops
    python -m loop_compression.policy --output team_runs/main --max-flops-ratio 0.95

``--match-removal-flops`` caps each window's policy at the FLOPs of its plain
equal-parameter removal, i.e. it asks "can the same compute be spent better across
timesteps?". FLOPs of a policy are weighted by how many of the 50 sampler steps
fall in each bucket; the error objective weights buckets equally, like the sweep.

Outputs (``<output>/profile/sensitivity/policy/``): ``policy_table.csv`` (best
config per budget and bucket), ``policy_summary.csv`` (fixed vs per-bucket policy),
``policy_<budget>.json`` (a ``schedule_by_t`` setting usable by
``loop_compression.method``) and ``policy_buckets.png``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sensitivity import assign_buckets

STEP_KEYS = ("K", "strategy", "h", "beta")


def bucket_columns(frame):
    columns = sorted((c for c in frame.columns if c.startswith("eps_err_b")),
                     key=lambda c: int(c[len("eps_err_b"):]))
    if not columns:
        raise ValueError("Sweep has no eps_err_b* columns")
    return columns


def candidate_groups(frame, allow_skip=True):
    """Configs that keep the same blocks: key (discard, first, last, kept) -> rows.

    Plain ``removal`` (K=1, naive) is the natural "do not loop" choice inside a
    group. ``window_removal`` (h=0) bypasses the kept unit; as a per-timestep
    choice it leaves the stored blocks, and so the parameters, unchanged, so it
    is allowed unless ``allow_skip`` is false.
    """
    kinds = ["removal", "loop"] + (["window_removal"] if allow_skip else [])
    usable = frame[frame.kind.isin(kinds) & (frame.iteration == "block")]
    groups = {}
    for key, rows in usable.groupby(["discard", "first", "last", "kept"], sort=True):
        if (rows.kind == "removal").any():
            groups[key] = rows.reset_index(drop=True)
    return groups


def best_fixed(rows, columns, max_flops_ratio=None):
    """Best single config for all buckets; never the bypass (it is not a looping config).

    With ``max_flops_ratio`` only configs within that FLOPs budget compete, the fair
    fixed baseline for a FLOPs-capped policy.
    """
    mean = rows[columns].mean(axis=1)
    allowed = rows.kind != "window_removal"
    if max_flops_ratio is not None:
        allowed &= rows.flops_ratio <= max_flops_ratio + 1e-12
    if not allowed.any():
        raise ValueError("No fixed config satisfies the FLOPs cap")
    index = int(mean.where(allowed).idxmin())
    return index, float(mean.iloc[index])


def best_policy(rows, columns, max_flops_ratio=None, weights=None):
    """Per-bucket choice minimising mean eps_err (optionally with a mean-FLOPs cap).

    ``weights`` (sum 1, one per bucket) weight the FLOPs average by sampler steps;
    default equal. Returns (indices per bucket, mean eps_err, mean flops_ratio).
    Exhaustive over configs^buckets, small (a dozen configs, five buckets).
    """
    errors = rows[columns].to_numpy()
    flops = rows.flops_ratio.to_numpy()
    n_configs, n_buckets = errors.shape
    w = np.full(n_buckets, 1.0 / n_buckets) if weights is None else np.asarray(weights, dtype=float)
    if max_flops_ratio is None:
        picks = errors.argmin(axis=0)
        return picks.tolist(), float(errors[picks, range(n_buckets)].mean()), float(flops[picks] @ w)
    best = None
    for picks in itertools.product(range(n_configs), repeat=n_buckets):
        mean_flops = float(flops[list(picks)] @ w)
        if mean_flops > max_flops_ratio + 1e-12:
            continue
        mean_error = float(errors[list(picks), range(n_buckets)].mean())
        if best is None or mean_error < best[1]:
            best = (list(picks), mean_error, mean_flops)
    if best is None:
        raise ValueError("No per-bucket policy satisfies the FLOPs cap")
    return best


def bucket_ranges(timesteps, bucket_count):
    """[(t_low, t_high)] per bucket, bucket 0 = highest (noisiest) timesteps."""
    buckets, count = assign_buckets(np.asarray(timesteps), bucket_count)
    ts = np.asarray(timesteps)
    return [(int(ts[buckets == b].min()), int(ts[buckets == b].max())) for b in range(count)]


def bucket_edges(ranges):
    """Contiguous [t_min, t_max] per bucket covering the whole axis 0..999 (bucket 0 on top)."""
    count = len(ranges)
    edges = []
    for b, (low, high) in enumerate(ranges):
        edges.append([0 if b == count - 1 else low, 999 if b == 0 else high])
    for b in range(count - 1):  # close the gap between this bucket's low and the next one's high
        edges[b][0] = edges[b + 1][1] + 1
    return edges


def step_weights(edges, steps=50):
    """Fraction of the respaced sampler steps (evenly spaced over 0..999) in each bucket."""
    timesteps = np.round(np.linspace(0, 999, steps)).astype(int)
    return np.array([np.mean((timesteps >= lo) & (timesteps <= hi)) for lo, hi in edges])


def schedule_by_t(rows, picks, ranges):
    """Convert per-bucket config picks into the ``schedule_by_t`` setting.

    The bucket edges cover the whole timestep axis [0, 999] with no gaps, so every
    timestep the sampler visits (including 0 and 999) maps to a rule.
    """
    edges = bucket_edges(ranges)
    schedule = []
    for b, pick in enumerate(picks):
        row = rows.iloc[pick]
        schedule.append(dict(t_min=int(edges[b][0]), t_max=int(edges[b][1]), K=int(row.K),
                             strategy=str(row.strategy), h=float(row.h), beta=float(row.beta)))
    return schedule


def build_policy_tables(frame, timesteps, max_flops_ratio=None, per_budget_top=3,
                        match_removal_flops=False, allow_skip=True):
    """(table, summary, policies) from a window sweep ``frame``."""
    columns = bucket_columns(frame)
    ranges = bucket_ranges(timesteps, len(columns))
    weights = step_weights(bucket_edges(ranges))
    groups = candidate_groups(frame, allow_skip)
    if not groups:
        raise ValueError("No window group contains a removal baseline")
    table_rows, summary_rows, policies = [], [], {}
    for budget in sorted({key[0] for key in groups}):
        scored = []
        for key, rows in groups.items():
            if key[0] != budget:
                continue
            cap = max_flops_ratio
            if match_removal_flops:
                cap = float(rows[rows.kind == "removal"].flops_ratio.iloc[0])
            fixed_index, fixed_error = best_fixed(rows, columns, cap)
            picks, policy_error, policy_flops = best_policy(rows, columns, cap, weights)
            scored.append((policy_error, key, rows, fixed_index, fixed_error, picks, policy_flops))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0])
        for rank, (policy_error, key, rows, fixed_index, fixed_error, picks, policy_flops) in enumerate(
                scored[:per_budget_top], 1):
            fixed = rows.iloc[fixed_index]
            _, first, last, kept = key
            removal = rows[rows.kind == "removal"].iloc[0]
            summary_rows.append(dict(
                budget=int(budget), rank=rank, first=int(first), last=int(last), kept=kept,
                removal_eps_err=float(removal[columns].mean()),
                best_fixed_eps_err=fixed_error, best_fixed=f"{fixed.strategy} K={int(fixed.K)} h={fixed.h:g}",
                best_fixed_flops_ratio=float(fixed.flops_ratio),
                policy_eps_err=policy_error, policy_flops_ratio=policy_flops,
                policy_gain_vs_fixed_pct=100 * (1 - policy_error / fixed_error)))
            for b, pick in enumerate(picks):
                row = rows.iloc[pick]
                table_rows.append(dict(
                    budget=int(budget), rank=rank, bucket=b, t_range=f"{ranges[b][0]}-{ranges[b][1]}",
                    first=int(first), last=int(last), kept=kept, kind=row.kind, strategy=row.strategy,
                    K=int(row.K), h=float(row.h), beta=float(row.beta),
                    eps_err=float(row[columns[b]]), flops_ratio=float(row.flops_ratio),
                    fixed_best_eps_err=float(fixed[columns[b]])))
            if rank == 1:
                policies[int(budget)] = dict(
                    window=[int(first), int(last)], kept=json.loads(kept), iteration="block",
                    schedule_by_t=schedule_by_t(rows, picks, ranges),
                    expected=dict(eps_err=policy_error, best_fixed_eps_err=fixed_error,
                                  mean_flops_ratio=policy_flops))
    return pd.DataFrame(table_rows), pd.DataFrame(summary_rows), policies


def plot_policy(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = table[table["rank"] == 1]
    budgets = sorted(best.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 3.8), squeeze=False,
                             constrained_layout=True)
    for ax, budget in zip(axes[0], budgets):
        sub = best[best.budget == budget].sort_values("bucket")
        ax.plot(sub.bucket, sub.fixed_best_eps_err, "o--", label="best fixed config")
        ax.plot(sub.bucket, sub.eps_err, "s-", label="best per bucket")
        for _, row in sub.iterrows():
            label = {"removal": "K1", "window_removal": "skip"}.get(row.kind, f"{row.strategy[0]}{row.K}h{row.h:g}")
            ax.annotate(label, (row.bucket, row.eps_err), fontsize=7, xytext=(0, 6), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel("timestep bucket (0 = noisiest)")
        ax.set_ylabel("eps_err")
        ax.set_title(f"{int(budget)} discarded blocks")
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-flops-ratio", type=float,
                        help="cap on mean flops_ratio of the per-bucket policy (1.0 = original DiT)")
    parser.add_argument("--match-removal-flops", action="store_true",
                        help="cap each window's policy at the FLOPs of its plain removal baseline")
    parser.add_argument("--no-skip", action="store_true", help="do not allow bypassing the unit at a bucket")
    parser.add_argument("--tag", default="", help="suffix for the output folder, e.g. matched")
    parser.add_argument("--top", type=int, default=3, help="windows kept per budget")
    args = parser.parse_args(argv)
    sens = Path(args.output) / "profile" / "sensitivity"
    frame = pd.read_csv(sens / "window_sweep.csv")
    timesteps = json.loads((sens / "block_sensitivity.json").read_text())["timesteps"]
    table, summary, policies = build_policy_tables(frame, timesteps, args.max_flops_ratio, args.top,
                                                   args.match_removal_flops, not args.no_skip)
    out = sens / (f"policy_{args.tag}" if args.tag else "policy")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "policy_table.csv", index=False)
    summary.to_csv(out / "policy_summary.csv", index=False)
    for budget, policy in policies.items():
        (out / f"policy_{budget}.json").write_text(json.dumps(policy, indent=2) + "\n")
    plot_policy(table, out / "policy_buckets.png")
    print(summary.round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
