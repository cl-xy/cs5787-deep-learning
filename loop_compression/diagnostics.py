"""Loop diagnostics: what happens to the residual stream as the kept unit is re-applied.

For each window (default: the windows of the frozen configs) and each strategy
(naive, h=1; damped, several h) the kept unit g is applied K = 1..Kmax times to
the REAL window input of the cached calibration states:

    x_k = x_{k-1} + h (g(x_{k-1}) - x_{k-1})        (h = 1 is naive replay)

Per iteration, averaged over tokens and over the conditional/unconditional CFG
halves, then over the states of each timestep bucket (bucket 0 = noisiest):

    norm_ratio  mean token L2 norm of x_k / that of x_0     (drift / blow-up)
    cos_prev    cosine(x_k, x_{k-1})                         (1 = stopped moving)
    cos_in      cosine(x_k, x_0)                             (representation drift)
    rel_step    ||x_k - x_{k-1}|| / ||x_{k-1}||              (size of the update)
    eps_err     CFG output error of the full looped model vs the original
                (optional, ``--no-eps`` skips it; needs full forward passes)

Onsets reported per (window, strategy, h, bucket):
    saturation_k   first k >= 2 with rel_step < ``--sat-step``      (extra loops change nothing)
    drift_k        first k with |norm_ratio - 1| > ``--drift-norm`` or cos_in < ``--drift-cos``
    best_k         k with the lowest eps_err; degrades_k first k > best_k whose eps_err is
                   more than ``--degrade-factor`` x the best (collapse in output space)
    best_euler_time  h * best_k, the total "time" integrated by the loop (naive: h = 1)
The reference ``loop_block_norms.csv`` gives the original model's own norm growth per
block, to judge how large a norm change is normal.

    python -m loop_compression.diagnostics --output team_runs/main
    python -m loop_compression.diagnostics --output team_runs/main --budgets 4 --kmax 6 --no-eps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from team_pipeline import experiment as exp

from .blocks import LoopedBlocks
from .sensitivity import (EpsilonScreen, NULL_LABEL, assign_buckets, select_states,
                          _load_run_context, _load_screen, _states_for_screening)

METRICS = ("norm_ratio", "cos_prev", "cos_in", "rel_step")
H_VALUES = (0.25, 0.5, 0.75)


def _token_norm(x):
    return x.norm(dim=-1).mean(1)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(1)


@torch.inference_mode()
def unit_trajectory(blocks, x0, c, h, kmax):
    """Metrics of iterating the unit ``blocks`` from ``x0``: dict of [kmax, batch] tensors."""
    def g(z):
        for block in blocks:
            z = block(z, c)
        return z

    base_norm = _token_norm(x0).clamp_min(1e-12)
    out = {name: [] for name in METRICS}
    previous = x0
    for _ in range(kmax):
        current = previous + h * (g(previous) - previous)
        out["norm_ratio"].append(_token_norm(current) / base_norm)
        out["cos_prev"].append(_cosine(current, previous))
        out["cos_in"].append(_cosine(current, x0))
        step = (current - previous).norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-12)
        out["rel_step"].append(step.mean(1))
        previous = current
    return {name: torch.stack(values).float().cpu() for name, values in out.items()}


@torch.inference_mode()
def window_inputs(model, base_blocks, first, x, t, y, cfg_batch=True):
    """(input to block ``first``, conditioning c) exactly as the sampler's CFG forward builds them."""
    if cfg_batch:
        x = torch.cat([x, x])
        t = torch.cat([t, t])
        y = torch.cat([y, torch.full_like(y, NULL_LABEL)])
    z = model.x_embedder(x) + model.pos_embed
    c = model.t_embedder(t) + model.y_embedder(y, False)
    for block in base_blocks[:first]:
        z = block(z, c)
    return z, c


def _fold_cfg(values, n):
    """Average the conditional and unconditional halves: [k, 2n] -> [k, n]."""
    return (values[:, :n] + values[:, n:]) / 2


def trajectories(screen, window, kept, kmax, h_values=H_VALUES, include_naive=True):
    """{(strategy, h): {metric: [kmax, states]}} for one window over all screened states."""
    first = window[0]
    units = [screen.base_blocks[i] for i in kept]
    settings = ([("naive", 1.0)] if include_naive else []) + [("damped", float(h)) for h in h_values]
    store = {s: {m: [] for m in METRICS} for s in settings}
    for start in range(0, screen.count, screen.batch):
        x, t, y = screen._batch(start)
        n = len(x)
        z, c = window_inputs(screen.model, screen.base_blocks, first, x, t, y)
        for strategy, h in settings:
            result = unit_trajectory(units, z, c, h, kmax)
            for metric in METRICS:
                store[(strategy, h)][metric].append(_fold_cfg(result[metric], n))
    return {s: {m: torch.cat(parts, dim=1).numpy() for m, parts in metrics.items()}
            for s, metrics in store.items()}


@torch.inference_mode()
def block_norm_reference(screen):
    """Per-block output norm / embedding norm of the ORIGINAL model, averaged per state: [states, depth+1]."""
    depth = len(screen.base_blocks)
    result = np.zeros((screen.count, depth + 1))
    for start in range(0, screen.count, screen.batch):
        x, t, y = screen._batch(start)
        n = len(x)
        z, c = window_inputs(screen.model, screen.base_blocks, 0, x, t, y)
        norms = [_token_norm(z)]
        for block in screen.base_blocks:
            z = block(z, c)
            norms.append(_token_norm(z))
        folded = _fold_cfg(torch.stack(norms).float().cpu(), n)
        folded = folded / folded[0:1].clamp_min(1e-12)
        result[start:start + n] = folded.T.numpy()
    return result


@torch.inference_mode()
def eps_by_k(screen, window, kept, kmax, settings, buckets, bucket_count):
    """{(strategy, h): eps_err [kmax, bucket_count]}: full-model output error after K loops.

    The K-th iterate of the unit is exactly what a ``LoopedBlocks`` container with
    K loops hands to the post-window blocks, so every iterate is pushed through the
    SUFFIX of the network only (blocks after the window, final layer and the model's
    own CFG combination). This equals ``screen.errors(LoopedBlocks(..., K=k))`` at a
    fraction of the cost: the prefix runs once and iterates are shared across K.
    """
    model, first, last = screen.model, window[0], window[1]
    units = [screen.base_blocks[i] for i in kept]
    suffix = screen.base_blocks[last + 1:]
    errors = {s: np.zeros((kmax, screen.count)) for s in settings}
    for start in range(0, screen.count, screen.batch):
        x, t, y = screen._batch(start)
        n = len(x)
        z0, c = window_inputs(model, screen.base_blocks, first, x, t, y)

        def g(z):
            for block in units:
                z = block(z, c)
            return z

        def run_suffix(*_):
            z = current
            for block in suffix:
                z = block(z, c)
            return model.unpatchify(model.final_layer(z, c))

        model.forward = run_suffix  # forward_with_cfg calls self.forward(...) and applies CFG itself
        try:
            for strategy, h in settings:
                current = z0
                for k in range(kmax):
                    current = current + h * (g(current) - current)
                    out = model.forward_with_cfg(torch.cat([x, x]), torch.cat([t, t]),
                                                 torch.cat([y, torch.full_like(y, NULL_LABEL)]),
                                                 screen.cfg)[:n].float().cpu()
                    errors[(strategy, h)][k, start:start + n] = screen._relative(
                        out, screen.reference[start:start + n]).eps_err.to_numpy()
        finally:
            del model.forward
        print(f"  eps batch {start // screen.batch + 1}/{-(-screen.count // screen.batch)}", flush=True)
    return {s: np.array([[float(e[k][buckets == b].mean()) for b in range(bucket_count)]
                         for k in range(kmax)]) for s, e in errors.items()}


def bucket_table(label, traj, buckets, bucket_count, eps=None):
    """Long table of per-bucket means for one window."""
    rows = []
    for (strategy, h), metrics in traj.items():
        kmax = next(iter(metrics.values())).shape[0]
        for b in range(bucket_count):
            mask = buckets == b
            for k in range(kmax):
                row = dict(window=label, strategy=strategy, h=h, bucket=b, k=k + 1, euler_time=h * (k + 1))
                for metric, values in metrics.items():
                    row[metric] = float(values[k, mask].mean())
                if eps is not None:
                    row["eps_err"] = float(eps[(strategy, h)][k, b])
                rows.append(row)
    return pd.DataFrame(rows)


def _first(mask, ks):
    hits = np.flatnonzero(mask)
    return int(ks[hits[0]]) if len(hits) else None


def onsets(table, sat_step=0.02, drift_norm=0.25, drift_cos=0.8, degrade_factor=2.0):
    """Saturation, drift and degradation onsets per (window, strategy, h, bucket)."""
    rows = []
    for (window, strategy, h, bucket), sub in table.groupby(["window", "strategy", "h", "bucket"]):
        sub = sub.sort_values("k")
        ks = sub.k.to_numpy()
        row = dict(window=window, strategy=strategy, h=h, bucket=bucket,
                   saturation_k=_first((sub.rel_step.to_numpy() < sat_step) & (ks >= 2), ks),
                   drift_k=_first((np.abs(sub.norm_ratio.to_numpy() - 1) > drift_norm)
                                  | (sub.cos_in.to_numpy() < drift_cos), ks))
        if "eps_err" in sub:
            eps = sub.eps_err.to_numpy()
            best = int(np.argmin(eps))
            row["best_k"] = int(ks[best])
            row["best_eps_err"] = float(eps[best])
            row["best_euler_time"] = float(h * ks[best])
            row["degrades_k"] = _first((ks > ks[best]) & (eps > degrade_factor * eps[best]), ks)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(table, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = table[table.window == label]
    metrics = [m for m in (*METRICS, "eps_err") if m in sub.columns]
    buckets = sorted(sub.bucket.unique())
    fig, axes = plt.subplots(len(buckets), len(metrics), figsize=(3.2 * len(metrics), 2.2 * len(buckets)),
                             squeeze=False, constrained_layout=True, sharex=True)
    for r, bucket in enumerate(buckets):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for (strategy, h), line in sub[sub.bucket == bucket].groupby(["strategy", "h"]):
                ax.plot(line.k, line[metric], marker="o", ms=3,
                        label="naive" if strategy == "naive" else f"damped h={h:g}")
            if metric == "eps_err":
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(metric, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"bucket {bucket}", fontsize=8)
            if r == len(buckets) - 1:
                ax.set_xlabel("loop count K")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle(f"window {label}", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def windows_from_frozen(output, budgets=None):
    """Unique (label, window, kept) of the frozen looped configs."""
    frozen = json.loads((Path(output) / "frozen_configs.json").read_text())
    found = {}
    for budget, item in sorted(frozen.items(), key=lambda kv: int(kv[0])):
        if budgets and int(budget) not in set(budgets):
            continue
        s = item["loop"]["settings"]
        label = f"{s['window'][0]}-{s['window'][1]}_kept{'-'.join(map(str, s['kept']))}"
        found.setdefault(label, (s["window"], s["kept"]))
    if not found:
        raise ValueError("No frozen looped windows found")
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="third_party/DiT")
    parser.add_argument("--budgets", type=int, nargs="+", help="frozen budgets whose windows to analyse")
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-eps", action="store_true", help="skip the (slower) full-model error per K")
    parser.add_argument("--sat-step", type=float, default=0.02)
    parser.add_argument("--drift-norm", type=float, default=0.25)
    parser.add_argument("--drift-cos", type=float, default=0.8)
    parser.add_argument("--degrade-factor", type=float, default=2.0)
    parser.add_argument("--reanalyze", action="store_true",
                        help="recompute loop_onsets.csv from the saved loop_trajectory.csv (no model run)")
    args = parser.parse_args(argv)
    if args.reanalyze:
        out = Path(args.output) / "profile" / "diagnostics"
        table = pd.read_csv(out / "loop_trajectory.csv")
        if "euler_time" not in table:
            table["euler_time"] = table.h * table.k
        table.to_csv(out / "loop_trajectory.csv", index=False)
        summary = onsets(table, args.sat_step, args.drift_norm, args.drift_cos, args.degrade_factor)
        summary.to_csv(out / "loop_onsets.csv", index=False)
        print(summary.to_string(index=False))
        return

    root, protocol, config = _load_run_context(args.output, args.repo)
    out = root / "profile" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    cache = _states_for_screening(root, config, protocol, args.images, args.stride)
    states = select_states(cache, args.images, args.stride)
    buckets, count = assign_buckets(states["t"].numpy(), args.buckets)
    screen, _ = _load_screen(config, states, args.batch)
    windows = windows_from_frozen(root, args.budgets)

    norms = block_norm_reference(screen)
    ref = pd.DataFrame([dict(block=b, bucket=k, norm_vs_embedding=float(norms[buckets == k, b].mean()))
                        for b in range(norms.shape[1]) for k in range(count)])
    ref.to_csv(out / "loop_block_norms.csv", index=False)

    tables = []
    for label, (window, kept) in windows.items():
        print(f"window {label}", flush=True)
        traj = trajectories(screen, window, kept, args.kmax)
        eps = None if args.no_eps else eps_by_k(screen, window, kept, args.kmax, list(traj), buckets, count)
        table = bucket_table(label, traj, buckets, count, eps)
        tables.append(table)
        plot_diagnostics(table, label, out / f"loop_diagnostics_{label}.png")
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "loop_trajectory.csv", index=False)
    summary = onsets(table, args.sat_step, args.drift_norm, args.drift_cos, args.degrade_factor)
    summary.to_csv(out / "loop_onsets.csv", index=False)
    exp.write_json(out / "diagnostics.json", dict(
        states=int(len(states["x"])), kmax=args.kmax, buckets=count, windows={k: v for k, v in windows.items()},
        thresholds=dict(sat_step=args.sat_step, drift_norm=args.drift_norm, drift_cos=args.drift_cos,
                        degrade_factor=args.degrade_factor)))
    print(summary.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
