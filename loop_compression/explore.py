"""Screening of the extension ideas on the cached calibration states (eps_err, no sampling).

    python -m loop_compression.explore gate    --output team_runs/main
    python -m loop_compression.explore quality --output team_runs/main
    python -m loop_compression.explore policy  --output team_runs/main \
        --policy team_runs/main/profile/sensitivity/policy_matched/policy_4.json

gate      Token gating (masked): loop only the selected tokens on the 2nd..Kth
          application. Grid over signal (update | grad | random control), keep
          fraction and K for the frozen looped windows, against the ungated loop and
          plain removal. Gating saves no compute here; see ``blocks`` for the estimate.
quality   Full-parameter arm: window blocks are ALL kept (nothing removed), looped
          with small damped steps. Parameters equal the original; FLOPs go up.
          eps_err only says how far the output moves from the original, not whether
          it is better: shortlist by small drift, then judge with CLIP/FID.
policy    Run ``schedule_by_t`` policies from ``loop_compression.policy`` through the
          real container and compare measured per-bucket eps_err with the table.

Results append to ``<output>/profile/explore/<name>.csv`` (resumable by config id).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from team_pipeline import experiment as exp

from .blocks import LoopedBlocks
from .diagnostics import windows_from_frozen
from .sensitivity import (assign_buckets, bucket_means, select_states, settings_id, _load_run_context,
                          _load_screen, _states_for_screening)

DEFAULT_IMAGES, DEFAULT_STRIDE = 8, 10


def container_errors(screen, settings):
    """Per-state eps_err of a LoopedBlocks built from ``settings`` (timestep-aware rules supported)."""
    container = LoopedBlocks(screen.base_blocks, **settings)
    if container.timed:
        with container.attached(screen.model):
            return container, screen.errors(container)
    return container, screen.errors(container)


def _log(*args):
    print(*args, flush=True)


def score_configs(screen, configs, buckets, bucket_count, csv_path, log=_log, flops_steps=50):
    """Score ``configs`` ([{group, settings, extra}]) and append to ``csv_path``; skips scored ids."""
    csv_path = Path(csv_path)
    done = set(pd.read_csv(csv_path, usecols=["config_id"]).config_id) if csv_path.exists() else set()
    model, depth = screen.model, len(screen.base_blocks)
    reference_flops = exp.estimate_dit_flops(model, flops_steps)["dit_sampling_flops"]
    reference_params = sum(p.numel() for p in model.parameters())
    todo = [c for c in configs if settings_id(c["settings"]) not in done]
    log(f"{len(configs)} configs, {len(done)} already scored, {len(todo)} to run")
    started = time.perf_counter()
    for count, config in enumerate(todo, 1):
        settings = config["settings"]
        container, errors = container_errors(screen, settings)
        model.blocks = container
        try:
            flops = exp.estimate_dit_flops(model, flops_steps)["dit_sampling_flops"]
            params = sum(p.numel() for p in model.parameters())
        finally:
            model.blocks = screen.original_blocks
        by_bucket = bucket_means(errors.eps_err.to_numpy(), buckets, bucket_count)
        meta = container.metadata()
        row = dict(config_id=settings_id(settings), group=config["group"],
                   eps_err=float(by_bucket.mean()), var_err=float(errors.var_err.mean()),
                   flops_ratio=flops / reference_flops, params_ratio=params / reference_params,
                   block_calls=len(container),
                   est_pruned_block_calls=meta.get("estimated_pruned_block_evaluations", len(container)),
                   settings=json.dumps(settings, sort_keys=True), **config.get("extra", {}),
                   **{f"eps_err_b{b}": float(v) for b, v in enumerate(by_bucket)})
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        log(f"{count}/{len(todo)} {config['group']}: eps_err={row['eps_err']:.4f} "
            f"({(time.perf_counter() - started) / count:.1f}s/config)")
    return pd.read_csv(csv_path)


# ------------------------------------------------------------------------- configs

def gate_configs(windows, loop_counts=(2, 3), h=0.75, fractions=(0.25, 0.5, 0.75),
                 signals=("update", "grad", "random")):
    """Ungated loops, plain removal and every gate setting for each (window, kept)."""
    configs = []
    for label, (window, kept) in windows.items():
        base = dict(window=list(window), kept=list(kept), iteration="block")
        configs.append(dict(group="removal", settings=dict(base, K=1, strategy="naive"),
                            extra=dict(window=label, K=1, signal="", keep_fraction=1.0, select="")))
        for K in loop_counts:
            loop = dict(base, K=K, strategy="damped", h=h)
            configs.append(dict(group="ungated", settings=loop,
                                extra=dict(window=label, K=K, signal="", keep_fraction=1.0, select="")))
            for signal in signals:
                for fraction in fractions:
                    for select in (("high", "low") if signal != "random" else ("high",)):
                        gate = dict(signal=signal, keep_fraction=fraction, select=select)
                        configs.append(dict(group="gated", settings=dict(loop, gate=gate),
                                            extra=dict(window=label, K=K, signal=signal,
                                                       keep_fraction=fraction, select=select)))
    return configs


def quality_configs(depth=28, sizes=(2, 4), starts=None, steps=((2, 0.25), (2, 0.5), (3, 0.5), (2, 0.75))):
    """Full-parameter windows (every block kept, none removed) looped with damped steps.

    ``steps`` are (K, h) pairs; K*h is the total Euler time the loop integrates
    (1 = a refinement of the original single step, > 1 over-relaxes it).
    """
    starts = range(8, depth - 6, 2) if starts is None else starts
    configs = []
    for size in sizes:
        for first in starts:
            last = first + size - 1
            if last >= depth - 1:
                continue
            centre = (first + last + 1) / (2 * depth)
            for K, h in steps:
                configs.append(dict(
                    group="quality",
                    settings=dict(window=[first, last], kept=list(range(first, last + 1)), K=K,
                                  iteration="block", strategy="damped", h=h),
                    extra=dict(window=f"{first}-{last}", K=K, signal="", keep_fraction=1.0, select="",
                               centre=round(centre, 4), h=h, euler_time=K * h)))
    return configs


def policy_configs(paths):
    configs = []
    for path in paths:
        policy = json.loads(Path(path).read_text())
        settings = {k: policy[k] for k in ("window", "kept", "iteration", "schedule_by_t")}
        settings.update(K=1, strategy="naive")
        configs.append(dict(group=f"policy:{Path(path).parent.name}/{Path(path).stem}", settings=settings,
                            extra=dict(window="-".join(map(str, policy["window"])), K=1, signal="",
                                       keep_fraction=1.0, select="",
                                       expected_eps_err=policy.get("expected", {}).get("eps_err"))))
    return configs


# ------------------------------------------------------------------------- plotting

def plot_gate(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = sorted(frame.window.unique())
    counts = sorted(frame[frame.group == "gated"].K.unique())
    fig, axes = plt.subplots(len(windows), len(counts), figsize=(4.6 * len(counts), 3.4 * len(windows)),
                             squeeze=False, constrained_layout=True)
    for r, window in enumerate(windows):
        for c, K in enumerate(counts):
            ax = axes[r][c]
            sub = frame[(frame.window == window) & (frame.K == K)]
            for (signal, select), line in sub[sub.group == "gated"].groupby(["signal", "select"]):
                line = line.sort_values("keep_fraction")
                ax.plot(line.keep_fraction, line.eps_err, marker="o", label=f"{signal}/{select}")
            ungated = sub[sub.group == "ungated"]
            if len(ungated):
                ax.axhline(float(ungated.eps_err.iloc[0]), color="k", linestyle="--", label="ungated loop")
            removal = frame[(frame.window == window) & (frame.group == "removal")]
            if len(removal):
                ax.axhline(float(removal.eps_err.iloc[0]), color="grey", linestyle=":", label="removal")
            ax.set_title(f"window {window}, K={K}", fontsize=9)
            ax.set_xlabel("keep_fraction of tokens looped")
            ax.set_ylabel("eps_err")
            ax.set_yscale("log")
            ax.legend(fontsize=6)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quality(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for (K, h, size), line in frame.assign(size=frame.window.map(
            lambda w: int(w.split("-")[1]) - int(w.split("-")[0]) + 1)).groupby(["K", "h", "size"]):
        line = line.sort_values("centre")
        ax.plot(line.centre, line.eps_err, marker="o", label=f"K={K} h={h:g}, {size} blocks")
    ax.set_yscale("log")
    ax.set_xlabel("window centre / depth")
    ax.set_ylabel("eps_err vs original (drift)")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["gate", "quality", "policy"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="third_party/DiT")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--budgets", type=int, nargs="+", help="frozen budgets whose windows gate uses")
    parser.add_argument("--policy", nargs="*", default=[], help="policy JSON files (policy command)")
    parser.add_argument("--max-configs", type=int, help="debug: first N configs only")
    args = parser.parse_args(argv)

    root, protocol, config = _load_run_context(args.output, args.repo)
    out = root / "profile" / "explore"
    out.mkdir(parents=True, exist_ok=True)
    cache = _states_for_screening(root, config, protocol, 16, 5)
    states = select_states(cache, args.images, args.stride)
    buckets, count = assign_buckets(states["t"].numpy(), args.buckets)
    screen, _ = _load_screen(config, states, args.batch)
    depth = len(screen.base_blocks)
    if args.command == "gate":
        configs = gate_configs(windows_from_frozen(root, args.budgets))
    elif args.command == "quality":
        configs = quality_configs(depth)
    else:
        if not args.policy:
            raise SystemExit("policy needs --policy FILE [FILE ...]")
        configs = policy_configs(args.policy)
    if args.max_configs:
        configs = configs[:args.max_configs]
    print(f"Screening {len(states['x'])} states ({args.images} images, stride {args.stride})", flush=True)
    frame = score_configs(screen, configs, buckets, count, out / f"{args.command}.csv")
    wanted = {settings_id(c["settings"]) for c in configs}
    frame = frame[frame.config_id.isin(wanted)]
    if args.command == "gate":
        plot_gate(frame, out / "gate.png")
    elif args.command == "quality":
        plot_quality(frame, out / "quality.png")
    keep = [c for c in ("group", "window", "K", "signal", "select", "keep_fraction", "h", "eps_err",
                        "flops_ratio", "est_pruned_block_calls", "expected_eps_err") if c in frame.columns]
    print(frame[keep].sort_values("eps_err").round(5).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
