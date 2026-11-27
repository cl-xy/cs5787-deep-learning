"""Layer-sensitivity profiling of a frozen DiT by one-step epsilon screening.

Screening avoids full 50-step sampling. The original sampler is run once on the
calibration split while recording the states (x_t, t, label) it visits. A
candidate block schedule is then scored on those SAME states by how far the
CFG-combined model output moves from the original model's output:

    eps_err = ||out_cand[:4] - out_ref[:4]||^2 / ||out_ref[:4]||^2

(channels 0-3 are the noise prediction the sampler uses; channels 4-7 are the
learned-variance values, reported as var_err). Errors are averaged per state,
then per timestep bucket, then over buckets with equal weight. Everything here
uses the calibration split only; nothing is trained.

Usage:
    python -m loop_compression.sensitivity blocks  --output team_runs/main
    python -m loop_compression.sensitivity windows --output team_runs/main
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from team_pipeline import experiment as exp

from .blocks import LoopedBlocks

NULL_LABEL = 1000
ERROR_COLUMNS = ("eps_err", "var_err")


def _log(*args):
    """Progress output must reach redirected logs immediately."""
    print(*args, flush=True)


# ------------------------------------------------------------ trajectory cache

@contextmanager
def record_states(model, store):
    """Record (x_t, t, label) at every sampler call of ``model.forward_with_cfg``.

    The sampler duplicates x for CFG, so only the first half is stored. Timesteps
    seen here are the ORIGINAL (un-respaced) values the model receives.
    """
    original = model.forward_with_cfg

    def wrapped(x, t, y, cfg_scale):
        half = len(x) // 2
        store["x"].append(x[:half].detach().float().cpu().clone())
        store["t"].append(t[:half].detach().cpu().clone())
        store["y"].append(y[:half].detach().cpu().clone())
        return original(x, t, y, cfg_scale)

    model.forward_with_cfg = wrapped
    try:
        yield
    finally:
        del model.forward_with_cfg  # restore the class method


def build_cache(model, vae, diffusion, pairs, cfg_scale):
    """Run the ORIGINAL sampler on ``pairs`` and return every visited state."""
    pairs = exp.validate_pairs(pairs)
    chunks = {"x": [], "t": [], "y": []}
    image_index, step_index = [], []
    for index, (label, seed) in enumerate(pairs):
        store = {"x": [], "t": [], "y": []}
        with record_states(model, store):
            exp.generate(model, vae, diffusion, label, seed, cfg_scale)
        steps = len(store["x"])
        for key in chunks:
            chunks[key].append(torch.cat(store[key]))
        image_index += [index] * steps
        step_index += list(range(steps))
        print(f"Cached {steps} states: image {index + 1}/{len(pairs)} "
              f"(label={label}, seed={seed})", flush=True)
    return dict(x=torch.cat(chunks["x"]), t=torch.cat(chunks["t"]), y=torch.cat(chunks["y"]),
                image_index=torch.tensor(image_index), step_index=torch.tensor(step_index),
                pairs=[list(p) for p in pairs], cfg_scale=float(cfg_scale))


def select_states(cache, images=None, stride=5, offset=None):
    """Subset of cached states: first ``images`` images, every ``stride``-th step."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    offset = stride // 2 if offset is None else offset
    keep = cache["step_index"] % stride == offset
    if images is not None:
        keep &= cache["image_index"] < images
    if not keep.any():
        raise ValueError("State selection is empty")
    return dict(x=cache["x"][keep], t=cache["t"][keep], y=cache["y"][keep],
                image_index=cache["image_index"][keep], step_index=cache["step_index"][keep],
                cfg_scale=cache["cfg_scale"])


def assign_buckets(timesteps, count):
    """Bucket id per state; 0 is the earliest (noisiest) part of sampling."""
    values = np.asarray(timesteps)
    unique = np.unique(values)[::-1]
    count = min(count, len(unique))
    mapping = {}
    for bucket, group in enumerate(np.array_split(unique, count)):
        mapping.update({int(t): bucket for t in group})
    return np.array([mapping[int(t)] for t in values]), count


# -------------------------------------------------------------- epsilon screen

class EpsilonScreen:
    """Score candidate block containers against the original model on fixed states."""

    def __init__(self, model, states, device, batch_states=16):
        self.model, self.states, self.device = model, states, torch.device(device)
        self.batch = int(batch_states)
        self.cfg = float(states["cfg_scale"])
        self.original_blocks = model.blocks
        self.base_blocks = list(model.blocks.retained.values())
        self.count = len(states["x"])
        self.reference = self._outputs(self.original_blocks)

    @torch.inference_mode()
    def _outputs(self, blocks):
        self.model.blocks = blocks
        try:
            results = []
            for start in range(0, self.count, self.batch):
                x, t, y = self._batch(start)
                n = len(x)
                out = self.model.forward_with_cfg(
                    torch.cat([x, x]), torch.cat([t, t]),
                    torch.cat([y, torch.full_like(y, NULL_LABEL)]), self.cfg)[:n]
                if not torch.isfinite(out).all():
                    raise RuntimeError("Nonfinite model output while screening")
                results.append(out.float().cpu())
            return torch.cat(results)
        finally:
            self.model.blocks = self.original_blocks

    def _batch(self, start):
        s = self.states
        stop = start + self.batch
        return tuple(s[key][start:stop].to(self.device) for key in ("x", "t", "y"))

    def errors(self, blocks):
        """Per-state relative errors of a candidate container vs the original."""
        out = self._outputs(blocks)
        return self._relative(out, self.reference)

    @staticmethod
    def _relative(out, ref):
        def rel(a, b):
            return ((a - b).flatten(1).square().sum(1) /
                    b.flatten(1).square().sum(1).clamp_min(1e-12)).numpy()
        return pd.DataFrame(dict(eps_err=rel(out[:, :4], ref[:, :4]),
                                 var_err=rel(out[:, 4:], ref[:, 4:])))

    def feature_changes(self):
        """Per-state, per-block L1 ratio and cosine Block Influence on the original model.

        Averaged over the conditional and unconditional CFG halves. Block
        influence is 1 - cos(block input, block output), averaged over tokens.
        """
        depth = len(self.base_blocks)
        l1 = np.zeros((self.count, depth))
        bi = np.zeros((self.count, depth))
        handles, position = [], {}

        def make_hook(index):
            def hook(module, args, output):
                x, out = args[0].float(), output.float()
                n = len(x) // 2
                ratio = (out - x).abs().flatten(1).sum(1) / (x.abs().flatten(1).sum(1) + 1e-8)
                cosine = torch.nn.functional.cosine_similarity(x, out, dim=-1).mean(1)
                start = position["start"]
                l1[start:start + n, index] = ((ratio[:n] + ratio[n:]) / 2).cpu().numpy()
                bi[start:start + n, index] = (1 - (cosine[:n] + cosine[n:]) / 2).cpu().numpy()
            return hook

        for index, block in enumerate(self.base_blocks):
            handles.append(block.register_forward_hook(make_hook(index)))
        try:
            self.model.blocks = self.original_blocks
            with torch.inference_mode():
                for start in range(0, self.count, self.batch):
                    position["start"] = start
                    x, t, y = self._batch(start)
                    self.model.forward_with_cfg(torch.cat([x, x]), torch.cat([t, t]),
                                                torch.cat([y, torch.full_like(y, NULL_LABEL)]), self.cfg)
        finally:
            for handle in handles:
                handle.remove()
        return l1, bi


def bucket_means(per_state, buckets, count):
    """Mean of a per-state vector within each timestep bucket, shape [count]."""
    per_state = np.asarray(per_state)
    return np.array([per_state[buckets == b].mean() for b in range(count)])


# ------------------------------------------------------------- block profiling

def profile_blocks(screen, buckets, bucket_count, loop_steps=(1.0, 0.5), log=None):
    """Per block: L1 ratio, cosine BI, removal error, and 'previous block in its place'."""
    log = log or _log
    depth = len(screen.base_blocks)
    base = screen.base_blocks
    l1, bi = screen.feature_changes()
    long_rows = []

    def add(metric, block, per_state):
        values = pd.Series(per_state).groupby(buckets).mean()
        for bucket, value in values.items():
            long_rows.append(dict(block=block, bucket=int(bucket), metric=metric, value=float(value)))

    for block in range(depth):
        add("l1_ratio", block, l1[:, block])
        add("cosine_bi", block, bi[:, block])
    for block in range(depth):
        started = time.perf_counter()
        removal = LoopedBlocks(base, [block, block], kept=[block], K=1, strategy="damped", h=0.0)
        errors = screen.errors(removal)
        removal_error = float(errors.eps_err.mean())
        add("removal_eps_err", block, errors.eps_err.to_numpy())
        add("removal_var_err", block, errors.var_err.to_numpy())
        if block >= 1:  # previous block replaces this one
            for h in loop_steps:
                strategy = "naive" if h == 1.0 else "damped"
                looped = LoopedBlocks(base, [block - 1, block], kept=[block - 1], K=2,
                                      strategy=strategy, h=h)
                add(f"loop_h{h:g}_eps_err", block, screen.errors(looped).eps_err.to_numpy())
        log(f"block {block:2d}/{depth - 1}: removal eps_err={removal_error:.4f} "
            f"({time.perf_counter() - started:.1f}s)")
    long = pd.DataFrame(long_rows)
    per_bucket = long.pivot_table(index=["block", "bucket"], columns="metric", values="value").reset_index()
    overall = per_bucket.groupby("block").mean(numeric_only=True).drop(columns="bucket")
    return overall, per_bucket


def plot_block_heatmaps(per_bucket, path, bucket_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [m for m in per_bucket.columns if m not in {"block", "bucket"} and not m.endswith("var_err")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 7), constrained_layout=True,
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        matrix = per_bucket.pivot(index="block", columns="bucket", values=metric)
        values = np.ma.masked_invalid(matrix.to_numpy())
        image = ax.imshow(values, aspect="auto", interpolation="nearest")
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(range(bucket_count), [f"b{b}" for b in range(bucket_count)])
        ax.set_yticks(range(len(matrix)))
        ax.set_xlabel("timestep bucket (b0 = noisiest)")
        fig.colorbar(image, ax=ax)
    axes[0][0].set_ylabel("block index")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ window sweep

BUDGETS = (2, 4, 6)
DAMPING = (0.25, 0.5, 0.75)


def window_candidates(depth, discards=BUDGETS, kept_sizes=(2, 4), a_min=4, a_stride=2,
                      protect_last=1):
    """Contiguous windows with ``discard`` removed blocks and ``kept`` central blocks.

    Window size is M = discard + kept; discarded blocks are split around the kept
    segment (``discard // 2`` before it). The last ``protect_last`` blocks are never
    part of a window.
    """
    for discard in discards:
        for kept_size in kept_sizes:
            size = discard + kept_size
            lead = discard // 2
            for first in range(a_min, depth - protect_last - size + 1, a_stride):
                yield dict(window=[first, first + size - 1], discard=discard, kept_size=kept_size,
                           kept=list(range(first + lead, first + lead + kept_size)))


def strategy_variants(loop_counts=(2,), reduced=False):
    """(kind, LoopedBlocks settings) for one window.

    removal         discarded blocks dropped, kept blocks run once (equal parameters)
    window_removal  the whole window dropped (control; fewer parameters)
    loop            the kept segment looped K times
    ``reduced`` keeps only block iteration with naive, damped h in {0.5, 0.75} and
    anchored h=0.75: the variants that won in the full sweep, for quick wide scans.
    """
    yield "removal", dict(K=1, iteration="block", strategy="naive")
    yield "window_removal", dict(K=1, iteration="block", strategy="damped", h=0.0)
    for K in loop_counts:
        yield "loop", dict(K=K, iteration="block", strategy="naive")
        for h in ((0.5, 0.75) if reduced else DAMPING):
            yield "loop", dict(K=K, iteration="block", strategy="damped", h=h)
        for h in ((0.75,) if reduced else DAMPING):
            yield "loop", dict(K=K, iteration="block", strategy="anchored", h=h, beta=0.5)
        if not reduced:
            yield "loop", dict(K=K, iteration="layer", strategy="naive")
            yield "loop", dict(K=K, iteration="layer", strategy="damped", h=0.5)


def full_settings(window, kept, extra):
    settings = dict(window=list(window), kept=list(kept), K=1, iteration="block",
                    strategy="damped", h=None, beta=0.0)
    settings.update(extra)
    if settings["h"] is None:
        settings["h"] = 1.0 if settings["strategy"] == "naive" else 1.0 / settings["K"]
    settings["h"] = float(settings["h"])
    return settings


def settings_id(settings):
    import hashlib
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


def sweep_plan(depth, discards, kept_sizes, loop_counts, a_stride, a_min=4, protect_last=1,
               reduced=False):
    plan = []
    for window in window_candidates(depth, discards, kept_sizes, a_min, a_stride, protect_last):
        for kind, extra in strategy_variants(loop_counts, reduced):
            settings = full_settings(window["window"], window["kept"], extra)
            plan.append(dict(kind=kind, discard=window["discard"], kept_size=window["kept_size"],
                             settings=settings, config_id=settings_id(settings)))
    return plan


def evaluate_plan(screen, plan, buckets, bucket_count, csv_path, log=None, flops_steps=50):
    """Score every planned config; append to ``csv_path`` so interrupted sweeps resume."""
    log = log or _log
    csv_path = Path(csv_path)
    done = set(pd.read_csv(csv_path, usecols=["config_id"]).config_id) if csv_path.exists() else set()
    todo = [item for item in plan if item["config_id"] not in done]
    log(f"{len(plan)} configs planned, {len(done)} already scored, {len(todo)} to run")
    model, depth = screen.model, len(screen.base_blocks)
    reference_flops = exp.estimate_dit_flops(model, flops_steps)["dit_sampling_flops"]
    reference_params = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    for count, item in enumerate(todo, 1):
        s = item["settings"]
        container = LoopedBlocks(screen.base_blocks, **s)
        errors = screen.errors(container)
        model.blocks = container
        try:
            flops = exp.estimate_dit_flops(model, flops_steps)["dit_sampling_flops"]
            params = sum(p.numel() for p in model.parameters())
        finally:
            model.blocks = screen.original_blocks
        by_bucket = bucket_means(errors.eps_err.to_numpy(), buckets, bucket_count)
        first, last = s["window"]
        row = dict(config_id=item["config_id"], kind=item["kind"], discard=item["discard"],
                   kept_size=item["kept_size"], first=first, last=last, size=last - first + 1,
                   center_depth_fraction=(first + last + 1) / (2 * depth),
                   kept=json.dumps(s["kept"]), K=s["K"], iteration=s["iteration"],
                   strategy=s["strategy"], h=s["h"], beta=s["beta"],
                   eps_err=float(by_bucket.mean()), var_err=float(errors.var_err.mean()),
                   flops_ratio=flops / reference_flops, params_ratio=params / reference_params,
                   block_calls=len(container), settings=json.dumps(s, sort_keys=True),
                   **{f"eps_err_b{b}": float(v) for b, v in enumerate(by_bucket)})
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        if count % 10 == 0 or count == len(todo):
            elapsed = time.perf_counter() - started
            log(f"{count}/{len(todo)} scored; {elapsed / count:.1f}s/config; "
                f"~{elapsed / count * (len(todo) - count) / 60:.0f} min left")
    return pd.read_csv(csv_path)


def build_shortlist(frame, per_budget=3, block_table=None):
    """Top looped configs per budget plus the matching baselines.

    Roles: ``loop`` (selected), ``naive_same_window`` and ``removal_same_window``
    (same window as a selected loop), ``removal_best_window`` (best contiguous
    removal at that budget) and, if a block table is given, ``removal_lowest_blocks``
    (the k individually least sensitive blocks, via the shared ``removal`` method).
    """
    entries = []
    for discard in sorted(frame.discard.unique()):
        budget = frame[frame.discard == discard]
        looped = budget[budget.kind == "loop"].sort_values("eps_err").head(per_budget)
        for rank, (_, row) in enumerate(looped.iterrows(), 1):
            base = f"d{discard}_r{rank}"
            entries.append(dict(name=f"{base}_loop", role="loop", budget=int(discard),
                                eps_err=float(row.eps_err), settings=json.loads(row.settings)))
            same = budget[(budget["first"] == row["first"]) & (budget["last"] == row["last"])
                          & (budget.kept_size == row.kept_size)]
            naive = same[(same.kind == "loop") & (same.strategy == "naive") & (same.iteration == "block")
                         & (same.K == row.K)]
            if len(naive):
                entries.append(dict(name=f"{base}_naive_same_window", role="naive_same_window",
                                    budget=int(discard), eps_err=float(naive.iloc[0].eps_err),
                                    settings=json.loads(naive.iloc[0].settings)))
            removal = same[same.kind == "removal"]
            if len(removal):
                entries.append(dict(name=f"{base}_removal_same_window", role="removal_same_window",
                                    budget=int(discard), eps_err=float(removal.iloc[0].eps_err),
                                    settings=json.loads(removal.iloc[0].settings)))
        removals = budget[budget.kind == "removal"].sort_values("eps_err")
        if len(removals):
            best = removals.iloc[0]
            entries.append(dict(name=f"d{discard}_removal_best_window", role="removal_best_window",
                                budget=int(discard), eps_err=float(best.eps_err),
                                settings=json.loads(best.settings)))
        if block_table is not None:
            candidates = block_table.drop(index=[0, block_table.index.max()], errors="ignore")
            lowest = sorted(int(i) for i in candidates.removal_eps_err.nsmallest(int(discard)).index)
            entries.append(dict(name=f"d{discard}_removal_lowest_blocks", role="removal_lowest_blocks",
                                budget=int(discard), settings=None, removed=lowest))
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate shortlist names")
    return entries


def write_shortlist(entries, folder):
    """Write settings JSONs into ``folder`` and the list to ``<folder>.json`` (shortlist -> shortlist.json)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if entry["settings"] is not None:
            (folder / f"{entry['name']}.json").write_text(
                json.dumps(entry["settings"], indent=2) + "\n")
    exp.write_json(folder.parent / f"{folder.name}.json", entries)


def plot_window_results(frame, out_dir, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    budgets = sorted(frame.discard.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 4), constrained_layout=True,
                             squeeze=False)
    for ax, discard in zip(axes[0], budgets):
        sub = frame[frame.discard == discard]
        for kind, label in [("loop", "best looped config"), ("removal", "removal (same discarded blocks)")]:
            curve = (sub[sub.kind == kind].groupby(["center_depth_fraction"]).eps_err.min().sort_index())
            ax.plot(curve.index, curve.values, marker="o", label=label)
        ax.axvspan(0.45, 0.60, color="grey", alpha=0.2, label="Chen et al. 0.45-0.60")
        ax.set_yscale("log")
        ax.set_title(f"{int(discard)} discarded blocks")
        ax.set_xlabel("window centre / depth")
        ax.set_ylabel("eps_err (lower = closer to original)")
        ax.legend(fontsize=7)
    fig.savefig(out_dir / f"window_depth_curve{tag}.png", dpi=160)
    plt.close(fig)
    loops = frame[frame.kind == "loop"]
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 4), constrained_layout=True,
                             squeeze=False)
    for ax, discard in zip(axes[0], budgets):
        sub = loops[loops.discard == discard]
        table = sub.pivot_table(index="size", columns="first", values="eps_err", aggfunc="min")
        image = ax.imshow(np.log10(table.to_numpy()), aspect="auto", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(table.columns)), table.columns, fontsize=7)
        ax.set_yticks(range(len(table.index)), table.index)
        ax.set_xlabel("window start block")
        ax.set_ylabel("window size M")
        ax.set_title(f"{int(discard)} discarded: log10 best eps_err")
        fig.colorbar(image, ax=ax)
    fig.savefig(out_dir / f"window_position_size_heatmap{tag}.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------- driver

def _load_run_context(output, repo):
    """Validated (root, protocol, config) for SCREENING tools.

    Same checks as ``team_pipeline.cli.load_run`` except the runtime-environment
    equality: screening writes only eps_err/diagnostic tables, never the comparable
    manifests, latency or metric results the environment check protects. A
    difference (e.g. an OS patch update) is reported, not fatal. The shared runner
    (``team_pipeline.cli run``) keeps its strict check.
    """
    from team_pipeline.cli import configure_dit, environment, load_protocol, protocol_hash, sha256
    from team_pipeline.experiment import shared_code_hash
    root = Path(output)
    protocol = load_protocol(root / "protocol.json")
    config = json.loads((root / "config.json").read_text())
    if config.get("shared_code_sha256") != shared_code_hash():
        raise ValueError("Shared pipeline changed after preparation; create a new run")
    if config["protocol_sha256"] != protocol_hash(protocol):
        raise ValueError("Protocol changed after preparation; create a new run")
    if any(config.get(k) != v for k, v in protocol["settings"].items()):
        raise ValueError("Config settings changed after preparation; create a new run")
    if configure_dit(repo) != config["dit_commit"]:
        raise ValueError("DiT source changed after preparation")
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preparation")
    recorded = json.loads((root / "environment.json").read_text())
    current = environment(config["device"])
    if current != recorded:
        changed = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        _log(f"WARNING: runtime environment differs from the prepared run ({', '.join(changed)}); "
             "screening results are not comparable manifests.")
    return root, protocol, config


def _states_for_screening(root, config, protocol, images, stride, batch_pairs=None):
    """Load the calibration cache, building it with the original sampler if absent."""
    cache_path = root / "profile" / "sensitivity" / "trajectory_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=True)
        if cache["pairs"] != [list(p) for p in protocol["splits"]["calibration"]]:
            raise ValueError("Trajectory cache was built for a different calibration split")
        return cache
    model, vae, diffusion = exp._prepare(config, "original", (), 1)
    try:
        device = exp.select_device(config.get("device", "auto"))
        model.to(device)
        vae.to(device)
        cache = build_cache(model, vae, diffusion, protocol["splits"]["calibration"], config["cfg_scale"])
    finally:
        del model, vae, diffusion
        exp._cleanup()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _load_screen(config, states, batch):
    device = exp.select_device(config.get("device", "auto"))
    model = exp.load_model(config["checkpoint"]).to(device)
    return EpsilonScreen(model, states, device, batch_states=batch), device


def run_blocks(args):
    root, protocol, config = _load_run_context(args.output, args.repo)
    out = root / "profile" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    cache = _states_for_screening(root, config, protocol, args.images, args.stride)
    states = select_states(cache, args.images, args.stride)
    buckets, bucket_count = assign_buckets(states["t"].numpy(), args.buckets)
    print(f"Screening {len(states['x'])} states ({args.images} images, stride {args.stride}), "
          f"{bucket_count} timestep buckets", flush=True)
    screen, _ = _load_screen(config, states, args.batch)
    overall, per_bucket = profile_blocks(screen, buckets, bucket_count)
    overall.to_csv(out / "block_sensitivity.csv", index_label="block")
    per_bucket.to_csv(out / "block_sensitivity_by_bucket.csv", index=False)
    plot_block_heatmaps(per_bucket, out / "block_sensitivity_heatmap.png", bucket_count)
    exp.write_json(out / "block_sensitivity.json", dict(
        states=int(len(states["x"])), images=args.images, stride=args.stride,
        buckets=bucket_count, timesteps=sorted({int(t) for t in states["t"]}, reverse=True),
        screening="one-step CFG output error vs original on cached calibration states"))
    print(overall.round(4).to_string())
    print(f"Wrote {out}")


def run_windows(args):
    root, protocol, config = _load_run_context(args.output, args.repo)
    out = root / "profile" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    cache = _states_for_screening(root, config, protocol, args.images, args.stride)
    states = select_states(cache, args.images, args.stride)
    buckets, bucket_count = assign_buckets(states["t"].numpy(), args.buckets)
    screen, _ = _load_screen(config, states, args.batch)
    depth = len(screen.base_blocks)
    plan = sweep_plan(depth, args.discards, args.kept_sizes, args.loop_counts, args.a_stride,
                      args.a_min, args.protect_last, args.reduced)
    if args.max_configs:
        plan = plan[:args.max_configs]
    # A tag keeps an extra scan (e.g. early windows) apart from the sweep that
    # ``drive`` reads: nothing in the untagged shortlist/CSV/plots is overwritten.
    tag = f"_{args.tag}" if args.tag else ""
    print(f"Screening {len(states['x'])} states, {bucket_count} buckets", flush=True)
    frame = evaluate_plan(screen, plan, buckets, bucket_count, out / f"window_sweep{tag}.csv")
    frame = frame[frame.config_id.isin({p["config_id"] for p in plan})]
    block_path = out / "block_sensitivity.csv"
    block_table = pd.read_csv(block_path, index_col="block") if block_path.exists() else None
    entries = build_shortlist(frame, args.per_budget, block_table)
    write_shortlist(entries, out / f"shortlist{tag}")
    plot_window_results(frame, out, tag)
    print(f"Shortlist ({len(entries)} entries) written to {out / f'shortlist{tag}.json'}")
    for entry in entries:
        print(f"  {entry['name']:<34} budget={entry['budget']} eps_err={entry.get('eps_err', float('nan')):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("blocks", "per-block sensitivity table and heatmap"),
                            ("windows", "window sweep, depth curve and shortlist")]:
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("--output", required=True, help="Prepared shared run directory")
        sub.add_argument("--repo", default="third_party/DiT")
        sub.add_argument("--images", type=int, default=16 if name == "blocks" else 8,
                         help="calibration images to screen on")
        sub.add_argument("--stride", type=int, default=5 if name == "blocks" else 10,
                         help="use every N-th sampler step")
        sub.add_argument("--buckets", type=int, default=5)
        sub.add_argument("--batch", type=int, default=16, help="states per forward (x2 for CFG)")
        if name == "windows":
            sub.add_argument("--discards", type=int, nargs="+", default=list(BUDGETS))
            sub.add_argument("--kept-sizes", type=int, nargs="+", default=[2, 4])
            sub.add_argument("--loop-counts", type=int, nargs="+", default=[2])
            sub.add_argument("--a-stride", type=int, default=3, help="step between window start blocks")
            sub.add_argument("--per-budget", type=int, default=3, help="looped configs shortlisted per budget")
            sub.add_argument("--max-configs", type=int, help="debug: only the first N planned configs")
            sub.add_argument("--a-min", type=int, default=4, help="first block a window may start at")
            sub.add_argument("--protect-last", type=int, default=1, help="final blocks never in a window")
            sub.add_argument("--reduced", action="store_true", help="fewer strategies per window")
            sub.add_argument("--tag", default="", help="suffix for all outputs, keeps the base sweep intact")
    args = parser.parse_args(argv)
    {"blocks": run_blocks, "windows": run_windows}[args.command](args)


if __name__ == "__main__":
    main()
