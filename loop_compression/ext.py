"""Candidates and freezing for the extension study (timestep policy, gating, quality arm).

Runs in its OWN prepared root (e.g. ``team_runs/ext``) because the shared runner
refuses to compare results across a changed runtime environment, and ``original``
must be regenerated under the current one. Screening artefacts are read from the
main root; nothing in the main root is modified.

    python -m team_pipeline.cli prepare --output team_runs/ext --device mps
    python -m loop_compression.ext candidates --main team_runs/main --output team_runs/ext
    python -m loop_compression.drive validate --output team_runs/ext \
        --shortlist team_runs/ext/shortlist_ext.json
    python -m team_pipeline.cli clip --output team_runs/ext --split validation --names original <names>
    python -m loop_compression.ext freeze --output team_runs/ext
    python -m loop_compression.final test --output team_runs/ext
    python -m loop_compression.final fid  --output team_runs/ext --budget 4 --stride 2 --limit 500 \
        --reference checkpoints/reference/VIRTUAL_imagenet256_labeled.npz

Selection uses VALIDATION only: lowest LPIPS for the compression arm (budget 4) and
highest paired CLIP gain over the original for the full-parameter quality arm
(budget key 0). Each rule is applied once; ``freeze`` writes ``frozen_configs.json``
in the schema ``loop_compression.final`` reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from team_pipeline import experiment as exp

from .blocks import LoopedBlocks

QUALITY_BUDGET = 0
COMPRESSION_BUDGET = 4


def policy_settings(path):
    policy = json.loads(Path(path).read_text())
    settings = {k: policy[k] for k in ("window", "kept", "iteration", "schedule_by_t")}
    settings.update(K=1, strategy="naive")
    return settings


def best_gate_candidates(frame, window):
    """(best gated config, random control with the same K/fraction, ungated same-K) from a gate CSV."""
    sub = frame[frame.window == window]
    informative = sub[(sub.group == "gated") & (sub.signal != "random") & (sub.keep_fraction < 1.0)]
    if informative.empty:
        raise ValueError(f"No informative gated configs for window {window}")
    best = informative.loc[informative.eps_err.idxmin()]
    control = sub[(sub.group == "gated") & (sub.signal == "random") & (sub.K == best.K)
                  & (sub.keep_fraction == best.keep_fraction)]
    ungated = sub[(sub.group == "ungated") & (sub.K == best.K)]
    return best, (control.iloc[0] if len(control) else None), (ungated.iloc[0] if len(ungated) else None)


def quality_settings(window, K, h):
    first, last = window
    return dict(window=[first, last], kept=list(range(first, last + 1)), K=K, iteration="block",
                strategy="damped", h=h)


def same_behaviour(a, b):
    """Equal settings once defaults are filled in (``beta`` defaults to 0)."""
    return json.dumps({"beta": 0.0, **a}, sort_keys=True) == json.dumps({"beta": 0.0, **b}, sort_keys=True)


def build_candidates(main, budgets=(4, 6), gate_window="16-21_kept18-19", quality_windows=(),
                     quality_steps=((2, 0.5), (3, 0.5)), include_bands=True, free_policy=False):
    """Shortlist entries (drive.validate schema) for the extension study."""
    main = Path(main)
    sens = main / "profile" / "sensitivity"
    frozen = json.loads((main / "frozen_configs.json").read_text())
    entries = []
    for budget in budgets:
        item = frozen[str(budget)]
        entries.append(dict(name=item["loop"]["name"], role="loop", budget=budget,
                            settings=item["loop"]["settings"]))
        same = item.get("removal_same_window")
        if same and same["settings"]:
            entries.append(dict(name=same["name"], role="removal_same_window", budget=budget,
                                settings=same["settings"]))
        # ``policy_free`` (no FLOPs cap) usually collapses to the best fixed loop; opt in to run it.
        for tag, folder in (("policy_matched", "policy_matched"),) + ((("policy_free", "policy"),) if free_policy else ()):
            path = sens / folder / f"policy_{budget}.json"
            if path.exists():
                entries.append(dict(name=f"d{budget}_{tag}", role="policy", budget=budget,
                                    settings=policy_settings(path)))
    gate_csv = main / "profile" / "explore" / "gate.csv"
    if gate_csv.exists():
        frame = pd.read_csv(gate_csv)
        best, control, ungated = best_gate_candidates(frame, gate_window)
        reference = next((e["settings"] for e in entries if e["role"] == "loop" and e["budget"] == COMPRESSION_BUDGET),
                         None)
        for tag, row in (("gate", best), ("gate_random_control", control), ("gate_ungated", ungated)):
            if row is not None and not (tag == "gate_ungated" and reference is not None
                                        and same_behaviour(json.loads(row.settings), reference)):
                entries.append(dict(name=f"d{COMPRESSION_BUDGET}_{tag}", role="gated",
                                    budget=COMPRESSION_BUDGET, settings=json.loads(row.settings)))
    for window in quality_windows:
        first, last = map(int, window.split("-"))
        for K, h in quality_steps:
            entries.append(dict(name=f"q_{first}_{last}_K{K}_h{str(h).replace('.', 'p')}", role="quality",
                                budget=QUALITY_BUDGET, settings=quality_settings((first, last), K, h)))
    if include_bands and (sens / "shortlist_bands.json").exists():
        entries += [e for e in json.loads((sens / "shortlist_bands.json").read_text())
                    if e["name"] not in {x["name"] for x in entries}]
    seen = {}
    for entry in entries:
        key = json.dumps(entry["settings"], sort_keys=True)
        if entry["name"] in seen and seen[entry["name"]] != key:
            raise ValueError(f"Duplicate candidate name with different settings: {entry['name']}")
        seen[entry["name"]] = key
    return list({e["name"]: e for e in entries}.values())


def write_candidates(output, entries):
    output = Path(output)
    folder = output / "shortlist_ext"
    folder.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (folder / f"{entry['name']}.json").write_text(json.dumps(entry["settings"], indent=2) + "\n")
    (output / "shortlist_ext.json").write_text(json.dumps(entries, indent=2) + "\n")
    return folder


# ---------------------------------------------------------------------------- freeze

def paired_clip_gain(clip_scores, name, baseline="original", resamples=5000, seed=0):
    """Mean paired CLIP cosine difference (variant - original) with a bootstrap 95% interval."""
    a = clip_scores[name].to_numpy(dtype=float)
    b = clip_scores[baseline].to_numpy(dtype=float)
    if len(a) != len(b) or not len(a):
        raise ValueError("CLIP scores must cover the same images")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(diff.mean()), ci95_low=float(low), ci95_high=float(high), images=len(diff))


def read_clip_scores(validation, names):
    frame = {}
    for name in ["original", *names]:
        path = Path(validation) / name / "clip_scores.csv"
        if path.exists():
            frame[name] = pd.read_csv(path).sort_values("index").cosine
    return pd.DataFrame(frame)


def describe(entry, results, extra=None):
    row = results.set_index("variant").loc[entry["name"]]
    out = dict(name=entry["name"], role=entry["role"], settings=entry["settings"], removed=None,
               validation_lpips=float(row.lpips_mean), unique_parameters=int(row.unique_parameters),
               dit_sampling_GFLOPs=None if pd.isna(row.dit_sampling_GFLOPs) else float(row.dit_sampling_GFLOPs))
    out.update(extra or {})
    return out


def select_frozen(results, entries, clip_scores):
    """Frozen entries: reference loop + best compression policy (budget 4), best quality arm (budget 0).

    Compression policy: lowest validation LPIPS among role ``policy``, kept only if it
    beats the plain equal-parameter removal on LPIPS (else recorded as rejected). Quality arm:
    highest mean paired CLIP gain over ``original`` among role ``quality``; ties to
    noise are reported through the bootstrap interval, not hidden.
    """
    available = set(results.variant)
    by_role = lambda role, budget: [e for e in entries if e["role"] == role and e["budget"] == budget
                                    and e["name"] in available]
    table = results.set_index("variant")
    frozen = {}
    reference = by_role("loop", COMPRESSION_BUDGET)
    policies = by_role("policy", COMPRESSION_BUDGET)
    removal = by_role("removal_same_window", COMPRESSION_BUDGET)
    if reference:
        item = dict(loop=describe(reference[0], results))
        if policies:
            best = min(policies, key=lambda e: table.loc[e["name"], "lpips_mean"])
            item["policy_candidates"] = {e["name"]: float(table.loc[e["name"], "lpips_mean"]) for e in policies}
            # A policy spends the same FLOPs as plain removal; it earns a FID slot only by beating it.
            if not removal or table.loc[best["name"], "lpips_mean"] < table.loc[removal[0]["name"], "lpips_mean"]:
                item["policy"] = describe(best, results)
            else:
                item["policy_rejected"] = dict(
                    name=best["name"], lpips=float(table.loc[best["name"], "lpips_mean"]),
                    removal_lpips=float(table.loc[removal[0]["name"], "lpips_mean"]))
        frozen[str(COMPRESSION_BUDGET)] = item
    quality = [e for e in by_role("quality", QUALITY_BUDGET) if e["name"] in clip_scores.columns]
    if quality and "original" in clip_scores.columns:
        gains = {e["name"]: paired_clip_gain(clip_scores, e["name"]) for e in quality}
        best = max(quality, key=lambda e: gains[e["name"]]["mean"])
        frozen[str(QUALITY_BUDGET)] = dict(
            quality=describe(best, results, dict(validation_clip_gain=gains[best["name"]])),
            quality_candidates={n: g["mean"] for n, g in gains.items()})
    if not frozen:
        raise ValueError("Nothing to freeze: run drive validate (and the CLIP report) first")
    return frozen


def freeze(args):
    output = Path(args.output)
    entries = json.loads((output / "shortlist_ext.json").read_text())
    results = pd.read_csv(output / "validation" / "results.csv")
    clip_scores = read_clip_scores(output / "validation", [e["name"] for e in entries])
    frozen = select_frozen(results, entries, clip_scores)
    path = output / "frozen_configs.json"
    path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    print(json.dumps({b: {k: (v["name"] if isinstance(v, dict) and "name" in v else v)
                          for k, v in item.items()} for b, item in frozen.items()}, indent=2))
    print(f"Wrote {path}")


# ------------------------------------------------------------------ exact FLOPs

def sampler_timesteps(steps=50):
    """Original timesteps the respaced sampler visits (checked against ``diffusion.create_diffusion``)."""
    return np.round(np.linspace(0, 999, steps)).astype(int)


def exact_sampling_flops(settings, steps=50, depth=28, hidden=1152, heads=16, mlp=4608, tokens=256):
    """(GFLOPs of ``steps`` CFG sampling steps, mean block calls per step) for a loop config.

    The shared runner counts the container's worst-case schedule, which overstates a
    timestep policy; this averages the rule that actually applies at each step.
    FLOPs are linear in block calls, so two evaluations of the shared estimator give
    the exact value for a fractional mean.
    """
    container = LoopedBlocks([nn.Identity() for _ in range(depth)], **settings)
    calls = container.mean_block_evaluations(sampler_timesteps(steps))

    def flops(n):
        block = SimpleNamespace(mlp=SimpleNamespace(fc1=SimpleNamespace(out_features=mlp)))
        model = SimpleNamespace(
            blocks=SimpleNamespace(retained={"0": block}, schedule=[(0, 1.0, False)] * n, __len__=lambda: n),
            x_embedder=SimpleNamespace(num_patches=tokens, patch_size=(2, 2)),
            pos_embed=torch.zeros(1, 1, hidden), in_channels=4, out_channels=8, num_heads=heads)
        return exp.estimate_dit_flops(model, steps)["dit_sampling_flops"]

    f0, f1 = flops(0), flops(1)
    return (f0 + calls * (f1 - f0)) / 1e9, calls


def exact_flops_report(output):
    """Add exact FLOPs to ``<output>/report/summary.csv``; writes ``summary_exact_flops.csv``."""
    output = Path(output)
    summary = pd.read_csv(output / "report" / "summary.csv")
    exact, calls = [], []
    for name in summary.variant:
        config = output / "frozen" / f"{name}.json"
        if name == "original":
            value = exact_sampling_flops(dict(window=[0, 0], kept=[0], K=1, strategy="naive"))
        elif config.exists():
            value = exact_sampling_flops(json.loads(config.read_text()))
        else:
            value = (np.nan, np.nan)
        exact.append(value[0])
        calls.append(value[1])
    summary["mean_block_calls_exact"] = calls
    summary["dit_sampling_GFLOPs_exact"] = exact
    base = float(summary.loc[summary.variant == "original", "dit_sampling_GFLOPs_exact"].iloc[0])
    summary["flops_reduction_pct_exact"] = 100 * (1 - summary.dit_sampling_GFLOPs_exact / base)
    summary.to_csv(output / "report" / "summary_exact_flops.csv", index=False)
    return summary[["variant", "dit_sampling_GFLOPs", "dit_sampling_GFLOPs_exact", "mean_block_calls_exact",
                    "flops_reduction_pct_exact"]]


def report(args):
    print(exact_flops_report(args.output).round(3).to_string(index=False))


def candidates(args):
    entries = build_candidates(args.main, args.budgets, args.gate_window, args.quality_windows,
                               [(int(s.split(":")[0]), float(s.split(":")[1])) for s in args.quality_steps],
                               not args.no_bands, args.free_policy)
    folder = write_candidates(args.output, entries)
    for e in entries:
        print(f"  {e['name']:<36} {e['role']:<20} budget {e['budget']}")
    print(f"{len(entries)} candidates written to {folder.parent / 'shortlist_ext.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    cand = subs.add_parser("candidates")
    cand.add_argument("--main", default="team_runs/main")
    cand.add_argument("--output", required=True)
    cand.add_argument("--budgets", type=int, nargs="+", default=[4, 6])
    cand.add_argument("--gate-window", default="16-21_kept18-19")
    cand.add_argument("--quality-windows", nargs="*", default=["12-15", "14-17"])
    cand.add_argument("--free-policy", action="store_true", help="also validate the uncapped policy")
    cand.add_argument("--quality-steps", nargs="*", default=["2:0.5", "3:0.5"], help="K:h pairs")
    cand.add_argument("--no-bands", action="store_true")
    fr = subs.add_parser("freeze")
    fr.add_argument("--output", required=True)
    rp = subs.add_parser("report", help="add exact (per-step averaged) FLOPs to report/summary.csv")
    rp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"candidates": candidates, "freeze": freeze, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
