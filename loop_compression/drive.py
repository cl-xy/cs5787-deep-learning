"""Validation driver: run the shortlist through the SHARED runner and freeze one config per budget.

    python -m loop_compression.drive validate --output team_runs/main
    python -m loop_compression.drive freeze   --output team_runs/main

Selection uses validation LPIPS only. The driver cannot run or read the test and
fid splits; those are produced later with the frozen configs by ``final``.
Each variant is a separate ``team_pipeline.cli run`` process, so a crash loses at
most one variant and finished folders are skipped on rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

SELECTION_SPLIT = "validation"
FORBIDDEN_SPLITS = {"test", "fid"}
METHOD = "loop_compression.method:build_model"


def check_split(split):
    if split in FORBIDDEN_SPLITS:
        raise ValueError(f"Settings are selected on validation only; '{split}' is reserved for frozen configs")
    if split != SELECTION_SPLIT:
        raise ValueError(f"Unknown selection split: {split}")
    return split


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_command(output, entry, config_dir, split, limit=None, stride=1):
    """The shared-runner command for one entry; no split policy (see ``run_command``)."""
    command = [sys.executable, "-m", "team_pipeline.cli", "run", "--output", str(output),
               "--split", split, "--name", entry["name"]]
    if limit:
        command += ["--limit", str(limit)]
    if stride != 1:
        command += ["--stride", str(stride)]
    if entry.get("settings") is not None:
        command += ["--method", METHOD, "--method-config", str(Path(config_dir) / f"{entry['name']}.json")]
    else:
        command += ["--method", "removal", "--removed", *map(str, entry["removed"])]
    return command


def run_command(output, entry, config_dir, split=SELECTION_SPLIT, limit=None, stride=1):
    """Selection-time command: validation only."""
    check_split(split)
    return build_command(output, entry, config_dir, split, limit=limit, stride=stride)


def load_shortlist(path, budgets=None):
    entries = json.loads(Path(path).read_text())
    if budgets is not None:
        entries = [e for e in entries if e["budget"] in set(budgets)]
    if not entries:
        raise ValueError("Shortlist is empty after filtering")
    return entries


def config_key(entry):
    """Identical settings (or removed blocks) give identical, deterministic runs."""
    if entry.get("settings") is not None:
        return json.dumps(entry["settings"], sort_keys=True)
    return "removed:" + json.dumps(sorted(entry["removed"]))


def validate(args):
    check_split(SELECTION_SPLIT)
    output = Path(args.output)
    sens = output / "profile" / "sensitivity"
    shortlist = Path(args.shortlist) if args.shortlist else sens / "shortlist.json"
    entries = load_shortlist(shortlist, args.budgets)
    config_dir = shortlist.parent / "shortlist"
    split_dir = output / SELECTION_SPLIT
    log, jobs, aliases, first_name = [], [], [], {}
    if not (split_dir / "original").exists():
        jobs.append(dict(name="original", command=[
            sys.executable, "-m", "team_pipeline.cli", "run", "--output", str(output), "--split", SELECTION_SPLIT,
            "--method", "original", "--name", "original", *(["--limit", str(args.limit)] if args.limit else [])]))
    for entry in entries:
        key = config_key(entry)
        if key in first_name:  # duplicate of an earlier entry: alias instead of re-running
            aliases.append((entry["name"], first_name[key]))
            continue
        first_name[key] = entry["name"]
        if (split_dir / entry["name"]).exists():
            continue
        record = dict(name=entry["name"], role=entry["role"], budget=entry["budget"])
        if entry.get("settings") is not None:
            record["config_sha256"] = file_sha256(config_dir / f"{entry['name']}.json")
        jobs.append(dict(name=entry["name"], command=run_command(output, entry, config_dir, limit=args.limit),
                         record=record))
    print(f"{len(jobs)} variants to run; {len(aliases)} duplicate configs will be aliased", flush=True)
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        if args.dry_run:
            print("   ", " ".join(job["command"]))
            continue
        subprocess.run(job["command"], check=True)
        if "record" in job:
            log.append(job["record"])
    if args.dry_run:
        return
    for name, target in aliases:
        link = split_dir / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)  # relative link inside the split folder
        log.append(dict(name=name, alias_of=target))
    driver_log = split_dir / "driver_log.json"
    previous = json.loads(driver_log.read_text()) if driver_log.exists() else []
    driver_log.write_text(json.dumps(previous + log, indent=2) + "\n")
    names = ["original"] + [e["name"] for e in entries]
    evaluate(output, names)


def evaluate(output, names):
    """LPIPS and resource table for all names, plus one comparison grid per budget."""
    from team_pipeline.data import load_protocol
    from team_pipeline import experiment as exp
    root = Path(output) / SELECTION_SPLIT
    results = exp.compare_results(root, names)
    print(results[["variant", "lpips_mean", "lpips_std", "unique_parameters", "dit_sampling_GFLOPs",
                   "latency_mean_s"]].to_string(index=False))
    pairs = load_protocol(Path(output) / "protocol.json")["splits"][SELECTION_SPLIT]
    by_budget = {}
    for name in names:
        if name.startswith("d") and "_" in name and name[1:name.index("_")].isdigit():
            by_budget.setdefault(int(name[1:name.index("_")]), []).append(name)
    for budget, group in by_budget.items():
        exp.comparison_grid(root, ["original"] + group, pairs)
        shutil.move(root / "comparison_grid.png", root / f"comparison_grid_d{budget}.png")
    return results


def paired_difference(paired, a, b, resamples=5000, seed=0):
    """Mean of (LPIPS_a - LPIPS_b) over images with a bootstrap 95% interval.

    Negative means ``a`` is closer to the original than ``b``.
    """
    left = paired[paired.variant == a].sort_values("index").paired_lpips.to_numpy()
    right = paired[paired.variant == b].sort_values("index").paired_lpips.to_numpy()
    if len(left) != len(right) or not len(left):
        raise ValueError("Variants need the same nonempty set of images")
    diff = left - right
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(diff.mean()), ci95_low=float(low), ci95_high=float(high), images=len(diff))


def select_frozen(results, paired, entries):
    """One looped config per budget, by validation LPIPS, with its baselines.

    Returns {budget: {...}}. ``loop_beats_removal`` compares the chosen loop with the
    STRONGEST equal-parameter removal baseline on validation (lowest LPIPS); the
    bootstrap interval shows whether that difference is distinguishable from noise.
    """
    check_split(SELECTION_SPLIT)
    table = results.set_index("variant")
    by_name = {e["name"]: e for e in entries}
    frozen = {}
    for budget in sorted({e["budget"] for e in entries}):
        group = [e for e in entries if e["budget"] == budget and e["name"] in table.index]
        loops = [e for e in group if e["role"] == "loop"]
        removals = [e for e in group if e["role"].startswith("removal")]
        if not loops or not removals:
            continue
        best_loop = min(loops, key=lambda e: table.loc[e["name"], "lpips_mean"])
        best_removal = min(removals, key=lambda e: table.loc[e["name"], "lpips_mean"])
        prefix = best_loop["name"][:-len("_loop")]
        companions = {role: by_name.get(f"{prefix}_{role}") for role in ("naive_same_window", "removal_same_window")}

        def describe(entry):
            row = table.loc[entry["name"]]
            return dict(name=entry["name"], role=entry["role"], settings=entry.get("settings"),
                        removed=entry.get("removed"), validation_lpips=float(row.lpips_mean),
                        unique_parameters=int(row.unique_parameters),
                        dit_sampling_GFLOPs=None if pd.isna(row.dit_sampling_GFLOPs) else float(row.dit_sampling_GFLOPs))

        difference = paired_difference(paired, best_loop["name"], best_removal["name"])
        frozen[int(budget)] = dict(
            loop=describe(best_loop), strongest_removal=describe(best_removal),
            naive_same_window=describe(companions["naive_same_window"]) if companions["naive_same_window"] else None,
            removal_same_window=describe(companions["removal_same_window"]) if companions["removal_same_window"] else None,
            loop_minus_removal_lpips=difference,
            loop_beats_removal=bool(difference["mean"] < 0 and difference["ci95_high"] < 0),
            equal_parameters=bool(table.loc[best_loop["name"], "unique_parameters"]
                                  == table.loc[best_removal["name"], "unique_parameters"]))
    if not frozen:
        raise ValueError("No budget has both a looped and a removal result on validation")
    return frozen


def freeze(args):
    output = Path(args.output)
    root = output / SELECTION_SPLIT
    entries = load_shortlist(Path(args.shortlist) if args.shortlist
                             else output / "profile" / "sensitivity" / "shortlist.json", args.budgets)
    results = pd.read_csv(root / "results.csv")
    paired = pd.read_csv(root / "paired_metrics.csv")
    frozen = select_frozen(results, paired, entries)
    path = output / "frozen_configs.json"
    path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    for budget, item in frozen.items():
        diff = item["loop_minus_removal_lpips"]
        print(f"budget {budget}: loop {item['loop']['name']} LPIPS {item['loop']['validation_lpips']:.3f} vs "
              f"removal {item['strongest_removal']['name']} {item['strongest_removal']['validation_lpips']:.3f}; "
              f"paired diff {diff['mean']:+.3f} [{diff['ci95_low']:+.3f}, {diff['ci95_high']:+.3f}] "
              f"-> {'LOOP BEATS removal' if item['loop_beats_removal'] else 'no reliable win over removal'}")
    print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "freeze"):
        sub = subs.add_parser(name)
        sub.add_argument("--output", required=True)
        sub.add_argument("--shortlist", help="default: <output>/profile/sensitivity/shortlist.json")
        sub.add_argument("--budgets", type=int, nargs="+")
        if name == "validate":
            sub.add_argument("--limit", type=int, help="smoke runs: first N validation pairs only")
            sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    {"validate": validate, "freeze": freeze}[args.command](args)


if __name__ == "__main__":
    main()
