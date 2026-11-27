"""Final benchmark with FROZEN configs only (Task 6).

    python -m loop_compression.final test   --output team_runs/main
    python -m loop_compression.final timing --output team_runs/main
    python -m loop_compression.final fid    --output team_runs/main --budget 4
    python -m loop_compression.final report --output team_runs/main

Nothing here selects settings. It reads ``frozen_configs.json`` (written from
validation by ``drive freeze``) and refuses to run if it is missing. Latency is
measured on the Mac MPS device and is labelled as such; never pool it with CUDA.
Close other GPU/CPU-heavy work during ``test``, ``timing`` and ``fid``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from team_pipeline.cli import select_pairs
from team_pipeline.data import load_protocol
from .drive import build_command

ROLES = ("loop", "strongest_removal")           # arms compared in the headline results
OPTIONAL_ROLES = ("naive_same_window", "removal_same_window")
EXTENSION_ROLES = ("policy", "gated", "quality")  # written by ``loop_compression.ext freeze``


def load_frozen(output, budgets=None, extras=False):
    """Frozen entries as [{name, role, budget, settings|removed, ...}] in a stable order."""
    path = Path(output) / "frozen_configs.json"
    if not path.exists():
        raise FileNotFoundError("frozen_configs.json missing: run `drive validate` and `drive freeze` first")
    frozen = json.loads(path.read_text())
    entries, seen = [], set()
    for budget in sorted(frozen, key=int):
        if budgets is not None and int(budget) not in set(budgets):
            continue
        for role in ROLES + (OPTIONAL_ROLES if extras else ()) + EXTENSION_ROLES:
            item = frozen[budget].get(role)
            if item is None or item["name"] in seen:  # one run can fill several roles
                continue
            seen.add(item["name"])
            entries.append(dict(name=item["name"], role=role, budget=int(budget),
                                settings=item.get("settings"), removed=item.get("removed"),
                                validation_lpips=item["validation_lpips"]))
    if not entries:
        raise ValueError("No frozen configurations for the requested budgets")
    return entries


def write_frozen_configs(output, entries):
    """Provenance copies of the exact settings used; returns the folder."""
    folder = Path(output) / "frozen"
    folder.mkdir(exist_ok=True)
    for e in entries:
        if e["settings"] is not None:
            path = folder / f"{e['name']}.json"
            text = json.dumps(e["settings"], indent=2) + "\n"
            if path.exists() and path.read_text() != text:
                raise ValueError(f"Frozen config changed on disk: {path}")
            path.write_text(text)
    return folder


def plan_commands(output, entries, split, config_dir, limit=None, stride=1):
    """(name, command) for the baseline and every frozen entry on ``split``."""
    if split not in {"test", "fid"}:
        raise ValueError("final runs use the test or fid split")
    subset = [*( ["--limit", str(limit)] if limit else []),
              *( ["--stride", str(stride)] if stride != 1 else [])]
    jobs = [("original", [sys.executable, "-m", "team_pipeline.cli", "run", "--output", str(output),
                          "--split", split, "--method", "original", "--name", "original", *subset])]
    for e in entries:
        jobs.append((e["name"], build_command(output, e, config_dir, split,
                                               limit=limit, stride=stride)))
    return jobs


def selected_split_pairs(output, split, stride=1, limit=None):
    """Load and select the exact protocol pairs expected from every final run."""
    protocol = load_protocol(Path(output) / "protocol.json")
    pairs = protocol["splits"].get(split)
    if pairs is None:
        raise ValueError(f"Protocol has no '{split}' split")
    return select_pairs(pairs, stride=stride, limit=limit)


def folder_state(folder, expected_pairs):
    """Return missing/complete/partial/mismatched for one generated variant."""
    folder = Path(folder)
    if not folder.exists():
        return "missing"
    manifest_path = folder / "manifest.json"
    samples_path = folder / "samples.csv"
    if not manifest_path.exists() or not samples_path.exists():
        return "partial"
    try:
        manifest = json.loads(manifest_path.read_text())
        actual_pairs = [list(pair) for pair in manifest["pairs"]]
        expected_pairs = [list(pair) for pair in expected_pairs]
        samples = pd.read_csv(samples_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return "mismatched"
    if actual_pairs != expected_pairs:
        return "mismatched"
    columns = ["index", "label", "seed"]
    if not set(columns) <= set(samples.columns):
        return "mismatched"
    expected_rows = pd.DataFrame(
        [(i, label, seed) for i, (label, seed) in enumerate(expected_pairs)], columns=columns)
    actual_rows = samples[columns]
    if len(actual_rows) < len(expected_rows) and actual_rows.equals(expected_rows.iloc[:len(actual_rows)]):
        return "partial"
    if not actual_rows.equals(expected_rows):
        return "mismatched"
    expected_images = {f"image_{i:04d}.pt" for i in range(len(expected_pairs))}
    actual_images = {path.name for path in folder.glob("image_*.pt")}
    if actual_images == expected_images:
        return "complete"
    return "partial" if actual_images < expected_images else "mismatched"


def run_jobs(output, split, jobs, expected_pairs, dry_run=False):
    root = Path(output) / split
    for index, (name, command) in enumerate(jobs, 1):
        folder = root / name
        state = folder_state(folder, expected_pairs)
        if state == "complete":
            print(f"[{index}/{len(jobs)}] {name}: complete ({len(expected_pairs)} images), skipping")
            continue
        if state != "missing":
            raise RuntimeError(
                f"Existing result folder is {state}: {folder}. "
                "Move it aside before rerunning; no files were changed.")
        print(f"[{index}/{len(jobs)}] {split}/{name}", flush=True)
        if dry_run:
            print("   ", " ".join(command))
            continue
        subprocess.run(command, check=True)


def run_test(args):
    entries = load_frozen(args.output, args.budgets, extras=args.extras)
    config_dir = write_frozen_configs(args.output, entries)
    expected_pairs = selected_split_pairs(args.output, "test", args.stride, args.limit)
    jobs = plan_commands(args.output, entries, "test", config_dir,
                         limit=args.limit, stride=args.stride)
    run_jobs(args.output, "test", jobs, expected_pairs, args.dry_run)
    if not args.dry_run:
        names = [name for name, _ in jobs]
        subprocess.run([sys.executable, "-m", "team_pipeline.cli", "evaluate", "--output", str(args.output),
                        "--split", "test", "--names", *names], check=True)


def run_fid(args):
    entries = load_frozen(args.output, [args.budget], extras=False)
    config_dir = write_frozen_configs(args.output, entries)
    expected_pairs = selected_split_pairs(args.output, "fid", args.stride, args.limit)
    jobs = plan_commands(args.output, entries, "fid", config_dir,
                         limit=args.limit, stride=args.stride)
    run_jobs(args.output, "fid", jobs, expected_pairs, args.dry_run)
    if args.dry_run:
        return
    names = [name for name, _ in jobs]
    base = [sys.executable, "-m", "team_pipeline.cli"]
    subprocess.run(base + ["clip", "--output", str(args.output), "--split", "fid", "--names", *names], check=True)
    fid = base + ["fid", "--output", str(args.output), "--split", "fid", "--names", *names]
    if args.reference:
        fid += ["--reference", args.reference]
    subprocess.run(fid, check=True)


# ----------------------------------------------------------------------- timing

def interleaved_order(names, repetitions):
    """Alternate the variant order each repetition so drift does not favour one variant."""
    order = []
    for rep in range(repetitions):
        names_rep = list(names) if rep % 2 == 0 else list(reversed(names))
        order += [(rep, name) for name in names_rep]
    return order


def summarize_timing(latencies):
    """latencies: DataFrame(variant, rep, latency_s) -> per-variant mean/std and relative speed.

    The std across repetition means shows run-to-run drift; the image-level std shows spread.
    """
    per_rep = latencies.groupby(["variant", "rep"]).latency_s.mean().reset_index()
    rows = []
    for variant, group in latencies.groupby("variant"):
        reps = per_rep[per_rep.variant == variant].latency_s
        rows.append(dict(variant=variant, repetitions=len(reps), images_per_rep=int(len(group) / len(reps)),
                         latency_mean_s=float(group.latency_s.mean()),
                         latency_image_std_s=float(group.latency_s.std(ddof=1)) if len(group) > 1 else 0.0,
                         latency_rep_mean_std_s=float(reps.std(ddof=1)) if len(reps) > 1 else 0.0,
                         device="mps"))
    table = pd.DataFrame(rows).set_index("variant")
    table["speedup_vs_original"] = table.loc["original", "latency_mean_s"] / table.latency_mean_s
    return table.reset_index()


def run_timing(args):
    """Repeat timing on a few test images with the shared run_variant, interleaving variants."""
    from team_pipeline import experiment as exp
    from .sensitivity import _load_run_context
    from .method import build_model
    root, protocol, config = _load_run_context(args.output, args.repo)
    entries = load_frozen(args.output, args.budgets, extras=False)
    pairs = protocol["splits"]["test"][:args.images]
    names = ["original"] + [e["name"] for e in entries]
    by_name = {e["name"]: e for e in entries}
    folder = root / "timing"
    rows = []
    for rep, name in interleaved_order(names, args.repetitions):
        out = folder / f"rep{rep}_{name}"
        if not (out / "samples.csv").exists():
            if args.dry_run:
                print("would time", out)
                continue
            if name == "original":
                exp.run_variant(config, pairs, out)
            elif by_name[name]["settings"] is not None:
                settings = by_name[name]["settings"]
                exp.run_variant(config, pairs, out, model_transform=lambda m, s=settings: build_model(m, s),
                                method_metadata=dict(method="loop_compression", settings=settings))
            else:
                exp.run_variant(config, pairs, out, mode="removal", removed=by_name[name]["removed"])
        if (out / "samples.csv").exists():
            samples = pd.read_csv(out / "samples.csv")
            rows.append(pd.DataFrame(dict(variant=name, rep=rep, latency_s=samples.latency_s)))
    if args.dry_run:
        return
    table = summarize_timing(pd.concat(rows, ignore_index=True))
    table.to_csv(folder / "timing_summary.csv", index=False)
    print(table.to_string(index=False))


# ----------------------------------------------------------------------- report

def build_summary(test_results, frozen_entries, fid_rows=None, clip_rows=None, timing=None):
    """One row per variant: resources, LPIPS, optional FID/CLIP/timing, role and budget."""
    roles = {e["name"]: (e["role"], e["budget"]) for e in frozen_entries}
    table = test_results.copy()
    table["role"] = table.variant.map(lambda v: roles.get(v, ("baseline", 0))[0])
    table["budget"] = table.variant.map(lambda v: roles.get(v, ("baseline", 0))[1])
    table["unique_parameters_M"] = table.unique_parameters / 1e6
    for extra, key in [(fid_rows, "fid"), (clip_rows, "clip"), (timing, "timing")]:
        if extra is not None and len(extra):
            extra = extra.copy()
            metadata = {column: f"{key}_{column}" for column in
                        ("images", "class_count", "metric", "rank_deficient")
                        if column in extra.columns}
            extra = extra.rename(columns=metadata)
            extra = extra.drop(columns=[c for c in extra.columns if c in table.columns and c != "variant"])
            table = table.merge(extra, on="variant", how="left")
    return table


def savings_table(summary):
    """Parameters vs compute: looping saves parameters but not FLOPs."""
    cols = ["variant", "role", "budget", "unique_parameters_M", "parameter_reduction_pct",
            "dit_sampling_GFLOPs", "flops_reduction_pct", "effective_block_evaluations_per_forward"]
    return summary[[c for c in cols if c in summary.columns]].round(3)


def read_json_rows(root, names, filename):
    rows = []
    for name in names:
        path = Path(root) / name / filename
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def plot_pareto(summary, out_dir, fid=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = {"baseline": "s", "loop": "o", "strongest_removal": "^",
               "naive_same_window": "x", "removal_same_window": "v"}
    panels = [("unique_parameters_M", "lpips_mean", "parameters (M)", "LPIPS vs original (lower = closer)"),
              ("dit_sampling_GFLOPs", "lpips_mean", "DiT sampling GFLOPs", "LPIPS vs original (lower = closer)")]
    if fid and "fid_vs_original" in summary.columns:
        panels += [("unique_parameters_M", "fid_vs_original", "parameters (M)", "FID-N vs original"),
                   ("dit_sampling_GFLOPs", "fid_vs_original", "DiT sampling GFLOPs", "FID-N vs original")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), constrained_layout=True, squeeze=False)
    for ax, (x, y, xlabel, ylabel) in zip(axes[0], panels):
        data = summary.dropna(subset=[x, y])
        for role, group in data.groupby("role"):
            ax.scatter(group[x], group[y], marker=markers.get(role, "o"), label=role, s=40)
            for _, row in group.iterrows():
                ax.annotate(row.variant, (row[x], row[y]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.savefig(out_dir / ("pareto_fid.png" if fid else "pareto_lpips.png"), dpi=170)
    plt.close(fig)


def run_report(args):
    output = Path(args.output)
    frozen = load_frozen(output, args.budgets, extras=True)
    results = pd.read_csv(output / "test" / "results.csv")
    names = list(results.variant)
    fid = read_json_rows(output / "fid", names, "fid.json")
    clip = read_json_rows(output / "fid", names, "clip.json")
    timing_path = output / "timing" / "timing_summary.csv"
    timing = pd.read_csv(timing_path).rename(columns=lambda c: c if c == "variant" else f"mps_{c}") \
        if timing_path.exists() else None
    summary = build_summary(results, frozen, fid, clip, timing)
    report = output / "report"
    report.mkdir(exist_ok=True)
    summary.to_csv(report / "summary.csv", index=False)
    savings = savings_table(summary)
    savings.to_csv(report / "parameter_vs_flop_savings.csv", index=False)
    (report / "parameter_vs_flop_savings.md").write_text(
        "```\n" + savings.to_string(index=False) + "\n```\n")
    plot_pareto(summary, report)
    if "fid_vs_original" in summary.columns and summary.fid_vs_original.notna().any():
        plot_pareto(summary, report, fid=True)
    print(savings.to_string(index=False))
    print(f"Wrote {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("test", "timing", "fid", "report"):
        sub = subs.add_parser(name)
        sub.add_argument("--output", required=True)
        sub.add_argument("--repo", default="third_party/DiT")
        if name == "fid":
            sub.add_argument("--budget", type=int, required=True)
            sub.add_argument("--reference", help="ADM ImageNet-256 reference .npz (mu/sigma)")
        else:
            sub.add_argument("--budgets", type=int, nargs="+")
        if name in ("test", "fid"):
            sub.add_argument("--limit", type=int, help="use at most N selected pairs")
            sub.add_argument("--stride", type=int, default=1,
                             help="select every Nth split pair before applying --limit")
            sub.add_argument("--dry-run", action="store_true")
        if name == "test":
            sub.add_argument("--extras", action="store_true", help="also run the naive/removal same-window arms")
        if name == "timing":
            sub.add_argument("--images", type=int, default=8)
            sub.add_argument("--repetitions", type=int, default=3)
            sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    {"test": run_test, "timing": run_timing, "fid": run_fid, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
