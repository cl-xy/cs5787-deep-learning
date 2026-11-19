"""Read-only report tables and figures from completed experiment artifacts.

Nothing here generates images, loads weights, reruns metrics or calls ``load_run``,
so a run produced by older code stays reportable. Figures use ``matplotlib.figure.Figure``
directly and never touch the global pyplot backend.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .evaluation import SAMPLE_KEYS, check_manifests_consistent
from .runtime import write_json

REPORT_COLORS = ("#334155", "#D97706", "#059669", "#7C3AED", "#DC2626", "#0284C7")
REPORT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}
MIN_FULL_RANK_FID_IMAGES = 2049
CAPTION_COLOR = "#475569"
FIGURE_DPI = 220
REPORT_SPLITS = ("validation", "test")


# ---- loading and validation --------------------------------------------------------
def _method_label(name, manifest):
    schedule = manifest.get("schedule", {})
    mode = schedule.get("mode", manifest.get("method_metadata", {}).get("method", name))
    if name == "original":
        return "Original DiT"
    if mode == "damped":
        alpha = schedule.get("alpha")
        policy = schedule.get("damping_policy", "inverse_k" if alpha == "inverse_k" else "fixed")
        return "Damped (1/k)" if policy == "inverse_k" else f"Damped (fixed α={alpha})"
    return {"removal": "Removal", "loop": "Looping", "xsa_reuse": "XSA reuse"}.get(mode, name)


def _damping_policy(manifest):
    schedule = manifest.get("schedule", {})
    return schedule.get("damping_policy", "fixed" if schedule.get("mode") == "damped" else "none")


def _check_names(names):
    if not names or len(set(names)) != len(names) or "original" not in names:
        raise ValueError("Use unique variant names including original")
    if any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}
           for name in names):
        raise ValueError("Variant names must be single folder names")


def _expected_samples(baseline):
    expected = pd.DataFrame([(i, *pair) for i, pair in enumerate(baseline["pairs"])],
                            columns=SAMPLE_KEYS)
    if len(expected) < 2 or expected["seed"].duplicated().any():
        raise ValueError("Expected at least two distinct generated samples")
    return expected


def _check_variant_rows(name, root, summary, paired, expected):
    """Validate one variant's saved rows and return its paired rows."""
    samples = pd.read_csv(root / name / "samples.csv")
    rows = paired.loc[paired["variant"] == name].reset_index(drop=True)
    images = int(summary.loc[summary["variant"] == name, "images"].iloc[0])
    if (not samples[SAMPLE_KEYS].equals(expected) or not rows[SAMPLE_KEYS].equals(expected)
            or images != len(expected)):
        raise ValueError(f"Incomplete or inconsistent saved samples: {name}")
    measurements = rows[["paired_lpips", "latency_s"]]
    if (not np.isfinite(measurements.to_numpy()).all()
            or (rows["latency_s"] <= 0).any() or (rows["paired_lpips"] < 0).any()):
        raise ValueError(f"Invalid saved measurements: {name}")
    if not np.allclose(samples["latency_s"], rows["latency_s"]):
        raise ValueError(f"Paired latency rows differ from samples.csv: {name}")
    result = summary.loc[summary["variant"] == name].iloc[0]
    if (not np.isclose(rows["paired_lpips"].mean(), result["lpips_mean"])
            or not np.isclose(rows["latency_s"].mean(), result["latency_mean_s"])):
        raise ValueError(f"Summary means differ from paired rows: {name}")
    return rows


def _read_variants(root, names, summary, paired, baseline):
    """Check every variant against the baseline; return ``(manifests, paired_rows)``."""
    expected = _expected_samples(baseline)
    manifests, frames = {}, []
    for name in names:
        manifest = json.loads((root / name / "manifest.json").read_text())
        check_manifests_consistent(baseline, manifest, name)
        frames.append(_check_variant_rows(name, root, summary, paired, expected))
        manifests[name] = manifest
    return manifests, pd.concat(frames, ignore_index=True)


def _read_fid(root, names, summary, baseline):
    """Attach saved FID scores to ``summary`` when their provenance matches. Returns metadata."""
    extended_path = root / "results_extended.csv"
    evaluation_path = root / "evaluation_extended.json"
    if not extended_path.exists():
        return {}
    extended = pd.read_csv(extended_path)
    if "fid" not in extended:
        return {}
    if not evaluation_path.exists():
        raise ValueError("FID CSV is missing its evaluation_extended.json provenance")
    evaluation = json.loads(evaluation_path.read_text())
    if (evaluation.get("generation_code_sha256") != baseline["shared_code_sha256"]
            or evaluation.get("generation_runtime") != baseline["runtime"]
            or evaluation.get("images_per_variant") != len(baseline["pairs"])
            or not set(names).issubset(evaluation.get("variants", []))
            or not evaluation.get("fid")):
        raise ValueError("FID evaluation provenance does not match the saved generation")
    if extended["variant"].duplicated().any():
        raise ValueError("Duplicate variants in results_extended.csv")
    extended = extended.set_index("variant").loc[names]
    if not extended["images"].astype(int).eq(len(baseline["pairs"])).all():
        raise ValueError("FID summary sample counts do not match")
    fid_values = pd.to_numeric(extended["fid"], errors="raise")
    if not np.isfinite(fid_values).all() or (fid_values < 0).any():
        raise ValueError("Invalid saved FID scores")
    summary["fid"] = summary["variant"].map(fid_values)
    return evaluation


def _derive_metrics(summary, paired, manifests, names):
    """Add labels and metrics relative to the original; return the per-class LPIPS table."""
    summary["method"] = [_method_label(name, manifests[name]) for name in names]
    summary["damping_policy"] = [_damping_policy(manifests[name]) for name in names]
    original = summary.loc[summary["variant"] == "original"].iloc[0]
    summary["speedup_vs_original"] = original["latency_mean_s"] / summary["latency_mean_s"]
    summary["parameter_reduction_pct"] = 100 * (
        1 - summary["unique_parameters"] / original["unique_parameters"])
    latency_sd = paired.groupby("variant")["latency_s"].std(ddof=1)
    summary["latency_sample_sd_s"] = summary["variant"].map(latency_sd)
    return paired.groupby(["variant", "label"], sort=False)["paired_lpips"].agg(
        samples="size", lpips_mean="mean", lpips_std="std").reset_index()


def _write_report_tables(run_root, root, split, summary, per_class, baseline, manifests,
                         evaluation, names):
    """Write report_summary.csv, per_class_lpips.csv and report_metadata.json."""
    output = run_root / "report" / split
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "report_summary.csv", index=False)
    per_class = per_class.copy()
    per_class.to_csv(output / "per_class_lpips.csv", index=False)
    inputs = [root / "results.csv", root / "paired_metrics.csv", root / "original/manifest.json",
              root / "results_extended.csv", root / "evaluation_extended.json"]
    inputs = [path for path in inputs if path.exists()]
    labels = pd.Series([label for label, _seed in baseline["pairs"]])
    write_json(output / "report_metadata.json", dict(
        run=str(run_root), split=split, images_per_variant=len(baseline["pairs"]),
        labels=sorted(labels.unique().tolist()),
        class_counts={str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        generation_code_sha256=baseline["shared_code_sha256"], runtime=baseline["runtime"],
        method_labels=dict(zip(names, summary["method"])),
        saved_schedules={name: manifests[name].get("schedule", {}) for name in names},
        fid=evaluation.get("fid"),
        uncertainty="Error bars show sample SD, not confidence intervals; FID has no error bar",
        input_sha256={str(p.relative_to(run_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in inputs}))
    return output


def load_report_data(run_root, split="validation", names=None):
    """Read CSV/JSON only; old generation code/environments remain reportable.

    Every method must have complete, matching manifests/sample rows. Old fixed
    damping is labelled from its saved manifest, never from today's ALPHA value.
    Extended FID is used only with matching recorded generation provenance.
    """
    if split not in REPORT_SPLITS:
        raise ValueError("Use validation or test")
    run_root = Path(run_root).expanduser().resolve()
    root = run_root / split
    summary = pd.read_csv(root / "results.csv")
    paired = pd.read_csv(root / "paired_metrics.csv")
    if summary["variant"].duplicated().any():
        raise ValueError("Duplicate variants in results.csv")
    names = list(summary["variant"] if names is None else names)
    _check_names(names)
    summary = summary.set_index("variant").loc[names].reset_index()
    baseline = json.loads((root / "original/manifest.json").read_text())
    manifests, paired = _read_variants(root, names, summary, paired, baseline)
    evaluation = _read_fid(root, names, summary, baseline)
    per_class = _derive_metrics(summary, paired, manifests, names)
    output = _write_report_tables(run_root, root, split, summary, per_class, baseline,
                                  manifests, evaluation, names)
    return dict(root=root, run_root=run_root, split=split, summary=summary, paired=paired,
                per_class=per_class, baseline=baseline, manifests=manifests,
                evaluation=evaluation, output=output)


def find_report_run(output_base, split="validation", current=None):
    """The run to report: ``current`` if it has results, else the newest completed run."""
    if current is not None and (Path(current) / split / "results.csv").is_file():
        return Path(current)
    completed = [p for p in Path(output_base).glob("*") if (p / split / "results.csv").is_file()]
    if not completed:
        raise FileNotFoundError("No completed results.csv found; set REPORT_RUN_ROOT to a saved run")
    return max(completed, key=lambda p: (p / split / "results.csv").stat().st_mtime_ns)


# ---- plotting ----------------------------------------------------------------------
def _report_context(report):
    manifest = report["baseline"]
    config = manifest["config"]
    classes = len({label for label, _seed in manifest["pairs"]})
    device = manifest.get("runtime", {}).get("device", config.get("device", "unknown"))
    return (f"{report['run_root'].name} | {report['split']} | "
            f"N={len(manifest['pairs'])}/method, {classes} classes | "
            f"{config.get('sampling_steps', '?')} steps, CFG={config.get('cfg_scale', '?')}, "
            f"{config.get('dtype', 'float32')} | {device}")


def _save_report_figure(report, figure, stem, note):
    width = max(70, int(figure.get_figwidth() * 12))
    caption = (textwrap.fill(_report_context(report), width=width) + "\n"
               + textwrap.fill(note, width=width))
    figure.text(0.015, 0.012, caption, fontsize=7, va="bottom", color=CAPTION_COLOR)
    figure.tight_layout(rect=(0, 0.13, 1, 0.96))
    paths = {}
    for extension in ("png", "pdf"):
        path = report["output"] / f"{stem}.{extension}"
        figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        paths[extension] = path
    return paths


def _report_bars(ax, report, values, ylabel, title, errors=None, decimals=2):
    values = np.asarray(values, dtype=float)
    labels = report["summary"]["method"].str.replace(" (", "\n(", regex=False)
    colors = [REPORT_COLORS[i % len(REPORT_COLORS)] for i in range(len(values))]
    finite = np.isfinite(values)
    if not finite.any():
        ax.set_title(title)
        ax.text(0.5, 0.5, "Metric not available in this saved run", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    heights = np.where(finite, values, 0)
    uncertainty = (np.zeros_like(heights) if errors is None
                   else np.nan_to_num(np.asarray(errors, float), nan=0))
    bars = ax.bar(np.arange(len(values)), heights, color=colors,
                  yerr=uncertainty if errors is not None else None, capsize=3,
                  error_kw={"elinewidth": 1, "ecolor": "#334155"}, width=0.68)
    ymax = max(float(np.max(heights + uncertainty)), 0.01)
    ax.set_ylim(0, ymax * 1.27)
    for bar, value, error in zip(bars, values, uncertainty):
        text = f"{value:.{decimals}f}" if np.isfinite(value) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + error + ymax * 0.025,
                text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(values)), labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", visible=False)


def _figure(*args, **kwargs):
    """A pyplot-free figure created under the report style."""
    with matplotlib.rc_context(REPORT_STYLE):
        figure = Figure(**kwargs)
        return figure, figure.subplots(*args, squeeze=False)


def plot_report_quality(report):
    summary = report["summary"]
    with matplotlib.rc_context(REPORT_STYLE):
        fig, axes = _figure(1, 2, figsize=(12.5, 4.8))
        fig.suptitle("Output preservation and real-reference FID", fontsize=13)
        _report_bars(axes[0, 0], report, summary["lpips_mean"], "Paired LPIPS ↓",
                     "Deviation from original DiT", errors=summary["lpips_std"], decimals=3)
        n = len(report["baseline"]["pairs"])
        fid_title = (f"Exploratory pooled FID-{n}" if n < MIN_FULL_RANK_FID_IMAGES
                     else f"Pooled FID-{n}")
        fid = summary["fid"] if "fid" in summary else pd.Series(np.nan, index=summary.index)
        _report_bars(axes[0, 1], report, fid, "FID vs. real reference ↓", fid_title, decimals=2)
        reference_count = report["evaluation"].get("fid", {}).get("reference_images",
                                                                  "not recorded")
        note = (f"LPIPS bars: mean ± sample SD. FID is pooled across classes against "
                f"{reference_count} real images; no FID confidence intervals are estimated. "
                "Small-sample FID rankings are exploratory.")
        return _save_report_figure(report, fig, "quality_metrics", note)


def _memory_panel(summary):
    """First available accelerator-memory column as ``(key, label, title)``, else None."""
    options = [
        ("peak_GPU_allocated_GiB", "Peak CUDA allocated memory (GiB) ↓",
         "Peak allocated during sampling"),
        ("max_post_sample_MPS_allocated_GiB", "Post-sample MPS allocated memory (GiB) ↓",
         "MPS allocated snapshot (not peak)"),
    ]
    return next(((key, label, title) for key, label, title in options
                 if key in summary and summary[key].notna().any()), None)


def plot_report_efficiency(report):
    summary = report["summary"]
    with matplotlib.rc_context(REPORT_STYLE):
        fig, axes = _figure(2, 2, figsize=(12.5, 8.3))
        fig.suptitle("Measured inference cost and model storage", fontsize=13)
        _report_bars(axes[0, 0], report, summary["latency_mean_s"], "Seconds / image ↓",
                     "Sampling + decode + CPU transfer",
                     errors=summary["latency_sample_sd_s"], decimals=2)
        _report_bars(axes[0, 1], report, summary["unique_parameters"] / 1e6,
                     "Unique parameters (millions) ↓", "DiT parameters; shared VAE excluded",
                     decimals=1)
        _report_bars(axes[1, 0], report, summary["checkpoint_MiB"] / 1024,
                     "Serialized checkpoint (GiB) ↓", "DiT checkpoint size", decimals=2)
        memory = _memory_panel(summary)
        if memory:
            key, label, title = memory
            _report_bars(axes[1, 1], report, summary[key], label, title, decimals=2)
        else:
            _report_bars(axes[1, 1], report, summary["block_evaluations_per_forward"],
                         "Block calls / forward", "Executed blocks", decimals=0)
        return _save_report_figure(
            report, fig, "efficiency_metrics",
            "Latency bars: mean ± sample SD. Timing excludes model loading, warmup, "
            "serialization and metrics. Unique weights and execution count are different "
            "quantities; MPS snapshots are not peak memory.")


def plot_report_per_class(report):
    names = [name for name in report["summary"]["variant"] if name != "original"]
    if not names:
        return None
    labels = sorted({label for label, _seed in report["baseline"]["pairs"]})
    pivot = report["per_class"].pivot(index="variant", columns="label")
    table = pivot["lpips_mean"].reindex(index=names, columns=labels)
    counts = pivot["samples"].reindex(index=names, columns=labels)
    method_labels = report["summary"].set_index("variant")["method"]
    with matplotlib.rc_context(REPORT_STYLE):
        fig, axes = _figure(1, 1, figsize=(max(9, len(labels) * 0.95), 4.8))
        ax = axes[0, 0]
        vmax = max(float(np.nanmax(table.to_numpy())), 0.01)
        heatmap = ax.imshow(table, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.grid(False)
        ax.set_xticks(range(len(labels)), [str(label) for label in labels])
        ax.set_yticks(range(len(names)), [method_labels[name] for name in names], fontsize=9)
        ax.set_xlabel("ImageNet class ID")
        fig.suptitle("Output preservation by class", fontsize=13)
        for row in range(len(names)):
            for column in range(len(labels)):
                value = table.iloc[row, column]
                if pd.notna(value):
                    ax.text(column, row, f"{value:.3f}\nn={int(counts.iloc[row, column])}",
                            ha="center", va="center", fontsize=8,
                            color="white" if value > 0.65 * vmax else "#1E293B")
        fig.colorbar(heatmap, ax=ax, label="Mean paired LPIPS ↓", shrink=0.85)
        return _save_report_figure(
            report, fig, "per_class_lpips",
            "Each cell averages matched original/variant pairs within one class. n is the "
            "number of pairs; per-class LPIPS is output preservation, not per-class FID or "
            "absolute image quality.")


def _first_of_each_class(pairs, count):
    """Indices of the first sample of up to ``count`` distinct classes."""
    indices, seen = [], set()
    for index, (label, _seed) in enumerate(pairs):
        if label not in seen:
            indices.append(index)
            seen.add(label)
        if len(indices) == count:
            break
    return indices


def plot_report_samples(report, count=6):
    from PIL import Image
    if type(count) is not int or count < 1:
        raise ValueError("Sample grid count must be a positive integer")
    pairs = report["baseline"]["pairs"]
    indices = _first_of_each_class(pairs, count)  # Different classes, not many seeds of one.
    names = list(report["summary"]["variant"])
    method_labels = report["summary"].set_index("variant")["method"]
    with matplotlib.rc_context(REPORT_STYLE):
        fig, axes = _figure(len(names), len(indices),
                            figsize=(2.15 * len(indices) + 1, 2.1 * len(names)))
        fig.suptitle("Matched generated samples", fontsize=13)
        for row, name in enumerate(names):
            for column, index in enumerate(indices):
                ax = axes[row, column]
                with Image.open(report["root"] / name / f"image_{index:04d}.png") as image:
                    ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    label, seed = pairs[index]
                    ax.set_title(f"Class {label}\nSeed {seed}", fontsize=9)
                if column == 0:
                    ax.set_ylabel(method_labels[name], fontsize=9)
        return _save_report_figure(
            report, fig, "matched_samples",
            "Columns use identical class/seed pairs across methods. Original DiT outputs are a "
            "generated baseline, not real ground-truth images. Examples are selected by first "
            "occurrence of each class, without quality ranking.")


def _schedule_matrices(report, names, depth):
    """Source block, residual scale and XSA flag for every variant x depth slot."""
    sources = np.full((len(names), depth), np.nan)
    scales = np.full_like(sources, np.nan)
    xsa = np.zeros_like(sources, dtype=bool)
    for row, name in enumerate(names):
        schedule = report["manifests"][name].get("schedule", {})
        for position, entry in enumerate(schedule.get("execution_schedule", [])):
            slot = entry.get("slot", entry["source"] if schedule.get("mode") == "removal"
                             else position)
            sources[row, slot] = entry["source"]
            scales[row, slot] = entry["scale"]
            xsa[row, slot] = entry.get("xsa", False)
    return sources, scales, xsa


def plot_report_schedules(report):
    names = list(report["summary"]["variant"])
    schedules = [m.get("schedule", {}) for m in report["manifests"].values()]
    depth = max(schedule.get("depth", 0) for schedule in schedules)
    if not depth:
        return None
    # Looping methods execute more block calls than the model has blocks.
    depth = max(depth, *(len(schedule.get("execution_schedule", [])) for schedule in schedules))
    sources, scales, xsa = _schedule_matrices(report, names, depth)
    method_labels = report["summary"].set_index("variant")["method"]
    panels = [
        (sources, "Original source block used at each depth slot (* = XSA)", max(depth - 1, 1)),
        (scales, "Residual scale at each executed depth slot", 1.0),
    ]
    with matplotlib.rc_context(REPORT_STYLE):
        fig, axes = _figure(2, 1, figsize=(16, 7.5))
        fig.suptitle("Saved block schedules and residual damping", fontsize=13)
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#E2E8F0")
        for panel, (ax, (matrix, title, upper)) in enumerate(zip(axes[:, 0], panels)):
            heatmap = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                                vmin=0, vmax=upper)
            ax.grid(False)
            ax.set_xticks(range(depth), [str(slot) for slot in range(depth)], fontsize=7)
            ax.set_yticks(range(len(names)), [method_labels[name] for name in names], fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Original depth slot (zero-based)")
            for row in range(len(names)):
                for slot in range(depth):
                    value = matrix[row, slot]
                    if np.isnan(value):
                        label = "—"
                    elif panel == 0:
                        label = f"{int(value)}{'*' if xsa[row, slot] else ''}"
                    else:
                        label = f"{value:.2g}"
                    ax.text(slot, row, label, ha="center", va="center", fontsize=6.5,
                            color="#1E293B" if np.isnan(value) or value > 0.6 * upper else "white")
            fig.colorbar(heatmap, ax=ax, fraction=0.018, pad=0.015)
        return _save_report_figure(
            report, fig, "block_schedules",
            "Schedules and damping labels are read from saved manifests. Gray cells were not "
            "executed. Historical fixed-alpha runs retain their recorded labels; this plot "
            "does not simulate the new 1/k policy.")


REPORT_PLOTS = {
    "quality": plot_report_quality,
    "efficiency": plot_report_efficiency,
    "per_class": plot_report_per_class,
    "samples": plot_report_samples,
    "schedules": plot_report_schedules,
}
