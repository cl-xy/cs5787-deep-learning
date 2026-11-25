"""Paired LPIPS evaluation, efficiency summaries and comparison grids for finished splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import preprocess_image, validate_pairs
from .provenance import shared_code_hash
from .runtime import write_json

MIB = 2**20
GIB = 2**30
GFLOP = 1e9
SAMPLE_KEYS = ["index", "label", "seed"]
# (manifest key, error message) in the order they are checked.
CONSISTENCY_CHECKS = (
    ("shared_code_sha256", "Shared pipeline code differs"),
    ("runtime", "Hardware or software environment differs"),
    ("config", "Mismatched configuration or sample pairs"),
    ("pairs", "Mismatched configuration or sample pairs"),
)


def check_manifests_consistent(reference, manifest, name, checks=CONSISTENCY_CHECKS):
    """Raise ValueError unless ``manifest`` was produced under the reference's conditions."""
    for key, message in checks:
        if key not in manifest or manifest[key] != reference.get(key):
            raise ValueError(f"{message}: {name}")


def reduction_pct(value, reference):
    """Percent reduction of ``value`` relative to ``reference``."""
    return 100 * (1 - value / reference)


def _load_variant(root, name, baseline, ref_manifest):
    """Read one variant's samples/manifest and verify they match the baseline."""
    folder = Path(root) / name
    samples = pd.read_csv(folder / "samples.csv")
    manifest = json.loads((folder / "manifest.json").read_text())
    check_manifests_consistent(ref_manifest, manifest, name)
    if not samples[SAMPLE_KEYS].equals(baseline[SAMPLE_KEYS]):
        raise ValueError(f"Mismatched sample order: {name}")
    pairs = validate_pairs(manifest["pairs"])
    expected = pd.DataFrame([(i, label, seed) for i, (label, seed) in enumerate(pairs)],
                            columns=SAMPLE_KEYS)
    if not samples[SAMPLE_KEYS].equals(expected):
        raise ValueError(f"Incomplete or invalid sample manifest: {name}")
    return samples, manifest


def _paired_lpips(root, name, samples, manifest, metric):
    """LPIPS between each saved image and the original's image for the same pair."""
    size = manifest["config"]["image_size"]
    scores = []
    for i in samples["index"]:
        ref = preprocess_image(torch.load(Path(root) / "original" / f"image_{i:04d}.pt",
                                          weights_only=True), size)
        image = preprocess_image(torch.load(Path(root) / name / f"image_{i:04d}.pt",
                                            weights_only=True), size)
        score = float(metric(ref, image).item())
        if not np.isfinite(score):
            raise ValueError("Nonfinite paired metric")
        scores.append(score)
    return scores


def _optional_max(samples, column, divisor):
    return samples[column].max() / divisor if column in samples else None


def _summary_row(name, samples, scores, stats, ref_stats, ref_latency):
    latency = float(samples["latency_s"].mean())
    parameters = int(stats["unique_parameters"])
    checkpoint_bytes = int(stats["checkpoint_bytes"])
    flops, ref_flops = stats.get("dit_sampling_flops"), ref_stats.get("dit_sampling_flops")
    evaluations = stats["block_evaluations_per_forward"]
    return dict(
        variant=name, images=len(samples), lpips_mean=float(np.mean(scores)),
        lpips_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        unique_parameters=parameters,
        parameter_reduction_pct=reduction_pct(parameters, ref_stats["unique_parameters"]),
        unique_blocks=stats["unique_blocks"],
        block_evaluations_per_forward=evaluations,
        effective_block_evaluations_per_forward=stats.get(
            "effective_block_evaluations_per_forward", evaluations),
        checkpoint_MiB=checkpoint_bytes / MIB,
        checkpoint_reduction_pct=reduction_pct(checkpoint_bytes, ref_stats["checkpoint_bytes"]),
        dit_sampling_GFLOPs=flops / GFLOP if flops is not None else None,
        flops_reduction_pct=(reduction_pct(flops, ref_flops) if flops is not None and ref_flops
                             else None),
        device=samples["device"].iloc[0] if "device" in samples else "cuda",
        peak_GPU_allocated_GiB=_optional_max(samples, "peak_allocated_bytes", GIB),
        peak_GPU_reserved_GiB=_optional_max(samples, "peak_reserved_bytes", GIB),
        max_post_sample_MPS_allocated_GiB=_optional_max(
            samples, "mps_allocated_bytes_after_sample", GIB),
        max_post_sample_MPS_driver_GiB=_optional_max(
            samples, "mps_driver_bytes_after_sample", GIB),
        latency_mean_s=latency, latency_std_s=samples["latency_s"].std(ddof=0),
        latency_reduction_pct=reduction_pct(latency, ref_latency),
        latency_speedup_x=ref_latency / latency,
        throughput_images_per_s=1 / latency,
    )


@torch.inference_mode()
def compare_results(root, names, metric=None):
    """LPIPS runs on CPU after all device models are released; never contaminates timing."""
    root = Path(root)
    if metric is None:
        import lpips
        metric = lpips.LPIPS(net="alex", version="0.1").cpu().eval()
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide nonempty, unique variant names")
    baseline = pd.read_csv(root / "original" / "samples.csv")
    ref_manifest = json.loads((root / "original" / "manifest.json").read_text())
    ref_latency = float(baseline["latency_s"].mean())
    all_rows, summary = [], []
    for name in names:
        samples, manifest = _load_variant(root, name, baseline, ref_manifest)
        scores = _paired_lpips(root, name, samples, manifest, metric)
        samples["paired_lpips"] = scores
        samples["variant"] = name
        all_rows.append(samples)
        summary.append(_summary_row(name, samples, scores, manifest["stats"],
                                    ref_manifest["stats"], ref_latency))
    result = pd.DataFrame(summary)
    result.to_csv(root / "results.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(root / "paired_metrics.csv", index=False)
    is_lpips = metric.__class__.__module__.startswith("lpips")
    write_json(root / "evaluation.json", dict(
        metric="lpips_alex_v0.1" if is_lpips else "injected_metric",
        preprocessing="saved CPU FP32 RGB tensors in [-1,1]; no resizing",
        variants=names, images=len(baseline), paired_reference="original",
        shared_code_sha256=shared_code_hash(),
        interpretation="distance from original output, not absolute generation quality"))
    return result


def comparison_grid(root, names, pairs, count=8):
    """Save a variants x images grid of the saved PNG previews to ``comparison_grid.png``."""
    from matplotlib.figure import Figure
    from PIL import Image
    root = Path(root)
    count = min(count, len(pairs))
    fig = Figure(figsize=(2.4 * count, 2.7 * len(names)), layout="constrained")
    axes = fig.subplots(len(names), count, squeeze=False)
    for row, name in enumerate(names):
        for column in range(count):
            ax = axes[row, column]
            with Image.open(root / name / f"image_{column:04d}.png") as image:
                ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"class {pairs[column][0]}\nseed {pairs[column][1]}", fontsize=9)
            if column == 0:
                ax.set_ylabel(name)
    fig.savefig(root / "comparison_grid.png", dpi=160)


def evaluate_split(run_root, protocol, split, names, metric=None):
    """LPIPS comparison plus grid for ``run_root/split``; returns the summary frame."""
    split_root = Path(run_root) / split
    result = compare_results(split_root, names, metric)
    comparison_grid(split_root, names, protocol["splits"][split])
    return result
