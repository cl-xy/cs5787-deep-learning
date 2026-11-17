"""Optional extra metrics on a finished, LPIPS-evaluated split: Clean-FID and TPIPS.

Run after ``python -m team_pipeline.cli evaluate`` has written ``results.csv`` and
``paired_metrics.csv``. Results go to separate ``*_extended`` files, so the base
LPIPS outputs are never modified. Lower is better for both metrics.

- FID: Clean-FID Inception features of the manifest-listed ``image_NNNN.png``
  files against a directory of REAL images (``pip install clean-fid``).
  ``build_imagenet_reference`` writes such a directory from ImageNet for the
  generated classes (``pip install datasets``; gated, needs a Hugging Face login).
- TPIPS: official embedding-model distance per factor (e.g. overall, shape,
  texture) against the matched ``original`` image (``pip install tpips``).

Both are opt-in. With both disabled, the LPIPS results are exported unchanged.
For FID against the ADM ImageNet statistics on the ``fid`` split, use the
``fid`` subcommand (``team_pipeline/metrics.py``) instead.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import torch

IMAGENET_DATASET = "ILSVRC/imagenet-1k"
REFERENCE_RECORD = "reference.json"


def _reference_labels(split_root):
    """Classes and image size from the run's actual ``original`` generation manifest."""
    manifest = json.loads((Path(split_root) / "original" / "manifest.json").read_text())
    labels = sorted({int(label) for label, _ in manifest["pairs"]})
    return labels, int(manifest["config"]["image_size"])


def build_imagenet_reference(split_root, output_root="fid_references", images_per_class=50,
                             dataset=IMAGENET_DATASET, split="validation", streaming=False):
    """Write a real-image FID reference folder matching the generated classes.

    Takes the first ``images_per_class`` images of every generated class from the
    ImageNet ``split`` in dataset order, converts them to RGB, resizes them with
    bicubic interpolation (keeping the aspect ratio), center-crops them to the
    generated size and saves PNGs. The folder must lie outside the generated split.
    ImageNet on the Hugging Face Hub is gated: log in first (``huggingface_hub.login()``,
    ``huggingface-cli login`` or ``HF_TOKEN``) with an account that has access.

    A complete existing folder with the same settings is reused without downloading.
    Returns ``(folder, description)``; pass both to ``evaluate_fid_tpips``.
    """
    from PIL import Image as PILImage, ImageOps

    if isinstance(images_per_class, bool) or not isinstance(images_per_class, int) or images_per_class < 1:
        raise ValueError("images_per_class must be a positive integer")
    split_root = Path(split_root).expanduser().resolve()
    labels, image_size = _reference_labels(split_root)
    tag = "_".join(map(str, labels))
    if len(labels) > 16:  # Keep folder names short for large class sets (e.g. the fid split).
        tag = f"{len(labels)}cls_" + hashlib.sha256(tag.encode()).hexdigest()[:12]
    folder = Path(output_root).expanduser().resolve() / f"imagenet_{split}_{image_size}_{images_per_class}pc_{tag}"
    if folder == split_root or split_root in folder.parents or folder in split_root.parents:
        raise ValueError("FID reference must be outside the generated split directory")
    description = (
        f"{dataset} {split}; classes={labels}; {images_per_class} images per class "
        f"(first in dataset order); RGB; aspect-preserving bicubic resize and center crop "
        f"to {image_size}x{image_size}; saved as PNG."
    )
    expected = {f"class_{label:04d}_{i:04d}.png" for label in labels for i in range(images_per_class)}
    record = dict(dataset=dataset, split=split, labels=labels, image_size=image_size,
                  images_per_class=images_per_class, description=description)
    record_path = folder / REFERENCE_RECORD
    if record_path.exists() and json.loads(record_path.read_text()) == record:
        if {p.name for p in folder.glob("*.png")} == expected:
            print("Reusing complete reference folder:", folder)
            return folder, description
    stray = {p.name for p in folder.glob("*.png")} - expected if folder.exists() else set()
    if stray:
        raise ValueError(f"Reference folder contains unexpected images; move it aside: {folder}")

    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=streaming, token=True)
    folder.mkdir(parents=True, exist_ok=True)
    record_path.unlink(missing_ok=True)  # Only a finished build is marked complete.
    counts = {label: 0 for label in labels}
    for example in ds:
        label = int(example["label"])
        if label not in counts or counts[label] >= images_per_class:
            continue
        image = ImageOps.fit(example["image"].convert("RGB"), (image_size, image_size),
                             method=PILImage.Resampling.BICUBIC, centering=(0.5, 0.5))
        image.save(folder / f"class_{label:04d}_{counts[label]:04d}.png")
        counts[label] += 1
        if all(count == images_per_class for count in counts.values()):
            break
    if any(count != images_per_class for count in counts.values()):
        raise RuntimeError(f"Incomplete reference images: {counts}")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print("Reference folder:", folder)
    print("Images per class:", counts)
    return folder, description


def _extra_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _extra_read_image(path, size):
    image = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(image, torch.Tensor) or image.shape != (1, 3, size, size)
            or not image.is_floating_point() or not torch.isfinite(image).all()
            or image.min() < -1.00001 or image.max() > 1.00001):
        raise ValueError(f"Invalid saved RGB image tensor: {path}")
    return image.detach().float().clamp(-1, 1)


def _extra_read_artifacts(root, names):
    """Validate paired inputs using the existing manifests, samples and LPIPS CSVs."""
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Use nonempty, unique variant names")
    if any(not isinstance(n, str) or Path(n).name != n or n in {".", "..", ""} for n in names):
        raise ValueError("Each variant name must be a single folder name")
    summary = pd.read_csv(root / "results.csv")
    paired = pd.read_csv(root / "paired_metrics.csv")
    if summary["variant"].duplicated().any():
        raise ValueError("Duplicate variants in results.csv")
    baseline = json.loads((root / "original/manifest.json").read_text())
    pairs = baseline["pairs"]
    if not pairs:
        raise ValueError("Empty baseline sample list")
    expected = pd.DataFrame([(i, *pair) for i, pair in enumerate(pairs)],
                            columns=["index", "label", "seed"])
    samples_by_name = {}
    for name in dict.fromkeys(["original", *names]):
        manifest = json.loads((root / name / "manifest.json").read_text())
        for key in ["config", "pairs", "shared_code_sha256", "runtime"]:
            if key not in manifest or manifest[key] != baseline.get(key):
                raise ValueError(f"Mismatched generation {key}: {name}")
        samples = pd.read_csv(root / name / "samples.csv")
        if not samples[["index", "label", "seed"]].equals(expected):
            raise ValueError(f"Incomplete or mismatched sample order: {name}")
        samples_by_name[name] = samples
        if name in names:
            rows = paired.loc[paired["variant"] == name]
            if not rows[["index", "label", "seed"]].reset_index(drop=True).equals(expected):
                raise ValueError(f"Incomplete paired_metrics.csv rows: {name}")
            result = summary.loc[summary["variant"] == name]
            if len(result) != 1 or int(result.iloc[0]["images"]) != len(expected):
                raise ValueError(f"Missing or inconsistent results.csv entry: {name}")
    summary = summary.set_index("variant").loc[names].reset_index()
    paired = pd.concat([paired.loc[paired["variant"] == name] for name in names], ignore_index=True)
    return summary, paired, baseline, samples_by_name


@torch.inference_mode()
def evaluate_fid_tpips(split_root, names, *, fid_reference_dir=None,
                       fid_reference_description=None,
                       tpips_factors=("overall", "shape", "texture"),
                       fid_device="cpu", tpips_device="cuda:0",
                       fid_batch_size=16, tpips_model_path=None):
    """Append optional metrics to separate extended CSVs; lower is better for both.

    FID uses pooled Clean-FID Inception features against a REAL-image directory.
    Only manifest-listed image_NNNN.png files are evaluated: grids are excluded.
    TPIPS uses official embedding-model distance against matched original images,
    from saved FP32 tensors with the explicit numeric range [-1, 1]. Prompts are
    separate metrics. With both extras disabled, export the existing LPIPS results.
    Install optional metric dependencies only after the base LPIPS evaluation.
    """
    root = Path(split_root).expanduser().resolve()
    names = list(names)
    factors = list(tpips_factors)
    if any(not isinstance(f, str) or not f.strip() for f in factors) or len(set(factors)) != len(factors):
        raise ValueError("Use unique, nonempty TPIPS factor strings")
    columns = ["tpips_" + re.sub(r"[^a-z0-9]+", "_", factor.lower()).strip("_") for factor in factors]
    if len(set(columns)) != len(columns) or "tpips_" in columns:
        raise ValueError("TPIPS factor names produce empty or colliding CSV column names")
    summary, paired, baseline, samples_by_name = _extra_read_artifacts(root, names)
    if fid_reference_dir is None and not factors:
        print("FID and TPIPS disabled; exporting LPIPS results only.", flush=True)
    indices = samples_by_name["original"]["index"].tolist()
    size = baseline["config"]["image_size"]
    metadata = dict(
        generation_code_sha256=baseline["shared_code_sha256"],
        generation_runtime=baseline["runtime"], variants=names,
        images_per_variant=len(indices), lower_is_better=True,
        packages={p: _extra_version(p) for p in ["torch", "torchvision", "numpy",
                   "scipy", "clean-fid", "tpips", "transformers", "pillow"]},
        fid=None, tpips=None,
    )

    if fid_reference_dir is not None:
        from cleanfid import fid
        from cleanfid.features import build_feature_extractor
        reference = Path(fid_reference_dir).expanduser().resolve()
        if not reference.is_dir():
            raise FileNotFoundError(f"Real-image FID reference folder missing: {reference}")
        record = reference / REFERENCE_RECORD
        if not fid_reference_description and record.is_file():
            fid_reference_description = json.loads(record.read_text()).get("description")
        if not fid_reference_description:
            raise ValueError("Describe the FID reference dataset, class mixture, split and preprocessing")
        if reference == root or root in reference.parents or reference in root.parents:
            raise ValueError("FID reference must be real images outside the generated split directory")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        reference_files = sorted(p for p in reference.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        if len(indices) < 2 or len(reference_files) < 2:
            raise ValueError("FID needs at least two generated and two reference images")
        if len(indices) < 2049 or len(reference_files) < 2049:
            warnings.warn("Small-sample FID: 2048-D covariance is rank-deficient below 2049 images. "
                          "Treat this as a noisy pilot diagnostic, not a benchmark-quality FID.", stacklevel=2)
        generated_files = {
            name: [root / name / f"image_{int(i):04d}.png" for i in indices]
            for name in names
        }
        for files in generated_files.values():
            for path in files:
                if not path.is_file():
                    raise FileNotFoundError(path)
        model = build_feature_extractor("clean", device=torch.device(fid_device), use_dataparallel=False)
        try:
            def features(files):
                values = fid.get_files_features(
                    [str(p) for p in files], model=model, num_workers=0,
                    batch_size=fid_batch_size, device=torch.device(fid_device), mode="clean")
                if values.ndim != 2 or values.shape[0] != len(files) or not np.isfinite(values).all():
                    raise ValueError("Invalid Clean-FID features")
                return values

            reference_features = features(reference_files)
            fid_scores = {}
            for name in names:
                score = float(fid.fid_from_feats(features(generated_files[name]), reference_features))
                if not np.isfinite(score):
                    raise ValueError(f"Nonfinite FID: {name}")
                fid_scores[name] = score
                print(f"FID {name}: {score:.6f}", flush=True)
            summary["fid"] = summary["variant"].map(fid_scores)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        inventory = [dict(path=str(p.relative_to(reference)), bytes=p.stat().st_size) for p in reference_files]
        (root / "fid_reference_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        metadata["fid"] = dict(
            implementation="clean-fid", mode="clean", features="inception_v3_2048",
            reference_dir=str(reference), reference_description=fid_reference_description,
            reference_images=len(reference_files), generated_images=len(indices),
            reference_inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            inventory_hash_scope="relative filenames and file sizes, not image content",
            input="only manifest-listed image_NNNN.png; grids excluded", device=fid_device,
            small_sample_pilot=len(indices) < 2049 or len(reference_files) < 2049,
        )

    if factors:
        import tpips
        metric = tpips.load_model("embedding", model_path=tpips_model_path, device=tpips_device)
        metric.eval().requires_grad_(False)
        try:
            for name in names:
                values = {column: [] for column in columns}
                for i in indices:
                    ref = _extra_read_image(root / "original" / f"image_{int(i):04d}.pt", size)
                    image = _extra_read_image(root / name / f"image_{int(i):04d}.pt", size)
                    for factor, column in zip(factors, columns):
                        score = float(metric.distance(ref, image, factor=factor, value_range=(-1.0, 1.0)).item())
                        if not np.isfinite(score):
                            raise ValueError(f"Nonfinite TPIPS: {name}, sample={i}, factor={factor}")
                        values[column].append(score)
                for column, scores in values.items():
                    paired.loc[paired["variant"] == name, column] = scores
                    summary.loc[summary["variant"] == name, column + "_mean"] = float(np.mean(scores))
                    summary.loc[summary["variant"] == name, column + "_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                print(f"TPIPS finished: {name}", flush=True)
        finally:
            del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metadata["tpips"] = dict(
            model_type="embedding", model_path=str(tpips_model_path) if tpips_model_path else "sywang/TPIPS-Embed-Qwen3VL-8B",
            factors=dict(zip(columns, factors)), paired_reference="original",
            input="saved FP32 RGB tensors; TPIPS value_range=(-1,1); official model preprocessing",
            metric="official model.distance (1 - cosine similarity)", device=tpips_device,
            interpretation="aspect-conditioned deviation from original, not absolute generation quality",
        )

    # Preserve the original LPIPS evaluation files and generation measurements.
    summary.to_csv(root / "results_extended.csv", index=False)
    paired.to_csv(root / "paired_metrics_extended.csv", index=False)
    (root / "evaluation_extended.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print("Saved:", root / "results_extended.csv")
    return summary
