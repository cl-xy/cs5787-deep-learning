"""Optional extra metrics on a finished, LPIPS-evaluated split: Clean-FID and TPIPS.

Run after ``python -m team_pipeline.cli evaluate`` has written ``results.csv`` and
``paired_metrics.csv``. Results go to separate ``*_extended`` files, so the base
LPIPS outputs are never modified. FID/TPIPS are lower-is-better; CLIP cosines
are higher-is-better.

- FID: Clean-FID Inception features of the manifest-listed ``image_NNNN.png``
  files against a directory of REAL images (``pip install clean-fid``).
  ``build_imagenet_reference`` writes such a directory from ImageNet for the
  generated classes (``pip install datasets``; gated, needs a Hugging Face login).
- TPIPS: official embedding-model distance per factor (e.g. overall, shape,
  texture) against the matched ``original`` image (``pip install tpips``).

FID and TPIPS are opt-in. CLIP reports image–text and paired image–image raw
cosines; the notebook and CLI enable it by default. Direct callers pass
``enable_clip=True``. With all extras disabled, LPIPS results are exported unchanged.
For FID against the ADM ImageNet statistics on the ``fid`` split, use the
``fid`` subcommand (``team_pipeline/metrics.py``) instead.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import torch

from .evaluation import SAMPLE_KEYS, check_manifests_consistent, evaluate_split
from .imagenette import prepare_imagenette_reference
from .runtime import cleanup, write_json
from .metrics import load_clip, imagenet_class_names, PROMPT

IMAGENET_DATASET = "ILSVRC/imagenet-1k"
REFERENCE_RECORD = "reference.json"
EXTRA_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "clean-fid", "tpips",
                  "transformers", "pillow")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FID_BATCH_SIZE = 16
FID_FULL_RANK_IMAGES = 2049  # 2048-D Inception covariance needs at least 2049 images.
TPIPS_DEFAULT_MODEL = "sywang/TPIPS-Embed-Qwen3VL-8B"


def _append_clip_metrics(root, names, summary, paired, baseline, *, device="cpu", batch_size=16):
    """Stream saved images; retain only small embeddings for paired comparisons."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("CLIP batch_size must be a positive integer")
    encode_image, encode_text, info = load_clip(device)
    pairs = baseline["pairs"]
    size = baseline["config"]["image_size"]
    labels = sorted({label for label, seed in pairs})
    class_names = imagenet_class_names()
    prompts = [PROMPT.format(class_names[label]) for label in labels]

    def normalized(features, count):
        features = features.detach().float().cpu()
        if (features.ndim != 2 or len(features) != count
                or not torch.isfinite(features).all()
                or (features.norm(dim=-1) <= 0).any()):
            raise ValueError("Invalid CLIP embeddings")
        return torch.nn.functional.normalize(features, dim=-1)

    def image_features(name):
        chunks = []
        for start in range(0, len(pairs), batch_size):
            images = torch.cat([_extra_read_image(root / name / f"image_{i:04d}.pt", size)
                                for i in range(start, min(start + batch_size, len(pairs)))])
            chunks.append(normalized(encode_image(images), len(images)))
        return torch.cat(chunks)

    try:
        text = normalized(encode_text(prompts), len(prompts))
        text = text[[labels.index(label) for label, seed in pairs]]
        reference = image_features("original")
        for name in names:
            features = reference if name == "original" else image_features(name)
            for column, target in [("clip_image_text_cosine", text),
                                   ("clip_image_image_cosine", reference)]:
                if features.shape != target.shape:
                    raise ValueError("Mismatched CLIP embedding shapes")
                scores = (features * target).sum(-1).clamp(-1, 1).numpy()
                paired.loc[paired["variant"] == name, column] = scores
                summary.loc[summary["variant"] == name, column + "_mean"] = float(scores.mean())
                summary.loc[summary["variant"] == name, column + "_std"] = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        return dict(**info, higher_is_better=True, range=[-1, 1], prompt_template=PROMPT,
                    prompts={str(label): prompt for label, prompt in zip(labels, prompts)},
                    image_reference="original (matched label and seed)",
                    preprocessing="saved FP32 RGB [-1,1]; bicubic 224x224; CLIP mean/std",
                    device=device, batch_size=batch_size,
                    interpretation="text alignment and semantic preservation; raw cosine, not 100x CLIPScore")
    finally:
        del encode_image, encode_text
        cleanup()

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


def package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def required_extra_packages(fid_enabled, tpips_factors):
    """Optional packages still missing for the requested metrics (pip names)."""
    missing = []
    if fid_enabled and package_version("clean-fid") == "not-installed":
        missing.append("clean-fid")
    if tpips_factors and package_version("tpips") == "not-installed":
        missing.append("tpips")
    return missing


def _extra_read_image(path, size):
    image = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(image, torch.Tensor) or image.shape != (1, 3, size, size)
            or not image.is_floating_point() or not torch.isfinite(image).all()
            or image.min() < -1.00001 or image.max() > 1.00001):
        raise ValueError(f"Invalid saved RGB image tensor: {path}")
    return image.detach().float().clamp(-1, 1)


def _check_variant_names(names):
    if not names or len(set(names)) != len(names):
        raise ValueError("Use nonempty, unique variant names")
    if any(not isinstance(n, str) or Path(n).name != n or n in {".", "..", ""} for n in names):
        raise ValueError("Each variant name must be a single folder name")


def _extra_read_artifacts(root, names):
    """Validate paired inputs using the existing manifests, samples and LPIPS CSVs."""
    names = list(names)
    _check_variant_names(names)
    summary = pd.read_csv(root / "results.csv")
    paired = pd.read_csv(root / "paired_metrics.csv")
    if summary["variant"].duplicated().any():
        raise ValueError("Duplicate variants in results.csv")
    baseline = json.loads((root / "original/manifest.json").read_text())
    if not baseline["pairs"]:
        raise ValueError("Empty baseline sample list")
    expected = pd.DataFrame([(i, *pair) for i, pair in enumerate(baseline["pairs"])],
                            columns=SAMPLE_KEYS)
    samples_by_name = {}
    for name in dict.fromkeys(["original", *names]):
        manifest = json.loads((root / name / "manifest.json").read_text())
        check_manifests_consistent(baseline, manifest, name)
        samples = pd.read_csv(root / name / "samples.csv")
        if not samples[SAMPLE_KEYS].equals(expected):
            raise ValueError(f"Incomplete or mismatched sample order: {name}")
        samples_by_name[name] = samples
        if name in names:
            rows = paired.loc[paired["variant"] == name]
            if not rows[SAMPLE_KEYS].reset_index(drop=True).equals(expected):
                raise ValueError(f"Incomplete paired_metrics.csv rows: {name}")
            result = summary.loc[summary["variant"] == name]
            if len(result) != 1 or int(result.iloc[0]["images"]) != len(expected):
                raise ValueError(f"Missing or inconsistent results.csv entry: {name}")
    summary = summary.set_index("variant").loc[names].reset_index()
    paired = pd.concat([paired.loc[paired["variant"] == name] for name in names],
                       ignore_index=True)
    return summary, paired, baseline, samples_by_name


def _tpips_columns(factors):
    if any(not isinstance(f, str) or not f.strip() for f in factors) or len(set(factors)) != len(factors):
        raise ValueError("Use unique, nonempty TPIPS factor strings")
    columns = ["tpips_" + re.sub(r"[^a-z0-9]+", "_", f.lower()).strip("_") for f in factors]
    if len(set(columns)) != len(columns) or "tpips_" in columns:
        raise ValueError("TPIPS factor names produce empty or colliding CSV column names")
    return columns


def _fid_reference_files(reference, root):
    if not reference.is_dir():
        raise FileNotFoundError(f"Real-image FID reference folder missing: {reference}")
    if reference == root or root in reference.parents or reference in root.parents:
        raise ValueError("FID reference must be real images outside the generated split directory")
    return sorted(p for p in reference.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _frechet_from_feats(feats1, feats2):
    """Frechet distance; replaces clean-fid's, which passes ``disp=`` to ``scipy.linalg.sqrtm``
    (removed in SciPy 1.16+)."""
    from scipy import linalg
    mu1, mu2 = feats1.mean(axis=0), feats2.mean(axis=0)
    s1, s2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    covmean = np.real(covmean)
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _compute_fid(root, names, indices, reference, reference_files, device, batch_size):
    """Clean-FID of each variant's PNGs against the real reference; returns {name: score}."""
    from cleanfid import fid
    from cleanfid.features import build_feature_extractor
    generated = {name: [root / name / f"image_{int(i):04d}.png" for i in indices]
                 for name in names}
    for path in (p for files in generated.values() for p in files):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = build_feature_extractor("clean", device=torch.device(device), use_dataparallel=False)
    try:
        def features(files, extractor=model):
            values = fid.get_files_features(
                [str(p) for p in files], model=extractor, num_workers=0, batch_size=batch_size,
                device=torch.device(device), mode="clean")
            if values.ndim != 2 or values.shape[0] != len(files) or not np.isfinite(values).all():
                raise ValueError("Invalid Clean-FID features")
            return values

        reference_features = features(reference_files)
        scores = {}
        for name in names:
            score = _frechet_from_feats(features(generated[name]), reference_features)
            if not np.isfinite(score):
                raise ValueError(f"Nonfinite FID: {name}")
            scores[name] = score
            print(f"FID {name}: {score:.6f}", flush=True)
        return scores
    finally:
        del model
        cleanup()


def _fid_stage(root, names, indices, summary, reference_dir, description, device, batch_size):
    """Add an ``fid`` column to ``summary``; returns the metadata record."""
    reference = Path(reference_dir).expanduser().resolve()
    record = reference / REFERENCE_RECORD
    if not description and record.is_file():
        description = json.loads(record.read_text()).get("description")
    if not description:
        raise ValueError("Describe the FID reference dataset, class mixture, split and "
                         "preprocessing")
    reference_files = _fid_reference_files(reference, root)
    if len(indices) < 2 or len(reference_files) < 2:
        raise ValueError("FID needs at least two generated and two reference images")
    small_sample = len(indices) < FID_FULL_RANK_IMAGES or len(reference_files) < FID_FULL_RANK_IMAGES
    if small_sample:
        warnings.warn("Small-sample FID: 2048-D covariance is rank-deficient below 2049 images. "
                      "Treat this as a noisy pilot diagnostic, not a benchmark-quality FID.",
                      stacklevel=3)
    scores = _compute_fid(root, names, indices, reference, reference_files, device, batch_size)
    summary["fid"] = summary["variant"].map(scores)
    inventory = [dict(path=str(p.relative_to(reference)), bytes=p.stat().st_size)
                 for p in reference_files]
    write_json(root / "fid_reference_inventory.json", inventory)
    inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    return dict(
        implementation="clean-fid", mode="clean", features="inception_v3_2048",
        reference_dir=str(reference), reference_description=description,
        reference_images=len(reference_files), generated_images=len(indices),
        reference_inventory_sha256=inventory_hash,
        inventory_hash_scope="relative filenames and file sizes, not image content",
        input="only manifest-listed image_NNNN.png; grids excluded", device=device,
        small_sample_pilot=small_sample,
    )


def _tpips_stage(root, names, indices, size, summary, paired, factors, columns, device,
                 model_path):
    """Add TPIPS columns to ``summary`` and ``paired``; returns the metadata record."""
    import tpips
    metric = tpips.load_model("embedding", model_path=model_path, device=device)
    metric.eval().requires_grad_(False)
    try:
        for name in names:
            values = {column: [] for column in columns}
            for i in indices:
                ref = _extra_read_image(root / "original" / f"image_{int(i):04d}.pt", size)
                image = _extra_read_image(root / name / f"image_{int(i):04d}.pt", size)
                for factor, column in zip(factors, columns):
                    score = float(metric.distance(ref, image, factor=factor,
                                                  value_range=(-1.0, 1.0)).item())
                    if not np.isfinite(score):
                        raise ValueError(f"Nonfinite TPIPS: {name}, sample={i}, factor={factor}")
                    values[column].append(score)
            for column, scores in values.items():
                std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                paired.loc[paired["variant"] == name, column] = scores
                summary.loc[summary["variant"] == name, column + "_mean"] = float(np.mean(scores))
                summary.loc[summary["variant"] == name, column + "_std"] = std
            print(f"TPIPS finished: {name}", flush=True)
    finally:
        del metric
        cleanup()
    return dict(
        model_type="embedding", model_path=str(model_path) if model_path else TPIPS_DEFAULT_MODEL,
        factors=dict(zip(columns, factors)), paired_reference="original",
        input="saved FP32 RGB tensors; TPIPS value_range=(-1,1); official model preprocessing",
        metric="official model.distance (1 - cosine similarity)", device=device,
        interpretation="aspect-conditioned deviation from original, not absolute generation quality",
    )


@torch.inference_mode()
def evaluate_fid_tpips(split_root, names, *, fid_reference_dir=None,
                       fid_reference_description=None,
                       tpips_factors=("overall", "shape", "texture"),
                       fid_device="cpu", tpips_device="cuda:0",
                       fid_batch_size=FID_BATCH_SIZE, tpips_model_path=None,
                       enable_clip=False, clip_device="cpu", clip_batch_size=16):
    """Append metrics to extended CSVs: FID/TPIPS lower, CLIP cosine higher.

    FID uses pooled Clean-FID Inception features against a REAL-image directory.
    Only manifest-listed image_NNNN.png files are evaluated: grids are excluded.
    TPIPS uses official embedding-model distance against matched original images,
    from saved FP32 tensors with the explicit numeric range [-1, 1]. Prompts are
    separate metrics. With all extras disabled, export the existing LPIPS results.
    Install optional metric dependencies only after the base LPIPS evaluation.
    """
    root = Path(split_root).expanduser().resolve()
    names = list(names)
    factors = list(tpips_factors)
    columns = _tpips_columns(factors)
    summary, paired, baseline, samples_by_name = _extra_read_artifacts(root, names)
    if fid_reference_dir is None and not factors and not enable_clip:
        print("FID and TPIPS disabled; exporting LPIPS results only.", flush=True)
    indices = samples_by_name["original"]["index"].tolist()
    metadata = dict(
        generation_code_sha256=baseline["shared_code_sha256"],
        generation_runtime=baseline["runtime"], variants=names,
        images_per_variant=len(indices), metric_directions={"fid": "lower", "tpips": "lower",
                           "clip_image_text_cosine": "higher", "clip_image_image_cosine": "higher"},
        packages={p: package_version(p) for p in EXTRA_PACKAGES}, fid=None, tpips=None, clip=None,
    )
    if fid_reference_dir is not None:
        metadata["fid"] = _fid_stage(root, names, indices, summary, fid_reference_dir,
                                     fid_reference_description, fid_device, fid_batch_size)
    if factors:
        metadata["tpips"] = _tpips_stage(root, names, indices, baseline["config"]["image_size"],
                                         summary, paired, factors, columns, tpips_device,
                                         tpips_model_path)
    if enable_clip:
        metadata["clip"] = _append_clip_metrics(
            root, names, summary, paired, baseline, device=clip_device, batch_size=clip_batch_size)

    # Preserve the original LPIPS evaluation files and generation measurements.
    summary.to_csv(root / "results_extended.csv", index=False)
    paired.to_csv(root / "paired_metrics_extended.csv", index=False)
    write_json(root / "evaluation_extended.json", metadata)
    print("Saved:", root / "results_extended.csv")
    return summary


def _lpips_is_current(split_root, names):
    """True when LPIPS outputs already exist and cover every requested variant."""
    results = split_root / "results.csv"
    required = [results, split_root / "paired_metrics.csv", split_root / "comparison_grid.png"]
    return (all(path.exists() for path in required)
            and set(names) <= set(pd.read_csv(results)["variant"]))


def evaluate_saved_split(run_root, protocol, split, names, *, imagenette_source_root=None,
                         imagenette_prepared_root=None, fid_real_images_per_class=500,
                         fid_reference_dir=None, fid_reference_description=None,
                         tpips_factors=(), fid_device="cpu", tpips_device="cpu",
                         installer=None, metric=None, enable_fid=True, enable_clip=False,
                         clip_device="cpu", clip_batch_size=16):
    """LPIPS (if missing) plus optional CLIP/FID/TPIPS for ``run_root/split``.

    ``enable_fid=False`` skips all FID work even with configured reference paths.
    ``enable_clip=True`` adds both CLIP cosine metrics independently of FID.
    FID is enabled by passing ``imagenette_source_root`` and ``imagenette_prepared_root``
    (builds a class-matched Imagenette reference) or an existing ``fid_reference_dir``.
    ``installer(packages)`` is called with any missing optional pip packages; without
    it a missing package raises instead of being installed silently. Reads and writes
    only ``run_root/split``; ``metric`` replaces LPIPS (for tests). Returns a dict with
    ``results`` and artifact ``paths``.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    names = list(names)
    run_root = Path(run_root)
    split_root = run_root / split
    if not _lpips_is_current(split_root, names):
        evaluate_split(run_root, protocol, split, names, metric)
    if not enable_fid:
        imagenette_source_root = None
        fid_reference_dir = None
        fid_reference_description = None
    if imagenette_source_root is not None:
        if imagenette_prepared_root is None:
            raise ValueError("Give imagenette_prepared_root with imagenette_source_root")
        baseline = json.loads((split_root / "original/manifest.json").read_text())
        fid_reference_dir, fid_reference_description = prepare_imagenette_reference(
            imagenette_source_root, imagenette_prepared_root, baseline["pairs"],
            images_per_class=fid_real_images_per_class,
            image_size=baseline["config"]["image_size"], seed=0)
    # Install optional packages only after the base evaluation checked the run.
    missing = required_extra_packages(fid_reference_dir is not None, tpips_factors)
    if enable_clip and package_version("transformers") == "not-installed":
        missing.append("transformers>=4.50,<5")
    if missing:
        if installer is None:
            raise RuntimeError(f"Install optional metric packages first: pip install {' '.join(missing)}")
        installer(missing)
    results = evaluate_fid_tpips(
        split_root, names, fid_reference_dir=fid_reference_dir,
        fid_reference_description=fid_reference_description, tpips_factors=tpips_factors,
        fid_device=fid_device, tpips_device=tpips_device,
        enable_clip=enable_clip, clip_device=clip_device, clip_batch_size=clip_batch_size)
    paths = dict(results=split_root / "results_extended.csv",
                 paired=split_root / "paired_metrics_extended.csv",
                 settings=split_root / "evaluation_extended.json",
                 grid=split_root / "comparison_grid.png")
    return dict(results=results, paths=paths)
