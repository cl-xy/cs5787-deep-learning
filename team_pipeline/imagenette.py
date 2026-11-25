"""Real-image FID reference built from the Imagenette v2 training split.

DiT is not trained here: the Imagenette TRAIN images only supply real statistics.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .data import IMAGENETTE_CLASSES, validate_pairs
from .runtime import write_json

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGENETTE_ARCHIVE_MD5 = "fe2fc210e6bb7c5664d602c3cd71e612"
SOURCE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
PREPROCESSING = "RGB; aspect-preserving bicubic resize and center crop; PNG"


def _check_options(images_per_class, image_size, seed):
    if type(images_per_class) is not int or images_per_class < 2:
        raise ValueError("images_per_class must be an integer >= 2")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("Reference seed must be a nonnegative integer")


def _selected_labels(pairs):
    counts = Counter(label for label, _seed in validate_pairs(pairs))
    labels = sorted(counts)
    if set(labels) - {label for label, _synset, _name in IMAGENETTE_CLASSES}:
        raise ValueError("Run contains classes outside Imagenette; prepare a new run using "
                         "CLASS_LABELS")
    if len(set(counts.values())) != 1:
        raise ValueError("Use balanced generated class counts to match the real reference")
    return labels


def _reuse_cached(output, settings, expected):
    """True when ``output`` is a complete, consistent earlier export of ``settings``."""
    manifest_path = output / "reference_manifest.json"
    if not manifest_path.exists():
        return False
    saved = json.loads(manifest_path.read_text())
    actual = {p.name for p in output.glob("*.png") if p.is_file()}
    if (saved.get("settings") != settings or actual != expected
            or any((output / filename).stat().st_size == 0 for filename in expected)):
        raise ValueError(f"Cached Imagenette reference is incomplete or inconsistent: {output}")
    return True


def _select_sources(source_root, labels, images_per_class, seed):
    class_map = {label: (synset, name) for label, synset, name in IMAGENETTE_CLASSES}
    selections = {}
    for label in labels:
        synset, name = class_map[label]
        folder = Path(source_root) / "imagenette2" / "train" / synset
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS)
        if len(files) < images_per_class:
            raise ValueError(f"{name} ({label}) has {len(files)} real images; "
                             f"requested {images_per_class}")
        selections[label] = sorted(random.Random(seed + label).sample(files, images_per_class))
    return selections


def prepare_imagenette_reference(source_root, prepared_root, pairs, *,
                                 images_per_class=500, image_size=256, seed=0):
    """Export a deterministic, class-balanced real-image reference and reuse it.

    Uses the dataset's TRAIN split as real FID reference data. Generated labels
    come from the saved run manifest, never assumed local ids. Source files and
    archives live outside the prepared image folder so Clean-FID cannot accidentally
    include raw images or grids in the reference. Returns ``(folder, description)``.
    """
    from PIL import Image, ImageOps

    labels = _selected_labels(pairs)
    _check_options(images_per_class, image_size, seed)
    settings = dict(
        dataset="Imagenette v2", source_split="train", source_size="full",
        source_url=IMAGENETTE_URL, archive_md5=IMAGENETTE_ARCHIVE_MD5, labels=labels,
        images_per_class=images_per_class, image_size=image_size, seed=seed,
        preprocessing=PREPROCESSING,
    )
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output = Path(prepared_root) / f"imagenette2_train_{images_per_class}perclass_{digest}"
    expected = {f"class_{label:04d}_{i:04d}.png"
                for label in labels for i in range(images_per_class)}
    description = (
        f"Imagenette v2 full-resolution train split; ImageNet labels={labels}; "
        f"{images_per_class} real images per class, selected with seed={seed}; "
        f"RGB; aspect-preserving bicubic resize and center crop to "
        f"{image_size}x{image_size}; PNG."
    )
    if _reuse_cached(output, settings, expected):
        print("Reusing real FID reference:", output, flush=True)
        return output, description

    from torchvision.datasets import Imagenette
    print("Preparing Imagenette v2 full-resolution reference (one-time download).", flush=True)
    Imagenette(root=str(source_root), split="train", size="full", download=True)
    selections = _select_sources(source_root, labels, images_per_class, seed)
    output.mkdir(parents=True, exist_ok=True)
    if {p.name for p in output.glob("*.png")} - expected:
        raise ValueError(f"Unexpected images in reference folder: {output}")
    inventory = []
    for label in labels:
        for i, source in enumerate(selections[label]):
            filename = f"class_{label:04d}_{i:04d}.png"
            with Image.open(source) as image:
                image = ImageOps.fit(image.convert("RGB"), (image_size, image_size),
                                     method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
                image.save(output / filename)
            inventory.append(dict(file=filename, label=label,
                                  source=str(source.relative_to(source_root))))
        print(f"Real reference class {label}: {images_per_class} images", flush=True)
    write_json(output / "reference_manifest.json",
               dict(settings=settings, images=len(inventory), inventory=inventory))
    print("Saved real FID reference:", output, flush=True)
    return output, description
