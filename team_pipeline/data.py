"""The team's input and image contract. No training data is needed for this project."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import torch


def validate_pairs(pairs):
    """Return nonempty, unique ImageNet (label, seed) pairs without silent coercion."""
    result = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each sample must contain a label and seed")
        label, seed = pair
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in pair):
            raise ValueError("Labels and seeds must be integers")
        if not 0 <= label < 1000 or not 0 <= seed < 2**63 - 1_000_000:
            raise ValueError("Invalid ImageNet label or seed")
        result.append((int(label), int(seed)))
    if not result or len(set(result)) != len(result):
        raise ValueError("Sample pairs must be nonempty and unique")
    if len({s for _, s in result}) != len(result):
        raise ValueError("Use a different seed for every image")
    return result


REQUIRED_SPLITS = ("calibration", "validation", "test")
OPTIONAL_SPLITS = ("fid",)  # final FID/CLIP only; never used to select settings


def make_protocol(labels=(207, 360, 387, 974, 88, 979, 417, 279),
                  calibration_per_class=2, validation_per_class=2, test_per_class=8,
                  fid_per_class=0):
    """Balanced pilot with disjoint seed sets; all members use this file.

    fid_per_class > 0 appends a ``fid`` split over ALL 1000 ImageNet classes
    (class-balanced, seeds continuing after the test split).
    """
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Use nonempty, unique class labels")
    counts = [calibration_per_class, validation_per_class, test_per_class]
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in counts):
        raise ValueError("Samples per class must be positive integers")
    if isinstance(fid_per_class, bool) or not isinstance(fid_per_class, Integral) or fid_per_class < 0:
        raise ValueError("fid_per_class must be a nonnegative integer")
    splits = {}
    next_seed = 10_000
    for name, count in zip(REQUIRED_SPLITS, counts):
        pairs = [(label, next_seed + i) for i, label in
                 enumerate(label for _ in range(count) for label in labels)]
        splits[name] = [list(pair) for pair in validate_pairs(pairs)]
        next_seed += len(pairs)
    if fid_per_class:
        pairs = [(label, next_seed + i) for i, label in
                 enumerate(label for _ in range(fid_per_class) for label in range(1000))]
        splits["fid"] = [list(pair) for pair in validate_pairs(pairs)]
    return dict(version=1, settings=dict(image_size=256, architecture="DiT-XL/2",
                sampling_steps=50, cfg_scale=4.0, sampler="official ancestral p_sample_loop",
                dtype="float32", tf32=False, deterministic_algorithms=True,
                batch_size_images=1, initial_noise_device="cpu", sampler_seed_offset=1_000_000,
                vae_id="stabilityai/sd-vae-ft-mse",
                vae_revision="31f26fdeee1355a5c34592e401dd41e45d25a493",
                dit_commit="ed81ce2229091fd4ecc9a223645f95cf379d582b",
                checkpoint_sha256="9ec1876e4c03471bca126663a30e2d1b20610b6d2f87850a39a36f25cc685521"),
                splits=splits)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    names = set(protocol.get("splits", {}))
    if (protocol.get("version") != 1 or not set(REQUIRED_SPLITS) <= names
            or names - set(REQUIRED_SPLITS) - set(OPTIONAL_SPLITS)):
        raise ValueError("Expected protocol version 1 with calibration, validation and test "
                         "splits, plus an optional fid split")
    all_seeds = set()
    for name, pairs in protocol["splits"].items():
        pairs = validate_pairs(pairs)
        seeds = {seed for _, seed in pairs}
        if all_seeds & seeds:
            raise ValueError(f"Seed leakage into {name}")
        all_seeds |= seeds
    settings = protocol["settings"]
    if settings["image_size"] != 256 or settings["architecture"] != "DiT-XL/2":
        raise ValueError("This protocol supports DiT-XL/2 at 256 pixels")
    if settings.get("tf32") is not False or settings.get("deterministic_algorithms") is not True:
        raise ValueError("Use deterministic FP32 settings with TF32 disabled")
    if settings["dtype"] != "float32" or settings["sampler"] != "official ancestral p_sample_loop":
        raise ValueError("Unsupported dtype or sampler")
    if settings.get("sampler_seed_offset") != 1_000_000 or settings.get("initial_noise_device") != "cpu" or settings.get("batch_size_images") != 1:
        raise ValueError("Unsupported input noise or batch settings")
    steps = settings["sampling_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1000:
        raise ValueError("sampling_steps must be an integer between 2 and 1000")
    if not isinstance(settings["cfg_scale"], (float, int)) or not 0 <= settings["cfg_scale"] < float("inf"):
        raise ValueError("cfg_scale must be finite and nonnegative")
    if not settings.get("vae_revision") or not settings.get("dit_commit") or not settings.get("checkpoint_sha256"):
        raise ValueError("Pin source, checkpoint and VAE revisions")
    return protocol


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def initial_latent(seed, device):
    """Same CPU FP32 initial noise on all machines; transfer only after generation."""
    return torch.randn((1, 4, 32, 32), dtype=torch.float32,
                       generator=torch.Generator(device="cpu").manual_seed(int(seed))).to(device)


def preprocess_image(image, expected_size=None):
    """Canonical LPIPS/storage input: one RGB FP32 CPU image in [-1, 1].

    Generated images are already 256x256: do not resize, crop or normalize each
    image independently. PNGs are previews; evaluation uses the saved tensors.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("Expected image shape [1, 3, H, W]")
    if expected_size is not None and tuple(image.shape[-2:]) != (expected_size, expected_size):
        raise ValueError("Image size differs from the shared protocol")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("Images must contain finite floating-point values")
    if image.min() < -1.00001 or image.max() > 1.00001:
        raise ValueError("Expected decoded image range [-1, 1]")
    return image.detach().to(device="cpu", dtype=torch.float32).clamp(-1, 1).contiguous()
