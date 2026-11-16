"""Generative metrics for the final comparison: CLIP score and FID-1K.

These read the saved FP32 image tensors of finished runs; they never generate
images and never touch timing. Heavy dependencies (transformers, torch-fidelity)
are imported lazily so the sampling pipeline does not require them.

FID here uses ~1,000 class-balanced images, so absolute values are biased high and
are NOT comparable with published FID-50K. Use it only to compare variants made
with the same protocol, and say "FID-1K".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import preprocess_image, validate_pairs

CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPT = "a photo of a {}"
ADM_REFERENCE_URL = ("https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                     "ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz")


def imagenet_class_names():
    """The 1000 ImageNet-1k class names, indexed like the DiT label ids."""
    from torchvision.models import ResNet50_Weights
    names = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
    if len(names) != 1000:
        raise RuntimeError("Expected 1000 ImageNet class names")
    return names


def to_uint8(images):
    """Saved [-1, 1] FP32 images to the uint8 [0, 255] range Inception expects."""
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def load_variant_images(root, name):
    """Return (images [N,3,H,W] float, pairs, manifest) for a COMPLETE run folder."""
    folder = Path(root) / name
    manifest = json.loads((folder / "manifest.json").read_text())
    pairs = validate_pairs(manifest["pairs"])
    samples = pd.read_csv(folder / "samples.csv")
    expected = pd.DataFrame([(i, label, seed) for i, (label, seed) in enumerate(pairs)],
                            columns=["index", "label", "seed"])
    if not samples[["index", "label", "seed"]].equals(expected):
        raise ValueError(f"Incomplete or inconsistent run: {name}")
    size = manifest["config"]["image_size"]
    images = torch.cat([
        preprocess_image(torch.load(folder / f"image_{i:04d}.pt", weights_only=True), size)
        for i in range(len(pairs))])
    return images, pairs, manifest


def check_same_setup(reference, manifest, name):
    """Variants may differ only in the method, exactly like compare_results."""
    for key in ("shared_code_sha256", "runtime", "config", "pairs"):
        if manifest.get(key) != reference.get(key):
            raise ValueError(f"{key} differs from the baseline: {name}")


# --------------------------------------------------------------------------- FID

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Frechet distance between Gaussians, stable for rank-deficient covariances.

    tr sqrt(S1 S2) is evaluated as tr sqrt(R S2 R) with R = sqrt(S1) from a
    symmetric eigendecomposition, so every matrix is real symmetric PSD.
    """
    mu1, mu2 = np.asarray(mu1, np.float64), np.asarray(mu2, np.float64)
    sigma1, sigma2 = np.asarray(sigma1, np.float64), np.asarray(sigma2, np.float64)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("Feature statistics have different shapes")
    values, vectors = np.linalg.eigh((sigma1 + sigma1.T) / 2)
    root = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
    middle = root @ sigma2 @ root
    eigenvalues = np.linalg.eigvalsh((middle + middle.T) / 2)
    trace_sqrt = np.sqrt(np.clip(eigenvalues, 0, None)).sum()
    diff = mu1 - mu2
    result = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_sqrt)
    if not np.isfinite(result):
        raise ValueError("Nonfinite FID")
    return max(result, 0.0)


def feature_statistics(features):
    features = np.asarray(features, np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("Need an [N>=2, D] feature matrix")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def fid_from_features(a, b):
    return frechet_distance(*feature_statistics(a), *feature_statistics(b))


def load_inception(device="cpu"):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    return FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()


@torch.inference_mode()
def inception_features(images, extractor=None, batch_size=50, device="cpu"):
    """2048-d pool features (TF-FID-compatible Inception) of [-1,1] images."""
    extractor = extractor if extractor is not None else load_inception(device)
    out = []
    for start in range(0, len(images), batch_size):
        batch = to_uint8(images[start:start + batch_size]).to(device)
        features = extractor(batch)
        features = features[0] if isinstance(features, (tuple, list)) else features
        out.append(features.float().cpu())
    return torch.cat(out).numpy()


def load_reference_statistics(path):
    """(mu, sigma) from an ADM-style reference npz; fail loudly if keys are absent."""
    with np.load(path) as data:
        if "mu" not in data.files or "sigma" not in data.files:
            raise ValueError(f"{path} has no mu/sigma statistics (keys: {data.files})")
        return data["mu"].astype(np.float64), data["sigma"].astype(np.float64)


def fid_report(root, names, reference_stats=None, extractor=None, device="cpu", batch_size=50):
    """Write ``<root>/<name>/fid.json`` for each name; the baseline is ``original``."""
    root = Path(root)
    names = list(names)
    if "original" not in names:
        names = ["original"] + names
    reference_manifest = json.loads((root / "original" / "manifest.json").read_text())
    extractor = extractor if extractor is not None else load_inception(device)
    features = {}
    for name in names:
        cache = root / name / "inception_features.npy"
        images, pairs, manifest = load_variant_images(root, name)
        check_same_setup(reference_manifest, manifest, name)
        if cache.exists() and len(np.load(cache)) == len(images):
            features[name] = np.load(cache)
        else:
            features[name] = inception_features(images, extractor, batch_size, device)
            np.save(cache, features[name])
    adm = load_reference_statistics(reference_stats) if reference_stats else None
    rows = []
    for name in names:
        row = dict(variant=name, images=len(features[name]), metric="FID-1K-style (relative use only)",
                   fid_vs_original=fid_from_features(features[name], features["original"]))
        if adm is not None:
            row["fid_vs_adm_imagenet256"] = frechet_distance(
                *feature_statistics(features[name]), *adm)
        (root / name / "fid.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- CLIP score

def clip_scores_from_embeddings(image_embeddings, text_embeddings):
    """Per-image cosine similarity between paired image and text embeddings."""
    a = torch.nn.functional.normalize(image_embeddings.float(), dim=-1)
    b = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)
    return (a * b).sum(-1)


def clip_preprocess(images):
    """[-1,1] 256px images to CLIP's 224px bicubic-resized, normalized input."""
    x = (images.clamp(-1, 1) + 1) / 2
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic",
                                        align_corners=False, antialias=True).clamp(0, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def load_clip(device="cpu"):
    """Return (encode_image(images), encode_text(prompts), info) for CLIP ViT-B/32."""
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)

    @torch.inference_mode()
    def encode_image(images):
        return model.get_image_features(pixel_values=clip_preprocess(images).to(device)).float().cpu()

    @torch.inference_mode()
    def encode_text(prompts):
        tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        return model.get_text_features(**tokens).float().cpu()

    return encode_image, encode_text, dict(model=CLIP_ID,
                                           revision=getattr(model.config, "_commit_hash", None))


def clip_report(root, names, encode_image=None, encode_text=None, info=None,
                class_names=None, batch_size=50):
    """Write ``<root>/<name>/clip.json`` and ``clip_scores.csv`` per variant."""
    root = Path(root)
    if encode_image is None:
        encode_image, encode_text, info = load_clip()
    class_names = class_names if class_names is not None else imagenet_class_names()
    reference_manifest = json.loads((root / "original" / "manifest.json").read_text())
    rows = []
    for name in names:
        images, pairs, manifest = load_variant_images(root, name)
        check_same_setup(reference_manifest, manifest, name)
        labels = [label for label, _ in pairs]
        prompts = [PROMPT.format(class_names[label]) for label in labels]
        image_emb = torch.cat([encode_image(images[i:i + batch_size])
                               for i in range(0, len(images), batch_size)])
        text_emb = encode_text(prompts)
        cosine = clip_scores_from_embeddings(image_emb, text_emb).numpy()
        pd.DataFrame(dict(index=range(len(pairs)), label=labels, cosine=cosine)).to_csv(
            root / name / "clip_scores.csv", index=False)
        row = dict(variant=name, images=len(cosine), clip_cosine_mean=float(cosine.mean()),
                   clip_cosine_std=float(cosine.std(ddof=1)) if len(cosine) > 1 else 0.0,
                   clip_score=float(100 * np.clip(cosine, 0, None).mean()),
                   prompt=PROMPT, **(info or {}))
        (root / name / "clip.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        rows.append(row)
    return pd.DataFrame(rows)
