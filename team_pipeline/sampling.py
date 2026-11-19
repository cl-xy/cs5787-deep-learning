"""Model loading, deterministic sampling, calibration and the shared variant runner."""
from __future__ import annotations

import importlib
from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
from torchvision.utils import save_image

from xsa_method.checkpoint import load_compact_checkpoint  # noqa: F401  (re-exported)

from .data import (LATENT_SHAPE, NUM_IMAGENET_CLASSES, SAMPLER_SEED_OFFSET,
                   initial_latent, preprocess_image, validate_pairs)
from .methods import DEFAULT_ALPHA, ScheduledBlocks
from .profiling import FeatureProfiler
from .provenance import shared_code_hash
from .runtime import (cleanup, environment, paired_rng, select_device, synchronize,
                      write_json)

VAE_SCALE = 0.18215                    # Stable Diffusion latent scaling used by DiT.
NULL_CLASS = NUM_IMAGENET_CLASSES      # Unconditional (classifier-free guidance) label.
WARMUP_TIMESTEP = 999
DIT_INPUT_SIZE = LATENT_SHAPE[-1]
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
PREVIEW_IMAGES = 16                    # Only a small preview grid is kept in RAM.
GRID_COLUMNS = 4
TIMING_DESCRIPTION = ("synchronized diffusion + VAE decode + CPU image transfer; excludes "
                      "loading, warmup, PNG/checkpoint writes, profiling and LPIPS")


def load_model(checkpoint, removed=(), mode="original", alpha=DEFAULT_ALPHA,
               schedule_cls=ScheduledBlocks):
    """Load the pretrained DiT-XL/2 and wrap its blocks in ``schedule_cls``."""
    models = importlib.import_module("models")
    # Avoid allocating a second full set of initialized XL weights in unified memory.
    with torch.device("meta"):
        model = models.DiT_XL_2(input_size=DIT_INPUT_SIZE, num_classes=NUM_IMAGENET_CLASSES)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.blocks = schedule_cls(model.blocks, removed, mode, alpha)
    return model.eval().requires_grad_(False)


def _effective_block_evaluations(blocks):
    """Count block calls that do real work (zero-scale damped slots are no-ops)."""
    schedule = getattr(blocks, "schedule", None)
    if schedule is None:
        return len(blocks)
    return sum(float(scale) != 0.0 for _, scale, _ in schedule)


def estimate_dit_flops(model, sampling_steps=1, cfg_batch_size=2):
    """Estimate DiT arithmetic FLOPs for CFG sampling.

    Counts multiply-add as two FLOPs for convolution/linear projections and the
    two attention matrix products. Elementwise operations, normalization,
    softmax, embeddings and VAE decoding are excluded. This makes the estimate
    deterministic and useful for relative comparisons across block schedules.
    Returns ``None`` for custom models that do not expose the official DiT shape.
    """
    steps = int(sampling_steps)
    batch = int(cfg_batch_size)
    if steps <= 0 or batch <= 0:
        raise ValueError("sampling_steps and cfg_batch_size must be positive")
    try:
        block = next(iter(model.blocks.retained.values()))
        tokens = int(model.x_embedder.num_patches)
        hidden = int(model.pos_embed.shape[-1])
        patch_height, patch_width = map(int, model.x_embedder.patch_size)
        input_channels = int(model.in_channels)
        output_channels = int(model.out_channels)
        mlp_hidden = int(block.mlp.fc1.out_features)
        heads = int(model.num_heads)
        if hidden % heads:
            return None
    except (AttributeError, StopIteration, TypeError, ValueError):
        return None

    # Patch projection and timestep MLP. Label lookup has no arithmetic FLOPs.
    patch_projection = 2 * batch * tokens * hidden * input_channels * patch_height * patch_width
    timestep_mlp = 2 * batch * (256 * hidden + hidden * hidden)

    # qkv + output projections, QK^T + attention-value products, MLP, adaLN.
    projections = 2 * batch * tokens * (3 * hidden * hidden + hidden * hidden)
    attention = 4 * batch * tokens * tokens * hidden
    mlp = 4 * batch * tokens * hidden * mlp_hidden
    adaptive_norm = 2 * batch * hidden * (6 * hidden)
    block_flops = projections + attention + mlp + adaptive_norm
    block_calls = _effective_block_evaluations(model.blocks)

    final_adaptive_norm = 2 * batch * hidden * (2 * hidden)
    final_projection = 2 * batch * tokens * hidden * (
        patch_height * patch_width * output_channels
    )
    forward = (
        patch_projection
        + timestep_mlp
        + block_calls * block_flops
        + final_adaptive_norm
        + final_projection
    )
    return dict(
        dit_forward_flops=int(forward),
        dit_sampling_flops=int(forward * steps),
        flops_convention=(
            "multiply-add=2; CFG batch=2; excludes elementwise ops, normalization, "
            "softmax, embeddings, sampler arithmetic and VAE decode"
        ),
    )

def _schedule_metadata(model):
    blocks = model.blocks
    return blocks.metadata() if hasattr(blocks, "metadata") else dict(mode="custom")


def checkpoint_stats(model, path, keep=False, sampling_steps=None):
    """Serialize the model to measure its on-disk size; delete it unless ``keep``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CPU serialization happens before the model goes to device; no hidden device copies.
    torch.save(dict(format_version=1, architecture="DiT-XL/2-256",
                    schedule=_schedule_metadata(model), state_dict=model.state_dict()), path)
    blocks = model.blocks
    unique_blocks = len(blocks.retained) if hasattr(blocks, "retained") else len(blocks)
    result = dict(
        unique_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_bytes=path.stat().st_size,
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        unique_blocks=unique_blocks,
        block_evaluations_per_forward=len(blocks),
        effective_block_evaluations_per_forward=_effective_block_evaluations(blocks),
    )
    if sampling_steps is not None:
        flops = estimate_dit_flops(model, sampling_steps)
        if flops is not None:
            result.update(flops)
    if not keep:
        path.unlink()
    return result


def download_checkpoint(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(path))
    return path


@torch.inference_mode()
def generate(model, vae, diffusion, label, seed, cfg_scale=4.0, progress=False,
             seed_offset=SAMPLER_SEED_OFFSET):
    """Sample one image for ``(label, seed)``; identical noise for every method."""
    device = next(model.parameters()).device
    # Independent CPU generator makes initial latents independent of prior work.
    z = initial_latent(seed, device)
    z = torch.cat([z, z], dim=0)
    labels = torch.tensor([int(label), NULL_CLASS], device=device)
    # Reset the complete ancestral-sampler noise stream for every paired sample.
    with paired_rng(device, int(seed) + seed_offset):
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, noise=z, clip_denoised=False,
            model_kwargs=dict(y=labels, cfg_scale=cfg_scale), progress=progress, device=device)
        image = vae.decode(samples[:1] / VAE_SCALE).sample.clamp(-1, 1)
    if not torch.isfinite(image).all():
        raise RuntimeError(f"Nonfinite image: label={label}, seed={seed}")
    return preprocess_image(image)


@torch.inference_mode()
def warmup(model, vae, cfg_scale):
    """Run one throwaway forward/decode so kernel compilation is not timed."""
    device = next(model.parameters()).device
    z = torch.zeros(2, *LATENT_SHAPE[1:], device=device)
    timesteps = torch.full((2,), WARMUP_TIMESTEP, device=device, dtype=torch.long)
    labels = torch.tensor([0, NULL_CLASS], device=device)
    model.forward_with_cfg(z, timesteps, labels, cfg_scale)
    vae.decode(z[:1] / VAE_SCALE)
    synchronize(device)


def _load_vae(config):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config["vae_id"], revision=config.get("vae_revision"))
    return vae.eval().requires_grad_(False)


def _prepare(config, mode, removed, alpha, model_transform=None, schedule_cls=ScheduledBlocks):
    """Build the model, VAE and diffusion sampler on CPU."""
    device = select_device(config.get("device", "auto"))
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")
    create_diffusion = importlib.import_module("diffusion").create_diffusion
    model = load_model(config["checkpoint"], removed, mode, alpha, schedule_cls)
    if model_transform is not None:
        model = model_transform(model)
        if not isinstance(model, nn.Module):
            raise TypeError("model_transform must return a torch.nn.Module")
        model.eval().requires_grad_(False)
    diffusion = create_diffusion(str(config["sampling_steps"]))
    return model, _load_vae(config), diffusion


def calibrate(config, pairs, output):
    """Profile per-block feature change of the ORIGINAL model on the calibration pairs."""
    pairs = validate_pairs(pairs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, vae, diffusion = _prepare(config, "original", (), 1)
    try:
        device = select_device(config.get("device", "auto"))
        model.to(device)
        vae.to(device)
        frames = []
        for label, seed in pairs:
            with FeatureProfiler(model) as profiler:
                generate(model, vae, diffusion, label, seed, config["cfg_scale"])
            frame = profiler.frame()
            frame["label"], frame["seed"] = label, seed
            frames.append(frame)
            print(f"Profiled label={label}, seed={seed}", flush=True)
            del profiler
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(output / "feature_changes.csv", index=False)
        write_json(output / "calibration_manifest.json", dict(config=config, pairs=pairs))
        return frame
    finally:
        del model, vae, diffusion
        cleanup()


def _write_manifest(output, config, pairs, model, vae, stats, method_metadata):
    write_json(output / "manifest.json", dict(
        config=config, pairs=pairs, schedule=_schedule_metadata(model), stats=stats,
        method_metadata=method_metadata or {}, shared_code_sha256=shared_code_hash(),
        runtime=environment(config.get("device", "auto")),
        vae_commit=getattr(vae.config, "_commit_hash", None), timing=TIMING_DESCRIPTION))


def _sample_row(index, label, seed, latency, device):
    """One samples.csv row: latency plus accelerator memory counters (None elsewhere)."""
    cuda = device.type == "cuda"
    mps = device.type == "mps"
    return dict(
        index=index, label=label, seed=seed, latency_s=latency, device=str(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None,
        mps_allocated_bytes_after_sample=torch.mps.current_allocated_memory() if mps else None,
        mps_driver_bytes_after_sample=torch.mps.driver_allocated_memory() if mps else None,
    )


def _save_sample(output, index, image, image_size):
    """Save the canonical tensor (used for metrics) and a PNG preview."""
    image = preprocess_image(image, image_size)
    torch.save(image, output / f"image_{index:04d}.pt")
    save_image(image, output / f"image_{index:04d}.png", normalize=True, value_range=(-1, 1))


def run_variant(config, pairs, output, mode="original", removed=(), alpha=DEFAULT_ALPHA,
                keep_checkpoint=False, model_transform=None, method_metadata=None,
                schedule_cls=ScheduledBlocks):
    """Generate every pair with one method and write images, timings and a manifest."""
    pairs = validate_pairs(pairs)
    device = select_device(config.get("device", "auto"))
    if model_transform is not None and (mode != "original" or removed):
        raise ValueError("Use either a custom transform or a built-in mode, not both")
    output = Path(output)
    # Never silently mix outputs from different settings or overwrite prior results.
    output.mkdir(parents=True, exist_ok=False)
    model, vae, diffusion = _prepare(config, mode, removed, alpha, model_transform, schedule_cls)
    try:
        stats = checkpoint_stats(model, output / "model.pt", keep_checkpoint,
                                 config.get("sampling_steps"))
        _write_manifest(output, config, pairs, model, vae, stats, method_metadata)
        model.to(device)
        vae.to(device)
        warmup(model, vae, config["cfg_scale"])
        rows, previews = [], []
        for index, (label, seed) in enumerate(pairs):
            cleanup()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            image = generate(model, vae, diffusion, label, seed, config["cfg_scale"])
            synchronize(device)
            elapsed = time.perf_counter() - start
            rows.append(_sample_row(index, label, seed, elapsed, device))
            if len(previews) < PREVIEW_IMAGES:
                previews.append(image)
            _save_sample(output, index, image, config["image_size"])
            pd.DataFrame(rows).to_csv(output / "samples.csv", index=False)
            print(f"{mode}: {index + 1}/{len(pairs)}, {elapsed:.2f} s", flush=True)
        save_image(torch.cat(previews), output / "grid.png", nrow=GRID_COLUMNS,
                   normalize=True, value_range=(-1, 1))
        return pd.DataFrame(rows)
    finally:
        del model, vae, diffusion
        cleanup()
