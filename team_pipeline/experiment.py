"""Shared input, sampling, profiling and evaluation for CPU, MPS and CUDA.

Both notebooks import this implementation. Keep method changes in methods.py
or supply model_transform to run_variant.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from .methods import ScheduledBlocks
from .data import validate_pairs, preprocess_image, initial_latent


def shared_code_hash():
    """Fingerprint shared processing, metrics and built-in schedules."""
    digest = hashlib.sha256()
    for name in ["data.py", "experiment.py", "methods.py"]:
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configure_dit(repo):
    repo = Path(repo).resolve()
    if not (repo / "models.py").exists():
        raise FileNotFoundError(f"Official DiT checkout missing: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Avoid silently using an unrelated notebook module called 'models'.
    import models
    if Path(models.__file__).resolve() != repo / "models.py":
        raise RuntimeError("An unrelated 'models' module is loaded. Restart the kernel.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=select_device().type == "mps")
    install_mps_sampler_compatibility()
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def environment(requested="auto"):
    device = select_device(requested)
    packages = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "timm", "diffusers", "accelerate", "lpips", "numpy"]}
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                device=str(device), machine=platform.machine(), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                mps_available=torch.backends.mps.is_available(), macos=platform.mac_ver()[0],
                mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def select_device(requested="auto"):
    device = str(requested)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps or cuda[:index]")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this Python process.")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: select a GPU machine and CUDA-enabled PyTorch.")
        if result.index is None:
            result = torch.device("cuda", torch.cuda.current_device())
        if result.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index outside available range")
    return result


def install_mps_sampler_compatibility():
    from diffusion import gaussian_diffusion as gd
    if getattr(gd._extract_into_tensor, "_mac_compatible", False):
        return
    original = gd._extract_into_tensor
    def extract(arr, timesteps, broadcast_shape):
        if timesteps.device.type != "mps":
            return original(arr, timesteps, broadcast_shape)
        # Same selected coefficients, cast BEFORE device transfer (MPS has no float64).
        values = torch.from_numpy(arr).float().to(timesteps.device)[timesteps]
        while values.ndim < len(broadcast_shape):
            values = values[..., None]
        return values.expand(broadcast_shape)
    extract._mac_compatible = True
    gd._extract_into_tensor = extract


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def paired_rng(device, seed):
    """Restore CPU and selected accelerator RNG, including after exceptions."""
    state = torch.mps.get_rng_state() if device.type == "mps" else None
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == "mps":
                torch.mps.manual_seed(int(seed))
            elif device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            yield
        finally:
            if state is not None:
                torch.mps.set_rng_state(state)


class FeatureProfiler:
    """Per-sample L1 ratios, averaged separately over CFG's two branches.

    Hooks run only on original-model calibration trajectories. Scalar tensors
    stay on the device until collection ends, avoiding per-block CPU synchronization.
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.records = []
        self.timestep = None

    def __enter__(self):
        if self.model.blocks.mode != "original":
            raise ValueError("Profile original model only.")
        # Official forward_with_cfg calls self.forward directly, bypassing root
        # Module hooks. The timestep embedder is still invoked through __call__.
        self.handles.append(self.model.t_embedder.register_forward_pre_hook(self._before))
        for index, block in self.model.blocks.retained.items():
            def hook(module, args, output, index=int(index)):
                x = args[0].float()
                ratio = (output.float() - x).abs().flatten(1).sum(1) / (
                    x.abs().flatten(1).sum(1) + 1e-8)
                cond, uncond = ratio.chunk(2)
                self.records.append(torch.stack([
                    self.timestep.float(), ratio.new_tensor(index), cond.mean(), uncond.mean()
                ]).detach())
            self.handles.append(block.register_forward_hook(hook))
        return self

    def _before(self, module, args):
        self.timestep = args[0][0].detach()

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def frame(self):
        import pandas as pd
        if not self.records:
            raise RuntimeError("No profile records collected.")
        values = torch.stack(self.records).cpu().numpy()
        frame = pd.DataFrame(values, columns=["timestep", "block", "conditional", "unconditional"])
        frame[["timestep", "block"]] = frame[["timestep", "block"]].astype(int)
        frame["mean_cfg"] = frame[["conditional", "unconditional"]].mean(axis=1)
        return frame


def summarize_profile(frame, output, remove_count=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Average samples within each timestep first, then give each timestep equal weight.
    matrix = frame.groupby(["block", "timestep"])["mean_cfg"].mean().unstack()
    matrix = matrix.reindex(columns=sorted(matrix.columns, reverse=True))
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Incomplete or nonfinite feature profile.")
    scores = matrix.mean(axis=1).rename("mean_feature_change").to_frame()
    depth = len(scores)
    candidates = scores.loc[~scores.index.isin([0, depth - 1])]
    if not 0 < remove_count <= len(candidates):
        raise ValueError("Invalid removal budget.")
    removed = sorted(candidates.sort_values("mean_feature_change", kind="stable").head(remove_count).index.tolist())
    scores["selected"] = scores.index.isin(removed)
    # Stages refer to sampling order, NOT increasing numerical diffusion t.
    for stage, columns in zip(["early_noisy", "middle", "late_clean"],
                              np.array_split(matrix.columns.to_numpy(), 3)):
        scores[stage] = matrix.loc[:, columns].mean(axis=1)
    scores.to_csv(output / "block_scores.csv", index_label="block")
    matrix.to_csv(output / "feature_change_matrix.csv")
    write_json(output / "selection.json", dict(removed=removed, protected=[0, depth - 1],
        scoring="mean of per-sample normalized L1 changes, equally averaged over CFG branches and timesteps"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, values, title in zip(axes, [matrix.to_numpy(), np.log10(matrix.to_numpy().clip(1e-12))],
                                  ["Mean normalized feature change", "log10 normalized feature change"]):
        im = ax.imshow(values, aspect="auto", interpolation="nearest")
        ticks = np.unique(np.linspace(0, len(matrix.columns) - 1, min(6, len(matrix.columns))).astype(int))
        ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks])
        ax.set_yticks(range(depth))
        ax.set_xlabel("Diffusion timestep (sampling proceeds left to right)")
        ax.set_ylabel("Block index (zero based)")
        ax.set_title(title)
        for i in removed:
            ax.axhline(i, color="red", linewidth=0.6, alpha=0.6)
        fig.colorbar(im, ax=ax)
    fig.savefig(output / "feature_change_heatmap.png", dpi=180)
    plt.close(fig)
    return removed, scores


def load_model(checkpoint, removed=(), mode="original", alpha=0.5):
    from models import DiT_XL_2
    # Avoid allocating a second full set of initialized XL weights in unified memory.
    with torch.device("meta"):
        model = DiT_XL_2(input_size=32, num_classes=1000)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.blocks = ScheduledBlocks(model.blocks, removed, mode, alpha)
    return model.eval().requires_grad_(False)


def load_compact_checkpoint(path):
    """Rebuild the schedule before loading; plain official DiT cannot load this format."""
    from models import DiT_XL_2
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta = payload["schedule"]
    if meta.get("mode") == "custom":
        raise ValueError("Custom checkpoints require the corresponding method builder before loading state_dict")
    # Avoid allocating a second full set of initialized XL weights in unified memory.
    with torch.device("meta"):
        model = DiT_XL_2(input_size=32, num_classes=1000)
    model.blocks = ScheduledBlocks(model.blocks, meta["removed"], meta["mode"], meta["alpha"])
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
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


def checkpoint_stats(model, path, keep=False, sampling_steps=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CPU serialization happens before the model goes to device; no hidden device copies.
    torch.save(dict(format_version=1, architecture="DiT-XL/2-256",
                    schedule=(model.blocks.metadata() if hasattr(model.blocks, "metadata") else dict(mode="custom")), state_dict=model.state_dict()), path)
    result = dict(unique_parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_bytes=path.stat().st_size,
                  parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  unique_blocks=(len(model.blocks.retained) if hasattr(model.blocks, "retained") else len(model.blocks)),
                  block_evaluations_per_forward=len(model.blocks),
                  effective_block_evaluations_per_forward=_effective_block_evaluations(model.blocks))
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
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt", str(path))
    return path


@torch.inference_mode()
def generate(model, vae, diffusion, label, seed, cfg_scale=4.0, progress=False):
    device = next(model.parameters()).device
    # Independent CPU generator makes initial latents independent of prior work.
    z = initial_latent(seed, device)
    z = torch.cat([z, z], dim=0)
    labels = torch.tensor([int(label), 1000], device=device)
    # Reset the complete ancestral-sampler noise stream for every paired sample.
    with paired_rng(device, int(seed) + 1_000_000):
        samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, noise=z,
            clip_denoised=False, model_kwargs=dict(y=labels, cfg_scale=cfg_scale),
            progress=progress, device=device)
        image = vae.decode(samples[:1] / 0.18215).sample.clamp(-1, 1)
    if not torch.isfinite(image).all():
        raise RuntimeError(f"Nonfinite image: label={label}, seed={seed}")
    return preprocess_image(image)


@torch.inference_mode()
def warmup(model, vae, cfg_scale):
    device = next(model.parameters()).device
    z = torch.zeros(2, 4, 32, 32, device=device)
    model.forward_with_cfg(z, torch.full((2,), 999, device=device, dtype=torch.long),
                           torch.tensor([0, 1000], device=device), cfg_scale)
    vae.decode(z[:1] / 0.18215)
    synchronize(device)


def _prepare(config, mode, removed, alpha, model_transform=None):
    device = select_device(config.get("device", "auto"))
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")
    from diffusion import create_diffusion
    from diffusers import AutoencoderKL
    model = load_model(config["checkpoint"], removed, mode, alpha)
    if model_transform is not None:
        model = model_transform(model)
        if not isinstance(model, nn.Module):
            raise TypeError("model_transform must return a torch.nn.Module")
        model.eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(config["vae_id"], revision=config.get("vae_revision"))
    vae.eval().requires_grad_(False)
    diffusion = create_diffusion(str(config["sampling_steps"]))
    return model, vae, diffusion


def _cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrate(config, pairs, output):
    import pandas as pd
    pairs = validate_pairs(pairs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, vae, diffusion = _prepare(config, "original", (), 1)
    try:
        device = select_device(config.get("device", "auto"))
        model.to(device); vae.to(device)
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
        _cleanup()


def run_variant(config, pairs, output, mode="original", removed=(), alpha=0.5,
                keep_checkpoint=False, model_transform=None, method_metadata=None):
    import pandas as pd
    from torchvision.utils import save_image
    pairs = validate_pairs(pairs)
    select_device(config.get("device", "auto"))
    if model_transform is not None and (mode != "original" or removed):
        raise ValueError("Use either a custom transform or a built-in mode, not both")
    output = Path(output)
    # Never silently mix outputs from different settings or overwrite prior results.
    output.mkdir(parents=True, exist_ok=False)
    model, vae, diffusion = _prepare(config, mode, removed, alpha, model_transform)
    try:
        schedule = model.blocks.metadata() if hasattr(model.blocks, "metadata") else dict(mode="custom")
        stats = checkpoint_stats(
            model, output / "model.pt", keep_checkpoint, config.get("sampling_steps")
        )
        write_json(output / "manifest.json", dict(config=config, pairs=pairs,
                   schedule=schedule, stats=stats, method_metadata=method_metadata or {},
                   shared_code_sha256=shared_code_hash(),
                   runtime=environment(config.get("device", "auto")),
                   vae_commit=getattr(vae.config, "_commit_hash", None),
                   timing="synchronized diffusion + VAE decode + CPU image transfer; excludes loading, warmup, PNG/checkpoint writes, profiling and LPIPS"))
        device = select_device(config.get("device", "auto"))
        model.to(device); vae.to(device)
        warmup(model, vae, config["cfg_scale"])
        rows, images = [], []
        for index, (label, seed) in enumerate(pairs):
            _cleanup()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            image = generate(model, vae, diffusion, label, seed, config["cfg_scale"])
            synchronize(device)
            elapsed = time.perf_counter() - start
            row = dict(index=index, label=label, seed=seed, latency_s=elapsed,
                       device=str(device),
                       peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                       peak_reserved_bytes=(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
                       mps_allocated_bytes_after_sample=(torch.mps.current_allocated_memory()
                           if device.type == "mps" else None),
                       mps_driver_bytes_after_sample=(torch.mps.driver_allocated_memory()
                           if device.type == "mps" else None))
            rows.append(row)
            images.append(image)
            image = preprocess_image(image, config["image_size"])
            torch.save(image, output / f"image_{index:04d}.pt")
            save_image(image, output / f"image_{index:04d}.png", normalize=True, value_range=(-1, 1))
            pd.DataFrame(rows).to_csv(output / "samples.csv", index=False)
            print(f"{mode}: {index + 1}/{len(pairs)}, {elapsed:.2f} s", flush=True)
        save_image(torch.cat(images), output / "grid.png", nrow=4,
                   normalize=True, value_range=(-1, 1))
        return pd.DataFrame(rows)
    finally:
        del model, vae, diffusion
        _cleanup()


@torch.inference_mode()
def compare_results(root, names, metric=None):
    """LPIPS runs on CPU after all device models are released; never contaminates timing."""
    import pandas as pd
    root = Path(root)
    if metric is None:
        import lpips
        metric = lpips.LPIPS(net="alex", version="0.1").cpu().eval()
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide nonempty, unique variant names")
    baseline = pd.read_csv(root / "original" / "samples.csv")
    ref_manifest = json.loads((root / "original" / "manifest.json").read_text())
    ref_stats = ref_manifest["stats"]
    ref_latency = float(baseline["latency_s"].mean())
    all_rows, summary = [], []
    for name in names:
        folder = root / name
        samples = pd.read_csv(folder / "samples.csv")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("shared_code_sha256") != ref_manifest.get("shared_code_sha256"):
            raise ValueError(f"Shared pipeline code differs: {name}")
        if manifest.get("runtime") != ref_manifest.get("runtime"):
            raise ValueError(f"Hardware or software environment differs: {name}")
        if manifest["config"] != ref_manifest["config"] or manifest["pairs"] != ref_manifest["pairs"]:
            raise ValueError(f"Mismatched configuration or sample pairs: {name}")
        if not samples[["index", "label", "seed"]].equals(baseline[["index", "label", "seed"]]):
            raise ValueError(f"Mismatched sample order: {name}")
        pairs = validate_pairs(manifest["pairs"])
        expected = pd.DataFrame([(i, label, seed) for i, (label, seed) in enumerate(pairs)],
                                columns=["index", "label", "seed"])
        if not samples[["index", "label", "seed"]].equals(expected):
            raise ValueError(f"Incomplete or invalid sample manifest: {name}")
        scores = []
        for i in samples["index"]:
            ref = torch.load(root / "original" / f"image_{i:04d}.pt", weights_only=True)
            image = torch.load(folder / f"image_{i:04d}.pt", weights_only=True)
            ref = preprocess_image(ref, manifest["config"]["image_size"])
            image = preprocess_image(image, manifest["config"]["image_size"])
            score = float(metric(ref, image).item())
            if not np.isfinite(score):
                raise ValueError("Nonfinite paired metric")
            scores.append(score)
        samples["paired_lpips"] = scores
        samples["variant"] = name
        all_rows.append(samples)
        stats = manifest["stats"]
        latency = float(samples["latency_s"].mean())
        parameters = int(stats["unique_parameters"])
        checkpoint_bytes = int(stats["checkpoint_bytes"])
        ref_flops = ref_stats.get("dit_sampling_flops")
        variant_flops = stats.get("dit_sampling_flops")
        summary.append(dict(variant=name, images=len(samples), lpips_mean=float(np.mean(scores)),
            lpips_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            unique_parameters=parameters,
            parameter_reduction_pct=100 * (1 - parameters / ref_stats["unique_parameters"]),
            unique_blocks=stats["unique_blocks"],
            block_evaluations_per_forward=stats["block_evaluations_per_forward"],
            effective_block_evaluations_per_forward=stats.get(
                "effective_block_evaluations_per_forward", stats["block_evaluations_per_forward"]),
            checkpoint_MiB=checkpoint_bytes / 2**20,
            checkpoint_reduction_pct=100 * (1 - checkpoint_bytes / ref_stats["checkpoint_bytes"]),
            dit_sampling_GFLOPs=(variant_flops / 1e9 if variant_flops is not None else None),
            flops_reduction_pct=(100 * (1 - variant_flops / ref_flops)
                                 if variant_flops is not None and ref_flops else None),
            device=samples["device"].iloc[0] if "device" in samples else "cuda",
            peak_GPU_allocated_GiB=(samples["peak_allocated_bytes"].max() / 2**30 if "peak_allocated_bytes" in samples else None),
            peak_GPU_reserved_GiB=(samples["peak_reserved_bytes"].max() / 2**30 if "peak_reserved_bytes" in samples else None),
            max_post_sample_MPS_allocated_GiB=(samples["mps_allocated_bytes_after_sample"].max() / 2**30 if "mps_allocated_bytes_after_sample" in samples else None),
            max_post_sample_MPS_driver_GiB=(samples["mps_driver_bytes_after_sample"].max() / 2**30 if "mps_driver_bytes_after_sample" in samples else None),
            latency_mean_s=latency, latency_std_s=samples["latency_s"].std(ddof=0),
            latency_reduction_pct=100 * (1 - latency / ref_latency),
            latency_speedup_x=ref_latency / latency,
            throughput_images_per_s=1 / latency))
    result = pd.DataFrame(summary)
    result.to_csv(root / "results.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(root / "paired_metrics.csv", index=False)
    write_json(root / "evaluation.json", dict(
        metric="lpips_alex_v0.1" if metric.__class__.__module__.startswith("lpips") else "injected_metric",
        preprocessing="saved CPU FP32 RGB tensors in [-1,1]; no resizing",
        variants=names, images=len(baseline), paired_reference="original",
        shared_code_sha256=shared_code_hash(),
        interpretation="distance from original output, not absolute generation quality"))
    return result


def comparison_grid(root, names, pairs, count=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(root)
    count = min(count, len(pairs))
    fig, axes = plt.subplots(len(names), count, figsize=(2.4 * count, 2.7 * len(names)),
                             squeeze=False, constrained_layout=True)
    for row, name in enumerate(names):
        for column in range(count):
            ax = axes[row, column]
            with Image.open(root / name / f"image_{column:04d}.png") as image:
                ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"class {pairs[column][0]}\nseed {pairs[column][1]}", fontsize=9)
            if column == 0:
                ax.set_ylabel(name)
    fig.savefig(root / "comparison_grid.png", dpi=160)
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
