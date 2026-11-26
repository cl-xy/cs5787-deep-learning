# Notebook cell 2
# SETUP — no experiment starts here.
from pathlib import Path
import os, sys, subprocess
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
WORKSPACE = Path.cwd()
USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
OUTPUT_BASE = (Path("/content/drive/MyDrive/DiT_team_runs")
               if USE_GOOGLE_DRIVE else WORKSPACE / "team_runs")
subprocess.run([sys.executable, "-m", "pip", "install",
                "timm", "diffusers", "accelerate", "lpips", "numpy",
                "pandas", "matplotlib", "pillow", "pytest", "huggingface_hub"], check=True)
print("Source workspace:", WORKSPACE)
print("Output base:", OUTPUT_BASE)


# Notebook cell 5
"""The team's input and image contract. Sampling uses label/seed pairs; real references are loaded only for evaluation."""

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


def make_protocol(labels=None,
                  calibration_per_class=2, validation_per_class=2, test_per_class=8):
    """Balanced pilot with three disjoint seed sets; all members use this file."""
    labels = list(CLASS_LABELS if labels is None else labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Use nonempty, unique class labels")
    counts = [calibration_per_class, validation_per_class, test_per_class]
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in counts):
        raise ValueError("Samples per class must be positive integers")
    splits = {}
    next_seed = 10_000
    for name, count in zip(["calibration", "validation", "test"], counts):
        pairs = [(label, next_seed + i) for i, label in
                 enumerate(label for _ in range(count) for label in labels)]
        splits[name] = [list(pair) for pair in validate_pairs(pairs)]
        next_seed += len(pairs)
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
    protocol = (json.loads(json.dumps(path)) if isinstance(path, dict)
                else json.loads(Path(path).read_text()))
    if protocol.get("version") != 1 or set(protocol.get("splits", {})) != {"calibration", "validation", "test"}:
        raise ValueError("Expected protocol version 1 with calibration, validation and test splits")
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


# Notebook cell 7
"""Method-specific block schedules. Team members can add model transformations here."""
from torch import nn


class ScheduledBlocks(nn.Module):
    """Register each retained block ONCE; execute it at one or more depth slots.

    Official DiT.forward iterates over model.blocks. This container preserves that
    interface, while its state_dict contains only retained modules.
    """
    def __init__(self, blocks, removed=(), mode="original", alpha="inverse_k"):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        self.mode = mode
        self.alpha = "inverse_k" if alpha == "inverse_k" else float(alpha)
        self.removed = sorted(set(int(i) for i in removed))
        if mode not in {"original", "removal", "loop", "damped"}:
            raise ValueError(f"Unknown mode: {mode}")
        if self.alpha != "inverse_k" and not 0 <= self.alpha <= 1:
            raise ValueError("Damping alpha must be inverse_k or a number in [0, 1].")
        if any(i < 0 or i >= self.depth for i in self.removed):
            raise ValueError("Removed index outside block range.")
        if mode == "original" and self.removed:
            raise ValueError("Original model cannot have removed blocks.")
        if 0 in self.removed:
            raise ValueError("Keep block 0: a replacement needs a preceding retained block.")
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        self.schedule = []
        previous = None
        iteration = 0
        for i in range(self.depth):
            if i not in self.removed:
                previous = i
                iteration = 1
                self.schedule.append((i, 1.0, False))
            elif mode != "removal":
                iteration += 1
                scale = (1.0 / iteration if self.alpha == "inverse_k" else self.alpha) if mode == "damped" else 1.0
                self.schedule.append((previous, scale, True))

    def __len__(self):
        return len(self.schedule)

    @staticmethod
    def _damped(block, scale):
        def call(x, c):
            if scale == 0:
                return x
            return x + scale * (block(x, c) - x)
        return call

    def __iter__(self):
        for index, scale, replacement in self.schedule:
            block = self.retained[str(index)]
            yield self._damped(block, scale) if replacement and scale != 1 else block

    def metadata(self):
        return dict(mode=self.mode, removed=self.removed, alpha=self.alpha,
                    damping_policy=("inverse_k" if self.alpha == "inverse_k" else "fixed") if self.mode == "damped" else "none",
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    execution_schedule=[dict(source=i, scale=a, replacement=r)
                                        for i, a, r in self.schedule])


# Notebook cell 9
"""Shared input, sampling, profiling and evaluation for CPU, MPS and CUDA.

Both notebooks import this implementation. Keep method changes in methods.py
or supply model_transform to run_variant.
"""

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


# Notebook cell 11
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


# Notebook cell 13
def load_model(checkpoint, removed=(), mode="original", alpha="inverse_k"):
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


def checkpoint_stats(model, path, keep=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CPU serialization happens before the model goes to device; no hidden device copies.
    torch.save(dict(format_version=1, architecture="DiT-XL/2-256",
                    schedule=(model.blocks.metadata() if hasattr(model.blocks, "metadata") else dict(mode="custom")), state_dict=model.state_dict()), path)
    result = dict(unique_parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_bytes=path.stat().st_size,
                  parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  unique_blocks=(len(model.blocks.retained) if hasattr(model.blocks, "retained") else len(model.blocks)),
                  block_evaluations_per_forward=len(model.blocks))
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


# Notebook cell 15
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


def run_variant(config, pairs, output, mode="original", removed=(), alpha="inverse_k",
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
        stats = checkpoint_stats(model, output / "model.pt", keep_checkpoint)
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
            # Keep only a small preview in RAM, even for 5,000-image runs.
            if len(images) < 16:
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


# Notebook cell 17
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
        summary.append(dict(variant=name, images=len(samples), lpips_mean=float(np.mean(scores)),
            lpips_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            unique_parameters=stats["unique_parameters"], unique_blocks=stats["unique_blocks"],
            block_evaluations_per_forward=stats["block_evaluations_per_forward"],
            checkpoint_MiB=stats["checkpoint_bytes"] / 2**20,
            device=samples["device"].iloc[0] if "device" in samples else "cuda",
            peak_GPU_allocated_GiB=(samples["peak_allocated_bytes"].max() / 2**30 if "peak_allocated_bytes" in samples else None),
            peak_GPU_reserved_GiB=(samples["peak_reserved_bytes"].max() / 2**30 if "peak_reserved_bytes" in samples else None),
            max_post_sample_MPS_allocated_GiB=(samples["mps_allocated_bytes_after_sample"].max() / 2**30 if "mps_allocated_bytes_after_sample" in samples else None),
            max_post_sample_MPS_driver_GiB=(samples["mps_driver_bytes_after_sample"].max() / 2**30 if "mps_driver_bytes_after_sample" in samples else None),
            latency_mean_s=samples["latency_s"].mean(), latency_std_s=samples["latency_s"].std(ddof=0)))
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


# Notebook cell 19
import inspect
from types import SimpleNamespace

SHARED_DEFINITION_NAMES = ['validate_pairs', 'make_protocol', 'load_protocol', 'protocol_hash', 'initial_latent', 'preprocess_image', 'ScheduledBlocks', 'write_json', 'configure_dit', 'environment', 'select_device', 'install_mps_sampler_compatibility', 'synchronize', 'paired_rng', 'FeatureProfiler', 'summarize_profile', 'load_model', 'load_compact_checkpoint', 'checkpoint_stats', 'download_checkpoint', 'generate', 'warmup', '_prepare', '_cleanup', 'calibrate', 'run_variant', 'compare_results', 'comparison_grid', 'sha256']

def notebook_sources():
    """Read executed function sources from IPython's source cache."""
    sources = {}
    for name in SHARED_DEFINITION_NAMES:
        obj = globals()[name]
        if inspect.isclass(obj):
            # Inspect methods individually: __main__ classes lack a module file.
            for member_name, member in vars(obj).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    sources[f"{name}.{member_name}"] = inspect.getsource(member)
        else:
            sources[name] = inspect.getsource(obj)
    return sources

def shared_code_hash():
    return hashlib.sha256(json.dumps(notebook_sources(), sort_keys=True).encode()).hexdigest()

def prepare(args):
    protocol = load_protocol(args.protocol)
    device = select_device(args.device)
    commit = configure_dit(args.repo)
    if commit != protocol["settings"]["dit_commit"]:
        raise ValueError("DiT source commit differs from the agreed protocol")
    checkpoint = download_checkpoint(Path(args.checkpoint).resolve())
    if sha256(checkpoint) != protocol["settings"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the agreed protocol")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = {**protocol["settings"], "checkpoint": str(checkpoint), "device": str(device),
              "protocol_sha256": protocol_hash(protocol), "shared_code_sha256": shared_code_hash()}
    write_json(root / "protocol.json", protocol)
    write_json(root / "notebook_sources.json", notebook_sources())
    write_json(root / "config.json", config)
    write_json(root / "environment.json", environment(str(device)))
    with (root / "pip-freeze.txt").open("w") as out:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=out, check=True)
    print(f"Prepared shared inputs: {root.resolve()}")

def load_run(args):
    root = Path(args.output)
    protocol = load_protocol(root / "protocol.json")
    config = json.loads((root / "config.json").read_text())
    if config.get("shared_code_sha256") != shared_code_hash():
        raise ValueError("Shared pipeline changed after preparation; create a new run")
    if config["protocol_sha256"] != protocol_hash(protocol):
        raise ValueError("Protocol changed after preparation; create a new run")
    if any(config.get(k) != v for k, v in protocol["settings"].items()):
        raise ValueError("Config settings changed after preparation; create a new run")
    if configure_dit(args.repo) != config["dit_commit"]:
        raise ValueError("DiT source changed after preparation")
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preparation")
    current = environment(config["device"])
    recorded = json.loads((root / "environment.json").read_text())
    if current != recorded:
        raise ValueError("Runtime environment changed; prepare a new run for comparable measurements")
    return root, protocol, config


# Notebook cell 21
# Imagenette v2: original ImageNet-1K ids, synsets and names (NOT local ids 0..9).
# Source: https://github.com/fastai/imagenette
IMAGENETTE_CLASSES = (
    (0,   "n01440764", "tench"),
    (217, "n02102040", "English springer"),
    (482, "n02979186", "cassette player"),
    (491, "n03000684", "chain saw"),
    (497, "n03028079", "church"),
    (566, "n03394916", "French horn"),
    (569, "n03417042", "garbage truck"),
    (571, "n03425413", "gas pump"),
    (574, "n03445777", "golf ball"),
    (701, "n03888257", "parachute"),
)
# The single class array used by every generation method and split.
# A subset of these ids is also supported by the reference preparation below.
CLASS_LABELS = tuple(label for label, synset, name in IMAGENETTE_CLASSES)
CALIBRATION_IMAGES_PER_CLASS = 2
VALIDATION_IMAGES_PER_CLASS = 2  # 20 images/method for selecting settings.
FINAL_IMAGES_PER_CLASS = 500   # 5,000 images/method in the held-out test.
FID_REAL_IMAGES_PER_CLASS = 500  # Balanced, fixed real reference: 5,000 images.

PROTOCOL = make_protocol(
    labels=CLASS_LABELS,
    calibration_per_class=CALIBRATION_IMAGES_PER_CLASS,
    validation_per_class=VALIDATION_IMAGES_PER_CLASS,
    test_per_class=FINAL_IMAGES_PER_CLASS,
)
CUSTOM_SETTINGS = {"removed": [4], "mode": "damped", "alpha": "inverse_k"}

"""Copy this file for each member's method; leave the shared pipeline unchanged."""


def build_model(model, settings):
    """Receive the same pretrained CPU DiT and return the modified model.

    settings is a JSON object supplied with --method-config. Use calibration and
    validation data to choose it, then freeze it for the held-out test split.
    A future training method can train here and return the resulting model; the
    runner always evaluates in inference mode with gradients disabled afterward.
    """
    model.blocks = ScheduledBlocks(
        model.blocks.retained.values(),
        removed=settings["removed"],
        mode=settings.get("mode", "damped"),
        alpha=settings.get("alpha", "inverse_k"),
    )
    return model


def run_method(split, method, name, *, alpha="inverse_k", builder=None, settings=None):
    """Direct notebook entry point; keeps the CLI's run checks."""
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    if Path(name).name != name or name in {".", "..", ""}:
        raise ValueError("name must be a single folder name")
    if (name == "original") != (method == "original" and builder is None):
        raise ValueError("Reserve original for the unmodified baseline")
    root, protocol, config = load_run(RUN_ARGS)
    removed, transform = (), None
    metadata = dict(method=method)
    if builder is not None:
        settings = json.loads(json.dumps(settings or {}))
        source = inspect.getsource(builder)
        metadata.update(settings=settings, source=source,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest())
        transform = lambda model: builder(model, settings)
        mode = "original"
    elif method in {"removal", "loop", "damped"}:
        removed = json.loads((root / "profile/selection.json").read_text())["removed"]
        mode = method
    elif method == "original":
        mode = method
    else:
        raise ValueError("Unknown method; pass builder= for a custom method")
    return run_variant(config, protocol["splits"][split], root / split / name,
                       mode=mode, removed=removed, alpha=alpha, model_transform=transform,
                       method_metadata=metadata, keep_checkpoint=KEEP_CHECKPOINTS)


def evaluate_split(split, names):
    root, protocol, config = load_run(RUN_ARGS)
    result = compare_results(root / split, names)
    comparison_grid(root / split, names, protocol["splits"][split])
    return result


# Notebook cell 23
"""Shared protocol and end-to-end artifact checks; no pretrained downloads."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn



def test_protocol_has_balanced_disjoint_splits_and_stable_hash(tmp_path):
    protocol = make_protocol(labels=[1, 2], calibration_per_class=2,
                             validation_per_class=1, test_per_class=3)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    assert load_protocol(path) == protocol
    assert [len(v) for v in protocol["splits"].values()] == [4, 2, 6]
    reordered = dict(reversed(list(protocol.items())))
    assert protocol_hash(protocol) == protocol_hash(reordered)
    changed = copy.deepcopy(protocol)
    changed["splits"]["test"][0][1] = changed["splits"]["calibration"][0][1]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="leakage"):
        load_protocol(path)


def test_input_validation_rejects_ambiguous_samples(pairs):
    with pytest.raises(ValueError):
        validate_pairs(pairs)


def test_preprocessing_and_initial_noise_are_fixed():
    before = torch.get_rng_state().clone()
    a = initial_latent(20, "cpu")
    torch.randn(5)
    assert torch.equal(a, initial_latent(20, "cpu"))
    torch.set_rng_state(before)
    x = torch.zeros(1, 3, 8, 8, dtype=torch.float64, requires_grad=True)
    y = preprocess_image(x, 8)
    assert y.dtype == torch.float32 and y.device.type == "cpu" and not y.requires_grad
    for invalid in [torch.full_like(x, float("nan")), x + 2, x.to(torch.uint8), x.squeeze(0)]:
        with pytest.raises(ValueError):
            preprocess_image(invalid, 8)
    with pytest.raises(ValueError, match="size"):
        preprocess_image(x, 16)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = ScheduledBlocks([nn.Linear(1, 1), nn.Linear(1, 1)])


def test_shared_runner_and_evaluator_reject_incomplete_or_mismatched_runs(tmp_path, monkeypatch):
    # The actual save/load, manifest, timing and evaluation code runs unchanged.
    vae = nn.Linear(1, 1)
    vae.config = SimpleNamespace()
    monkeypatch.setitem(globals(), "_prepare", lambda *a: (Tiny(), vae, None))
    monkeypatch.setitem(globals(), "warmup", lambda *a: None)
    monkeypatch.setitem(globals(), "generate", lambda model, vae, diffusion, label, seed, cfg:
                        torch.full((1, 3, 8, 8), seed / 100))
    config = dict(device="cpu", image_size=8, cfg_scale=4)
    pairs = [(1, 10), (2, 20)]
    for name in ["original", "member1"]:
        run_variant(config, pairs, tmp_path / name)
    metric = lambda x, y: (x - y).square().mean()
    result = compare_results(tmp_path, ["original", "member1"], metric=metric)
    assert result.lpips_mean.eq(0).all()
    assert result.images.eq(2).all()
    assert result.peak_GPU_allocated_GiB.isna().all()
    assert (tmp_path / "paired_metrics.csv").exists()
    with pytest.raises(FileExistsError):
        run_variant(config, pairs, tmp_path / "member1")
    samples_path = tmp_path / "member1/samples.csv"
    samples = pd.read_csv(samples_path)
    samples.iloc[:1].to_csv(samples_path, index=False)
    with pytest.raises(ValueError, match="sample order"):
        compare_results(tmp_path, ["member1"], metric=metric)
    samples.to_csv(samples_path, index=False)
    path = tmp_path / "member1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["cfg_scale"] = 5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration"):
        compare_results(tmp_path, ["member1"], metric=metric)


def test_custom_builder_keeps_shared_preparation_and_freezes_model(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(globals(), "load_model", lambda *a: Tiny())
    vae = nn.Linear(1, 1)
    factory = types.SimpleNamespace(from_pretrained=lambda *a, **k: vae)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(AutoencoderKL=factory))
    monkeypatch.setitem(sys.modules, "diffusion", types.SimpleNamespace(create_diffusion=lambda *a: "sampler"))
    seen = []
    def builder(model):
        seen.append(model)
        model.blocks = ScheduledBlocks(model.blocks.retained.values(), [1], "damped", .25)
        return model
    model, loaded_vae, diffusion = _prepare(dict(device="cpu", checkpoint="unused",
        vae_id="unused", sampling_steps=50), "original", (), .5, builder)
    assert seen == [model] and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert len(model.blocks.retained) == 1
    assert loaded_vae is vae and diffusion == "sampler"


# Notebook cell 24
# XSA definitions — execute after the shared definitions and before tests/preparation.
from contextlib import contextmanager
import math
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def xsa_filter_heads(output, value, eps=1e-8):
    """Remove each token/head's current self-value direction; last axis is head_dim."""
    if output.shape != value.shape or output.ndim != 4:
        raise ValueError("XSA needs matching four-dimensional head output/value tensors")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("XSA epsilon must be finite and positive")
    # FP32 projection also makes eps safe if this helper is used with FP16/BF16.
    dtype = torch.float32 if output.dtype in {torch.float16, torch.bfloat16} else output.dtype
    o, v = output.to(dtype), value.to(dtype)
    v_normalized = F.normalize(v, dim=-1, eps=eps)
    z = o - (o * v_normalized).sum(dim=-1, keepdim=True) * v_normalized
    return z.to(output.dtype)


@contextmanager
def xsa_attention_scope(attention, enabled=True, eps=1e-8):
    """Filter exactly one ordinary timm Attention call, with no persistent flags.

    The original forward computes fresh Q/K/V and attention (including its fused
    or unfused path, Q/K norms, scaling and masks). At the head-concatenation
    boundary we view the tensor as separate heads, filter, and concatenate again.
    This is exactly the pre-concatenation projection: reshape mixes no values.
    timm's optional post-attention norm/gate and output projection run afterward.
    Hooks and this call's value tensor are always discarded, even on failure.
    Intended for this notebook's sequential eager inference, not concurrent calls
    to the same module or torch.compile.
    """
    if not enabled:
        yield
        return
    from timm.models.vision_transformer import Attention
    if type(attention) is not Attention:
        raise TypeError("XSA adapter supports the standard timm Attention used by official DiT")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("XSA epsilon must be finite and positive")
    state = {}
    counts = {"qkv": 0, "filter": 0}

    def capture_value(module, args, qkv):
        if counts["qkv"]:
            raise RuntimeError("Expected exactly one fresh QKV evaluation per XSA call")
        b, n, three_c = qkv.shape
        h = attention.num_heads
        if three_c % (3 * h) or three_c // 3 != attention.proj.in_features:
            raise RuntimeError("Unsupported timm QKV/head layout")
        # Only THIS evaluation's V is retained until its attention output arrives.
        state["value"] = qkv.reshape(b, n, 3, h, three_c // (3 * h))[:, :, 2]
        counts["qkv"] += 1

    def filter_before_projection(module, args):
        if "value" not in state:
            raise RuntimeError("XSA output did not follow a fresh QKV evaluation")
        value = state.pop("value")
        output = args[0]
        if output.shape != (*value.shape[:2], value.shape[2] * value.shape[3]):
            raise RuntimeError("Unsupported timm attention-output layout")
        heads = output.reshape_as(value)
        filtered = xsa_filter_heads(heads, value, eps).reshape_as(output)
        counts["filter"] += 1
        return (filtered, *args[1:])

    handles = []
    try:
        handles.append(attention.qkv.register_forward_hook(capture_value))
        # New timm has a post-head-concatenation norm; old timm goes straight to
        # proj. Filtering BEFORE that norm preserves it, rather than bypassing it.
        boundary = getattr(attention, "norm", None)
        if boundary is None:
            boundary = attention.proj
        handles.append(boundary.register_forward_pre_hook(filter_before_projection))
        yield
        if counts != {"qkv": 1, "filter": 1}:
            raise RuntimeError("XSA hooks did not observe exactly one attention evaluation")
    finally:
        for handle in handles:
            handle.remove()
        state.clear()


class ScheduledBlocks(nn.Module):
    """Retain original indices and register each retained block exactly once."""
    def __init__(self, blocks, removed=(), mode="original", alpha="inverse_k", xsa_eps=1e-8):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        self.mode = mode
        self.alpha = "inverse_k" if alpha == "inverse_k" else float(alpha)
        self.xsa_eps = float(xsa_eps)
        self.removed = sorted(set(int(i) for i in removed))
        if mode not in {"original", "removal", "loop", "damped", "xsa_reuse"}:
            raise ValueError(f"Unknown mode: {mode}")
        if self.alpha != "inverse_k" and not 0 <= self.alpha <= 1:
            raise ValueError("Damping alpha must be inverse_k or a number in [0, 1]")
        if not math.isfinite(self.xsa_eps) or self.xsa_eps <= 0:
            raise ValueError("XSA epsilon must be finite and positive")
        if any(i < 0 or i >= self.depth for i in self.removed):
            raise ValueError("Removed index outside block range")
        if mode == "original" and self.removed:
            raise ValueError("Original model cannot have removed blocks")
        if 0 in self.removed:
            raise ValueError("Keep block 0: a replacement needs a preceding retained block")
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        self.schedule = []
        self.slots = []
        self.iterations = []
        previous = None
        iteration = 0
        for slot in range(self.depth):
            if slot not in self.removed:
                previous = slot
                iteration = 1
                self.schedule.append((slot, 1.0, False))
                self.slots.append(slot)
                self.iterations.append(iteration)
            elif mode != "removal":
                # k includes the original call; each retained block starts at k=1.
                # Precompute per-slot values so k resets on every model forward.
                iteration += 1
                scale = (1.0 / iteration if self.alpha == "inverse_k" else self.alpha) if mode == "damped" else 1.0
                self.schedule.append((previous, scale, True))
                self.slots.append(slot)
                self.iterations.append(iteration)

    def __len__(self):
        return len(self.schedule)

    @staticmethod
    def _damped(block, scale):
        def call(x, c):
            if scale == 0:
                return x
            return x + scale * (block(x, c) - x)
        return call

    @staticmethod
    def _xsa(block, eps):
        def call(x, c):
            # Keep DiTBlock.forward itself: adaLN, both gates, residuals and MLP.
            with xsa_attention_scope(block.attn, eps=eps):
                return block(x, c)
        return call

    def __iter__(self):
        for index, scale, replacement in self.schedule:
            block = self.retained[str(index)]
            if replacement and self.mode == "xsa_reuse":
                yield self._xsa(block, self.xsa_eps)
            elif replacement and scale != 1:
                yield self._damped(block, scale)
            else:
                yield block

    def metadata(self):
        return dict(mode=self.mode, removed=self.removed, alpha=self.alpha,
                    damping_policy=("inverse_k" if self.alpha == "inverse_k" else "fixed") if self.mode == "damped" else "none",
                    k_definition="call index of the retained source block, including its original call",
                    replacement_residual_scale=(None if self.alpha == "inverse_k" else self.alpha) if self.mode == "damped" else 1.0,
                    xsa_eps=self.xsa_eps, index_base=0,
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    execution_schedule=[dict(slot=slot, source=i, scale=a,
                        replacement=r, iteration=k, xsa=r and self.mode == "xsa_reuse")
                        for slot, (i, a, r), k in zip(self.slots, self.schedule, self.iterations)])


# %% Compact checkpoint loader: same weight format, now restores XSA metadata.
def load_compact_checkpoint(path):
    from models import DiT_XL_2
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta = payload["schedule"]
    if meta.get("mode") == "custom":
        raise ValueError("Custom checkpoints require the corresponding method builder before loading state_dict")
    if meta.get("index_base", 0) != 0:
        raise ValueError("Expected zero-based block indices")
    with torch.device("meta"):
        model = DiT_XL_2(input_size=32, num_classes=1000)
    model.blocks = ScheduledBlocks(model.blocks, meta["removed"], meta["mode"],
                                   meta["alpha"], xsa_eps=meta.get("xsa_eps", 1e-8))
    if "retained" in meta and meta["retained"] != model.blocks.metadata()["retained"]:
        raise ValueError("Checkpoint original-index mapping is inconsistent")
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    return model.eval().requires_grad_(False)



def load_fixed_selection(root):
    selection = json.loads(
        (Path(root) / "profile/selection.json").read_text()
    )
    if selection.get("index_base", 0) != 0:
        raise ValueError("Expected zero-based block indices")
    return selection


def write_fixed_selection(root, removed=(4, 7)):
    removed = list(removed)
    if (
        not removed
        or any(type(i) is not int for i in removed)
        or removed != sorted(set(removed))
        or any(i <= 0 or i >= 28 for i in removed)
    ):
        raise ValueError("Use sorted, unique block indices from 1 through 27")

    path = Path(root) / "profile/selection.json"

    if path.exists():
        selection = load_fixed_selection(root)
        if selection["removed"] != removed:
            raise ValueError(
                "Selection is frozen for this run; create a new run"
            )
        return selection

    selection = {
        "removed": removed,
        "index_base": 0,
        "scoring": "explicitly configured fixed selection",
    }
    write_json(path, selection)
    return selection


def run_method(split, method, name, *, alpha=None, builder=None, settings=None):
    """Use the existing sampler/artifact runner with the frozen selection."""
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    if Path(name).name != name or name in {".", "..", ""}:
        raise ValueError("name must be a single folder name")
    if (name == "original") != (method == "original" and builder is None):
        raise ValueError("Reserve original for the unmodified baseline")
    if alpha is None:
        alpha = "inverse_k" if method == "damped" else 1.0
    elif alpha != "inverse_k":
        alpha = float(alpha)
    if builder is None and method != "damped" and alpha != 1.0:
        raise ValueError("Only damped mode scales the block residual; use alpha=1.0 for other built-ins")
    root, protocol, config = load_run(RUN_ARGS)
    removed, transform = (), None
    metadata = dict(method=method)
    if builder is not None:
        settings = json.loads(json.dumps(settings or {}))
        source = inspect.getsource(builder)
        metadata.update(settings=settings, source=source,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest())
        transform = lambda model: builder(model, settings)
        mode = "original"
    elif method in {"removal", "loop", "damped", "xsa_reuse"}:
        selection = load_fixed_selection(root)
        removed = selection["removed"]
        metadata["selection"] = selection
        mode = method
    elif method == "original":
        mode = method
    else:
        raise ValueError("Unknown method; pass builder= for a custom method")
    if builder is None:
        metadata["damping_policy"] = (("inverse_k" if alpha == "inverse_k" else "fixed")
                                      if method == "damped" else "none")
        metadata["replacement_residual_scale"] = None if alpha == "inverse_k" else alpha
    return run_variant(config, protocol["splits"][split], root / split / name,
                       mode=mode, removed=removed, alpha=alpha, model_transform=transform,
                       method_metadata=metadata, keep_checkpoint=KEEP_CHECKPOINTS)


# Include the new executed definitions in preparation/resume/evaluation fingerprints.
for _name in ["xsa_filter_heads", "xsa_attention_scope", "load_fixed_selection",
              "write_fixed_selection", "run_method"]:
    if _name not in SHARED_DEFINITION_NAMES:
        SHARED_DEFINITION_NAMES.append(_name)


# %% Focused CPU checks (also included in the notebook's normal test runner).
def test_xsa_projection():
    with torch.random.fork_rng():
        torch.manual_seed(184)
        o, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
        z = xsa_filter_heads(o, v)
        torch.testing.assert_close((z * F.normalize(v, dim=-1)).sum(-1),
                                   torch.zeros(2, 3, 5), atol=1e-6, rtol=0)
        # Zero/epsilon-clamped vectors must remain finite; zero V leaves O alone.
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for magnitude in [0.0, 1e-30, 1e-10, 1e-7]:
                small = torch.full_like(v, magnitude).to(dtype)
                result = xsa_filter_heads(o.to(dtype), small)
                assert torch.isfinite(result).all()
                if magnitude == 0:
                    torch.testing.assert_close(result, o.to(dtype), atol=0, rtol=0)


def test_xsa_attention_paths():
    from timm.models.vision_transformer import Attention
    with torch.random.fork_rng():
        torch.manual_seed(185)
        attn = Attention(24, num_heads=3, qkv_bias=True).eval().requires_grad_(False)
        x = torch.randn(2, 5, 24)
        # Both timm attention kernels stay in use; no change when XSA is off.
        paths = [False, True] if hasattr(attn, "fused_attn") else [None]
        for fused in paths:
            if fused is not None:
                attn.fused_attn = fused
            expected_off = attn(x)
            with xsa_attention_scope(attn, enabled=False):
                actual_off = attn(x)
            torch.testing.assert_close(actual_off, expected_off, atol=0, rtol=0)
            with xsa_attention_scope(attn):
                actual_on = attn(x)
            qkv = attn.qkv(x).reshape(2, 5, 3, 3, 8).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = attn.q_norm(q) if hasattr(attn, "q_norm") else q
            k = attn.k_norm(k) if hasattr(attn, "k_norm") else k
            o = F.scaled_dot_product_attention(q, k, v) if fused else (
                ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1) @ v)
            expected_on = xsa_filter_heads(o, v).transpose(1, 2).reshape(2, 5, 24)
            norm = getattr(attn, "norm", nn.Identity())
            expected_on = attn.proj_drop(attn.proj(norm(expected_on)))
            torch.testing.assert_close(actual_on, expected_on, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(attn(x), expected_off, atol=0, rtol=0)
        # Failure during a replacement must not leave XSA active on later calls.
        boundary = getattr(attn, "norm", attn.proj)
        before = (len(attn.qkv._forward_hooks), len(boundary._forward_pre_hooks))
        with pytest.raises(RuntimeError, match="deliberate"):
            with xsa_attention_scope(attn):
                raise RuntimeError("deliberate")
        assert before == (len(attn.qkv._forward_hooks), len(boundary._forward_pre_hooks))
        # Newer timm additionally supports masks, Q/K norms, post-attention norm
        # and an optional attention gate: exercise them without changing forward.
        import inspect
        parameters = inspect.signature(Attention).parameters
        if {"qk_norm", "scale_norm", "gated"} <= parameters.keys():
            normalized = Attention(24, 3, qkv_bias=True, qk_norm=True,
                                   scale_norm=True, gated=True, norm_layer=nn.LayerNorm).eval()
            normalized.requires_grad_(False)
            mask = torch.ones(5, 5, dtype=torch.bool).tril()
            for fused in [False, True]:
                normalized.fused_attn = fused
                off = normalized(x, attn_mask=mask)
                with xsa_attention_scope(normalized, enabled=False):
                    torch.testing.assert_close(normalized(x, attn_mask=mask), off, atol=0, rtol=0)
                with xsa_attention_scope(normalized):
                    actual = normalized(x, attn_mask=mask)
                q, k, v = normalized.qkv(x).reshape(2, 5, 3, 3, 8).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = normalized.q_norm(q), normalized.k_norm(k)
                if fused:
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    logits = ((q * normalized.scale) @ k.transpose(-2, -1)).masked_fill(~mask, -float("inf"))
                    o = logits.softmax(-1) @ v
                heads = xsa_filter_heads(o, v).transpose(1, 2).reshape(2, 5, 24)
                expected = normalized.norm(heads) * normalized.gate(x).sigmoid()
                expected = normalized.proj_drop(normalized.proj(expected))
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_xsa_schedule_and_registration(monkeypatch):
    from timm.models.vision_transformer import Attention

    class ProbeBlock(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.attn = Attention(12, num_heads=3, qkv_bias=True)
            self.inputs = []

        def forward(self, x, c):
            self.inputs.append((x.detach().clone(), c))
            return x + self.attn(x) + c[:, None, :]

    with torch.random.fork_rng():
        torch.manual_seed(186)
        blocks = [ProbeBlock(i) for i in range(28)]
        schedule = ScheduledBlocks(blocks, [4, 7], "xsa_reuse", alpha=1.0)
        schedule.eval().requires_grad_(False)
        expected_sources = [0, 1, 2, 3, 3, 5, 6, 6, *range(8, 28)]
        metadata = schedule.metadata()
        assert [s["source"] for s in metadata["execution_schedule"]] == expected_sources
        assert [s["slot"] for s in metadata["execution_schedule"] if s["xsa"]] == [4, 7]
        assert metadata["index_base"] == 0 and metadata["retained"] == [i for i in range(28) if i not in [4, 7]]
        assert schedule.retained["3"] is blocks[3] and schedule.retained["6"] is blocks[6]
        retained_ids = {id(p) for b in blocks if b.index not in [4, 7] for p in b.parameters()}
        discarded_ids = {id(p) for b in [blocks[4], blocks[7]] for p in b.parameters()}
        registered_ids = {id(p) for p in schedule.parameters()}
        assert registered_ids == retained_ids and not registered_ids & discarded_ids
        assert len(schedule.retained) == 26 and len(schedule) == 28
        assert all(not p.requires_grad for p in schedule.parameters())
        assert not any(key.split(".")[1] in {"4", "7"} for key in schedule.state_dict())

        events, qkv_inputs, ordinary_outputs = [], {3: [], 6: []}, {}
        original_filter = xsa_filter_heads

        def record_filter(o, v, eps=1e-8):
            events.append(active_slot[0])
            return original_filter(o, v, eps)

        monkeypatch.setitem(globals(), "xsa_filter_heads", record_filter)
        handles = []
        for index in [3, 6]:
            handles.append(blocks[index].attn.qkv.register_forward_pre_hook(
                lambda module, args, index=index: qkv_inputs[index].append(args[0].detach().clone())))
        x, c = torch.randn(1, 5, 12) * 0.01, torch.randn(1, 12) * 0.01
        active_slot = [None]
        try:
            for slot, call in enumerate(schedule):
                active_slot[0] = slot
                prior = x.clone()
                x = call(x, c)
                if slot in [3, 6]:
                    ordinary_outputs[slot] = x.clone()
                # Each additional call receives the immediately preceding output.
                if slot in [4, 7]:
                    source = 3 if slot == 4 else 6
                    torch.testing.assert_close(blocks[source].inputs[-1][0], prior)
                    assert blocks[source].inputs[-1][1] is c
            assert events == [4, 7]
            assert torch.isfinite(x).all()
            for index in [3, 6]:
                assert len(qkv_inputs[index]) == 2
                assert not torch.equal(*qkv_inputs[index])
                # The ordinary retained call remains the exact original block path.
                input_x, input_c = blocks[index].inputs[0]
                ordinary = blocks[index](input_x, input_c)
                torch.testing.assert_close(ordinary, ordinary_outputs[index], atol=0, rtol=0)
        finally:
            for handle in handles:
                handle.remove()


def test_inverse_k_damping():
    class AddBlock(nn.Module):
        def forward(self, x, c):
            return x + 1

    def evaluate(schedule):
        x = torch.zeros(1)
        for call in schedule:
            x = call(x, None)
        return x

    schedule = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "damped")
    entries = schedule.metadata()["execution_schedule"]
    assert [entry["source"] for entry in entries] == [0, 0, 0, 3, 3, 5]
    assert [entry["iteration"] for entry in entries] == [1, 2, 3, 1, 2, 1]
    assert [entry["scale"] for entry in entries] == [1.0, 0.5, 1 / 3, 1.0, 0.5, 1.0]
    assert schedule.metadata()["damping_policy"] == "inverse_k"
    assert schedule.metadata()["replacement_residual_scale"] is None
    torch.testing.assert_close(evaluate(schedule), torch.tensor([13 / 3]))
    # Calling a forward again starts from the same precomputed k values.
    torch.testing.assert_close(evaluate(schedule), evaluate(schedule), atol=0, rtol=0)
    fixed = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "damped", alpha=0.25)
    assert fixed.metadata()["damping_policy"] == "fixed"
    torch.testing.assert_close(evaluate(fixed), torch.tensor([3.75]))
    ordinary = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "loop")
    torch.testing.assert_close(evaluate(ordinary), torch.tensor([6.0]))


def test_xsa_checkpoint_and_fixed_selection(tmp_path, monkeypatch):
    import sys
    import types
    from timm.models.vision_transformer import Attention

    class SmallModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([nn.ModuleDict({"attn": Attention(12, 3)}) for _ in range(28)])

    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(DiT_XL_2=SmallModel))
    model = SmallModel()
    model.blocks = ScheduledBlocks(model.blocks, [4, 7], "xsa_reuse", 1.0, xsa_eps=1e-7)
    path = tmp_path / "compact.pt"
    stats = checkpoint_stats(model, path, keep=True)
    loaded = load_compact_checkpoint(path)
    assert loaded.blocks.metadata() == model.blocks.metadata()
    assert stats["unique_blocks"] == 26 and stats["block_evaluations_per_forward"] == 28
    assert all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    selection = write_fixed_selection(tmp_path)
    assert selection["removed"] == [4, 7] and selection["index_base"] == 0
    assert load_fixed_selection(tmp_path) == selection
    # Verify the public entry point uses shared config/pairs and fixed indices.
    config, pairs = {"device": "cpu"}, [[207, 10016]]
    protocol = {"splits": {"validation": pairs}}
    monkeypatch.setitem(globals(), "RUN_ARGS", object())
    monkeypatch.setitem(globals(), "KEEP_CHECKPOINTS", False)
    monkeypatch.setitem(globals(), "load_run", lambda args: (tmp_path, protocol, config))
    calls = []

    def capture_run(config_arg, pairs_arg, output, **kwargs):
        calls.append((config_arg, pairs_arg, output, kwargs))

    monkeypatch.setitem(globals(), "run_variant", capture_run)
    for mode in ["removal", "loop", "xsa_reuse"]:
        run_method("validation", mode, mode, alpha=1.0)
        actual_config, actual_pairs, output, kwargs = calls[-1]
        assert actual_config is config and actual_pairs is pairs
        assert output == tmp_path / "validation" / mode
        assert kwargs["mode"] == mode and kwargs["removed"] == [4, 7]
        assert kwargs["alpha"] == 1.0 and kwargs["model_transform"] is None
    with pytest.raises(ValueError, match="alpha=1.0"):
        run_method("validation", "xsa_reuse", "bad_alpha", alpha="inverse_k")
    selection["removed"] = [3, 6]
    write_json(tmp_path / "profile/selection.json", selection)
    with pytest.raises(ValueError, match="frozen"):
        write_fixed_selection(tmp_path)


# %% Combined notebook test runner. No checkpoint or image downloads.
from tempfile import TemporaryDirectory


def run_notebook_tests():
    tests = [
        ("protocol splits/hash", lambda path, patch: test_protocol_has_balanced_disjoint_splits_and_stable_hash(path)),
        ("preprocessing/noise", lambda path, patch: test_preprocessing_and_initial_noise_are_fixed()),
        ("runner/evaluator artifacts", lambda path, patch: test_shared_runner_and_evaluator_reject_incomplete_or_mismatched_runs(path, patch)),
        ("custom builder", lambda path, patch: test_custom_builder_keeps_shared_preparation_and_freezes_model(patch)),
        ("inverse-k damping/reset/fixed compatibility", lambda path, patch: test_inverse_k_damping()),
        ("XSA projection/zero values", lambda path, patch: test_xsa_projection()),
        ("XSA attention on/off and hook cleanup", lambda path, patch: test_xsa_attention_paths()),
        ("XSA schedule/shared frozen weights", lambda path, patch: test_xsa_schedule_and_registration(patch)),
        ("XSA checkpoint/fixed selection", lambda path, patch: test_xsa_checkpoint_and_fixed_selection(path, patch)),
    ]
    for pairs in [[], [(1, 3), (2, 3)], [(1000, 3)], [(1.5, 3)], [(True, 3)]]:
        tests.append((f"invalid input {pairs}", lambda path, patch, pairs=pairs:
                      test_input_validation_rejects_ambiguous_samples(pairs)))
    for name, test in tests:
        with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
            test(Path(directory), patch)
        print("PASS:", name)
    print(f"All {len(tests)} notebook test cases passed.")


# Definitions only. Continue with tests, configuration and block selection below.


# Notebook cell 26
# RUN TESTS — CPU only, no generated .py files.
from tempfile import TemporaryDirectory

run_notebook_tests()


# Notebook cell 28
# CONFIGURE — does not generate images.
import json
from datetime import datetime, timezone
import torch
from IPython.display import display, Image

DEVICE = "auto"
RESUME_RUN = None  # Leave None for the new Imagenette protocol; old eight-class runs differ.
CALIBRATE_BLOCKS = False
FIXED_REMOVED = [4, 7]  # Zero-based, shared by removal / loop / damped / XSA.
REMOVE_COUNT = 2  # Used only when CALIBRATE_BLOCKS=True.

protocol = load_protocol(PROTOCOL)
REPO = WORKSPACE / "third_party/DiT"
if not (REPO / "models.py").is_file():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/DiT.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", protocol["settings"]["dit_commit"]], check=True)
MEMBER = "team"  # Use your name for independent runs.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RUN_ROOT = (Path(RESUME_RUN).expanduser().resolve() if RESUME_RUN
            else OUTPUT_BASE / f"{MEMBER}_{RUN_ID}")
CHECKPOINT = WORKSPACE / "checkpoints/DiT-XL-2-256x256.pt"
ALPHA = "inverse_k"  # Reused calls use 1/2, 1/3, ...; original call counts as k=1.
KEEP_CHECKPOINTS = False  # True retains several GB per variant.
NAMES = ["original", "member1_removal", "member2_loop", "member3_damped", "member4_xsa_reuse"]

if RESUME_RUN:
    saved_config = json.loads((RUN_ROOT / "config.json").read_text())
    DEVICE = saved_config["device"]
    CHECKPOINT = Path(saved_config["checkpoint"])
device = select_device(DEVICE)
RUN_ARGS = SimpleNamespace(output=RUN_ROOT, repo=REPO, protocol=PROTOCOL,
                           device=str(device), checkpoint=CHECKPOINT)

print("Device:", device, "| PyTorch:", torch.__version__)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(device))
print("Run directory:", RUN_ROOT)
print("ImageNet class ids:", list(CLASS_LABELS))
print("Images per split:", {key: len(value) for key, value in protocol["splits"].items()})


# Notebook cell 30
# RUN PREPARE — skip downloading/preparing again when resuming.
if RESUME_RUN:
    load_run(RUN_ARGS)  # Verify the saved code, environment and inputs.
    print("Resumed:", RUN_ROOT)
else:
    prepare(RUN_ARGS)


# Notebook cell 32
# SELECT BLOCKS — fixed indices avoid calibration inference by default.
root, protocol, config = load_run(RUN_ARGS)
selection_path = root / "profile/selection.json"
if selection_path.exists():
    selection = load_fixed_selection(root)
elif CALIBRATE_BLOCKS:
    profile = calibrate(config, protocol["splits"]["calibration"], root / "profile")
    summarize_profile(profile, root / "profile", REMOVE_COUNT)
    selection = load_fixed_selection(root)
else:
    selection = write_fixed_selection(root, FIXED_REMOVED)
print("Selection:", selection)
heatmap = root / "profile/feature_change_heatmap.png"
if heatmap.exists():
    display(Image(filename=str(heatmap)))


# Notebook cell 35
# RUN VALIDATION — original baseline
run_method("validation", "original", "original")
display(Image(filename=str(RUN_ROOT / "validation/original/grid.png")))


# Notebook cell 37
# RUN VALIDATION — member 1: removal
run_method("validation", "removal", "member1_removal")
display(Image(filename=str(RUN_ROOT / "validation/member1_removal/grid.png")))


# Notebook cell 39
# RUN VALIDATION — member 2: looping
run_method("validation", "loop", "member2_loop")
display(Image(filename=str(RUN_ROOT / "validation/member2_loop/grid.png")))


# Notebook cell 41
# RUN VALIDATION — member 3: damped looping
run_method("validation", "damped", "member3_damped", alpha=ALPHA)
display(Image(filename=str(RUN_ROOT / "validation/member3_damped/grid.png")))


# Notebook cell 43
# RUN VALIDATION — member 4: XSA reuse
run_method("validation", "xsa_reuse", "member4_xsa_reuse", alpha=1.0)
display(Image(filename=str(RUN_ROOT / "validation/member4_xsa_reuse/grid.png")))


# Notebook cell 45
# 8B - FIDs & TPIPS
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


# Imagenette reference preparation uses torchvision, which is already installed.
# Download: full-resolution Imagenette v2 (~1.45 GiB archive), not all ImageNet.
def prepare_imagenette_reference(source_root, prepared_root, pairs, *,
                                 images_per_class=500, image_size=256, seed=0):
    """Export a deterministic, class-balanced real-image reference and reuse it.

    Uses the dataset's TRAIN split as real FID reference data; DiT is not trained.
    Generated labels come from the saved run manifest, never assumed local ids.
    Source files/archives live outside the prepared image folder so Clean-FID
    cannot accidentally include raw images or grids in the reference.
    """
    from collections import Counter
    import random
    from PIL import Image as PILImage, ImageOps

    pairs = validate_pairs(pairs)
    counts = Counter(label for label, seed_value in pairs)
    class_map = {label: (synset, name) for label, synset, name in IMAGENETTE_CLASSES}
    labels = sorted(counts)
    if set(labels) - class_map.keys():
        raise ValueError("Run contains classes outside Imagenette; prepare a new run using CLASS_LABELS")
    if len(set(counts.values())) != 1:
        raise ValueError("Use balanced generated class counts to match the real reference")
    if type(images_per_class) is not int or images_per_class < 2:
        raise ValueError("images_per_class must be an integer >= 2")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("Reference seed must be a nonnegative integer")
    settings = dict(
        dataset="Imagenette v2", source_split="train", source_size="full",
        source_url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        archive_md5="fe2fc210e6bb7c5664d602c3cd71e612",
        labels=labels, images_per_class=images_per_class, image_size=image_size,
        seed=seed, preprocessing="RGB; aspect-preserving bicubic resize and center crop; PNG",
    )
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output = Path(prepared_root) / f"imagenette2_train_{images_per_class}perclass_{digest}"
    manifest_path = output / "reference_manifest.json"
    expected = {f"class_{label:04d}_{i:04d}.png"
                for label in labels for i in range(images_per_class)}
    description = (
        f"Imagenette v2 full-resolution train split; ImageNet labels={labels}; "
        f"{images_per_class} real images per class, selected with seed={seed}; "
        f"RGB; aspect-preserving bicubic resize and center crop to {image_size}x{image_size}; PNG."
    )
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        actual = {p.name for p in output.glob("*.png") if p.is_file()}
        if (saved.get("settings") != settings or actual != expected
                or any((output / filename).stat().st_size == 0 for filename in expected)):
            raise ValueError(f"Cached Imagenette reference is incomplete or inconsistent: {output}")
        print("Reusing real FID reference:", output, flush=True)
        return output, description

    from torchvision.datasets import Imagenette
    source_root = Path(source_root)
    print("Preparing Imagenette v2 full-resolution reference (one-time download).", flush=True)
    Imagenette(root=str(source_root), split="train", size="full", download=True)
    selections = {}
    extensions = {".jpeg", ".jpg", ".png"}
    for label in labels:
        synset, name = class_map[label]
        folder = source_root / "imagenette2" / "train" / synset
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        if len(files) < images_per_class:
            raise ValueError(f"{name} ({label}) has {len(files)} real images; requested {images_per_class}")
        selections[label] = sorted(random.Random(seed + label).sample(files, images_per_class))
    output.mkdir(parents=True, exist_ok=True)
    if {p.name for p in output.glob("*.png")} - expected:
        raise ValueError(f"Unexpected images in reference folder: {output}")
    inventory = []
    for label in labels:
        for i, source in enumerate(selections[label]):
            filename = f"class_{label:04d}_{i:04d}.png"
            with PILImage.open(source) as image:
                image = ImageOps.fit(image.convert("RGB"), (image_size, image_size),
                                     method=PILImage.Resampling.BICUBIC, centering=(0.5, 0.5))
                image.save(output / filename)
            inventory.append(dict(file=filename, label=label,
                                  source=str(source.relative_to(source_root))))
        print(f"Real reference class {label}: {images_per_class} images", flush=True)
    write_json(manifest_path, dict(settings=settings, images=len(inventory), inventory=inventory))
    print("Saved real FID reference:", output, flush=True)
    return output, description


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


# Notebook cell 46
# METRIC SETTINGS — Imagenette FID enabled, TPIPS disabled.
USE_IMAGENETTE_REFERENCE = True
IMAGENETTE_SOURCE_ROOT = WORKSPACE / "fid_references" / "source"
IMAGENETTE_PREPARED_ROOT = WORKSPACE / "fid_references" / "prepared"
FID_REFERENCE_DIR = None  # Filled automatically from the evaluated run's classes.
FID_REFERENCE_DESCRIPTION = None
TPIPS_FACTORS = ()  # No TPIPS package/weights are loaded with this default.
FID_DEVICE = "cpu"
TPIPS_DEVICE = str(select_device("auto"))


# Notebook cell 47
# RUN EVALUATION — works for validation now and the larger held-out test later.
import pandas as pd


def evaluate_saved_split(split):
    global FID_REFERENCE_DIR, FID_REFERENCE_DESCRIPTION
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    split_root = RUN_ROOT / split
    results_path = split_root / "results.csv"
    paired_path = split_root / "paired_metrics.csv"
    grid_path = split_root / "comparison_grid.png"
    if (not results_path.exists() or not paired_path.exists() or not grid_path.exists()
            or not set(NAMES).issubset(set(pd.read_csv(results_path)["variant"]))):
        evaluate_split(split, NAMES)

    # Prepare real references once; validation and test reuse the same image set.
    if USE_IMAGENETTE_REFERENCE:
        baseline = json.loads((split_root / "original/manifest.json").read_text())
        FID_REFERENCE_DIR, FID_REFERENCE_DESCRIPTION = prepare_imagenette_reference(
            IMAGENETTE_SOURCE_ROOT, IMAGENETTE_PREPARED_ROOT, baseline["pairs"],
            images_per_class=FID_REAL_IMAGES_PER_CLASS,
            image_size=baseline["config"]["image_size"], seed=0,
        )

    # Install optional packages after base evaluation checks the run environment.
    extra_packages = []
    if FID_REFERENCE_DIR is not None and _extra_version("clean-fid") == "not-installed":
        extra_packages.append("clean-fid")
    if TPIPS_FACTORS and _extra_version("tpips") == "not-installed":
        extra_packages.append("tpips")
    if extra_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_packages,
                        "scipy<1.18"], check=True)

    result = evaluate_fid_tpips(
        split_root, NAMES,
        fid_reference_dir=FID_REFERENCE_DIR,
        fid_reference_description=FID_REFERENCE_DESCRIPTION,
        tpips_factors=TPIPS_FACTORS,
        fid_device=FID_DEVICE,
        tpips_device=TPIPS_DEVICE,
    )
    display(result)
    display(Image(filename=str(grid_path)))
    print("Summary:", split_root / "results_extended.csv")
    print("Per-image metrics:", split_root / "paired_metrics_extended.csv")
    print("Evaluation settings:", split_root / "evaluation_extended.json")
    return result


results = evaluate_saved_split("validation")


# Notebook cell 48
print("Saved run:", RUN_ROOT)

# Notebook cell 52
# OPTIONAL CUSTOM — disabled by default.
RUN_CUSTOM = False
CUSTOM_NAME = "custom_example"
if RUN_CUSTOM:
    run_method("validation", "custom", CUSTOM_NAME,
               builder=build_model, settings=CUSTOM_SETTINGS)
    evaluate_split("validation", [*NAMES, CUSTOM_NAME])
    display(pd.read_csv(RUN_ROOT / "validation/results.csv"))
else:
    print("Skipped custom method; set RUN_CUSTOM=True to run it.")


# Notebook cell 54
# RUN HELD-OUT TEST — enable only after validation choices are frozen.
# Default: 500 images/class x 10 classes = 5,000 images for EACH of five methods.
# Set FINAL_IMAGES_PER_CLASS in section 2.9 BEFORE preparing a new run.
RUN_HELD_OUT_TEST = False
if RUN_HELD_OUT_TEST:
    run_method("test", "original", "original")
    run_method("test", "removal", "member1_removal")
    run_method("test", "loop", "member2_loop")
    run_method("test", "damped", "member3_damped", alpha=ALPHA)
    run_method("test", "xsa_reuse", "member4_xsa_reuse", alpha=1.0)
    test_results = evaluate_saved_split("test")
else:
    print("Skipped held-out testing; freeze validation choices before enabling.")
    print("Final images per method:", len(PROTOCOL["splits"]["test"]))


# Notebook cell 56
# EXPORT — optional; no inference.
EXPORT_RUN = False
if EXPORT_RUN:
    import shutil
    archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT.parent,
                                  base_dir=RUN_ROOT.name)
    print("Archive:", archive)
    if "google.colab" in sys.modules:
        from google.colab import files
        files.download(archive)
else:
    print("Set EXPORT_RUN=True to archive:", RUN_ROOT)


# Notebook cell 58
# REPORT HELPERS — standalone, read-only access to completed experiment artifacts.
from pathlib import Path
import hashlib
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage

REPORT_COLORS = ("#334155", "#D97706", "#059669", "#7C3AED", "#DC2626", "#0284C7")
REPORT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}


def _report_method_label(name, manifest):
    schedule = manifest.get("schedule", {})
    mode = schedule.get("mode", manifest.get("method_metadata", {}).get("method", name))
    if name == "original":
        return "Original DiT"
    if mode == "damped":
        alpha = schedule.get("alpha")
        policy = schedule.get("damping_policy", "inverse_k" if alpha == "inverse_k" else "fixed")
        return "Damped (1/k)" if policy == "inverse_k" else f"Damped (fixed α={alpha})"
    return {"removal": "Removal", "loop": "Looping", "xsa_reuse": "XSA reuse"}.get(mode, name)


def load_report_data(run_root, split="validation", names=None):
    """Read CSV/JSON only; old generation code/environments remain reportable.

    Every method must have complete, matching manifests/sample rows. Old fixed
    damping is labelled from its saved manifest, never from today's ALPHA value.
    Extended FID is used only with matching recorded generation provenance.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    run_root = Path(run_root).expanduser().resolve()
    root = run_root / split
    summary = pd.read_csv(root / "results.csv")
    paired = pd.read_csv(root / "paired_metrics.csv")
    if summary["variant"].duplicated().any():
        raise ValueError("Duplicate variants in results.csv")
    names = list(summary["variant"] if names is None else names)
    if not names or len(set(names)) != len(names) or "original" not in names:
        raise ValueError("Use unique variant names including original")
    if any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}
           for name in names):
        raise ValueError("Variant names must be single folder names")
    summary = summary.set_index("variant").loc[names].reset_index()
    baseline = json.loads((root / "original/manifest.json").read_text())
    pairs = baseline["pairs"]
    expected = pd.DataFrame([(i, *pair) for i, pair in enumerate(pairs)],
                            columns=["index", "label", "seed"])
    if len(expected) < 2 or expected["seed"].duplicated().any():
        raise ValueError("Expected at least two distinct generated samples")
    manifests, frames = {}, []
    for name in names:
        manifest = json.loads((root / name / "manifest.json").read_text())
        for key in ["config", "pairs", "shared_code_sha256", "runtime"]:
            if key not in manifest or manifest[key] != baseline.get(key):
                raise ValueError(f"Mismatched generation {key}: {name}")
        samples = pd.read_csv(root / name / "samples.csv")
        rows = paired.loc[paired["variant"] == name].reset_index(drop=True)
        if (not samples[["index", "label", "seed"]].equals(expected)
                or not rows[["index", "label", "seed"]].equals(expected)
                or int(summary.loc[summary["variant"] == name, "images"].iloc[0]) != len(expected)):
            raise ValueError(f"Incomplete or inconsistent saved samples: {name}")
        if (not np.isfinite(rows[["paired_lpips", "latency_s"]].to_numpy()).all()
                or (rows["latency_s"] <= 0).any() or (rows["paired_lpips"] < 0).any()):
            raise ValueError(f"Invalid saved measurements: {name}")
        if not np.allclose(samples["latency_s"], rows["latency_s"]):
            raise ValueError(f"Paired latency rows differ from samples.csv: {name}")
        result_row = summary.loc[summary["variant"] == name].iloc[0]
        if (not np.isclose(rows["paired_lpips"].mean(), result_row["lpips_mean"])
                or not np.isclose(rows["latency_s"].mean(), result_row["latency_mean_s"])):
            raise ValueError(f"Summary means differ from paired rows: {name}")
        manifests[name] = manifest
        frames.append(rows)
    paired = pd.concat(frames, ignore_index=True)
    evaluation = {}
    extended_path = root / "results_extended.csv"
    evaluation_path = root / "evaluation_extended.json"
    if extended_path.exists():
        extended = pd.read_csv(extended_path)
        if "fid" in extended:
            if not evaluation_path.exists():
                raise ValueError("FID CSV is missing its evaluation_extended.json provenance")
            evaluation = json.loads(evaluation_path.read_text())
            if (evaluation.get("generation_code_sha256") != baseline["shared_code_sha256"]
                    or evaluation.get("generation_runtime") != baseline["runtime"]
                    or evaluation.get("images_per_variant") != len(expected)
                    or not set(names).issubset(evaluation.get("variants", []))
                    or not evaluation.get("fid")):
                raise ValueError("FID evaluation provenance does not match the saved generation")
            if extended["variant"].duplicated().any():
                raise ValueError("Duplicate variants in results_extended.csv")
            extended = extended.set_index("variant").loc[names]
            if not extended["images"].astype(int).eq(len(expected)).all():
                raise ValueError("FID summary sample counts do not match")
            fid_values = pd.to_numeric(extended["fid"], errors="raise")
            if not np.isfinite(fid_values).all() or (fid_values < 0).any():
                raise ValueError("Invalid saved FID scores")
            summary["fid"] = summary["variant"].map(fid_values)
    summary["method"] = [_report_method_label(name, manifests[name]) for name in names]
    summary["damping_policy"] = [manifests[name].get("schedule", {}).get(
        "damping_policy", "fixed" if manifests[name].get("schedule", {}).get("mode") == "damped" else "none")
        for name in names]
    baseline_row = summary.loc[summary["variant"] == "original"].iloc[0]
    summary["speedup_vs_original"] = baseline_row["latency_mean_s"] / summary["latency_mean_s"]
    summary["parameter_reduction_pct"] = 100 * (1 - summary["unique_parameters"] / baseline_row["unique_parameters"])
    summary["latency_sample_sd_s"] = summary["variant"].map(paired.groupby("variant")["latency_s"].std(ddof=1))
    per_class = paired.groupby(["variant", "label"], sort=False)["paired_lpips"].agg(
        samples="size", lpips_mean="mean", lpips_std="std").reset_index()
    output = run_root / "report" / split
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "report_summary.csv", index=False)
    per_class.to_csv(output / "per_class_lpips.csv", index=False)
    inputs = [root / "results.csv", root / "paired_metrics.csv", root / "original/manifest.json"]
    inputs += [p for p in [extended_path, evaluation_path] if p.exists()]
    provenance = dict(
        run=str(run_root), split=split, images_per_variant=len(expected),
        labels=sorted(expected["label"].unique().tolist()),
        class_counts={str(k): int(v) for k, v in expected["label"].value_counts().sort_index().items()},
        generation_code_sha256=baseline["shared_code_sha256"], runtime=baseline["runtime"],
        method_labels=dict(zip(names, summary["method"])),
        saved_schedules={name: manifests[name].get("schedule", {}) for name in names},
        fid=evaluation.get("fid"), uncertainty="Error bars show sample SD, not confidence intervals; FID has no error bar",
        input_sha256={str(p.relative_to(run_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    )
    (output / "report_metadata.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return dict(root=root, run_root=run_root, split=split, summary=summary, paired=paired,
                per_class=per_class, baseline=baseline, manifests=manifests,
                evaluation=evaluation, output=output)


def _report_context(report):
    manifest = report["baseline"]
    config = manifest["config"]
    n = len(manifest["pairs"])
    classes = len({label for label, seed in manifest["pairs"]})
    device = manifest.get("runtime", {}).get("device", config.get("device", "unknown"))
    return (f"{report['run_root'].name} | {report['split']} | N={n}/method, {classes} classes | "
            f"{config.get('sampling_steps', '?')} steps, CFG={config.get('cfg_scale', '?')}, "
            f"{config.get('dtype', 'float32')} | {device}")


def _save_report_figure(report, figure, stem, note):
    width = max(70, int(figure.get_figwidth() * 12))
    caption = textwrap.fill(_report_context(report), width=width) + "\n" + textwrap.fill(note, width=width)
    figure.text(0.015, 0.012, caption, fontsize=7, va="bottom", color="#475569")
    figure.tight_layout(rect=(0, 0.13, 1, 0.96))
    paths = {}
    for extension in ["png", "pdf"]:
        path = report["output"] / f"{stem}.{extension}"
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths[extension] = path
    plt.close(figure)
    return paths


def _report_bars(ax, report, values, ylabel, title, errors=None, decimals=2):
    values = np.asarray(values, dtype=float)
    labels = report["summary"]["method"].str.replace(" (", "\n(", regex=False)
    colors = [REPORT_COLORS[i % len(REPORT_COLORS)] for i in range(len(values))]
    finite = np.isfinite(values)
    if not finite.any():
        ax.set_title(title)
        ax.text(0.5, 0.5, "Metric not available in this saved run", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    heights = np.where(finite, values, 0)
    uncertainty = np.zeros_like(heights) if errors is None else np.nan_to_num(np.asarray(errors, float), nan=0)
    bars = ax.bar(np.arange(len(values)), heights, color=colors,
                  yerr=uncertainty if errors is not None else None,
                  capsize=3, error_kw={"elinewidth": 1, "ecolor": "#334155"}, width=0.68)
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


def plot_report_quality(report):
    summary = report["summary"]
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        fig.suptitle("Output preservation and real-reference FID", fontsize=13)
        _report_bars(axes[0], report, summary["lpips_mean"], "Paired LPIPS ↓", "Deviation from original DiT",
                     errors=summary["lpips_std"], decimals=3)
        n = len(report["baseline"]["pairs"])
        fid_title = f"Exploratory pooled FID-{n}" if n < 2049 else f"Pooled FID-{n}"
        _report_bars(axes[1], report, summary.get("fid", pd.Series(np.nan, index=summary.index)),
                     "FID vs. real reference ↓", fid_title, decimals=2)
        reference_count = report["evaluation"].get("fid", {}).get("reference_images", "not recorded")
        note = (f"LPIPS bars: mean ± sample SD. FID is pooled across classes against {reference_count} real images; "
                "no FID confidence intervals are estimated. Small-sample FID rankings are exploratory.")
        return _save_report_figure(report, fig, "quality_metrics", note)


def plot_report_efficiency(report):
    summary = report["summary"]
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
        fig.suptitle("Measured inference cost and model storage", fontsize=13)
        _report_bars(axes[0, 0], report, summary["latency_mean_s"], "Seconds / image ↓", "Sampling + decode + CPU transfer",
                     errors=summary["latency_sample_sd_s"], decimals=2)
        _report_bars(axes[0, 1], report, summary["unique_parameters"] / 1e6, "Unique parameters (millions) ↓", "DiT parameters; shared VAE excluded", decimals=1)
        _report_bars(axes[1, 0], report, summary["checkpoint_MiB"] / 1024, "Serialized checkpoint (GiB) ↓", "DiT checkpoint size", decimals=2)
        memory_options = [
            ("peak_GPU_allocated_GiB", "Peak CUDA allocated memory (GiB) ↓", "Peak allocated during sampling"),
            ("max_post_sample_MPS_allocated_GiB", "Post-sample MPS allocated memory (GiB) ↓", "MPS allocated snapshot (not peak)"),
        ]
        memory = next(((key, label, title) for key, label, title in memory_options
                       if key in summary and summary[key].notna().any()), None)
        if memory:
            key, label, title = memory
            _report_bars(axes[1, 1], report, summary[key], label, title, decimals=2)
        else:
            _report_bars(axes[1, 1], report, summary["block_evaluations_per_forward"], "Block calls / forward", "Executed blocks", decimals=0)
        return _save_report_figure(report, fig, "efficiency_metrics",
            "Latency bars: mean ± sample SD. Timing excludes model loading, warmup, serialization and metrics. "
            "Unique weights and execution count are different quantities; MPS snapshots are not peak memory.")


def plot_report_per_class(report):
    names = [name for name in report["summary"]["variant"] if name != "original"]
    if not names:
        return None
    labels = sorted({label for label, seed in report["baseline"]["pairs"]})
    table = report["per_class"].pivot(index="variant", columns="label", values="lpips_mean").reindex(index=names, columns=labels)
    counts = report["per_class"].pivot(index="variant", columns="label", values="samples").reindex(index=names, columns=labels)
    method_labels = report["summary"].set_index("variant")["method"]
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.95), 4.8))
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
                            ha="center", va="center", fontsize=8, color="white" if value > 0.65 * vmax else "#1E293B")
        fig.colorbar(heatmap, ax=ax, label="Mean paired LPIPS ↓", shrink=0.85)
        return _save_report_figure(report, fig, "per_class_lpips",
            "Each cell averages matched original/variant pairs within one class. n is the number of pairs; "
            "per-class LPIPS is output preservation, not per-class FID or absolute image quality.")


def plot_report_samples(report, count=6):
    if type(count) is not int or count < 1:
        raise ValueError("Sample grid count must be a positive integer")
    # Prefer different classes rather than accidentally showing six seeds of one class.
    indices, seen = [], set()
    for index, (label, seed) in enumerate(report["baseline"]["pairs"]):
        if label not in seen:
            indices.append(index)
            seen.add(label)
        if len(indices) == count:
            break
    names = list(report["summary"]["variant"])
    method_labels = report["summary"].set_index("variant")["method"]
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(len(names), len(indices), squeeze=False,
                                 figsize=(2.15 * len(indices) + 1, 2.1 * len(names)))
        fig.suptitle("Matched generated samples", fontsize=13)
        for row, name in enumerate(names):
            for column, index in enumerate(indices):
                ax = axes[row, column]
                path = report["root"] / name / f"image_{index:04d}.png"
                with PILImage.open(path) as image:
                    ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    label, seed = report["baseline"]["pairs"][index]
                    ax.set_title(f"Class {label}\nSeed {seed}", fontsize=9)
                if column == 0:
                    ax.set_ylabel(method_labels[name], fontsize=9)
        return _save_report_figure(report, fig, "matched_samples",
            "Columns use identical class/seed pairs across methods. Original DiT outputs are a generated baseline, "
            "not real ground-truth images. Examples are selected by first occurrence of each class, without quality ranking.")


def plot_report_schedules(report):
    names = list(report["summary"]["variant"])
    depth = max(manifest.get("schedule", {}).get("depth", 0) for manifest in report["manifests"].values())
    if not depth:
        return None
    sources = np.full((len(names), depth), np.nan)
    scales = np.full_like(sources, np.nan)
    xsa = np.zeros_like(sources, dtype=bool)
    for row, name in enumerate(names):
        schedule = report["manifests"][name].get("schedule", {})
        for position, entry in enumerate(schedule.get("execution_schedule", [])):
            slot = entry.get("slot", entry["source"] if schedule.get("mode") == "removal" else position)
            sources[row, slot] = entry["source"]
            scales[row, slot] = entry["scale"]
            xsa[row, slot] = entry.get("xsa", False)
    method_labels = report["summary"].set_index("variant")["method"]
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(16, 7.5))
        fig.suptitle("Saved block schedules and residual damping", fontsize=13)
        for ax, matrix, title, upper in [
            (axes[0], sources, "Original source block used at each depth slot (* = XSA)", max(depth - 1, 1)),
            (axes[1], scales, "Residual scale at each executed depth slot", 1.0),
        ]:
            cmap = plt.get_cmap("viridis").with_extremes(bad="#E2E8F0")
            heatmap = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=upper)
            ax.grid(False)
            ax.set_xticks(range(depth), [str(slot) for slot in range(depth)], fontsize=7)
            ax.set_yticks(range(len(names)), [method_labels[name] for name in names], fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Original depth slot (zero-based)")
            for row in range(len(names)):
                for slot in range(depth):
                    value = matrix[row, slot]
                    label = "—" if np.isnan(value) else (f"{int(value)}{'*' if xsa[row, slot] else ''}" if ax is axes[0] else f"{value:.2g}")
                    ax.text(slot, row, label, ha="center", va="center", fontsize=6.5,
                            color="#1E293B" if np.isnan(value) or value > 0.6 * upper else "white")
            fig.colorbar(heatmap, ax=ax, fraction=0.018, pad=0.015)
        return _save_report_figure(report, fig, "block_schedules",
            "Schedules and damping labels are read from saved manifests. Gray cells were not executed. "
            "Historical fixed-alpha runs retain their recorded labels; this plot does not simulate the new 1/k policy.")


# Notebook cell 60
# REPORT CONFIG — no sampling or metric calculation.
from IPython.display import display, Image as ReportImage

REPORT_RUN_ROOT = None  # e.g. Path("team_runs/team_20261117T003651069722Z")
REPORT_SPLIT = "validation"
REPORT_GRID_IMAGES = 6

if REPORT_RUN_ROOT is None:
    current = globals().get("RUN_ROOT")
    report_base = Path(globals().get("OUTPUT_BASE", Path.cwd() / "team_runs"))
    if current is not None and (Path(current) / REPORT_SPLIT / "results.csv").is_file():
        REPORT_RUN_ROOT = Path(current)
    else:
        completed = [p for p in report_base.glob("*")
                     if (p / REPORT_SPLIT / "results.csv").is_file()]
        if not completed:
            raise FileNotFoundError("No completed results.csv found; set REPORT_RUN_ROOT to a saved run")
        REPORT_RUN_ROOT = max(completed, key=lambda p: (p / REPORT_SPLIT / "results.csv").stat().st_mtime_ns)
REPORT = load_report_data(REPORT_RUN_ROOT, REPORT_SPLIT)
print("Reporting saved run:", REPORT["run_root"])
print("Report exports:", REPORT["output"])
print("Saved damping:", REPORT["summary"].loc[REPORT["summary"]["damping_policy"] != "none", "method"].tolist())
columns = ["method", "images", "lpips_mean", "lpips_std", "fid", "latency_mean_s",
           "speedup_vs_original", "parameter_reduction_pct", "unique_parameters",
           "checkpoint_MiB", "block_evaluations_per_forward"]
display(REPORT["summary"][[c for c in columns if c in REPORT["summary"]]].round(4))


# Notebook cell 62
paths = plot_report_quality(REPORT)
if paths is not None:
    display(ReportImage(filename=str(paths["png"])))
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])


# Notebook cell 64
paths = plot_report_efficiency(REPORT)
if paths is not None:
    display(ReportImage(filename=str(paths["png"])))
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])


# Notebook cell 66
paths = plot_report_per_class(REPORT)
if paths is not None:
    display(ReportImage(filename=str(paths["png"])))
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])


# Notebook cell 68
paths = plot_report_samples(REPORT, REPORT_GRID_IMAGES)
if paths is not None:
    display(ReportImage(filename=str(paths["png"])))
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])


# Notebook cell 70
paths = plot_report_schedules(REPORT)
if paths is not None:
    display(ReportImage(filename=str(paths["png"])))
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])

