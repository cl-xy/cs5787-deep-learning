"""Device selection, determinism, RNG, JSON and hashing helpers for CPU, MPS and CUDA."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

DEVICE_TYPES = {"cpu", "mps", "cuda"}
ENVIRONMENT_PACKAGES = ("torch", "torchvision", "timm", "diffusers", "accelerate", "lpips", "numpy")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def write_json(path, value):
    """Write strict JSON (no NaN/Infinity), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device(requested="auto"):
    device = str(requested)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    result = torch.device(device)
    if result.type not in DEVICE_TYPES:
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
    """Make the official sampler's coefficient lookup work on MPS (no float64 there)."""
    gd = importlib.import_module("diffusion.gaussian_diffusion")
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


def configure_dit(repo):
    """Put the official DiT checkout on sys.path, fix determinism flags, return its commit."""
    repo = Path(repo).resolve()
    if not (repo / "models.py").exists():
        raise FileNotFoundError(f"Official DiT checkout missing: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Avoid silently using an unrelated notebook module called 'models'.
    models = importlib.import_module("models")
    if Path(models.__file__).resolve() != repo / "models.py":
        raise RuntimeError("An unrelated 'models' module is loaded. Restart the kernel.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=select_device().type == "mps")
    install_mps_sampler_compatibility()
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def environment(requested="auto"):
    """Record everything that must match for timing comparisons to be valid."""
    device = select_device(requested)
    packages = {name: importlib.metadata.version(name) for name in ENVIRONMENT_PACKAGES}
    return dict(
        python=sys.version, platform=platform.platform(), packages=packages,
        device=str(device), machine=platform.machine(), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        mps_available=torch.backends.mps.is_available(), macos=platform.mac_ver()[0],
        mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    """Release cached accelerator memory between samples and variants."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def paired_rng(device, seed):
    """Restore CPU and selected accelerator RNG, including after exceptions."""
    state = torch.mps.get_rng_state() if device.type == "mps" else None
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
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


def pip_install(packages, *extra_args):
    """Install pip ``packages`` into the running interpreter's environment."""
    subprocess.run([sys.executable, "-m", "pip", "install", *packages, *extra_args], check=True)
