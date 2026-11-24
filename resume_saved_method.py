"""Resume a built-in method using the live notebook's unchanged generation code.

In a notebook cell, after interrupting generation:

    from resume_saved_method import resume_method
    resume_method("test", "member2_loop", namespace=globals())

This helper is separate from the fingerprinted notebook definitions so existing
run provenance remains verifiable. It never edits the saved manifest or config.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.utils import save_image


def _saved_progress(folder, pairs, size, device):
    path = folder / "samples.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        # Python's float parser round-trips the existing latency values.
        frame = pd.read_csv(path, converters={"latency_s": float})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError("samples.csv is unreadable; keep the folder for recovery") from exc
    columns = ["index", "label", "seed"]
    expected = pd.DataFrame([(i, *pair) for i, pair in enumerate(pairs)], columns=columns)
    if (len(frame) > len(pairs)
            or not set(columns + ["latency_s", "device"]).issubset(frame.columns)
            or (len(frame) and not frame[columns].equals(expected.iloc[:len(frame)]))):
        raise ValueError("Saved sample rows must be a matching, contiguous prefix of the protocol")
    latency = frame["latency_s"].to_numpy(dtype=float)
    if not np.isfinite(latency).all() or (latency <= 0).any() or not frame["device"].eq(device).all():
        raise ValueError("Saved timing/device records are invalid")
    # A committed sample needs its timing row AND both valid image files.
    for index in range(len(frame)):
        tensor = torch.load(folder / f"image_{index:04d}.pt", map_location="cpu", weights_only=True)
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (1, 3, size, size)
                or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
                or tensor.min() < -1 or tensor.max() > 1):
            raise ValueError(f"Invalid saved tensor at index {index}; no samples were overwritten")
        with Image.open(folder / f"image_{index:04d}.png") as image:
            if image.size != (size, size) or image.mode != "RGB":
                raise ValueError(f"Invalid saved PNG at index {index}")
            image.verify()
    return frame


def _save_grid(folder, count):
    images = [torch.load(folder / f"image_{i:04d}.pt", map_location="cpu", weights_only=True)
              for i in range(min(16, count))]
    temporary = folder / ".grid.png.tmp"
    save_image(torch.cat(images), temporary, format="png", nrow=4,
               normalize=True, value_range=(-1, 1))
    temporary.replace(folder / "grid.png")


def resume_method(split, name, *, namespace, validate_only=False):
    """Continue after the last committed sample, preserving the saved method.

    Pass ``namespace=globals()`` from the original notebook. Its ``load_run``
    validates the source fingerprint, protocol, weights and runtime first.
    A file with no corresponding CSV row is an uncommitted sample and will be
    regenerated with its original seed. Completed sample files are untouched.
    ``validate_only=True`` checks progress without loading models or writing.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test")
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("name must be a single folder name")
    ns = namespace
    root, protocol, config = ns["load_run"](ns["RUN_ARGS"])
    root = Path(root).resolve()
    folder = root / split / name
    manifest = json.loads((folder / "manifest.json").read_text())
    pairs = [list(pair) for pair in ns["validate_pairs"](protocol["splits"][split])]
    recorded_runtime = json.loads((root / "environment.json").read_text())
    for key, expected in [("config", config), ("pairs", pairs),
                          ("shared_code_sha256", config["shared_code_sha256"]),
                          ("runtime", recorded_runtime)]:
        if manifest.get(key) != expected:
            raise ValueError(f"Saved method {key} does not match this run")
    schedule = manifest["schedule"]
    mode = schedule["mode"]
    if mode not in {"original", "removal", "loop", "damped", "xsa_reuse"}:
        raise ValueError("Resume supports the built-in methods, not custom builders")
    if (name == "original") != (mode == "original"):
        raise ValueError("The original folder must contain the original method")
    device = ns["select_device"](config["device"])
    frame = _saved_progress(folder, pairs, config["image_size"], str(device))
    start = len(frame)
    print(f"Verified {start}/{len(pairs)} saved samples in {folder}", flush=True)
    if validate_only:
        return frame
    if start == len(pairs):
        if not (folder / "grid.png").exists():
            _save_grid(folder, start)
        print("Already complete; no generation needed.", flush=True)
        return frame

    model = vae = diffusion = None
    try:
        # Restore the actual saved schedule, including the old damping policy.
        model, vae, diffusion = ns["_prepare"](
            config, mode, schedule["removed"], schedule["alpha"])
        if model.blocks.metadata() != schedule:
            raise ValueError("Reconstructed block schedule differs from the saved method")
        if getattr(vae.config, "_commit_hash", None) != manifest.get("vae_commit"):
            raise ValueError("VAE revision differs from the saved method")
        model.to(device)
        vae.to(device)
        ns["warmup"](model, vae, config["cfg_scale"])

        # Keep continuation provenance separate from the original manifest.
        log_path = folder / "resume_sessions.json"
        sessions = json.loads(log_path.read_text()) if log_path.exists() else []
        sessions.append(dict(start_index=start, total_images=len(pairs),
                             started_utc=datetime.now(timezone.utc).isoformat(),
                             helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             generation_code_sha256=config["shared_code_sha256"],
                             runtime=recorded_runtime))
        log_tmp = folder / ".resume_sessions.json.tmp"
        log_tmp.write_text(json.dumps(sessions, indent=2) + "\n")
        log_tmp.replace(log_path)

        rows = frame.to_dict("records")
        for index in range(start, len(pairs)):
            label, seed = pairs[index]
            ns["_cleanup"]()
            ns["synchronize"](device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            image = ns["generate"](model, vae, diffusion, label, seed, config["cfg_scale"])
            ns["synchronize"](device)
            elapsed = time.perf_counter() - began
            row = dict(index=index, label=label, seed=seed, latency_s=elapsed,
                       device=str(device),
                       peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                             if device.type == "cuda" else None),
                       peak_reserved_bytes=(torch.cuda.max_memory_reserved(device)
                                            if device.type == "cuda" else None),
                       mps_allocated_bytes_after_sample=(torch.mps.current_allocated_memory()
                                                        if device.type == "mps" else None),
                       mps_driver_bytes_after_sample=(torch.mps.driver_allocated_memory()
                                                     if device.type == "mps" else None))
            image = ns["preprocess_image"](image, config["image_size"])
            pt_tmp = folder / f".image_{index:04d}.pt.tmp"
            png_tmp = folder / f".image_{index:04d}.png.tmp"
            torch.save(image, pt_tmp)
            save_image(image, png_tmp, format="png", normalize=True, value_range=(-1, 1))
            pt_tmp.replace(folder / f"image_{index:04d}.pt")
            png_tmp.replace(folder / f"image_{index:04d}.png")
            rows.append(row)
            frame = pd.DataFrame(rows)
            csv_tmp = folder / ".samples.csv.tmp"
            frame.to_csv(csv_tmp, index=False)
            csv_tmp.replace(folder / "samples.csv")
            print(f"{mode}: {index + 1}/{len(pairs)}, {elapsed:.2f} s", flush=True)
        _save_grid(folder, len(pairs))
        return frame
    finally:
        del model, vae, diffusion
        ns["_cleanup"]()
