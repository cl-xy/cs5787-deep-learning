"""One runner for all team members: python -m team_pipeline.cli --help."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

from .data import load_protocol, protocol_hash
from .experiment import (calibrate, compare_results, comparison_grid, configure_dit,
                         download_checkpoint, environment, run_variant, select_device,
                         sha256, shared_code_hash, summarize_profile, write_json)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "profile", "run", "evaluate"]:
        sub = subs.add_parser(name)
        sub.add_argument("--output", required=True, help="Shared run directory")
        if name != "evaluate":
            sub.add_argument("--repo", default="third_party/DiT")
        if name == "prepare":
            sub.add_argument("--protocol", default="team_protocol.json")
            sub.add_argument("--device", default="auto")
            sub.add_argument("--checkpoint", default="checkpoints/DiT-XL-2-256x256.pt")
        elif name == "profile":
            sub.add_argument("--remove-count", type=int, default=2)
        elif name == "run":
            sub.add_argument("--split", choices=["validation", "test", "fid"], default="validation")
            sub.add_argument("--limit", type=int, help="Use only the first N pairs of the split (smoke runs)")
            sub.add_argument("--method", required=True, help="original/removal/loop/damped or module:function")
            sub.add_argument("--name", required=True, help="Unique folder name; baseline must be original")
            sub.add_argument("--removed", type=int, nargs="+", help="Otherwise use profile/selection.json")
            sub.add_argument("--alpha", type=float, default=0.5)
            sub.add_argument("--method-config", help="JSON file passed to a custom method")
            sub.add_argument("--keep-checkpoint", action="store_true")
        else:
            sub.add_argument("--split", choices=["validation", "test", "fid"], default="validation")
            sub.add_argument("--names", nargs="+", required=True)
    for name in ["clip", "fid"]:
        sub = subs.add_parser(name, help=f"{name.upper()} metric on finished fid-split runs")
        sub.add_argument("--output", required=True, help="Shared run directory")
        sub.add_argument("--split", choices=["validation", "test", "fid"], default="fid")
        sub.add_argument("--names", nargs="+", required=True)
        sub.add_argument("--device", default="cpu", help="Device for the metric network only")
        if name == "fid":
            sub.add_argument("--reference", help="ADM ImageNet-256 reference .npz with mu/sigma")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return
    if args.command in {"clip", "fid"}:
        from . import metrics
        root = Path(args.output) / args.split
        if args.command == "clip":
            print(metrics.clip_report(root, args.names).to_string(index=False))
        else:
            print(metrics.fid_report(root, args.names, reference_stats=args.reference,
                                     device=args.device).to_string(index=False))
        return
    if args.command == "evaluate":
        root = Path(args.output) / args.split
        print(compare_results(root, args.names).to_string(index=False))
        pairs = load_protocol(Path(args.output) / "protocol.json")["splits"][args.split]
        comparison_grid(root, args.names, pairs)
        return
    root, protocol, config = load_run(args)
    if args.command == "profile":
        frame = calibrate(config, protocol["splits"]["calibration"], root / "profile")
        removed, _ = summarize_profile(frame, root / "profile", args.remove_count)
        print("Selected blocks:", removed)
        return
    if Path(args.name).name != args.name or args.name in {".", "..", ""}:
        raise ValueError("name must be a single folder name")
    if (args.name == "original") != (args.method == "original"):
        raise ValueError("Reserve the name original for the unmodified baseline")
    transform = None
    metadata = dict(method=args.method)
    removed = ()
    mode = args.method
    if ":" in args.method:
        module, function = args.method.rsplit(":", 1)
        builder = getattr(importlib.import_module(module), function)
        method_config = json.loads(Path(args.method_config).read_text()) if args.method_config else {}
        if args.removed:
            raise ValueError("Specify custom method settings in --method-config")
        source = inspect.getsourcefile(builder)
        metadata.update(settings=method_config, source_sha256=sha256(source) if source else None)
        transform = lambda model: builder(model, method_config)
        mode = "original"
    elif mode in {"removal", "loop", "damped"}:
        removed = args.removed if args.removed is not None else json.loads(
            (root / "profile" / "selection.json").read_text())["removed"]
    elif mode != "original":
        raise ValueError("Unknown method")
    elif args.removed or args.method_config:
        raise ValueError("Original baseline cannot have method settings")
    pairs = protocol["splits"].get(args.split)
    if pairs is None:
        raise ValueError(f"Protocol has no '{args.split}' split; prepare a root from the updated team_protocol.json")
    if args.limit is not None:
        if args.limit < 2:
            raise ValueError("--limit must be at least 2")
        pairs = pairs[:args.limit]
    run_variant(config, pairs, root / args.split / args.name,
                mode=mode, removed=removed, alpha=args.alpha, model_transform=transform,
                method_metadata=metadata, keep_checkpoint=args.keep_checkpoint)


if __name__ == "__main__":
    main()
