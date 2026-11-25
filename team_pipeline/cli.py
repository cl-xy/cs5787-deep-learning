"""One runner for all team members: python -m team_pipeline.cli --help."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

from .experiment import (calibrate, compare_results, comparison_grid, load_fixed_selection,
                         run_variant, sha256, summarize_profile, write_fixed_selection)
from .runs import load_run, prepare  # noqa: F401  (load_run is imported by loop_compression)

BUILTIN_METHODS = ("removal", "loop", "damped", "xsa_reuse")


def select_pairs(pairs, stride=1, limit=None):
    """Select a deterministic split subset by striding first, then limiting."""
    if stride < 1:
        raise ValueError("--stride must be at least 1")
    if limit is not None and limit < 2:
        raise ValueError("--limit must be at least 2")
    selected = pairs[::stride]
    if limit is not None:
        selected = selected[:limit]
    if len(selected) < 2:
        raise ValueError("Pair selection must contain at least 2 images")
    return selected


def resolve_alpha(method, alpha):
    """Only damped scales the replacement residual; other built-ins require alpha=1."""
    if method == "damped":
        return 0.5 if alpha is None else float(alpha)
    if method in BUILTIN_METHODS:
        if alpha is not None and float(alpha) != 1.0:
            raise ValueError("Only damped mode scales the block residual; use --alpha 1.0 or omit it")
        return 1.0
    if alpha is not None:
        raise ValueError("--alpha applies only to built-in methods")
    return 0.5  # Unused by original/custom methods; keeps ScheduledBlocks' default.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "profile", "select", "run", "evaluate"]:
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
        elif name == "select":
            sub.add_argument("--removed", type=int, nargs="+", default=[4, 7],
                             help="Zero-based blocks frozen for every built-in method (no calibration)")
        elif name == "run":
            sub.add_argument("--split", choices=["validation", "test", "fid"], default="validation")
            sub.add_argument("--limit", type=int, help="Use at most N selected pairs")
            sub.add_argument("--stride", type=int, default=1,
                             help="Select every Nth split pair before applying --limit")
            sub.add_argument("--method", required=True,
                             help="original/removal/loop/damped/xsa_reuse or module:function")
            sub.add_argument("--name", required=True, help="Unique folder name; baseline must be original")
            sub.add_argument("--removed", type=int, nargs="+", help="Otherwise use profile/selection.json")
            sub.add_argument("--alpha", type=float,
                             help="Damped residual scale (default 0.5); other built-ins use 1.0")
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
    sub = subs.add_parser("extended", help="CLIP cosine scores and optional Clean-FID / TPIPS after evaluate")
    sub.add_argument("--output", required=True, help="Shared run directory")
    sub.add_argument("--split", choices=["validation", "test", "fid"], default="validation")
    sub.add_argument("--names", nargs="+", required=True)
    sub.add_argument("--fid-reference-dir", help="Folder of REAL images; enables Clean-FID")
    sub.add_argument("--fid-reference-description",
                     help="Dataset, split, class mixture, preprocessing (read from reference.json if omitted)")
    sub.add_argument("--fid-device", default="cpu")
    sub.add_argument("--fid-batch-size", type=int, default=16)
    sub.add_argument("--tpips-factors", nargs="*", default=[],
                     help="e.g. overall shape texture; enables TPIPS")
    sub.add_argument("--tpips-device", default="cuda:0")
    sub.add_argument("--tpips-model-path")
    sub.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True,
                     help="Evaluate both CLIP cosine scores (default: enabled)")
    sub.add_argument("--clip-device", default="cpu")
    sub.add_argument("--clip-batch-size", type=int, default=16)
    sub = subs.add_parser("fid-reference",
                          help="Build a real ImageNet FID reference folder for the generated classes")
    sub.add_argument("--output", required=True, help="Shared run directory")
    sub.add_argument("--split", choices=["validation", "test", "fid"], default="validation",
                     help="Generated split whose original manifest defines the classes")
    sub.add_argument("--reference-root", default="fid_references",
                     help="Parent folder for reference images (outside the run)")
    sub.add_argument("--images-per-class", type=int, default=50)
    sub.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    sub.add_argument("--dataset-split", default="validation")
    sub.add_argument("--streaming", action="store_true",
                     help="Stream instead of downloading the whole dataset split")
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
    if args.command == "fid-reference":
        from .extended_metrics import build_imagenet_reference
        folder, description = build_imagenet_reference(
            Path(args.output) / args.split, args.reference_root,
            images_per_class=args.images_per_class, dataset=args.dataset,
            split=args.dataset_split, streaming=args.streaming)
        print("FID reference:", folder)
        print("Description:", description)
        return
    if args.command == "extended":
        from .extended_metrics import evaluate_fid_tpips
        result = evaluate_fid_tpips(
            Path(args.output) / args.split, args.names,
            fid_reference_dir=args.fid_reference_dir,
            fid_reference_description=args.fid_reference_description,
            tpips_factors=args.tpips_factors, fid_device=args.fid_device,
            tpips_device=args.tpips_device, fid_batch_size=args.fid_batch_size,
            tpips_model_path=args.tpips_model_path, enable_clip=args.clip,
            clip_device=args.clip_device, clip_batch_size=args.clip_batch_size)
        print(result.to_string(index=False))
        return
    if args.command == "evaluate":
        root = Path(args.output) / args.split
        print(compare_results(root, args.names).to_string(index=False))
        # Use the pairs the baseline actually ran: a --limit run covers only a prefix of the split.
        manifest = json.loads((root / "original" / "manifest.json").read_text())
        comparison_grid(root, args.names, [tuple(pair) for pair in manifest["pairs"]])
        return
    root, protocol, config = load_run(args)
    if args.command == "profile":
        frame = calibrate(config, protocol["splits"]["calibration"], root / "profile")
        removed, _ = summarize_profile(frame, root / "profile", args.remove_count)
        print("Selected blocks:", removed)
        return
    if args.command == "select":
        print("Selection:", write_fixed_selection(root, args.removed))
        return
    if Path(args.name).name != args.name or args.name in {".", "..", ""}:
        raise ValueError("name must be a single folder name")
    if (args.name == "original") != (args.method == "original"):
        raise ValueError("Reserve the name original for the unmodified baseline")
    alpha = resolve_alpha(args.method, args.alpha)
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
    elif mode in BUILTIN_METHODS:
        if args.removed is not None:
            removed = args.removed
        else:
            selection = load_fixed_selection(root)
            removed = selection["removed"]
            metadata["selection"] = selection
        metadata["replacement_residual_scale"] = alpha
    elif mode != "original":
        raise ValueError("Unknown method")
    elif args.removed or args.method_config:
        raise ValueError("Original baseline cannot have method settings")
    pairs = protocol["splits"].get(args.split)
    if pairs is None:
        raise ValueError(f"Protocol has no '{args.split}' split; prepare a root from the updated team_protocol.json")
    pairs = select_pairs(pairs, stride=args.stride, limit=args.limit)
    run_variant(config, pairs, root / args.split / args.name,
                mode=mode, removed=removed, alpha=alpha, model_transform=transform,
                method_metadata=metadata, keep_checkpoint=args.keep_checkpoint)


if __name__ == "__main__":
    main()
