"""Connect XSA and its baselines to the shared experiment runner."""
import hashlib
import inspect
import json
from pathlib import Path

from .selection import load_fixed_selection


def run_method(run_args, split, method, name, *, load_run, run_variant,
               keep_checkpoints=False, alpha=None, builder=None, settings=None):
    """Run a method using explicit shared-pipeline dependencies.

    ``load_run(run_args)`` validates provenance and returns root/protocol/config;
    ``run_variant`` performs the unchanged sampling, timing and artifact writes.
    This module does not depend on a notebook kernel or a global current run.
    """
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
    root, protocol, config = load_run(run_args)
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
                       method_metadata=metadata, keep_checkpoint=keep_checkpoints)




def bind_runner(run_args, *, load_run, run_variant, keep_checkpoints=False):
    """Return ``run(split, method, name, *, alpha=None, builder=None, settings=None)``.

    Fixes the run, the shared pipeline functions and the checkpoint policy once, so a
    notebook can call ``run("validation", "loop", "member2_loop")`` without globals.
    """
    def run(split, method, name, *, alpha=None, builder=None, settings=None):
        return run_method(run_args, split, method, name, load_run=load_run,
                          run_variant=run_variant, keep_checkpoints=keep_checkpoints,
                          alpha=alpha, builder=builder, settings=settings)
    return run
