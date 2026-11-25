"""Backward-compatible import surface for the shared pipeline.

The implementation lives in focused modules; import from them in new code:

* ``runtime``    device selection, determinism, RNG, JSON and hashing helpers
* ``profiling``  feature-change profiling and block selection scoring
* ``sampling``   model loading, deterministic sampling and ``run_variant``
* ``evaluation`` paired LPIPS, efficiency summaries and comparison grids
* ``provenance`` run fingerprint (import-time source snapshot)

Tests that replace a function must patch the module that *calls* it, e.g.
``monkeypatch.setattr(team_pipeline.sampling, "generate", ...)``; patching this
module has no effect on the implementation.
"""
from xsa_method.selection import load_fixed_selection, write_fixed_selection

from .data import initial_latent, preprocess_image, validate_pairs
from .evaluation import compare_results, comparison_grid, evaluate_split
from .methods import ScheduledBlocks
from .profiling import FeatureProfiler, summarize_profile
from .provenance import shared_code_hash
from .runtime import (cleanup, configure_dit, environment, install_mps_sampler_compatibility,
                      paired_rng, select_device, sha256, synchronize, write_json)
from .sampling import (calibrate, checkpoint_stats, download_checkpoint, estimate_dit_flops,
                       generate, load_compact_checkpoint, load_model, run_variant, warmup)

__all__ = [
    "FeatureProfiler", "ScheduledBlocks", "calibrate", "checkpoint_stats", "cleanup",
    "compare_results", "comparison_grid", "configure_dit", "download_checkpoint",
    "environment", "estimate_dit_flops", "evaluate_split", "generate", "initial_latent",
    "install_mps_sampler_compatibility", "load_compact_checkpoint", "load_fixed_selection",
    "load_model", "paired_rng", "preprocess_image", "run_variant", "select_device",
    "sha256", "shared_code_hash", "summarize_profile", "synchronize", "validate_pairs",
    "warmup", "write_fixed_selection", "write_json",
]
