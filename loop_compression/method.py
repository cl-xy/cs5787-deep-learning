"""Builder for the shared runner:
python -m team_pipeline.cli run ... --method loop_compression.method:build_model --method-config cfg.json
"""
from .blocks import LoopedBlocks

SETTING_KEYS = {"window", "kept", "K", "iteration", "strategy", "h", "beta"}


def build_model(model, settings):
    """Replace the middle of ``model.blocks`` per ``settings``; weights are untouched."""
    unknown = set(settings) - SETTING_KEYS
    if unknown:
        raise ValueError(f"Unknown loop settings: {sorted(unknown)}")
    if "window" not in settings:
        raise ValueError("settings must contain 'window': [first, last]")
    model.blocks = LoopedBlocks(model.blocks.retained.values(), **settings)
    return model
