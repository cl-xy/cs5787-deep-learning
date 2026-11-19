"""Example custom method. Copy this file for each member's method; leave the shared
pipeline unchanged. Use it from the notebook via ``bind_runner(...)(builder=build_model)``.
"""
from xsa_method import ScheduledBlocks


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
