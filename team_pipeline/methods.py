"""Block schedules for the CLI and custom methods.

The implementation lives in ``xsa_method.schedule``. This subclass only keeps the
CLI's historical default of a fixed damping ``alpha=0.5``; the notebook uses
``xsa_method.ScheduledBlocks`` directly (default ``alpha="inverse_k"``).
"""
from xsa_method.schedule import ScheduledBlocks as _ScheduledBlocks

MODES = ("original", "removal", "loop", "damped", "xsa_reuse")
DEFAULT_ALPHA = 0.5


class ScheduledBlocks(_ScheduledBlocks):
    """``xsa_method.ScheduledBlocks`` with a fixed-alpha default."""

    def __init__(self, blocks, removed=(), mode="original", alpha=DEFAULT_ALPHA, xsa_eps=1e-8):
        super().__init__(blocks, removed, mode, alpha, xsa_eps=xsa_eps)
