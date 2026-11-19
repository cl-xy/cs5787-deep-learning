"""Training-free XSA reuse for DiT; notebook-independent method code.

Settings remain explicit at the call site. The schedule defaults to inverse-k
for damped comparisons; XSA itself applies no residual damping.
"""
from pathlib import Path

from .attention import xsa_attention_scope, xsa_filter_heads
from .checkpoint import load_compact_checkpoint
from .experiment import bind_runner, run_method
from .schedule import ScheduledBlocks
from .selection import load_fixed_selection, write_fixed_selection

__all__ = [
    "ScheduledBlocks", "load_compact_checkpoint", "load_fixed_selection",
    "write_fixed_selection", "run_method", "bind_runner", "xsa_attention_scope",
    "xsa_filter_heads", "method_sources",
]

# Capture at import time: later unexecuted file edits must not change the
# fingerprint of the implementation currently loaded in the notebook kernel.
_SOURCES = {f"xsa_method/{path.name}": path.read_text()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def method_sources():
    """Return the package source snapshot captured when this kernel imported it."""
    return dict(_SOURCES)
