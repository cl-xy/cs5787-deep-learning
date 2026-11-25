"""Frozen block selections shared by all methods in a comparison."""
import json
from pathlib import Path

DEFAULT_DEPTH = 28  # DiT-XL/2 transformer blocks.
DEFAULT_REMOVED = (4, 7)


def load_fixed_selection(root):
    """Read ``<root>/profile/selection.json`` (calibrated or fixed); indices are zero based."""
    selection = json.loads((Path(root) / "profile/selection.json").read_text())
    if selection.get("index_base", 0) != 0:
        raise ValueError("Expected zero-based block indices")
    return selection


def write_fixed_selection(root, removed=DEFAULT_REMOVED, depth=DEFAULT_DEPTH):
    """Freeze an explicit block selection for a run, skipping calibration inference.

    Every built-in compressed method (removal/loop/damped/xsa_reuse) then uses the
    same blocks. An existing selection is never replaced: create a new run instead.
    """
    removed = list(removed)
    if (
        not removed
        or any(type(i) is not int for i in removed)
        or removed != sorted(set(removed))
        or any(i <= 0 or i >= depth for i in removed)
    ):
        raise ValueError(f"Use sorted, unique block indices from 1 through {depth - 1}")

    path = Path(root) / "profile/selection.json"

    if path.exists():
        selection = load_fixed_selection(root)
        if selection["removed"] != removed:
            raise ValueError("Selection is frozen for this run; create a new run")
        return selection

    selection = {
        "removed": removed,
        "index_base": 0,
        "scoring": "explicitly configured fixed selection",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection
