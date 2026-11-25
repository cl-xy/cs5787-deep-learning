"""Display helpers for Jupyter/Colab. Not part of the run fingerprint (no effect on results)."""
from __future__ import annotations

from pathlib import Path


def _display(*objects):
    from IPython.display import display
    for obj in objects:
        display(obj)


def show_image(path):
    """Display a PNG file inline."""
    from IPython.display import Image
    _display(Image(filename=str(path)))


def show_figure_paths(paths):
    """Display the PNG of a ``{"png": ..., "pdf": ...}`` figure result and print both paths."""
    if paths is None:
        return
    show_image(paths["png"])
    print("PNG:", paths["png"])
    print("PDF:", paths["pdf"])


def run_methods(run, run_root, split, methods, only=None, show=True):
    """Run ``methods`` (``(method, name, run_kwargs)`` tuples) and show each grid.

    ``run`` comes from ``xsa_method.bind_runner``. Pass ``only=name`` to run one method,
    which keeps long jobs in separate notebook cells.
    """
    selected = [m for m in methods if only is None or m[1] == only]
    if not selected:
        raise ValueError(f"No method named {only!r}")
    for method, name, kwargs in selected:
        run(split, method, name, **kwargs)
        if show:
            show_image(Path(run_root) / split / name / "grid.png")


def show_evaluation(outcome):
    """Display an ``evaluate_saved_split`` outcome: metric table, grid and file locations."""
    _display(outcome["results"])
    show_image(outcome["paths"]["grid"])
    print("Summary:", outcome["paths"]["results"])
    print("Per-image metrics:", outcome["paths"]["paired"])
    print("Evaluation settings:", outcome["paths"]["settings"])
