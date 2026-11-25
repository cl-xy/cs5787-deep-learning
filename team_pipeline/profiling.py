"""Feature-change profiling of the original model and block-removal selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .runtime import write_json

STAGE_NAMES = ("early_noisy", "middle", "late_clean")  # Sampling order, not increasing t.
RATIO_EPSILON = 1e-8
LOG_FLOOR = 1e-12
SCORING_DESCRIPTION = ("mean of per-sample normalized L1 changes, equally averaged "
                       "over CFG branches and timesteps")


class FeatureProfiler:
    """Per-sample L1 ratios, averaged separately over CFG's two branches.

    Hooks run only on original-model calibration trajectories. Scalar tensors
    stay on the device until collection ends, avoiding per-block CPU synchronization.
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.records = []
        self.timestep = None

    def __enter__(self):
        if self.model.blocks.mode != "original":
            raise ValueError("Profile original model only.")
        # Official forward_with_cfg calls self.forward directly, bypassing root
        # Module hooks. The timestep embedder is still invoked through __call__.
        self.handles.append(self.model.t_embedder.register_forward_pre_hook(self._before))
        for index, block in self.model.blocks.retained.items():
            def hook(module, args, output, index=int(index)):
                x = args[0].float()
                ratio = (output.float() - x).abs().flatten(1).sum(1) / (
                    x.abs().flatten(1).sum(1) + RATIO_EPSILON)
                cond, uncond = ratio.chunk(2)
                self.records.append(torch.stack([
                    self.timestep.float(), ratio.new_tensor(index), cond.mean(), uncond.mean()
                ]).detach())
            self.handles.append(block.register_forward_hook(hook))
        return self

    def _before(self, module, args):
        self.timestep = args[0][0].detach()

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def frame(self):
        if not self.records:
            raise RuntimeError("No profile records collected.")
        values = torch.stack(self.records).cpu().numpy()
        frame = pd.DataFrame(values, columns=["timestep", "block", "conditional", "unconditional"])
        frame[["timestep", "block"]] = frame[["timestep", "block"]].astype(int)
        frame["mean_cfg"] = frame[["conditional", "unconditional"]].mean(axis=1)
        return frame


def block_scores(frame, remove_count=4):
    """Rank blocks by mean feature change. Returns ``(matrix, scores, removed)``.

    ``matrix`` is block x timestep (columns in sampling order); ``scores`` has one
    row per block. The first and last blocks are protected from removal.
    """
    # Average samples within each timestep first, then give each timestep equal weight.
    matrix = frame.groupby(["block", "timestep"])["mean_cfg"].mean().unstack()
    matrix = matrix.reindex(columns=sorted(matrix.columns, reverse=True))
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Incomplete or nonfinite feature profile.")
    scores = matrix.mean(axis=1).rename("mean_feature_change").to_frame()
    depth = len(scores)
    candidates = scores.loc[~scores.index.isin([0, depth - 1])]
    if not 0 < remove_count <= len(candidates):
        raise ValueError("Invalid removal budget.")
    ranked = candidates.sort_values("mean_feature_change", kind="stable")
    removed = sorted(ranked.head(remove_count).index.tolist())
    scores["selected"] = scores.index.isin(removed)
    for stage, columns in zip(STAGE_NAMES, np.array_split(matrix.columns.to_numpy(), 3)):
        scores[stage] = matrix.loc[:, columns].mean(axis=1)
    return matrix, scores, removed


def plot_profile_heatmap(matrix, removed, path):
    """Save the linear and log10 feature-change heatmaps.

    Uses a standalone Figure, so the caller's matplotlib backend is untouched.
    """
    from matplotlib.figure import Figure
    values = matrix.to_numpy()
    fig = Figure(figsize=(15, 7), layout="constrained")
    axes = fig.subplots(1, 2)
    panels = [(values, "Mean normalized feature change"),
              (np.log10(values.clip(LOG_FLOOR)), "log10 normalized feature change")]
    for ax, (panel, title) in zip(axes, panels):
        image = ax.imshow(panel, aspect="auto", interpolation="nearest")
        ticks = np.unique(np.linspace(0, len(matrix.columns) - 1,
                                      min(6, len(matrix.columns))).astype(int))
        ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks])
        ax.set_yticks(range(len(matrix)))
        ax.set_xlabel("Diffusion timestep (sampling proceeds left to right)")
        ax.set_ylabel("Block index (zero based)")
        ax.set_title(title)
        for index in removed:
            ax.axhline(index, color="red", linewidth=0.6, alpha=0.6)
        fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=180)


def write_profile(output, matrix, scores, removed):
    """Write block_scores.csv, feature_change_matrix.csv and selection.json."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    depth = len(scores)
    scores.to_csv(output / "block_scores.csv", index_label="block")
    matrix.to_csv(output / "feature_change_matrix.csv")
    write_json(output / "selection.json", dict(
        removed=removed, protected=[0, depth - 1], index_base=0, scoring=SCORING_DESCRIPTION))


def summarize_profile(frame, output, remove_count=4):
    """Score blocks, write the CSV/JSON/heatmap artifacts and return ``(removed, scores)``."""
    matrix, scores, removed = block_scores(frame, remove_count)
    write_profile(output, matrix, scores, removed)
    plot_profile_heatmap(matrix, removed, Path(output) / "feature_change_heatmap.png")
    return removed, scores
