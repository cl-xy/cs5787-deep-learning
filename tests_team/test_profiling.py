"""Block scoring, selection artifacts and the heatmap, on a synthetic profile."""
import json

import matplotlib
import pandas as pd
import pytest

from team_pipeline.profiling import block_scores, plot_profile_heatmap, summarize_profile

DEPTH = 8
TIMESTEPS = (900, 600, 300, 0)


def synthetic_frame(change):
    """``change[block]`` is that block's constant feature change at every timestep."""
    rows = [dict(timestep=t, block=b, conditional=c, unconditional=c, mean_cfg=c, seed=s)
            for s in (1, 2) for t in TIMESTEPS for b, c in enumerate(change)]
    return pd.DataFrame(rows)


CHANGE = [0.5, 0.4, 0.01, 0.3, 0.02, 0.2, 0.1, 0.6]  # blocks 2 and 4 change least


def test_block_scores_pick_least_changing_interior_blocks():
    matrix, scores, removed = block_scores(synthetic_frame(CHANGE), remove_count=2)
    assert removed == [2, 4]
    assert list(matrix.columns) == sorted(TIMESTEPS, reverse=True)  # sampling order
    assert scores["selected"].tolist() == [b in removed for b in range(DEPTH)]
    assert {"early_noisy", "middle", "late_clean"} <= set(scores.columns)


def test_first_and_last_blocks_are_protected():
    frame = synthetic_frame([0.0, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.0])
    _, _, removed = block_scores(frame, remove_count=2)
    assert 0 not in removed and DEPTH - 1 not in removed


def test_invalid_budget_and_incomplete_profile_are_rejected():
    frame = synthetic_frame(CHANGE)
    with pytest.raises(ValueError, match="removal budget"):
        block_scores(frame, remove_count=DEPTH)
    with pytest.raises(ValueError, match="Incomplete"):
        missing_cell = (frame["block"] == 3) & (frame["timestep"] == 300)
        block_scores(frame[~missing_cell], remove_count=2)


def test_summarize_profile_writes_artifacts_without_changing_backend(tmp_path):
    backend = matplotlib.get_backend()
    removed, scores = summarize_profile(synthetic_frame(CHANGE), tmp_path, remove_count=2)
    assert removed == [2, 4] and len(scores) == DEPTH
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["removed"] == [2, 4] and selection["index_base"] == 0
    assert selection["protected"] == [0, DEPTH - 1]
    for name in ("block_scores.csv", "feature_change_matrix.csv", "feature_change_heatmap.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert matplotlib.get_backend() == backend


def test_heatmap_can_be_written_alone(tmp_path):
    matrix, _, removed = block_scores(synthetic_frame(CHANGE), remove_count=2)
    plot_profile_heatmap(matrix, removed, tmp_path / "heat.png")
    assert (tmp_path / "heat.png").read_bytes()[:4] == b"\x89PNG"
