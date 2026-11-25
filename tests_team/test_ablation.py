"""Block-aligned analysis and no-loop experiment regression tests."""
import json

import numpy as np
import pandas as pd
import pytest

from team_pipeline import ablation_report as report


def artifacts(tmp_path):
    profile = tmp_path / 'profile'
    root = tmp_path / 'validation'
    profile.mkdir()
    root.mkdir()
    pd.DataFrame({'block': [2, 0, 1], 'mean_feature_change': [0.3, 0.1, 0.2]}).to_csv(
        profile / 'block_scores.csv', index=False)
    base = dict(config={}, pairs=[[1, 1]], runtime={}, shared_code_sha256='same')
    rows = []
    for name, block in [('original', None), ('remove_2', 2), ('remove_0', 0), ('remove_1', 1)]:
        folder = root / name
        folder.mkdir()
        schedule = dict(mode='original' if block is None else 'removal',
                        removed=[] if block is None else [block], depth=3)
        (folder / 'manifest.json').write_text(json.dumps(dict(base, schedule=schedule)))
        rows.append(dict(variant=name, lpips_mean=0 if block is None else (block + 1) / 10))
    pd.DataFrame(rows).to_csv(root / 'results.csv', index=False)
    return profile, root


def test_analysis_joins_by_block_and_rejects_unmatched_runs(tmp_path):
    profile, root = artifacts(tmp_path)
    frame = report.load_block_ablation(profile, root)
    assert frame.block.tolist() == [0, 1, 2]
    assert frame.mean_feature_change.tolist() == [0.1, 0.2, 0.3]
    assert frame.lpips_mean.tolist() == [0.1, 0.2, 0.3]
    path = root / 'remove_1/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['pairs'] = [[1, 2]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='pairs'):
        report.load_block_ablation(profile, root)


def test_analysis_requires_complete_finite_sweep(tmp_path):
    profile, root = artifacts(tmp_path)
    results = pd.read_csv(root / 'results.csv')
    results.iloc[:-1].to_csv(root / 'results.csv', index=False)
    with pytest.raises(ValueError, match='every block'):
        report.load_block_ablation(profile, root)
    results.loc[1, 'lpips_mean'] = np.inf
    results.to_csv(root / 'results.csv', index=False)
    with pytest.raises(ValueError, match='finite'):
        report.load_block_ablation(profile, root)


def test_depth_thirds_cover_all_blocks_and_position_plot(tmp_path):
    groups = report.depth_partitions(28)
    assert [len(g) for g in groups] == [10, 9, 9]
    assert np.concatenate(groups).tolist() == list(range(28))
    profile, root = artifacts(tmp_path)
    paths = report.plot_block_lpips(report.load_block_ablation(profile, root), tmp_path)
    assert paths['png'].read_bytes().startswith(b'\x89PNG')


def test_single_removal_rules_and_boundary_successor():
    from team_pipeline.ablation import single_removal_plan
    # Last block has the largest delta but no successor: next largest is block 3.
    scores = pd.DataFrame(dict(block=range(6), mean_feature_change=[2, 1, 3, 8, 4, 9]))
    plan = single_removal_plan(scores)
    assert plan == {'A_min_delta': [1], 'B_after_max_delta': [4],
                    'C_early': [0], 'C_middle': [2], 'C_late': [4]}
    with pytest.raises(ValueError, match='finite'):
        single_removal_plan(scores.assign(mean_feature_change=np.nan))


def test_removal_can_skip_first_block_but_loop_cannot():
    import torch
    from torch import nn
    from xsa_method.schedule import ScheduledBlocks
    blocks = [nn.Linear(1, 1) for _ in range(3)]
    schedule = ScheduledBlocks(blocks, [0], 'removal')
    assert list(schedule) == blocks[1:]
    assert schedule.metadata()['removed'] == [0]
    with pytest.raises(ValueError, match='Keep block 0'):
        ScheduledBlocks(blocks, [0], 'loop')


def test_plan_generates_matched_no_loop_variants_and_checks_resume(tiny_run):
    from team_pipeline.ablation import run_plan
    from team_pipeline.evaluation import compare_results
    TINY_CONFIG = dict(device='cpu', image_size=8, cfg_scale=4, sampling_steps=2)
    TINY_PAIRS = [(1, 10), (2, 20)]
    output = tiny_run / 'study'
    plan = {'first': [0], 'last': [3]}
    names = run_plan(TINY_CONFIG, TINY_PAIRS, output, plan)
    results = compare_results(output, names, metric=lambda x, y: (x-y).square().mean())
    assert results.lpips_mean.iloc[0] == 0
    assert results.block_evaluations_per_forward.tolist() == [4, 3, 3]
    assert results.lpips_mean.iloc[1] > 0
    assert run_plan(TINY_CONFIG, TINY_PAIRS, output, plan) == names
    with pytest.raises(ValueError, match='Frozen study'):
        run_plan(TINY_CONFIG, TINY_PAIRS, output, {'first': [2]})
    (output / 'first/image_0000.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        run_plan(TINY_CONFIG, TINY_PAIRS, output, plan)
