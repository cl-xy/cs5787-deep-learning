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
    assert plan == {'A_min_delta': [1], 'B_before_max_delta': [4], 'B_after_max_delta': [4],
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


def test_top_r_uses_descending_frozen_scores_and_nested_sets():
    from team_pipeline.ablation import top_r_plan
    sweep = pd.DataFrame(dict(block=[2, 0, 1], lpips_mean=[0.5, 0.1, 0.5]))
    plan, ranking = top_r_plan(sweep, [1, 2, 3])
    assert ranking == [1, 2, 0]
    assert plan == {'top_r_01': [1], 'top_r_02': [1, 2], 'top_r_03': [0, 1, 2]}
    for counts in ([0], [4], [2, 1], [1, 1], [1.5], []):
        with pytest.raises(ValueError, match='counts'):
            top_r_plan(sweep, counts)
    with pytest.raises(ValueError, match='finite'):
        top_r_plan(sweep.assign(lpips_mean=np.inf), [1])


def test_top_r_report_includes_baseline_and_adjacent_slopes(tmp_path):
    frame = pd.DataFrame(dict(variant=['top_r_03', 'original', 'top_r_01'],
                             lpips_mean=[0.6, 0, 0.1],
                             clip_image_text_cosine_mean=[0.1, 0.3, 0.2],
                             clip_image_image_cosine_mean=[0.4, 1, 0.9]))
    table = report.write_top_r_report(frame, {'top_r_01': [2], 'top_r_03': [0, 1, 2]}, tmp_path)
    assert table.R.tolist() == [0, 1, 3]
    assert table.lpips_increase_per_block.iloc[1:].tolist() == [0.1, 0.25]
    assert (tmp_path / 'top_r_metrics.png').is_file()
    assert json.loads((tmp_path / 'largest_lpips_rise.json').read_text())['R'] == 3


def test_sweep_rules_and_top_r_end_to_end_with_tiny_models(tiny_run, monkeypatch):
    from types import SimpleNamespace
    import torch
    from team_pipeline import ablation, extended_metrics, runs
    from team_pipeline.evaluation import compare_results
    config = dict(device='cpu', image_size=8, cfg_scale=4, sampling_steps=2)
    pairs = [[1, 10], [2, 20]]
    protocol = {'splits': {'validation': pairs, 'calibration': [[1, 30], [2, 40]]}}
    monkeypatch.setattr(runs, 'load_run', lambda args: (tiny_run, protocol, config))
    monkeypatch.setattr(ablation, 'package_version', lambda package: 'test-encoder')
    profile = tiny_run / 'ablations/profile'
    profile.mkdir(parents=True)
    pd.DataFrame(dict(block=range(4), timestep=[9]*4, mean_cfg=[.1, .4, .2, .3])).to_csv(
        profile / 'feature_changes.csv', index=False)
    # Exercise real aggregation and pair checks with lightweight metric encoders.
    monkeypatch.setattr(ablation, 'compare_results', lambda root, names:
                        compare_results(root, names, metric=lambda x, y: (x-y).square().mean()))
    monkeypatch.setattr(extended_metrics, 'load_clip', lambda device: (
        lambda images: torch.stack([images.mean((1, 2, 3)), torch.ones(len(images))], dim=1),
        lambda prompts: torch.ones(len(prompts), 2), {'model': 'tiny'}))
    for study in ['sweep', 'rules', 'topk']:
        result = ablation.run_study(SimpleNamespace(study=study, split='validation', counts=[1, 2, 4]))
        assert {'lpips_mean', 'clip_image_text_cosine_mean',
                'clip_image_image_cosine_mean'} <= set(result.columns)
        assert result.loc[result.variant == 'original', 'lpips_mean'].item() == 0
    assert result.block_evaluations_per_forward.tolist() == [4, 3, 2, 0]
    ranking = json.loads((tiny_run / 'ablations/topk/ranking.json').read_text())
    assert ranking['source_split'] == 'validation'
    assert ranking['ranking'] == [0, 1, 2, 3]  # equal single-removal scores: stable ties
    assert (tiny_run / 'ablations/sweep/validation/analysis/block_ablation_heatmap.png').exists()


def test_missing_clip_dependency_fails_before_loading_run(monkeypatch):
    from types import SimpleNamespace
    from team_pipeline import ablation
    monkeypatch.setattr(ablation, 'package_version', lambda package: 'not-installed')
    with pytest.raises(RuntimeError, match='before preparing'):
        ablation.run_study(SimpleNamespace(study='sweep'))


@pytest.mark.parametrize('keep', [False, True])
def test_checkpoint_retention_is_honored_and_checked_on_resume(tiny_run, keep):
    import torch
    from team_pipeline.ablation import run_plan
    config = dict(device='cpu', image_size=8, cfg_scale=4, sampling_steps=2)
    pairs = [(1, 10), (2, 20)]
    output = tiny_run / 'checkpoint_study'
    names = run_plan(config, pairs, output, {'remove': [1]}, keep_checkpoints=keep)
    for name in names:
        path = output / name / 'model.pt'
        assert path.exists() is keep
        if keep:
            assert 'state_dict' in torch.load(path, weights_only=True)
    assert run_plan(config, pairs, output, {'remove': [1]}, keep_checkpoints=keep) == names
    if keep:
        (output / 'remove/model.pt').unlink()
        with pytest.raises(ValueError, match='checkpoint'):
            run_plan(config, pairs, output, {'remove': [1]}, keep_checkpoints=keep)


@pytest.mark.parametrize('deltas,before,after', [
    ([1, 2, 9, 3, 4], 1, 3),
    ([9, 2, 8, 3, 4], 1, 1),  # block 0 has no predecessor
    ([1, 8, 8, 3, 4], 0, 2),  # ties choose the lower eligible index
])
def test_before_and_after_max_delta_select_adjacent_blocks(deltas, before, after):
    from team_pipeline.ablation import single_removal_plan
    scores = pd.DataFrame(dict(block=range(len(deltas)), mean_feature_change=deltas))
    plan = single_removal_plan(scores)
    assert plan['B_before_max_delta'] == [before]
    assert plan['B_after_max_delta'] == [after]
