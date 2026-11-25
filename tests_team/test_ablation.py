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
