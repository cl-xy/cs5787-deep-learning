"""No-loop block ablation studies. Run ``python -m team_pipeline.ablation --help``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ablation_report import depth_partitions, write_ablation_report
from .evaluation import compare_results, _load_variant
from .extended_metrics import evaluate_fid_tpips
from .profiling import block_scores
from .provenance import shared_code_hash
from .runtime import write_json
from .sampling import calibrate, run_variant


def _feature_scores(scores):
    scores = scores.sort_values('block').reset_index(drop=True)
    depth = len(scores)
    if depth < 3 or scores.block.tolist() != list(range(depth)):
        raise ValueError('Feature scores must contain every block exactly once')
    if not np.isfinite(scores.mean_feature_change).all():
        raise ValueError('Feature scores must be finite')
    return scores


def single_removal_plan(scores):
    """R=1 for each rule; ties favor lower indices, C uses lower partition centers."""
    scores = _feature_scores(scores)
    minimum = scores.sort_values(['mean_feature_change', 'block']).iloc[0]
    predecessor = scores.iloc[:-1].sort_values(
        ['mean_feature_change', 'block'], ascending=[False, True]).iloc[0]
    plan = {'A_min_delta': [int(minimum.block)],
            'B_after_max_delta': [int(predecessor.block) + 1]}
    for name, group in zip(('early', 'middle', 'late'), depth_partitions(len(scores))):
        plan[f'C_{name}'] = [int(group[(len(group) - 1) // 2])]
    return plan


def _freeze(path, payload):
    """An existing experiment definition is immutable."""
    payload = json.loads(json.dumps(payload))
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'Frozen study changed; use a new run: {path}')
    else:
        write_json(path, payload)


def run_plan(config, pairs, output, plan, *, evidence=None):
    """Run intact reference and removal variants; resume only complete matching runs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not plan or any(Path(n).name != n or n in {'original', '.', '..', ''} for n in plan):
        raise ValueError('Use nonempty removal plan with safe, unique variant names')
    _freeze(output / 'plan.json', dict(config=config, pairs=pairs, variants=plan,
                                     evidence=evidence, shared_code_sha256=shared_code_hash()))
    for name, removed in {'original': [], **plan}.items():
        folder = output / name
        mode = 'original' if name == 'original' else 'removal'
        if folder.exists():
            manifest = json.loads((folder / 'manifest.json').read_text())
            expected_pairs = [list(p) for p in pairs]
            if (manifest['config'] != config or manifest['pairs'] != expected_pairs
                    or manifest['shared_code_sha256'] != shared_code_hash()
                    or manifest['schedule']['mode'] != mode
                    or manifest['schedule']['removed'] != sorted(removed)):
                raise ValueError(f'Existing variant differs from plan: {name}')
            baseline = pd.read_csv(output / 'original/samples.csv')
            reference = json.loads((output / 'original/manifest.json').read_text())
            samples, _ = _load_variant(output, name, baseline, reference)
            if any(not (folder / f'image_{i:04d}.{ext}').is_file()
                   for i in samples['index'] for ext in ('pt', 'png')):
                raise ValueError(f'Incomplete variant: {name}; move it aside before retrying')
            continue
        run_variant(config, pairs, folder, mode=mode, removed=removed, alpha=1.0,
                    method_metadata=dict(study='block_ablation', rule=name, looping=False))
    return ['original', *plan]


def evaluate_plan(output, names):
    """Matched LPIPS plus raw CLIP image–text and image–image cosines."""
    compare_results(output, names)
    return evaluate_fid_tpips(output, names, tpips_factors=(), enable_clip=True)


def study_profile(root, config, calibration_pairs):
    """Dedicated calibration artifacts do not overwrite the main method selection."""
    folder = Path(root) / 'ablations/profile'
    folder.mkdir(parents=True, exist_ok=True)
    _freeze(folder / 'inputs.json', dict(config=config, pairs=calibration_pairs,
                                       shared_code_sha256=shared_code_hash()))
    raw = folder / 'feature_changes.csv'
    frame = pd.read_csv(raw) if raw.exists() else calibrate(config, calibration_pairs, folder)
    matrix, scores, _ = block_scores(frame, remove_count=1)
    scores.to_csv(folder / 'block_scores.csv', index_label='block')
    matrix.to_csv(folder / 'feature_change_matrix.csv')
    return folder, scores.reset_index()


def run_study(args):
    """Notebook/CLI entry point using the prepared run's checked configuration."""
    from .runs import load_run
    root, protocol, config = load_run(args)
    if args.study == 'report':
        return write_ablation_report(root / 'ablations/profile', root / 'ablations/sweep/validation')
    profile, scores = study_profile(root, config, protocol['splits']['calibration'])
    if args.study == 'sweep':
        if args.split != 'validation':
            raise ValueError('Single-block ranking sweep uses validation only')
        plan = {f'remove_{i:02d}': [i] for i in range(len(scores))}
    else:
        plan = single_removal_plan(scores)
    output = root / 'ablations' / args.study / args.split
    names = run_plan(config, protocol['splits'][args.split], output, plan,
                     evidence=scores[['block', 'mean_feature_change']].to_dict('records'))
    result = evaluate_plan(output, names)
    if args.study == 'sweep':
        write_ablation_report(profile, output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=['sweep', 'rules', 'report'])
    parser.add_argument('--output', required=True, help='Prepared run root')
    parser.add_argument('--repo', default='third_party/DiT')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args(argv)
    print(run_study(args).to_string(index=False))


if __name__ == '__main__':
    main()
