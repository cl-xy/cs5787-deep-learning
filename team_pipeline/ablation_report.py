"""Block ablation plots from saved, matched single-block removal results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .evaluation import check_manifests_consistent


def load_block_ablation(profile_root, split_root):
    """Join by original block ID, requiring a complete, finite removal-only sweep."""
    root = Path(split_root)
    scores = pd.read_csv(Path(profile_root) / 'block_scores.csv')
    if scores.block.tolist() and sorted(scores.block.tolist()) != list(range(len(scores))):
        raise ValueError('Feature scores must contain every block exactly once')
    reference = json.loads((root / 'original/manifest.json').read_text())
    depth = reference['schedule']['depth']
    if reference['schedule']['mode'] != 'original' or reference['schedule']['removed']:
        raise ValueError('Reference must be the intact DiT')
    rows = []
    for row in pd.read_csv(root / 'results.csv').to_dict('records'):
        name = row['variant']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid variant name')
        manifest = json.loads((root / name / 'manifest.json').read_text())
        check_manifests_consistent(reference, manifest, name)
        schedule = manifest['schedule']
        if name == 'original':
            continue
        if schedule['mode'] != 'removal' or len(schedule['removed']) != 1:
            raise ValueError('Expected single-block removal without looping')
        rows.append(dict(block=schedule['removed'][0], variant=name,
                         lpips_mean=row['lpips_mean']))
    removed = pd.DataFrame(rows, columns=['block', 'variant', 'lpips_mean'])
    if (len(scores) != depth or sorted(removed.block.tolist()) != list(range(depth))):
        raise ValueError('Sweep must contain every block exactly once')
    frame = scores[['block', 'mean_feature_change']].merge(removed, on='block', validate='one_to_one')
    if not np.isfinite(frame[['mean_feature_change', 'lpips_mean']].to_numpy()).all():
        raise ValueError('Feature changes and LPIPS must be finite')
    return frame.sort_values('block').reset_index(drop=True)


def _save(fig, output, stem):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {ext: output / f'{stem}.{ext}' for ext in ('png', 'pdf')}
    for path in paths.values():
        fig.savefig(path, dpi=180)
    return paths


def plot_feature_change_vs_lpips(frame, output):
    """Each point is one block; LPIPS compares its removal with intact DiT."""
    fig = Figure(figsize=(8, 6), layout='constrained')
    ax = fig.subplots()
    ax.scatter(frame.mean_feature_change, frame.lpips_mean)
    for row in frame.itertuples():
        ax.annotate(str(row.block), (row.mean_feature_change, row.lpips_mean),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Mean normalized feature change Δ (calibration)',
           ylabel='Paired LPIPS vs intact DiT after removal',
           title='Single-block removal: feature change vs perceptual deviation')
    return _save(fig, output, 'feature_change_vs_lpips')


def depth_partitions(depth):
    """Three contiguous thirds; indivisible remainder goes to earlier partitions."""
    if not isinstance(depth, int) or depth < 3:
        raise ValueError('Depth must be an integer of at least three')
    return np.array_split(np.arange(depth), 3)


def _mark_partitions(ax, depth):
    groups = depth_partitions(depth)
    for name, group in zip(('early', 'middle', 'late'), groups):
        ax.text(float(group.mean()), 1.01, name, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom')
    for group in groups[:-1]:
        ax.axvline(group[-1] + 0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(-0.5, depth - 0.5)


def plot_block_lpips(frame, output):
    """Depth-wise removal LPIPS with early/middle/late boundaries."""
    frame = frame.sort_values('block')
    fig = Figure(figsize=(11, 5), layout='constrained')
    ax = fig.subplots()
    ax.plot(frame.block, frame.lpips_mean, 'o-')
    ax.set(xlabel='Block removed (zero based)', ylabel='Paired LPIPS vs intact DiT')
    ax.set_xticks(frame.block)
    _mark_partitions(ax, len(frame))
    return _save(fig, output, 'block_removal_lpips')


def plot_block_ablation_heatmap(frame, output):
    """Align feature change and removal LPIPS; separate color scales retain units."""
    frame = frame.sort_values('block')
    fig = Figure(figsize=(15, 5), layout='constrained')
    axes = fig.subplots(2, 1)
    for ax, column, label in zip(axes, ('mean_feature_change', 'lpips_mean'),
                                  ('Normalized feature change Δ', 'Removal paired LPIPS')):
        values = frame[column].to_numpy()[None, :]
        image = ax.imshow(values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks([0], [label])
        ax.set_xticks(range(len(frame)), frame.block)
        for index, value in enumerate(values[0]):
            ax.text(index, 0, f'{value:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if image.norm(value) > 0.6 else 'black')
        _mark_partitions(ax, len(frame))
        fig.colorbar(image, ax=ax, label=label)
    axes[-1].set_xlabel('Block removed (zero based)')
    return _save(fig, output, 'block_ablation_heatmap')


def write_ablation_report(profile_root, split_root):
    """Export the joined table and all three requested plots without inference."""
    frame = load_block_ablation(profile_root, split_root)
    output = Path(split_root) / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'block_ablation.csv', index=False)
    for plot in (plot_feature_change_vs_lpips, plot_block_lpips, plot_block_ablation_heatmap):
        plot(frame, output)
    return frame
