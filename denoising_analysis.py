# 10A — standalone matched-latent analysis definitions (no pipeline prerequisites).
from contextlib import contextmanager
from pathlib import Path
import json
from time import perf_counter
import numpy as np
import torch
from torch import nn


class AnalysisBlocks(nn.Module):
    """Temporary execution order; weights are shared with the full model."""
    def __init__(self, blocks, schedule):
        super().__init__()
        self.original = nn.ModuleList(blocks)
        self.schedule = schedule

    def __iter__(self):
        for index, scale in self.schedule:
            block = self.original[index]
            if scale == 1.:
                yield block
            else:
                def damped(x, c, block=block, scale=scale):
                    return x + scale * (block(x, c) - x)
                yield damped


@contextmanager
def analysis_blocks(model, schedule):
    original = model.blocks
    try:
        model.blocks = AnalysisBlocks(list(original), schedule)
        yield
    finally:
        model.blocks = original


def analysis_schedules(depth, removed, windows, loop_alpha):
    def check(indices):
        return (indices and all(type(i) is int and 0 <= i < depth for i in indices)
                and indices == list(range(indices[0], indices[-1] + 1)))
    removed = list(removed)
    windows = [list(w) for w in windows]
    if not check(removed) or not windows or not 0 < loop_alpha <= 1:
        raise ValueError('Require a contiguous removed group, candidate windows, and 0 < alpha <= 1')
    width = len(windows[0])
    if any(not check(w) or len(w) != width or set(w) & set(removed) for w in windows):
        raise ValueError('Loop windows must be contiguous, equally wide, and contain retained blocks only')
    if len({tuple(w) for w in windows}) != len(windows):
        raise ValueError('Duplicate loop windows')
    prune = [(i, 1.) for i in range(depth) if i not in removed]
    loops = []
    for window in windows:
        schedule = []
        for i, scale in prune:
            schedule.append((i, scale))
            if i == window[-1]:
                schedule.extend((j, float(loop_alpha)) for j in window)
        loops.append(schedule)
    return prune, loops


@contextmanager
def analysis_rng(device, seed):
    # Restore the caller's RNG state, including MPS (not covered by CPU fork_rng).
    mps_state = torch.mps.get_rng_state() if device.type == 'mps' else None
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            if device.type == 'mps':
                torch.mps.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


@torch.inference_mode()
def run_denoising_analysis(model, diffusion, pairs, removed, windows, *,
                           loop_alpha=1., cfg_scale=4., output=None):
    """Probe every full-model ancestral sampling call, then advance using full only.

    MSE covers the first sample's four epsilon channels in forward_with_cfg;
    learned variance is excluded. Official DiT guides channels 0..2 and leaves
    channel 3 conditional. The duplicated unconditional sample is not averaged in.
    Rows are denoising CALLS, with actual original training timesteps captured
    inside SpacedDiffusion's model callback (no double timestep mapping).
    """
    if model.training:
        raise ValueError('Use model.eval() so dropout cannot confound comparisons')
    blocks = list(model.blocks)
    # Refuse a previously pruned/replayed schedule as the reference model.
    if not isinstance(model.blocks, nn.ModuleList):
        raise ValueError('Load a fresh full DiT with its original ModuleList blocks')
    depth = len(blocks)
    prune, loops = analysis_schedules(depth, removed, windows, loop_alpha)
    pairs = [tuple(p) for p in pairs]
    if not pairs or any(len(p) != 2 or type(p[0]) is not int or not 0 <= p[0] < 1000
                        or type(p[1]) is not int or not 0 <= p[1] < 2**63-1_000_000 for p in pairs):
        raise ValueError('pairs must contain (ImageNet class 0..999, nonnegative integer seed)')
    if len(set(pairs)) != len(pairs):
        raise ValueError('Use distinct class-seed pairs')
    if not np.isfinite(cfg_scale):
        raise ValueError('cfg_scale must be finite')
    device = next(model.parameters()).device
    ablations, prunes, loop_errors, all_times = [], [], [], []
    started = perf_counter()
    for pair_index, (label, seed) in enumerate(pairs):
        rows_a, rows_p, rows_l, times = [], [], [], []
        z = torch.randn(1, 4, 32, 32, generator=torch.Generator(device='cpu').manual_seed(seed)).to(device)
        z = torch.cat([z, z], dim=0)
        y = torch.tensor([label, 1000], device=device)

        def probe(x, t, **kwargs):
            full = model.forward_with_cfg(x, t, **kwargs)
            epsilon = full[:1, :4].float()

            def error(schedule):
                with analysis_blocks(model, schedule):
                    prediction = model.forward_with_cfg(x, t, **kwargs)[:1, :4].float()
                return (prediction - epsilon).square().mean()

            rows_a.append(torch.stack([error([(i, 1.) for i in range(depth) if i != skipped])
                                       for skipped in range(depth)]).cpu())
            rows_p.append(error(prune).cpu())
            rows_l.append(torch.stack([error(schedule) for schedule in loops]).cpu())
            times.append(int(t[0].item()))
            return full  # Only the FULL prediction advances the common trajectory.

        with analysis_rng(device, seed + 1_000_000):
            diffusion.p_sample_loop(probe, z.shape, noise=z, clip_denoised=False,
                                    model_kwargs=dict(y=y, cfg_scale=cfg_scale),
                                    progress=False, device=device)
        ablations.append(torch.stack(rows_a).numpy())
        prunes.append(torch.stack(rows_p).numpy())
        loop_errors.append(torch.stack(rows_l).numpy())
        all_times.append(times)
        elapsed = perf_counter() - started
        remaining = elapsed / (pair_index + 1) * (len(pairs) - pair_index - 1)
        print(f'Analysis pair {pair_index + 1}/{len(pairs)}: class={label}, seed={seed}, '
              f'calls={len(times)} | elapsed={elapsed/60:.1f} min | estimated remaining={remaining/60:.1f} min',
              flush=True)
    if any(times != all_times[0] for times in all_times):
        raise ValueError('Timestep schedule differs between pairs')
    result = dict(ablation_mse=np.stack(ablations), prune_mse=np.stack(prunes),
                  loop_mse=np.stack(loop_errors), timesteps=np.array(all_times[0]))
    result['recovery'] = result['prune_mse'][:, :, None] - result['loop_mse']
    if any(not np.isfinite(v).all() for v in result.values()):
        raise RuntimeError('Nonfinite prediction errors; results not saved')
    result['metadata'] = dict(pairs=pairs, removed=list(removed), windows=[list(w) for w in windows],
        loop_alpha=float(loop_alpha), cfg_scale=float(cfg_scale), depth=depth,
        trajectory='full-model ancestral p_sample_loop',
        metric='MSE of forward_with_cfg sample 0, epsilon channels 0:4; variance excluded',
        replay='one ordered extra pass immediately after the window first executes',
        extra_block_calls=len(windows[0]), torch_version=str(torch.__version__))
    if output is not None:
        save_denoising_analysis(result, output)
    return result


def save_denoising_analysis(result, output):
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'per_pair.npz', **{k: v for k, v in result.items() if k != 'metadata'})
    (output / 'metadata.json').write_text(json.dumps(result['metadata'], indent=2))
    rows = []
    for metric in ('ablation_mse', 'prune_mse', 'loop_mse', 'recovery'):
        values = result[metric]
        if values.ndim == 2:
            values = values[:, :, None]
        for call, timestep in enumerate(result['timesteps']):
            for column in range(values.shape[2]):
                data = values[:, call, column]
                candidate = (str(column) if metric == 'ablation_mse' else
                             str(result['metadata']['windows'][column]) if metric != 'prune_mse' else 'fixed removed group')
                rows.append(dict(metric=metric, call=call+1, timestep=int(timestep), candidate=candidate,
                                 mean=float(data.mean()), std=float(data.std(ddof=1)) if len(data)>1 else 0.,
                                 pairs=len(data)))
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)


def plot_denoising_analysis(result, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    mean_a = result['ablation_mse'].mean(axis=0)
    mean_r = result['recovery'].mean(axis=0)
    mean_p = result['prune_mse'].mean(axis=0)[:, None]
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), gridspec_kw={'width_ratios': [2, 2, .55]}, constrained_layout=True)
    limit = max(float(np.abs(mean_r).max()), 1e-12)
    panels = [(mean_a, 'Single-block bypass: mean epsilon MSE', 'inferno', dict(vmin=0)),
              (mean_r, 'Recovery: prune MSE − prune+loop MSE', 'RdBu', dict(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))),
              (mean_p, 'Prune-only error', 'inferno', dict(vmin=0))]
    ticks = np.unique(np.linspace(0, len(mean_a)-1, min(10, len(mean_a)), dtype=int))
    for ax, (values, title, cmap, options) in zip(axes, panels):
        im = ax.imshow(values, origin='upper', aspect='auto', interpolation='nearest', cmap=cmap, **options)
        ax.set_title(title)
        ax.set_yticks(ticks, [f'{i+1} (t={result["timesteps"][i]})' for i in ticks])
        ax.set_ylabel('Denoising call: high noise (top) → low noise (bottom)')
        fig.colorbar(im, ax=ax, label='Mean epsilon MSE' if ax is not axes[1] else 'Recovery (positive = closer)')
    axes[0].set_xticks(range(mean_a.shape[1]))
    axes[0].set_xlabel('Bypassed block index')
    labels = ['–'.join(map(str, (w[0], w[-1]))) if len(w)>1 else str(w[0]) for w in result['metadata']['windows']]
    axes[1].set_xticks(range(len(labels)), labels, rotation=90)
    axes[1].set_xlabel('Candidate loop window (inclusive; equal extra block calls)')
    axes[2].set_xticks([0], [str(result['metadata']['removed'])], rotation=90)
    axes[2].set_xlabel('Fixed removed group')
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output) / 'denoising_heatmaps.png', dpi=180)
        fig.savefig(Path(output) / 'denoising_heatmaps.pdf')
    return fig
