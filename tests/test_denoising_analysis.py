"""Numerical contracts for matched-latent diagnostics, without pretrained weights."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

def definitions():
    path = ROOT / "denoising_analysis.py"
    assert path.exists(), "Missing standalone denoising analysis"
    spec = importlib.util.spec_from_file_location("analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def forward(self, x, c):
        return x + self.value

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleList([Add(1), Add(2), Add(4)])
        self.inputs = []
    def forward_with_cfg(self, x, t, y, cfg_scale):
        self.inputs.append((x.clone(), t.clone(), y.clone()))
        for block in self.blocks:
            x = block(x, None)
        # Variance channels deliberately huge: must never enter epsilon MSE.
        return torch.cat([x, 1000*x], dim=1)

class Diffusion:
    def p_sample_loop(self, model, shape, noise, model_kwargs, **kwargs):
        x = noise
        for t in (999, 0):
            out = model(x, torch.full((2,), t, device=x.device), **model_kwargs)
            x = x + out[:, :4] * .01
        return x

def test_matched_inputs_recovery_sign_and_noise_channels(tmp_path):
    a = definitions()
    model = Model().eval()
    original = model.blocks
    r = a.run_denoising_analysis(model, Diffusion(), [(0, 1), (1, 2)],
        removed=[1], windows=[[0], [2]], loop_alpha=1., output=tmp_path)
    assert model.blocks is original
    np.testing.assert_allclose(r["ablation_mse"], np.broadcast_to([1.,4.,16.], (2,2,3)), atol=1e-4)
    np.testing.assert_allclose(r["prune_mse"], 4., atol=1e-4)
    np.testing.assert_allclose(r["recovery"], np.broadcast_to([3.,0.], (2,2,2)), atol=1e-4)
    np.testing.assert_array_equal(r["timesteps"], [999,0])
    # Each step has full + three ablations + prune + two equal-cost loops.
    for start in range(0, len(model.inputs), 7):
        for sample in model.inputs[start:start+7]:
            for actual, expected in zip(sample, model.inputs[start]):
                torch.testing.assert_close(actual, expected)
    assert (tmp_path / "per_pair.npz").exists()
    assert (tmp_path / "summary.csv").exists()

def test_negative_recovery_and_validation(tmp_path):
    a = definitions()
    r = a.run_denoising_analysis(Model().eval(), Diffusion(), [(0, 1)],
        removed=[0], windows=[[2]], loop_alpha=1., output=tmp_path)
    np.testing.assert_allclose(r["recovery"], -8., atol=1e-4)
    for windows in ([[0], [0,2]], [[1]], [[2,0]]):
        with pytest.raises(ValueError):
            a.analysis_schedules(3, [1], windows, 1.)

def test_schedule_restored_after_error():
    a = definitions()
    m = Model().eval()
    original = m.blocks
    with pytest.raises(RuntimeError):
        with a.analysis_blocks(m, [(0,1.)]):
            raise RuntimeError("test")
    assert m.blocks is original

def test_notebooks_embed_identical_standalone_definitions():
    import json
    a = definitions()
    for name in ("Damped", "Pruned"):
        n = json.loads((ROOT / f"DiT_Layer_Looping_{name}_colab.ipynb").read_text())
        cells = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
        assert (ROOT / "denoising_analysis.py").read_text() in cells


def test_real_respaced_sampler_trajectory_and_plot(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'third_party' / 'DiT'))
    from diffusion import create_diffusion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = definitions()
    diffusion = create_diffusion('3')
    model = Model().eval()
    # Bounded learned variance, while preserving the model's recorded inputs.
    base = model.forward_with_cfg
    def forward(x, t, **kwargs):
        out = base(x, t, **kwargs)
        return torch.cat([out[:, :4], torch.zeros_like(out[:, :4])], dim=1)
    model.forward_with_cfg = forward
    rng_before = torch.random.get_rng_state().clone()
    result = a.run_denoising_analysis(model, diffusion, [(0,1)], [1], [[0]], output=tmp_path)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    np.testing.assert_array_equal(result['timesteps'], [999,500,0])
    measured_inputs = [model.inputs[i][0] for i in range(0,len(model.inputs),6)]
    model.inputs.clear()
    z = torch.randn(1,4,32,32,generator=torch.Generator().manual_seed(1)).repeat(2,1,1,1)
    with a.analysis_rng(torch.device('cpu'), 1_000_001):
        diffusion.p_sample_loop(model.forward_with_cfg, z.shape, noise=z, clip_denoised=False,
            model_kwargs=dict(y=torch.tensor([0,1000]),cfg_scale=4.),device=torch.device('cpu'))
    for actual, reference in zip(measured_inputs, model.inputs):
        torch.testing.assert_close(actual, reference[0])
    figure = a.plot_denoising_analysis(result,tmp_path)
    assert (tmp_path/'denoising_heatmaps.png').is_file()
    assert (tmp_path/'denoising_heatmaps.pdf').is_file()
    assert figure.axes[1].images[0].norm(0.) == .5
    plt.close(figure)
