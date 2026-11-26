"""schedule_cls plumbing, deterministic seeding and artifact writing in sampling.py."""
import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from team_pipeline import methods, sampling
from team_pipeline.data import SAMPLER_SEED_OFFSET
from xsa_method import ScheduledBlocks as XsaScheduledBlocks


class Stub(nn.Module):
    """Stand-in for the official DiT_XL_2 with four tiny blocks."""
    def __init__(self, input_size=32, num_classes=1000):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(4))


@pytest.fixture
def stub_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "stub.pt"
    torch.save(Stub().state_dict(), path)
    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(DiT_XL_2=Stub))
    return path


def test_load_model_defaults_to_cli_schedule_and_alpha(stub_checkpoint):
    model = sampling.load_model(stub_checkpoint, (1,), "damped")
    assert isinstance(model.blocks, methods.ScheduledBlocks)
    assert model.blocks.alpha == methods.DEFAULT_ALPHA == 0.5
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_load_model_applies_schedule_cls_and_inverse_k(stub_checkpoint):
    model = sampling.load_model(stub_checkpoint, (1,), "damped", "inverse_k", XsaScheduledBlocks)
    assert type(model.blocks) is XsaScheduledBlocks
    assert model.blocks.alpha == "inverse_k"
    assert len(model.blocks.retained) == 3 and len(model.blocks) == 4


def test_cli_schedule_equals_xsa_schedule_for_the_same_settings():
    blocks = [nn.Linear(1, 1) for _ in range(6)]
    for mode, alpha in [("removal", 1.0), ("loop", 1.0), ("damped", 0.25), ("damped", "inverse_k")]:
        base = XsaScheduledBlocks(blocks, (2, 4), mode, alpha)
        cli = methods.ScheduledBlocks(blocks, (2, 4), mode, alpha)
        assert cli.metadata() == base.metadata()


def test_run_variant_passes_schedule_cls_to_prepare(tmp_path, monkeypatch):
    seen = {}
    vae = nn.Linear(1, 1)
    vae.config = SimpleNamespace()

    def prepare(config, mode, removed, alpha, model_transform=None, schedule_cls=None):
        seen["schedule_cls"] = schedule_cls
        model = nn.Module()
        model.blocks = schedule_cls([nn.Linear(1, 1) for _ in range(3)], removed, mode, alpha)
        return model, vae, None

    monkeypatch.setattr(sampling, "_prepare", prepare)
    monkeypatch.setattr(sampling, "warmup", lambda *a: None)
    monkeypatch.setattr(sampling, "generate", lambda *a: torch.zeros(1, 3, 8, 8))
    config = dict(device="cpu", image_size=8, cfg_scale=4, sampling_steps=2)
    frame = sampling.run_variant(config, [(1, 10), (2, 20)], tmp_path / "run", mode="damped",
                                 removed=(1,), alpha="inverse_k",
                                 schedule_cls=XsaScheduledBlocks)
    assert seen["schedule_cls"] is XsaScheduledBlocks
    run = tmp_path / "run"
    for name in ("manifest.json", "samples.csv", "grid.png", "image_0000.pt", "image_0001.png"):
        assert (run / name).exists(), name
    assert not (run / "model.pt").exists()
    assert len(frame) == 2 and pd.read_csv(run / "samples.csv")["seed"].tolist() == [10, 20]


class Recorder:
    def __init__(self):
        self.draw = None

    def p_sample_loop(self, forward, shape, noise, **kwargs):
        self.draw = torch.rand(1).item()
        return noise


def generate_draw(seed, **kwargs):
    model = nn.Linear(1, 1)
    vae = SimpleNamespace(decode=lambda z: SimpleNamespace(sample=torch.zeros(1, 3, 8, 8)))
    model.forward_with_cfg = None
    diffusion = Recorder()
    sampling.generate(model, vae, diffusion, 3, seed, **kwargs)
    return diffusion.draw


def test_generate_seeds_sampler_noise_with_the_protocol_offset():
    torch.manual_seed(SAMPLER_SEED_OFFSET + 7)
    expected = torch.rand(1).item()
    assert generate_draw(7) == expected
    assert generate_draw(7, seed_offset=5) != expected
    torch.manual_seed(12)
    assert generate_draw(7, seed_offset=5) == generate_draw(7, seed_offset=5)


def test_checkpoint_measurement_does_not_write_or_replace_files(tmp_path):
    """Disabled checkpoint retention must work without storage for a checkpoint."""
    model = Stub()
    path = tmp_path / 'existing.pt'
    path.write_bytes(b'preserve existing artifact')
    measured = sampling.checkpoint_stats(model, path, keep=False)
    assert path.read_bytes() == b'preserve existing artifact'
    saved = tmp_path / 'saved.pt'
    retained = sampling.checkpoint_stats(model, saved, keep=True)
    assert measured['checkpoint_bytes'] == retained['checkpoint_bytes'] == saved.stat().st_size
    assert measured['parameter_bytes'] == 32
    restored = torch.load(saved, weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored['state_dict'][key], value)


def test_checkpoint_measurement_needs_no_output_directory(tmp_path):
    path = tmp_path / 'not-created/model.pt'
    result = sampling.checkpoint_stats(Stub(), path, keep=False)
    assert result['checkpoint_bytes'] > 32
    assert not path.parent.exists()
