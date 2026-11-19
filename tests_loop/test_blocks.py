"""LoopedBlocks contracts on tiny residual blocks; no pretrained downloads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from loop_compression.blocks import LoopedBlocks
from loop_compression.method import build_model
from team_pipeline import experiment as exp
from team_pipeline import sampling
from team_pipeline.methods import ScheduledBlocks

D = 8


class Res(nn.Module):
    """x + 0.4 tanh(Wx + c): residual, nonlinear, takes (x, c) like DiTBlock."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.mlp = SimpleNamespace(fc1=SimpleNamespace(out_features=16))

    def forward(self, x, c):
        return x + 0.4 * torch.tanh(self.lin(x) + c)


def make_blocks(n=10, seed=0):
    torch.manual_seed(seed)
    return [Res().eval() for _ in range(n)]


def run(blocks, x, c):
    for block in blocks:
        x = block(x, c)
    return x


@pytest.fixture
def xc():
    torch.manual_seed(1)
    return torch.randn(2, 5, D), torch.randn(2, 1, D)


def test_k1_h1_whole_window_is_the_original(xc):
    x, c = xc
    blocks = make_blocks()
    for strategy, iteration in [("naive", "block"), ("damped", "block"), ("naive", "layer")]:
        h = 1.0 if strategy == "damped" else None
        looped = LoopedBlocks(blocks, [3, 6], K=1, strategy=strategy, iteration=iteration, h=h)
        assert torch.allclose(run(looped, x, c), run(blocks, x, c), atol=1e-6)


def test_h0_equals_removing_the_whole_window(xc):
    x, c = xc
    blocks = make_blocks()
    looped = LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=3, strategy="damped", h=0.0)
    expected = run(blocks[:3] + blocks[7:], x, c)
    assert torch.allclose(run(looped, x, c), expected, atol=1e-6)
    assert not [e for e in looped.schedule if e[2]]


def test_parameters_match_removal_of_discarded_blocks_and_state_dict_is_unique():
    blocks = make_blocks()
    looped = LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=2, h=0.5)
    removal = ScheduledBlocks(blocks, removed=[3, 6], mode="removal")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(looped) == count(removal)
    assert looped.discarded == (3, 6)
    keys = {k.split(".")[1] for k in looped.state_dict()}
    assert keys == {str(i) for i in [0, 1, 2, 4, 5, 7, 8, 9]}
    assert [int(i) for i in looped.retained] == [0, 1, 2, 4, 5, 7, 8, 9]


def test_schedule_and_flops_scale_with_kept_blocks_times_loops():
    class Model(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.in_channels, self.out_channels, self.num_heads = 4, 8, 2
            self.pos_embed = nn.Parameter(torch.zeros(1, 4, D))
            self.x_embedder = SimpleNamespace(num_patches=4, patch_size=(2, 2))
            self.blocks = blocks

    blocks = make_blocks()
    effective = lambda m: sampling._effective_block_evaluations(m.blocks)
    base = Model(LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=1, strategy="naive"))
    twice = Model(LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=2, strategy="naive"))
    layer = Model(LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=3, strategy="damped", iteration="layer"))
    assert effective(base) == 6 + 2 * 1
    assert effective(twice) == 6 + 2 * 2
    assert effective(layer) == 6 + 2 * 3
    f = lambda m: exp.estimate_dit_flops(m, sampling_steps=1)["dit_forward_flops"]
    # base -> twice adds 2 block calls; twice -> layer (4 -> 6 calls) adds 2 more.
    two_calls = f(twice) - f(base)
    assert two_calls > 0 and f(layer) - f(twice) == two_calls
    assert len(twice.blocks) == effective(twice)
    # Zero-scale (h=0) loops add no FLOPs relative to removing the whole window.
    zero = Model(LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=4, strategy="damped", h=0))
    assert effective(zero) == 6


def test_damped_matches_the_euler_formula(xc):
    x, c = xc
    blocks = make_blocks()
    looped = LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=2, strategy="damped", h=0.5)
    y = run(blocks[:3], x, c)
    for _ in range(2):
        y = y + 0.5 * (run(blocks[4:6], y, c) - y)
    assert torch.allclose(run(looped, x, c), run(blocks[7:], y, c), atol=1e-6)


def test_naive_applies_the_kept_segment_k_times(xc):
    x, c = xc
    blocks = make_blocks()
    looped = LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=2, strategy="naive")
    expected = run(blocks[:3] + blocks[4:6] * 2 + blocks[7:], x, c)
    assert torch.allclose(run(looped, x, c), expected, atol=1e-6)


def test_layer_mode_repeats_each_layer_before_the_next(xc):
    x, c = xc
    blocks = make_blocks()
    looped = LoopedBlocks(blocks, [3, 6], kept=[4, 5], K=2, strategy="naive", iteration="layer")
    expected = run(blocks[:3] + [blocks[4]] * 2 + [blocks[5]] * 2 + blocks[7:], x, c)
    assert torch.allclose(run(looped, x, c), expected, atol=1e-6)
    assert [e[0] for e in looped.schedule if e[2]] == [4, 4, 5, 5]


def test_anchored_endpoints(xc):
    x, c = xc
    blocks = make_blocks()
    kw = dict(window=[3, 6], kept=[4, 5], K=3, h=1 / 3)
    single = run(blocks[:3] + blocks[4:6] + blocks[7:], x, c)
    anchored1 = LoopedBlocks(blocks, strategy="anchored", beta=1.0, **kw)
    assert torch.allclose(run(anchored1, x, c), single, atol=1e-6)
    anchored0 = LoopedBlocks(blocks, strategy="anchored", beta=0.0, **kw)
    damped = LoopedBlocks(blocks, strategy="damped", **kw)
    assert torch.allclose(run(anchored0, x, c), run(damped, x, c), atol=1e-6)
    # Anchor reuses the first evaluation: still K evaluations of the kept segment.
    assert len([e for e in anchored0.schedule if e[2]]) == 3 * 2


def test_default_step_is_one_over_k_and_validation():
    blocks = make_blocks()
    assert LoopedBlocks(blocks, [3, 6], K=4).h == 0.25
    assert LoopedBlocks(blocks, [3, 6], strategy="naive").h == 1.0
    bad = [dict(window=[3]), dict(window=[6, 3]), dict(window=[3, 10]),
           dict(window=[3, 6], kept=[]), dict(window=[3, 6], kept=[2]),
           dict(window=[3, 6], kept=[5, 4]), dict(window=[3, 6], K=0),
           dict(window=[3, 6], h=1.5), dict(window=[3, 6], h=float("nan")),
           dict(window=[3, 6], strategy="naive", h=0.5), dict(window=[3, 6], beta=0.5),
           dict(window=[3, 6], strategy="x"), dict(window=[3, 6], iteration="x"),
           dict(window=[3, 6], strategy="anchored", beta=2.0)]
    for kwargs in bad:
        with pytest.raises(ValueError):
            LoopedBlocks(blocks, **kwargs)


def test_metadata_is_json_serializable():
    import json
    meta = LoopedBlocks(make_blocks(), [3, 6], kept=[4, 5], K=2, h=0.5).metadata()
    json.dumps(meta, allow_nan=False)
    assert meta["removed"] == [3, 6] and meta["kept"] == [4, 5]


def test_build_model_swaps_blocks_and_rejects_unknown_settings():
    model = SimpleNamespace(blocks=ScheduledBlocks(make_blocks()))
    out = build_model(model, dict(window=[3, 6], kept=[4, 5], K=2, h=0.5))
    assert isinstance(out.blocks, LoopedBlocks) and len(out.blocks.retained) == 8
    with pytest.raises(ValueError, match="Unknown"):
        build_model(SimpleNamespace(blocks=ScheduledBlocks(make_blocks())), dict(window=[3, 6], bogus=1))
    with pytest.raises(ValueError, match="window"):
        build_model(SimpleNamespace(blocks=ScheduledBlocks(make_blocks())), {})
