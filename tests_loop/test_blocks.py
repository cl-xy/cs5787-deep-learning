"""LoopedBlocks contracts on tiny residual blocks; no pretrained downloads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from loop_compression.blocks import LoopedBlocks, gate_mask, make_gate
from loop_compression.method import build_model
from team_pipeline import experiment as exp
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
    effective = lambda m: exp._effective_block_evaluations(m.blocks)
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


# ------------------------------------------------------------ timestep schedule

SKIP = dict(t_min=0, t_max=499, K=1, strategy="damped", h=0.0)
TWICE = dict(t_min=500, t_max=999, K=2, strategy="naive")
WINDOW = dict(window=[3, 6], kept=[4, 5])


def timed(blocks, schedule, **extra):
    return LoopedBlocks(blocks, **WINDOW, schedule_by_t=schedule, **extra)


def run_at(looped, x, c, t):
    looped.set_timestep(torch.as_tensor(t))
    return run(looped, x, c)


def test_schedule_selects_the_rule_of_each_timestep(xc):
    x, c = xc
    blocks = make_blocks()
    looped = timed(blocks, [SKIP, TWICE])
    skipped = LoopedBlocks(blocks, **WINDOW, K=3, strategy="damped", h=0.0)
    naive2 = LoopedBlocks(blocks, **WINDOW, K=2, strategy="naive")
    assert torch.allclose(run_at(looped, x, c, [100, 100]), run(skipped, x, c), atol=1e-6)
    assert torch.allclose(run_at(looped, x, c, [800, 800]), run(naive2, x, c), atol=1e-6)
    assert looped.rule_for(499).h == 0.0 and looped.rule_for(500).strategy == "naive"


def test_constant_schedule_equals_the_unscheduled_container(xc):
    x, c = xc
    blocks = make_blocks()
    rule = dict(t_min=0, t_max=999, K=2, strategy="damped", h=0.5)
    plain = LoopedBlocks(blocks, **WINDOW, K=2, strategy="damped", h=0.5)
    scheduled = timed(blocks, [rule])
    for t in ([0, 0], [999, 999], [250, 700]):
        assert torch.allclose(run_at(scheduled, x, c, t), run(plain, x, c), atol=1e-6)


def test_mixed_timestep_batch_is_grouped_per_rule(xc):
    x, c = xc
    blocks = make_blocks()
    looped = timed(blocks, [SKIP, TWICE])
    mixed = run_at(looped, x, c, [100, 800])
    low = run_at(looped, x, c, [100, 100])
    high = run_at(looped, x, c, [800, 800])
    assert torch.allclose(mixed[0], low[0], atol=1e-6) and torch.allclose(mixed[1], high[1], atol=1e-6)
    assert not torch.allclose(low[0], high[0])


def test_uncovered_timesteps_use_the_default_rule(xc):
    x, c = xc
    blocks = make_blocks()
    looped = timed(blocks, [TWICE], K=1, strategy="naive")
    plain = LoopedBlocks(blocks, **WINDOW, K=1, strategy="naive")
    assert torch.allclose(run_at(looped, x, c, [10, 10]), run(plain, x, c), atol=1e-6)


def test_missing_timestep_raises_instead_of_guessing(xc):
    x, c = xc
    with pytest.raises(RuntimeError, match="attach"):
        run(timed(make_blocks(), [SKIP, TWICE]), x, c)
    with pytest.raises(ValueError, match="batch"):
        run_at(timed(make_blocks(), [SKIP, TWICE]), x, c, [1, 2, 3])


def test_schedule_validation():
    blocks = make_blocks()
    bad = [[], [SKIP, dict(TWICE, t_min=499)], [dict(SKIP, t_min=-1)], [dict(SKIP, bogus=1)],
           [{k: v for k, v in SKIP.items() if k != "K"}], [dict(SKIP, h=2.0)],
           [dict(TWICE, h=0.5)], ["x"], [dict(SKIP, t_min=True)]]
    for schedule in bad:
        with pytest.raises(ValueError):
            timed(blocks, schedule)


def test_schedule_keeps_parameters_and_reports_worst_case_flops_and_exact_mean():
    blocks = make_blocks()
    plain = LoopedBlocks(blocks, **WINDOW, K=1, strategy="naive")
    looped = timed(blocks, [SKIP, TWICE], K=1, strategy="naive")
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(looped) == count(plain)
    assert len(looped) == 6 + 2 * 2                      # worst case: the K=2 rule
    # Half of the steps skip the unit (0 calls), half run it twice (4 calls): mean = 6 + 2.
    assert looped.mean_block_evaluations([100, 200, 600, 900]) == pytest.approx(6 + (0 + 0 + 4 + 4) / 4)
    import json
    meta = looped.metadata()
    json.dumps(meta, allow_nan=False)
    assert [r["t_min"] for r in meta["schedule_by_t"]] == [0, 500]
    assert "schedule_by_t" not in plain.metadata()


def test_attach_feeds_the_timestep_through_t_embedder_and_detaches():
    class Embedder(nn.Module):
        def forward(self, t):
            return t.float()[:, None]

    class Model(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.t_embedder = Embedder()
            self.blocks = blocks

    model = Model(timed(make_blocks(), [SKIP, TWICE]))
    with model.blocks.attached(model):
        model.t_embedder(torch.tensor([7, 7]))
        assert model.blocks._t.tolist() == [7, 7]
    assert model.blocks._t is None
    model.t_embedder(torch.tensor([9, 9]))
    assert model.blocks._t is None  # hook removed


def test_build_model_attaches_only_for_scheduled_settings():
    class Embedder(nn.Module):
        def forward(self, t):
            return t

    model = SimpleNamespace(blocks=ScheduledBlocks(make_blocks()), t_embedder=Embedder())
    handles = []
    model.t_embedder.register_forward_pre_hook = lambda hook: handles.append(hook) or SimpleNamespace(remove=lambda: None)
    build_model(model, dict(window=[3, 6], kept=[4, 5], K=2, h=0.5))
    assert handles == []
    model.blocks = ScheduledBlocks(make_blocks())
    out = build_model(model, dict(window=[3, 6], kept=[4, 5], K=1, strategy="naive",
                                  schedule_by_t=[SKIP, TWICE]))
    assert len(handles) == 1 and out.blocks.timed


# ------------------------------------------------------------------ token gating

def gated(blocks, gate, **kw):
    base = dict(window=[3, 6], kept=[4, 5], K=2, strategy="naive")
    base.update(kw)
    return LoopedBlocks(blocks, gate=gate, **base)


def tokens_xc(tokens=16, seed=3):
    torch.manual_seed(seed)
    return torch.randn(2, tokens, D), torch.randn(2, 1, D)


def test_gate_extremes_reduce_to_the_ungated_loop_and_to_one_application():
    x, c = tokens_xc()
    blocks = make_blocks()
    for strategy, h in [("naive", None), ("damped", 0.5)]:
        plain = LoopedBlocks(blocks, window=[3, 6], kept=[4, 5], K=2, strategy=strategy, h=h)
        full = gated(blocks, dict(signal="update", keep_fraction=1.0), strategy=strategy, h=h)
        none = gated(blocks, dict(signal="update", keep_fraction=0.0), strategy=strategy, h=h)
        once = LoopedBlocks(blocks, window=[3, 6], kept=[4, 5], K=1, strategy=strategy, h=h if h else None)
        assert torch.allclose(run(full, x, c), run(plain, x, c), atol=1e-6)
        assert torch.allclose(run(none, x, c), run(once, x, c), atol=1e-6)


def test_gate_loops_exactly_the_selected_tokens():
    x, c = tokens_xc()
    blocks = make_blocks()
    looped = gated(blocks, dict(signal="update", keep_fraction=0.25))
    y = run(blocks[:3], x, c)
    unit = lambda z: run(blocks[4:6], z, c)
    y1 = unit(y)
    mask = gate_mask(y1 - y, make_gate(dict(signal="update", keep_fraction=0.25)))
    assert mask.sum().item() == 2 * 4                      # ceil(0.25 * 16) tokens per sample
    expected = torch.where(mask, unit(y1), y1)
    assert torch.allclose(run(looped, x, c), run(blocks[7:], expected, c), atol=1e-6)
    # The selected tokens are those with the largest first update (select=high) ...
    size = (y1 - y).norm(dim=-1)
    for b in range(2):
        assert set(size[b].topk(4).indices.tolist()) == set(mask[b, :, 0].nonzero().flatten().tolist())
    # ... and select=low picks the complement-end.
    low = gate_mask(y1 - y, make_gate(dict(signal="update", keep_fraction=0.25, select="low")))
    for b in range(2):
        assert set(size[b].topk(4, largest=False).indices.tolist()) == set(low[b, :, 0].nonzero().flatten().tolist())


def test_grad_and_random_signals_select_the_requested_count():
    update = torch.zeros(2, 16, D)
    update[:, :8] = 1.0                                      # a sharp edge between token rows 1 and 2
    grad = gate_mask(update, make_gate(dict(signal="grad", keep_fraction=0.25)))
    rows = grad[0, :, 0].reshape(4, 4).any(dim=1).tolist()
    assert grad.sum().item() == 2 * 4 and rows == [False, True, False, False]  # the edge row only
    rand = gate_mask(update, make_gate(dict(signal="random", keep_fraction=0.5)))
    assert rand.sum().item() == 2 * 8
    assert torch.equal(rand, gate_mask(update, make_gate(dict(signal="random", keep_fraction=0.5))))
    with pytest.raises(ValueError, match="square"):
        gate_mask(torch.zeros(1, 6, D), make_gate(dict(signal="grad", keep_fraction=0.5)))


def test_gate_validation_and_metadata():
    blocks = make_blocks()
    for bad in [dict(signal="x", keep_fraction=0.5), dict(signal="update"), dict(signal="update", keep_fraction=1.5),
                dict(signal="update", keep_fraction=0.5, select="mid"), dict(signal="update", keep_fraction=0.5, z=1),
                dict(signal="update", keep_fraction=True), "update"]:
        with pytest.raises(ValueError):
            gated(blocks, bad)
    with pytest.raises(ValueError, match="naive and damped"):
        gated(blocks, dict(signal="update", keep_fraction=0.5), strategy="anchored", beta=0.5, h=0.5)
    import json
    meta = gated(blocks, dict(signal="update", keep_fraction=0.5)).metadata()
    json.dumps(meta, allow_nan=False)
    assert meta["estimated_pruned_block_evaluations"] == 6 + 2 * (1 + 0.5)
    assert len(gated(blocks, dict(signal="update", keep_fraction=0.5)).schedule) == 6 + 2 * 2  # measured: not reduced
    assert "gate" not in LoopedBlocks(blocks, window=[3, 6], kept=[4, 5]).metadata()


def test_gate_inside_a_timestep_schedule(xc):
    x, c = tokens_xc()
    blocks = make_blocks()
    rule = dict(t_min=0, t_max=999, K=2, strategy="naive", gate=dict(signal="update", keep_fraction=0.5))
    looped = LoopedBlocks(blocks, **WINDOW, schedule_by_t=[rule])
    direct = gated(blocks, dict(signal="update", keep_fraction=0.5))
    looped.set_timestep(torch.tensor([10, 10]))
    assert torch.allclose(run(looped, x, c), run(direct, x, c), atol=1e-6)
    assert looped.metadata()["schedule_by_t"][0]["gate"]["keep_fraction"] == 0.5
