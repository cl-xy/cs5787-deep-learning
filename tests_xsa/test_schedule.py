"""Weight sharing, replacement-only XSA, and inverse-k damping."""
import torch
from torch import nn

from xsa_method import ScheduledBlocks, xsa_filter_heads
from xsa_method import attention


def test_xsa_schedule_and_registration(monkeypatch):
    from timm.models.vision_transformer import Attention

    class ProbeBlock(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.attn = Attention(12, num_heads=3, qkv_bias=True)
            self.inputs = []

        def forward(self, x, c):
            self.inputs.append((x.detach().clone(), c))
            return x + self.attn(x) + c[:, None, :]

    with torch.random.fork_rng():
        torch.manual_seed(186)
        blocks = [ProbeBlock(i) for i in range(28)]
        schedule = ScheduledBlocks(blocks, [4, 7], "xsa_reuse", alpha=1.0)
        schedule.eval().requires_grad_(False)
        expected_sources = [0, 1, 2, 3, 3, 5, 6, 6, *range(8, 28)]
        metadata = schedule.metadata()
        assert [s["source"] for s in metadata["execution_schedule"]] == expected_sources
        assert [s["slot"] for s in metadata["execution_schedule"] if s["xsa"]] == [4, 7]
        assert metadata["index_base"] == 0 and metadata["retained"] == [i for i in range(28) if i not in [4, 7]]
        assert schedule.retained["3"] is blocks[3] and schedule.retained["6"] is blocks[6]
        retained_ids = {id(p) for b in blocks if b.index not in [4, 7] for p in b.parameters()}
        discarded_ids = {id(p) for b in [blocks[4], blocks[7]] for p in b.parameters()}
        registered_ids = {id(p) for p in schedule.parameters()}
        assert registered_ids == retained_ids and not registered_ids & discarded_ids
        assert len(schedule.retained) == 26 and len(schedule) == 28
        assert all(not p.requires_grad for p in schedule.parameters())
        assert not any(key.split(".")[1] in {"4", "7"} for key in schedule.state_dict())

        events, qkv_inputs, ordinary_outputs = [], {3: [], 6: []}, {}
        original_filter = xsa_filter_heads

        def record_filter(o, v, eps=1e-8):
            events.append(active_slot[0])
            return original_filter(o, v, eps)

        monkeypatch.setattr(attention, "xsa_filter_heads", record_filter)
        handles = []
        for index in [3, 6]:
            handles.append(blocks[index].attn.qkv.register_forward_pre_hook(
                lambda module, args, index=index: qkv_inputs[index].append(args[0].detach().clone())))
        x, c = torch.randn(1, 5, 12) * 0.01, torch.randn(1, 12) * 0.01
        active_slot = [None]
        try:
            for slot, call in enumerate(schedule):
                active_slot[0] = slot
                prior = x.clone()
                x = call(x, c)
                if slot in [3, 6]:
                    ordinary_outputs[slot] = x.clone()
                # Each additional call receives the immediately preceding output.
                if slot in [4, 7]:
                    source = 3 if slot == 4 else 6
                    torch.testing.assert_close(blocks[source].inputs[-1][0], prior)
                    assert blocks[source].inputs[-1][1] is c
            assert events == [4, 7]
            assert torch.isfinite(x).all()
            for index in [3, 6]:
                assert len(qkv_inputs[index]) == 2
                assert not torch.equal(*qkv_inputs[index])
                # The ordinary retained call remains the exact original block path.
                input_x, input_c = blocks[index].inputs[0]
                ordinary = blocks[index](input_x, input_c)
                torch.testing.assert_close(ordinary, ordinary_outputs[index], atol=0, rtol=0)
        finally:
            for handle in handles:
                handle.remove()


def test_inverse_k_damping():
    class AddBlock(nn.Module):
        def forward(self, x, c):
            return x + 1

    def evaluate(schedule):
        x = torch.zeros(1)
        for call in schedule:
            x = call(x, None)
        return x

    schedule = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "damped")
    entries = schedule.metadata()["execution_schedule"]
    assert [entry["source"] for entry in entries] == [0, 0, 0, 3, 3, 5]
    assert [entry["iteration"] for entry in entries] == [1, 2, 3, 1, 2, 1]
    assert [entry["scale"] for entry in entries] == [1.0, 0.5, 1 / 3, 1.0, 0.5, 1.0]
    assert schedule.metadata()["damping_policy"] == "inverse_k"
    assert schedule.metadata()["replacement_residual_scale"] is None
    torch.testing.assert_close(evaluate(schedule), torch.tensor([13 / 3]))
    # Calling a forward again starts from the same precomputed k values.
    torch.testing.assert_close(evaluate(schedule), evaluate(schedule), atol=0, rtol=0)
    fixed = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "damped", alpha=0.25)
    assert fixed.metadata()["damping_policy"] == "fixed"
    torch.testing.assert_close(evaluate(fixed), torch.tensor([3.75]))
    ordinary = ScheduledBlocks([AddBlock() for _ in range(6)], [1, 2, 4], "loop")
    torch.testing.assert_close(evaluate(ordinary), torch.tensor([6.0]))


