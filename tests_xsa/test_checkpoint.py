"""Compact checkpoint and frozen selection round trips on tiny models."""
import json

import pytest
import torch
from torch import nn

from xsa_method import (
    ScheduledBlocks, load_compact_checkpoint, load_fixed_selection,
    write_fixed_selection, run_method,
)


def test_xsa_checkpoint_and_fixed_selection(tmp_path, monkeypatch):
    import sys
    import types
    from timm.models.vision_transformer import Attention

    class SmallModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([nn.ModuleDict({"attn": Attention(12, 3)}) for _ in range(28)])

    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(DiT_XL_2=SmallModel))
    model = SmallModel()
    model.blocks = ScheduledBlocks(model.blocks, [4, 7], "xsa_reuse", 1.0, xsa_eps=1e-7)
    path = tmp_path / "compact.pt"
    torch.save(dict(schedule=model.blocks.metadata(), state_dict=model.state_dict()), path)
    loaded = load_compact_checkpoint(path)
    assert loaded.blocks.metadata() == model.blocks.metadata()
    assert len(loaded.blocks.retained) == 26 and len(loaded.blocks) == 28
    assert all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    selection = write_fixed_selection(tmp_path)
    assert selection["removed"] == [4, 7] and selection["index_base"] == 0
    assert load_fixed_selection(tmp_path) == selection
    # Verify the public entry point uses shared config/pairs and fixed indices.
    config, pairs = {"device": "cpu"}, [[207, 10016]]
    protocol = {"splits": {"validation": pairs}}
    run_args = object()
    load_run = lambda args: (tmp_path, protocol, config)
    calls = []

    def capture_run(config_arg, pairs_arg, output, **kwargs):
        calls.append((config_arg, pairs_arg, output, kwargs))

    for mode in ["removal", "loop", "xsa_reuse"]:
        run_method(run_args, "validation", mode, mode, alpha=1.0,
                   load_run=load_run, run_variant=capture_run)
        actual_config, actual_pairs, output, kwargs = calls[-1]
        assert actual_config is config and actual_pairs is pairs
        assert output == tmp_path / "validation" / mode
        assert kwargs["mode"] == mode and kwargs["removed"] == [4, 7]
        assert kwargs["alpha"] == 1.0 and kwargs["model_transform"] is None
    with pytest.raises(ValueError, match="alpha=1.0"):
        run_method(run_args, "validation", "xsa_reuse", "bad_alpha", alpha="inverse_k",
                   load_run=load_run, run_variant=capture_run)
    selection["removed"] = [3, 6]
    (tmp_path / "profile/selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="frozen"):
        write_fixed_selection(tmp_path)


