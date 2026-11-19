"""bind_runner routing and the example custom method, without downloads or globals."""
import inspect
import sys
import types

import pytest
from torch import nn

from team_pipeline import example_method, sampling
from xsa_method import ScheduledBlocks, bind_runner, write_fixed_selection

METHODS = [("original", "original", None), ("removal", "member1_removal", None),
           ("loop", "member2_loop", None), ("damped", "member3_damped", "inverse_k"),
           ("xsa_reuse", "member4_xsa_reuse", 1.0)]


@pytest.fixture
def bound(tmp_path):
    write_fixed_selection(tmp_path)
    protocol = {"splits": {"validation": [[1, 10]], "test": [[1, 11]]}}
    calls, run_args = [], object()

    def load_run(args):
        assert args is run_args
        return tmp_path, protocol, {"device": "cpu"}

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    run = bind_runner(run_args, load_run=load_run, run_variant=capture, keep_checkpoints=True)
    return run, calls, tmp_path


def test_routes_every_builtin_method_for_both_splits(bound):
    run, calls, root = bound
    for split in ("validation", "test"):
        for method, name, alpha in METHODS:
            run(split, method, name, **({} if alpha is None else {"alpha": alpha}))
            args, kwargs = calls[-1]
            assert args[2] == root / split / name
            assert kwargs["mode"] == method and kwargs["keep_checkpoint"] is True
            assert kwargs["removed"] == (() if method == "original" else [4, 7])
    assert calls[-1][0][1] == [[1, 11]]


def test_custom_builder_records_its_source_and_settings(bound):
    run, calls, _ = bound

    def builder(model, settings):
        return (model, settings)

    run("validation", "custom", "custom_run", builder=builder, settings={"alpha": 0.1})
    kwargs = calls[-1][1]
    assert kwargs["model_transform"]("m") == ("m", {"alpha": 0.1})
    assert kwargs["method_metadata"]["source"] == inspect.getsource(builder)
    assert kwargs["mode"] == "original" and kwargs["removed"] == ()


@pytest.mark.parametrize("split, method, name", [
    ("calibration", "loop", "member"), ("test", "loop", "../escape"),
    ("test", "loop", "original"), ("test", "original", "member"), ("test", "loop", ""),
])
def test_invalid_routes_are_rejected_before_any_work(bound, split, method, name):
    run, calls, _ = bound
    with pytest.raises(ValueError):
        run(split, method, name)
    assert not calls


def test_non_damped_methods_reject_a_residual_scale(bound):
    run, calls, _ = bound
    with pytest.raises(ValueError, match="alpha=1.0"):
        run("validation", "loop", "member", alpha=0.5)
    assert not calls


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = ScheduledBlocks([nn.Linear(1, 1) for _ in range(3)])


def test_example_builder_uses_the_notebook_schedule_and_default_alpha():
    model = example_method.build_model(Tiny(), {"removed": [1]})
    assert type(model.blocks) is ScheduledBlocks
    assert model.blocks.alpha == "inverse_k" and model.blocks.mode == "damped"
    assert len(model.blocks.retained) == 2


def test_custom_builder_keeps_shared_preparation_and_freezes_model(monkeypatch):
    vae = nn.Linear(1, 1)
    monkeypatch.setattr(sampling, "load_model", lambda *a: Tiny())
    monkeypatch.setattr(sampling, "_load_vae", lambda config: vae)
    monkeypatch.setitem(sys.modules, "diffusion",
                        types.SimpleNamespace(create_diffusion=lambda *a: "sampler"))
    seen = []

    def builder(model):
        seen.append(model)
        return example_method.build_model(model, {"removed": [1], "alpha": 0.25})

    config = dict(device="cpu", checkpoint="unused", vae_id="unused", sampling_steps=50)
    model, loaded_vae, diffusion = sampling._prepare(config, "original", (), 0.5, builder)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert len(model.blocks.retained) == 2 and model.blocks.alpha == 0.25
    assert loaded_vae is vae and diffusion == "sampler" and len(seen) == 1
    with pytest.raises(TypeError, match="nn.Module"):
        sampling._prepare(config, "original", (), 0.5, lambda m: None)
