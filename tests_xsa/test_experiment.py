"""Runner routing without model downloads or notebook globals."""
import json

import pytest

from xsa_method import run_method, write_fixed_selection


def test_routes_methods_and_custom_builder_with_explicit_dependencies(tmp_path):
    write_fixed_selection(tmp_path)
    pairs = [[207, 10016]]
    protocol = {"splits": {"validation": pairs, "test": [[207, 10017]]}}
    config, run_args, calls = {"device": "cpu"}, object(), []

    def load_run(args):
        assert args is run_args
        return tmp_path, protocol, config

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    def run(split, method, name, **kwargs):
        return run_method(run_args, split, method, name,
                          load_run=load_run, run_variant=capture, **kwargs)

    for method in ["original", "removal", "loop", "damped", "xsa_reuse"]:
        assert run("validation", method, method, keep_checkpoints=True) == "result"
        args, kwargs = calls[-1]
        assert args == (config, pairs, tmp_path / "validation" / method)
        assert kwargs["mode"] == method
        assert kwargs["removed"] == (() if method == "original" else [4, 7])
        assert kwargs["alpha"] == ("inverse_k" if method == "damped" else 1.0)
        assert kwargs["keep_checkpoint"] is True
    run("test", "damped", "fixed", alpha=0.25)
    assert calls[-1][0][1] == protocol["splits"]["test"]
    assert calls[-1][1]["alpha"] == 0.25

    def builder(model, settings):
        return model, settings

    run("validation", "custom", "custom", builder=builder, settings={"epsilon": 1e-7})
    kwargs = calls[-1][1]
    assert kwargs["model_transform"]("model") == ("model", {"epsilon": 1e-7})
    assert kwargs["mode"] == "original" and kwargs["removed"] == ()
    assert "def builder" in kwargs["method_metadata"]["source"]
    json.dumps(kwargs["method_metadata"])

    for split, method, name in [("calibration", "loop", "loop"),
                                 ("test", "loop", "../escape"),
                                 ("test", "loop", "original"),
                                 ("test", "original", "renamed"),
                                 ("test", "unknown", "unknown")]:
        with pytest.raises(ValueError):
            run(split, method, name)
    for alpha in ["inverse_k", 0.5]:
        with pytest.raises(ValueError, match="alpha=1.0"):
            run("validation", "xsa_reuse", "bad_alpha", alpha=alpha)
