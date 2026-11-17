"""Notebook integration checks that do not require PyTorch or downloads."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook_code():
    notebook = json.loads((ROOT / "DiT_Layer_Looping_GCP.ipynb").read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def definitions():
    return {node.name: node for source in notebook_code() for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def load_definitions(names, namespace):
    nodes = definitions()
    for name in names:
        exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), "notebook-test", "exec"), namespace)


def test_notebook_uses_direct_cells_and_preserves_shared_implementation():
    code = notebook_code()
    for index, source in enumerate(code):
        compile(source, f"cell-{index}", "exec")
        assert "FILES =" not in source
        assert "team_pipeline.cli" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("team_pipeline")
    actual = definitions()
    for filename in ["data.py", "methods.py", "experiment.py", "example_method.py"]:
        tree = ast.parse((ROOT / "team_pipeline" / filename).read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in {"load_protocol", "shared_code_hash"}:
                    continue  # Deliberate dictionary-input and notebook-source adaptations.
                assert ast.dump(actual[node.name]) == ast.dump(node), node.name


def test_notebook_protocol_matches_team_protocol():
    namespace = {"json": json, "Path": Path}
    from numbers import Integral
    namespace["Integral"] = Integral
    load_definitions(["validate_pairs", "make_protocol", "load_protocol"], namespace)
    protocol = namespace["make_protocol"]()
    assert protocol == json.loads((ROOT / "team_protocol.json").read_text())
    assert namespace["load_protocol"](protocol) == protocol
    assert namespace["load_protocol"](protocol) is not protocol


def test_direct_runner_routes_builtin_and_custom_methods(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "selection.json").write_text('{"removed": [2, 4]}')
    calls = []
    protocol = {"splits": {"validation": [(1, 2)], "test": [(1, 3)]}}
    namespace = dict(Path=Path, json=json, inspect=inspect, hashlib=hashlib,
                     RUN_ARGS=object(), KEEP_CHECKPOINTS=False,
                     load_run=lambda args: (tmp_path, protocol, {"device": "cpu"}),
                     run_variant=lambda *args, **kwargs: calls.append((args, kwargs)))
    load_definitions(["run_method"], namespace)
    run = namespace["run_method"]
    run("validation", "original", "original")
    assert calls[-1][1]["removed"] == ()
    run("test", "damped", "member3_damped", alpha=.25)
    args, kwargs = calls[-1]
    assert args[1] == [(1, 3)] and args[2] == tmp_path / "test/member3_damped"
    assert kwargs["removed"] == [2, 4] and kwargs["alpha"] == .25
    def builder(model, settings):
        return (model, settings)
    run("validation", "custom", "custom_run", builder=builder, settings={"alpha": .1})
    assert calls[-1][1]["model_transform"]("model") == ("model", {"alpha": .1})
    assert calls[-1][1]["method_metadata"]["source"] == inspect.getsource(builder)
    for split, method, name in [("calibration", "loop", "member"),
                                 ("test", "loop", "../escape"),
                                 ("test", "loop", "original"),
                                 ("test", "original", "member")]:
        with pytest.raises(ValueError):
            run(split, method, name)


def test_run_cells_call_live_functions_and_optional_runs_default_off():
    calls = []
    namespace = dict(prepare=lambda *a: calls.append("prepare"), RUN_ARGS=object(),
                     run_method=lambda *a, **k: calls.append((a, k)),
                     evaluate_split=lambda *a: calls.append("evaluate"),
                     RUN_ROOT=Path("unused"), ALPHA=.5,
                     display=lambda *a: None, Image=lambda **k: None)
    for source in notebook_code():
        if source.startswith(("# RUN PREPARE", "# RUN VALIDATION", "# OPTIONAL CUSTOM", "# RUN HELD-OUT TEST")):
            exec(source, namespace)
    assert calls[0] == "prepare"
    assert [entry[0][1] for entry in calls[1:]] == ["original", "removal", "loop", "damped"]
    assert calls[-1][1] == {"alpha": .5}


def test_fingerprint_handles_ipython_functions_and_classes():
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell()
    try:
        result = shell.run_cell("def helper():\n    return 1\n\nclass Block:\n    def forward(self):\n        return 1\n", store_history=True)
        assert result.success
        namespace = dict(shell.user_ns, inspect=inspect, json=json, hashlib=hashlib,
                         SHARED_DEFINITION_NAMES=["helper", "Block"])
        load_definitions(["notebook_sources", "shared_code_hash"], namespace)
        before = namespace["shared_code_hash"]()
        assert "Block.forward" in namespace["notebook_sources"]()
        shell.run_cell("def helper():\n    return 2\n", store_history=True)
        namespace["helper"] = shell.user_ns["helper"]
        assert before != namespace["shared_code_hash"]()
    finally:
        shell.history_manager.end_session()
