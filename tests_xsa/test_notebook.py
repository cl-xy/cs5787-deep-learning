"""The colab notebook is a thin driver: no definitions, valid imports, correct wiring.

Cells are located by their leading ``# LABEL`` comment, never by position. Section 3
(which runs pytest) is never executed here, so these tests cannot recurse.
"""
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from xsa_method import method_sources

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "DiT_Layer_Looping_colab.ipynb"
LEGACY_PACKAGE = "dit_" + "looping"  # written into the GCP notebook before this refactor
RUN_VALIDATION_NAMES = ["original", "member1_removal", "member2_loop", "member3_damped",
                        "member4_xsa_reuse"]


def notebook():
    return json.loads(NOTEBOOK.read_text())


def code_cells():
    return ["".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code"]


def cells_starting(prefix):
    found = [source for source in code_cells() if source.startswith(prefix)]
    assert found, f"no cell starts with {prefix!r}"
    return found


def cell(prefix):
    (source,) = cells_starting(prefix)
    return source


# ---- static contract ----------------------------------------------------------------
def test_notebook_is_a_thin_driver_with_no_definitions():
    for index, source in enumerate(code_cells()):
        tree = ast.parse(source)
        compile(tree, f"cell-{index}", "exec")
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)), (
                f"cell {index} defines {node.name}; move it into team_pipeline/ or xsa_method/")
        assert len(source.splitlines()) <= 60, f"cell {index} is too long for a driver"


def test_every_package_import_resolves():
    checked = 0
    for source in code_cells():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
                top = module_name.split(".")[0]
                assert top != LEGACY_PACKAGE, "the generated helper package no longer exists"
                if top in {"team_pipeline", "xsa_method"}:
                    module = importlib.import_module(module_name)
                    for alias in node.names:
                        assert (hasattr(module, alias.name)
                                or importlib.util.find_spec(f"{module_name}.{alias.name}")), (
                            f"{module_name}.{alias.name}")
                        checked += 1
    assert checked >= 20


def test_notebook_does_not_set_environment_or_hash_sources_itself():
    text = "\n".join(code_cells())
    assert "CUBLAS_WORKSPACE_CONFIG" not in text and "PYTORCH_ENABLE_MPS_FALLBACK" not in text
    assert "inspect.getsource" not in text and "notebook_sources" not in text


def test_notebook_is_saved_without_outputs():
    for source_cell in notebook()["cells"]:
        if source_cell["cell_type"] == "code":
            assert source_cell["outputs"] == [] and source_cell["execution_count"] is None


# ---- execution on CPU with network/pip/git stubbed ----------------------------------
@pytest.fixture
def ns(monkeypatch):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: commands.append(cmd))
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__main__", "commands": commands}
    for prefix in ["# SETUP", "# IMPORTS", "# PROTOCOL", "# CONFIGURE", "# METRIC SETTINGS"]:
        exec(compile(cell(prefix), prefix, "exec"), namespace)
    return namespace


def test_setup_installs_dependencies_and_finds_the_project(ns):
    install = ns["commands"][0]
    assert install[1:4] == ["-m", "pip", "install"] and "pytest" in install
    assert (ns["PROJECT_ROOT"] / "team_pipeline/runs.py").is_file()
    assert str(ns["PROJECT_ROOT"]) in sys.path


def test_protocol_uses_30_held_out_images_per_class_with_frozen_sampling_settings(ns):
    from collections import Counter
    golden = json.loads((ROOT / "tests_team/golden_notebook_protocol.json").read_text())
    protocol = ns["PROTOCOL"]
    assert protocol["settings"] == golden["settings"]
    assert {k: len(v) for k, v in protocol["splits"].items()} == {
        "calibration": 20, "validation": 20, "test": 300}
    assert Counter(label for label, seed in protocol["splits"]["test"]) == {
        label: 30 for label in golden["class_labels"]}
    for split, pairs in protocol["splits"].items():
        assert pairs[:3] == golden["first_pairs"][split]
    assert protocol["splits"]["test"][-1] == [701, 10339]


def test_configuration_defines_one_method_list_and_a_bound_runner(ns):
    assert ns["NAMES"] == RUN_VALIDATION_NAMES
    assert [m for m, _, _ in ns["METHODS"]] == ["original", "removal", "loop", "damped", "xsa_reuse"]
    assert dict((name, kw) for _, name, kw in ns["METHODS"])["member3_damped"] == {"alpha": "inverse_k"}
    assert ns["RUN_ARGS"].output == ns["RUN_ROOT"] and ns["RUN_ARGS"].protocol is ns["PROTOCOL"]
    assert callable(ns["run_method"])
    assert ns["EVAL_OPTIONS"]["enable_fid"] is False
    assert ns["EVAL_OPTIONS"]["enable_clip"] is True
    # No run state is created by configuring.
    assert not ns["RUN_ROOT"].exists()


def test_run_cells_call_package_functions_and_optional_runs_default_off(ns):
    calls = []
    for name in ["prepare", "run_methods", "evaluate_saved_split", "show_evaluation",
                 "evaluate_split", "load_run"]:
        ns[name] = lambda *a, _n=name, **k: calls.append((_n, a, k)) or SimpleNamespace()
    exec(cell("# RUN PREPARE"), ns)
    assert calls == [("prepare", (ns["RUN_ARGS"],), {})]
    calls.clear()
    for source in cells_starting("# RUN VALIDATION"):
        exec(source, ns)
    assert [c[2]["only"] for c in calls] == RUN_VALIDATION_NAMES
    assert all(c[0] == "run_methods" and c[1][2:4] == ("validation", ns["METHODS"]) for c in calls)
    calls.clear()
    exec(cell("# OPTIONAL CUSTOM"), ns)
    exec(cell("# RUN HELD-OUT TEST"), ns)
    assert calls == [] and ns["RUN_CUSTOM"] is False and ns["RUN_HELD_OUT_TEST"] is False


def test_enabling_held_out_test_runs_all_methods_then_evaluates_the_test_split(ns):
    calls = []
    for name in ["run_methods", "evaluate_saved_split", "show_evaluation"]:
        ns[name] = lambda *a, _n=name, **k: calls.append((_n, a, k)) or {}
    ns["protocol"] = ns["PROTOCOL"]
    source = cell("# RUN HELD-OUT TEST").replace("RUN_HELD_OUT_TEST = False", "RUN_HELD_OUT_TEST = True")
    exec(source, ns)
    assert [c[0] for c in calls] == ["run_methods", "evaluate_saved_split", "show_evaluation"]
    assert calls[0][1][2] == "test" and calls[0][2] == {"show": False}
    assert calls[1][1][2:4] == ("test", ns["NAMES"]) and "installer" in calls[1][2]


def test_enabling_custom_method_runs_the_example_builder_on_validation(ns):
    calls = []
    ns["run_method"] = lambda *a, **k: calls.append(("run", a, k))
    ns["evaluate_saved_split"] = lambda *a, **k: calls.append(("evaluate", a, k)) or {}
    ns["show_evaluation"] = lambda outcome: None
    ns["protocol"] = ns["PROTOCOL"]
    ns["RUN_ROOT"] = ROOT / "does-not-exist-so-read_csv-is-stubbed"
    ns["pd"] = SimpleNamespace(read_csv=lambda path: pd.DataFrame())
    ns["display"] = lambda frame: None
    exec(cell("# OPTIONAL CUSTOM").replace("RUN_CUSTOM = False", "RUN_CUSTOM = True"), ns)
    (_, run_args, run_kwargs), (_, eval_args, _) = calls
    assert run_args == ("validation", "custom", "custom_example")
    assert run_kwargs["builder"] is ns["build_model"] and run_kwargs["settings"] == ns["CUSTOM_SETTINGS"]
    assert eval_args[3] == [*ns["NAMES"], "custom_example"]


def test_report_cells_use_package_functions_only(ns, tmp_path):
    shown = []
    summary = pd.DataFrame(dict(method=["Original DiT"], damping_policy=["none"], images=[2],
                                lpips_mean=[0.0]))
    ns.update(
        find_report_run=lambda base, split, current=None: tmp_path,
        load_report_data=lambda root, split: dict(run_root=root, output=root, summary=summary),
        display=lambda obj: None,
        show_figure_paths=lambda paths: shown.append(paths),
        plot_report_samples=lambda report, count: ("samples", count),
        REPORT_PLOTS={name: (lambda report, _n=name: _n)
                      for name in ["quality", "efficiency", "per_class", "schedules"]},
    )
    exec(cell("# REPORT CONFIG"), ns)
    assert ns["REPORT_RUN_ROOT"] == tmp_path and ns["REPORT"]["summary"] is summary
    for source in code_cells():
        if source.startswith("show_figure_paths("):
            exec(source, ns)
    assert shown == ["quality", "efficiency", "per_class", ("samples", 6), "schedules"]


def test_export_cell_is_off_by_default(ns, capsys):
    exec(cell("# EXPORT"), ns)
    assert "Set EXPORT_RUN=True" in capsys.readouterr().out


# ---- the xsa_method source snapshot used by the run fingerprint ---------------------
def test_import_snapshot_is_complete_and_defensive():
    sources = method_sources()
    assert set(sources) == {f"xsa_method/{p.name}" for p in (ROOT / "xsa_method").glob("*.py")}
    for path, source in sources.items():
        assert source == (ROOT / path).read_text()
    sources.clear()
    assert method_sources()
