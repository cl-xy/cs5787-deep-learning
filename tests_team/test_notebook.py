"""The GCP/Colab notebook drives the shared packages; it embeds no copy of them."""
import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "DiT_Layer_Looping_GCP.ipynb"
LEGACY_PACKAGE = "dit_" + "looping"  # the helper package this notebook used to generate


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_every_cell_compiles_and_nothing_is_generated_at_runtime():
    for index, source in enumerate(notebook_code()):
        compile(source, f"cell-{index}", "exec")
        assert "FILES =" not in source, "embedded source blobs are not allowed"
        assert LEGACY_PACKAGE not in source
        assert "write_text(content)" not in source


def test_pipeline_imports_resolve_to_real_package_members():
    checked = []
    for source in notebook_code():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {
                    "team_pipeline", "loop_compression", "xsa_method"}:
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert (hasattr(module, alias.name)
                            or importlib.util.find_spec(f"{node.module}.{alias.name}")), (
                        f"{node.module}.{alias.name}")
                    checked.append(alias.name)
    assert {"configure_dit", "run_variant", "compare_results", "calibrate"} <= set(checked)


def test_notebook_defines_only_small_local_helpers():
    defined = [node.name for source in notebook_code() for node in ast.parse(source).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    # `sh` streams subprocess output; `forward` is a closure in the Level B smoke check.
    assert defined == ["sh", "forward"]


def test_cli_and_notebook_use_the_same_run_preparation():
    from team_pipeline import cli, runs
    assert cli.prepare is runs.prepare and cli.load_run is runs.load_run


def test_cli_commands_used_by_the_notebook_exist():
    from team_pipeline import cli
    text = "\n".join(notebook_code())
    for command in ["prepare", "profile", "select", "run", "evaluate", "clip", "fid"]:
        assert f'"{command}"' in text
        with pytest.raises(SystemExit) as stop:
            cli.main([command, "--help"])
        assert stop.value.code == 0
