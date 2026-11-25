"""The GCP/Colab notebook drives the shared packages; it embeds no copy of them."""
import ast
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "DiT_Layer_Looping_GCP.ipynb"
LEGACY_PACKAGE = "dit_" + "looping"  # the helper package this notebook used to generate


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("enabled,automatic", [(False, True), (False, False), (True, True), (True, False)])
def test_colab_fid_toggle_controls_reference_and_evaluation(tmp_path, split, enabled, automatic, monkeypatch):
    """Disabling FID must bypass downloads/install even with a cached reference path."""
    import pandas as pd
    from unittest.mock import Mock

    notebook = json.loads((ROOT / "DiT_Layer_Looping_colab.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    folder = tmp_path / split
    (folder / "original").mkdir(parents=True)
    (folder / "original/manifest.json").write_text(json.dumps({
        "pairs": [[0, 42]], "config": {"image_size": 256}}))
    prepared = Mock(return_value=(tmp_path / "prepared", "Imagenette"))
    extended = Mock(return_value="summary")
    install = Mock()
    base = Mock()
    namespace = dict(
        WORKSPACE=tmp_path, pip_install=install, select_device=lambda _: "cpu", RUN_ROOT=tmp_path,
        NAMES=["original"], pd=pd, json=json,
        prepare_imagenette_reference=prepared, evaluate_fid_tpips=extended,
        evaluate_split=base, _extra_version=lambda _: "not-installed",
        subprocess=SimpleNamespace(run=install), sys=SimpleNamespace(executable="python"),
        display=lambda _: None, Image=lambda **kw: None, FID_REAL_IMAGES_PER_CLASS=500,
    )
    exec(next(s for s in sources if s.startswith("# METRIC SETTINGS")), namespace)
    from team_pipeline import extended_metrics as em
    monkeypatch.setattr(em, "evaluate_split", base)
    monkeypatch.setattr(em, "prepare_imagenette_reference", prepared)
    monkeypatch.setattr(em, "evaluate_fid_tpips", extended)
    monkeypatch.setattr(em, "package_version", lambda _: "not-installed")
    options = namespace["EVAL_OPTIONS"]
    options.update(enable_clip=False, enable_fid=enabled,
                   imagenette_source_root=tmp_path / "source" if automatic else None,
                   fid_reference_dir=tmp_path / "cached", installer=install)
    assert em.evaluate_saved_split(tmp_path, {}, split, ["original"], **options)["results"] == "summary"
    base.assert_called_once_with(tmp_path, {}, split, ["original"], None)
    assert prepared.call_count == int(enabled and automatic)
    assert install.call_count == int(enabled)
    expected = (tmp_path / ("prepared" if automatic else "cached")) if enabled else None
    assert extended.call_args.kwargs["fid_reference_dir"] == expected
    assert extended.call_args.kwargs["tpips_factors"] == ()


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
