"""The run fingerprint is an import-time snapshot, not a live read of the files."""
import importlib.util
from pathlib import Path
import shutil

from team_pipeline import provenance

PACKAGE = Path(provenance.__file__).parent


def load_copy(directory):
    """Import a private copy of provenance.py that snapshots files in ``directory``."""
    target = directory / "provenance.py"
    if not target.exists():
        shutil.copy(PACKAGE / "provenance.py", target)
        for name in provenance.FINGERPRINTED_MODULES:
            (directory / f"{name}.py").write_text(f"# {name}\n")
    spec = importlib.util.spec_from_file_location(f"provenance_copy_{id(directory)}", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_covers_pipeline_modules_and_xsa_method():
    sources = provenance.code_sources()
    for name in provenance.FINGERPRINTED_MODULES:
        assert sources[f"team_pipeline/{name}.py"] == (PACKAGE / f"{name}.py").read_text()
    assert "xsa_method/attention.py" in sources
    # Plotting, CLI and optional metrics must not invalidate prepared runs.
    for excluded in ("report", "cli", "metrics", "extended_metrics", "runs"):
        assert f"team_pipeline/{excluded}.py" not in sources


def test_editing_a_file_after_import_does_not_change_the_hash(tmp_path):
    module = load_copy(tmp_path)
    before = module.shared_code_hash()
    (tmp_path / "sampling.py").write_text("# edited after import\n")
    assert module.shared_code_hash() == before
    assert load_copy(tmp_path).shared_code_hash() != before  # a fresh import sees the edit


def test_changed_xsa_method_source_changes_the_hash(monkeypatch):
    before = provenance.shared_code_hash()
    changed = {**provenance.method_sources(), "xsa_method/attention.py": "changed"}
    monkeypatch.setattr(provenance, "method_sources", lambda: changed)
    assert provenance.shared_code_hash() != before


def test_returned_sources_are_a_copy():
    before = provenance.shared_code_hash()
    provenance.code_sources().clear()
    assert provenance.code_sources()
    assert provenance.shared_code_hash() == before
