"""Run fingerprint: a hash of the pipeline code that produced a run's artifacts.

Sources are read ONCE, when this module is imported. Editing a file afterwards does
not change the fingerprint of the code already loaded in the kernel, so a stale
kernel cannot silently stamp new code onto old runs. Restart and re-prepare to pick
up edits.

Only modules that affect generated images, timings or paired metrics are covered.
Plotting, the CLI and optional metrics (FID/CLIP) are deliberately excluded so
editing them never invalidates a prepared run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xsa_method import method_sources

FINGERPRINTED_MODULES = ("data", "runtime", "profiling", "sampling", "evaluation",
                         "methods", "xsa")

_PACKAGE_DIR = Path(__file__).parent
_SOURCES = {f"team_pipeline/{name}.py": (_PACKAGE_DIR / f"{name}.py").read_text()
            for name in FINGERPRINTED_MODULES}


def code_sources():
    """Return a copy of the source snapshot: this package's pipeline plus xsa_method."""
    return {**_SOURCES, **method_sources()}


def shared_code_hash():
    canonical = json.dumps(code_sources(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
