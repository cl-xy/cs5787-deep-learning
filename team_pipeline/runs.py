"""Prepare a run folder and re-validate it before every later step.

``args`` is any object with ``output`` and ``repo`` (plus ``protocol``, ``device`` and
``checkpoint`` for :func:`prepare`). ``args.protocol`` may be a protocol dict or a path.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .data import load_protocol, protocol_hash
from .provenance import code_sources, shared_code_hash
from .runtime import (configure_dit, environment, select_device, sha256, write_json)
from .sampling import download_checkpoint


def prepare(args):
    """Create ``args.output`` with the frozen protocol, config, environment and code snapshot."""
    protocol = load_protocol(args.protocol)
    device = select_device(args.device)
    commit = configure_dit(args.repo)
    if commit != protocol["settings"]["dit_commit"]:
        raise ValueError("DiT source commit differs from the agreed protocol")
    checkpoint = download_checkpoint(Path(args.checkpoint).resolve())
    if sha256(checkpoint) != protocol["settings"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the agreed protocol")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = {**protocol["settings"], "checkpoint": str(checkpoint), "device": str(device),
              "protocol_sha256": protocol_hash(protocol),
              "shared_code_sha256": shared_code_hash()}
    write_json(root / "protocol.json", protocol)
    write_json(root / "code_sources.json", code_sources())
    write_json(root / "config.json", config)
    write_json(root / "environment.json", environment(str(device)))
    with (root / "pip-freeze.txt").open("w") as out:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=out, check=True)
    print(f"Prepared shared inputs: {root.resolve()}")


def load_run(args):
    """Verify a prepared run still matches the code, DiT source, checkpoint and hardware."""
    root = Path(args.output)
    protocol = load_protocol(root / "protocol.json")
    config = json.loads((root / "config.json").read_text())
    if config.get("shared_code_sha256") != shared_code_hash():
        raise ValueError("Shared pipeline changed after preparation; create a new run")
    if config["protocol_sha256"] != protocol_hash(protocol):
        raise ValueError("Protocol changed after preparation; create a new run")
    if any(config.get(k) != v for k, v in protocol["settings"].items()):
        raise ValueError("Config settings changed after preparation; create a new run")
    if configure_dit(args.repo) != config["dit_commit"]:
        raise ValueError("DiT source changed after preparation")
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preparation")
    current = environment(config["device"])
    recorded = json.loads((root / "environment.json").read_text())
    if current != recorded:
        raise ValueError("Runtime environment changed; prepare a new run for comparable "
                         "measurements")
    return root, protocol, config
