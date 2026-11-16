"""Validation driver: split discipline, commands, hashes and frozen selection."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from loop_compression import drive


def entry(name, role, budget, settings=True, removed=None):
    return dict(name=name, role=role, budget=budget, eps_err=0.1,
                settings=dict(window=[8, 11], kept=[9, 10], K=1, tag=name) if settings else None, removed=removed)


ENTRIES = [
    entry("d2_r1_loop", "loop", 2), entry("d2_r1_naive_same_window", "naive_same_window", 2),
    entry("d2_r1_removal_same_window", "removal_same_window", 2), entry("d2_r2_loop", "loop", 2),
    entry("d2_removal_best_window", "removal_best_window", 2),
    entry("d2_removal_lowest_blocks", "removal_lowest_blocks", 2, settings=False, removed=[5, 9]),
]


def test_selection_split_is_validation_only():
    assert drive.check_split("validation") == "validation"
    for split in ("test", "fid"):
        with pytest.raises(ValueError, match="frozen"):
            drive.check_split(split)
        with pytest.raises(ValueError):
            drive.run_command("out", ENTRIES[0], "cfg", split=split)
    with pytest.raises(ValueError):
        drive.check_split("calibration")


def test_run_commands_use_the_shared_runner(tmp_path):
    loop = drive.run_command("out", ENTRIES[0], tmp_path / "cfg")
    assert loop[1:4] == ["-m", "team_pipeline.cli", "run"]
    assert loop[loop.index("--method") + 1] == drive.METHOD
    assert loop[loop.index("--method-config") + 1] == str(tmp_path / "cfg" / "d2_r1_loop.json")
    assert loop[loop.index("--split") + 1] == "validation" and loop[loop.index("--name") + 1] == "d2_r1_loop"
    shared = drive.run_command("out", ENTRIES[-1], tmp_path / "cfg", limit=4, stride=2)
    assert shared[shared.index("--method") + 1] == "removal"
    assert shared[shared.index("--removed") + 1:shared.index("--removed") + 3] == ["5", "9"]
    assert shared[shared.index("--limit") + 1] == "4"
    assert shared[shared.index("--stride") + 1] == "2"
    assert "--stride" not in loop


def prepare_run(tmp_path):
    out = tmp_path / "run"
    sens = out / "profile" / "sensitivity"
    (sens / "shortlist").mkdir(parents=True)
    (sens / "shortlist.json").write_text(json.dumps(ENTRIES))
    for e in ENTRIES:
        if e["settings"] is not None:
            (sens / "shortlist" / f"{e['name']}.json").write_text(json.dumps(e["settings"]))
    return out


def test_validate_runs_missing_variants_records_hashes_and_never_touches_other_splits(tmp_path, monkeypatch):
    out = prepare_run(tmp_path)
    (out / "validation" / "d2_r2_loop").mkdir(parents=True)   # already finished: must be skipped
    commands = []

    def fake_run(command, check):
        commands.append(command)
        name = command[command.index("--name") + 1]
        (out / "validation" / name).mkdir(parents=True, exist_ok=True)

    evaluated = []
    monkeypatch.setattr(drive.subprocess, "run", fake_run)
    monkeypatch.setattr(drive, "evaluate", lambda output, names: evaluated.append(names))
    drive.main(["validate", "--output", str(out), "--limit", "4"])
    names = [c[c.index("--name") + 1] for c in commands]
    assert names == ["original"] + [e["name"] for e in ENTRIES if e["name"] != "d2_r2_loop"]
    assert len(names) == len(set(names))
    assert all(c[c.index("--split") + 1] == "validation" for c in commands)
    assert not any("test" in c or "fid" in c for c in commands)
    log = json.loads((out / "validation" / "driver_log.json").read_text())
    loop = next(r for r in log if r["name"] == "d2_r1_loop")
    expected = hashlib.sha256((out / "profile/sensitivity/shortlist/d2_r1_loop.json").read_bytes()).hexdigest()
    assert loop["config_sha256"] == expected
    assert "config_sha256" not in next(r for r in log if r["name"] == "d2_removal_lowest_blocks")
    assert evaluated == [["original"] + [e["name"] for e in ENTRIES]]
    # Second invocation has nothing left to run.
    commands.clear()
    drive.main(["validate", "--output", str(out), "--limit", "4"])
    assert commands == []


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    out = prepare_run(tmp_path)
    monkeypatch.setattr(drive.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    drive.main(["validate", "--output", str(out), "--dry-run"])
    assert not (out / "validation").exists()


def test_paired_difference_bootstrap():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.7, 16)
    paired = pd.concat([
        pd.DataFrame(dict(variant="a", index=range(16), paired_lpips=base - 0.1)),
        pd.DataFrame(dict(variant="b", index=range(16), paired_lpips=base))])
    result = drive.paired_difference(paired, "a", "b")
    assert result["mean"] == pytest.approx(-0.1) and result["ci95_high"] == pytest.approx(-0.1)
    noisy = pd.concat([
        pd.DataFrame(dict(variant="a", index=range(16), paired_lpips=base + rng.normal(0, 0.2, 16))),
        pd.DataFrame(dict(variant="b", index=range(16), paired_lpips=base))])
    result = drive.paired_difference(noisy, "a", "b")
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]
    with pytest.raises(ValueError):
        drive.paired_difference(paired[paired.variant == "a"], "a", "b")


def synthetic_results(lpips, parameters=None):
    parameters = parameters or {}
    names = ["original"] + list(lpips)
    return pd.DataFrame(dict(variant=names,
                             lpips_mean=[0.0] + [lpips[n] for n in lpips],
                             unique_parameters=[100] + [parameters.get(n, 90) for n in lpips],
                             dit_sampling_GFLOPs=[10.0] + [9.0] * len(lpips)))


def synthetic_paired(lpips, noise=0.01, images=16, seed=0):
    rng = np.random.default_rng(seed)
    rows = [pd.DataFrame(dict(variant=n, index=range(images),
                              paired_lpips=value + rng.normal(0, noise, images))) for n, value in lpips.items()]
    return pd.concat(rows)


def test_select_frozen_picks_by_validation_lpips_and_reports_honestly():
    lpips = {"d2_r1_loop": 0.40, "d2_r1_naive_same_window": 0.60, "d2_r1_removal_same_window": 0.50,
             "d2_r2_loop": 0.45, "d2_removal_best_window": 0.46, "d2_removal_lowest_blocks": 0.55}
    frozen = drive.select_frozen(synthetic_results(lpips), synthetic_paired(lpips), ENTRIES)[2]
    assert frozen["loop"]["name"] == "d2_r1_loop"
    assert frozen["strongest_removal"]["name"] == "d2_removal_best_window"
    assert frozen["naive_same_window"]["name"] == "d2_r1_naive_same_window"
    assert frozen["removal_same_window"]["name"] == "d2_r1_removal_same_window"
    assert frozen["loop_beats_removal"] and frozen["equal_parameters"]
    json.dumps(frozen, allow_nan=False)
    # A loop that is slightly worse than the strongest removal is NOT reported as a win.
    lpips["d2_r1_loop"] = 0.47
    lpips["d2_r2_loop"] = 0.48
    frozen = drive.select_frozen(synthetic_results(lpips), synthetic_paired(lpips), ENTRIES)[2]
    assert frozen["loop"]["name"] == "d2_r1_loop" and not frozen["loop_beats_removal"]
    # A small gap inside the noise is not a reliable win either.
    lpips["d2_r1_loop"] = 0.455
    frozen = drive.select_frozen(synthetic_results(lpips), synthetic_paired(lpips, noise=0.1), ENTRIES)[2]
    assert not frozen["loop_beats_removal"]
    # Unequal parameter counts are flagged.
    frozen = drive.select_frozen(synthetic_results(lpips, {"d2_r1_loop": 95}), synthetic_paired(lpips), ENTRIES)[2]
    assert not frozen["equal_parameters"]


def test_freeze_writes_json_and_needs_both_arms(tmp_path):
    out = prepare_run(tmp_path)
    lpips = {e["name"]: 0.4 + 0.01 * i for i, e in enumerate(ENTRIES)}
    root = out / "validation"
    root.mkdir()
    synthetic_results(lpips).to_csv(root / "results.csv", index=False)
    synthetic_paired(lpips).to_csv(root / "paired_metrics.csv", index=False)
    drive.main(["freeze", "--output", str(out)])
    frozen = json.loads((out / "frozen_configs.json").read_text())
    assert set(frozen) == {"2"}
    with pytest.raises(ValueError):
        drive.select_frozen(synthetic_results({"d2_r1_loop": 0.4}),
                            synthetic_paired({"d2_r1_loop": 0.4}), [ENTRIES[0]])


def test_duplicate_configs_run_once_and_are_aliased(tmp_path, monkeypatch):
    out = prepare_run(tmp_path)
    duplicate = dict(ENTRIES[3], name="d2_r2_dup")          # same settings as d2_r2_loop
    duplicate["settings"] = dict(ENTRIES[3]["settings"])
    shortlist = ENTRIES[:4] + [duplicate]
    sens = out / "profile" / "sensitivity"
    (sens / "shortlist.json").write_text(json.dumps(shortlist))
    (sens / "shortlist" / "d2_r2_dup.json").write_text(json.dumps(duplicate["settings"]))
    commands = []

    def fake_run(command, check):
        commands.append(command)
        (out / "validation" / command[command.index("--name") + 1]).mkdir(parents=True, exist_ok=True)

    evaluated = []
    monkeypatch.setattr(drive.subprocess, "run", fake_run)
    monkeypatch.setattr(drive, "evaluate", lambda output, names: evaluated.append(names))
    drive.main(["validate", "--output", str(out)])
    ran = [c[c.index("--name") + 1] for c in commands]
    assert "d2_r2_dup" not in ran and ran.count("d2_r2_loop") == 1
    assert (out / "validation" / "d2_r2_dup").resolve() == (out / "validation" / "d2_r2_loop").resolve()
    log = json.loads((out / "validation" / "driver_log.json").read_text())
    assert dict(name="d2_r2_dup", alias_of="d2_r2_loop") in log
    assert "d2_r2_dup" in evaluated[0]
