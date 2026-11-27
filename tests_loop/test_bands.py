import json

import numpy as np
import pandas as pd
import pytest

from loop_compression import bands


def row(cid, kind, first, last, kept, errs, K=1, strategy="naive", h=1.0, discard=2, depth=28):
    settings = dict(window=[first, last], kept=kept, K=K, iteration="block", strategy=strategy, h=h, beta=0.0)
    out = dict(config_id=cid, kind=kind, discard=discard, first=first, last=last, kept=json.dumps(kept),
               K=K, iteration="block", strategy=strategy, h=h, beta=0.0, flops_ratio=1.0,
               center_depth_fraction=(first + last + 1) / (2 * depth), settings=json.dumps(settings))
    out.update({f"eps_err_b{i}": e for i, e in enumerate(errs)})
    return out


def frame():
    rows = []
    for name, (first, last, kept) in dict(early=(1, 4, [2, 3]), middle=(10, 13, [11, 12]),
                                          late=(20, 23, [21, 22])).items():
        gain = dict(early=1.0, middle=0.8, late=0.5)[name]
        rows.append(row(f"{name}_rm", "removal", first, last, kept, [1.0, 2.0]))
        rows.append(row(f"{name}_lp", "loop", first, last, kept, [1.0 * gain, 2.0 * gain], K=2,
                        strategy="damped", h=0.5))
        rows.append(row(f"{name}_wr", "window_removal", first, last, kept, [0.1, 0.1], strategy="damped", h=0.0))
    rows[0]["iteration"] = "layer"  # layer-mode rows are not part of the block-iteration analysis
    return pd.DataFrame(rows)


def test_band_assignment_bounds():
    assert [bands.assign_band(v) for v in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)] == [
        "early", "early", "middle", "middle", "late", "late"]
    with pytest.raises(ValueError):
        bands.assign_band(1.5)


def test_load_sweeps_merges_extras_and_dedupes(tmp_path):
    frame().iloc[3:].to_csv(tmp_path / "window_sweep.csv", index=False)
    frame().iloc[:6].to_csv(tmp_path / "window_sweep_early.csv", index=False)
    merged = bands.load_sweeps(tmp_path, ["early"])
    assert merged.config_id.is_unique and set(merged.band) == {"early", "middle", "late"}


def test_table_and_summary_pick_the_best_loop_per_band():
    data = frame()
    data["band"] = data.center_depth_fraction.map(bands.assign_band)
    data = data[data.iteration == "block"]
    summary = bands.band_summary(data).set_index("band")
    assert summary.loc["late", "loop_eps_err"] == pytest.approx(1.5 * 0.5)
    assert summary.loc["late", "same_window_removal_eps_err"] == pytest.approx(1.5)
    table = bands.band_table(data)
    late = table[(table.band == "late") & (table.bucket == 1)].iloc[0]
    assert late.loop_over_removal == pytest.approx(0.5) and late.loop_window == "20-23"


def test_finalists_pair_each_loop_with_its_equal_parameter_removal(tmp_path):
    data = frame()
    data["band"] = data.center_depth_fraction.map(bands.assign_band)
    entries = bands.finalists(bands.band_summary(data[data.iteration == "block"]), [2])
    assert {e["name"] for e in entries} >= {"d2_late_loop", "d2_late_removal_same_window"}
    loop = next(e for e in entries if e["name"] == "d2_late_loop")
    removal = next(e for e in entries if e["name"] == "d2_late_removal_same_window")
    assert removal["settings"]["kept"] == loop["settings"]["kept"] and removal["settings"]["K"] == 1
    assert bands.finalists(bands.band_summary(data[data.iteration == "block"]), [99]) == []


def test_main_writes_outputs(tmp_path):
    sens = tmp_path / "profile" / "sensitivity"
    sens.mkdir(parents=True)
    frame().to_csv(sens / "window_sweep.csv", index=False)
    bands.main(["--output", str(tmp_path), "--finalist-budgets", "2"])
    for name in ("band_table.csv", "band_summary.csv", "band_heatmap.png"):
        assert (sens / "bands" / name).exists()
    assert (sens / "shortlist_bands.json").exists() and (sens / "shortlist_bands" / "d2_late_loop.json").exists()
