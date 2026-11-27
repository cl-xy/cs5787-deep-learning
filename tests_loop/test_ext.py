import json

import numpy as np
import pandas as pd
import pytest

from loop_compression import ext, final


def make_main(tmp_path):
    main = tmp_path / "main"
    sens = main / "profile" / "sensitivity"
    (sens / "policy_matched").mkdir(parents=True, exist_ok=True)
    (sens / "policy").mkdir(exist_ok=True)
    (main / "profile" / "explore").mkdir(exist_ok=True)
    loop = dict(window=[16, 21], kept=[18, 19], K=2, iteration="block", strategy="damped", h=0.75, beta=0.0)
    removal = dict(loop, K=1, strategy="naive", h=1.0)
    frozen = {"4": dict(loop=dict(name="d4_r1_loop", settings=loop),
                        removal_same_window=dict(name="d4_r1_removal_same_window", settings=removal))}
    (main / "frozen_configs.json").write_text(json.dumps(frozen))
    policy = dict(window=[16, 21], kept=[18, 19], iteration="block", expected=dict(eps_err=0.01),
                  schedule_by_t=[dict(t_min=0, t_max=999, K=1, strategy="naive")])
    for folder in ("policy_matched", "policy"):
        (sens / folder / "policy_4.json").write_text(json.dumps(policy))
    rows = []
    for group, signal, fraction, K, err in [("removal", "", 1.0, 1, 0.05), ("ungated", "", 1.0, 2, 0.04),
                                            ("gated", "update", 0.5, 2, 0.03), ("gated", "update", 0.25, 2, 0.035),
                                            ("gated", "random", 0.5, 2, 0.045)]:
        rows.append(dict(group=group, window="16-21_kept18-19", K=K, signal=signal, keep_fraction=fraction,
                         select="high", eps_err=err,
                         settings=json.dumps(dict(loop, K=K, **(dict(gate=signal) if signal else {})))))
    pd.DataFrame(rows).to_csv(main / "profile" / "explore" / "gate.csv", index=False)
    return main


def test_candidates_cover_the_planned_arms(tmp_path):
    entries = ext.build_candidates(make_main(tmp_path), budgets=(4,), quality_windows=["14-17"],
                                   quality_steps=[(2, 0.5)], include_bands=False)
    names = {e["name"] for e in entries}
    assert {"d4_r1_loop", "d4_r1_removal_same_window", "d4_policy_matched", "d4_gate",
            "d4_gate_random_control", "q_14_17_K2_h0p5"} <= names
    assert "d4_policy_free" not in names                # opt-in
    assert "d4_gate_ungated" not in names               # identical to the K=2 reference loop
    with_free = ext.build_candidates(make_main(tmp_path), budgets=(4,), include_bands=False, free_policy=True)
    assert "d4_policy_free" in {e["name"] for e in with_free}
    quality = next(e for e in entries if e["role"] == "quality")
    assert quality["settings"]["kept"] == [14, 15, 16, 17] and quality["budget"] == ext.QUALITY_BUDGET
    gate = next(e for e in entries if e["name"] == "d4_gate")
    assert json.loads(json.dumps(gate["settings"]))["gate"] == "update"  # best informative gate, not the control
    policy = next(e for e in entries if e["name"] == "d4_policy_matched")
    assert policy["settings"]["schedule_by_t"] and policy["settings"]["K"] == 1 and "expected" not in policy["settings"]
    folder = ext.write_candidates(tmp_path / "ext", entries)
    assert json.loads((folder / "d4_policy_matched.json").read_text())["window"] == [16, 21]
    assert (tmp_path / "ext" / "shortlist_ext.json").exists()


def test_gate_selection_ignores_random_and_ungated_fraction_one():
    frame = pd.DataFrame(dict(
        group=["gated", "gated", "gated", "ungated"], window="w", K=2, signal=["random", "update", "update", ""],
        keep_fraction=[0.5, 1.0, 0.5, 1.0], eps_err=[0.001, 0.002, 0.03, 0.04], select="high", settings="{}"))
    best, control, ungated = ext.best_gate_candidates(frame, "w")
    assert best.signal == "update" and best.keep_fraction == 0.5 and best.eps_err == 0.03
    assert control.signal == "random" and ungated.group == "ungated"
    with pytest.raises(ValueError):
        ext.best_gate_candidates(frame[frame.group != "gated"], "w")


def test_paired_clip_gain_and_selection(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 0.02, 16)
    clip = pd.DataFrame(dict(original=base, q_a=base + 0.01, q_b=base - 0.01))
    gain = ext.paired_clip_gain(clip, "q_a")
    assert gain["mean"] == pytest.approx(0.01) and gain["ci95_low"] == pytest.approx(0.01)
    results = pd.DataFrame(dict(
        variant=["original", "d4_r1_loop", "d4_policy_matched", "d4_policy_free", "q_a", "q_b"],
        lpips_mean=[0, 0.43, 0.40, 0.41, 0.05, 0.06], unique_parameters=1, dit_sampling_GFLOPs=1.0))
    results.loc[len(results)] = ["d4_r1_removal_same_window", 0.50, 1, 1.0]
    entries = [dict(name="d4_r1_loop", role="loop", budget=4, settings=dict(window=[1, 2])),
               dict(name="d4_r1_removal_same_window", role="removal_same_window", budget=4,
                    settings=dict(window=[1, 2])),
               dict(name="d4_policy_matched", role="policy", budget=4, settings=dict(window=[1, 2])),
               dict(name="d4_policy_free", role="policy", budget=4, settings=dict(window=[1, 2])),
               dict(name="q_a", role="quality", budget=0, settings=dict(window=[1, 2])),
               dict(name="q_b", role="quality", budget=0, settings=dict(window=[1, 2]))]
    frozen = ext.select_frozen(results, entries, clip)
    assert frozen["4"]["policy"]["name"] == "d4_policy_matched" and frozen["4"]["loop"]["name"] == "d4_r1_loop"
    worse = results.assign(lpips_mean=results.lpips_mean.where(~results.variant.str.startswith("d4_policy"), 0.6))
    rejected = ext.select_frozen(worse, entries, clip)["4"]
    assert "policy" not in rejected and rejected["policy_rejected"]["removal_lpips"] == 0.5
    assert frozen["0"]["quality"]["name"] == "q_a"
    assert frozen["0"]["quality"]["validation_clip_gain"]["mean"] == pytest.approx(0.01)
    json.dumps(frozen, allow_nan=False)
    with pytest.raises(ValueError, match="Nothing to freeze"):
        ext.select_frozen(results[results.variant == "original"], entries, clip)


def test_final_reads_extension_roles(tmp_path):
    item = lambda name, role: dict(name=name, role=role, settings=dict(window=[1, 2]), removed=None,
                                   validation_lpips=0.1)
    (tmp_path / "frozen_configs.json").write_text(json.dumps({
        "4": dict(loop=item("d4_r1_loop", "loop"), policy=item("d4_policy_matched", "policy"),
                  policy_candidates={"d4_policy_matched": 0.4}),
        "0": dict(quality=item("q_a", "quality"))}))
    assert [e["name"] for e in final.load_frozen(tmp_path, [4])] == ["d4_r1_loop", "d4_policy_matched"]
    assert [(e["name"], e["budget"]) for e in final.load_frozen(tmp_path, [0])] == [("q_a", 0)]


def test_exact_flops_average_the_rule_applied_at_each_step():
    plain = dict(window=[16, 21], kept=[18, 19], K=1, strategy="naive")
    twice = dict(window=[16, 21], kept=[18, 19], K=2, strategy="damped", h=0.75)
    skip_half = dict(plain, schedule_by_t=[dict(t_min=0, t_max=299, K=1, strategy="damped", h=0.0),
                                           dict(t_min=300, t_max=999, K=2, strategy="damped", h=0.75)])
    f_plain, c_plain = ext.exact_sampling_flops(plain)
    f_twice, c_twice = ext.exact_sampling_flops(twice)
    f_mixed, c_mixed = ext.exact_sampling_flops(skip_half)
    assert (c_plain, c_twice) == (24, 26)
    skipped = int((ext.sampler_timesteps(50) <= 299).sum())
    assert c_mixed == pytest.approx((skipped * 22 + (50 - skipped) * 26) / 50)
    assert f_plain < f_mixed < f_twice
    # Matches the shared estimator for an original DiT-XL/2 (28 calls) in GFLOPs.
    original, calls = ext.exact_sampling_flops(dict(window=[0, 0], kept=[0], K=1, strategy="naive"))
    assert calls == 28 and original == pytest.approx(23724.2843136)


def test_exact_flops_report_adds_columns(tmp_path):
    (tmp_path / "report").mkdir()
    (tmp_path / "frozen").mkdir()
    pd.DataFrame(dict(variant=["original", "a"], dit_sampling_GFLOPs=[23724.28, 22029.96])).to_csv(
        tmp_path / "report" / "summary.csv", index=False)
    (tmp_path / "frozen" / "a.json").write_text(json.dumps(
        dict(window=[16, 21], kept=[18, 19], K=1, strategy="naive")))
    table = ext.exact_flops_report(tmp_path).set_index("variant")
    assert table.loc["original", "flops_reduction_pct_exact"] == pytest.approx(0.0)
    assert table.loc["a", "dit_sampling_GFLOPs_exact"] == pytest.approx(20335.6274688, rel=1e-6)
    assert (tmp_path / "report" / "summary_exact_flops.csv").exists()
