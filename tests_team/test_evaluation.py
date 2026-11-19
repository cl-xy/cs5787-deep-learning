"""LPIPS comparison, manifest consistency and evaluate_split on a tiny real run."""
import json

import pytest

from team_pipeline import evaluation

NAMES = ["original", "member1"]
PAIRS = [(1, 10), (2, 20)]


def metric(x, y):
    return (x - y).square().mean()


def edit_manifest(run, name, **changes):
    path = run / "validation" / name / "manifest.json"
    manifest = json.loads(path.read_text())
    for key, value in changes.items():
        manifest[key] = value
    path.write_text(json.dumps(manifest))


def test_compare_results_reports_paired_distance_and_efficiency(tiny_run):
    result = evaluation.compare_results(tiny_run / "validation", NAMES, metric=metric)
    original, member = result.set_index("variant").loc["original"], result.set_index("variant").loc["member1"]
    assert original.lpips_mean == 0 and member.lpips_mean > 0  # one block fewer changes the pixels
    assert member.unique_blocks == 3 and member.block_evaluations_per_forward == 3
    assert member.parameter_reduction_pct > 0 and original.parameter_reduction_pct == 0
    assert member.latency_speedup_x > 0
    evaluation_json = json.loads((tiny_run / "validation/evaluation.json").read_text())
    assert evaluation_json["metric"] == "injected_metric"
    assert (tiny_run / "validation/paired_metrics.csv").exists()


@pytest.mark.parametrize("key, value, message", [
    ("shared_code_sha256", "0" * 64, "Shared pipeline code differs"),
    ("runtime", {"device": "other"}, "environment differs"),
    ("config", {"image_size": 8}, "configuration"),
    ("pairs", [[9, 9]], "configuration"),
])
def test_mismatched_manifests_are_rejected(tiny_run, key, value, message):
    edit_manifest(tiny_run, "member1", **{key: value})
    with pytest.raises(ValueError, match=message):
        evaluation.compare_results(tiny_run / "validation", NAMES, metric=metric)


def test_duplicate_or_empty_names_are_rejected(tiny_run):
    for names in ([], ["original", "original"]):
        with pytest.raises(ValueError, match="unique"):
            evaluation.compare_results(tiny_run / "validation", names, metric=metric)


def test_check_manifests_consistent_names_the_variant():
    reference = dict(shared_code_sha256="a", runtime={}, config={}, pairs=[])
    evaluation.check_manifests_consistent(reference, dict(reference), "ok")
    with pytest.raises(ValueError, match="bad"):
        evaluation.check_manifests_consistent(reference, dict(reference, config={"x": 1}), "bad")


def test_evaluate_split_writes_results_and_grid(tiny_run):
    protocol = {"splits": {"validation": [list(p) for p in PAIRS]}}
    result = evaluation.evaluate_split(tiny_run, protocol, "validation", NAMES, metric)
    assert result["variant"].tolist() == NAMES
    assert (tiny_run / "validation/results.csv").exists()
    assert (tiny_run / "validation/comparison_grid.png").read_bytes()[:4] == b"\x89PNG"
