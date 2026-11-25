"""CLIP evaluation with deterministic encoders; no model downloads."""
import json

import pandas as pd
import pytest
import torch

from team_pipeline import extended_metrics
from pathlib import Path
from types import ModuleType


@pytest.fixture(params=["module", "notebook"])
def evaluation(request):
    if request.param == "module":
        return extended_metrics
    notebook = json.loads((Path(__file__).resolve().parents[1] / "DiT_Layer_Looping_colab.ipynb").read_text())
    module = ModuleType("notebook_evaluation")
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "def evaluate_fid_tpips(" in "".join(c["source"]))
    exec(compile(source, "notebook-evaluation", "exec"), module.__dict__)
    return module


def saved_run(root):
    pairs = [[0, 100], [1, 101]]
    rows = []
    for name, values in [('original', [1., -1.]), ('changed', [-1., -1.])]:
        folder = root / name
        folder.mkdir()
        (folder / 'manifest.json').write_text(json.dumps(dict(
            pairs=pairs, config={'image_size': 8}, runtime={}, shared_code_sha256='test')))
        pd.DataFrame([(i, *p, 1.) for i, p in enumerate(pairs)],
                     columns=['index', 'label', 'seed', 'latency_s']).to_csv(folder / 'samples.csv', index=False)
        for i, value in enumerate(values):
            torch.save(torch.full((1, 3, 8, 8), value), folder / f'image_{i:04d}.pt')
            rows.append(dict(variant=name, index=i, label=pairs[i][0], seed=pairs[i][1], paired_lpips=0., latency_s=1.))
    pd.DataFrame(rows).to_csv(root / 'paired_metrics.csv', index=False)
    pd.DataFrame([dict(variant=n, images=2, lpips_mean=0., lpips_std=0., latency_mean_s=1., unique_parameters=100) for n in ['original', 'changed']]).to_csv(root / 'results.csv', index=False)


def encoders(device='cpu'):
    def image(x):
        assert len(x) <= 1  # Streaming batch size is respected.
        v = x.mean((1, 2, 3))
        return torch.stack([v, torch.zeros_like(v)], -1)
    def text(prompts):
        assert prompts == ['a photo of a tench', 'a photo of a goldfish']
        return torch.tensor([[1., 0.], [-1., 0.]])
    return image, text, {'model': 'test-clip', 'revision': 'test'}


def test_extended_clip_scores_and_saved_results(tmp_path, monkeypatch, evaluation):
    saved_run(tmp_path)
    monkeypatch.setattr(evaluation, 'load_clip', encoders, raising=False)
    result = evaluation.evaluate_fid_tpips(tmp_path, ['original', 'changed'],
        tpips_factors=(), enable_clip=True, clip_batch_size=1).set_index('variant')
    for column in ['clip_image_text_cosine', 'clip_image_image_cosine']:
        assert result.loc['original', column + '_mean'] == pytest.approx(1.)
        assert result.loc['changed', column + '_mean'] == pytest.approx(0.)
        assert result.loc['changed', column + '_std'] == pytest.approx(2 ** .5)
    paired = pd.read_csv(tmp_path / 'paired_metrics_extended.csv')
    assert paired.loc[paired.variant == 'changed', 'clip_image_image_cosine'].tolist() == [-1., 1.]
    assert 'clip_image_text_cosine_mean' in pd.read_csv(tmp_path / 'results_extended.csv')
    meta = json.loads((tmp_path / 'evaluation_extended.json').read_text())
    assert meta['clip']['higher_is_better'] is True
    assert meta['clip']['image_reference'] == 'original (matched label and seed)'
    assert meta['fid'] is None


def test_clip_rejects_mismatched_pairs_before_loading_model(tmp_path, monkeypatch, evaluation):
    saved_run(tmp_path)
    path = tmp_path / 'changed/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['pairs'].reverse()
    path.write_text(json.dumps(manifest))
    def unexpected(*args, **kwargs):
        pytest.fail('CLIP must not load for mismatched runs')
    monkeypatch.setattr(evaluation, 'load_clip', unexpected, raising=False)
    with pytest.raises(ValueError, match='Mismatched generation pairs'):
        evaluation.evaluate_fid_tpips(tmp_path, ['original', 'changed'], tpips_factors=(), enable_clip=True)


def test_report_exports_and_plots_clip_without_fid(tmp_path, monkeypatch, evaluation):
    root = tmp_path / "validation"
    root.mkdir()
    saved_run(root)
    monkeypatch.setattr(evaluation, "load_clip", encoders)
    evaluation.evaluate_fid_tpips(root, ["original", "changed"],
        tpips_factors=(), enable_clip=True, clip_batch_size=1)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "DiT_Layer_Looping_colab.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "def load_report_data(" in "".join(c["source"]))
    import matplotlib
    matplotlib.use("Agg")
    ns = {}
    exec(compile(source, "notebook-report", "exec"), ns)
    report = ns["load_report_data"](tmp_path)
    exported = pd.read_csv(report["output"] / "report_summary.csv").set_index("variant")
    assert exported.loc["changed", "clip_image_text_cosine_mean"] == pytest.approx(0.)
    assert exported.loc["original", "clip_image_image_cosine_mean"] == pytest.approx(1.)
    assert "fid" not in exported
    paths = ns["plot_report_quality"](report)
    assert all(path.stat().st_size > 0 for path in paths.values())
    metadata_path = root / "evaluation_extended.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["generation_code_sha256"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        ns["load_report_data"](tmp_path)


def test_disabled_clip_does_not_load_model(tmp_path, monkeypatch, evaluation):
    saved_run(tmp_path)
    def unexpected(*a, **k):
        pytest.fail("Disabled CLIP must not load a model")
    monkeypatch.setattr(evaluation, "load_clip", unexpected)
    result = evaluation.evaluate_fid_tpips(tmp_path, ["original"], tpips_factors=())
    assert "clip_image_text_cosine_mean" not in result
    assert json.loads((tmp_path / "evaluation_extended.json").read_text())["clip"] is None


def test_cli_extended_enables_both_clip_scores_by_default(tmp_path, monkeypatch):
    from team_pipeline import cli
    root = tmp_path / "validation"
    root.mkdir()
    saved_run(root)
    monkeypatch.setattr(extended_metrics, "load_clip", encoders)
    cli.main(["extended", "--output", str(tmp_path), "--names", "original", "changed",
              "--clip-batch-size", "1"])
    summary = pd.read_csv(root / "results_extended.csv").set_index("variant")
    assert summary.loc["changed", "clip_image_image_cosine_mean"] == pytest.approx(0.)
    assert summary.loc["original", "clip_image_text_cosine_mean"] == pytest.approx(1.)
