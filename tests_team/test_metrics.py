"""fid split, FID and CLIP score contracts. Injected extractors: no downloads."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from team_pipeline import metrics
from team_pipeline.data import load_protocol, make_protocol, validate_pairs


def write_protocol(tmp_path, protocol):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path


def test_protocol_accepts_a_disjoint_class_balanced_fid_split(tmp_path):
    protocol = make_protocol(labels=[1, 2], calibration_per_class=1,
                             validation_per_class=1, test_per_class=1, fid_per_class=1)
    fid = protocol["splits"]["fid"]
    assert sorted(label for label, _ in fid) == list(range(1000))
    last_test_seed = max(seed for _, seed in protocol["splits"]["test"])
    assert min(seed for _, seed in fid) == last_test_seed + 1
    assert load_protocol(write_protocol(tmp_path, protocol)) == protocol
    validate_pairs(fid)


def test_default_protocol_file_has_the_agreed_fid_split():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "team_protocol.json")
    assert len(protocol["splits"]["fid"]) == 1000
    assert protocol["splits"]["fid"][0] == [0, 10096]
    assert protocol["splits"]["fid"][-1] == [999, 11095]
    assert protocol == make_protocol(fid_per_class=1)


def test_protocol_without_fid_is_unchanged_and_overlap_or_unknown_splits_fail(tmp_path):
    plain = make_protocol(labels=[1, 2], calibration_per_class=1,
                          validation_per_class=1, test_per_class=1)
    assert "fid" not in plain["splits"]
    with_fid = make_protocol(labels=[1, 2], calibration_per_class=1,
                             validation_per_class=1, test_per_class=1, fid_per_class=1)
    for name in ("calibration", "validation", "test"):
        assert plain["splits"][name] == with_fid["splits"][name]
    leaked = copy.deepcopy(with_fid)
    leaked["splits"]["fid"][0][1] = leaked["splits"]["validation"][0][1]
    with pytest.raises(ValueError, match="leakage"):
        load_protocol(write_protocol(tmp_path, leaked))
    unknown = copy.deepcopy(plain)
    unknown["splits"]["extra"] = [[1, 99999]]
    with pytest.raises(ValueError, match="splits"):
        load_protocol(write_protocol(tmp_path, unknown))
    with pytest.raises(ValueError):
        make_protocol(fid_per_class=-1)


# ---------------------------------------------------------------- FID

def test_frechet_distance_matches_the_closed_form():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 6))
    assert metrics.fid_from_features(a, a) == pytest.approx(0, abs=1e-6)
    # Same covariance, shifted mean: FID = squared mean shift.
    shift = np.array([1, 2, 0, 0, 0, 0.5])
    mu, sigma = metrics.feature_statistics(a)
    assert metrics.frechet_distance(mu, sigma, mu + shift, sigma) == pytest.approx(shift @ shift, rel=1e-9)
    # Diagonal covariances: tr(S1)+tr(S2)-2 tr sqrt(S1 S2) = sum (sqrt s1 - sqrt s2)^2.
    s1, s2 = np.diag([1.0, 4.0]), np.diag([9.0, 1.0])
    expected = (1 - 3) ** 2 + (2 - 1) ** 2
    assert metrics.frechet_distance(np.zeros(2), s1, np.zeros(2), s2) == pytest.approx(expected)


def test_self_fid_is_zero_for_rank_deficient_covariance():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 64))  # N < D: covariance is singular, as with FID-1K
    assert metrics.fid_from_features(a, a) == pytest.approx(0, abs=1e-6)
    b = a + 0.5
    assert metrics.fid_from_features(a, b) == pytest.approx(0.25 * 64, rel=1e-6)


def test_frechet_distance_rejects_bad_inputs():
    with pytest.raises(ValueError):
        metrics.frechet_distance(np.zeros(2), np.eye(2), np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        metrics.feature_statistics(np.zeros((1, 4)))
    with pytest.raises(ValueError):
        metrics.frechet_distance(np.zeros(2), np.full((2, 2), np.nan), np.zeros(2), np.eye(2))


def test_to_uint8_covers_the_full_range():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0]).view(1, 1, 2, 2)
    out = metrics.to_uint8(x)
    assert out.dtype == torch.uint8 and out.flatten().tolist() == [0, 128, 255, 255]


def make_run(root, name, labels, pixel, runtime="rt", config=None):
    """A finished run folder; image pixels are constant per image (list `pixel`)."""
    folder = root / name
    folder.mkdir(parents=True)
    pairs = [(label, 100 + i) for i, label in enumerate(labels)]
    config = config or dict(image_size=8)
    (folder / "manifest.json").write_text(json.dumps(dict(
        pairs=pairs, config=config, shared_code_sha256="h", runtime=runtime)))
    pd.DataFrame([(i, l, s) for i, (l, s) in enumerate(pairs)],
                 columns=["index", "label", "seed"]).to_csv(folder / "samples.csv", index=False)
    for i, value in enumerate(pixel):
        torch.save(torch.full((1, 3, 8, 8), float(value)), folder / f"image_{i:04d}.pt")


def fake_extractor(x):
    assert x.dtype == torch.uint8
    return (x.float().flatten(1)[:, ::12],)  # 16 features per image


def test_fid_report_writes_files_and_baseline_is_zero(tmp_path):
    rng = np.random.default_rng(2)
    labels = list(range(24))
    base = rng.uniform(-0.8, 0.8, 24)
    make_run(tmp_path, "original", labels, base)
    make_run(tmp_path, "same", labels, base)
    make_run(tmp_path, "shifted", labels, np.clip(base + 0.15, -1, 1))
    report = metrics.fid_report(tmp_path, ["same", "shifted"], extractor=fake_extractor).set_index("variant")
    assert report.loc["original", "fid_vs_original"] == pytest.approx(0, abs=1e-6)
    assert report.loc["same", "fid_vs_original"] == pytest.approx(0, abs=1e-6)
    assert report.loc["shifted", "fid_vs_original"] > 1.0
    saved = json.loads((tmp_path / "shifted" / "fid.json").read_text())
    assert saved["images"] == 24 and saved["class_count"] == 24
    assert saved["metric"] == "FID-24 (relative use only)"
    assert saved["rank_deficient"] is False
    assert "fid_vs_adm_imagenet256" not in saved
    assert (tmp_path / "shifted" / "inception_features.npy").exists()


def test_fid_metadata_marks_small_feature_covariance_rank_deficient(tmp_path):
    labels = list(range(8))
    pixels = np.linspace(-0.8, 0.8, len(labels))
    make_run(tmp_path, "original", labels, pixels)
    saved = metrics.fid_report(tmp_path, [], extractor=fake_extractor).iloc[0]
    assert saved["metric"] == "FID-8 (relative use only)"
    assert bool(saved["rank_deficient"]) is True
    assert saved["class_count"] == 8


def test_fid_report_with_reference_statistics(tmp_path):
    rng = np.random.default_rng(3)
    base = rng.uniform(-0.8, 0.8, 24)
    make_run(tmp_path, "original", list(range(24)), base)
    features = metrics.inception_features(
        torch.cat([torch.load(tmp_path / "original" / f"image_{i:04d}.pt") for i in range(24)]),
        fake_extractor)
    mu, sigma = metrics.feature_statistics(features)
    np.savez(tmp_path / "ref.npz", mu=mu, sigma=sigma)
    report = metrics.fid_report(tmp_path, [], reference_stats=tmp_path / "ref.npz",
                                extractor=fake_extractor)
    assert report.loc[0, "fid_vs_adm_imagenet256"] == pytest.approx(0, abs=1e-6)
    np.savez(tmp_path / "bad.npz", arr_0=np.zeros(3))
    with pytest.raises(ValueError, match="mu/sigma"):
        metrics.load_reference_statistics(tmp_path / "bad.npz")


def test_reports_reject_runs_from_a_different_setup(tmp_path):
    make_run(tmp_path, "original", [1, 2, 3], [0.1, 0.2, 0.3])
    make_run(tmp_path, "other_env", [1, 2, 3], [0.1, 0.2, 0.3], runtime="different")
    make_run(tmp_path, "other_cfg", [1, 2, 3], [0.1, 0.2, 0.3], config=dict(image_size=8, cfg_scale=9))
    for name in ("other_env", "other_cfg"):
        with pytest.raises(ValueError, match="differs"):
            metrics.fid_report(tmp_path, [name], extractor=fake_extractor)
    (tmp_path / "other_env" / "image_0002.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        metrics.load_variant_images(tmp_path, "other_env")
    (tmp_path / "other_env" / "samples.csv").write_text("index,label,seed\n0,1,100\n")
    with pytest.raises(ValueError, match="Incomplete"):
        metrics.load_variant_images(tmp_path, "other_env")


# ---------------------------------------------------------------- CLIP

def test_clip_cosine_and_preprocessing():
    a = torch.eye(3)
    assert metrics.clip_scores_from_embeddings(a, a).tolist() == pytest.approx([1, 1, 1])
    assert metrics.clip_scores_from_embeddings(a, a.roll(1, 0)).abs().max() < 1e-6
    images = torch.zeros(2, 3, 256, 256)
    out = metrics.clip_preprocess(images)
    assert out.shape == (2, 3, 224, 224)
    expected = (0.5 - torch.tensor(metrics.CLIP_MEAN)) / torch.tensor(metrics.CLIP_STD)
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_clip_report_scores_matching_images_higher_than_shuffled(tmp_path):
    classes = ["c0", "c1", "c2", "c3"]
    labels = [0, 1, 2, 3] * 3
    pixel = [-1 + 0.5 * l for l in labels]           # image encodes its true class
    make_run(tmp_path, "original", labels, pixel)
    wrong = [-1 + 0.5 * ((l + 1) % 4) for l in labels]  # every image shows the wrong class
    make_run(tmp_path, "wrong", labels, wrong)

    def encode_image(images):
        index = ((images.mean(dim=(1, 2, 3)) + 1) / 0.5).round().long()
        return torch.nn.functional.one_hot(index, 4).float()

    def encode_text(prompts):
        assert all(p.startswith("a photo of a c") for p in prompts)
        return torch.nn.functional.one_hot(torch.tensor([int(p[-1]) for p in prompts]), 4).float()

    report = metrics.clip_report(tmp_path, ["original", "wrong"], encode_image, encode_text,
                                 info=dict(model="fake"), class_names=classes).set_index("variant")
    assert report.loc["original", "clip_cosine_mean"] == pytest.approx(1.0)
    assert report.loc["wrong", "clip_cosine_mean"] == pytest.approx(0.0)
    assert report.loc["original", "clip_score"] == pytest.approx(100.0)
    saved = json.loads((tmp_path / "wrong" / "clip.json").read_text())
    assert saved["model"] == "fake" and saved["prompt"] == "a photo of a {}"
    assert saved["class_count"] == 4 and saved["images"] == 12
    assert len(pd.read_csv(tmp_path / "original" / "clip_scores.csv")) == 12
    # Shuffling the text labels lowers a perfectly aligned score.
    image_emb = encode_image(torch.cat([torch.load(tmp_path / "original" / f"image_{i:04d}.pt")
                                        for i in range(12)]))
    text_emb = encode_text([f"a photo of a c{l}" for l in labels])
    shuffled = text_emb[torch.randperm(12, generator=torch.Generator().manual_seed(0))]
    aligned = metrics.clip_scores_from_embeddings(image_emb, text_emb).mean()
    assert metrics.clip_scores_from_embeddings(image_emb, shuffled).mean() < aligned


def test_imagenet_class_names_are_indexed_like_dit_labels():
    names = metrics.imagenet_class_names()
    assert len(names) == 1000
    assert "tench" in names[0] and "golden retriever" in names[207]
