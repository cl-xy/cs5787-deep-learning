"""Offline checks for resuming partial runs without changing their provenance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torchvision.utils import save_image

from resume_saved_method import resume_method


@pytest.fixture
def partial_run(tmp_path):
    root = tmp_path / "run"
    folder = root / "test/member2_loop"
    folder.mkdir(parents=True)
    pairs = [[0, 100 + i] for i in range(5)]
    config = dict(image_size=8, device="cpu", cfg_scale=4.0, shared_code_sha256="original-code")
    runtime = {"device": "cpu", "packages": {"torch": torch.__version__}}
    (root / "environment.json").write_text(json.dumps(runtime))
    schedule = dict(mode="loop", removed=[1], alpha=1.0, depth=4)
    manifest = dict(config=config, pairs=pairs, shared_code_sha256="original-code",
                    runtime=runtime, schedule=schedule, vae_commit="pinned-vae")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    generated, prepared = [], []

    def image_for(seed):
        return torch.full((1, 3, 8, 8), (seed - 102) / 10)

    for i in range(2):
        tensor = image_for(pairs[i][1])
        torch.save(tensor, folder / f"image_{i:04d}.pt")
        save_image(tensor, folder / f"image_{i:04d}.png", normalize=True, value_range=(-1, 1))
    pd.DataFrame([dict(index=i, label=label, seed=seed, latency_s=1.2 + i, device="cpu")
                  for i, (label, seed) in enumerate(pairs[:2])]).to_csv(folder / "samples.csv", index=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = SimpleNamespace(metadata=lambda: schedule.copy())

    def prepare(*args):
        prepared.append(args)
        vae = nn.Module()
        vae.config = SimpleNamespace(_commit_hash="pinned-vae")
        return Model(), vae, object()

    def generate(model, vae, diffusion, label, seed, cfg):
        generated.append(seed)
        return image_for(seed)

    namespace = dict(RUN_ARGS=object(), load_run=lambda args: (root, {"splits": {"test": pairs}}, config),
                     validate_pairs=lambda p: p, select_device=torch.device, _prepare=prepare,
                     generate=generate, warmup=lambda *args: None, synchronize=lambda *args: None,
                     _cleanup=lambda: None, preprocess_image=lambda image, size: image)
    return SimpleNamespace(root=root, folder=folder, pairs=pairs, namespace=namespace,
                           generated=generated, prepared=prepared, image_for=image_for)


def test_resume_preserves_samples_and_regenerates_only_uncommitted_tail(partial_run):
    run = partial_run
    preserved = {p: p.read_bytes() for p in run.folder.glob("image_000[01].*")}
    manifest_bytes = (run.folder / "manifest.json").read_bytes()
    old_rows = pd.read_csv(run.folder / "samples.csv")
    # An interrupted write produced a tensor with no timing row or PNG.
    torch.save(torch.zeros(1, 3, 8, 8), run.folder / "image_0002.pt")
    result = resume_method("test", "member2_loop", namespace=run.namespace)
    assert run.generated == [102, 103, 104]
    assert result["index"].tolist() == list(range(5))
    pd.testing.assert_frame_equal(result.iloc[:2][old_rows.columns], old_rows)
    assert all(path.read_bytes() == content for path, content in preserved.items())
    assert (run.folder / "manifest.json").read_bytes() == manifest_bytes
    for index, (_, seed) in enumerate(run.pairs):
        torch.testing.assert_close(torch.load(run.folder / f"image_{index:04d}.pt", weights_only=True),
                                   run.image_for(seed))
    assert (run.folder / "grid.png").exists()
    assert json.loads((run.folder / "resume_sessions.json").read_text())[0]["start_index"] == 2
    # A completed method is a no-op: no model reload and no regenerated images.
    result2 = resume_method("test", "member2_loop", namespace=run.namespace)
    assert len(result2) == 5 and len(run.prepared) == 1
    assert run.generated == [102, 103, 104]


def test_resume_can_be_interrupted_and_continued_again(partial_run):
    run = partial_run
    original_generate = run.namespace["generate"]

    def interrupted(*args):
        if args[4] == 103:
            raise KeyboardInterrupt
        return original_generate(*args)

    run.namespace["generate"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        resume_method("test", "member2_loop", namespace=run.namespace)
    assert pd.read_csv(run.folder / "samples.csv")["index"].tolist() == [0, 1, 2]
    run.namespace["generate"] = original_generate
    resume_method("test", "member2_loop", namespace=run.namespace)
    assert run.generated == [102, 103, 104]


@pytest.mark.parametrize("damage", ["pair", "tensor", "config", "runtime"])
def test_resume_rejects_inconsistent_saved_data_without_overwriting(partial_run, damage):
    run = partial_run
    if damage == "pair":
        frame = pd.read_csv(run.folder / "samples.csv")
        frame.loc[1, "seed"] = 999
        frame.to_csv(run.folder / "samples.csv", index=False)
    elif damage == "tensor":
        torch.save(torch.full((1, 3, 8, 8), float("nan")), run.folder / "image_0001.pt")
    else:
        path = run.folder / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest[damage] = {"changed": True}
        path.write_text(json.dumps(manifest))
    preserved = {p: p.read_bytes() for p in run.folder.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        resume_method("test", "member2_loop", namespace=run.namespace)
    assert not run.generated and not run.prepared
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_resume_validation_only_is_read_only(partial_run):
    run = partial_run
    preserved = {p: p.read_bytes() for p in run.folder.iterdir() if p.is_file()}
    frame = resume_method("test", "member2_loop", namespace=run.namespace, validate_only=True)
    assert len(frame) == 2 and not run.prepared and not run.generated
    assert {p: p.read_bytes() for p in run.folder.iterdir() if p.is_file()} == preserved
