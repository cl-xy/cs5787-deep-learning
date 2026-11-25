"""XSA reuse, fixed block selection and the optional Clean-FID/TPIPS export.

CPU only, no pretrained downloads. The real-DiT smoke test runs only when
DIT_TEST_REPO points at the official DiT checkout.
"""
import inspect
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from team_pipeline import cli, experiment as exp, xsa
from team_pipeline.extended_metrics import evaluate_fid_tpips
from team_pipeline.methods import ScheduledBlocks
from team_pipeline.xsa import xsa_attention_scope, xsa_filter_heads


def test_xsa_projection():
    with torch.random.fork_rng():
        torch.manual_seed(184)
        o, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
        z = xsa_filter_heads(o, v)
        torch.testing.assert_close((z * F.normalize(v, dim=-1)).sum(-1),
                                   torch.zeros(2, 3, 5), atol=1e-6, rtol=0)
        # Zero/epsilon-clamped vectors must remain finite; zero V leaves O alone.
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for magnitude in [0.0, 1e-30, 1e-10, 1e-7]:
                small = torch.full_like(v, magnitude).to(dtype)
                result = xsa_filter_heads(o.to(dtype), small)
                assert torch.isfinite(result).all()
                if magnitude == 0:
                    torch.testing.assert_close(result, o.to(dtype), atol=0, rtol=0)


def test_xsa_attention_paths():
    from timm.models.vision_transformer import Attention
    with torch.random.fork_rng():
        torch.manual_seed(185)
        attn = Attention(24, num_heads=3, qkv_bias=True).eval().requires_grad_(False)
        x = torch.randn(2, 5, 24)
        # Both timm attention kernels stay in use; no change when XSA is off.
        paths = [False, True] if hasattr(attn, "fused_attn") else [None]
        for fused in paths:
            if fused is not None:
                attn.fused_attn = fused
            expected_off = attn(x)
            with xsa_attention_scope(attn, enabled=False):
                actual_off = attn(x)
            torch.testing.assert_close(actual_off, expected_off, atol=0, rtol=0)
            with xsa_attention_scope(attn):
                actual_on = attn(x)
            qkv = attn.qkv(x).reshape(2, 5, 3, 3, 8).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = attn.q_norm(q) if hasattr(attn, "q_norm") else q
            k = attn.k_norm(k) if hasattr(attn, "k_norm") else k
            o = F.scaled_dot_product_attention(q, k, v) if fused else (
                ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1) @ v)
            expected_on = xsa_filter_heads(o, v).transpose(1, 2).reshape(2, 5, 24)
            norm = getattr(attn, "norm", nn.Identity())
            expected_on = attn.proj_drop(attn.proj(norm(expected_on)))
            torch.testing.assert_close(actual_on, expected_on, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(attn(x), expected_off, atol=0, rtol=0)
        # Failure during a replacement must not leave XSA active on later calls.
        boundary = getattr(attn, "norm", attn.proj)
        before = (len(attn.qkv._forward_hooks), len(boundary._forward_pre_hooks))
        with pytest.raises(RuntimeError, match="deliberate"):
            with xsa_attention_scope(attn):
                raise RuntimeError("deliberate")
        assert before == (len(attn.qkv._forward_hooks), len(boundary._forward_pre_hooks))
        # Newer timm additionally supports masks, Q/K norms, post-attention norm
        # and an optional attention gate: exercise them without changing forward.
        parameters = inspect.signature(Attention).parameters
        if {"qk_norm", "scale_norm", "gated"} <= parameters.keys():
            normalized = Attention(24, 3, qkv_bias=True, qk_norm=True,
                                   scale_norm=True, gated=True, norm_layer=nn.LayerNorm).eval()
            normalized.requires_grad_(False)
            mask = torch.ones(5, 5, dtype=torch.bool).tril()
            for fused in [False, True]:
                normalized.fused_attn = fused
                off = normalized(x, attn_mask=mask)
                with xsa_attention_scope(normalized, enabled=False):
                    torch.testing.assert_close(normalized(x, attn_mask=mask), off, atol=0, rtol=0)
                with xsa_attention_scope(normalized):
                    actual = normalized(x, attn_mask=mask)
                q, k, v = normalized.qkv(x).reshape(2, 5, 3, 3, 8).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = normalized.q_norm(q), normalized.k_norm(k)
                if fused:
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    logits = ((q * normalized.scale) @ k.transpose(-2, -1)).masked_fill(~mask, -float("inf"))
                    o = logits.softmax(-1) @ v
                heads = xsa_filter_heads(o, v).transpose(1, 2).reshape(2, 5, 24)
                expected = normalized.norm(heads) * normalized.gate(x).sigmoid()
                expected = normalized.proj_drop(normalized.proj(expected))
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_xsa_schedule_and_registration(monkeypatch):
    from timm.models.vision_transformer import Attention

    class ProbeBlock(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.attn = Attention(12, num_heads=3, qkv_bias=True)
            self.inputs = []

        def forward(self, x, c):
            self.inputs.append((x.detach().clone(), c))
            return x + self.attn(x) + c[:, None, :]

    with torch.random.fork_rng():
        torch.manual_seed(186)
        blocks = [ProbeBlock(i) for i in range(28)]
        schedule = ScheduledBlocks(blocks, [4, 7], "xsa_reuse", alpha=1.0)
        schedule.eval().requires_grad_(False)
        expected_sources = [0, 1, 2, 3, 3, 5, 6, 6, *range(8, 28)]
        metadata = schedule.metadata()
        assert [s["source"] for s in metadata["execution_schedule"]] == expected_sources
        assert [s["slot"] for s in metadata["execution_schedule"] if s["xsa"]] == [4, 7]
        assert metadata["index_base"] == 0 and metadata["retained"] == [i for i in range(28) if i not in [4, 7]]
        assert schedule.retained["3"] is blocks[3] and schedule.retained["6"] is blocks[6]
        retained_ids = {id(p) for b in blocks if b.index not in [4, 7] for p in b.parameters()}
        discarded_ids = {id(p) for b in [blocks[4], blocks[7]] for p in b.parameters()}
        registered_ids = {id(p) for p in schedule.parameters()}
        assert registered_ids == retained_ids and not registered_ids & discarded_ids
        assert len(schedule.retained) == 26 and len(schedule) == 28
        assert all(not p.requires_grad for p in schedule.parameters())
        assert not any(key.split(".")[1] in {"4", "7"} for key in schedule.state_dict())

        events, qkv_inputs, ordinary_outputs = [], {3: [], 6: []}, {}
        original_filter = xsa.xsa_filter_heads

        def record_filter(o, v, eps=1e-8):
            events.append(active_slot[0])
            return original_filter(o, v, eps)

        monkeypatch.setattr(xsa, "xsa_filter_heads", record_filter)
        handles = []
        for index in [3, 6]:
            handles.append(blocks[index].attn.qkv.register_forward_pre_hook(
                lambda module, args, index=index: qkv_inputs[index].append(args[0].detach().clone())))
        x, c = torch.randn(1, 5, 12) * 0.01, torch.randn(1, 12) * 0.01
        active_slot = [None]
        try:
            for slot, call in enumerate(schedule):
                active_slot[0] = slot
                prior = x.clone()
                x = call(x, c)
                if slot in [3, 6]:
                    ordinary_outputs[slot] = x.clone()
                # Each additional call receives the immediately preceding output.
                if slot in [4, 7]:
                    source = 3 if slot == 4 else 6
                    torch.testing.assert_close(blocks[source].inputs[-1][0], prior)
                    assert blocks[source].inputs[-1][1] is c
            assert events == [4, 7]
            assert torch.isfinite(x).all()
            for index in [3, 6]:
                assert len(qkv_inputs[index]) == 2
                assert not torch.equal(*qkv_inputs[index])
                # The ordinary retained call remains the exact original block path.
                input_x, input_c = blocks[index].inputs[0]
                ordinary = blocks[index](input_x, input_c)
                torch.testing.assert_close(ordinary, ordinary_outputs[index], atol=0, rtol=0)
        finally:
            for handle in handles:
                handle.remove()


def test_xsa_checkpoint_and_fixed_selection(tmp_path, monkeypatch):
    from timm.models.vision_transformer import Attention

    class SmallModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([nn.ModuleDict({"attn": Attention(12, 3)}) for _ in range(28)])

    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(DiT_XL_2=SmallModel))
    model = SmallModel()
    model.blocks = ScheduledBlocks(model.blocks, [4, 7], "xsa_reuse", 1.0, xsa_eps=1e-7)
    path = tmp_path / "compact.pt"
    stats = exp.checkpoint_stats(model, path, keep=True)
    loaded = exp.load_compact_checkpoint(path)
    assert loaded.blocks.metadata() == model.blocks.metadata()
    assert stats["unique_blocks"] == 26 and stats["block_evaluations_per_forward"] == 28
    assert all(not p.requires_grad for p in loaded.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    selection = exp.write_fixed_selection(tmp_path)
    assert selection["removed"] == [4, 7] and selection["index_base"] == 0
    assert exp.load_fixed_selection(tmp_path) == selection
    assert exp.write_fixed_selection(tmp_path, [4, 7]) == selection  # Idempotent.
    for bad in [[], [0, 4], [7, 4], [4, 4], [28], [4.0]]:
        with pytest.raises(ValueError):
            exp.write_fixed_selection(tmp_path / "other", bad)
    exp.write_json(tmp_path / "profile/selection.json", dict(selection, removed=[3, 6]))
    with pytest.raises(ValueError, match="frozen"):
        exp.write_fixed_selection(tmp_path)


def test_cli_routes_fixed_selection_and_alpha_rules(tmp_path, monkeypatch):
    pairs = [[207, 10016], [360, 10017]]
    protocol = {"splits": {"validation": pairs}}
    config = {"device": "cpu"}
    monkeypatch.setattr(cli, "load_run", lambda args: (tmp_path, protocol, config))
    calls = []
    monkeypatch.setattr(cli, "run_variant", lambda *a, **k: calls.append((a, k)))
    cli.main(["select", "--output", str(tmp_path)])
    assert exp.load_fixed_selection(tmp_path)["removed"] == [4, 7]
    with pytest.raises(ValueError, match="frozen"):
        cli.main(["select", "--output", str(tmp_path), "--removed", "3", "6"])
    for mode in ["removal", "loop", "damped", "xsa_reuse"]:
        cli.main(["run", "--output", str(tmp_path), "--method", mode, "--name", mode])
        (config_arg, pairs_arg, output), kwargs = calls[-1]
        assert config_arg is config and pairs_arg == pairs
        assert output == tmp_path / "validation" / mode
        assert kwargs["mode"] == mode and kwargs["removed"] == [4, 7]
        assert kwargs["alpha"] == (0.5 if mode == "damped" else 1.0)
        assert kwargs["model_transform"] is None
        assert kwargs["method_metadata"]["selection"]["removed"] == [4, 7]
    cli.main(["run", "--output", str(tmp_path), "--method", "damped", "--name", "d25", "--alpha", "0.25"])
    assert calls[-1][1]["alpha"] == 0.25
    cli.main(["run", "--output", str(tmp_path), "--method", "xsa_reuse", "--name", "x1", "--alpha", "1"])
    assert calls[-1][1]["alpha"] == 1.0
    with pytest.raises(ValueError, match="alpha"):
        cli.main(["run", "--output", str(tmp_path), "--method", "xsa_reuse", "--name", "bad", "--alpha", "0.5"])
    with pytest.raises(ValueError, match="alpha"):
        cli.main(["run", "--output", str(tmp_path), "--method", "original", "--name", "original", "--alpha", "1"])


def _tiny_run(split_root, names, monkeypatch):
    """Generate real (tiny) artifacts with every built-in schedule, without downloads."""
    vae = nn.Linear(1, 1)
    vae.config = SimpleNamespace()

    def prepare(config, mode, removed, alpha, model_transform=None):
        model = nn.Module()
        blocks = [nn.Linear(1, 1) for _ in range(8)]
        model.blocks = ScheduledBlocks(blocks, removed, mode, alpha)
        return model, vae, None

    monkeypatch.setattr(exp, "_prepare", prepare)
    monkeypatch.setattr(exp, "warmup", lambda *a: None)
    monkeypatch.setattr(exp, "generate", lambda model, vae, diffusion, label, seed, cfg:
                        torch.full((1, 3, 8, 8), (seed % 100) / 100 + (len(model.blocks) - 8) / 50))
    config = dict(device="cpu", image_size=8, cfg_scale=4)
    pairs = [(207, 10016), (360, 10017)]
    for name, mode in zip(names, ["original", "removal", "loop", "damped", "xsa_reuse"]):
        exp.run_variant(config, pairs, split_root / name, mode=mode,
                        removed=() if mode == "original" else (4, 7),
                        alpha=0.5 if mode == "damped" else 1.0)
    return exp.compare_results(split_root, names, metric=lambda x, y: (x - y).square().mean())


def test_extended_metrics_disabled_exports_lpips_without_optional_packages(tmp_path, monkeypatch):
    names = ["original", "member1_removal", "member2_loop", "member3_damped", "member4_xsa_reuse"]
    split_root = tmp_path / "validation"
    summary = _tiny_run(split_root, names, monkeypatch)
    xsa_manifest = json.loads((split_root / "member4_xsa_reuse/manifest.json").read_text())
    assert xsa_manifest["schedule"]["mode"] == "xsa_reuse"
    assert [s["slot"] for s in xsa_manifest["schedule"]["execution_schedule"] if s["xsa"]] == [4, 7]
    # Disabled extras must never import TPIPS or Clean-FID.
    monkeypatch.setitem(sys.modules, "tpips", None)
    monkeypatch.setitem(sys.modules, "cleanfid", None)
    cli.main(["extended", "--no-clip", "--output", str(tmp_path), "--split", "validation", "--names", *names])
    assert len(summary) == len(names)
    pd.testing.assert_frame_equal(pd.read_csv(split_root / "results.csv"),
                                  pd.read_csv(split_root / "results_extended.csv"))
    metadata = json.loads((split_root / "evaluation_extended.json").read_text())
    assert metadata["fid"] is None and metadata["tpips"] is None
    # Incomplete artifacts are still rejected when extras are off.
    samples = split_root / "member4_xsa_reuse/samples.csv"
    pd.read_csv(samples).iloc[:1].to_csv(samples, index=False)
    with pytest.raises(ValueError, match="sample order"):
        evaluate_fid_tpips(split_root, names, tpips_factors=())


def test_real_dit_xsa_reuse_forward():
    """Official DiT block + timm Attention: XSA changes only replacement slots."""
    repo = os.environ.get("DIT_TEST_REPO")
    if not repo or not (Path(repo) / "models.py").exists():
        pytest.skip("Set DIT_TEST_REPO to test against the official DiT source.")
    sys.path.insert(0, str(Path(repo).resolve()))
    from models import DiT
    import copy
    with torch.random.fork_rng():
        torch.manual_seed(77)
        original = DiT(input_size=8, patch_size=2, hidden_size=24, depth=8, num_heads=3).eval()
        with torch.no_grad():
            for block in original.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
            nn.init.normal_(original.final_layer.linear.weight, std=0.05)
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([500, 500])
        y = torch.tensor([207, 1000])
        outputs = {}
        for mode in ["loop", "xsa_reuse"]:
            model = copy.deepcopy(original)
            model.blocks = ScheduledBlocks(model.blocks, [2, 5], mode, 1.0)
            with torch.inference_mode():
                outputs[mode] = model.forward_with_cfg(x, t, y, 4.0)
        assert torch.isfinite(outputs["xsa_reuse"]).all()
        assert not torch.equal(outputs["loop"], outputs["xsa_reuse"])


def test_real_sampler_runs_all_five_schedules(tmp_path, monkeypatch):
    """load_model -> run_variant -> compare_results with the official DiT sampler (tiny, random)."""
    repo = os.environ.get("DIT_TEST_REPO")
    if not repo or not (Path(repo) / "models.py").exists():
        pytest.skip("Set DIT_TEST_REPO to test against the official DiT source.")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # Keep the pinned checkout clean.
    monkeypatch.syspath_prepend(str(Path(repo).resolve()))
    import models

    def small_dit(**kwargs):
        return models.DiT(hidden_size=24, depth=8, num_heads=3, **kwargs)

    monkeypatch.setattr(models, "DiT_XL_2", small_dit)
    with torch.random.fork_rng():
        torch.manual_seed(24)
        model = small_dit(input_size=32, num_classes=1000)
        # Nonzero adaLN gates/output make this a real attention/sampler exercise.
        with torch.no_grad():
            for block in model.blocks:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.01)
            nn.init.normal_(model.final_layer.linear.weight, std=0.01)
        checkpoint = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), checkpoint)
    del model

    class Decoder(nn.Module):
        config = SimpleNamespace()

        def decode(self, latent):
            return SimpleNamespace(sample=F.interpolate(latent[:, :3], size=256).tanh())

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoencoderKL=SimpleNamespace(from_pretrained=lambda *a, **kw: Decoder())))
    config = dict(device="cpu", image_size=256, cfg_scale=4.0,
                  checkpoint=str(checkpoint), vae_id="offline-decoder", sampling_steps=2)
    pairs = [(207, 10016), (360, 10017)]
    names = ["original", "member1_removal", "member2_loop", "member3_damped", "member4_xsa_reuse"]
    split_root = tmp_path / "run" / "validation"
    for name, mode in zip(names, ["original", "removal", "loop", "damped", "xsa_reuse"]):
        exp.run_variant(config, pairs, split_root / name, mode=mode,
                        removed=() if mode == "original" else (4, 7),
                        alpha=0.5 if mode == "damped" else 1.0)
    summary = exp.compare_results(split_root, names,
                                  metric=lambda x, y: (x - y).square().mean()).set_index("variant")
    assert summary["images"].eq(2).all()
    assert summary.loc["original", "lpips_mean"] == 0
    assert summary.loc["member4_xsa_reuse", "lpips_mean"] != summary.loc["member2_loop", "lpips_mean"]
    assert summary.loc["member4_xsa_reuse", "unique_blocks"] == 6
    exp.comparison_grid(split_root, names, pairs)


def _fake_imagenet(monkeypatch, labels, per_class, calls):
    """Install a fake ``datasets`` module yielding non-square RGBA images in mixed order."""
    from PIL import Image as PILImage

    def load_dataset(name, split, streaming, token):
        calls.append((name, split, streaming, token))
        for i in range(per_class):
            for label in [*labels, 999]:  # 999 is never generated: must be skipped.
                yield dict(label=label, image=PILImage.new("RGBA", (40, 24), (label % 255, i, 0, 255)))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))


def test_imagenet_reference_build_reuse_and_fid(tmp_path, monkeypatch):
    from PIL import Image as PILImage
    from team_pipeline import extended_metrics as em
    names = ["original", "member1_removal", "member2_loop", "member3_damped", "member4_xsa_reuse"]
    split_root = tmp_path / "run" / "validation"
    _tiny_run(split_root, names, monkeypatch)
    calls = []
    _fake_imagenet(monkeypatch, [207, 360], 3, calls)
    references = tmp_path / "refs"
    with pytest.raises(ValueError, match="outside"):
        em.build_imagenet_reference(split_root, split_root, images_per_class=2)
    cli.main(["fid-reference", "--output", str(tmp_path / "run"), "--reference-root", str(references),
              "--images-per-class", "2"])
    folder = next(references.iterdir())
    files = sorted(p.name for p in folder.glob("*.png"))
    assert files == [f"class_{c:04d}_{i:04d}.png" for c in [207, 360] for i in range(2)]
    with PILImage.open(folder / files[0]) as image:
        assert image.mode == "RGB" and image.size == (8, 8)
    record = json.loads((folder / "reference.json").read_text())
    assert record["labels"] == [207, 360] and "center crop to 8x8" in record["description"]
    assert calls == [("ILSVRC/imagenet-1k", "validation", False, True)]
    # A complete folder is reused without touching the dataset.
    assert em.build_imagenet_reference(split_root, references, images_per_class=2)[0] == folder
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Incomplete"):
        em.build_imagenet_reference(split_root, references, images_per_class=5)

    # FID path with a fake Clean-FID; description comes from reference.json.
    import numpy as np

    def features(files, **kwargs):
        return np.stack([np.r_[float(len(Path(f).name)), float(i), float(i * i % 7)]
                         for i, f in enumerate(files)])

    fid_module = SimpleNamespace(get_files_features=features,
                                 fid_from_feats=lambda a, b: float(np.abs(a.mean(0) - b.mean(0)).sum()))
    monkeypatch.setitem(sys.modules, "cleanfid", SimpleNamespace(fid=fid_module))
    monkeypatch.setitem(sys.modules, "cleanfid.fid", fid_module)
    monkeypatch.setitem(sys.modules, "cleanfid.features",
                        SimpleNamespace(build_feature_extractor=lambda *a, **k: object()))
    with pytest.warns(UserWarning, match="Small-sample"):
        cli.main(["extended", "--no-clip", "--output", str(tmp_path / "run"), "--names", *names,
                  "--fid-reference-dir", str(folder)])
    result = pd.read_csv(split_root / "results_extended.csv")
    assert result["fid"].notna().all()
    metadata = json.loads((split_root / "evaluation_extended.json").read_text())
    assert metadata["fid"]["reference_description"] == record["description"]
    assert metadata["fid"]["reference_images"] == 4
