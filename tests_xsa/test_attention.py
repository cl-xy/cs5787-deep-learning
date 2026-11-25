"""XSA projection and timm attention checks; no pretrained downloads."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from xsa_method import xsa_attention_scope, xsa_filter_heads


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
        import inspect
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


