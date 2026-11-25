"""Exclusive self-attention projection and scoped timm hooks."""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def xsa_filter_heads(output, value, eps=1e-8):
    """Remove each token/head's current self-value direction; last axis is head_dim."""
    if output.shape != value.shape or output.ndim != 4:
        raise ValueError("XSA needs matching four-dimensional head output/value tensors")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("XSA epsilon must be finite and positive")
    # FP32 projection also makes eps safe if this helper is used with FP16/BF16.
    dtype = torch.float32 if output.dtype in {torch.float16, torch.bfloat16} else output.dtype
    o, v = output.to(dtype), value.to(dtype)
    v_normalized = F.normalize(v, dim=-1, eps=eps)
    z = o - (o * v_normalized).sum(dim=-1, keepdim=True) * v_normalized
    return z.to(output.dtype)


@contextmanager
def xsa_attention_scope(attention, enabled=True, eps=1e-8):
    """Filter exactly one ordinary timm Attention call, with no persistent flags.

    The original forward computes fresh Q/K/V and attention (including its fused
    or unfused path, Q/K norms, scaling and masks). At the head-concatenation
    boundary we view the tensor as separate heads, filter, and concatenate again.
    This is exactly the pre-concatenation projection: reshape mixes no values.
    timm's optional post-attention norm/gate and output projection run afterward.
    Hooks and this call's value tensor are always discarded, even on failure.
    Intended for sequential eager inference, not concurrent calls
    to the same module or torch.compile.
    """
    if not enabled:
        yield
        return
    from timm.models.vision_transformer import Attention
    if type(attention) is not Attention:
        raise TypeError("XSA adapter supports the standard timm Attention used by official DiT")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("XSA epsilon must be finite and positive")
    state = {}
    counts = {"qkv": 0, "filter": 0}

    def capture_value(module, args, qkv):
        if counts["qkv"]:
            raise RuntimeError("Expected exactly one fresh QKV evaluation per XSA call")
        b, n, three_c = qkv.shape
        h = attention.num_heads
        if three_c % (3 * h) or three_c // 3 != attention.proj.in_features:
            raise RuntimeError("Unsupported timm QKV/head layout")
        # Only THIS evaluation's V is retained until its attention output arrives.
        state["value"] = qkv.reshape(b, n, 3, h, three_c // (3 * h))[:, :, 2]
        counts["qkv"] += 1

    def filter_before_projection(module, args):
        if "value" not in state:
            raise RuntimeError("XSA output did not follow a fresh QKV evaluation")
        value = state.pop("value")
        output = args[0]
        if output.shape != (*value.shape[:2], value.shape[2] * value.shape[3]):
            raise RuntimeError("Unsupported timm attention-output layout")
        heads = output.reshape_as(value)
        filtered = xsa_filter_heads(heads, value, eps).reshape_as(output)
        counts["filter"] += 1
        return (filtered, *args[1:])

    handles = []
    try:
        handles.append(attention.qkv.register_forward_hook(capture_value))
        # New timm has a post-head-concatenation norm; old timm goes straight to
        # proj. Filtering BEFORE that norm preserves it, rather than bypassing it.
        boundary = getattr(attention, "norm", None)
        if boundary is None:
            boundary = attention.proj
        handles.append(boundary.register_forward_pre_hook(filter_before_projection))
        yield
        if counts != {"qkv": 1, "filter": 1}:
            raise RuntimeError("XSA hooks did not observe exactly one attention evaluation")
    finally:
        for handle in handles:
            handle.remove()
        state.clear()


