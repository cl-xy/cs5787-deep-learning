**This is a reading guide for the XSA code in `DiT_Layer_Looping_GCP.ipynb`.**

The experiment's executable definitions stay unchanged. The notebook records their
exact source text to check that a resumed run uses the same code. The commented
copy below is documentation; it does not need to be executed or pasted over the
existing definitions.

**Yes: filtering happens separately for every token in every attention head.**
It subtracts a component from the head's output vector. It does not delete a head
or set the attention matrix's diagonal to zero. A particular output vector can
become zero if it points entirely along its own value direction.

Here are the words used in the comments:

| Word | Plain meaning here |
|---|---|
| Token | A vector describing one patch of the latent image. |
| Head | One of several parallel attention calculations, with its own learned projections. |
| Q / query | The vector used to ask which tokens are relevant. |
| K / key | The vector used to calculate how relevant a token is to a query. |
| V / value | The message/content a token contributes to the weighted mixture. |
| Attention output `o` | The mixture of value vectors that attention produces for a token and head. |
| Own value `v` | The value vector of that output's token in that same head. |
| Projection | Removing the part of `o` pointing along the direction of its own `v`. |
| Hook | A callback that runs before or after an existing layer. |
| Residual | The route that adds an update to the existing hidden features. |
| Conditioning `c` | Features representing the diffusion timestep and class label. |

For example, let both heads have attention output `(3, 4)`:

| Head | Its own value direction | Output after filtering |
|---|---|---|
| Head 1 | `(1, 0)` | `(0, 4)` |
| Head 2 | `(0, 1)` | `(3, 0)` |

Each head uses **its own** value direction. The code does this for all tokens and
heads in one array operation. It does not need an explicit Python loop over heads.
`dim=-1` means “operate within the features of each head,” so the other axes remain
separate. For normal-length value vectors, the remaining output is perpendicular
to the token's own value. Epsilon stabilizes very small values; zero V leaves the
attention output unchanged.

In your usual DiT-XL/2 sampling call, the shapes are:

| Array | Shape | How to read it |
|---|---|---|
| Combined Q/K/V | `[2, 256, 3456]` | Two CFG branches, 256 patches, three sets of 1152 features. |
| Separated values | `[2, 256, 16, 72]` | Each branch/patch has 16 heads, each with 72 features. |
| Concatenated head outputs | `[2, 256, 1152]` | The 16 head vectors sit side by side. |

The batch of two represents the class-conditioned and unconditioned predictions
used for classifier-free guidance. Filtering also keeps these two branches separate.

The following is the actual core implementation with explanatory comments added.
The Python operations, original docstrings and parameter defaults are unchanged.

```python
# XSA definitions — execute after the shared definitions and before tests/preparation.
# Lets us use a temporary setup with a Python "with" statement.
from contextlib import contextmanager
# Provides isfinite(), which checks that a number is not infinity or NaN.
import math
# Used later in this notebook cell to read/write the experiment records.
import json
# Used later to work with saved-run file paths.
from pathlib import Path
# PyTorch: the library that stores tensors and runs the neural network.
import torch
# nn contains PyTorch building blocks such as modules and ModuleDict.
from torch import nn
# F is a short name for helper operations, including vector normalization.
from torch.nn import functional as F


# THE ACTUAL XSA MATH. Run it separately for every token in every head.
# output = the messages attention has mixed together for each token.
# value = that same token's OWN value/message in each head.
# Both arrays have shape [batch, tokens, heads, features_per_head].
# eps = 0.00000001: a tiny safety floor when dividing by a vector length.
def xsa_filter_heads(output, value, eps=1e-8):
    """Remove each token/head's current self-value direction; last axis is head_dim."""
    # Refuse mismatched arrays: each output needs its matching token/head value.
    if output.shape != value.shape or output.ndim != 4:
        # Stop with an explanation instead of calculating with the wrong arrangement.
        raise ValueError("XSA needs matching four-dimensional head output/value tensors")
    # The safety floor must be a positive, ordinary finite number.
    if not math.isfinite(eps) or eps <= 0:
        # Reject an invalid safety floor.
        raise ValueError("XSA epsilon must be finite and positive")
    # FP32 projection also makes eps safe if this helper is used with FP16/BF16.
    # If inputs use 16-bit numbers, do this subtraction with safer 32-bit numbers.
    # Otherwise keep their precision. Your current experiment already uses FP32.
    dtype = torch.float32 if output.dtype in {torch.float16, torch.bfloat16} else output.dtype
    # Make working versions of both arrays with that numeric precision.
    o, v = output.to(dtype), value.to(dtype)
    # Turn each own-value vector into a direction by dividing by its length.
    # dim=-1 means work along the last dimension: FEATURES WITHIN ONE HEAD.
    # No averaging or summing across heads happens here.
    # The divisor is max(length, eps), so a zero-length vector does not cause NaN.
    v_normalized = F.normalize(v, dim=-1, eps=eps)
    # 1. o * v_normalized multiplies matching feature entries.
    # 2. sum(..., dim=-1) adds them into ONE dot product per token and head.
    # 3. keepdim=True leaves that number in a size-one final dimension.
    # 4. Multiplying it by v_normalized rebuilds the part pointing along own V.
    # 5. Subtract that part from o. Keep the part perpendicular to own V.
    # Different heads use THEIR OWN directions; entire heads are not deleted.
    z = o - (o * v_normalized).sum(dim=-1, keepdim=True) * v_normalized
    # Return the filtered array using the original numeric precision.
    return z.to(output.dtype)


# Makes the following function work as a temporary "with ...:" scope.
@contextmanager
# THE ADAPTER: temporarily put the XSA math inside an existing attention layer.
# A hook is a small callback: run some extra code before/after a module runs.
# We keep normal attention and add two temporary callbacks around it.
def xsa_attention_scope(attention, enabled=True, eps=1e-8):
    """Filter exactly one ordinary timm Attention call, with no persistent flags.

    The original forward computes fresh Q/K/V and attention (including its fused
    or unfused path, Q/K norms, scaling and masks). At the head-concatenation
    boundary we view the tensor as separate heads, filter, and concatenate again.
    This is exactly the pre-concatenation projection: reshape mixes no values.
    timm's optional post-attention norm/gate and output projection run afterward.
    Hooks and this call's value tensor are always discarded, even on failure.
    Intended for this notebook's sequential eager inference, not concurrent calls
    to the same module or torch.compile.
    """
    # If filtering is switched off, let the original code run unchanged.
    if not enabled:
        # Pause this setup function and let the code INSIDE the with statement run.
        # When that code finishes, execution returns to the line after this yield.
        yield
        # Finish this function; in the disabled branch there is nothing to clean up.
        return
    # Get the particular attention implementation that the official DiT uses.
    from timm.models.vision_transformer import Attention
    # Only accept the attention class whose internal array layout we understand.
    if type(attention) is not Attention:
        # Reject unsupported attention types instead of guessing how their data is laid out.
        raise TypeError("XSA adapter supports the standard timm Attention used by official DiT")
    # The safety floor must be a positive, ordinary finite number.
    if not math.isfinite(eps) or eps <= 0:
        # Reject an invalid safety floor.
        raise ValueError("XSA epsilon must be finite and positive")
    # An empty dictionary holding temporary data for THIS attention call only.
    state = {}
    # Count how often our two callbacks run. Each must run exactly once.
    # These are callback checks, NOT the block-repetition count or sampling steps.
    counts = {"qkv": 0, "filter": 0}

    # FIRST CALLBACK: runs AFTER the attention layer calculates combined Q/K/V.
    # module is the layer, args are its inputs, qkv is its computed output.
    # The callback interface supplies module/args; we only need qkv here.
    def capture_value(module, args, qkv):
        # A nonzero count means we unexpectedly captured Q/K/V twice in one scope.
        if counts["qkv"]:
            # Stop rather than pairing an attention output with the wrong set of values.
            raise RuntimeError("Expected exactly one fresh QKV evaluation per XSA call")
        # Read the array sizes: batch, tokens, and the combined Q/K/V feature width.
        # For the usual run: [2, 256, 3456], where 3456 = 3 * 1152.
        b, n, three_c = qkv.shape
        # Read the head count from the model itself: 16 for your DiT-XL/2.
        h = attention.num_heads
        # Check that Q/K/V divide into heads and match the later output layer width.
        if three_c % (3 * h) or three_c // 3 != attention.proj.in_features:
            # Stop if this attention implementation has a different internal arrangement.
            raise RuntimeError("Unsupported timm QKV/head layout")
        # Only THIS evaluation's V is retained until its attention output arrives.
        # Re-label the same entries as [batch, tokens, Q-or-K-or-V, heads, head_features].
        # Q is position 0, K is 1, V is 2. [:, :, 2] selects ONLY V.
        # Keep that V as [batch, tokens, heads, head_features] for the second callback.
        # This is the CURRENT call's V, not V cached from a previous block call.
        state["value"] = qkv.reshape(b, n, 3, h, three_c // (3 * h))[:, :, 2]
        # Record one successful capture. There is no returned replacement tensor:
        # normal attention keeps using its original Q/K/V output.
        counts["qkv"] += 1

    # SECOND CALLBACK: runs just before a module downstream of attention.
    # It gets to replace that module's input with the XSA-filtered version.
    def filter_before_projection(module, args):
        # We cannot filter safely unless the first callback captured matching V.
        if "value" not in state:
            # Stop if the expected first callback did not run.
            raise RuntimeError("XSA output did not follow a fresh QKV evaluation")
        # Fetch V and remove it from the dictionary, so it can only be consumed once.
        value = state.pop("value")
        # The first input argument is the attention output we want to change.
        output = args[0]
        # At this point heads sit side by side: [batch, tokens, heads * head_features].
        # Check that this matches the separate-head V captured earlier.
        if output.shape != (*value.shape[:2], value.shape[2] * value.shape[3]):
            # Stop if the output cannot be matched to the captured heads.
            raise RuntimeError("Unsupported timm attention-output layout")
        # View the output as separate heads again: [batch, tokens, heads, head_features].
        # This just re-labels dimensions; it does not mix one head with another.
        heads = output.reshape_as(value)
        # Run the XSA subtraction independently in every token/head.
        # Then put heads side by side again so the original next layer can read them.
        filtered = xsa_filter_heads(heads, value, eps).reshape_as(output)
        # Record one successful filtering operation.
        counts["filter"] += 1
        # Give PyTorch the replacement input arguments as a tuple.
        # Replace the first argument with filtered; keep any other arguments unchanged.
        return (filtered, *args[1:])

    # Keep the handles/tickets needed to remove the callbacks afterward.
    handles = []
    # Set up cleanup that will run even if attention fails or is interrupted.
    try:
        # Install callback 1 AFTER Q/K/V calculation and remember its removal handle.
        handles.append(attention.qkv.register_forward_hook(capture_value))
        # New timm has a post-head-concatenation norm; old timm goes straight to
        # proj. Filtering BEFORE that norm preserves it, rather than bypassing it.
        # Look for an optional normalization module just after the attention result.
        # We want to filter before normalization/gating/output projection continues.
        boundary = getattr(attention, "norm", None)
        # Older timm versions may not have this optional normalization module.
        if boundary is None:
            # For those versions, intercept the input to the output projection instead.
            boundary = attention.proj
        # Install callback 2 BEFORE that next module reads the attention output.
        handles.append(boundary.register_forward_pre_hook(filter_before_projection))
        # Pause this setup function and let the code INSIDE the with statement run.
        # When that code finishes, execution returns to the line after this yield.
        yield
        # After the with-body finishes, check that both callbacks ran exactly once.
        if counts != {"qkv": 1, "filter": 1}:
            # Stop if our assumptions about this attention call were not met.
            raise RuntimeError("XSA hooks did not observe exactly one attention evaluation")
    # This cleanup runs whether the with-body succeeded, failed, or was interrupted.
    finally:
        # Go through every installed callback handle.
        for handle in handles:
            # Uninstall that callback so the next ordinary block call stays ordinary.
            handle.remove()
        # Release any temporary value tensor still stored in the dictionary.
        state.clear()


# THE EXECUTION PLAN: stores retained blocks once and decides when to reuse them.
# An nn.Module is a PyTorch object that can own model weights and submodules.
class ScheduledBlocks(nn.Module):
    """Retain original indices and register each retained block exactly once."""
    # Build the plan from original blocks, removed positions, method, and numeric settings.
    def __init__(self, blocks, removed=(), mode="original", alpha="inverse_k", xsa_eps=1e-8):
        # Initialize PyTorch's module machinery before registering blocks.
        super().__init__()
        # Turn the incoming collection into a list we can count and walk through.
        blocks = list(blocks)
        # Remember the original number of positions: 28 in this model.
        self.depth = len(blocks)
        # Remember which method this container implements.
        self.mode = mode
        # Store the separate DAMPED method's setting as a policy name or number.
        # XSA calls use alpha=1.0; the inverse-k setting does not control XSA.
        self.alpha = "inverse_k" if alpha == "inverse_k" else float(alpha)
        # Store the tiny positive denominator floor used by the XSA projection.
        self.xsa_eps = float(xsa_eps)
        # Make removed positions into sorted, unique integers, e.g. [4, 7].
        self.removed = sorted(set(int(i) for i in removed))
        # Check that the requested method is one this code knows how to execute.
        if mode not in {"original", "removal", "loop", "damped", "xsa_reuse"}:
            # Reject an unknown method name.
            raise ValueError(f"Unknown mode: {mode}")
        # For a numeric damping setting, require a value from zero to one.
        if self.alpha != "inverse_k" and not 0 <= self.alpha <= 1:
            # Explain an invalid damping setting.
            raise ValueError("Damping alpha must be inverse_k or a number in [0, 1]")
        # Validate the XSA safety floor stored in the container.
        if not math.isfinite(self.xsa_eps) or self.xsa_eps <= 0:
            # Reject an invalid safety floor.
            raise ValueError("XSA epsilon must be finite and positive")
        # Removed positions must be valid zero-based positions in the original model.
        if any(i < 0 or i >= self.depth for i in self.removed):
            # Reject a position that does not exist.
            raise ValueError("Removed index outside block range")
        # The original baseline must keep all its blocks.
        if mode == "original" and self.removed:
            # Reject a baseline that would secretly remove blocks.
            raise ValueError("Original model cannot have removed blocks")
        # Position zero has no preceding block available as a replacement.
        if 0 in self.removed:
            # Reject removal of the first block under this replacement rule.
            raise ValueError("Keep block 0: a replacement needs a preceding retained block")
        # Register each kept block ONCE, under its original position, e.g. "3".
        # ModuleDict is a dictionary that PyTorch recognizes as containing model parts.
        # It stores the original block objects, without making extra copies of weights.
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        # Will hold the actual order of calls: source block, scale, replacement flag.
        self.schedule = []
        # Will record which original depth position each call fills.
        self.slots = []
        # Will record call numbers for reporting and the separate damped method.
        self.iterations = []
        # At the beginning we have not encountered a retained source block yet.
        previous = None
        # Start the per-source call counter; it is not a head count.
        iteration = 0
        # Walk through original positions 0 through 27.
        for slot in range(self.depth):
            # This position still has its original block.
            if slot not in self.removed:
                # Remember this block as the source to reuse if a following slot is removed.
                previous = slot
                # This is the first call to that source block.
                iteration = 1
                # Schedule a normal call: its own block, full residual scale, not a replacement.
                self.schedule.append((slot, 1.0, False))
                # Record the original depth position being filled.
                self.slots.append(slot)
                # Record whether this is the first, second, etc. call to the source block.
                self.iterations.append(iteration)
            # The position was removed. Reuse methods fill it with the preceding block.
            # Plain removal mode adds no call at this position.
            elif mode != "removal":
                # k includes the original call; each retained block starts at k=1.
                # Precompute per-slot values so k resets on every model forward.
                # Count this extra use of the same preceding source block.
                iteration += 1
                # Only the damped method calculates a residual damping scale here.
                # For XSA, mode is xsa_reuse, so this line chooses 1.0 every time.
                scale = (1.0 / iteration if self.alpha == "inverse_k" else self.alpha) if mode == "damped" else 1.0
                # Add the previous block again and mark this call as a replacement.
                self.schedule.append((previous, scale, True))
                # Record the original depth position being filled.
                self.slots.append(slot)
                # Record whether this is the first, second, etc. call to the source block.
                self.iterations.append(iteration)

    # Define what len(this_container) means.
    def __len__(self):
        # Return the number of scheduled calls, not the number of unique blocks.
        return len(self.schedule)

    # This helper does not need a container instance passed in as self.
    @staticmethod
    # Separate comparison method: reduce the whole block's residual update.
    def _damped(block, scale):
        # Prepare a callable that receives current hidden features x and conditioning c.
        def call(x, c):
            # For the separate damped method, zero scale means skip its update entirely.
            if scale == 0:
                # Return the current features unchanged when that scale is zero.
                return x
            # Take the original block update, shrink it by scale, and add it to x.
            return x + scale * (block(x, c) - x)
        # Hand back the callable; the model will execute it later when it reaches this slot.
        return call

    # This helper does not need a container instance passed in as self.
    @staticmethod
    # Prepare the XSA version of one reused block call.
    # This wrapper creates no new learned weights.
    def _xsa(block, eps):
        # Prepare a callable that receives current hidden features x and conditioning c.
        def call(x, c):
            # Keep DiTBlock.forward itself: adaLN, both gates, residuals and MLP.
            # Turn on the temporary attention callbacks only while this block call runs.
            with xsa_attention_scope(block.attn, eps=eps):
                # Run the original WHOLE block with the current features and conditioning.
                # Its attention output is filtered, but its normal residual path, gates and MLP remain.
                # Leaving the with statement removes the callbacks even though this line returns.
                return block(x, c)
        # Hand back the callable; the model will execute it later when it reaches this slot.
        return call

    # Define how the model gets one callable at a time from this container.
    def __iter__(self):
        # Read the next source block, residual scale and replacement flag.
        for index, scale, replacement in self.schedule:
            # Get the kept block object by its ORIGINAL index. No weights are copied.
            block = self.retained[str(index)]
            # Only an EXTRA/replacement call in XSA mode gets attention filtering.
            if replacement and self.mode == "xsa_reuse":
                # Supply the temporarily filtered block callable for the model to execute.
                yield self._xsa(block, self.xsa_eps)
            # Otherwise, an extra call with a smaller scale uses the damped-method wrapper.
            elif replacement and scale != 1:
                # Supply that separate residual-damped callable.
                yield self._damped(block, scale)
            # For a normal call, or an unscaled ordinary reuse, use the untouched block.
            else:
                # Supply the actual block object as the next callable.
                yield block

    # Describe the execution plan so it can be inspected and saved with the images.
    def metadata(self):
        # Record the selected method, removed positions, and residual damping setting.
        return dict(mode=self.mode, removed=self.removed, alpha=self.alpha,
                    # Report a damping policy only for damped mode; XSA records none.
                    damping_policy=("inverse_k" if self.alpha == "inverse_k" else "fixed") if self.mode == "damped" else "none",
                    # Explain the separate iteration field used by the existing damped-method code.
                    k_definition="call index of the retained source block, including its original call",
                    # XSA uses full block residual scale 1.0; its projection is a different operation.
                    replacement_residual_scale=(None if self.alpha == "inverse_k" else self.alpha) if self.mode == "damped" else 1.0,
                    # Record the XSA denominator floor and that positions are numbered from zero.
                    xsa_eps=self.xsa_eps, index_base=0,
                    # Record original depth and which original block weights are actually stored.
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    # Record which original position each call fills, which block it uses, and its scale.
                    execution_schedule=[dict(slot=slot, source=i, scale=a,
                        # Record replacement status, call number, and whether this call receives XSA.
                        replacement=r, iteration=k, xsa=r and self.mode == "xsa_reuse")
                        # Match the position, scheduled call and call number to create one record per call.
                        for slot, (i, a, r), k in zip(self.slots, self.schedule, self.iterations)])
```

The code is activated by this existing notebook call:

```python
run_method(
    "test",                 # Use the held-out class/seed list: 5,000 images.
    "xsa_reuse",            # Reuse weights, filtering attention on extra calls.
    "member4_xsa_reuse",    # Name of the folder that receives this method's images.
    alpha=1.0,              # No additional whole-block residual damping.
)
```

The runner loads the frozen removed positions from `profile/selection.json` and
passes the mode, positions and alpha into the ordinary model-loading/sampling code.
The XSA denominator floor defaults to `xsa_eps=1e-8` in `ScheduledBlocks` and is
recorded in the saved schedule.

For `[4, 7]`, this is the relevant part of the plan:

| Original position | Which weights run? | What happens to attention? |
|---:|---|---|
| 3 | Block 3, first use | Ordinary attention. |
| 4 | Block 3 again, replacing block 4 | XSA on every token and every head. |
| 6 | Block 6, first use | Ordinary attention. |
| 7 | Block 6 again, replacing block 7 | XSA on every token and every head. |

This plan runs again at each of the 50 denoising steps. Each repeated block receives
the features produced by the previous call and calculates fresh Q/K/V. Nothing is
copied from an earlier attention result. There are 26 unique blocks but 28 block
applications per denoising step.

The original DiT block still applies its normal conditioning, residual connections,
gates and MLP. Therefore, removing the own-value direction from the attention output
does not remove all of a token's own information from the model. The residual route
still carries information forward. Also, the projection can remove components of
OTHER tokens' contributions that happen to point along that same own-value direction.

**A short explanation you can say aloud:**

> We remove two blocks and fill their positions by calling the preceding retained
> blocks again, so weights are shared. On each extra call, attention runs normally.
> Then, for every image-patch token in every head, we subtract the part of its
> attention output that points along its own value vector. The heads remain
> separate during filtering. We keep the rest of the DiT block intact and introduce
> no new learned weights.

**What still needs work:**

- Complete the same generated-image count for XSA and the other methods before
  comparing FID. A correct projection does not by itself prove better image quality.
- Compare ordinary looping with XSA under the same positions, seeds and sampling
  settings to isolate the effect of this filtering.
- Test more repetitions and the choice to filter only extra calls versus all calls
  in a reused group. Neither choice is established as best by this code.
- Broaden backend/precision tests. Existing CPU checks exercise the math, fused and
  unfused attention, cleanup, weight sharing and checkpoint restoration. That is
  separate from validating full-model performance across MPS/CUDA and precisions.
- The temporary-hook adapter assumes one sequential attention call. Concurrency,
  reentrant calls and `torch.compile` require separate engineering/testing.
- This project inserts XSA into a frozen class-conditioned latent DiT. Looped-DiT
  trains a different looped model with additional supervision. Describe this as an
  adaptation of its attention operation rather than a complete paper reproduction.

The projection is described in [Looped-DiT, Appendix 6.3](https://arxiv.org/html/2609.40305v1),
with a direct implementation in the authors' [model.py](https://github.com/OpenSenseNova/Looped-DiT/blob/main/looped_dit/model.py).
