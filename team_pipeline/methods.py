"""Method-specific block schedules. Team members can add model transformations here."""
import math

from torch import nn

from . import xsa

MODES = ("original", "removal", "loop", "damped", "xsa_reuse")


class ScheduledBlocks(nn.Module):
    """Register each retained block ONCE; execute it at one or more depth slots.

    Official DiT.forward iterates over model.blocks. This container preserves that
    interface, while its state_dict contains only retained modules. Indices are
    zero based and refer to the original block positions.

    Modes: ``removal`` skips removed slots; ``loop`` reruns the nearest preceding
    retained block there; ``damped`` applies ``h + alpha * (B(h, c) - h)`` at
    those slots; ``xsa_reuse`` reruns the preceding block with exclusive
    self-attention (see ``team_pipeline.xsa``) at those slots.
    """
    def __init__(self, blocks, removed=(), mode="original", alpha=0.5, xsa_eps=1e-8):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        self.mode = mode
        self.alpha = float(alpha)
        self.xsa_eps = float(xsa_eps)
        self.removed = sorted(set(int(i) for i in removed))
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if not 0 <= self.alpha <= 1:
            raise ValueError("Damping alpha must be in [0, 1].")
        if not math.isfinite(self.xsa_eps) or self.xsa_eps <= 0:
            raise ValueError("XSA epsilon must be finite and positive")
        if any(i < 0 or i >= self.depth for i in self.removed):
            raise ValueError("Removed index outside block range.")
        if mode == "original" and self.removed:
            raise ValueError("Original model cannot have removed blocks.")
        if 0 in self.removed:
            raise ValueError("Keep block 0: a replacement needs a preceding retained block.")
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        self.schedule = []
        self.slots = []
        previous = None
        for slot in range(self.depth):
            if slot not in self.removed:
                previous = slot
                self.schedule.append((slot, 1.0, False))
                self.slots.append(slot)
            elif mode != "removal":
                scale = self.alpha if mode == "damped" else 1.0
                self.schedule.append((previous, scale, True))
                self.slots.append(slot)

    def __len__(self):
        return len(self.schedule)

    @staticmethod
    def _damped(block, scale):
        def call(x, c):
            if scale == 0:
                return x
            return x + scale * (block(x, c) - x)
        return call

    @staticmethod
    def _xsa(block, eps):
        def call(x, c):
            # Keep DiTBlock.forward itself: adaLN, both gates, residuals and MLP.
            with xsa.xsa_attention_scope(block.attn, eps=eps):
                return block(x, c)
        return call

    def __iter__(self):
        for index, scale, replacement in self.schedule:
            block = self.retained[str(index)]
            if replacement and self.mode == "xsa_reuse":
                yield self._xsa(block, self.xsa_eps)
            elif replacement and scale != 1:
                yield self._damped(block, scale)
            else:
                yield block

    def metadata(self):
        return dict(mode=self.mode, removed=self.removed, alpha=self.alpha,
                    replacement_residual_scale=self.alpha if self.mode == "damped" else 1.0,
                    xsa_eps=self.xsa_eps, index_base=0,
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    execution_schedule=[dict(slot=slot, source=i, scale=a,
                                             replacement=r, xsa=r and self.mode == "xsa_reuse")
                                        for slot, (i, a, r) in zip(self.slots, self.schedule)])
