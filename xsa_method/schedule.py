"""Weight-sharing schedules for XSA and its comparison baselines."""
import math

from torch import nn

from .attention import xsa_attention_scope


class ScheduledBlocks(nn.Module):
    """Retain original indices and register each retained block exactly once."""
    def __init__(self, blocks, removed=(), mode="original", alpha="inverse_k", xsa_eps=1e-8):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        self.mode = mode
        self.alpha = "inverse_k" if alpha == "inverse_k" else float(alpha)
        self.xsa_eps = float(xsa_eps)
        self.removed = sorted(set(int(i) for i in removed))
        if mode not in {"original", "removal", "loop", "damped", "xsa_reuse"}:
            raise ValueError(f"Unknown mode: {mode}")
        if self.alpha != "inverse_k" and not 0 <= self.alpha <= 1:
            raise ValueError("Damping alpha must be inverse_k or a number in [0, 1]")
        if not math.isfinite(self.xsa_eps) or self.xsa_eps <= 0:
            raise ValueError("XSA epsilon must be finite and positive")
        if any(i < 0 or i >= self.depth for i in self.removed):
            raise ValueError("Removed index outside block range")
        if mode == "original" and self.removed:
            raise ValueError("Original model cannot have removed blocks")
        if 0 in self.removed and mode != "removal":
            raise ValueError("Keep block 0: a replacement needs a preceding retained block")
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        self.schedule = []
        self.slots = []
        self.iterations = []
        previous = None
        iteration = 0
        for slot in range(self.depth):
            if slot not in self.removed:
                previous = slot
                iteration = 1
                self.schedule.append((slot, 1.0, False))
                self.slots.append(slot)
                self.iterations.append(iteration)
            elif mode != "removal":
                # k includes the original call; each retained block starts at k=1.
                # Precompute per-slot values so k resets on every model forward.
                iteration += 1
                scale = (1.0 / iteration if self.alpha == "inverse_k" else self.alpha) if mode == "damped" else 1.0
                self.schedule.append((previous, scale, True))
                self.slots.append(slot)
                self.iterations.append(iteration)

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
            with xsa_attention_scope(block.attn, eps=eps):
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
                    damping_policy=("inverse_k" if self.alpha == "inverse_k" else "fixed") if self.mode == "damped" else "none",
                    k_definition="call index of the retained source block, including its original call",
                    replacement_residual_scale=(None if self.alpha == "inverse_k" else self.alpha) if self.mode == "damped" else 1.0,
                    xsa_eps=self.xsa_eps, index_base=0,
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    execution_schedule=[dict(slot=slot, source=i, scale=a,
                        replacement=r, iteration=k, xsa=r and self.mode == "xsa_reuse")
                        for slot, (i, a, r), k in zip(self.slots, self.schedule, self.iterations)])


