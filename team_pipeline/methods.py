"""Method-specific block schedules. Team members can add model transformations here."""
from torch import nn


class ScheduledBlocks(nn.Module):
    """Register each retained block ONCE; execute it at one or more depth slots.

    Official DiT.forward iterates over model.blocks. This container preserves that
    interface, while its state_dict contains only retained modules.
    """
    def __init__(self, blocks, removed=(), mode="original", alpha=0.5):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        self.mode = mode
        self.alpha = float(alpha)
        self.removed = sorted(set(int(i) for i in removed))
        if mode not in {"original", "removal", "loop", "damped"}:
            raise ValueError(f"Unknown mode: {mode}")
        if not 0 <= self.alpha <= 1:
            raise ValueError("Damping alpha must be in [0, 1].")
        if any(i < 0 or i >= self.depth for i in self.removed):
            raise ValueError("Removed index outside block range.")
        if mode == "original" and self.removed:
            raise ValueError("Original model cannot have removed blocks.")
        if 0 in self.removed:
            raise ValueError("Keep block 0: a replacement needs a preceding retained block.")
        self.retained = nn.ModuleDict({str(i): b for i, b in enumerate(blocks)
                                       if i not in self.removed})
        self.schedule = []
        previous = None
        for i in range(self.depth):
            if i not in self.removed:
                previous = i
                self.schedule.append((i, 1.0, False))
            elif mode != "removal":
                self.schedule.append((previous, self.alpha if mode == "damped" else 1.0, True))

    def __len__(self):
        return len(self.schedule)

    @staticmethod
    def _damped(block, scale):
        def call(x, c):
            if scale == 0:
                return x
            return x + scale * (block(x, c) - x)
        return call

    def __iter__(self):
        for index, scale, replacement in self.schedule:
            block = self.retained[str(index)]
            yield self._damped(block, scale) if replacement and scale != 1 else block

    def metadata(self):
        return dict(mode=self.mode, removed=self.removed, alpha=self.alpha,
                    depth=self.depth, retained=list(map(int, self.retained.keys())),
                    execution_schedule=[dict(source=i, scale=a, replacement=r)
                                        for i, a, r in self.schedule])
