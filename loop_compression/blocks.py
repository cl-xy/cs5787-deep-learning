"""Training-free looping of a contiguous middle window of transformer blocks.

Chen et al. (arXiv 2605.23872) view a residual block as one forward-Euler step
(h = 1) of dx/dt = g(x) - x. Re-applying g with h = 1 ("naive") leaves the
regime the later blocks expect; damped sub-steps stay closer to it.

For compression we discard window blocks that are not in ``kept`` and run the
kept blocks as a looped unit instead. Only unique kept blocks are registered, so
parameters equal those of plain removal of ``window - kept``. No weights change.

Strategies (g is the unit being looped, x0 its input):
    naive    x <- g(x), K times                         (h fixed to 1)
    damped   x <- x + h (g(x) - x), K times
    anchored out = beta g(x0) + (1 - beta) F^K(x0), F(x) = x + h (g(x) - x);
             the first F step reuses g(x0), so it costs K evaluations of g.
Iteration modes:
    block    the unit is the whole kept segment (kept blocks in order)
    layer    each kept block is its own unit, finished before the next
"""
import math

from torch import nn

STRATEGIES = {"naive", "damped", "anchored"}
ITERATIONS = {"block", "layer"}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


class LoopedBlocks(nn.Module):
    """Drop-in replacement for DiT's ``model.blocks`` (iterated as f(x, c)).

    Iterating yields pre-window blocks, ONE closure that runs the whole window,
    then post-window blocks. ``schedule`` lists every real block call as
    (source_index, scale, replacement) so FLOP counting stays honest, and
    ``len(self)`` is the number of real block calls, not the number of yielded
    items.
    """

    def __init__(self, blocks, window, kept=None, K=2, iteration="block",
                 strategy="damped", h=None, beta=0.0):
        super().__init__()
        blocks = list(blocks)
        self.depth = len(blocks)
        if len(window) != 2 or not all(_is_int(v) for v in window):
            raise ValueError("window must be two integers [first, last]")
        first, last = window
        if not 0 <= first <= last < self.depth:
            raise ValueError("window must satisfy 0 <= first <= last < depth")
        self.window = (first, last)
        span = range(first, last + 1)
        kept = list(span) if kept is None else kept
        if not all(_is_int(v) for v in kept):
            raise ValueError("kept must contain integer block indices")
        if len(set(kept)) != len(kept) or sorted(kept) != list(kept):
            raise ValueError("kept must be sorted and unique")
        if not kept or any(i not in span for i in kept):
            raise ValueError("kept must be a nonempty subset of the window")
        self.kept = tuple(kept)
        self.discarded = tuple(i for i in span if i not in self.kept)
        if not _is_int(K) or K < 1:
            raise ValueError("K must be an integer >= 1")
        if iteration not in ITERATIONS:
            raise ValueError(f"iteration must be one of {sorted(ITERATIONS)}")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")
        if h is None:
            h = 1.0 if strategy == "naive" else 1.0 / K
        h = float(h)
        if strategy == "naive" and h != 1.0:
            raise ValueError("naive looping has step size h = 1")
        if not (math.isfinite(h) and 0.0 <= h <= 1.0):
            raise ValueError("h must be in [0, 1]")
        beta = float(beta)
        if not (math.isfinite(beta) and 0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1]")
        if strategy != "anchored" and beta != 0.0:
            raise ValueError("beta applies only to the anchored strategy")
        self.K, self.iteration, self.strategy = K, iteration, strategy
        self.h, self.beta = h, beta
        self.mode = "loop_compression"

        self.retained = nn.ModuleDict({
            str(i): b for i, b in enumerate(blocks)
            if not (first <= i <= last) or i in self.kept})
        self._units = ([list(self.kept)] if iteration == "block"
                       else [[i] for i in self.kept])
        self.schedule = []
        for i in range(first):
            self.schedule.append((i, 1.0, False))
        for unit in self._units:
            evaluations, scale = self._evaluations()
            for _ in range(evaluations):
                self.schedule.extend((i, scale, True) for i in unit)
        for i in range(last + 1, self.depth):
            self.schedule.append((i, 1.0, False))

    def _evaluations(self):
        """(number of evaluations of g per unit, scale of each) for the strategy."""
        if self.strategy == "naive":
            return self.K, 1.0
        if self.h == 0.0:
            return (1, 1.0) if self.strategy == "anchored" and self.beta > 0 else (0, 0.0)
        return self.K, self.h

    def __len__(self):
        return len(self.schedule)

    def _apply_unit(self, x, c, unit):
        blocks = [self.retained[str(i)] for i in unit]

        def g(z):
            for block in blocks:
                z = block(z, c)
            return z

        h, K = self.h, self.K
        if self.strategy == "naive":
            for _ in range(K):
                x = g(x)
            return x
        if h == 0.0:
            if self.strategy == "anchored" and self.beta > 0:
                return self.beta * g(x) + (1 - self.beta) * x
            return x
        if self.strategy == "damped":
            for _ in range(K):
                x = x + h * (g(x) - x)
            return x
        anchor = g(x)
        y = x + h * (anchor - x)
        for _ in range(K - 1):
            y = y + h * (g(y) - y)
        return self.beta * anchor + (1 - self.beta) * y

    def _run_window(self, x, c):
        for unit in self._units:
            x = self._apply_unit(x, c, unit)
        return x

    def __iter__(self):
        first, last = self.window
        for i in range(first):
            yield self.retained[str(i)]
        yield self._run_window
        for i in range(last + 1, self.depth):
            yield self.retained[str(i)]

    def metadata(self):
        return dict(
            mode=self.mode, depth=self.depth, window=list(self.window),
            kept=list(self.kept), removed=list(self.discarded), K=self.K,
            iteration=self.iteration, strategy=self.strategy, h=self.h,
            beta=self.beta, retained=[int(i) for i in self.retained.keys()],
            execution_schedule=[dict(source=i, scale=s, replacement=r)
                                for i, s, r in self.schedule])
