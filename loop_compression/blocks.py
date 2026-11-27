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

Timestep-aware looping (``schedule_by_t``): a list of rules
``{t_min, t_max, K, strategy, [h], [beta]}`` selecting HOW the kept unit is applied
for the sampler timesteps they cover (original, un-respaced values 0..999). The
retained blocks, and so the parameters, never change; uncovered timesteps use the
top-level ``K/strategy/h/beta``. ``damped`` with ``h = 0`` skips the unit at those
timesteps. The container reads the timestep from the model through ``attach``,
which hooks ``model.t_embedder`` (``DiT.forward_with_cfg`` calls ``forward``
directly, so a hook on the model itself would never fire). Samples of one batch
may have different timesteps; they are grouped by rule. ``schedule`` (used for
FLOP counting) is the worst-case rule, an upper bound; ``mean_block_evaluations``
gives the exact per-step average.

Token gating (``gate``, naive and damped only): the first application of the unit
updates every token; later iterations update only the ``keep_fraction`` of tokens
selected from that first update, the rest pass through unchanged.
    signal  update  per-token size of the first update ||g(x) - x||
            grad    spatial gradient of that size map on the token grid (an
                    "object boundary" proxy: where the update changes abruptly)
            random  seeded random tokens (control: is the signal informative?)
    select  high (default) loops the largest-signal tokens, low the smallest
This is a MASKED loop: g still runs on all tokens (they are the keys/values the
selected tokens attend to), so it saves no compute; ``metadata()`` reports what a
token-pruning implementation would cost as an estimate only. keep_fraction = 1 is
the ungated loop; 0 stops after the first application.
"""
from collections import namedtuple
from contextlib import contextmanager
import math

import torch
from torch import nn

STRATEGIES = {"naive", "damped", "anchored"}
ITERATIONS = {"block", "layer"}
Rule = namedtuple("Rule", "strategy K h beta gate", defaults=(None,))
Gate = namedtuple("Gate", "signal keep_fraction select")
RULE_KEYS = {"t_min", "t_max", "K", "strategy", "h", "beta", "gate"}
GATE_SIGNALS = {"update", "grad", "random"}
GATE_KEYS = {"signal", "keep_fraction", "select"}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def make_gate(gate):
    """Validated ``Gate`` from a dict (or None)."""
    if gate is None:
        return None
    if isinstance(gate, Gate):
        gate = gate._asdict()
    if not isinstance(gate, dict) or set(gate) - GATE_KEYS or "signal" not in gate or "keep_fraction" not in gate:
        raise ValueError(f"gate must be an object with signal and keep_fraction (optional select): {GATE_KEYS}")
    signal, fraction = gate["signal"], gate["keep_fraction"]
    select = gate.get("select", "high")
    if signal not in GATE_SIGNALS:
        raise ValueError(f"gate signal must be one of {sorted(GATE_SIGNALS)}")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not (
            math.isfinite(fraction) and 0.0 <= fraction <= 1.0):
        raise ValueError("gate keep_fraction must be a number in [0, 1]")
    if select not in {"high", "low"}:
        raise ValueError("gate select must be 'high' or 'low'")
    return Gate(signal, float(fraction), select)


def gate_mask(update, gate):
    """Boolean token mask [B, T, 1] of the tokens a gate keeps looping.

    ``update`` is the first-iteration change x1 - x0, [B, T, D].
    """
    batch, tokens, _ = update.shape
    count = int(math.ceil(gate.keep_fraction * tokens - 1e-9))
    if count >= tokens:
        return torch.ones(batch, tokens, 1, dtype=torch.bool, device=update.device)
    if count <= 0:
        return torch.zeros(batch, tokens, 1, dtype=torch.bool, device=update.device)
    size = update.float().norm(dim=-1)                                  # [B, T]
    if gate.signal == "grad":
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("gate signal 'grad' needs a square token grid")
        grid = size.reshape(batch, side, side)
        gx = torch.nn.functional.pad((grid[:, :, 1:] - grid[:, :, :-1]).abs(), (0, 1))
        gy = torch.nn.functional.pad((grid[:, 1:, :] - grid[:, :-1, :]).abs(), (0, 0, 0, 1))
        size = (gx + gy).reshape(batch, tokens)
    elif gate.signal == "random":
        generator = torch.Generator().manual_seed(1234)
        size = torch.rand(batch, tokens, generator=generator).to(update.device)
    order = size.argsort(dim=1, descending=gate.select == "high")
    chosen = order[:, :count]
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=update.device)
    mask.scatter_(1, chosen, True)
    return mask.unsqueeze(-1)


def make_rule(strategy, K, h=None, beta=0.0, gate=None):
    """Validated ``Rule``; ``h`` defaults to 1 (naive) or 1/K."""
    if not _is_int(K) or K < 1:
        raise ValueError("K must be an integer >= 1")
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
    gate = make_gate(gate)
    if gate is not None and strategy == "anchored":
        raise ValueError("gate supports naive and damped looping only")
    return Rule(strategy, K, h, beta, gate)


def rule_evaluations(rule):
    """(number of evaluations of g per unit, scale of each) for a rule."""
    if rule.strategy == "naive":
        return rule.K, 1.0
    if rule.h == 0.0:
        return (1, 1.0) if rule.strategy == "anchored" and rule.beta > 0 else (0, 0.0)
    return rule.K, rule.h


class LoopedBlocks(nn.Module):
    """Drop-in replacement for DiT's ``model.blocks`` (iterated as f(x, c)).

    Iterating yields pre-window blocks, ONE closure that runs the whole window,
    then post-window blocks. ``schedule`` lists every real block call as
    (source_index, scale, replacement) so FLOP counting stays honest, and
    ``len(self)`` is the number of real block calls, not the number of yielded
    items.
    """

    def __init__(self, blocks, window, kept=None, K=2, iteration="block",
                 strategy="damped", h=None, beta=0.0, schedule_by_t=None, gate=None):
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
        if iteration not in ITERATIONS:
            raise ValueError(f"iteration must be one of {sorted(ITERATIONS)}")
        self.default_rule = make_rule(strategy, K, h, beta, gate)
        self.K, self.iteration = K, iteration
        self.strategy, self.h, self.beta = (self.default_rule.strategy, self.default_rule.h,
                                            self.default_rule.beta)
        self.mode = "loop_compression"
        self.timed = self._parse_schedule(schedule_by_t)
        self._t = None

        self.retained = nn.ModuleDict({
            str(i): b for i, b in enumerate(blocks)
            if not (first <= i <= last) or i in self.kept})
        self._units = ([list(self.kept)] if iteration == "block"
                       else [[i] for i in self.kept])
        # Worst case over every rule that can apply: an upper bound for FLOP counting.
        worst = max([self.default_rule] + [rule for _, _, rule in self.timed],
                    key=lambda r: rule_evaluations(r)[0])
        self.schedule = []
        for i in range(first):
            self.schedule.append((i, 1.0, False))
        for unit in self._units:
            evaluations, scale = rule_evaluations(worst)
            for _ in range(evaluations):
                self.schedule.extend((i, scale, True) for i in unit)
        for i in range(last + 1, self.depth):
            self.schedule.append((i, 1.0, False))

    # ------------------------------------------------------------ timestep schedule

    def _parse_schedule(self, schedule_by_t):
        if schedule_by_t is None:
            return []
        if not isinstance(schedule_by_t, (list, tuple)) or not schedule_by_t:
            raise ValueError("schedule_by_t must be a nonempty list of rules")
        parsed = []
        for entry in schedule_by_t:
            if not isinstance(entry, dict):
                raise ValueError("schedule_by_t entries must be objects")
            unknown = set(entry) - RULE_KEYS
            if unknown:
                raise ValueError(f"Unknown schedule_by_t keys: {sorted(unknown)}")
            missing = {"t_min", "t_max", "K", "strategy"} - set(entry)
            if missing:
                raise ValueError(f"schedule_by_t entry is missing {sorted(missing)}")
            lo, hi = entry["t_min"], entry["t_max"]
            if not (_is_int(lo) and _is_int(hi) and 0 <= lo <= hi):
                raise ValueError("t_min and t_max must be integers with 0 <= t_min <= t_max")
            rule = make_rule(entry["strategy"], entry["K"], entry.get("h"), entry.get("beta", 0.0),
                             entry.get("gate"))
            parsed.append((lo, hi, rule))
        parsed.sort(key=lambda item: item[0])
        for (_, hi, _), (lo, _, _) in zip(parsed, parsed[1:]):
            if lo <= hi:
                raise ValueError("schedule_by_t ranges must not overlap")
        return parsed

    def rule_for(self, t):
        """The rule used at integer timestep ``t``."""
        for lo, hi, rule in self.timed:
            if lo <= t <= hi:
                return rule
        return self.default_rule

    def set_timestep(self, t):
        self._t = t

    def attach(self, model):
        """Forward-pre-hook on ``model.t_embedder`` feeding this container the batch timesteps."""
        return model.t_embedder.register_forward_pre_hook(
            lambda module, args: self.set_timestep(args[0]))

    @contextmanager
    def attached(self, model):
        handle = self.attach(model)
        try:
            yield self
        finally:
            handle.remove()
            self._t = None

    def mean_block_evaluations(self, timesteps):
        """Average real block calls per forward over the sampler ``timesteps``."""
        outside = self.depth - (self.window[1] - self.window[0] + 1)
        total = 0.0
        for t in timesteps:
            total += outside + len(self.kept) * rule_evaluations(self.rule_for(int(t)))[0]
        return total / len(timesteps)

    def __len__(self):
        return len(self.schedule)

    # ------------------------------------------------------------------- execution

    def _apply_unit(self, x, c, unit, rule=None):
        rule = rule or self.default_rule
        blocks = [self.retained[str(i)] for i in unit]

        def g(z):
            for block in blocks:
                z = block(z, c)
            return z

        strategy, K, h, beta, gate = rule
        if gate is not None and K > 1 and h > 0.0:
            def step(z):
                return g(z) if strategy == "naive" else z + h * (g(z) - z)

            y = step(x)                                  # first application: every token
            mask = gate_mask(y - x, gate)
            for _ in range(K - 1):
                y = torch.where(mask, step(y), y)        # later iterations: selected tokens only
            return y
        if strategy == "naive":
            for _ in range(K):
                x = g(x)
            return x
        if h == 0.0:
            if strategy == "anchored" and beta > 0:
                return beta * g(x) + (1 - beta) * x
            return x
        if strategy == "damped":
            for _ in range(K):
                x = x + h * (g(x) - x)
            return x
        anchor = g(x)
        y = x + h * (anchor - x)
        for _ in range(K - 1):
            y = y + h * (g(y) - y)
        return beta * anchor + (1 - beta) * y

    def _run_rule(self, x, c, rule):
        for unit in self._units:
            x = self._apply_unit(x, c, unit, rule)
        return x

    def _run_window(self, x, c):
        if not self.timed:
            return self._run_rule(x, c, self.default_rule)
        t = self._t
        if t is None:
            raise RuntimeError("schedule_by_t needs the sampler timestep: call attach(model) first")
        t = t.to(x.device)
        if t.shape[0] != x.shape[0]:
            raise ValueError("timestep batch does not match the activations")
        rules = [rule for _, _, rule in self.timed] + [self.default_rule]
        assign = t.new_full((len(x),), len(self.timed))
        for j, (lo, hi, _) in enumerate(self.timed):
            assign[(t >= lo) & (t <= hi)] = j
        used = assign.unique().tolist()
        if len(used) == 1:
            return self._run_rule(x, c, rules[used[0]])
        out = x.new_empty(x.shape)
        for j in used:
            mask = assign == j
            out[mask] = self._run_rule(x[mask], c[mask], rules[j])
        return out

    def __iter__(self):
        first, last = self.window
        for i in range(first):
            yield self.retained[str(i)]
        yield self._run_window
        for i in range(last + 1, self.depth):
            yield self.retained[str(i)]

    def metadata(self):
        meta = dict(
            mode=self.mode, depth=self.depth, window=list(self.window),
            kept=list(self.kept), removed=list(self.discarded), K=self.K,
            iteration=self.iteration, strategy=self.strategy, h=self.h,
            beta=self.beta, retained=[int(i) for i in self.retained.keys()],
            execution_schedule=[dict(source=i, scale=s, replacement=r)
                                for i, s, r in self.schedule])
        gate = self.default_rule.gate
        if gate is not None:
            meta["gate"] = gate._asdict()
            meta["gate_note"] = ("masked loop: g runs on all tokens, so execution_schedule and measured "
                                 "FLOPs are NOT reduced; estimated_pruned_block_evaluations is what token "
                                 "pruning would cost (attention keys/values still need all tokens)")
            meta["estimated_pruned_block_evaluations"] = (
                self.depth - (self.window[1] - self.window[0] + 1)
                + len(self.kept) * (1 + gate.keep_fraction * (self.default_rule.K - 1)))
        if self.timed:
            meta["schedule_by_t"] = [dict(t_min=lo, t_max=hi, K=r.K, strategy=r.strategy, h=r.h, beta=r.beta,
                                          gate=r.gate._asdict() if r.gate else None)
                                     for lo, hi, r in self.timed]
            meta["schedule_note"] = ("execution_schedule is the worst-case rule (upper bound); "
                                     "use mean_block_evaluations(timesteps) for the per-step average")
        return meta
