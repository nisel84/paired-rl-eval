"""Bucketed episode profiling — where the measurement window shows its teeth.

A single number at the end of an episode hides the shape of the run.  A single
number at the *start* of one hides the collapse.  This module reports a metric
at several window lengths from the same rollout, so the window length becomes
visible instead of implicit.
"""
from __future__ import annotations

import numpy as np

from .env import Config, SwarmEnv
from .policies import Policy

__all__ = ["profile", "trace", "crossings", "flip_point"]


def trace(
    policy: Policy,
    steps: int = 900,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Per-step mean of ``metric`` across the batch.  Shape ``(steps,)``."""
    env = SwarmEnv(cfg or Config(), batch=batch, seed=seed)
    out = np.zeros(steps)
    for i in range(steps):
        target, home = policy.act(env)
        out[i] = env.step(target, home)[metric].mean()
    return out


def _cumulative(per_step: np.ndarray) -> np.ndarray:
    """Running mean: element ``w`` is the mean over steps ``0..w`` inclusive."""
    return np.cumsum(per_step) / np.arange(1, per_step.size + 1)


def profile(
    policy: Policy,
    steps: int = 900,
    buckets: int = 9,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (window_lengths, metric_mean_up_to_that_window).

    The value at window ``w`` is the mean of ``metric`` over steps ``0..w``,
    which is exactly what a study that measured for ``w`` steps would report.
    """
    per_step = trace(policy, steps, metric, cfg, batch, seed)
    run = _cumulative(per_step)
    edges = np.linspace(steps // buckets, steps, buckets).astype(int)
    return edges, run[edges - 1]


def crossings(
    policy_a: Policy,
    policy_b: Policy,
    steps: int = 900,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
) -> list[int]:
    """Every window length at which the ranking of two policies changes hands.

    Usually there is more than one, and that is the uncomfortable part.  A
    single crossing would mean there is an early regime and a late regime and
    you only have to measure past the boundary.  Several crossings mean the
    ranking is a property of the stopping time over most of the episode, so a
    result quoted without its window is not a weaker claim — it is not a claim.
    """
    a = _cumulative(trace(policy_a, steps, metric, cfg, batch, seed))
    b = _cumulative(trace(policy_b, steps, metric, cfg, batch, seed))
    sign = np.sign(b - a)
    return (np.nonzero(np.diff(sign) != 0)[0] + 1).tolist()


def flip_point(
    policy_a: Policy,
    policy_b: Policy,
    steps: int = 900,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
    min_window: int = 30,
) -> int | None:
    """The window length after which the ranking never changes again.

    Not the *first* reversal — the last one.  The first is the easy number to
    report and the wrong one to act on, because the ranking can change hands
    again later; on the bundled policies it changes hands four times before it
    settles.  What a study needs to know is the shortest window whose verdict
    survives to the end of the horizon, and that is the final crossing.

    Crossings before ``min_window`` are ignored: the running mean has barely
    any samples behind it there and its ordering is noise.

    Returns ``None`` if the ranking never changes — which is the result you
    hope for, and the one you cannot claim without looking.  Note that the
    answer is only ever "never changes *within* ``steps``": a longer horizon
    can always produce another crossing, so this function reports what was
    measured, not what is true forever.
    """
    found = [c for c in crossings(policy_a, policy_b, steps, metric, cfg,
                                  batch, seed) if c >= min_window]
    return found[-1] if found else None
