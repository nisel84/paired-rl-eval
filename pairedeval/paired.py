"""Paired measurement with common random numbers.

The whole point of this module is one sentence: compare two policies on the
*same* task realisations, from the *same* state, with the *same* random draws,
and report the smallest difference the measurement could have resolved.

An unpaired comparison of two policies on independently sampled worlds is
dominated by task variance.  Pairing removes it.  The visibility threshold then
says whether a difference you can see is a difference you can trust.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import Config, SwarmEnv
from .policies import Policy

__all__ = ["Result", "rollout", "paired_compare", "unpaired_compare"]


@dataclass
class Result:
    """One comparison, with everything needed to judge it."""

    name_a: str
    name_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    diff: float            # b - a, per-world mean of the paired difference
    stderr: float
    t: float
    visibility: float      # 2 standard errors: the smallest resolvable effect
    positives: int         # worlds where b beat a

    @property
    def visible(self) -> bool:
        return abs(self.diff) > self.visibility

    @property
    def verdict(self) -> str:
        if not self.visible:
            return "indistinguishable from zero"
        return "B better" if self.diff > 0 else "B worse"

    def __str__(self) -> str:
        return (
            f"{self.metric}: {self.name_a} {self.mean_a:+.4f} -> "
            f"{self.name_b} {self.mean_b:+.4f} | diff {self.diff:+.4f} "
            f"+-{self.stderr:.4f} | t {self.t:+.2f} | {self.positives}/{self.n} "
            f"positive | visibility {self.visibility:.4f} | {self.verdict}"
        )


def rollout(
    env: SwarmEnv,
    policy: Policy,
    steps: int,
    metric: str = "coverage",
    snap: dict | None = None,
) -> np.ndarray:
    """Run ``policy`` for ``steps`` and return the per-world mean of ``metric``.

    If ``snap`` is given the environment is restored to it first, which is what
    makes two rollouts share both the starting state and the random stream.
    """
    if snap is not None:
        env.restore(snap)
    total = np.zeros(env.batch)
    for _ in range(steps):
        target, home = policy.act(env)
        out = env.step(target, home)
        total += out[metric]
    return total / steps


def _finish(name_a, name_b, metric, a, b) -> Result:
    d = b - a
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return Result(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        n=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=float(d.mean()),
        stderr=se,
        t=float(d.mean() / se) if se > 0 else 0.0,
        visibility=2.0 * se,
        positives=int((d > 0).sum()),
    )


def paired_compare(
    policy_a: Policy,
    policy_b: Policy,
    steps: int = 900,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
) -> Result:
    """Compare on identical worlds, from an identical state, with identical draws."""
    env = SwarmEnv(cfg or Config(), batch=batch, seed=seed)
    snap = env.snapshot()
    a = rollout(env, policy_a, steps, metric, snap)
    b = rollout(env, policy_b, steps, metric, snap)
    return _finish(policy_a.name, policy_b.name, metric, a, b)


def unpaired_compare(
    policy_a: Policy,
    policy_b: Policy,
    steps: int = 900,
    metric: str = "coverage",
    cfg: Config | None = None,
    batch: int = 32,
    seed: int = 0,
) -> Result:
    """The same comparison done carelessly: different worlds for each policy.

    Kept here on purpose.  Running both and looking at the two visibility
    thresholds is the clearest argument for pairing that exists.
    """
    env_a = SwarmEnv(cfg or Config(), batch=batch, seed=seed)
    env_b = SwarmEnv(cfg or Config(), batch=batch, seed=seed + 10_000)
    a = rollout(env_a, policy_a, steps, metric)
    b = rollout(env_b, policy_b, steps, metric)
    return _finish(policy_a.name, policy_b.name, metric, a, b)
