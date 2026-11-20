"""Determinism is the whole claim.  These tests are the claim's teeth.

If any of these fail, every number this package prints is unreliable, because
paired measurement is only paired as long as the second rollout really does see
the same state and the same random stream as the first.
"""
from __future__ import annotations

import numpy as np
import pytest

from pairedeval import (
    Config,
    Greedy,
    Sustaining,
    SwarmEnv,
    paired_compare,
    rollout,
    unpaired_compare,
)


def _advance(env: SwarmEnv, policy, steps: int) -> None:
    for _ in range(steps):
        env.step(*policy.act(env))


def test_same_seed_same_world():
    a = SwarmEnv(Config(), batch=8, seed=11)
    b = SwarmEnv(Config(), batch=8, seed=11)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.soc, b.soc)


def test_different_seed_different_world():
    a = SwarmEnv(Config(), batch=8, seed=11)
    b = SwarmEnv(Config(), batch=8, seed=12)
    assert not np.array_equal(a.pos, b.pos)


def test_restore_is_bit_exact():
    env = SwarmEnv(Config(), batch=8, seed=7)
    policy = Greedy()
    _advance(env, policy, 40)
    snap = env.snapshot()

    first = rollout(env, policy, 60, "coverage", snap)
    second = rollout(env, policy, 60, "coverage", snap)
    assert np.array_equal(first, second), "restore did not reproduce the rollout"


def test_restore_rewinds_every_field():
    env = SwarmEnv(Config(), batch=4, seed=3)
    policy = Sustaining()
    _advance(env, policy, 25)
    snap = env.snapshot()
    before = (env.pos.copy(), env.soc.copy(), env.docked.copy(),
              env.staleness.copy(), env.t)

    _advance(env, policy, 25)
    env.restore(snap)

    for got, want, field in zip(
        (env.pos, env.soc, env.docked, env.staleness),
        before[:4],
        ("pos", "soc", "docked", "staleness"),
    ):
        assert np.array_equal(got, want), f"{field} was not restored"
    assert env.t == before[4]


def test_snapshot_is_a_copy_not_a_view():
    env = SwarmEnv(Config(), batch=4, seed=5)
    snap = env.snapshot()
    env.pos += 1.0
    env.staleness += 1.0
    assert not np.array_equal(snap["pos"], env.pos)
    assert not np.array_equal(snap["staleness"], env.staleness)


def test_paired_compare_is_repeatable():
    kw = dict(steps=200, batch=12, seed=4)
    first = paired_compare(Greedy(), Sustaining(), **kw)
    second = paired_compare(Greedy(), Sustaining(), **kw)
    assert first.diff == second.diff
    assert first.stderr == second.stderr
    assert first.positives == second.positives


def test_policy_against_itself_is_exactly_zero():
    """The sharpest determinism test there is: no difference, no error bar."""
    result = paired_compare(Greedy(), Greedy(), steps=150, batch=12, seed=2)
    assert result.diff == 0.0
    assert result.stderr == 0.0
    assert not result.visible
    assert result.verdict == "indistinguishable from zero"


def test_unpaired_against_itself_is_not_zero():
    """Which is the point: the same policy looks different on different worlds."""
    result = unpaired_compare(Greedy(), Greedy(), steps=150, batch=12, seed=2)
    assert result.diff != 0.0


def test_pairing_pays_off_when_the_arms_are_close():
    """Pairing is not a free win, and this test says where the win lives.

    An earlier version of this file asserted that the paired standard error is
    never larger than the unpaired one.  That is false, and the suite caught
    it: for two policies as different as Greedy and Sustaining, pairing is a
    coin flip (narrower in 12 of 30 seeds, median ratio 0.98).  Common random
    numbers help in proportion to how correlated the two arms are, so the real
    guarantee is about near-identical arms.
    """
    close_a, close_b = Greedy(return_at=0.15), Greedy(return_at=0.155)
    paired_se, unpaired_se = [], []
    for seed in range(8):
        kw = dict(steps=300, batch=24, seed=seed)
        paired_se.append(paired_compare(close_a, close_b, **kw).stderr)
        unpaired_se.append(unpaired_compare(close_a, close_b, **kw).stderr)

    gain = np.mean(unpaired_se) / np.mean(paired_se)
    assert gain > 1.5, f"pairing bought only {gain:.2f}x on near-identical arms"


def test_pairing_gain_shrinks_as_the_arms_diverge():
    """The shape of the previous test's claim, stated as an ordering."""
    def gain(delta: float) -> float:
        a, b = Greedy(return_at=0.15), Greedy(return_at=0.15 + delta)
        kw = lambda s: dict(steps=300, batch=24, seed=s)
        p = np.mean([paired_compare(a, b, **kw(s)).stderr for s in range(6)])
        u = np.mean([unpaired_compare(a, b, **kw(s)).stderr for s in range(6)])
        return u / p

    assert gain(0.005) > gain(0.10), (
        "pairing should buy more on close arms than on distant ones"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_soc_stays_in_range(seed):
    env = SwarmEnv(Config(), batch=8, seed=seed)
    policy = Greedy()
    for _ in range(300):
        env.step(*policy.act(env))
        assert env.soc.min() >= 0.0
        assert env.soc.max() <= 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agents_stay_on_site(seed):
    env = SwarmEnv(Config(), batch=8, seed=seed)
    policy = Greedy()
    for _ in range(300):
        env.step(*policy.act(env))
    assert env.pos.min() >= 0.0
    assert env.pos.max() <= env.cfg.side
