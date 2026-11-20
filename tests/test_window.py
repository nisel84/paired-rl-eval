"""The window-length effect this package exists to expose.

These tests do not assert that one policy is better.  They assert that the
answer depends on how long you measured, and that the tooling reports that
dependence instead of hiding it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pairedeval import (
    Circular,
    Config,
    Greedy,
    crossings,
    Sustaining,
    flip_point,
    paired_compare,
    profile,
    trace,
)
from pairedeval.env import SwarmEnv


def test_ranking_reverses_between_the_two_windows():
    short = paired_compare(Greedy(), Sustaining(), steps=100, batch=32, seed=0)
    long = paired_compare(Greedy(), Sustaining(), steps=900, batch=32, seed=0)

    assert short.visible and long.visible, "both windows must resolve a difference"
    assert np.sign(short.diff) != np.sign(long.diff), (
        "the demonstration is gone: these two windows now agree"
    )
    assert short.verdict == "B worse"
    assert long.verdict == "B better"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flip_point_is_found_and_stable(seed):
    flip = flip_point(Greedy(), Sustaining(), steps=900, batch=24, seed=seed)
    assert flip is not None, "no reversal found"
    assert 60 < flip < 400, f"reversal at step {flip} is outside the expected band"


def test_flip_point_ignores_the_warm_up():
    """A sign change in the first few steps is noise, not a reversal."""
    flip = flip_point(Greedy(), Sustaining(), steps=900, batch=24, seed=0)
    assert flip >= 30


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flip_point_is_the_last_crossing(seed):
    """The distinction the whole function turns on.

    With a badly staggered rotation the ranking changed hands four times and
    reporting the first crossing gave 76 where the honest answer was 165.  The
    default no longer does that, so this test asserts the rule rather than the
    symptom: whatever the crossings are, the reported step is the last one.
    """
    kw = dict(steps=900, batch=24, seed=seed)
    all_c = [c for c in crossings(Greedy(), Sustaining(), **kw) if c >= 30]
    assert all_c, "no reversal found"
    assert flip_point(Greedy(), Sustaining(), **kw) == all_c[-1]


def test_verdict_is_stable_after_the_flip_point():
    """Past the reported step the ordering must not change again."""
    flip = flip_point(Greedy(), Sustaining(), steps=900, batch=24, seed=0)
    later = [c for c in crossings(Greedy(), Sustaining(), steps=900,
                                  batch=24, seed=0) if c > flip]
    assert later == []


def test_no_flip_against_itself():
    assert flip_point(Greedy(), Greedy(), steps=400, batch=8, seed=0) is None


def test_profile_is_a_running_mean():
    edges, means = profile(Greedy(), steps=200, buckets=4, batch=8, seed=0)
    per_step = trace(Greedy(), steps=200, batch=8, seed=0)
    for e, m in zip(edges, means):
        assert m == pytest.approx(per_step[:e].mean())


def test_circular_wins_short_and_loses_long():
    """A third policy, the same trap: best at 100 steps, worst at 900."""
    scores = {}
    for p in (Greedy(), Circular(), Sustaining()):
        _, m = profile(p, steps=900, buckets=9, batch=24, seed=0)
        scores[p.name] = (m[0], m[-1])

    best_short = max(scores, key=lambda k: scores[k][0])
    worst_long = min(scores, key=lambda k: scores[k][1])
    assert best_short == "circular"
    assert worst_long == "circular"


def test_travel_cost_is_not_what_drives_the_effect():
    """The check that overturned this package's original explanation.

    The environment was built on the belief that cruise power punishes a greedy
    policy over a long horizon.  Make travel free and the reversal survives, by
    a wider margin than before, so that belief was wrong.  The test is kept
    pointing the other way so the claim cannot quietly return to the docs.
    """
    free = Config(cruise_w=0.0)
    kw = dict(cfg=free, batch=24, seed=0)
    short = paired_compare(Greedy(), Sustaining(), steps=100, **kw)
    long = paired_compare(Greedy(), Sustaining(), steps=900, **kw)

    assert np.sign(short.diff) != np.sign(long.diff), (
        "with free travel the reversal disappeared, which would mean cruise "
        "power really is the cause after all - update the docs"
    )
    assert long.diff > 0


def test_travel_cost_sharpens_the_early_advantage():
    """What cruise power does do, stated as the measurement that shows it."""
    kw = dict(steps=100, batch=24, seed=0)
    paid = paired_compare(Greedy(), Sustaining(), cfg=Config(), **kw)
    free = paired_compare(Greedy(), Sustaining(), cfg=Config(cruise_w=0.0), **kw)
    assert paid.diff < free.diff < 0, (
        "charging for travel should widen the greedy policy's early lead"
    )


def test_overlap_is_what_separates_the_policies():
    """The mechanism, measured rather than asserted.

    Greedy agents converge on the same stale ground and watch it together.
    Partitioning is what stops that, and it is the part of `Sustaining` that
    carries the long-run result.
    """
    def redundancy(policy, steps=900, batch=8, seed=0, after=200):
        env = SwarmEnv(Config(), batch=batch, seed=seed)
        footprints = union = 0.0
        for i in range(steps):
            env.step(*policy.act(env))
            observing = (~env.docked) & (env.soc > 1e-9)
            d = np.linalg.norm(
                env.pos[:, :, None, :] - env.sector_centres[None, None, :, :],
                axis=-1,
            )
            foot = (d < env.cfg.view_radius) & observing[..., None]
            if i >= after:
                footprints += foot.sum()
                union += foot.any(axis=1).sum()
        return footprints / max(union, 1)

    greedy = redundancy(Greedy())
    partitioned = redundancy(Sustaining())
    unpartitioned = redundancy(Sustaining(partition=False))

    assert greedy > 1.2, "the greedy policy stopped duplicating itself"
    assert partitioned < 1.05, "the partition stopped preventing overlap"
    assert partitioned < unpartitioned < greedy, (
        "removing the partition should move overlap back toward greedy"
    )
