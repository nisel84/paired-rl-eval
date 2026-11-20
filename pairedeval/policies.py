"""Reference policies.

These are hand-written, not learned.  They exist so the measurement harness has
something to measure, and because two of them reproduce the effect the harness
was built to catch: a policy that wins on a short window and loses on a long
one.  Swap in a trained policy by implementing ``act``.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import SwarmEnv

__all__ = ["Policy", "Greedy", "Circular", "Sustaining"]


def _deconflict(score: np.ndarray) -> np.ndarray:
    """Give each agent its own sector: agent i takes its i-th best choice.

    Without this every agent chases the same stalest sector and the swarm
    behaves like a single aircraft.  The fix is deliberately crude — the point
    of these policies is to be measurable, not to be good.
    """
    order = np.argsort(-score, axis=-1)                      # (B, N, S)
    n = score.shape[1]
    return order[:, np.arange(n), np.arange(n)]


class Policy(Protocol):
    name: str

    def act(self, env: SwarmEnv) -> tuple[np.ndarray, np.ndarray]:
        """Return (target_xy, go_home) for the current state."""
        ...


class Greedy:
    """Fly at whatever sector is stalest, anywhere on the site.

    The distance term is deliberately weak: staleness grows every step while
    distance does not, so within a minute or two the stalest sector wins no
    matter how far away it is.  That is the pathology — it buys coverage now
    with cruise energy, and cruise energy is what the battery cannot spare.
    It does come home — ``return_at`` is set high enough that the return leg
    is affordable from anywhere on the site, so this baseline is not broken by
    construction.  What it does not do is take the leg into account when it
    chooses where to fly, so it pays for one every cycle.
    """

    name = "greedy"

    def __init__(self, distance_weight: float = 0.02, return_at: float = 0.15):
        self.distance_weight = distance_weight
        self.return_at = return_at

    def act(self, env: SwarmEnv):
        B = env.batch
        stale = env.staleness.reshape(B, -1)                     # (B, S)
        centres = env.sector_centres                             # (S, 2)
        d = np.linalg.norm(
            env.pos[:, :, None, :] - centres[None, None, :, :], axis=-1
        )                                                        # (B, N, S)
        score = stale[:, None, :] - self.distance_weight * d
        pick = _deconflict(score)
        return centres[pick], env.soc <= self.return_at


class Circular:
    """Sweep a fixed ring.  Predictable, and blind to the energy budget."""

    name = "circular"

    def __init__(self, radius_frac: float = 0.33):
        self.radius_frac = radius_frac

    def act(self, env: SwarmEnv):
        c = env.cfg
        B, N = env.batch, c.n_agents
        phase = 2 * np.pi * (np.arange(N) / N + env.t * c.dt / 600.0)
        r = self.radius_frac * c.side
        ring = np.stack([np.cos(phase), np.sin(phase)], -1) * r + c.side / 2
        return np.broadcast_to(ring, (B, N, 2)).copy(), env.soc <= 1e-3


class Sustaining:
    """Partitioned patrol with a staggered rotation.

    Two differences from ``Greedy``, both aimed at the energy budget rather
    than at coverage:

    * the site is split into one patch per agent, and an agent only ever flies
      at the stalest sector *inside its own patch*, so two aircraft cannot end
      up watching the same ground.  This is what carries the long-run result:
      it holds sensor overlap at 1.01 aircraft per observed sector against the
      greedy policy's 1.32;
    * agents return on a reserve that differs per agent, so the swarm does not
      all land in the same minute.  The spread is wide on purpose: a narrow
      one lets the fleet re-synchronise after the first cycle, and the site
      then empties out periodically even though the mean looks healthy.  At
      the default the site is never left with less than a tenth of its sectors
      observed; at ``stagger=0.22`` that figure is a hundredth.

    It covers less than ``Greedy`` early, while staleness is still uniform and
    the greedy swarm is still spread out, and more once the staleness field has
    developed enough to pull the greedy agents onto the same ground.  Note what
    is *not* the reason: the two policies spend almost the same fraction of the
    episode airborne (0.48 against 0.53), so this is not a duty-cycle result.

    Both mechanisms can be switched off independently, which is what makes an
    ablation possible: ``partition=False`` lets every agent roam the whole site,
    ``stagger=0.0`` sends the swarm down together.
    """

    name = "sustaining"

    def __init__(self, stagger: float = 0.45, partition: bool = True):
        self.stagger = stagger
        self.partition = partition
        self._patch: tuple[int, ...] | None = None
        self._assign: np.ndarray | None = None

    # Patches are a function of the grid alone, so build them once.
    def _patches(self, env: SwarmEnv) -> np.ndarray:
        key = (env.cfg.grid, env.cfg.n_agents)
        if self._patch == key and self._assign is not None:
            return self._assign
        centres = env.sector_centres
        N = env.cfg.n_agents
        # Split by angle about the centre: contiguous wedges, every sector owned.
        mid = centres.mean(axis=0)
        ang = np.arctan2(*(centres - mid).T[::-1])            # (S,)
        owner = np.floor((ang + np.pi) / (2 * np.pi) * N).astype(int) % N
        self._patch, self._assign = key, owner
        return owner

    def act(self, env: SwarmEnv):
        B, N = env.batch, env.cfg.n_agents
        centres = env.sector_centres
        owner = self._patches(env)

        stale = env.staleness.reshape(B, -1)                     # (B, S)
        d = np.linalg.norm(
            env.pos[:, :, None, :] - centres[None, None, :, :], axis=-1
        )                                                        # (B, N, S)
        if self.partition:
            mine = owner[None, :] == np.arange(N)[:, None]        # (N, S)
            score = np.where(
                mine[None, :, :], stale[:, None, :] - 0.02 * d, -np.inf
            )
            pick = score.argmax(axis=-1)                         # (B, N)
        else:
            pick = _deconflict(stale[:, None, :] - 0.02 * d)

        margin = env.cfg.reserve + self.stagger * (np.arange(N) / max(N - 1, 1))
        return centres[pick], env.needs_return(margin[None, :])
