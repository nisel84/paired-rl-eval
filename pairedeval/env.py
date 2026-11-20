"""A compact persistent-surveillance environment.

Deliberately small: a square site divided into sectors, a swarm of agents with
finite batteries, and a base they must return to in order to recharge.  It is
not the environment from any real study — it is the smallest thing that still
produces the phenomenon the measurement harness was built to detect: a cost
that a short measurement window cannot see.

That cost turned out not to be the one this file was built around.  Power is
split into a hover term and a cruise term, so crossing the site costs more than
holding station, and the first version of this docstring claimed that travel
energy was what punished a greedy policy over a long horizon.  Setting
``cruise_w=0`` refutes it: with travel made free the ranking still reverses,
slightly later and by a wider margin.  ``tests/test_window.py`` keeps that
check so the claim cannot quietly come back.

What the travel cost does do is sharpen the *early* advantage of chasing
distant stale sectors — the paired difference at 100 steps goes from -0.009 to
-0.061 when it is switched on — which makes the reversal more dramatic without
causing it.

The mechanism that does cause it is sensor overlap.  Agents that all chase the
globally stalest sector converge on the same region and watch it together: over
the back two thirds of an episode each observed sector is seen by 1.32 aircraft
on average, so about a quarter of the fleet's sensor capacity goes on
duplicating a neighbour.  A partitioned patrol holds that figure at 1.01.  The
waste is invisible at first, while staleness is still uniform and the agents
are still spread out, and it builds as the staleness field develops.  That
delay is what makes a short window lie, and
``examples/03_ablation.py`` measures it.

Everything is vectorised over a batch of worlds and driven by an explicit
``numpy`` Generator, so a run is reproducible bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Config", "SwarmEnv"]


@dataclass(frozen=True)
class Config:
    """Task parameters.  One instance describes one task distribution."""

    n_agents: int = 6
    grid: int = 8                 # grid x grid sectors
    side: float = 500.0           # metres
    speed: float = 12.0           # m/s, cruise
    dt: float = 2.0               # s per decision step
    view_radius: float = 70.0     # m, ground footprint radius
    capacity_wh: float = 32.0     # usable energy per agent
    hover_w: float = 240.0        # power while airborne and stationary
    cruise_w: float = 180.0       # additional power at full cruise speed
    recharge_w: float = 380.0     # power delivered on the pad
    reserve: float = 0.20         # state of charge held back for the return leg
    base_xy: tuple[float, float] = (0.5, 0.5)   # fraction of side
    soc_init: tuple[float, float] = (0.55, 1.0)  # fleet arrives part-charged

    def scaled(self, **kw) -> "Config":
        return replace(self, **kw)

    # ------------------------------------------------------------- endurance
    @property
    def hover_steps(self) -> float:
        """Steps a full battery lasts holding station."""
        return self.capacity_wh * 3600.0 / self.hover_w / self.dt

    @property
    def cruise_steps(self) -> float:
        """Steps a full battery lasts at full cruise speed."""
        return self.capacity_wh * 3600.0 / (self.hover_w + self.cruise_w) / self.dt

    @property
    def recharge_steps(self) -> float:
        """Steps to fill an empty battery."""
        return self.capacity_wh * 3600.0 / self.recharge_w / self.dt


class SwarmEnv:
    """Vectorised over ``batch`` independent worlds sharing one Config."""

    def __init__(self, cfg: Config = Config(), batch: int = 16, seed: int = 0):
        self.cfg = cfg
        self.batch = batch
        self.seed = seed
        self.reset(seed)

    # ---------------------------------------------------------------- state
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        c, B, N = self.cfg, self.batch, self.cfg.n_agents
        self.pos = self.rng.random((B, N, 2)) * c.side
        lo, hi = c.soc_init
        self.soc = self.rng.uniform(lo, hi, size=(B, N))
        self.docked = np.zeros((B, N), dtype=bool)
        self.staleness = np.zeros((B, c.grid, c.grid))
        self.t = 0
        self._centres = self._sector_centres()
        self.base = np.array(c.base_xy) * c.side

    def _sector_centres(self) -> np.ndarray:
        c = self.cfg
        k = (np.arange(c.grid) + 0.5) / c.grid * c.side
        gx, gy = np.meshgrid(k, k, indexing="ij")
        return np.stack([gx, gy], -1).reshape(-1, 2)      # (grid*grid, 2)

    # ------------------------------------------------------- snapshot/restore
    def snapshot(self) -> dict:
        """Full state, deep-copied.  ``restore`` of a snapshot is bit-exact."""
        return {
            "pos": self.pos.copy(),
            "soc": self.soc.copy(),
            "docked": self.docked.copy(),
            "staleness": self.staleness.copy(),
            "t": self.t,
            "rng": self.rng.bit_generator.state,
        }

    def restore(self, snap: dict) -> None:
        self.pos = snap["pos"].copy()
        self.soc = snap["soc"].copy()
        self.docked = snap["docked"].copy()
        self.staleness = snap["staleness"].copy()
        self.t = snap["t"]
        self.rng.bit_generator.state = snap["rng"]

    # ----------------------------------------------------------------- step
    def step(self, target: np.ndarray, go_home: np.ndarray) -> dict:
        """Advance one decision step.

        Parameters
        ----------
        target : (batch, n_agents, 2) desired positions in metres.
        go_home : (batch, n_agents) bool, request to dock and recharge.
        """
        c = self.cfg
        step_m = c.speed * c.dt

        # An agent with a flat battery cannot fly, and cannot reach the pad.
        flat = self.soc <= 1e-9
        flying = (~self.docked) & (~flat)

        want = np.where(go_home[..., None], self.base[None, None, :], target)
        delta = want - self.pos
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        move = np.where(dist > step_m, delta / np.maximum(dist, 1e-9) * step_m, delta)
        self.pos = np.where(flying[..., None], self.pos + move, self.pos)
        self.pos = np.clip(self.pos, 0.0, c.side)

        travelled = np.linalg.norm(move[..., 0:2], axis=-1) * flying
        duty = np.clip(travelled / max(step_m, 1e-9), 0.0, 1.0)

        at_base = np.linalg.norm(self.pos - self.base, axis=-1) < step_m
        self.docked = np.where(at_base & go_home, True, self.docked)
        self.docked = np.where(self.soc >= 0.995, False, self.docked)

        # Power: hover plus a cruise term proportional to how far it actually
        # moved this step.  Travel is what makes a battery expensive.
        watts = c.hover_w + c.cruise_w * duty
        spent = np.where(flying, watts * c.dt / 3600.0 / c.capacity_wh, 0.0)
        gained = c.recharge_w * c.dt / 3600.0 / c.capacity_wh
        self.soc = np.where(self.docked, self.soc + gained, self.soc - spent)
        self.soc = np.clip(self.soc, 0.0, 1.0)

        observing = (~self.docked) & (self.soc > 1e-9)

        d = np.linalg.norm(
            self.pos[:, :, None, :] - self._centres[None, None, :, :], axis=-1
        )                                                   # (B, N, S)
        seen = ((d < c.view_radius) & observing[..., None]).any(axis=1)
        seen = seen.reshape(self.batch, c.grid, c.grid)

        self.staleness = np.where(seen, 0.0, self.staleness + c.dt)
        self.t += 1

        return {
            "coverage": seen.mean(axis=(1, 2)),
            "mean_staleness": self.staleness.mean(axis=(1, 2)) / 60.0,
            "max_staleness": self.staleness.max(axis=(1, 2)) / 60.0,
            "soc": self.soc.mean(axis=1),
            "airborne": observing.mean(axis=1),
        }

    # -------------------------------------------------------------- helpers
    @property
    def sector_centres(self) -> np.ndarray:
        return self._centres

    def needs_return(self, margin: np.ndarray | float | None = None) -> np.ndarray:
        """True where the remaining charge only just covers the trip home.

        ``margin`` overrides ``Config.reserve`` per agent, which is how a policy
        staggers its rotation instead of sending the whole swarm down at once.
        """
        c = self.cfg
        if margin is None:
            margin = c.reserve
        home = np.linalg.norm(self.pos - self.base, axis=-1)
        legs = home / max(c.speed, 1e-9) / c.dt
        need = legs * (c.hover_w + c.cruise_w) * c.dt / 3600.0 / c.capacity_wh
        return self.soc <= need + margin
