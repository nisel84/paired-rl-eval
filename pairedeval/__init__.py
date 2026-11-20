"""Paired evaluation for multi-agent reinforcement learning.

A small, dependency-light harness for comparisons you can defend:
common random numbers, bit-exact snapshot/restore, a visibility threshold
printed beside every difference, and window-length profiling that exposes
rankings which reverse when you measure for longer.
"""
from .env import Config, SwarmEnv
from .policies import Circular, Greedy, Policy, Sustaining
from .paired import Result, paired_compare, rollout, unpaired_compare
from .profile import crossings, flip_point, profile, trace

__version__ = "0.1.0"
__all__ = [
    "Config", "SwarmEnv",
    "Policy", "Greedy", "Circular", "Sustaining",
    "Result", "paired_compare", "unpaired_compare", "rollout",
    "profile", "trace", "crossings", "flip_point",
]
