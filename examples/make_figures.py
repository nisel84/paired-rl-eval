"""Regenerate the two figures in the README.

Run:  python examples/make_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pairedeval import (Circular, Greedy, Sustaining, crossings, flip_point,
                        trace)
from pairedeval.profile import _cumulative

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
STEPS, BATCH, SEED = 900, 24, 0

INK = "#1f2328"
MUTED = "#8b949e"
COLOUR = {"greedy": "#c0392b", "circular": "#b07d2b", "sustaining": "#1f6f8b"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def _minutes(steps: int) -> np.ndarray:
    return (np.arange(steps) + 1) * 2 / 60.0


def figure_one(traces: dict[str, np.ndarray]) -> None:
    """Instantaneous coverage: the shape a single end-of-episode number hides."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=160)
    x = _minutes(STEPS)
    smooth = 25
    kernel = np.ones(smooth) / smooth
    for name, y in traces.items():
        ax.plot(x, np.convolve(y, kernel, mode="same"), lw=1.6,
                color=COLOUR[name], label=name)
    ax.set_xlabel("simulated minutes", color=INK, fontsize=9)
    ax.set_ylabel("sectors observed", color=INK, fontsize=9)
    ax.set_title("Coverage within the episode, 25-step moving average",
                 color=INK, fontsize=10, loc="left", pad=10)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_trace.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def figure_two(traces: dict[str, np.ndarray], flip: int | None,
               marks: list[int]) -> None:
    """What a study would have reported, as a function of how long it ran."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=160)
    x = _minutes(STEPS)
    for name, y in traces.items():
        ax.plot(x, _cumulative(y), lw=1.8, color=COLOUR[name], label=name)

    top = ax.get_ylim()[1]
    for c in marks:
        if c != flip:
            ax.axvline(c * 2 / 60.0, color=MUTED, lw=0.7, ls=(0, (2, 3)),
                       alpha=0.75)
    if flip is not None:
        fx = flip * 2 / 60.0
        ax.axvline(fx, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
        ax.annotate(
            f"greedy and sustaining swap places here, at step {flip}\n"
            f"({fx:.1f} min); before it the shorter study wins the argument",
            xy=(fx, top * 0.42), xytext=(fx + 1.6, top * 0.17),
            fontsize=8, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, alpha=0.6),
        )

    ax.set_xlabel("length of the measurement window, minutes", color=INK,
                  fontsize=9)
    ax.set_ylabel("mean coverage reported", color=INK, fontsize=9)
    ax.set_title("The same experiment, stopped at different times",
                 color=INK, fontsize=10, loc="left", pad=10)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "window_flip.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    traces = {
        p.name: trace(p, steps=STEPS, batch=BATCH, seed=SEED)
        for p in (Greedy(), Circular(), Sustaining())
    }
    flip = flip_point(Greedy(), Sustaining(), steps=STEPS, batch=BATCH,
                      seed=SEED)
    marks = crossings(Greedy(), Sustaining(), steps=STEPS, batch=BATCH,
                      seed=SEED)
    figure_one(traces)
    figure_two(traces, flip, marks)
    print(f"wrote {OUT / 'coverage_trace.png'}")
    print(f"wrote {OUT / 'window_flip.png'}")


if __name__ == "__main__":
    main()
