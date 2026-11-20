"""When pairing pays, and when it buys nothing.

Common random numbers are usually sold as a free improvement.  They are not.
Pairing removes the variance the two arms *share*, so it helps in proportion to
how alike they are.  Two near-identical variants respond to a world the same
way and almost all of the variance cancels; two genuinely different policies
respond differently and there is little left to cancel.

The table below is the whole argument.  It also sets expectations honestly: if
you are comparing a trained policy against a hand-written baseline, pairing
will tidy your bookkeeping and will not shrink your error bars much.  If you
are running an ablation, it is worth a factor of two.

Each row is an average over several seeds, because a standard error estimated
from one seed is itself noisy.  That makes this the slow example: about a
minute, against a second or two for the others.

Run:  python examples/04_when_pairing_pays.py
"""
import numpy as np

from pairedeval import (Circular, Greedy, Sustaining, paired_compare,
                        unpaired_compare)

STEPS, BATCH, SEEDS = 250, 24, 6


def gain(a, b) -> tuple[float, float, float]:
    paired, careless = [], []
    for seed in range(SEEDS):
        kw = dict(steps=STEPS, batch=BATCH, seed=seed)
        paired.append(paired_compare(a, b, **kw).stderr)
        careless.append(unpaired_compare(a, b, **kw).stderr)
    p, u = float(np.mean(paired)), float(np.mean(careless))
    return p, u, (u / p if p > 0 else float("inf"))


def main() -> None:
    base = 0.15
    rows = [(f"return threshold +{d * 100:.1f} pp",
             Greedy(return_at=base), Greedy(return_at=base + d))
            for d in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)]
    rows += [
        ("sustaining ablation", Sustaining(), Sustaining(partition=False)),
        ("greedy vs sustaining", Greedy(), Sustaining()),
        ("greedy vs circular", Greedy(), Circular()),
    ]

    print(f"{SEEDS} seeds, {BATCH} worlds, {STEPS} steps "
          f"(this takes about a minute)\n")
    print(f"{'arms':<30}{'paired se':>12}{'unpaired se':>14}{'gain':>9}")
    for label, a, b in rows:
        p, u, g = gain(a, b)
        shown = "exact" if p == 0 else f"{g:.1f}x"
        print(f"{label:<30}{p:>12.5f}{u:>14.5f}{shown:>9}")

    print("\nThe first row is a policy against itself.  Paired, the difference "
          "is\nexactly zero with no error bar at all; unpaired, the same policy "
          "looks\ndifferent from itself by more than some of the real effects "
          "below it.")
    print("\nThe gain decays from about 2x to about 1x as the arms diverge. "
          "Below\nroughly 1.2x, pairing is still worth keeping for "
          "reproducibility, but it\nis no longer buying statistical power, and "
          "claiming otherwise is\noverselling the method.")


if __name__ == "__main__":
    main()
