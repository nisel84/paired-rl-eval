"""The same two policies, ranked opposite ways by two honest measurements.

`Greedy` chases whichever sector is stalest, anywhere on the site.  It wins on
coverage for the first minute or two, while staleness is still uniform and its
agents are still spread out.  As the staleness field develops they converge on
the same ground and start watching it together, and by the half-hour mark
`Sustaining`, which gives each agent its own patch, is ahead.

See `03_ablation.py` for the measurement behind that sentence.

Neither number is wrong.  The window length is the hidden variable, so this
example makes it explicit.

Run:  python examples/02_window_flip.py
"""
from pairedeval import (Circular, Greedy, Sustaining, crossings, flip_point,
                        paired_compare, profile)

BATCH, SEED, STEPS = 24, 0, 900


def main() -> None:
    policies = [Greedy(), Circular(), Sustaining()]

    edges, _ = profile(policies[0], steps=STEPS, buckets=9, batch=BATCH, seed=SEED)
    print("mean coverage over the first N steps (N across the top)\n")
    print("              " + "".join(f"{e:>7d}" for e in edges))
    for p in policies:
        _, means = profile(p, steps=STEPS, buckets=9, batch=BATCH, seed=SEED)
        print(f"{p.name:>12}  " + "".join(f"{v:>7.3f}" for v in means))

    print("\nbest at 100 steps :", max(
        policies, key=lambda p: profile(p, STEPS, 9, "coverage", None, BATCH, SEED)[1][0]
    ).name)
    print("best at 900 steps :", max(
        policies, key=lambda p: profile(p, STEPS, 9, "coverage", None, BATCH, SEED)[1][-1]
    ).name)

    marks = crossings(Greedy(), Sustaining(), steps=STEPS, batch=BATCH, seed=SEED)
    flip = flip_point(Greedy(), Sustaining(), steps=STEPS, batch=BATCH, seed=SEED)
    print(f"\ngreedy vs sustaining: ranking changes hands at steps {marks}")
    print(f"and settles from step {flip} ({flip * 2 / 60:.1f} min of simulated "
          f"time) onward.")
    print("flip_point reports the last crossing, not the first: the first is "
          "the\neasy number to quote and the wrong one to act on.")

    print("\nthe same comparison, measured twice:")
    for steps in (100, STEPS):
        r = paired_compare(Greedy(), Sustaining(), steps=steps, batch=32, seed=SEED)
        print(f"  {steps:>3} steps: {r}")

    print("\nReport one of those two lines and you have said something true.")
    print("Report it without the window length and you have said nothing.")


if __name__ == "__main__":
    main()
