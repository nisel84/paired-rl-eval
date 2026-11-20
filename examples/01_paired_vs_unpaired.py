"""What pairing buys you.

Two policies that differ only in when they turn for home: one at 15% charge,
one at 16%.  The difference is real but small.  Measured on independently
sampled worlds it disappears into task variance; measured on the same worlds,
from the same state, with the same random draws, it is resolvable with the
same number of episodes.

Pairing is not a free win, and this example is deliberately set up where it
pays.  Common random numbers help in proportion to how alike the two arms are:
about 2x on arms this close, nothing at all on two genuinely different
policies.  `04_when_pairing_pays.py` measures the whole curve.

Run:  python examples/01_paired_vs_unpaired.py
"""
from pairedeval import Greedy, paired_compare, unpaired_compare

STEPS, BATCH, SEED = 400, 24, 0


def main() -> None:
    a = Greedy(return_at=0.15)
    b = Greedy(return_at=0.16)
    a.name, b.name = "return@15%", "return@16%"

    paired = paired_compare(a, b, steps=STEPS, batch=BATCH, seed=SEED)
    careless = unpaired_compare(a, b, steps=STEPS, batch=BATCH, seed=SEED)

    print(f"{BATCH} worlds, {STEPS} steps each\n")
    for label, r in (("paired  ", paired), ("unpaired", careless)):
        print(f"{label}  diff {r.diff:+.5f}   visibility {r.visibility:.5f}   "
              f"t {r.t:+6.2f}   -> {r.verdict}")

    ratio = careless.visibility / paired.visibility
    print(f"\nPairing shrank the smallest resolvable effect by {ratio:.1f}x.")
    if paired.visible and not careless.visible:
        print("The effect is real.  Only one of these two measurements can see it.")
    print("\nThe unpaired point estimate is also further off: "
          f"{careless.diff:+.5f} against {paired.diff:+.5f}.  "
          "Noise moves the estimate, not just the error bar.")


if __name__ == "__main__":
    main()
