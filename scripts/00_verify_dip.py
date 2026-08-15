"""Why stats.dip_statistic imports diptest instead of implementing the dip.

Not a test suite -- a one-time numerical verification, kept because it is the
evidence for a dependency the build spec does not list. Run it to reproduce
the comparison quoted in ASSETS.md.

Three independent computations of Hartigan's dip:
  hull   -- the tube-characterisation derivation written first (WRONG: it
            decouples the two segments at the mode)
  LP     -- direct linear program, minimise tube half-width subject to
            convex-left / concave-right shape constraints, scanning the mode
  diptest -- the reference AS 217 port

LP and diptest agree everywhere. The hull version agrees on most samples and
silently understates the dip on the rest.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def dip_via_hull(values):
    """The wrong derivation, kept so the failure is reproducible."""

    def gcm_dev(xs, upper, lower):
        n = xs.size
        if n < 3:
            return 0.0
        hull = []
        for i in range(n):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (lower[b] - lower[a]) * (xs[i] - xs[a]) >= (
                    lower[i] - lower[a]
                ) * (xs[b] - xs[a]):
                    hull.pop()
                else:
                    break
            hull.append(i)
        return float(np.max(upper - np.interp(xs, xs[hull], lower[hull])))

    x = np.sort(np.asarray(values, float))
    n = x.size
    lower, upper = np.arange(n) / n, np.arange(1, n + 1) / n
    best = np.inf
    for m in range(n):
        left = gcm_dev(x[: m + 1], upper[: m + 1], lower[: m + 1])
        right = gcm_dev(-x[m:][::-1], (-lower[m:])[::-1], (-upper[m:])[::-1])
        best = min(best, max(left, right))
    return float(best / 2.0)


def dip_via_lp(values):
    """Exact: one LP per candidate mode, minimising the tube half-width."""
    x = np.sort(np.asarray(values, float))
    n = x.size
    lower, upper = np.arange(n) / n, np.arange(1, n + 1) / n
    best = np.inf
    for m in range(n):
        nv = n + 1
        rows, rhs = [], []
        for i in range(n):  # G_i - t <= lower_i
            r = np.zeros(nv); r[i] = 1.0; r[n] = -1.0
            rows.append(r); rhs.append(lower[i])
        for i in range(n):  # -G_i - t <= -upper_i
            r = np.zeros(nv); r[i] = -1.0; r[n] = -1.0
            rows.append(r); rhs.append(-upper[i])
        for i in range(n - 2):
            d1, d2 = x[i + 1] - x[i], x[i + 2] - x[i + 1]
            if d1 <= 0 or d2 <= 0:
                continue
            r = np.zeros(nv)
            r[i], r[i + 1], r[i + 2] = 1.0 / d1, -1.0 / d1 - 1.0 / d2, 1.0 / d2
            if i + 2 <= m:      # convex left of the mode
                rows.append(-r); rhs.append(0.0)
            elif i >= m:        # concave right of the mode
                rows.append(r); rhs.append(0.0)
            # the triple straddling the mode is the kink: unconstrained
        c = np.zeros(nv); c[n] = 1.0
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=[(None, None)] * n + [(0, None)], method="highs")
        if res.success:
            best = min(best, res.x[n])
    return float(best)


def main():
    from stats import dip_statistic  # noqa: E402  (imports diptest)

    rng = np.random.default_rng(7)
    cases = []
    for t in range(6):
        cases.append((f"uniform n=30 #{t}", rng.random(30)))
    for t in range(6):
        cases.append((f"bimodal sep=6 n=30 #{t}",
                      np.concatenate([rng.normal(0, 1, 15), rng.normal(6, 1, 15)])))
    for t in range(4):
        cases.append((f"normal n=25 #{t}", rng.normal(0, 1, 25)))
    for t in range(4):
        cases.append((f"2-cluster n=24 #{t}",
                      np.concatenate([rng.random(12) * .05, 1 - rng.random(12) * .05])))

    print(f"{'case':26} {'hull':>12} {'LP':>12} {'diptest':>12} "
          f"{'|hull-LP|':>10} {'|dt-LP|':>9}")
    print("-" * 88)
    worst_hull = worst_ref = 0.0
    for name, sample in cases:
        h, l, d = dip_via_hull(sample), dip_via_lp(sample), dip_statistic(sample)
        worst_hull = max(worst_hull, abs(h - l))
        worst_ref = max(worst_ref, abs(d - l))
        print(f"{name:26} {h:12.9f} {l:12.9f} {d:12.9f} "
              f"{abs(h-l):10.2e} {abs(d-l):9.2e}")
    print("-" * 88)
    print(f"max |hull - LP|    over {len(cases)} samples: {worst_hull:.3e}")
    print(f"max |diptest - LP| over {len(cases)} samples: {worst_ref:.3e}")


if __name__ == "__main__":
    main()
