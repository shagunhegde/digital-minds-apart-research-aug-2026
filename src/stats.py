"""Statistics shared by every gate.

Built first (Phase 0) because every gate report depends on it. Pure
numpy/scipy: this module runs without a GPU and without the model, so gate
reports can be re-derived from cached artifacts on any machine.

Conventions used throughout the sprint:
  - every proportion carries a Wilson 95% interval
  - every comparison carries an effect size, not only a p-value
  - every distribution reports median and IQR, not only the mean

Nothing in here interprets a result. Functions return numbers; gates print
them; a human reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval. `n` is the support size."""

    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] n={self.n}"


def wilson(successes: int, total: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because the sprint reports many
    proportions near 0 and 1 (per-concept detection rates especially), where
    the Wald interval leaves the unit interval and undercovers badly.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"bad counts: {successes}/{total}")
    if total == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    z = sps.norm.ppf(0.5 + conf / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), total)


def newcombe_diff(
    succ_a: int, tot_a: int, succ_b: int, tot_b: int, conf: float = 0.95
) -> Interval:
    """Newcombe's hybrid-score interval for a difference of two proportions.

    The order contrast (§6, figure 2) needs a CI on the *difference*, not two
    separate intervals; overlapping marginal intervals do not imply the
    difference straddles zero.
    """
    a = wilson(succ_a, tot_a, conf)
    b = wilson(succ_b, tot_b, conf)
    diff = a.point - b.point
    lo = diff - np.sqrt((a.point - a.lo) ** 2 + (b.hi - b.point) ** 2)
    hi = diff + np.sqrt((a.hi - a.point) ** 2 + (b.point - b.lo) ** 2)
    return Interval(diff, max(-1.0, lo), min(1.0, hi), tot_a + tot_b)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10_000,
    conf: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for `statistic` over a 1-D sample."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    reps = statistic(values[idx], axis=1)
    lo, hi = np.quantile(reps, [(1 - conf) / 2, 0.5 + conf / 2])
    return Interval(float(statistic(values)), float(lo), float(hi), values.size)


def jackknife(values: np.ndarray, statistic=np.mean) -> dict:
    """Leave-one-out sensitivity.

    Reported for each cascade factor: "a factor that swings on one concept is
    not a factor". Returns the full estimate, the leave-one-out range, and the
    index whose removal moves the statistic most.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return {"full": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "range": float("nan"), "most_influential": -1}
    full = float(statistic(values))
    loo = np.array(
        [float(statistic(np.delete(values, i))) for i in range(n)], dtype=float
    )
    return {
        "full": full,
        "lo": float(loo.min()),
        "hi": float(loo.max()),
        "range": float(loo.max() - loo.min()),
        "most_influential": int(np.argmax(np.abs(loo - full))),
    }


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    Non-parametric and rank-based, so it is meaningful for the J-lens vs
    logit-lens *rank* distributions in G1, which are heavy-tailed and
    censored at the vocabulary size.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # rank-based O(n log n) rather than the O(n*m) pairwise form
    combined = np.concatenate([a, b])
    ranks = sps.rankdata(combined)
    rank_a = ranks[: a.size].sum()
    u_a = rank_a - a.size * (a.size + 1) / 2.0
    return float(2.0 * u_a / (a.size * b.size) - 1.0)


def cohens_kappa(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Cohen's kappa plus the confusion matrix, for judge-vs-human agreement."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label shape mismatch: {a.shape} vs {b.shape}")
    categories = np.unique(np.concatenate([a, b]))
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    confusion = np.zeros((k, k), dtype=int)
    for x, y in zip(a, b):
        confusion[index[x], index[y]] += 1
    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = (confusion.sum(0) * confusion.sum(1)).sum() / (total * total)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {
        "kappa": float(kappa),
        "observed_agreement": float(observed),
        "expected_agreement": float(expected),
        "confusion": confusion,
        "categories": categories,
        "n": int(total),
    }


def spearman_ci(x: np.ndarray, y: np.ndarray, conf: float = 0.95) -> dict:
    """Spearman rho with a Fisher-z interval.

    Used for the cross-model rank correlation in G3 (our per-concept rates vs
    Macar's Gemma rates) and for the dose-response monotonicity in G2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": n}
    rho, p = sps.spearmanr(x, y)
    # Fieller/Bonett-Wright standard error for the Spearman z-transform
    stderr = 1.0 / np.sqrt(n - 3) * np.sqrt(1.0 + rho * rho / 2.0)
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    crit = sps.norm.ppf(0.5 + conf / 2.0)
    return {
        "rho": float(rho),
        "lo": float(np.tanh(z - crit * stderr)),
        "hi": float(np.tanh(z + crit * stderr)),
        "p": float(p),
        "n": n,
    }


# --------------------------------------------------------------------------
# distribution shape
# --------------------------------------------------------------------------


def median_iqr(values: np.ndarray) -> dict:
    """Median, quartiles and range. The sprint reports these, not the mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("median", "q1", "q3", "iqr", "min", "max")} | {"n": 0}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max()),
        "n": int(v.size),
    }


def mad_outliers(values: np.ndarray, threshold: float = 3.0) -> dict:
    """Indices beyond `threshold` median-absolute-deviations of the median."""
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scaled = 1.4826 * mad  # consistency constant for the normal
    if scaled == 0:
        return {"n_outliers": 0, "indices": np.array([], dtype=int), "mad": 0.0}
    deviation = np.abs(v - med) / scaled
    idx = np.flatnonzero(deviation > threshold)
    return {"n_outliers": int(idx.size), "indices": idx, "mad": float(scaled)}


def dip_statistic(values: np.ndarray) -> float:
    """Hartigan's dip: sup-distance from the ECDF to the nearest unimodal CDF.

    Delegates to the `diptest` package (a port of Hartigan's AS 217). This is
    the one place the sprint takes a dependency beyond the list in the build
    spec, and it is deliberate.

    A hand-rolled version was written first, from the tube characterisation:
    a unimodal G within sup-distance d of Fn, with mode at m, exists iff
    2d >= max(Fn - GCM Fn) on the left and 2d >= max(LCM Fn - Fn) on the
    right. That derivation is wrong, and quietly so. It treats the two
    segments as independent, but a single G takes one value at the mode, and
    Hartigan's dip minimises over a modal *interval* rather than a point.
    The consequence is a dip that is too small -- it agreed with an
    independent LP to machine precision on 15 of 21 samples and understated
    it by up to 1.7e-2 on the other six, which would understate bimodality
    exactly where G3 reads it as a headline. `scripts/00_verify_dip.py`
    reproduces that comparison; `diptest` matches the LP on every case.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4 or v.min() == v.max():
        return 0.0
    import diptest

    return float(diptest.dipstat(v))


def dip_test(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Dip statistic with a bootstrap p-value against the uniform null.

    The uniform is the least favourable unimodal null (Hartigan & Hartigan
    1985), so calibrating against uniform samples of the same n bounds the
    p-value for the whole unimodal class.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4 or v.min() == v.max():
        return {"dip": 0.0, "p": float("nan"), "n": int(v.size)}
    import diptest

    dip, p = diptest.diptest(v, boot_pval=True, n_boot=n_boot, seed=seed)
    return {"dip": float(dip), "p": float(p), "n": int(v.size), "n_boot": n_boot}


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample KS. G4a's temporal-drift probe compares the first and last
    decile of shards; a shift in the control statistic means something changed
    mid-run rather than between conditions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"D": float("nan"), "p": float("nan"), "n_a": int(a.size),
                "n_b": int(b.size)}
    result = sps.ks_2samp(a, b)
    return {"D": float(result.statistic), "p": float(result.pvalue),
            "n_a": int(a.size), "n_b": int(b.size)}


def ks_uniform(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> dict:
    """KS statistic against a uniform on [lo, hi].

    G1's random-token calibration: ranks of prompt-irrelevant tokens should be
    roughly uniform over the vocabulary. Systematic deviation indicates a
    dominant-direction pathology in the lens.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"D": float("nan"), "p": float("nan"), "n": 0}
    scaled = (v - lo) / (hi - lo)
    result = sps.kstest(scaled, "uniform")
    return {"D": float(result.statistic), "p": float(result.pvalue), "n": int(v.size)}


# --------------------------------------------------------------------------
# multiplicity and power
# --------------------------------------------------------------------------


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def min_detectable_effect(
    baseline: float, n_per_arm: int, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest absolute difference in proportions detectable at `power`.

    Required by G5: "a null without this is not a null". Solved by bisection
    on the two-proportion normal approximation.
    """
    if n_per_arm < 2:
        return float("nan")
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)

    def achieved(delta: float) -> float:
        p2 = min(1.0, baseline + delta)
        pooled = (baseline + p2) / 2
        se_null = np.sqrt(2 * pooled * (1 - pooled) / n_per_arm)
        se_alt = np.sqrt(
            (baseline * (1 - baseline) + p2 * (1 - p2)) / n_per_arm
        )
        if se_alt == 0:
            return np.inf
        return (delta - z_a * se_null) / se_alt

    lo, hi = 0.0, 1.0 - baseline
    if achieved(hi) < z_b:
        return float("nan")  # not detectable at any effect within range
    for _ in range(200):
        mid = (lo + hi) / 2
        if achieved(mid) < z_b:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------


def participation_ratio(singular_values: np.ndarray) -> float:
    """Effective rank as the participation ratio of the singular spectrum.

    (sum s^2)^2 / sum s^4 -- equals the true rank for a flat spectrum and
    collapses toward 1 when one direction dominates. Reported per layer in G0.
    """
    s = np.asarray(singular_values, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return float("nan")
    energy = s**2
    return float(energy.sum() ** 2 / (energy**2).sum())
