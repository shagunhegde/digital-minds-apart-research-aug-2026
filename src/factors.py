"""The three-factor decomposition.

    P(report) = P(represented) x P(verbalizable | represented) x P(reported | verbalizable)

Pure numpy: everything GPU-shaped (lens readouts, probe residuals, the position
control) is done by scripts/04_factors.py and written to disk, so this module
and the gate that uses it run anywhere.
"""

from __future__ import annotations

import numpy as np


def cascade_residual(f1: float, f2: float, f3: float, observed_rate: float) -> float:
    """Chain rule makes the product exact. Nonzero residual = a wrong denominator.

    With nested sets C subset B subset A subset N, the product telescopes:
    (|A|/N)(|B|/|A|)(|C|/|B|) = |C|/N. So on a correct implementation this is
    zero to floating point. It is a bug detector, not a finding: it fires when
    a stage silently drops trials, when a denominator is taken over a different
    trial set, or when NaNs are dropped at one stage and not another.
    """
    return abs(f1 * f2 * f3 - observed_rate)


def cluster_bootstrap(
    concepts: np.ndarray, fn, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Resample CONCEPTS with replacement, not trials.

    Trials inside a concept share one prefill residual -- f1 and f2 are
    constant across the samples drawn from it, and only f3 varies. Resampling
    trials would treat those as independent and understate every interval.
    Returns (point, lo, hi).
    """
    unique = np.unique(concepts)
    rng = np.random.default_rng(seed)
    index_of = {c: np.flatnonzero(concepts == c) for c in unique}
    reps = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        value = fn(np.concatenate([index_of[c] for c in drawn]))
        if np.isfinite(value):
            reps.append(value)
    point = fn(np.arange(concepts.size))
    if not reps:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Fold indices split by GROUP, not by trial.

    The probe must never see a concept in training that it is scored on: with
    trial-level folds it can memorise concept identity and f1 becomes a
    measure of how well concepts are distinguishable from each other, not of
    whether an injection is present.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    buckets = np.array_split(shuffled, min(n_folds, len(unique)))
    return [np.flatnonzero(np.isin(groups, held)) for held in buckets]


def mass_mean_probe(
    injected: np.ndarray,
    control: np.ndarray,
    injected_groups: np.ndarray,
    control_groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    shuffle_labels: bool = False,
) -> dict:
    """Cross-validated mass-mean probe separating injected from control residuals.

    Direction is mean(injected) - mean(control) on the training folds; the
    score is the projection onto it. Every score is out-of-fold.

    `shuffle_labels` is the null arm: labels are permuted within the training
    folds before the direction is computed, so the probe should collapse to
    chance and f1 should land at the FPR the threshold was set to.
    """
    features = np.concatenate([injected, control])
    labels = np.concatenate([np.ones(len(injected)), np.zeros(len(control))])
    groups = np.concatenate([injected_groups, control_groups])
    scores = np.full(len(features), np.nan)
    rng = np.random.default_rng(seed)

    for held in grouped_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(features)), held)
        train_labels = labels[train]
        if shuffle_labels:
            train_labels = rng.permutation(train_labels)
        pos = features[train][train_labels == 1]
        neg = features[train][train_labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        direction = pos.mean(axis=0) - neg.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm
        scores[held] = features[held] @ direction

    return {
        "scores_injected": scores[: len(injected)],
        "scores_control": scores[len(injected):],
        "n_folds": min(n_folds, len(np.unique(groups))),
    }


def threshold_at_fpr(control_scores: np.ndarray, fpr: float = 0.05) -> float:
    """The score above which `fpr` of control trials fall."""
    valid = control_scores[np.isfinite(control_scores)]
    if valid.size == 0:
        return float("nan")
    return float(np.quantile(valid, 1.0 - fpr))


def compute_f1(injected_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Per-trial boolean: does this injection trial clear the probe threshold."""
    return np.isfinite(injected_scores) & (injected_scores > threshold)


def compute_f2(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-trial boolean: is the concept token inside the top-k lens readout.

    `ranks` is 1-indexed, so "inside top-k" is rank <= k.
    """
    return np.isfinite(ranks) & (ranks <= k)


def cascade(
    f1_pass: np.ndarray, f2_pass: np.ndarray, f3_pass: np.ndarray
) -> dict:
    """The three conditional rates and their survivorship counts.

    Each factor is conditioned on the previous stage surviving, which is what
    makes the product telescope to the end-to-end rate.
    """
    n = int(f1_pass.size)
    a = f1_pass
    b = a & f2_pass
    c = b & f3_pass
    n_a, n_b, n_c = int(a.sum()), int(b.sum()), int(c.sum())
    f1 = n_a / n if n else float("nan")
    f2 = n_b / n_a if n_a else float("nan")
    f3 = n_c / n_b if n_b else float("nan")
    observed = n_c / n if n else float("nan")
    return {
        "f1": f1, "f2": f2, "f3": f3,
        "observed_cascade_rate": observed,
        "n_entering_f1": n, "n_surviving_f1": n_a,
        "n_surviving_f2": n_b, "n_surviving_f3": n_c,
        "residual": cascade_residual(f1, f2, f3, observed)
        if n_a and n_b else float("nan"),
    }


def k_curve(
    f1_pass: np.ndarray, ranks: np.ndarray, f3_pass: np.ndarray, ks: list[int]
) -> list[dict]:
    """f2 (and the whole cascade) as a function of k.

    f2 moves monotonically with k by construction, so a single k is not a
    reportable quantity -- the curve is.
    """
    rows = []
    for k in ks:
        row = cascade(f1_pass, compute_f2(ranks, k), f3_pass)
        row["k"] = k
        rows.append(row)
    return rows


def jackknife_factor(
    concepts: np.ndarray,
    f1_pass: np.ndarray,
    f2_pass: np.ndarray,
    f3_pass: np.ndarray,
    which: str,
) -> dict:
    """Leave-one-concept-out range for one factor.

    A factor that swings on the removal of a single concept is not a factor.
    """
    unique = np.unique(concepts)
    full = cascade(f1_pass, f2_pass, f3_pass)[which]
    values = []
    for concept in unique:
        keep = concepts != concept
        values.append(cascade(f1_pass[keep], f2_pass[keep], f3_pass[keep])[which])
    values = np.array(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"full": full, "lo": float("nan"), "hi": float("nan"),
                "range": float("nan"), "most_influential": ""}
    return {
        "full": full,
        "lo": float(finite.min()),
        "hi": float(finite.max()),
        "range": float(finite.max() - finite.min()),
        "most_influential": str(unique[int(np.nanargmax(np.abs(values - full)))]),
    }


def per_concept_profiles(
    concepts: np.ndarray,
    f1_pass: np.ndarray,
    f2_pass: np.ndarray,
    f3_pass: np.ndarray,
) -> dict:
    """The factor triple for each concept separately.

    The aggregate can be a mixture of concepts that always survive and
    concepts that never do; only the per-concept view distinguishes those.
    """
    out = {}
    for concept in np.unique(concepts):
        mask = concepts == concept
        out[str(concept)] = cascade(f1_pass[mask], f2_pass[mask], f3_pass[mask])
    return out
