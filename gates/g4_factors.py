"""GATE G4: the three-factor decomposition.

    P(report) = P(represented) x P(verbalizable | represented) x P(reported | verbalizable)

Runs on artifacts/factors/factors_input.npz -- no GPU, so it can be re-run
freely while the sweep is still going.

Emits a report and stops. It judges nothing.

    python gates/g4_factors.py [--fpr 0.05] [--probe-folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import factors as fac  # noqa: E402
import stats  # noqa: E402

RULE = "=" * 78
KS = [1, 5, 10, 25, 50]


def fmt(interval: stats.Interval) -> str:
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def concept_bootstrap(concepts, fn, n_boot=2000, seed=0) -> stats.Interval:
    """Cluster bootstrap over concepts, wrapped as an Interval for printing."""
    point, lo, hi = fac.cluster_bootstrap(concepts, fn, n_boot, seed)
    return stats.Interval(point, lo, hi, int(np.unique(concepts).size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=Path, default=ROOT / "artifacts" / "factors")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--probe-folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=10, help="headline k")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    path = args.factors / "factors_input.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/04_factors.py first")
    blob = np.load(path, allow_pickle=False)
    meta = json.loads((args.factors / "factors_meta.json").read_text())
    seed = meta["seed"]
    op_layer = meta["op_layer"]
    readout_layers = meta["readout_layers"]
    # f2 is read where the lens works, not where we inject. G1 measured the
    # readout as ~25x better at 59 than at the injection layers.
    readout_primary = meta.get("readout_primary", op_layer)
    orders = meta["orders"]

    cell_concept = blob["cell_concept"].astype(str)
    cell_condition = blob["cell_condition"].astype(str)
    features = blob["cell_features"]
    trial_cell = blob["trial_cell"]
    trial_concept = blob["trial_concept"].astype(str)
    trial_order = blob["trial_order"].astype(str)
    trial_condition = blob["trial_condition"].astype(str)
    trial_ident = blob["trial_identifies"]
    trial_parse = blob["trial_parse_ok"]

    # ------------------------------------------------------------ f1 probe
    inj_cells = np.flatnonzero(cell_condition == "injected")
    ctl_cells = np.flatnonzero(cell_condition == "control_random")
    zero_cells = np.flatnonzero(cell_condition == "control_zero")

    probe = fac.mass_mean_probe(
        features[inj_cells], features[ctl_cells],
        cell_concept[inj_cells], cell_concept[ctl_cells],
        n_folds=args.probe_folds, seed=seed)
    threshold = fac.threshold_at_fpr(probe["scores_control"], args.fpr)
    probe_null = fac.mass_mean_probe(
        features[inj_cells], features[ctl_cells],
        cell_concept[inj_cells], cell_concept[ctl_cells],
        n_folds=args.probe_folds, seed=seed, shuffle_labels=True)
    threshold_null = fac.threshold_at_fpr(probe_null["scores_control"], args.fpr)

    cell_score = np.full(len(cell_concept), np.nan)
    cell_score[inj_cells] = probe["scores_injected"]
    cell_score[ctl_cells] = probe["scores_control"]
    cell_score_null = np.full(len(cell_concept), np.nan)
    cell_score_null[inj_cells] = probe_null["scores_injected"]

    # ------------------------------------------------- expand cells -> trials
    def cell_to_trial(values):
        return values[trial_cell]

    is_injected = trial_condition == "injected"
    is_random = trial_condition == "control_random"

    def arrays_for(mask, rank_key, score=cell_score, thr=None):
        thr = threshold if thr is None else thr
        f1 = fac.compute_f1(cell_to_trial(score)[mask], thr)
        ranks = cell_to_trial(blob[rank_key])[mask]
        f3 = trial_ident[mask]
        return f1, ranks, f3

    rank_key = f"cached_rank_L{readout_primary}"
    f1_pass, ranks, f3_pass = arrays_for(is_injected, rank_key)
    concepts_inj = trial_concept[is_injected]
    orders_inj = trial_order[is_injected]

    headline = fac.cascade(f1_pass, fac.compute_f2(ranks, args.k), f3_pass)

    # ------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G4 : three-factor decomposition ================")

    w("\nCONFIG")
    w(f"  injection layer          {op_layer}, alpha_rel {meta['strength']}")
    w(f"  f2 readout layer         {readout_primary}   <- not the injection layer")
    w(f"  probe layer              {meta['probe_layer']}")
    w(f"  readout layers swept     {readout_layers}")
    w(f"  orders                   {orders}")
    w(f"  cells / trials           {meta['n_cells']} / {meta['n_trials']}")
    w(f"  generations dropped      {meta['dropped_generations']} (no matching cell)")
    w(f"  probe folds              {probe['n_folds']} (grouped by concept)")
    w(f"  f1 threshold at FPR      {args.fpr}  ->  score {threshold:+.4f}")
    w(f"  headline k               {args.k}")
    w(f"  R-lens                   {meta['r_lens_used']}"
      f"{'' if meta['r_lens_used'] else '  (' + str(meta['r_lens_error']) + ')'}")

    w("\nINVARIANTS")
    w("  cascade_residual = |f1*f2*f3 - observed|. The chain rule makes the")
    w("  product exact when the denominators nest, so this is a bug detector:")
    w("  nonzero means a stage dropped trials or took a different denominator.")
    w(f"    {'k':>4}  {'order':<18}{'f1':>8}{'f2':>8}{'f3':>8}"
      f"{'observed':>10}{'residual':>12}")
    residual_rows = []
    for k in KS:
        for order in ["ALL", *orders]:
            mask = (np.ones_like(orders_inj, dtype=bool) if order == "ALL"
                    else orders_inj == order)
            row = fac.cascade(f1_pass[mask], fac.compute_f2(ranks[mask], k),
                              f3_pass[mask])
            residual_rows.append({"k": k, "order": order, **row})
            w(f"    {k:>4}  {order:<18}{row['f1']:>8.4f}{row['f2']:>8.4f}"
              f"{row['f3']:>8.4f}{row['observed_cascade_rate']:>10.4f}"
              f"{row['residual']:>12.3e}")
    w("")
    w("  survivorship at the headline k")
    w(f"    entering f1                      {headline['n_entering_f1']}")
    w(f"    surviving f1                     {headline['n_surviving_f1']}")
    w(f"    surviving f2                     {headline['n_surviving_f2']}")
    w(f"    surviving f3                     {headline['n_surviving_f3']}")
    if headline["n_surviving_f2"] < 30:
        w("    n at f3 is small, so every f3 number below is fragile; read the")
        w("    jackknife range and the judge-noise interval before quoting it.")

    w("\nPRIMARY")
    w(f"  cascade at k={args.k}, cluster-bootstrapped over concepts")
    for name in ("f1", "f2", "f3", "observed_cascade_rate"):
        def stat(idx, name=name):
            return fac.cascade(f1_pass[idx], fac.compute_f2(ranks[idx], args.k),
                               f3_pass[idx])[name]
        w(f"    {name:<22} {fmt(concept_bootstrap(concepts_inj, stat, args.n_boot, seed))}")
    naive = float(np.mean(f3_pass))
    w(f"    naive report rate      {naive:.4f}   (names the concept, ignoring")
    w("                             whether it passed f1 and f2)")
    w(f"    cascade vs naive gap   {abs(naive - headline['observed_cascade_rate']):.4f}")
    w("      the gap is trials that name the concept without clearing the probe")
    w("      or the lens readout -- measurement leakage in the operationalisation,")
    w("      not a property of the model.")
    w("")
    w("  by order")
    w(f"    {'order':<18}{'f1':>9}{'f2':>9}{'f3':>9}{'observed':>11}{'n':>7}")
    for order in orders:
        mask = orders_inj == order
        row = fac.cascade(f1_pass[mask], fac.compute_f2(ranks[mask], args.k),
                          f3_pass[mask])
        w(f"    {order:<18}{row['f1']:>9.4f}{row['f2']:>9.4f}{row['f3']:>9.4f}"
          f"{row['observed_cascade_rate']:>11.4f}{row['n_entering_f1']:>7}")
    if len(orders) == 2:
        a, b = orders
        ma, mb = orders_inj == a, orders_inj == b
        for name in ("f1", "f2", "f3"):
            ra = fac.cascade(f1_pass[ma], fac.compute_f2(ranks[ma], args.k), f3_pass[ma])
            rb = fac.cascade(f1_pass[mb], fac.compute_f2(ranks[mb], args.k), f3_pass[mb])
            w(f"    difference {name:<8} {a} - {b} = {ra[name] - rb[name]:+.4f}")
        w("    (G5 carries the CI on the difference; two separate intervals do")
        w("     not settle whether the difference straddles zero)")
    w("")
    w("  per-concept profiles (first 12 by f1, then the rest counted)")
    profiles = fac.per_concept_profiles(
        concepts_inj, f1_pass, fac.compute_f2(ranks, args.k), f3_pass)
    ordered = sorted(profiles.items(), key=lambda kv: -kv[1]["f1"])
    w(f"    {'concept':<18}{'f1':>7}{'f2':>7}{'f3':>7}{'n':>5}")
    for concept, row in ordered[:12]:
        w(f"    {concept:<18}{row['f1']:>7.3f}{row['f2']:>7.3f}"
          f"{row['f3']:>7.3f}{row['n_entering_f1']:>5}")
    n_all_zero = sum(1 for _, r in profiles.items() if r["n_surviving_f1"] == 0)
    w(f"    concepts with zero f1 survivors  {n_all_zero} of {len(profiles)}")

    w("\nCONTROLS")
    f1_null = fac.compute_f1(cell_to_trial(cell_score_null)[is_injected],
                             threshold_null)
    w(f"  probe null (labels shuffled)     f1 = {float(f1_null.mean()):.4f}")
    w(f"    threshold was set at FPR {args.fpr}, so a collapsed probe lands there")
    w(f"  probe separation, Cliff's delta  "
      f"{stats.cliffs_delta(probe['scores_injected'], probe['scores_control']):+.4f}")
    w(f"  probe null, Cliff's delta        "
      f"{stats.cliffs_delta(probe_null['scores_injected'], probe_null['scores_control']):+.4f}")
    w("")
    w("  f2 null band from matched-norm random vectors through the identical")
    w("  pipeline. Cosine has no absolute scale at this dimension, so an f2")
    w("  threshold has to be a quantile of THIS, not an absolute value.")
    rand_ranks = cell_to_trial(blob[rank_key])[is_random]
    w(f"    {'k':>4}{'f2 injected':>14}{'f2 random null':>16}{'lift':>9}")
    for k in KS:
        f2_i = float(np.mean(fac.compute_f2(ranks, k)))
        f2_n = float(np.mean(fac.compute_f2(rand_ranks, k)))
        w(f"    {k:>4}{f2_i:>14.4f}{f2_n:>16.4f}{f2_i - f2_n:>9.4f}")
    w(f"    rank median, injected / random   "
      f"{np.nanmedian(ranks):.0f} / {np.nanmedian(rand_ranks):.0f}")
    w("")
    w("  position control: f2 read at three positions. Only the report position")
    w("  supports the claim; the others say whether this is position-specific")
    w("  structure or a global shift.")
    w(f"    {'position':<12}{'f2':>9}{'median rank':>13}")
    for label in ("report", "injection", "random"):
        key = f"pos_rank_L{readout_primary}_{label}"
        if key not in blob:
            continue
        pos_ranks = cell_to_trial(blob[key])[is_injected]
        w(f"    {label:<12}{float(np.mean(fac.compute_f2(pos_ranks, args.k))):>9.4f}"
          f"{np.nanmedian(pos_ranks):>13.0f}")
    cached = ranks
    fresh_key = f"pos_rank_L{readout_primary}_report"
    if fresh_key in blob:
        fresh = cell_to_trial(blob[fresh_key])[is_injected]
        both = np.isfinite(cached) & np.isfinite(fresh)
        agree = float(np.mean(cached[both] == fresh[both])) if both.any() else float("nan")
        w(f"    cached vs recomputed report rank agreement  {agree:.4f}")
        w("      (the sweep cache and a fresh forward should give the same rank;")
        w("       disagreement means the sweep and this pass built different prompts)")

    w("\nCROSS-CHECK")
    w("  k-sensitivity: f2 moves monotonically with k by construction, so a")
    w("  scalar f2 is not reportable. Null band from the random-vector arm.")
    w(f"    {'k':>4}{'f1':>8}{'f2':>8}{'f3':>8}{'observed':>10}{'f2 null':>10}")
    for row in fac.k_curve(f1_pass, ranks, f3_pass, KS):
        f2_n = float(np.mean(fac.compute_f2(rand_ranks, row["k"])))
        w(f"    {row['k']:>4}{row['f1']:>8.4f}{row['f2']:>8.4f}{row['f3']:>8.4f}"
          f"{row['observed_cascade_rate']:>10.4f}{f2_n:>10.4f}")
    w("")
    w(f"  layer sensitivity of f2 at k={args.k} (readout layer varies, injection"
      f" stays at {op_layer})")
    w(f"    {'layer':>6}{'f2':>9}{'median rank':>13}{'null f2':>10}")
    for layer in readout_layers:
        key = f"cached_rank_L{layer}"
        if key not in blob:
            continue
        lr = cell_to_trial(blob[key])[is_injected]
        ln = cell_to_trial(blob[key])[is_random]
        w(f"    {layer:>6}{float(np.mean(fac.compute_f2(lr, args.k))):>9.4f}"
          f"{np.nanmedian(lr):>13.0f}"
          f"{float(np.mean(fac.compute_f2(ln, args.k))):>10.4f}")
    w("")
    w("  J-lens vs logit lens through the identical pipeline. G1 found the")
    w("  logit lens ahead at k=10 and k=25 on the eval sets, so neither is")
    w("  assumed correct here; both are carried through the cascade.")
    w(f"    {'layer':>6}{'k':>5}{'f2 J-lens':>11}{'f2 logit':>11}{'J - logit':>11}")
    for layer in readout_layers:
        jk, lk = f"cached_rank_L{layer}", f"logit_rank_L{layer}"
        if lk not in blob.files:
            continue
        jr = cell_to_trial(blob[jk])[is_injected]
        lr = cell_to_trial(blob[lk])[is_injected]
        for k in (1, args.k, 25):
            fj = float(np.mean(fac.compute_f2(jr, k)))
            fl = float(np.mean(fac.compute_f2(lr, k)))
            mark = "  <- primary" if (layer == readout_primary and k == args.k) else ""
            w(f"    {layer:>6}{k:>5}{fj:>11.4f}{fl:>11.4f}{fj - fl:>+11.4f}{mark}")
    lk_primary = f"logit_rank_L{readout_primary}"
    if lk_primary in blob.files:
        lr = cell_to_trial(blob[lk_primary])[is_injected]
        row_logit = fac.cascade(f1_pass, fac.compute_f2(lr, args.k), f3_pass)
        w(f"    full cascade under the logit lens: f1={row_logit['f1']:.4f} "
          f"f2={row_logit['f2']:.4f} f3={row_logit['f3']:.4f} "
          f"observed={row_logit['observed_cascade_rate']:.4f} "
          f"residual={row_logit['residual']:.3e}")

    w("")
    w("  jackknife, leave one concept out")
    f2_pass = fac.compute_f2(ranks, args.k)
    w(f"    {'factor':<8}{'full':>9}{'lo':>9}{'hi':>9}{'range':>9}  most influential")
    for which in ("f1", "f2", "f3"):
        j = fac.jackknife_factor(concepts_inj, f1_pass, f2_pass, f3_pass, which)
        w(f"    {which:<8}{j['full']:>9.4f}{j['lo']:>9.4f}{j['hi']:>9.4f}"
          f"{j['range']:>9.4f}  {j['most_influential']}")
    w("")
    w("  judge-noise propagation: f3 recomputed with every ambiguous trial")
    w("  flipped in each direction. Ambiguous = the response did not yield a")
    w("  parseable three-key JSON object, so identification rests on a string")
    w("  match over free text.")
    amb = ~trial_parse[is_injected]
    w(f"    ambiguous trials                 {int(amb.sum())} of {amb.size}")
    for label, forced in (("all ambiguous -> named", True),
                          ("all ambiguous -> not named", False)):
        flipped = f3_pass.copy()
        flipped[amb] = forced
        row = fac.cascade(f1_pass, f2_pass, flipped)
        w(f"    {label:<32} f3={row['f3']:.4f}  observed={row['observed_cascade_rate']:.4f}")
    w("      the width between those two rows bounds how much of the headline is")
    w("      scorer artifact rather than model behaviour")

    if meta["r_lens_used"]:
        w("")
        w(f"  R-lens comparison at k={args.k}, same pipeline")
        w(f"    {'layer':>6}{'J-lens f2':>11}{'R-lens f2':>11}")
        for layer in readout_layers:
            jk, rk = f"cached_rank_L{layer}", f"rlens_rank_L{layer}"
            if rk not in blob:
                continue
            jr = cell_to_trial(blob[jk])[is_injected]
            rr = cell_to_trial(blob[rk])[is_injected]
            w(f"    {layer:>6}{float(np.mean(fac.compute_f2(jr, args.k))):>11.4f}"
              f"{float(np.mean(fac.compute_f2(rr, args.k))):>11.4f}")

    w("\nANOMALIES")
    anomalies = []
    worst_residual = max((r["residual"] for r in residual_rows
                          if np.isfinite(r["residual"])), default=float("nan"))
    if np.isfinite(worst_residual) and worst_residual > 1e-12:
        anomalies.append(
            f"largest cascade residual is {worst_residual:.3e}, not floating-point "
            f"zero: a denominator does not nest")
    if headline["n_surviving_f2"] < 30:
        anomalies.append(
            f"only {headline['n_surviving_f2']} trials survive to f3")
    if meta["dropped_generations"]:
        anomalies.append(
            f"{meta['dropped_generations']} generations had no matching sweep cell")
    if zero_cells.size:
        anomalies.append(
            f"control_zero cells exist ({zero_cells.size}) but the probe uses "
            f"control_random: zero-strength carries no per-concept variation, so "
            f"it cannot supply a threshold with a meaningful FPR")
    anomalies.append(
        "f1/f2 are properties of one prefill residual and are shared across the "
        "samples drawn from it; only f3 varies within a cell. Trial counts are "
        "therefore not independent, which is why every interval here is a "
        "cluster bootstrap over concepts.")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  04_factors wall-clock              {meta['wall_s']:8.1f} s")
    w(f"  gate wall-clock                    {time.time() - t0:8.1f} s")

    w("\nARTIFACTS")
    out = {"headline": headline, "residual_rows": residual_rows,
           "threshold": threshold, "naive_report_rate": naive,
           "per_concept": profiles, "meta": meta}
    (args.out / "g4_factors.json").write_text(json.dumps(out, indent=1, default=float))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g4_report.txt").write_text(report)
    print(report)
    for name in ("g4_report.txt", "g4_factors.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
