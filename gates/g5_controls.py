"""GATE G5: controls, multiplicity, power, preregistration.

Collects every control arm the sprint ran into one table, then asks the four
questions that decide whether the headline survives contact with statistics:
is each cross-condition claim stated at a common FPR, how many comparisons
were made, is any claimed null actually powered, and which analyses were
specified in advance.

Runs on artifacts. No GPU. Tolerates missing gates -- it reports what has not
been run rather than failing.

Emits a report and stops. It judges nothing.

    python gates/g5_controls.py
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
    return f"{interval.point:.4f} [{interval.lo:.4f},{interval.hi:.4f}]"


def maybe(path: Path):
    return np.load(path, allow_pickle=False) if path.exists() else None


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided permutation p on the difference of means."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[: a.size].mean() - pool[a.size:].mean()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    A = args.artifacts
    g1 = maybe(A / "g1" / "g1_ranks.npz")
    g2 = maybe(A / "g2" / "g2.npz")
    g3 = maybe(A / "g3" / "g3.npz")
    fx = maybe(A / "factors" / "factors_input.npz")
    fx_meta = (json.loads((A / "factors" / "factors_meta.json").read_text())
               if (A / "factors" / "factors_meta.json").exists() else None)
    prereg = json.loads((ROOT / "configs" / "prereg.json").read_text())

    rows = []          # the one control table
    claims = []        # (name, p) for multiplicity
    nulls = []         # (name, baseline, n_per_arm) for power

    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G5 : controls, multiplicity, power ================")

    w("\nCONFIG")
    w(f"  artifacts root           {A}")
    for name, blob in (("G1", g1), ("G2", g2), ("G3", g3), ("factors", fx)):
        w(f"  {name:<24} {'present' if blob is not None else 'ABSENT -- gate not run'}")
    w(f"  headline k / FPR         {args.k} / {args.fpr}")

    # ------------------------------------------------------------- PRIMARY
    w("\nPRIMARY -- every control arm, one table")
    w(f"    {'gate':<5}{'arm':<38}{'rate / statistic':>26}{'n':>8}")

    if g1 is not None:
        n_words = g1["word_ranks_jacobian"].shape[0]
        best = {m: np.nanmin(g1[f"word_ranks_{m}"], axis=1)
                for m in ("jacobian", "logit", "null")}
        shuf = np.nanmin(g1["shuffled_word_ranks"], axis=1)
        for label, arr in (("J-lens pass@1", best["jacobian"]),
                           ("logit-lens pass@1", best["logit"]),
                           ("random-rotation null pass@1", best["null"]),
                           ("shuffled-position null pass@1", shuf)):
            iv = stats.wilson(int((arr <= 1).sum()), arr.size)
            rows.append(("G1", label, fmt(iv), arr.size))
            w(f"    {'G1':<5}{label:<38}{fmt(iv):>26}{arr.size:>8}")
        nulls.append(("G1 null lens vs J-lens", float((best['null'] <= 1).mean()),
                      n_words))
        claims.append(("G1 J-lens vs random-rotation null",
                       permutation_p(best["jacobian"], best["null"],
                                     args.n_perm, 0)))

    if g2 is not None:
        rho_c, rho_r = g2["rho_concept"], g2["rho_random"]
        for label, arr in (("dose-response rho, concepts", rho_c),
                           ("dose-response rho, random directions", rho_r)):
            d = stats.median_iqr(arr)
            rows.append(("G2", label, f"median {d['median']:+.4f} IQR {d['iqr']:.4f}",
                         arr.size))
            w(f"    {'G2':<5}{label:<38}"
              f"{'median ' + format(d['median'], '+.4f'):>26}{arr.size:>8}")
        conf = g2["confusion"]
        diag = int(sum(conf[i].argmax() == i for i in range(conf.shape[0])))
        iv = stats.wilson(diag, conf.shape[0])
        rows.append(("G2", "specificity: row max on diagonal", fmt(iv), conf.shape[0]))
        w(f"    {'G2':<5}{'specificity: row max on diagonal':<38}{fmt(iv):>26}"
          f"{conf.shape[0]:>8}")
        claims.append(("G2 concept vs random dose-response",
                       permutation_p(rho_c, rho_r, args.n_perm, 1)))
        nulls.append(("G2 random-direction dose-response is null",
                      float(np.mean(rho_r > 0)), rho_r.size))

    if g3 is not None:
        control = g3["control"]
        iv_fpr = stats.wilson(int((control > 0.5).sum()), control.size)
        rows.append(("G3", "zero-strength FPR", fmt(iv_fpr), control.size))
        w(f"    {'G3':<5}{'zero-strength FPR':<38}{fmt(iv_fpr):>26}{control.size:>8}")
        inj_keys = sorted(k for k in g3.files if k.startswith("injected_"))
        rnd_keys = sorted(k for k in g3.files if k.startswith("random_"))
        if inj_keys:
            pooled_i = np.concatenate([g3[k].ravel() for k in inj_keys])
            pooled_r = np.concatenate([g3[k].ravel() for k in rnd_keys])
            for label, arr in (("injected detection rate (pooled)", pooled_i),
                               ("random-direction rate (pooled)", pooled_r)):
                iv = stats.wilson(int((arr > 0.5).sum()), arr.size)
                rows.append(("G3", label, fmt(iv), arr.size))
                w(f"    {'G3':<5}{label:<38}{fmt(iv):>26}{arr.size:>8}")
            claims.append(("G3 injected vs random direction",
                           permutation_p(pooled_i, pooled_r, args.n_perm, 2)))
            nulls.append(("G3 random direction is not detected",
                          float((pooled_r > 0.5).mean()), pooled_r.size))

    # factors-derived arms
    cascade_ctx = None
    if fx is not None and fx_meta is not None:
        op_layer, seed = fx_meta["op_layer"], fx_meta["seed"]
        cell_condition = fx["cell_condition"].astype(str)
        cell_concept = fx["cell_concept"].astype(str)
        features = fx["cell_features"]
        trial_cell = fx["trial_cell"]
        trial_condition = fx["trial_condition"].astype(str)
        trial_concept = fx["trial_concept"].astype(str)
        trial_order = fx["trial_order"].astype(str)
        trial_ident = fx["trial_identifies"]

        inj_cells = np.flatnonzero(cell_condition == "injected")
        ctl_cells = np.flatnonzero(cell_condition == "control_random")
        probe = fac.mass_mean_probe(features[inj_cells], features[ctl_cells],
                                    cell_concept[inj_cells],
                                    cell_concept[ctl_cells], seed=seed)
        probe_null = fac.mass_mean_probe(features[inj_cells], features[ctl_cells],
                                         cell_concept[inj_cells],
                                         cell_concept[ctl_cells], seed=seed,
                                         shuffle_labels=True)
        cell_score = np.full(len(cell_concept), np.nan)
        cell_score[inj_cells] = probe["scores_injected"]
        cell_score[ctl_cells] = probe["scores_control"]
        is_inj = trial_condition == "injected"
        is_rand = trial_condition == "control_random"
        rank_key = f"cached_rank_L{op_layer}"
        ranks = fx[rank_key][trial_cell][is_inj]
        rand_ranks = fx[rank_key][trial_cell][is_rand]
        f3_pass = trial_ident[is_inj]
        concepts_inj = trial_concept[is_inj]
        orders_inj = trial_order[is_inj]

        thr = fac.threshold_at_fpr(probe["scores_control"], args.fpr)
        thr_null = fac.threshold_at_fpr(probe_null["scores_control"], args.fpr)
        cell_score_null = np.full(len(cell_concept), np.nan)
        cell_score_null[inj_cells] = probe_null["scores_injected"]
        f1_pass = fac.compute_f1(cell_score[trial_cell][is_inj], thr)
        f1_null = fac.compute_f1(cell_score_null[trial_cell][is_inj], thr_null)

        for label, arr in (("f1 (probe at 5% FPR)", f1_pass),
                           ("f1 probe null (labels shuffled)", f1_null)):
            iv = stats.wilson(int(arr.sum()), arr.size)
            rows.append(("G4", label, fmt(iv), arr.size))
            w(f"    {'G4':<5}{label:<38}{fmt(iv):>26}{arr.size:>8}")
        f2_obs = fac.compute_f2(ranks, args.k)
        f2_null = fac.compute_f2(rand_ranks, args.k)
        for label, arr in ((f"f2 at k={args.k}", f2_obs),
                           (f"f2 null band at k={args.k}", f2_null)):
            iv = stats.wilson(int(arr.sum()), arr.size)
            rows.append(("G4", label, fmt(iv), arr.size))
            w(f"    {'G4':<5}{label:<38}{fmt(iv):>26}{arr.size:>8}")
        for label in ("report", "injection", "random"):
            key = f"pos_rank_L{op_layer}_{label}"
            if key not in fx.files:
                continue
            arr = fac.compute_f2(fx[key][trial_cell][is_inj], args.k)
            iv = stats.wilson(int(arr.sum()), arr.size)
            rows.append(("G4", f"f2 at {label} position", fmt(iv), arr.size))
            w(f"    {'G4':<5}{('f2 at ' + label + ' position'):<38}{fmt(iv):>26}"
              f"{arr.size:>8}")
        claims.append(("G4 f2 vs random-vector null",
                       permutation_p(f2_obs.astype(float),
                                     f2_null.astype(float), args.n_perm, 3)))
        nulls.append(("G4 probe null collapses to chance",
                      float(f1_null.mean()), f1_null.size))
        cascade_ctx = (f1_pass, ranks, f3_pass, concepts_inj, orders_inj,
                       rand_ranks, seed)

    if not rows:
        w("    no artifacts found -- nothing to tabulate")

    # -------------------------------------------------------- CROSS-CHECK
    w("\nCROSS-CHECK")
    w("  FPR-matched comparison. Each arm below is restated at the SAME probe")
    w("  FPR, so the comparison is not each condition at its own best threshold.")
    if cascade_ctx is not None:
        f1_pass, ranks, f3_pass, concepts_inj, orders_inj, rand_ranks, seed = cascade_ctx
        w(f"    {'FPR':>8}{'threshold':>12}{'f1':>9}{'f2|f1':>9}{'f3|f2':>9}"
          f"{'observed':>11}")
        for fpr in (0.01, 0.05, 0.10, 0.20):
            t = fac.threshold_at_fpr(probe["scores_control"], fpr)
            f1_at = fac.compute_f1(cell_score[trial_cell][is_inj], t)
            row = fac.cascade(f1_at, fac.compute_f2(ranks, args.k), f3_pass)
            w(f"    {fpr:>8.2f}{t:>12.4f}{row['f1']:>9.4f}{row['f2']:>9.4f}"
              f"{row['f3']:>9.4f}{row['observed_cascade_rate']:>11.4f}")
        w("    f1 rises with the FPR allowance by construction; what matters is")
        w("    whether f2 and f3 move with it. If they do, the cascade is partly")
        w("    a restatement of where the threshold was put.")
    else:
        w("    factors artifacts absent -- cannot restate at a common FPR")

    w("")
    w("  order contrast, CI on the DIFFERENCE")
    if cascade_ctx is not None and len(set(orders_inj)) == 2:
        a, b = sorted(set(orders_inj))
        for name in ("f1", "f2", "f3"):
            def diff(idx, name=name, a=a, b=b):
                oa, ob = orders_inj[idx] == a, orders_inj[idx] == b
                ra = fac.cascade(f1_pass[idx][oa],
                                 fac.compute_f2(ranks[idx][oa], args.k),
                                 f3_pass[idx][oa])[name]
                rb = fac.cascade(f1_pass[idx][ob],
                                 fac.compute_f2(ranks[idx][ob], args.k),
                                 f3_pass[idx][ob])[name]
                return ra - rb
            point, lo, hi = fac.cluster_bootstrap(concepts_inj, diff,
                                                  args.n_boot, seed)
            w(f"    {name}  {a} - {b} = {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        w("    two overlapping marginal intervals would not settle this; the")
        w("    interval above is on the difference itself.")
    else:
        w("    not available")

    w("")
    w("  multiple comparisons")
    grid = 0
    if fx_meta:
        cfg_layers = len(fx_meta["readout_layers"])
        grid = cfg_layers * len(KS) * (len(fx_meta["orders"]) + 1)
    w(f"    comparisons implied by the condition grid   {grid}")
    w(f"      (readout layers x k values x orders including the pooled arm)")
    w(f"    headline claims tested                      {len(claims)}")
    if claims:
        ps = np.array([p for _, p in claims], dtype=float)
        qs = stats.benjamini_hochberg(ps)
        w(f"    {'claim':<44}{'p':>10}{'BH q':>10}")
        for (name, p), q in zip(claims, qs):
            w(f"    {name:<44}{p:>10.4f}{q:>10.4f}")
        w("    q-values adjust the headline claims among themselves. They do NOT")
        w("    absolve the grid above: every cell inspected while choosing an")
        w("    operating point is a comparison too, and those are not in this set.")
    else:
        w("    no claims computable from the artifacts present")

    w("")
    w("  power: minimum detectable effect at 80% power, given realised n.")
    w("  A null without this is not a null.")
    if nulls:
        w(f"    {'claimed null':<44}{'baseline':>10}{'n/arm':>9}{'MDE':>9}")
        for name, baseline, n in nulls:
            mde = stats.min_detectable_effect(float(np.clip(baseline, 1e-6, 1 - 1e-6)),
                                              int(n))
            w(f"    {name:<44}{baseline:>10.4f}{n:>9}{mde:>9.4f}")
        w("    read as: an effect smaller than MDE would not have been detected,")
        w("    so 'no difference' means 'no difference larger than MDE'.")
    else:
        w("    no nulls claimed from the artifacts present")

    w("")
    w("  preregistration adherence")
    spec, expl = prereg["specified"], prereg["exploratory"]
    w(f"    specified in advance                        {len(spec)}")
    w(f"    exploratory (added during the build)        {len(expl)}")
    not_done = [s for s in spec if s.get("status")]
    w(f"    specified but NOT delivered                 {len(not_done)}")
    for item in not_done:
        w(f"      - {item['id']}: {item['status']}")
    w("    exploratory additions:")
    for item in expl:
        w(f"      - {item['id']}: {item['what'][:88]}")

    w("\nANOMALIES")
    anomalies = []
    missing = [n for n, b in (("G1", g1), ("G2", g2), ("G3", g3), ("factors", fx))
               if b is None]
    if missing:
        anomalies.append(f"artifacts absent for {missing}; those arms are blank")
    if not_done:
        anomalies.append(
            f"{len(not_done)} preregistered analyses were not delivered: "
            f"{[i['id'] for i in not_done]}")
    anomalies.append(
        "the multiplicity count covers the condition grid this gate can see. "
        "Choices made while building -- operating point, injection window, "
        "task prompt -- are researcher degrees of freedom that no q-value here "
        "corrects for.")
    anomalies.append(
        "every interval on a factor is a cluster bootstrap over concepts; "
        "Wilson intervals in the PRIMARY table treat trials as independent and "
        "are therefore narrower than they should be wherever trials share a "
        "prefill residual.")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  gate wall-clock                    {time.time() - t0:8.1f} s")

    w("\nARTIFACTS")
    (args.out / "g5_controls.json").write_text(json.dumps(
        {"rows": rows, "claims": claims, "nulls": nulls,
         "prereg_specified": len(spec), "prereg_exploratory": len(expl),
         "prereg_not_delivered": [i["id"] for i in not_done]},
        indent=1, default=float))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g5_report.txt").write_text(report)
    print(report)
    for name in ("g5_report.txt", "g5_controls.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
