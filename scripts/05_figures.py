"""Render the three figures from artifacts. No GPU, no model, no network.

    python scripts/05_figures.py [--k 10]

Recomputes the cascade from factors_input.npz rather than reading the G4
report, so the figures stand alone: a judge can regenerate them without having
run any gate, which is what the Colab does.

Also writes RESULTS.md -- the numbers block the README quotes, generated rather
than typed, so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import factors as fac  # noqa: E402
import plots  # noqa: E402
import stats  # noqa: E402

KS = [1, 5, 10, 25, 50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="concept",
                    help="vector arm; selects artifacts/factors/<arm>")
    ap.add_argument("--factors", type=Path, default=None,
                    help="default artifacts/factors/<arm>")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--probe-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None,
                    help="default artifacts/figures/<arm>")
    args = ap.parse_args()
    if args.factors is None:
        args.factors = ROOT / "artifacts" / "factors" / args.arm
    if args.out is None:
        args.out = ROOT / "artifacts" / "figures" / args.arm
    args.out.mkdir(parents=True, exist_ok=True)

    path = args.factors / "factors_input.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/04_factors.py first")
    blob = np.load(path, allow_pickle=False)
    meta = json.loads((args.factors / "factors_meta.json").read_text())
    seed, op_layer, orders = meta["seed"], meta["op_layer"], meta["orders"]
    readout_primary = meta.get("readout_primary", op_layer)
    band = meta.get("band_layers") or []
    arm = meta.get("vector_arm", "concept")
    # figures name the band, not a layer, once the policy is workspace_band
    site = f"band L{band[0]}-{band[-1]}" if len(band) > 1 else f"L{op_layer}"

    cell_concept = blob["cell_concept"].astype(str)
    cell_condition = blob["cell_condition"].astype(str)
    features = blob["cell_features"]
    trial_cell = blob["trial_cell"]
    trial_concept = blob["trial_concept"].astype(str)
    trial_order = blob["trial_order"].astype(str)
    trial_condition = blob["trial_condition"].astype(str)
    trial_ident = blob["trial_identifies"]

    inj_cells = np.flatnonzero(cell_condition == "injected")
    ctl_cells = np.flatnonzero(cell_condition == "control_random")
    probe = fac.mass_mean_probe(
        features[inj_cells], features[ctl_cells],
        cell_concept[inj_cells], cell_concept[ctl_cells],
        n_folds=args.probe_folds, seed=seed)
    threshold = fac.threshold_at_fpr(probe["scores_control"], args.fpr)
    cell_score = np.full(len(cell_concept), np.nan)
    cell_score[inj_cells] = probe["scores_injected"]
    cell_score[ctl_cells] = probe["scores_control"]

    # Per-order threshold, matching G4. A pooled quantile put the task-first
    # controls at 0.000 and the report-first controls at 0.100 against a
    # nominal 0.05, which manufactured most of the f1 order effect.
    cell_order = blob["cell_order"].astype(str)
    threshold_by_order = {
        order: fac.threshold_at_fpr(
            cell_score[ctl_cells[cell_order[ctl_cells] == order]], args.fpr)
        for order in orders
    }

    is_inj = trial_condition == "injected"
    is_rand = trial_condition == "control_random"
    # f2 at the pre-registered primary slot when the artifact carries it
    slot_keys = {
        "naming_output": "naming_output_rank",
        "naming_jlens": f"naming_rank_L{readout_primary}",
        "boolean": f"cached_rank_L{readout_primary}",
    }
    slot = next((s for s in (args.f2_slot, "naming_output", "boolean")
                 if slot_keys.get(s) in blob), "boolean")
    rank_key = slot_keys[slot]
    slot_label = {
        "naming_output": "f₂ at the naming slot, model output distribution",
        "naming_jlens": f"f₂ at the naming slot, J-lens L{readout_primary}",
        "boolean": f"f₂ at the boolean slot, J-lens L{readout_primary}",
    }[slot]
    thr_per_trial = np.array([threshold_by_order.get(o, threshold)
                              for o in trial_order[is_inj]])
    scores_inj = cell_score[trial_cell][is_inj]
    f1_pass = np.isfinite(scores_inj) & (scores_inj > thr_per_trial)
    ranks = blob[rank_key][trial_cell][is_inj]
    rand_ranks = blob[rank_key][trial_cell][is_rand]
    f3_pass = trial_ident[is_inj]
    concepts_inj = trial_concept[is_inj]
    concepts_rand = trial_concept[is_rand]
    orders_inj = trial_order[is_inj]

    headline = fac.cascade(f1_pass, fac.compute_f2(ranks, args.k), f3_pass)

    # ---------------------------------------------------------- figure 1
    fig1 = plots.cascade_bar(headline["f1"], headline["f2"], headline["f3"],
                             headline["n_entering_f1"],
                             title=f"Where the injected concept is lost "
                                   f"(inject {site}, arm {arm}, read L{readout_primary}, k={args.k})")
    written = plots.save(fig1, args.out / "fig1_cascade")

    # ---------------------------------------------------------- figure 2
    per_order, differences = {}, {}
    for order in orders:
        mask = orders_inj == order
        per_order[order] = fac.cascade(
            f1_pass[mask], fac.compute_f2(ranks[mask], args.k), f3_pass[mask])
    if len(orders) == 2:
        a, b = orders
        for name in ("f1", "f2", "f3"):
            def diff(idx, name=name, a=a, b=b):
                oa = orders_inj[idx] == a
                ob = orders_inj[idx] == b
                ra = fac.cascade(f1_pass[idx][oa],
                                 fac.compute_f2(ranks[idx][oa], args.k),
                                 f3_pass[idx][oa])[name]
                rb = fac.cascade(f1_pass[idx][ob],
                                 fac.compute_f2(ranks[idx][ob], args.k),
                                 f3_pass[idx][ob])[name]
                return ra - rb
            differences[name] = fac.cluster_bootstrap(
                concepts_inj, diff, args.n_boot, seed)
    else:
        differences = {n: (float("nan"),) * 3 for n in ("f1", "f2", "f3")}
    fig2 = plots.order_contrast(per_order, differences,
                                title=f"Order contrast (inject {site}, arm {arm}, read L{readout_primary}, k={args.k})")
    written += plots.save(fig2, args.out / "fig2_order_contrast")

    # ---------------------------------------------------------- figure 3
    f2_obs, null_lo, null_hi, null_med = [], [], [], []
    for k in KS:
        f2_obs.append(float(np.mean(fac.compute_f2(ranks, k))))
        point, lo, hi = fac.cluster_bootstrap(
            concepts_rand,
            lambda idx, k=k: float(np.mean(fac.compute_f2(rand_ranks[idx], k))),
            args.n_boot, seed)
        null_med.append(point)
        null_lo.append(lo)
        null_hi.append(hi)
    fig3 = plots.f2_sensitivity(KS, f2_obs, null_lo, null_hi, null_med,
                                title=f"f₂ sensitivity to k (read L{readout_primary})")
    written += plots.save(fig3, args.out / "fig3_f2_sensitivity")

    # ------------------------------------------------------------ numbers
    def boot(name):
        point, lo, hi = fac.cluster_bootstrap(
            concepts_inj,
            lambda idx, name=name: fac.cascade(
                f1_pass[idx], fac.compute_f2(ranks[idx], args.k),
                f3_pass[idx])[name],
            args.n_boot, seed)
        return point, lo, hi

    triple = {name: boot(name) for name in ("f1", "f2", "f3")}
    results = {
        "operating_point": {"layer": op_layer, "band_layers": band,
                            "vector_arm": arm, "readout_layer": readout_primary,
                            "strength": meta["strength"],
                            "k": args.k, "fpr": args.fpr},
        "cascade": headline,
        "cascade_ci": triple,
        "per_order": per_order,
        "order_differences": differences,
        "f2_curve": {"k": KS, "observed": f2_obs, "null_median": null_med,
                     "null_lo": null_lo, "null_hi": null_hi},
        "naive_report_rate": float(np.mean(f3_pass)),
        "n_trials": int(f1_pass.size),
        "n_concepts": int(np.unique(concepts_inj).size),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    lines = [
        "# Results",
        "",
        "Generated by `scripts/05_figures.py`. Do not edit by hand -- the README",
        "quotes these numbers, and typing them twice is how they drift apart.",
        "",
        f"Operating point: inject {site} (arm {arm}), read L{readout_primary}, "
        f"alpha_rel {meta['strength']}, "
        f"k={args.k}, probe FPR {args.fpr}.",
        f"n = {results['n_trials']} injection trials over "
        f"{results['n_concepts']} concepts.",
        "",
        "## The cascade",
        "",
        "| factor | meaning | value | 95% CI (cluster bootstrap over concepts) |",
        "|---|---|---|---|",
        f"| f₁ | P(represented) | {triple['f1'][0]:.3f} | "
        f"[{triple['f1'][1]:.3f}, {triple['f1'][2]:.3f}] |",
        f"| f₂ | P(verbalizable \\| represented) | {triple['f2'][0]:.3f} | "
        f"[{triple['f2'][1]:.3f}, {triple['f2'][2]:.3f}] |",
        f"| f₃ | P(reported \\| verbalizable) | {triple['f3'][0]:.3f} | "
        f"[{triple['f3'][1]:.3f}, {triple['f3'][2]:.3f}] |",
        "",
        f"Product `f₁·f₂·f₃` = **{headline['f1'] * headline['f2'] * headline['f3']:.4f}**, "
        f"observed end-to-end rate = **{headline['observed_cascade_rate']:.4f}**, "
        f"`cascade_residual` = `{headline['residual']:.3e}`.",
        "",
        f"Survivorship: {headline['n_entering_f1']} → {headline['n_surviving_f1']} "
        f"→ {headline['n_surviving_f2']} → {headline['n_surviving_f3']}.",
        "",
        "## Order contrast",
        "",
        "| factor | " + " | ".join(orders) + " | difference (95% CI) |",
        "|---|" + "---|" * (len(orders) + 1),
    ]
    for name in ("f1", "f2", "f3"):
        cells = " | ".join(f"{per_order[o][name]:.3f}" for o in orders)
        d = differences[name]
        lines.append(f"| {name} | {cells} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")
    lines += [
        "",
        "## f₂ over k, against the matched-norm random-vector null",
        "",
        "| k | f₂ observed | null median | null 95% band |",
        "|---|---|---|---|",
    ]
    for i, k in enumerate(KS):
        lines.append(f"| {k} | {f2_obs[i]:.3f} | {null_med[i]:.3f} | "
                     f"[{null_lo[i]:.3f}, {null_hi[i]:.3f}] |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    for target in written:
        print(f"  wrote {target}")
    print(f"  wrote {args.out / 'results.json'}")
    print(f"  wrote {ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
