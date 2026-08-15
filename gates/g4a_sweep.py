"""GATE G4a: sweep integrity.

Runs on the artifacts, not the model, so it needs no GPU and can be re-run
after a disconnect to audit whatever landed.

This gate exists to catch the failures that leave no error message: a missing
cell, a shard written twice, a NaN that propagated, or something changing
mid-run -- a reloaded model, a different dtype, a hardware switch.

Emits a report and stops. It judges nothing.

    python gates/g4a_sweep.py [--sweep artifacts/sweep] [--gen artifacts/generations]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stats  # noqa: E402

RULE = "=" * 78


def fmt(interval: stats.Interval) -> str:
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=ROOT / "artifacts" / "sweep")
    ap.add_argument("--gen", type=Path, default=ROOT / "artifacts" / "generations")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g4a")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest_path = args.sweep / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run scripts/02_sweep.py first")
    manifest = json.loads(manifest_path.read_text())
    concepts = manifest["concepts"]
    expected_cells = manifest["cells_per_concept"]

    shards = sorted(args.sweep.glob("shard_*.npz"))
    strays = sorted(args.sweep.glob("*.tmp.npz"))

    # ------------------------------------------------------------- load
    loaded, unreadable, digests = {}, [], {}
    for path in shards:
        try:
            blob = np.load(path, allow_pickle=False)
            name = str(blob["concept"])
            loaded[name] = {k: blob[k] for k in blob.files}
            digests.setdefault(
                hashlib.sha256(blob["residuals"].tobytes()).hexdigest(), []
            ).append(name)
        except Exception as exc:  # noqa: BLE001
            unreadable.append((path.name, f"{type(exc).__name__}: {exc}"))

    present = set(loaded)
    missing = [c for c in concepts if c not in present]
    unexpected = sorted(present - set(concepts))
    duplicate_payloads = {d: names for d, names in digests.items() if len(names) > 1}

    # ------------------------------------------------- invariants over cells
    bad_cell_count, nan_count, inf_count = [], 0, 0
    order_of = condition_of = arm_of = strength_of = None
    for name, shard in loaded.items():
        if shard["p_true"].shape[0] != expected_cells:
            bad_cell_count.append((name, int(shard["p_true"].shape[0])))
        for key in ("residuals", "p_true", "p_false", "top_logits"):
            arr = shard[key]
            nan_count += int(np.isnan(arr).sum())
            inf_count += int(np.isinf(arr).sum())
        if order_of is None:
            order_of = shard["cell_order"]
            condition_of = shard["cell_condition"]
            arm_of = shard["cell_arm"]
            strength_of = shard["cell_strength"]

    # cell-grid consistency across shards
    grid_mismatch = []
    for name, shard in loaded.items():
        if shard["p_true"].shape[0] != expected_cells:
            continue
        if not np.array_equal(shard["cell_order"], order_of) or \
           not np.array_equal(shard["cell_condition"], condition_of) or \
           not np.array_equal(shard["cell_arm"], arm_of) or \
           not np.array_equal(shard["cell_strength"], strength_of):
            grid_mismatch.append(name)

    # ---------------------------------------------------------- drift probe
    order_arr = np.array([str(x) for x in order_of]) if order_of is not None else np.array([])
    cond_arr = np.array([str(x) for x in condition_of]) if condition_of is not None else np.array([])
    arm_arr = np.array([str(x) for x in arm_of]) if arm_of is not None else np.array([])
    zero_mask = cond_arr == "control_zero"
    written = np.array([float(loaded[c]["written_at"]) for c in loaded])
    names_by_time = [c for _, c in sorted(zip(written, loaded))]
    decile = max(1, len(names_by_time) // 10)
    first_names, last_names = names_by_time[:decile], names_by_time[-decile:]

    def zero_stat(names):
        vals = []
        for n in names:
            shard = loaded[n]
            if shard["p_true"].shape[0] != expected_cells:
                continue
            vals.append(shard["p_true"][zero_mask])
        return np.concatenate(vals) if vals else np.array([])

    first_zero, last_zero = zero_stat(first_names), zero_stat(last_names)

    # per-shard timing
    timings = manifest.get("timings", [])
    per_concept_s = np.array(
        [t["seconds"] / max(1, len(t["concepts"])) for t in timings], dtype=float)

    # -------------------------------------------------------- generations
    gen_files = sorted(args.gen.glob("gen_*.json")) if args.gen.exists() else []
    gen_records = []
    for path in gen_files:
        gen_records.extend(json.loads(path.read_text()))
    gen_meta = json.loads((args.gen / "meta.json").read_text()) \
        if (args.gen / "meta.json").exists() else {}

    # -------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G4a : sweep integrity ================")

    band = manifest.get("band_layers", [])
    arms = manifest.get("arms", [])
    tasks = manifest.get("tasks", [])
    task_of = manifest.get("task_of", {})

    w("\nCONFIG")
    w(f"  model revision           {manifest['model_revision'][:12]}")
    w(f"  injection policy         {manifest.get('injection_policy')}"
      f"  norm_mode {manifest.get('norm_mode')}"
      f"  median scope {manifest.get('median_scope')}")
    w(f"  band                     {len(band)} layers "
      f"{band[0] if band else '-'}..{band[-1] if band else '-'}  {band}")
    w(f"  strengths (per layer)    {manifest['strengths']}")
    w(f"  arms                     {arms}")
    w(f"  orders                   {manifest['orders']}")
    w(f"  conditions               {manifest['conditions']}")
    w(f"  cells per concept        {expected_cells}")
    w(f"  nominal grid             {manifest['nominal_grid']} cells")
    w(f"  batch / seed             {manifest['batch']} / {manifest['seed']}")
    w(f"  tasks                    {len(tasks)}, round-robin over concepts")
    for task in tasks:
        members = [c for c, t in task_of.items() if t == task]
        answer = manifest.get("clean_task_answers", {}).get(task)
        w(f"    {task!r:<52} n={len(members):<3} answer={answer!r}")
    for key, meta in manifest["prompt_meta"].items():
        w(f"  {key:<62} seq={meta['seq_len']} "
          f"span={meta['position_span']} n_inj={meta['n_injected_positions']}")
    w("  the report position is the token after a prefill that ends at the")
    w("  detection key, so both orders read the same quantity; without that")
    w("  the order contrast would compare a detection verdict against a task")
    w("  answer.")

    w("\nINVARIANTS")
    w(f"  shards found                       {len(shards)}")
    w(f"  shards readable                    {len(loaded)}")
    w(f"  concepts expected                  {len(concepts)}")
    w(f"  concepts missing                   {len(missing)}  {missing[:8]}")
    w(f"  shards not in the manifest         {len(unexpected)}  {unexpected[:8]}")
    w(f"  half-written .tmp files            {len(strays)}  {[p.name for p in strays][:4]}")
    w(f"  unreadable shards                  {len(unreadable)}")
    for name, err in unreadable[:5]:
        w(f"    {name}: {err}")
    w(f"  shards with wrong cell count       {len(bad_cell_count)}  {bad_cell_count[:5]}")
    w(f"  shards whose cell grid differs     {len(grid_mismatch)}  {grid_mismatch[:5]}")
    w(f"  duplicate residual payloads        {len(duplicate_payloads)}")
    for digest, names in list(duplicate_payloads.items())[:4]:
        w(f"    {digest[:12]}  {names}")
    w("    (two concepts with byte-identical residuals across every cell means")
    w("     a shard was written under the wrong name, or the same vector was")
    w("     injected twice)")
    w(f"  NaN entries across cached tensors  {nan_count}")
    w(f"  Inf entries across cached tensors  {inf_count}")
    cells_done = sum(1 for s in loaded.values()
                     if s["p_true"].shape[0] == expected_cells) * expected_cells
    w(f"  cells complete                     {cells_done} of "
      f"{manifest['nominal_grid']}   "
      f"{fmt(stats.wilson(cells_done, manifest['nominal_grid']))}")

    w("\n  missing-cell map, by (strength, order, condition, arm)")
    if missing:
        w(f"    every cell of {len(missing)} concept(s) absent: {missing[:12]}")
    if order_of is not None:
        counts = Counter(
            (float(s), str(o), str(c), str(a))
            for s, o, c, a in zip(strength_of, order_arr, cond_arr, arm_arr))
        # injected cells carry an arm; the two controls are shared
        expected_distinct = (len(manifest["strengths"]) * len(manifest["orders"])
                             * (len(arms) + len(manifest["conditions"]) - 1))
        w(f"    distinct cell coordinates          {len(counts)}")
        w(f"    coordinates appearing more than once "
          f"{sum(1 for v in counts.values() if v > 1)}")
        w(f"    expected distinct                  {expected_distinct}")

    w("\n  band integrity: did the injection fire where the cell says it did")
    if loaded and band:
        complete = [s for s in loaded.values()
                    if s["p_true"].shape[0] == expected_cells
                    and "cell_fires" in s]
        if complete:
            fires = np.stack([s["cell_fires"] for s in complete])
            ratio = np.stack([s["cell_delta_ratio"] for s in complete])
            live = ~zero_mask
            w(f"    fires per injected cell            "
              f"min {int(fires[:, live].min())}  max {int(fires[:, live].max())}"
              f"   (expect {len(band)}: one per band layer, prefill only)")
            w(f"    fires per zero cell                "
              f"max {int(fires[:, zero_mask].max()) if zero_mask.any() else 0}"
              f"   (expect 0: no hook is registered at all)")
            w("    realised ||delta_l|| / base_l vs the strength the cell claims.")
            w("    The hook's contract fixes this at alpha_rel exactly, so a")
            w("    departure localises a mis-scaled cell rather than hinting at one.")
            w(f"    {'alpha_rel':>10}  {'median realised':>16}  {'max |error|':>12}"
              f"  {'cells':>7}")
            for strength in sorted(set(float(s) for s in strength_of)):
                sel = live & np.isclose(strength_of.astype(float), strength)
                if not sel.any():
                    continue
                vals = ratio[:, sel]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    w(f"    {strength:>10.3f}  {np.median(vals):>16.6f}"
                      f"  {np.abs(vals - strength).max():>12.3e}  {vals.size:>7}")
        else:
            w("    shards predate the fire/displacement recording")

    w("\nCROSS-CHECK")
    w("  temporal drift: the zero-strength control at the report position,")
    w("  first vs last decile of shards by write time. This cell is recomputed")
    w("  once per batch precisely so drift is visible; a systematic shift means")
    w("  something changed mid-run.")
    if first_zero.size and last_zero.size:
        fd, ld = stats.median_iqr(first_zero), stats.median_iqr(last_zero)
        w(f"    first decile ({len(first_names)} shards)  median {fd['median']:.6f}"
          f"  IQR {fd['iqr']:.6f}")
        w(f"    last  decile ({len(last_names)} shards)  median {ld['median']:.6f}"
          f"  IQR {ld['iqr']:.6f}")
        w(f"    Cliff's delta first vs last      "
          f"{stats.cliffs_delta(first_zero, last_zero):+.4f}")
        ks = stats.ks_two_sample(first_zero, last_zero)
        w(f"    KS two-sample                    D={ks['D']:.4f}  p={ks['p']:.3e}")
        w(f"    |median difference|              "
          f"{abs(fd['median'] - ld['median']):.3e}")
    else:
        w("    not enough shards to compare deciles")

    w("  per-concept wall-clock distribution. Outliers usually mean silent")
    w("  recomputation or an OOM retry, not a slow concept.")
    if per_concept_s.size:
        td = stats.median_iqr(per_concept_s)
        outl = stats.mad_outliers(per_concept_s, threshold=3.0)
        w(f"    median {td['median']:.3f}s  IQR {td['iqr']:.3f}s"
          f"  min {td['min']:.3f}s  max {td['max']:.3f}s")
        w(f"    batches beyond 3 MAD             {outl['n_outliers']} of "
          f"{per_concept_s.size}")
    else:
        w("    no timing records (resumed run with nothing new to do)")

    w("  detection channel sanity, by condition x arm (pooled over shards)")
    if order_of is not None and loaded:
        complete_shards = [s for s in loaded.values()
                           if s["p_true"].shape[0] == expected_cells]
        w(f"    {'condition':<16}{'arm':<11}{'order':<18}"
          f"{'median P(true)':>15}{'IQR':>10}{'n':>8}")
        for condition in manifest["conditions"]:
            cell_arms = arms if condition == "injected" else ["shared"]
            for cell_arm in cell_arms:
                for order in manifest["orders"]:
                    mask = ((cond_arr == condition) & (order_arr == order)
                            & (arm_arr == cell_arm))
                    vals = np.concatenate([s["p_true"][mask] for s in complete_shards]) \
                        if mask.any() and complete_shards else np.array([])
                    if vals.size:
                        d = stats.median_iqr(vals)
                        w(f"    {condition:<16}{cell_arm:<11}{order:<18}"
                          f"{d['median']:>15.4f}{d['iqr']:>10.4f}{vals.size:>8}")

        w("")
        w("  zero-strength detection BY TASK. This is the variance the FPR")
        w("  interval rests on. Under one fixed task every zero cell was the")
        w("  same forward pass and this table would have one row and no spread;")
        w("  a flat column here means the 10-way assignment is not reaching the")
        w("  control channel.")
        w(f"    {'task':<52}{'median P(true)':>15}{'n':>6}")
        by_task = {}
        for name, shard in loaded.items():
            if shard["p_true"].shape[0] != expected_cells:
                continue
            by_task.setdefault(str(shard["task"]) if "task" in shard else "",
                               []).append(shard["p_true"][zero_mask])
        for task in sorted(by_task):
            vals = np.concatenate(by_task[task])
            w(f"    {task[:50]:<52}{np.median(vals):>15.4f}{vals.size:>6}")
        if len(by_task) > 1:
            task_medians = np.array(
                [float(np.median(np.concatenate(v))) for v in by_task.values()])
            w(f"    spread across tasks   min {task_medians.min():.4f}  "
              f"max {task_medians.max():.4f}  "
              f"range {np.ptp(task_medians):.4f}")
        w("    p_true + p_false should be near 1 if the next token really is a")
        w("    JSON boolean; a small sum means the prefill is not landing where")
        w("    it was meant to.")
        sums = np.concatenate([
            s["p_true"] + s["p_false"] for s in loaded.values()
            if s["p_true"].shape[0] == expected_cells]) if loaded else np.array([])
        if sums.size:
            sd = stats.median_iqr(sums)
            w(f"    P(true)+P(false)  median {sd['median']:.4f}  "
              f"min {sd['min']:.4f}  IQR {sd['iqr']:.4f}")
            w(f"    cells below 0.5                  {int((sums < 0.5).sum())} of {sums.size}")

    w("\n  generation pass")
    if gen_records:
        w(f"    records                          {len(gen_records)}")
        w(f"    files                            {len(gen_files)}")
        w(f"    operating point                  layer {gen_meta.get('layer')}, "
          f"alpha_rel {gen_meta.get('strength')}, T={gen_meta.get('temperature')}, "
          f"{gen_meta.get('samples')} samples")
        parse_ok = sum(1 for r in gen_records if r["parse_ok"])
        w(f"    three-key JSON parse rate        "
          f"{fmt(stats.wilson(parse_ok, len(gen_records)))}")
        for order in manifest["orders"]:
            rows = [r for r in gen_records if r["order"] == order]
            ok = sum(1 for r in rows if r["parse_ok"])
            if rows:
                w(f"      {order:<18} {fmt(stats.wilson(ok, len(rows)))}")
        lens_ = np.array([r["new_tokens"] for r in gen_records], dtype=float)
        ld = stats.median_iqr(lens_)
        w(f"    new tokens  median {ld['median']:.0f}  IQR {ld['iqr']:.0f}  "
          f"max {ld['max']:.0f}")
        w(f"    truncated at the cap             "
          f"{int((lens_ >= gen_meta.get('max_new_tokens', 96)).sum())} of {lens_.size}")
    else:
        w("    no generations found -- run scripts/03_generate.py")

    w("\nANOMALIES")
    anomalies = []
    if missing:
        anomalies.append(f"{len(missing)} concepts have no shard: {missing[:10]}")
    if strays:
        anomalies.append(f"{len(strays)} .tmp files: a run was killed mid-write")
    if unreadable:
        anomalies.append(f"{len(unreadable)} shards failed to load")
    if duplicate_payloads:
        anomalies.append(
            f"{len(duplicate_payloads)} residual payloads are byte-identical "
            f"across different concepts")
    if nan_count or inf_count:
        anomalies.append(f"non-finite values: {nan_count} NaN, {inf_count} Inf")
    if grid_mismatch:
        anomalies.append(f"{len(grid_mismatch)} shards disagree on the cell grid")
    anomalies.append(
        "control_zero cells are computed once per (batch, order) and broadcast "
        "across strengths AND arms. This is exact -- alpha=0 adds exactly zero "
        "whatever the vector is, which G2 measures as the zero-strength "
        "identity -- but it means the condition carries no per-concept "
        "variation WITHIN a task, so f1's control set should come from "
        "control_random, which is matched-norm and does vary.")
    anomalies.append(
        f"tasks vary across concepts ({len(tasks)} of them, round-robin) but "
        f"not within one: a concept's injected and control cells share its "
        f"task, which is what pairs them. So intervals are over concepts and "
        f"tasks jointly, and no per-task rate is estimated from more than "
        f"{len(concepts) // max(1, len(tasks))} concepts.")
    if len(arms) > 1:
        anomalies.append(
            f"the two arms {arms} share their controls. That is exact for "
            f"control_zero and deliberate for control_random -- one null for "
            f"both arms is what makes the A-vs-B contrast a contrast of the "
            f"injected object -- but it means the two cascades are not "
            f"independent and a bootstrap over concepts is shared between them.")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  sweep wall-clock                   {manifest.get('wall_s', float('nan')):8.1f} s")
    if per_concept_s.size:
        w(f"  per-concept median                 {np.median(per_concept_s):8.3f} s")
        w(f"  implied full-grid time             "
          f"{np.median(per_concept_s) * len(concepts) / 60:8.2f} min")
    actual_forwards = None
    if order_of is not None:
        n_zero_cells = int(zero_mask.sum())
        # zero cells dedupe to one forward per order per batch
        distinct_zero = len(manifest["orders"])
        per_batch = (expected_cells - n_zero_cells) + distinct_zero
        # a batch is a task group, so the batch count is set by the task
        # assignment, not by ceil(n_concepts / batch)
        n_batches = len(set(
            (t, i // manifest["batch"])
            for t in tasks
            for i, _ in enumerate([c for c in concepts if task_of.get(c) == t])
        )) or int(np.ceil(len(concepts) / manifest["batch"]))
        actual_forwards = per_batch * n_batches
        batched_only = int(np.ceil(manifest["nominal_grid"] / manifest["batch"]))
        w(f"  nominal grid cells                 {manifest['nominal_grid']}")
        w(f"  forwards if batched only           {batched_only}"
          f"   (x{manifest['batch']} concepts per forward)")
        w(f"  actual forward passes              {actual_forwards}"
          f"   ({per_batch}/batch x {n_batches} batches)")
        w("    no cell was skipped. The first reduction is batching concepts")
        w("    through one forward; the second is the zero-strength dedupe,")
        w("    which is exact because alpha=0 adds exactly zero.")
    if gen_records:
        w(f"  generation wall-clock              {gen_meta.get('wall_s', float('nan')):8.1f} s")
    total_bytes = sum(p.stat().st_size for p in shards)
    w(f"  shard bytes on disk                {total_bytes:,} "
      f"({total_bytes / 2**20:.1f} MiB)")
    w(f"  gate wall-clock                    {time.time() - t0:8.1f} s")

    w("\nARTIFACTS")
    summary = {
        "n_shards": len(shards), "n_readable": len(loaded),
        "missing": missing, "unexpected": unexpected,
        "unreadable": unreadable, "strays": [p.name for p in strays],
        "nan": nan_count, "inf": inf_count,
        "duplicate_payloads": {k: v for k, v in duplicate_payloads.items()},
        "cells_complete": cells_done, "nominal_grid": manifest["nominal_grid"],
        "actual_forwards": actual_forwards,
        "n_generations": len(gen_records),
    }
    (args.out / "g4a_summary.json").write_text(json.dumps(summary, indent=1))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g4a_report.txt").write_text(report)
    print(report)
    for name in ("g4a_report.txt", "g4a_summary.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
