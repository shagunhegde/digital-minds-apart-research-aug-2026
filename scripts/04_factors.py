"""Turn sweep shards + generations into the inputs the cascade needs.

    python scripts/04_factors.py [--layer 27] [--strength 4.0]

Everything GPU-shaped happens here -- lens readouts from cached residuals, and
the position-control pass -- and is written to factors_input.npz so that
src/factors.py and gates/g4_factors.py run on any machine.

The position control needs residuals at the injection positions and at a
random position, which the sweep did not cache (it caches the report position
only, deliberately -- that is what keeps shards at 350 MB). So this script
re-runs the operating-point cells recording every position. That fresh pass
also recomputes the REPORT position, which the sweep already has, giving a
free consistency check between the cache and a recomputation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inject  # noqa: E402
import lens as lens_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None, help="injection layer")
    ap.add_argument("--strength", type=float, default=4.0)
    ap.add_argument("--probe-layer", type=int, default=None,
                    help="layer whose residual feeds the f1 probe; default = --layer")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4)
    ap.add_argument("--r-lens", action="store_true",
                    help="also read out with the R-lens from camilablank/workspace-lenses")
    ap.add_argument("--sweep", type=Path, default=ROOT / "artifacts" / "sweep")
    ap.add_argument("--gen", type=Path, default=ROOT / "artifacts" / "generations")
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "factors")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layers = cfg["planned"]["layers"]
    op_layer = args.layer if args.layer is not None else layers[0]
    probe_layer = args.probe_layer if args.probe_layer is not None else op_layer
    orders = cfg["planned"]["orders"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    manifest = json.loads((args.sweep / "manifest.json").read_text())
    selection = json.loads((args.vectors / "selection.json").read_text())
    concepts = [c for c in selection["selected"]
                if sweep_mod.shard_path(args.sweep, c).exists()]
    cache_layers = list(manifest["cache_layers"])
    task = manifest["task"]

    gen_records = []
    for path in sorted(args.gen.glob("gen_*.json")):
        gen_records.extend(json.loads(path.read_text()))
    if not gen_records:
        raise SystemExit("no generations found -- run scripts/03_generate.py first")

    # ---------------------------------------------------------------- model
    import jlens
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["repo"], revision=cfg["model"]["revision"])
    hf_model = None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            hf_model = cls.from_pretrained(
                cfg["model"]["repo"], revision=cfg["model"]["revision"],
                dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[load] {name} failed: {type(exc).__name__}", file=sys.stderr)
    if hf_model is None:
        raise RuntimeError("no transformers auto-class loaded this checkpoint")
    model = jlens.from_hf(hf_model, tok)
    device = model.input_device

    lens = lens_mod.load_lens(cfg["lens"]["repo"], cfg["lens"]["filename"],
                              cfg["lens"]["revision"], device=device)
    r_lens, r_lens_error = None, None
    if args.r_lens:
        second = cfg.get("lens_secondary", {})
        try:
            r_lens = lens_mod.load_lens(second["repo"], second["r_lens"],
                                        None, device=device)
            if r_lens.d_model != model.d_model:
                raise ValueError(
                    f"r-lens d_model {r_lens.d_model} != {model.d_model}")
        except Exception as exc:  # noqa: BLE001
            r_lens, r_lens_error = None, f"{type(exc).__name__}: {exc}"
            print(f"[r-lens] unavailable: {r_lens_error}", file=sys.stderr)

    concept_token_ids = {c: vec_mod.single_token_ids(tok, c) for c in concepts}

    # ------------------------------------------------- cells at the op point
    cell_key = {}
    cell_rows = []
    for concept in concepts:
        shard = np.load(sweep_mod.shard_path(args.sweep, concept), allow_pickle=False)
        mask = ((shard["cell_layer"] == op_layer)
                & (np.isclose(shard["cell_strength"], args.strength)))
        for order in orders:
            for condition in sweep_mod.CONDITIONS:
                sel = (mask
                       & (np.array([str(x) for x in shard["cell_order"]]) == order)
                       & (np.array([str(x) for x in shard["cell_condition"]]) == condition))
                idx = np.flatnonzero(sel)
                if idx.size == 0:
                    continue
                residuals = shard["residuals"][idx[0]]  # [n_cache_layers, d_model]
                cell_key[(concept, order, condition)] = len(cell_rows)
                cell_rows.append({
                    "concept": concept, "order": order, "condition": condition,
                    "residuals": residuals,
                    "p_true": float(shard["p_true"][idx[0]]),
                    "p_false": float(shard["p_false"][idx[0]]),
                })
    if not cell_rows:
        raise SystemExit(
            f"no sweep cells at layer {op_layer} strength {args.strength}")
    print(f"[cells] {len(cell_rows)} operating-point cells from shards")

    def ranks_from_residual(residual: torch.Tensor, concept: str, which_lens,
                            readout_layer: int) -> float:
        """1-indexed rank of the concept token in a lens readout at one layer."""
        ids = concept_token_ids[concept]
        if not ids:
            return float("nan")
        logits = model.unembed(
            lens_mod.transport(which_lens, residual, readout_layer))
        r = lens_mod.ranks_of(logits.float(),
                              torch.as_tensor(ids, device=logits.device))
        return float(r.min())

    # cached-report-position ranks, per readout layer
    cached_ranks = {l: np.full(len(cell_rows), np.nan) for l in layers}
    cached_ranks_r = {l: np.full(len(cell_rows), np.nan) for l in layers} if r_lens else {}
    with torch.inference_mode():
        for i, row in enumerate(cell_rows):
            for readout_layer in layers:
                j = cache_layers.index(readout_layer)
                h = torch.as_tensor(row["residuals"][j], device=device).unsqueeze(0)
                cached_ranks[readout_layer][i] = ranks_from_residual(
                    h, row["concept"], lens, readout_layer)
                if r_lens:
                    cached_ranks_r[readout_layer][i] = ranks_from_residual(
                        h, row["concept"], r_lens, readout_layer)
            if i % 60 == 0:
                print(f"[readout] {i}/{len(cell_rows)}", file=sys.stderr)

    # ------------------------------------------------------ position control
    vec_blob = torch.load(args.vectors / f"vectors_layer{op_layer}.pt",
                          map_location="cpu", weights_only=False)
    vecs = {w: v.to(device) for w, v in vec_blob["vectors"].items()}
    randoms = vec_mod.matched_random_directions(vecs, seed + op_layer)
    answer = manifest["prompt_meta"][orders[0]]["clean_task_answer"]

    pos_ranks = {(l, p): np.full(len(cell_rows), np.nan)
                 for l in layers for p in ("report", "injection", "random")}
    for order in orders:
        ids, positions, _ = sweep_mod.sweep_prompt(
            model, tok, order, task, answer, args.inject_width,
            args.inject_tail_offset)
        seq_len = int(ids.shape[1])
        median_norm = inject.median_residual_norm(model, ids, op_layer, positions)
        rand_pos = int(rng.integers(0, max(1, positions.start)))
        for condition in sweep_mod.CONDITIONS:
            members = [c for c in concepts if (c, order, condition) in cell_key]
            for begin in range(0, len(members), args.batch):
                chunk = members[begin:begin + args.batch]
                batch_ids = ids.expand(len(chunk), -1).contiguous()
                if condition == "control_zero":
                    source, alpha_rel = vecs, 0.0
                elif condition == "control_random":
                    source, alpha_rel = randoms, args.strength
                else:
                    source, alpha_rel = vecs, args.strength
                stacked = torch.stack([source[c] for c in chunk]).to(device)
                alpha = torch.full((len(chunk),), alpha_rel * median_norm,
                                   device=device)
                result = inject.injected_prefill(
                    model, batch_ids, op_layer, stacked, alpha, positions,
                    record_layers=layers, record_positions=slice(None))
                with torch.inference_mode():
                    for readout_layer in layers:
                        full = result["residuals"][readout_layer]
                        for k, concept in enumerate(chunk):
                            row = cell_key[(concept, order, condition)]
                            for label, index in (("report", seq_len - 1),
                                                 ("injection", positions.stop - 1),
                                                 ("random", rand_pos)):
                                h = full[k, index, :].unsqueeze(0)
                                pos_ranks[(readout_layer, label)][row] = \
                                    ranks_from_residual(h, concept, lens,
                                                        readout_layer)
        print(f"[position] {order} done", file=sys.stderr)

    # ------------------------------------------------------------- trials
    trial_cell, trial_sample, trial_ident, trial_parse = [], [], [], []
    trial_concept, trial_order, trial_condition = [], [], []
    dropped = 0
    for rec in gen_records:
        key = (rec["concept"], rec["order"], rec["condition"])
        if key not in cell_key:
            dropped += 1
            continue
        trial_cell.append(cell_key[key])
        trial_sample.append(rec["sample"])
        trial_ident.append(bool(rec["identifies"]))
        trial_parse.append(bool(rec["parse_ok"]))
        trial_concept.append(rec["concept"])
        trial_order.append(rec["order"])
        trial_condition.append(rec["condition"])

    payload = {
        "cell_concept": np.array([r["concept"] for r in cell_rows]),
        "cell_order": np.array([r["order"] for r in cell_rows]),
        "cell_condition": np.array([r["condition"] for r in cell_rows]),
        "cell_p_true": np.array([r["p_true"] for r in cell_rows]),
        "cell_features": np.stack([
            r["residuals"][cache_layers.index(probe_layer)] for r in cell_rows]),
        "trial_cell": np.array(trial_cell, dtype=np.int32),
        "trial_sample": np.array(trial_sample, dtype=np.int32),
        "trial_identifies": np.array(trial_ident, dtype=bool),
        "trial_parse_ok": np.array(trial_parse, dtype=bool),
        "trial_concept": np.array(trial_concept),
        "trial_order": np.array(trial_order),
        "trial_condition": np.array(trial_condition),
    }
    for l in layers:
        payload[f"cached_rank_L{l}"] = cached_ranks[l]
        for p in ("report", "injection", "random"):
            payload[f"pos_rank_L{l}_{p}"] = pos_ranks[(l, p)]
        if r_lens:
            payload[f"rlens_rank_L{l}"] = cached_ranks_r[l]
    np.savez(args.out / "factors_input.npz", **payload)

    meta = {
        "op_layer": op_layer, "strength": args.strength,
        "probe_layer": probe_layer, "readout_layers": layers,
        "orders": orders, "conditions": list(sweep_mod.CONDITIONS),
        "n_cells": len(cell_rows), "n_trials": len(trial_cell),
        "dropped_generations": dropped,
        "r_lens_used": bool(r_lens), "r_lens_error": r_lens_error,
        "task": task, "seed": seed, "wall_s": time.time() - t0,
    }
    (args.out / "factors_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {len(cell_rows)} cells, {len(trial_cell)} trials, "
          f"{dropped} generations dropped, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
