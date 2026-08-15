"""Run the sweep. Resumable: one shard per concept, skipped if already on disk.

    python scripts/02_sweep.py [--batch 8] [--out artifacts/sweep]

Grid: 60 concepts x 4 strengths x 2 orders x (2 arms + zero + random) = 1,920
cells, all under Garcia's workspace_band policy -- one intervention at every
layer 24-40, each scaled by the live median residual norm at that layer.

Logits-first. Each cell caches the residual at the report position for the
readout layers plus the final layer, the JSON-boolean probabilities, and the
top-k model logits. Full-vocabulary logits are never stored because they are
recoverable exactly from the cached final-layer residual -- unembed is
deterministic -- which is the difference between a 350 MB run and a 4 GB one.

The band layers themselves are NOT cached: nothing downstream reads a residual
at layer 31 of a 17-layer band, and caching all 17 would quadruple the shards
to record the injection we already know we made. f1 reads `probe_layer` (40,
the band's end), f2 reads `readout_layers`.

Generation happens separately, in scripts/03_generate.py, at the operating
point only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated; default planned.vector_arms")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "sweep")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = cfg_mod.load(ROOT)
    seed = cfg["seed"]
    policy = cfg_mod.injection_policy(cfg)
    band_layers = cfg_mod.injection_layers(cfg)
    norm_mode = cfg_mod.norm_mode(cfg)
    strengths = cfg_mod.strengths(cfg)
    orders = cfg["planned"]["orders"]
    arms = ([a.strip() for a in args.arms.split(",") if a.strip()]
            if args.arms else cfg_mod.vector_arms(cfg))
    tasks = prompt_mod.TASK_PROMPTS[:cfg["planned"].get("n_tasks", 10)]
    torch.manual_seed(seed)

    selection_path = args.vectors / "selection.json"
    if not selection_path.exists():
        raise SystemExit(
            f"{selection_path} not found -- run scripts/01_concept_vectors.py first")
    selection = json.loads(selection_path.read_text())
    concepts = selection["selected"]

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

    vectors_by_arm = {arm: sweep_mod.load_arm(args.vectors, arm, band_layers, device)
                      for arm in arms}
    # One random-direction set, shared by both arms: the band hook
    # unit-normalises whatever it is given and scales by the live residual
    # norm, so "matched norm" is now automatic and a second set would only add
    # noise to the comparison between the arms.
    randoms_by_layer = {
        layer: vec_mod.matched_random_directions(
            vectors_by_arm[arms[0]][layer], seed + layer)
        for layer in band_layers
    }

    cells = sweep_mod.build_cells(strengths, orders, arms)
    # f2 reads out at readout_layers, which are NOT the injection layers:
    # G1 measured the lens as ~25x better at 59 than at 27/31/35. The probe
    # layer and the final block (for model logits) are cached too.
    readout_layers = cfg["planned"]["readout_layers"]
    probe_layer = cfg_mod.probe_layer(cfg)
    cache_layers = sorted({*readout_layers, probe_layer, model.n_layers - 1})
    true_ids, false_ids = sweep_mod.boolean_token_ids(tok)
    if not true_ids or not false_ids:
        raise RuntimeError(
            f"JSON booleans are not single tokens: true={true_ids} false={false_ids}")
    true_t = torch.as_tensor(true_ids, device=device)
    false_t = torch.as_tensor(false_ids, device=device)

    # 20 prompts (2 orders x 10 tasks) rather than 2, each with its own greedy
    # clean answer for the task-first prefill and its own per-band-layer clean
    # norms for the report.
    prompt_cache, answers, prompt_meta = sweep_mod.build_prompt_cache(
        model, hf_model, tok, orders, tasks, band_layers)
    task_of = sweep_mod.assign_tasks(concepts, tasks)
    for task in tasks:
        members = [c for c in concepts if task_of[c] == task]
        print(f"[task] {task!r} -> {len(members)} concepts, "
              f"clean answer {answers[task]!r}")

    fingerprint = sweep_mod.config_fingerprint(
        policy, band_layers, strengths, orders, readout_layers, tasks, arms,
        norm_mode)
    done = sweep_mod.completed_concepts(args.out, concepts, fingerprint)
    todo = [c for c in concepts if c not in done]
    groups = sweep_mod.task_groups(todo, task_of, args.batch)
    print(f"[sweep] {policy} over layers {band_layers[0]}-{band_layers[-1]} "
          f"({len(band_layers)} layers), arms {arms}, norm_mode {norm_mode}")
    print(f"[sweep] {len(cells)} cells/concept | {len(done)} shards done, "
          f"{len(todo)} to run in {len(groups)} task groups")

    cell_arrays = sweep_mod.cells_to_arrays(cells)
    timings = []
    n_done = 0
    for task, chunk in groups:
        started = time.time()
        prompts_by_order = {order: prompt_cache[(order, task)] for order in orders}
        results = sweep_mod.run_batch(
            model, chunk, vectors_by_arm, randoms_by_layer, cells,
            band_layers, cache_layers, prompts_by_order, true_t, false_t,
            args.topk, norm_mode=norm_mode)
        for concept, payload in results.items():
            sweep_mod.write_shard_atomic(
                sweep_mod.shard_path(args.out, concept),
                concept=np.array(concept),
                task=np.array(task),
                cache_layers=np.array(cache_layers, dtype=np.int32),
                band_layers=np.array(band_layers, dtype=np.int32),
                written_at=np.array(time.time()),
                fingerprint=np.array(fingerprint),
                **cell_arrays, **payload)
        elapsed = time.time() - started
        n_done += len(chunk)
        timings.append({"concepts": chunk, "task": task, "seconds": elapsed,
                        "written_at": time.time()})
        print(f"[sweep] {n_done}/{len(todo)}  {elapsed:.1f}s "
              f"({elapsed / len(chunk):.2f}s/concept)")

    manifest = {
        "concepts": concepts,
        "cells_per_concept": len(cells),
        "nominal_grid": len(concepts) * len(cells),
        "injection_policy": policy,
        "band_layers": band_layers,
        "norm_mode": norm_mode,
        "median_scope": "per_element",
        "arms": arms,
        "strengths": strengths, "orders": orders,
        "conditions": list(sweep_mod.CONDITIONS),
        "cache_layers": cache_layers,
        "readout_layers": readout_layers,
        "probe_layer": probe_layer,
        "f2_primary_layer": cfg["planned"]["f2_primary_layer"],
        "tasks": tasks,
        "task_of": task_of,
        "clean_task_answers": answers,
        "prompt_meta": prompt_meta,
        "true_ids": true_ids, "false_ids": false_ids,
        "topk": args.topk, "batch": args.batch, "seed": seed,
        "model_revision": cfg["model"]["revision"],
        "fingerprint": fingerprint,
        "timings": timings,
        "wall_s": time.time() - t0,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {time.time() - t0:.1f}s -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
