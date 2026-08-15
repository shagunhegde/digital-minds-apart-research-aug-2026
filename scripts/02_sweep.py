"""Run the sweep. Resumable: one shard per concept, skipped if already on disk.

    python scripts/02_sweep.py [--batch 8] [--out artifacts/sweep]

Logits-first. Each cell caches the residual at the report position for the
swept layers plus the final layer, the JSON-boolean probabilities, and the
top-k model logits. Full-vocabulary logits are never stored because they are
recoverable exactly from the cached final-layer residual -- unembed is
deterministic -- which is the difference between a 350 MB run and a 4 GB one.

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inject  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--task", type=str, default=None,
                    help="the fixed sweep task; default TASK_PROMPTS[0]")
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "sweep")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layers = cfg["planned"]["layers"]
    strengths = cfg["planned"]["strengths_rel"]
    orders = cfg["planned"]["orders"]
    task = args.task or prompt_mod.TASK_PROMPTS[0]
    torch.manual_seed(seed)

    selection = json.loads((args.vectors / "selection.json").read_text())
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

    vectors_by_layer, randoms_by_layer = {}, {}
    for layer in layers:
        blob = torch.load(args.vectors / f"vectors_layer{layer}.pt",
                          map_location="cpu", weights_only=False)
        vectors_by_layer[layer] = {w: v.to(device) for w, v in blob["vectors"].items()}
        randoms_by_layer[layer] = vec_mod.matched_random_directions(
            vectors_by_layer[layer], seed + layer)

    cells = sweep_mod.build_cells(layers, strengths, orders)
    # f2 reads out at readout_layers, which are NOT the injection layers:
    # G1 measured the lens as ~25x better at 59 than at 27/31/35. The final
    # block is cached too (for model logits), though it carries no lens.
    readout_layers = cfg['planned']['readout_layers']
    cache_layers = sorted({*layers, *readout_layers, model.n_layers - 1})
    true_ids, false_ids = sweep_mod.boolean_token_ids(tok)
    if not true_ids or not false_ids:
        raise RuntimeError(
            f"JSON booleans are not single tokens: true={true_ids} false={false_ids}")
    true_t = torch.as_tensor(true_ids, device=device)
    false_t = torch.as_tensor(false_ids, device=device)

    # one prompt per order, plus the per-layer norm that makes alpha comparable
    # order-independent: the task-first prefill quotes the model's own greedy
    # answer to the fixed task, so it is computed once, not once per order
    answer = sweep_mod.clean_task_answer(model, hf_model, tok, task)
    prompts_by_order, prompt_meta = {}, {}
    for order in orders:
        ids, positions, rendered = sweep_mod.sweep_prompt(
            model, tok, order, task, answer, args.inject_width,
            args.inject_tail_offset)
        norms = {l: inject.median_residual_norm(model, ids, l, positions)
                 for l in layers}
        prompts_by_order[order] = (ids, positions, norms)
        prompt_meta[order] = {
            "clean_task_answer": answer,
            "seq_len": int(ids.shape[1]),
            "window": [positions.start, positions.stop],
            "median_norms": norms,
            "rendered_tail": rendered[-220:],
        }
        print(f"[prompt] {order}: seq={ids.shape[1]} window={positions.start}:"
              f"{positions.stop} answer={answer!r}")

    done = sweep_mod.completed_concepts(args.out, concepts)
    todo = [c for c in concepts if c not in done]
    print(f"[sweep] {len(cells)} cells/concept | {len(done)} shards done, "
          f"{len(todo)} to run")

    cell_arrays = sweep_mod.cells_to_arrays(cells)
    timings = []
    for begin in range(0, len(todo), args.batch):
        chunk = todo[begin:begin + args.batch]
        started = time.time()
        results = sweep_mod.run_batch(
            model, tok, chunk, vectors_by_layer, randoms_by_layer, cells,
            cache_layers, prompts_by_order, true_t, false_t, args.topk)
        for concept, payload in results.items():
            sweep_mod.write_shard_atomic(
                sweep_mod.shard_path(args.out, concept),
                concept=np.array(concept),
                cache_layers=np.array(cache_layers, dtype=np.int32),
                written_at=np.array(time.time()),
                **cell_arrays, **payload)
        elapsed = time.time() - started
        timings.append({"concepts": chunk, "seconds": elapsed,
                        "written_at": time.time()})
        print(f"[sweep] {begin + len(chunk)}/{len(todo)}  {elapsed:.1f}s "
              f"({elapsed / len(chunk):.2f}s/concept)")

    manifest = {
        "concepts": concepts,
        "cells_per_concept": len(cells),
        "nominal_grid": len(concepts) * len(cells),
        "layers": layers, "strengths": strengths, "orders": orders,
        "conditions": list(sweep_mod.CONDITIONS),
        "cache_layers": cache_layers,
        "readout_layers": readout_layers,
        "f2_primary_layer": cfg["planned"]["f2_primary_layer"],
        "task": task,
        "prompt_meta": prompt_meta,
        "true_ids": true_ids, "false_ids": false_ids,
        "topk": args.topk, "batch": args.batch, "seed": seed,
        "model_revision": cfg["model"]["revision"],
        "timings": timings,
        "wall_s": time.time() - t0,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[done] {time.time() - t0:.1f}s -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
