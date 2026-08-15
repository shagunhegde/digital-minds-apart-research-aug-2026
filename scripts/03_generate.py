"""Generate at the operating point only. Resumable per concept.

    python scripts/03_generate.py [--layer 27] [--strength 4.0]

T=1.0 with 4 samples per condition. Garcia used T=0 with one completion and
flags it as a limitation; sampling at temperature with several draws is a cheap
and legible improvement over the closest prior work, and it is what makes the
f3 rate an estimate with a spread rather than a single draw.

Generation is the expensive channel, so it runs at one (layer, strength) for
both orders and all three conditions -- everything else in the sweep is
logits-first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inject  # noqa: E402
import judge as judge_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--strength", type=float, default=4.0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "generations")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layer = args.layer if args.layer is not None else cfg["planned"]["layers"][0]
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

    blob = torch.load(args.vectors / f"vectors_layer{layer}.pt",
                      map_location="cpu", weights_only=False)
    vecs = {w: v.to(device) for w, v in blob["vectors"].items()}
    randoms = vec_mod.matched_random_directions(vecs, seed + layer)

    # Generation continues from EXACTLY the prompt the sweep cached residuals
    # for -- the one prefilled up to the detection key. This is not cosmetic:
    # f1 and f2 are properties of that prefill's residual and f3 is a property
    # of the continuation, so if the two used different prompts then "the same
    # trial" would be undefined and cascade_residual would be measuring the
    # mismatch instead of a denominator bug. The model simply finishes the JSON
    # object it has been started on.
    answer = sweep_mod.clean_task_answer(model, hf_model, tok, task)
    prompts, prefills = {}, {}
    for order in orders:
        ids, positions, _rendered = sweep_mod.sweep_prompt(
            model, tok, order, task, answer, args.inject_width,
            args.inject_tail_offset)
        prefills[order] = ('{"change_detected":' if order == "report_then_task"
                           else '{"task_answer": "' + answer + '", "change_detected":')
        prompts[order] = (ids, positions,
                          inject.median_residual_norm(model, ids, layer, positions))

    for begin in range(0, len(concepts), args.batch):
        chunk = concepts[begin:begin + args.batch]
        path = args.out / f"gen_{begin:04d}.json"
        if path.exists():
            print(f"[gen] {path.name} exists, skipping")
            continue
        records = []
        for order in orders:
            ids, positions, median_norm = prompts[order]
            batch_ids = ids.expand(len(chunk), -1).contiguous()
            for condition in sweep_mod.CONDITIONS:
                if condition == "control_zero":
                    source, alpha_rel = vecs, 0.0
                elif condition == "control_random":
                    source, alpha_rel = randoms, args.strength
                else:
                    source, alpha_rel = vecs, args.strength
                stacked = torch.stack([source[c] for c in chunk]).to(device)
                alpha = torch.full((len(chunk),), alpha_rel * median_norm,
                                   device=device)
                for sample in range(args.samples):
                    gen = inject.generate_with_injection(
                        model, hf_model, tok, batch_ids, layer, stacked, alpha,
                        positions, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, seed=seed + 1000 * sample)
                    if gen["n_fires"] != 1:
                        raise RuntimeError(
                            f"hook fired {gen['n_fires']} times, expected 1 "
                            f"(prefill only) for {order}/{condition}")
                    for k, concept in enumerate(chunk):
                        text = gen["completions"][k]
                        # the opening brace lives in the prompt, so the object
                        # only parses once the prefill is put back in front
                        full = prefills[order] + text
                        parsed = judge_mod.parse_three_key_json(full)
                        records.append({
                            "concept": concept, "order": order,
                            "condition": condition, "sample": sample,
                            "layer": layer, "strength": alpha_rel,
                            "response": text,
                            "full_json": full,
                            "parsed": parsed,
                            "parse_ok": parsed is not None,
                            "identifies": judge_mod.mention_identifies(text, concept),
                            "new_tokens": gen["new_token_counts"][k],
                        })
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=1, ensure_ascii=False))
        tmp.replace(path)
        print(f"[gen] wrote {path.name}  {len(records)} records "
              f"({begin + len(chunk)}/{len(concepts)} concepts)")

    meta = {
        "layer": layer, "strength": args.strength, "samples": args.samples,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "orders": orders, "conditions": list(sweep_mod.CONDITIONS),
        "task": task, "clean_task_answer": answer, "seed": seed,
        "n_concepts": len(concepts), "wall_s": time.time() - t0,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {time.time() - t0:.1f}s -> {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
