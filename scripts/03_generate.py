"""Generate at the operating point only. Resumable per task group.

    python scripts/03_generate.py [--strength 0.09]

T=1.0 with 4 samples per condition. Garcia used T=0 with one completion and
flags it as a limitation; sampling at temperature with several draws is a cheap
and legible improvement over the closest prior work, and it is what makes the
f3 rate an estimate with a spread rather than a single draw.

Generation is the expensive channel, so it runs at one strength for both
orders, both arms and the two controls -- everything else in the sweep is
logits-first. 60 concepts x 2 orders x (2 arms + zero + random) x 4 samples =
1,920 generations. The change order budgeted 2,880 by counting the controls
once per arm; they are shared here for the same reason they are shared in the
sweep, which is that alpha=0 adds exactly zero whatever the vector is and one
random-direction set is the null for both arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import band_inject  # noqa: E402
import config as cfg_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=None,
                    help="default: planned.operating_strength from sprint.yaml")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated; default planned.vector_arms")
    ap.add_argument("--task-first-prefill", choices=("clean", "natural"),
                    default="clean",
                    help="clean: quote the model's own CLEAN greedy answer in "
                         "the task-first prefill, matching the sweep's cached "
                         "residuals. natural: prefill only '{\"task_answer\": \"' "
                         "and let the model write its own, possibly steered, "
                         "answer -- Garcia's actual condition.")
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "generations")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = cfg_mod.load(ROOT)
    seed = cfg["seed"]
    policy = cfg_mod.injection_policy(cfg)
    band_layers = cfg_mod.injection_layers(cfg)
    norm_mode = cfg_mod.norm_mode(cfg)
    strength = cfg_mod.operating_strength(cfg, args.strength)
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
    randoms_by_layer = {
        layer: vec_mod.matched_random_directions(
            vectors_by_arm[arms[0]][layer], seed + layer)
        for layer in band_layers
    }

    # Generation continues from EXACTLY the prompt the sweep cached residuals
    # for -- the one prefilled up to the detection key, for this concept's own
    # task. This is not cosmetic: f1 and f2 are properties of that prefill's
    # residual and f3 is a property of the continuation, so if the two used
    # different prompts then "the same trial" would be undefined and
    # cascade_residual would be measuring the mismatch instead of a
    # denominator bug. The model simply finishes the JSON object it has been
    # started on.
    prompt_cache, answers, _meta = sweep_mod.build_prompt_cache(
        model, hf_model, tok, orders, tasks, band_layers,
        task_first_prefill=args.task_first_prefill)
    task_of = sweep_mod.assign_tasks(concepts, tasks)
    groups = sweep_mod.task_groups(concepts, task_of, args.batch)

    # (condition, arm) pairs: the controls are shared across arms.
    combos = [("injected", arm) for arm in arms]
    combos += [("control_zero", sweep_mod.SHARED_ARM),
               ("control_random", sweep_mod.SHARED_ARM)]

    n_done = 0
    for group_index, (task, chunk) in enumerate(groups):
        path = args.out / f"gen_{group_index:04d}.json"
        n_done += len(chunk)
        if path.exists():
            print(f"[gen] {path.name} exists, skipping")
            continue
        records = []
        for order in orders:
            ids, positions, clean_norms = prompt_cache[(order, task)]
            prefill = sweep_mod.report_prefill(
                order, answers[task], args.task_first_prefill)
            batch_ids = ids.expand(len(chunk), -1).contiguous()
            for condition, arm in combos:
                if condition == "control_zero":
                    source, alpha_rel = vectors_by_arm[arms[0]], 0.0
                elif condition == "control_random":
                    source, alpha_rel = randoms_by_layer, strength
                else:
                    source, alpha_rel = vectors_by_arm[arm], strength
                stacked = {l: torch.stack([source[l][c] for c in chunk]).to(device)
                           for l in band_layers}
                alpha = torch.full((len(chunk),), alpha_rel, device=device)
                for sample in range(args.samples):
                    gen = band_inject.generate_with_injection_band(
                        model, hf_model, tok, batch_ids, band_layers, stacked,
                        alpha, positions, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, seed=seed + 1000 * sample,
                        norm_mode=norm_mode, clean_norms=clean_norms)
                    if gen["n_layers_fired"] != len(band_layers) or \
                            gen["max_fires_per_layer"] != 1:
                        raise RuntimeError(
                            f"band fired on {gen['n_layers_fired']} of "
                            f"{len(band_layers)} layers, max "
                            f"{gen['max_fires_per_layer']} times each; expected "
                            f"every layer exactly once (prefill only) for "
                            f"{order}/{condition}/{arm}")
                    for k, concept in enumerate(chunk):
                        text = gen["completions"][k]
                        # the opening brace lives in the prompt, so the object
                        # only parses once the prefill is put back in front
                        full = prefill + text
                        parsed = judge_mod.parse_three_key_json(full)
                        # The REPORT channel and the STEERING channel are
                        # scored separately and must never be pooled. In the
                        # first band run they were: `identifies` matched the
                        # whole response, so a trial that reported
                        # change_detected=false and was steered into answering
                        # "Granite" for the concept Granite counted as an
                        # identification. All 16 "reports" were that.
                        hit, how = judge_mod.report_identifies(parsed, text, concept)
                        steered = judge_mod.answer_steered(parsed, concept)
                        if (order == "task_then_report"
                                and args.task_first_prefill == "clean"):
                            # the answer was handed to the model; whatever it
                            # contains is not evidence about steering
                            steered = None
                        records.append({
                            "concept": concept, "order": order,
                            "condition": condition, "arm": arm,
                            "sample": sample, "task": task,
                            "strength": alpha_rel,
                            "task_first_prefill": args.task_first_prefill,
                            "response": text,
                            "full_json": full,
                            "parsed": parsed,
                            "parse_ok": parsed is not None,
                            "identifies": hit,
                            "identifies_scored_on": how,
                            "identifies_anywhere":
                                judge_mod.mention_identifies(text, concept),
                            "steered": steered,
                            "claims_detection": bool(
                                parsed and parsed.get("change_detected") is True),
                            "new_tokens": gen["new_token_counts"][k],
                        })
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=1, ensure_ascii=False))
        tmp.replace(path)
        print(f"[gen] wrote {path.name}  {len(records)} records "
              f"({n_done}/{len(concepts)} concepts, task {task!r})")

    meta = {
        "injection_policy": policy, "band_layers": band_layers,
        "norm_mode": norm_mode,
        "strength": strength, "samples": args.samples,
        "task_first_prefill": args.task_first_prefill,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "orders": orders, "conditions": list(sweep_mod.CONDITIONS),
        "arms": arms, "tasks": tasks, "task_of": task_of,
        "clean_task_answers": answers, "seed": seed,
        "n_concepts": len(concepts), "wall_s": time.time() - t0,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {time.time() - t0:.1f}s -> {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
