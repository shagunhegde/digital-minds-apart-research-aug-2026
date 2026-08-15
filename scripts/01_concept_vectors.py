"""Select the 60 concepts and extract their vectors. Writes to artifacts/vectors/.

Resumable: every stage checks for its output file first, so a disconnect costs
only the stage in flight.

    python scripts/01_concept_vectors.py [--per-tier 20] [--pilot-strength 0.09]

TWO ARMS, EXTRACTED PER BAND LAYER.

  concept    activation("Tell me about X") - mean(baseline), at EVERY band
             layer, from one forward per word. Injecting a layer-27 vector at
             layer 38 is a cross-layer mismatch, and under a 17-layer band 16
             of the 17 injections would be that mismatch; per-layer extraction
             removes it and costs nothing, because the 17 layers come off the
             same forward.
  jlens_row  (W_U[t] @ J_l) in layer-l space -- the exact object Garcia
             injects. The positive control: if this arm reproduces his
             steering and report rates, the harness is validated end to end.
             Needs the fitted lens, so it is behind --arms and skipped if the
             lens will not load.

The pilot that stratifies the pool now runs under the BAND too. The previous
pilot scored 315 concepts at one layer, where the whole ladder was ~20x too
weak: every rate sat on floor noise, so the 60 it selected were effectively
random. Re-running it here is P0.5 step 2 of the change order.

STRATIFICATION -- READ THIS.

The build spec stratifies the 60 concepts by Macar's per-concept Gemma
detection rates. Those rates are not published: `plotting/data/*.parquet` are
aggregates over (layer_idx, strength, arm), the abliterated checkpoint is
weights only, and the README reports aggregate findings. Nothing reachable
carries a per-concept number.

So this script measures its OWN per-concept detection over the full
single-token pool, using next-token logits (Macar's other prior, ~10x cheaper
than generating), and stratifies on that. Consequences, all reported in G3:

  + no cross-model transfer, so no regression to the mean from another
    model's noisy labels -- the trap the build spec's own table warns about
  + bimodality can be tested on the FULL pool rather than on 60 concepts
    chosen to span the range, which would have manufactured the spread
  - the cross-model Spearman against Gemma cannot be computed at all
  - selecting on a measured outcome means tier membership is not independent
    of detection, so tiers are a sampling device only and G3 never reports
    "high-tier detected at X%"

Pass --stratify-file to supply external per-concept rates instead (JSON:
{"word": rate}), which restores the spec's original design if the numbers
turn up.
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
import lens as lens_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402
import vectors as vec_mod  # noqa: E402


def load_model(cfg):
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
    return jlens.from_hf(hf_model, tok), hf_model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tier", type=int, default=20)
    ap.add_argument("--pilot-strength", type=float, default=None,
                    help="default: planned.operating_strength from sprint.yaml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated; default planned.vector_arms")
    ap.add_argument("--stratify-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "vectors")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = cfg_mod.load(ROOT)
    seed = cfg["seed"]
    layers = cfg_mod.injection_layers(cfg)
    policy = cfg_mod.injection_policy(cfg)
    norm_mode = cfg_mod.norm_mode(cfg)
    arms = ([a.strip() for a in args.arms.split(",") if a.strip()]
            if args.arms else cfg_mod.vector_arms(cfg))
    pilot_strength = cfg_mod.operating_strength(cfg, args.pilot_strength)
    torch.manual_seed(seed)

    model, hf_model, tok = load_model(cfg)
    device = model.input_device

    # ------------------------------------------------ pool + composition audit
    pool, pool_meta = vec_mod.load_concept_pool(ROOT / "configs" / "concepts.json")
    concept_ids = {}
    for item in pool:
        ids = vec_mod.single_token_ids(tok, item["word"])
        if ids:
            concept_ids[item["word"]] = ids
    audit = vec_mod.composition_audit(pool, set(concept_ids))
    print(f"[pool] {audit['n_before']} -> {audit['n_after']} single-token")
    for bucket, keep in audit["bucket_retention"].items():
        print(f"       {bucket:<9} retention {keep:.3f}")

    survivors = [c["word"] for c in pool if c["word"] in concept_ids]
    baseline_words, baseline_diag = vec_mod.load_baseline_words(
        ROOT / "configs" / "baseline_words.json")

    # ------------------------------------------------------------- pilot
    tasks = prompt_mod.TASK_PROMPTS[:cfg["planned"].get("n_tasks", 10)]
    pilot_path = args.out / "pilot.json"
    pool_vectors = None      # {layer: {word: vec}} for the whole survivor pool
    if args.stratify_file is not None:
        rates = json.loads(args.stratify_file.read_text())
        rates = {w: float(r) for w, r in rates.items() if w in concept_ids}
        rate_source = str(args.stratify_file)
    elif pilot_path.exists():
        payload = json.loads(pilot_path.read_text())
        rates, rate_source = payload["rates"], payload["source"]
        print(f"[pilot] resumed from {pilot_path}")
    else:
        print(f"[pilot] scoring {len(survivors)} concepts under the {policy} "
              f"({len(layers)} layers {layers[0]}-{layers[-1]}), "
              f"alpha_rel={pilot_strength} per layer")
        pool_vectors, _ = vec_mod.extract_concept_vectors_band(
            model, tok, survivors, baseline_words, layers)
        yes_ids, no_ids = prompt_mod.boolean_token_ids(tok)
        yes_t = torch.as_tensor(yes_ids, device=device)
        no_t = torch.as_tensor(no_ids, device=device)

        # The pilot gets the same 10-way task spread the sweep does. Selecting
        # 60 concepts on their response to one fixed prompt makes the whole
        # selection a property of that prompt; this is the cheapest place to
        # stop that, since the batching is by task group either way.
        task_of = sweep_mod.assign_tasks(survivors, tasks)
        prompt_cache = {}
        scores, done = {}, 0
        for task, chunk in sweep_mod.task_groups(survivors, task_of, args.batch):
            if task not in prompt_cache:
                prompt_cache[task] = prompt_mod.prepare(
                    model, tok, prompt_mod.detect_messages(task), prefill=True,
                    enable_thinking=cfg["planned"]["enable_thinking"])
            base_ids, positions, _ = prompt_cache[task]
            batch = base_ids.expand(len(chunk), -1).contiguous()
            result = band_inject.injected_prefill_band(
                model, batch, layers,
                {l: torch.stack([pool_vectors[l][w] for w in chunk]).to(device)
                 for l in layers},
                torch.full((len(chunk),), pilot_strength, device=device),
                positions, record_layers=[layers[-1]], record_positions=[-1],
                norm_mode=norm_mode)
            probs = torch.softmax(result["logits"], dim=-1)
            p_yes = probs.index_select(-1, yes_t).sum(-1)
            p_no = probs.index_select(-1, no_t).sum(-1)
            ratio = (p_yes / (p_yes + p_no)).cpu().numpy()
            for word, value in zip(chunk, ratio):
                scores[word] = float(value)
            done += len(chunk)
            if done % (args.batch * 10) < args.batch:
                print(f"       {done}/{len(survivors)}")
        rates = scores
        rate_source = (
            f"own pilot, {policy} over layers {layers[0]}-{layers[-1]}, "
            f"alpha_rel {pilot_strength} per layer, norm_mode {norm_mode}, "
            f"{len(tasks)} tasks round-robin, "
            f"P(yes)/(P(yes)+P(no)) from next-token logits")
        pilot_path.write_text(json.dumps({
            "source": rate_source, "injection_policy": policy,
            "band_layers": layers, "strength": pilot_strength,
            "norm_mode": norm_mode, "n_tasks": len(tasks),
            "task_of": task_of, "n_scored": len(rates), "rates": rates},
            indent=1))
        print(f"[pilot] wrote {pilot_path}")

    # --------------------------------------------------------- selection
    selected, strat_meta = vec_mod.stratify_by_rate(
        [w for w in survivors if w in rates], rates, args.per_tier, seed)
    print(f"[select] {len(selected)} concepts  {strat_meta}")

    # -------------------------------------------- arm A: concept vectors
    def arm_path(arm: str, layer: int) -> Path:
        stem = "vectors" if arm == "concept" else "jlens_rows"
        return args.out / f"{stem}_layer{layer}.pt"

    def on_disk(arm: str) -> bool:
        """Whether every band layer of `arm` is already saved for THIS 60.

        Resuming on file existence alone is not enough: these vectors are for
        whichever concepts the previous selection picked. If the selection
        changed -- and after the re-pilot it has -- silently keeping them
        would sweep the wrong concept set with no error anywhere.
        """
        for layer in layers:
            path = arm_path(arm, layer)
            if not path.exists():
                return False
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if list(saved.get("concepts", [])) != list(selected):
                print(f"[extract] {arm} L{layer} on disk but for a different "
                      f"selection ({len(saved.get('concepts', []))} concepts)"
                      f" -- re-extracting the arm")
                return False
        return True

    if "concept" in arms:
        if on_disk("concept"):
            print(f"[extract] concept arm already on disk for all {len(layers)} "
                  f"band layers, skipping")
        else:
            if pool_vectors is not None:
                # the pilot already extracted the whole pool at every band
                # layer; the 60 are a subset of it, so re-extracting would be
                # 160 forwards for nothing
                vecs = {l: {w: pool_vectors[l][w] for w in selected} for l in layers}
                means = None
                print("[extract] concept arm subset from the pilot extraction")
            else:
                vecs, means = vec_mod.extract_concept_vectors_band(
                    model, tok, selected, baseline_words, layers)
            for layer in layers:
                torch.save({
                    "arm": "concept",
                    "layer": layer,
                    "concepts": selected,
                    "vectors": {w: v.cpu() for w, v in vecs[layer].items()},
                    "baseline_mean": (means[layer].cpu() if means else None),
                    "model_revision": cfg["model"]["revision"],
                }, arm_path("concept", layer))
            print(f"[extract] wrote concept vectors for {len(layers)} band layers")
    del pool_vectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------- arm B: J-lens row vectors
    jlens_error = None
    if "jlens_row" in arms:
        if on_disk("jlens_row"):
            print("[extract] jlens_row arm already on disk, skipping")
        else:
            try:
                lens = lens_mod.load_lens(
                    cfg["lens"]["repo"], cfg["lens"]["filename"],
                    cfg["lens"]["revision"], device=device, layers=layers)
                rows = band_inject.rows_by_layer(
                    lens, model, {w: concept_ids[w] for w in selected}, layers)
                for layer in layers:
                    torch.save({
                        "arm": "jlens_row",
                        "layer": layer,
                        "concepts": selected,
                        "vectors": {w: v.cpu() for w, v in rows[layer].items()},
                        "fold_final_norm": True,
                        "lens": {"repo": cfg["lens"]["repo"],
                                 "filename": cfg["lens"]["filename"],
                                 "revision": cfg["lens"]["revision"]},
                        "model_revision": cfg["model"]["revision"],
                    }, arm_path("jlens_row", layer))
                print(f"[extract] wrote J-lens rows for {len(layers)} band layers")
                del lens, rows
            except Exception as exc:  # noqa: BLE001
                # The lens is 6.6 GB resident and this arm is a positive
                # control, not the headline. Report the failure and let the
                # concept arm proceed rather than losing both.
                jlens_error = f"{type(exc).__name__}: {exc}"
                print(f"[extract] jlens_row arm unavailable: {jlens_error}",
                      file=sys.stderr)

    selection = {
        "selected": selected,
        "rate_source": rate_source,
        "rates_of_selected": {w: rates[w] for w in selected},
        "stratification": strat_meta,
        "composition_audit": audit,
        "pool_meta": pool_meta,
        "baseline_diag": baseline_diag,
        "concept_ids": {w: concept_ids[w] for w in selected},
        "injection_policy": policy,
        "band_layers": layers,
        "layers": layers,
        "arms": arms,
        "jlens_row_error": jlens_error,
        "pilot_strength": pilot_strength,
        "norm_mode": norm_mode,
        "seed": seed,
        "wall_s": time.time() - t0,
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=1))
    print(f"[done] wrote {args.out / 'selection.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
