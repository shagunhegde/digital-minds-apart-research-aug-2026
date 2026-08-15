"""Select the 60 concepts and extract their vectors. Writes to artifacts/vectors/.

Resumable: every stage checks for its output file first, so a disconnect costs
only the stage in flight.

    python scripts/01_concept_vectors.py [--per-tier 20] [--pilot-strength 4.0]

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

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inject  # noqa: E402
import prompts as prompt_mod  # noqa: E402
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
    ap.add_argument("--pilot-strength", type=float, default=4.0)
    ap.add_argument("--pilot-layer", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4)
    ap.add_argument("--stratify-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "vectors")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layers = cfg["planned"]["layers"]
    pilot_layer = args.pilot_layer if args.pilot_layer is not None else layers[0]
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
    pilot_path = args.out / "pilot.json"
    if args.stratify_file is not None:
        rates = json.loads(args.stratify_file.read_text())
        rates = {w: float(r) for w, r in rates.items() if w in concept_ids}
        rate_source = str(args.stratify_file)
    elif pilot_path.exists():
        payload = json.loads(pilot_path.read_text())
        rates, rate_source = payload["rates"], payload["source"]
        print(f"[pilot] resumed from {pilot_path}")
    else:
        print(f"[pilot] scoring {len(survivors)} concepts at layer {pilot_layer}, "
              f"alpha_rel={args.pilot_strength}")
        vecs_pilot, _ = vec_mod.extract_concept_vectors(
            model, tok, survivors, baseline_words, pilot_layer)
        yes_ids, no_ids = prompt_mod.boolean_token_ids(tok)
        yes_t = torch.as_tensor(yes_ids, device=device)
        no_t = torch.as_tensor(no_ids, device=device)

        base_ids, positions, _ = prompt_mod.prepare(
            model, tok, prompt_mod.detect_messages(), prefill=True,
            enable_thinking=cfg["planned"]["enable_thinking"])
        median_norm = inject.median_residual_norm(
            model, base_ids, pilot_layer, positions)

        scores = {}
        for begin in range(0, len(survivors), args.batch):
            chunk = survivors[begin:begin + args.batch]
            batch = base_ids.expand(len(chunk), -1).contiguous()
            result = inject.injected_prefill(
                model, batch, pilot_layer,
                torch.stack([vecs_pilot[w] for w in chunk]).to(device),
                torch.full((len(chunk),), args.pilot_strength * median_norm,
                           device=device),
                positions, record_layers=[pilot_layer], record_positions=positions)
            probs = torch.softmax(result["logits"], dim=-1)
            p_yes = probs.index_select(-1, yes_t).sum(-1)
            p_no = probs.index_select(-1, no_t).sum(-1)
            ratio = (p_yes / (p_yes + p_no)).cpu().numpy()
            for word, value in zip(chunk, ratio):
                scores[word] = float(value)
            if begin % (args.batch * 10) == 0:
                print(f"       {begin + len(chunk)}/{len(survivors)}")
        rates = scores
        rate_source = (
            f"own pilot, layer {pilot_layer}, alpha_rel {args.pilot_strength}, "
            f"P(yes)/(P(yes)+P(no)) from next-token logits")
        pilot_path.write_text(json.dumps({
            "source": rate_source, "layer": pilot_layer,
            "strength": args.pilot_strength, "median_norm": median_norm,
            "n_scored": len(rates), "rates": rates}, indent=1))
        print(f"[pilot] wrote {pilot_path}")
        del vecs_pilot
        torch.cuda.empty_cache()

    # --------------------------------------------------------- selection
    selected, strat_meta = vec_mod.stratify_by_rate(
        [w for w in survivors if w in rates], rates, args.per_tier, seed)
    print(f"[select] {len(selected)} concepts  {strat_meta}")

    # --------------------------------------------------------- extraction
    for layer in layers:
        path = args.out / f"vectors_layer{layer}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if list(saved.get("concepts", [])) == list(selected):
                print(f"[extract] layer {layer} already on disk, skipping")
                continue
            # Resuming on file existence alone is not enough: these vectors are
            # for whichever concepts the previous selection picked. If the
            # selection changed, silently keeping them would sweep the wrong
            # concept set with no error anywhere.
            print(f"[extract] layer {layer} on disk but for a different "
                  f"selection ({len(saved.get('concepts', []))} concepts) -- "
                  f"re-extracting")
        vecs, baseline_mean = vec_mod.extract_concept_vectors(
            model, tok, selected, baseline_words, layer)
        torch.save({
            "layer": layer,
            "concepts": selected,
            "vectors": {w: v.cpu() for w, v in vecs.items()},
            "baseline_mean": baseline_mean.cpu(),
            "model_revision": cfg["model"]["revision"],
        }, path)
        print(f"[extract] wrote {path}")

    selection = {
        "selected": selected,
        "rate_source": rate_source,
        "rates_of_selected": {w: rates[w] for w in selected},
        "stratification": strat_meta,
        "composition_audit": audit,
        "pool_meta": pool_meta,
        "baseline_diag": baseline_diag,
        "concept_ids": {w: concept_ids[w] for w in selected},
        "layers": layers,
        "seed": seed,
        "wall_s": time.time() - t0,
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=1))
    print(f"[done] wrote {args.out / 'selection.json'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
