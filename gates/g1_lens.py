"""GATE G1: lens validation.

The most consequential gate in the project. A lens that loads but does not
work produces a plausible f2 that is noise, and nothing downstream would
notice. So this compares three lenses -- Jacobian, vanilla logit, and a
random-rotation null -- on the published eval sets, and reports ranks rather
than only hits.

Emits a report and stops. It judges nothing.

    python gates/g1_lens.py [--per-category 20] [--k 10] [--out artifacts/g1]

Three properties of the published eval sets shape what this gate can measure
(data/evaluations/README.md, plus the files themselves):

  - `intermediates` is what gets scored, and an item may have several. The
    published pass@k is the mean over items of the FRACTION of that item's
    intermediates that hit -- not the best one. Every multilingual item has
    several ("Spanish", "opposite", "big", "small"), so scoring the best would
    inflate the number and stop it comparing to anything.
  - `target` exists only for multihop and multilingual, where it fixes the
    readout position and is not itself scored. association and typo have no
    `target` at all, so the build spec's "filter to questions the model
    answers correctly" is undefined for them and is reported as inapplicable
    rather than silently skipped.
  - The published pass@k takes the MIN over layers. The build spec asks for
    pass@k *by layer*. Both are reported: the by-layer curve is the
    diagnostic, the min-over-layers scalar is what compares to the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lens as lens_mod  # noqa: E402
import stats  # noqa: E402

RULE = "=" * 78
RAW = "https://raw.githubusercontent.com/anthropics/jacobian-lens"


def fmt(interval: stats.Interval) -> str:
    """Compact interval, for cells inside a table."""
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def fetch_eval_sets(data_dir: Path, commit: str) -> None:
    """Pull the published eval sets at the pinned commit if absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for category in lens_mod.FINAL_TOKEN_CATEGORIES:
        path = data_dir / f"lens-eval-{category}.json"
        if path.exists():
            continue
        url = f"{RAW}/{commit}/data/evaluations/lens-eval-{category}.json"
        print(f"[fetch] {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())


def per_item_fraction(best: np.ndarray, item_of_word: np.ndarray,
                      n_items: int, k: int) -> np.ndarray:
    """Fraction of each item's intermediates whose rank is <= k."""
    out = np.zeros(n_items, dtype=float)
    for i in range(n_items):
        rows = item_of_word == i
        out[i] = float(np.mean(best[rows] <= k)) if rows.any() else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--k", type=int, default=10, help="top-k for first-appearance")
    ap.add_argument("--n-random-tokens", type=int, default=200,
                    help="prompt-irrelevant tokens for the uniformity calibration")
    ap.add_argument("--spearman-vocab", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- load
    import jlens
    import transformers
    from transformers import AutoTokenizer

    fetch_eval_sets(ROOT / "vendor" / "evaluations", cfg["code"]["jlens"]["commit"])
    items = lens_mod.load_eval_items(ROOT / "vendor" / "evaluations")

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
    lens = lens_mod.load_lens(
        cfg["lens"]["repo"], cfg["lens"]["filename"], cfg["lens"]["revision"],
        device=device)
    layers = list(lens.source_layers)
    n_layers = len(layers)
    rotation = lens_mod.random_rotation(lens.d_model, seed, device)
    special_ids = set(tok.all_special_ids)
    vocab_size = int(model._lm_head.weight.shape[0])
    load_s = time.time() - t_start

    # ------------------------------------------------------------- filter
    n_raw = len(items)
    resolved, dropped_no_word, dropped_words = [], 0, 0
    for item in items:
        scored = []
        for word in item["intermediates"]:
            ids = lens_mod.single_token_ids(tok, word)
            if ids:
                scored.append((word, ids))
            else:
                dropped_words += 1
        if not scored:
            dropped_no_word += 1
            continue
        target = item.get("target")
        resolved.append({
            **item,
            "scored": scored,
            "target_ids": lens_mod.single_token_ids(tok, target) if target else [],
        })

    # competence arm: only defined where the item has a target to check
    kept, wrong_answer, no_target = [], 0, 0
    for item in resolved:
        if not item["target_ids"]:
            no_target += 1
            kept.append(item)
            continue
        ids = model.encode(item["prompt"])
        with torch.inference_mode():
            generated = hf_model.generate(
                input_ids=ids, max_new_tokens=8, do_sample=False,
                pad_token_id=tok.eos_token_id)
        continuation = tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
        if item["target"].strip().lower() in continuation.strip().lower():
            kept.append(item)
        else:
            wrong_answer += 1

    by_category: dict[str, list] = {}
    for item in kept:
        by_category.setdefault(item["category"], []).append(item)
    evaluated = []
    for category in lens_mod.FINAL_TOKEN_CATEGORIES:
        evaluated.extend(by_category.get(category, [])[: args.per_category])

    if not evaluated:
        print(f"{RULE}\n================ GATE G1 : lens validation ================")
        print("\nINVARIANTS")
        print(f"  eval items published               {n_raw}")
        print(f"  dropped, no single-token word      {dropped_no_word}")
        print(f"  dropped, model answers wrongly     {wrong_answer}")
        print("  nothing survived both filters, so no lens comparison was run")
        print(f"{RULE}")
        return

    # ------------------------------------------------------------- measure
    modes = ("jacobian", "logit", "null")
    item_of_word = np.array(
        [i for i, it in enumerate(evaluated) for _ in it["scored"]], dtype=int)
    n_words = item_of_word.size
    word_ranks = {m: np.full((n_words, n_layers), np.nan) for m in modes}
    target_ranks = np.full((len(evaluated), n_layers), np.nan)
    shuffled_word_ranks = np.full((n_words, n_layers), np.nan)
    trash_hits = np.zeros(n_layers)
    trash_total = np.zeros(n_layers)
    spearman_by_layer = np.full((len(evaluated), n_layers), np.nan)
    random_token_ranks: list[np.ndarray] = []
    n_nan_readouts = 0
    trash_cache: dict[int, bool] = {}
    vocab_sample = torch.as_tensor(
        rng.choice(vocab_size, size=args.spearman_vocab, replace=False), device=device)

    word_cursor = 0
    for i, item in enumerate(evaluated):
        input_ids = model.encode(item["prompt"])
        seq_len = input_ids.shape[1]
        start = word_cursor
        word_cursor += len(item["scored"])

        with torch.inference_mode():
            residuals = lens_mod.residuals_at(model, input_ids, layers, position=-1)
            shuffled_pos = int(rng.integers(0, max(1, seq_len - 1)))
            shuffled = lens_mod.residuals_at(
                model, input_ids, layers, position=shuffled_pos)
            rand_ids = torch.as_tensor(
                rng.choice(vocab_size, size=args.n_random_tokens, replace=False),
                device=device)

            logits_by_mode = {}
            for mode in modes:
                logits = model.unembed(torch.cat([
                    lens_mod.transport(lens, residuals[l], l, mode, rotation)
                    for l in layers
                ], dim=0)).float()  # [n_layers, vocab]
                logits_by_mode[mode] = logits
                n_nan_readouts += int(torch.isnan(logits).sum())

                # one row per scored word: best rank over that word's variants
                for offset, (_word, ids) in enumerate(item["scored"]):
                    word_ranks[mode][start + offset] = (
                        lens_mod.ranks_of(logits, torch.as_tensor(ids, device=device))
                        .min(dim=-1).values.cpu().numpy()
                    )

                if mode == "jacobian":
                    if item["target_ids"]:
                        target_ranks[i] = (
                            lens_mod.ranks_of(
                                logits,
                                torch.as_tensor(item["target_ids"], device=device))
                            .min(dim=-1).values.cpu().numpy()
                        )
                    top = logits.topk(args.k, dim=-1).indices.cpu().numpy()
                    for layer_idx in range(n_layers):
                        for token_id in top[layer_idx]:
                            token_id = int(token_id)
                            if token_id not in trash_cache:
                                trash_cache[token_id] = lens_mod.is_trash_token(
                                    tok, token_id, special_ids)
                            trash_hits[layer_idx] += trash_cache[token_id]
                            trash_total[layer_idx] += 1
                    random_token_ranks.append(
                        lens_mod.ranks_of(logits, rand_ids).cpu().numpy().ravel())

            # shuffled-position arm, Jacobian lens only
            shuffled_logits = model.unembed(torch.cat([
                lens_mod.transport(lens, shuffled[l], l, "jacobian", rotation)
                for l in layers
            ], dim=0)).float()
            for offset, (_word, ids) in enumerate(item["scored"]):
                shuffled_word_ranks[start + offset] = (
                    lens_mod.ranks_of(
                        shuffled_logits, torch.as_tensor(ids, device=device))
                    .min(dim=-1).values.cpu().numpy()
                )

            a = logits_by_mode["jacobian"].index_select(-1, vocab_sample).cpu().numpy()
            b = logits_by_mode["logit"].index_select(-1, vocab_sample).cpu().numpy()
            for layer_idx in range(n_layers):
                spearman_by_layer[i, layer_idx] = stats.spearman_ci(
                    a[layer_idx], b[layer_idx])["rho"]

            del logits_by_mode, shuffled_logits
        torch.cuda.empty_cache()

    wall_s = time.time() - t_start
    n_items = len(evaluated)
    best = {m: np.nanmin(word_ranks[m], axis=1) for m in modes}
    best_shuffled = np.nanmin(shuffled_word_ranks, axis=1)

    # -------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G1 : lens validation ================")

    w("\nCONFIG")
    w(f"  model @ revision         {cfg['model']['repo']} @ {cfg['model']['revision'][:12]}")
    w(f"  lens @ revision          {cfg['lens']['repo']} @ {cfg['lens']['revision'][:12]}")
    w(f"  lens file                {cfg['lens']['filename']}")
    w(f"  eval sets @ commit       jacobian-lens @ {cfg['code']['jlens']['commit'][:12]}")
    w(f"  seed / device / dtype    {seed} / {device} / {torch.bfloat16}")
    w(f"  fitted layers            {n_layers}  [{layers[0]}..{layers[-1]}]")
    w(f"  vocabulary               {vocab_size}")
    w(f"  null lens                Q @ J_l, Q orthogonal, seed {seed}")

    w("\nINVARIANTS")
    w(f"  eval items published               {n_raw}")
    w(f"  intermediate words with no single-token variant  {dropped_words}")
    w(f"  items dropped, no scorable word    {dropped_no_word}")
    w(f"  items with no target (filter n/a)  {no_target}")
    w(f"  items dropped, model answers wrongly  {wrong_answer}")
    w(f"  items retained                     {len(kept)}")
    w(f"  items evaluated (<= {args.per_category}/category)      {n_items}")
    w(f"  scored intermediate words          {n_words}")
    for category in lens_mod.FINAL_TOKEN_CATEGORIES:
        rows = [i for i, it in enumerate(evaluated) if it["category"] == category]
        n_w = int((np.isin(item_of_word, rows)).sum())
        w(f"    {category:<14} items {len(rows):>3}   scored words {n_w:>4}"
          f"   pool {len(by_category.get(category, [])):>3}")
    w(f"  NaN entries in any readout         {n_nan_readouts}")
    w("  readout position                   last prompt token (index -1)")

    w("\nPRIMARY")
    w("  Three lenses on identical residuals. The null is the placebo: if it")
    w("  scores near the Jacobian lens, the readout is coming through W_U")
    w("  alone and the lens is contributing nothing.")
    w("")
    w("  published convention: min over layers, mean over items of the")
    w("  fraction of that item's intermediates that hit. CI is a bootstrap")
    w("  over items (a mean of per-item fractions is not a raw proportion, so")
    w("  Wilson does not apply); the word-level proportion below is, and")
    w("  carries Wilson.")
    w(f"    {'lens':<10} {'pass@1':>22} {'pass@10':>22} {'pass@25':>22}")
    for mode in modes:
        cells = []
        for k in (1, 10, 25):
            frac = per_item_fraction(best[mode], item_of_word, n_items, k)
            cells.append(fmt(stats.bootstrap_ci(frac, n_boot=10_000, seed=seed)))
        w(f"    {mode:<10} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    w("")
    w(f"    {'lens':<10} {'word pass@1':>22} {'word pass@10':>22} {'word pass@25':>22}")
    for mode in modes:
        cells = [fmt(stats.wilson(int((best[mode] <= k).sum()), n_words))
                 for k in (1, 10, 25)]
        w(f"    {mode:<10} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    w("")
    w("  by layer (the build spec's diagnostic): word-level pass@1 and median")
    w("  rank of the scored intermediates")
    w(f"    {'layer':>5} " + " ".join(
        f"{m[:4] + '_p@1':>9} {m[:4] + '_med':>10}" for m in modes))
    for layer_idx, layer in enumerate(layers):
        cells = []
        for mode in modes:
            column = word_ranks[mode][:, layer_idx]
            cells.append(f"{np.mean(column <= 1):>9.3f} {np.median(column):>10.0f}")
        w(f"    {layer:>5} " + " ".join(cells))
    w("")
    w("  rank distribution over scored words, pooled across layers")
    for mode in modes:
        dist = stats.median_iqr(word_ranks[mode].ravel())
        w(f"    {mode:<10} median {dist['median']:>10.1f}  IQR {dist['iqr']:>10.1f}"
          f"  min {dist['min']:>7.0f}  max {dist['max']:>9.0f}")
    w("  mean reciprocal rank (min over layers, per scored word)")
    for mode in modes:
        w(f"    {mode:<10} "
          f"{fmt(stats.bootstrap_ci(1.0 / best[mode], n_boot=10_000, seed=seed))}")
    w(f"  Cliff's delta, J vs logit rank     "
      f"{stats.cliffs_delta(word_ranks['jacobian'].ravel(), word_ranks['logit'].ravel()):+.4f}")
    w(f"  Cliff's delta, J vs null rank      "
      f"{stats.cliffs_delta(word_ranks['jacobian'].ravel(), word_ranks['null'].ravel()):+.4f}")
    w("    (negative = J-lens ranks are smaller, i.e. better)")

    w("\nCONTROLS")
    w(f"  random-rotation null, word pass@1  "
      f"{fmt(stats.wilson(int((best['null'] <= 1).sum()), n_words))}")
    w(f"  shuffled-position, word pass@1     "
      f"{fmt(stats.wilson(int((best_shuffled <= 1).sum()), n_words))}")
    shuffled_dist = stats.median_iqr(shuffled_word_ranks.ravel())
    w(f"  shuffled-position rank median/IQR  "
      f"{shuffled_dist['median']:.0f} / {shuffled_dist['iqr']:.0f}")
    w(f"  Cliff's delta, true vs shuffled position  "
      f"{stats.cliffs_delta(word_ranks['jacobian'].ravel(), shuffled_word_ranks.ravel()):+.4f}")
    w("  random-token calibration: ranks of prompt-irrelevant tokens should be")
    w("  approximately uniform over the vocabulary. Systematic deviation means")
    w("  a dominant-direction pathology.")
    pooled_random = (np.concatenate(random_token_ranks)
                     if random_token_ranks else np.array([]))
    ks = stats.ks_uniform(pooled_random, lo=1.0, hi=float(vocab_size))
    w(f"    KS vs uniform                    D={ks['D']:.4f}  p={ks['p']:.3e}  n={ks['n']}")
    w("    (n pools items x layers x probe tokens, so p is near zero for any")
    w("     deviation at all; D is the quantity to read here, not p)")
    rand_dist = stats.median_iqr(pooled_random)
    w(f"    median rank / expected           "
      f"{rand_dist['median']:.0f} / {vocab_size / 2:.0f}")

    w("\nCROSS-CHECK")
    w("  J-lens vs logit-lens agreement by layer (Spearman over a fixed")
    w(f"  {args.spearman_vocab}-token vocabulary subsample). The two should")
    w("  converge near the deepest fitted layer.")
    trash_rate = np.divide(trash_hits, np.maximum(trash_total, 1))
    w(f"    {'layer':>5}  {'rho median':>11}  {'trash@' + str(args.k):>9}"
      f"  {'J med rank':>11}  {'null med rank':>13}")
    for layer_idx, layer in enumerate(layers):
        w(f"    {layer:>5}  {np.nanmedian(spearman_by_layer[:, layer_idx]):>11.4f}"
          f"  {trash_rate[layer_idx]:>9.3f}"
          f"  {np.median(word_ranks['jacobian'][:, layer_idx]):>11.0f}"
          f"  {np.median(word_ranks['null'][:, layer_idx]):>13.0f}")
    w(f"  rho at deepest fitted layer        "
      f"{np.nanmedian(spearman_by_layer[:, -1]):.4f}")

    w("")
    w("  depth ordering on multihop: the intermediate entity should surface")
    w("  earlier in depth than the final answer. Anthropic report the")
    w("  intermediate taking effect ~17% earlier. First appearance = shallowest")
    w(f"  fitted layer whose top-{args.k} contains the token.")
    w("  NOTE: this is the one place `target` is scored. The published")
    w("  convention uses it only to fix the readout position, so this number")
    w("  is not comparable to a published pass@k.")
    multihop_rows = [i for i, it in enumerate(evaluated)
                     if it["category"] == "multihop" and it["target_ids"]]
    if multihop_rows:
        def first_layer(matrix: np.ndarray, row: int) -> float:
            hit = np.flatnonzero(matrix[row] <= args.k)
            return float(layers[hit[0]]) if hit.size else np.nan

        inter_first, target_first = [], []
        for i in multihop_rows:
            words = np.flatnonzero(item_of_word == i)
            per_word = [first_layer(word_ranks["jacobian"], r) for r in words]
            per_word = [v for v in per_word if np.isfinite(v)]
            inter_first.append(min(per_word) if per_word else np.nan)
            target_first.append(first_layer(target_ranks, i))
        inter_first = np.array(inter_first)
        target_first = np.array(target_first)
        both = np.isfinite(inter_first) & np.isfinite(target_first)
        w(f"    items with both surfacing        {int(both.sum())} of {len(multihop_rows)}")
        if both.sum():
            gap = target_first[both] - inter_first[both]
            w(f"    median layer, intermediate       {np.median(inter_first[both]):.1f}")
            w(f"    median layer, final answer       {np.median(target_first[both]):.1f}")
            w(f"    median gap (target - inter)      "
              f"{fmt(stats.bootstrap_ci(gap, statistic=np.median, seed=seed))}")
            w(f"    gap as fraction of model depth   "
              f"{np.median(gap) / model.n_layers:+.4f}   (reference ~0.17)")
    else:
        w("    no multihop items with a target survived the filters")

    w("")
    w("  per-category breakdown, word-level pass@1 (min over layers)")
    for category in lens_mod.FINAL_TOKEN_CATEGORIES:
        rows = [i for i, it in enumerate(evaluated) if it["category"] == category]
        words = np.flatnonzero(np.isin(item_of_word, rows))
        if words.size == 0:
            w(f"    {category:<14} no items")
            continue
        cells = [f"{m[:4]}={fmt(stats.wilson(int((best[m][words] <= 1).sum()), words.size))}"
                 for m in modes]
        w(f"    {category:<14} n={words.size:<4} " + "  ".join(cells))

    w("\nANOMALIES")
    anomalies = []
    worse = [layers[i] for i in range(n_layers)
             if np.median(word_ranks["jacobian"][:, i])
             >= np.median(word_ranks["null"][:, i])]
    if worse:
        anomalies.append(f"J-lens median rank not better than null at layers {worse}")
    if n_nan_readouts:
        anomalies.append(f"{n_nan_readouts} NaN entries in readouts")
    if dropped_words:
        anomalies.append(
            f"{dropped_words} intermediate words had no single-token variant and "
            f"were dropped; multi-token words are not scorable by this metric")
    anomalies.append(
        "association and typo carry no `target`, so the competence filter is "
        f"undefined for them; {no_target} items entered unfiltered")
    anomalies.append(
        "poetry and order-ops eval sets exist but are excluded: poetry reads at "
        "the last newline rather than the final token, and order-ops scores "
        "intermediates as synonym sets")
    anomalies.append(
        "lens was fitted on WikiText and is applied here to eval prompts; the "
        "sprint's chat-formatted injection prompts are a further shift")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  model + lens load                  {load_s:6.1f} s")
    w(f"  peak VRAM                          "
      f"{torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB")
    w(f"  items x layers x lenses            {n_items} x {n_layers} x {len(modes)}")
    w(f"  gate wall-clock                    {wall_s:6.1f} s")

    w("\nARTIFACTS")
    np.savez(
        args.out / "g1_ranks.npz",
        layers=np.array(layers),
        item_of_word=item_of_word,
        **{f"word_ranks_{m}": word_ranks[m] for m in modes},
        target_ranks=target_ranks,
        shuffled_word_ranks=shuffled_word_ranks,
        trash_rate=trash_rate,
        spearman_by_layer=spearman_by_layer,
        random_token_ranks=pooled_random,
    )
    (args.out / "g1_items.json").write_text(json.dumps(
        [{"name": it["name"], "category": it["category"],
          "prompt": it["prompt"], "target": it.get("target"),
          "scored_words": [wd for wd, _ in it["scored"]]} for it in evaluated],
        indent=2, ensure_ascii=False))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g1_report.txt").write_text(report)
    print(report)
    for name in ("g1_report.txt", "g1_ranks.npz", "g1_items.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
