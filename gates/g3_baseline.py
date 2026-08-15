"""GATE G3: concept vectors and baseline reproduction.

Reads what scripts/01_concept_vectors.py wrote, then measures detection and
identification across the (layer, strength) grid with every control arm.

Emits a report and stops. It judges nothing.

    python gates/g3_baseline.py [--gen-layer 27] [--gen-strength 0.09]

Detection is read from next-token logits, not generation (Macar's prior, ~10x
cheaper). Identification needs text, so it runs at one operating point only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inject  # noqa: E402
import judge as judge_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import stats  # noqa: E402
import vectors as vec_mod  # noqa: E402

RULE = "=" * 78


def fmt(interval: stats.Interval) -> str:
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=ROOT / "artifacts" / "vectors")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4)
    ap.add_argument("--gen-layer", type=int, default=None)
    ap.add_argument("--gen-strength", type=float, default=None,
                    help="default: planned.operating_strength from sprint.yaml")
    ap.add_argument("--gen-samples", type=int, default=4)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layers = cfg["planned"]["layers"]
    strengths = cfg["planned"]["strengths_rel"]
    gen_layer = args.gen_layer if args.gen_layer is not None else layers[0]
    if args.gen_strength is None:
        args.gen_strength = cfg["planned"]["operating_strength"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    selection_path = args.vectors / "selection.json"
    if not selection_path.exists():
        raise SystemExit(
            f"{selection_path} not found -- run scripts/01_concept_vectors.py first")
    selection = json.loads(selection_path.read_text())
    concepts = selection["selected"]

    # ---------------------------------------------------------------- load
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
    load_s = time.time() - t0

    vectors_by_layer, randoms_by_layer = {}, {}
    for layer in layers:
        blob = torch.load(args.vectors / f"vectors_layer{layer}.pt",
                          map_location="cpu", weights_only=False)
        vectors_by_layer[layer] = {w: v.to(device) for w, v in blob["vectors"].items()}
        randoms_by_layer[layer] = vec_mod.matched_random_directions(
            vectors_by_layer[layer], seed + layer)

    yes_ids, no_ids = prompt_mod.boolean_token_ids(tok)
    yes_t = torch.as_tensor(yes_ids, device=device)
    no_t = torch.as_tensor(no_ids, device=device)
    tasks = prompt_mod.TASK_PROMPTS[: args.n_tasks]

    def prepared(messages):
        ids, positions, _ = prompt_mod.prepare(
            model, tok, messages, prefill=True,
            enable_thinking=cfg["planned"]["enable_thinking"])
        return ids, positions

    boolean_mass: list = []

    def detect_scores(ids, positions, layer, source, words, alpha_rel, median_norm):
        """P(true)/(P(true)+P(false)) for each word, batched over one prompt.

        Also accumulates P(true)+P(false). If that sum is not near 1 the
        prefill did not land on a JSON boolean and the whole channel is
        reading the wrong position -- which is how G2's dose-response read
        0.00000 everywhere before the prefill was made forcing.
        """
        out = []
        for begin in range(0, len(words), args.batch):
            chunk = words[begin:begin + args.batch]
            batch = ids.expand(len(chunk), -1).contiguous()
            result = inject.injected_prefill(
                model, batch, layer,
                torch.stack([source[w] for w in chunk]).to(device),
                torch.full((len(chunk),), alpha_rel * median_norm, device=device),
                positions, record_layers=[layer], record_positions=positions)
            probs = torch.softmax(result["logits"], dim=-1)
            p_yes = probs.index_select(-1, yes_t).sum(-1)
            p_no = probs.index_select(-1, no_t).sum(-1)
            boolean_mass.append((p_yes + p_no).cpu().numpy())
            out.append((p_yes / (p_yes + p_no)).cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------ the grid
    norms = {}
    injected = {}   # (layer, strength) -> [n_tasks, n_concepts]
    randomarm = {}
    control = np.zeros(len(tasks))
    for t_idx, task in enumerate(tasks):
        ids, positions = prepared(prompt_mod.detect_messages(task))
        clean = inject.injected_prefill(
            model, ids, layers[0], None, None, positions,
            record_layers=[layers[0]], record_positions=positions)
        probs = torch.softmax(clean["logits"], dim=-1)
        p_yes = probs.index_select(-1, yes_t).sum(-1)
        p_no = probs.index_select(-1, no_t).sum(-1)
        control[t_idx] = float(p_yes / (p_yes + p_no))
        for layer in layers:
            norms.setdefault((layer, t_idx), inject.median_residual_norm(
                model, ids, layer, positions))
            for strength in strengths:
                injected.setdefault((layer, strength),
                                    np.zeros((len(tasks), len(concepts))))
                randomarm.setdefault((layer, strength),
                                     np.zeros((len(tasks), len(concepts))))
                injected[(layer, strength)][t_idx] = detect_scores(
                    ids, positions, layer, vectors_by_layer[layer], concepts,
                    strength, norms[(layer, t_idx)])
                randomarm[(layer, strength)][t_idx] = detect_scores(
                    ids, positions, layer, randoms_by_layer[layer], concepts,
                    strength, norms[(layer, t_idx)])
        print(f"[grid] task {t_idx + 1}/{len(tasks)} done", file=sys.stderr)

    # yes-bias arm: same vectors, unrelated questions whose answer is "no"
    yes_bias = {}
    for q_idx, question in enumerate(prompt_mod.YES_BIAS_QUESTIONS[: args.n_tasks]):
        ids, positions = prepared(prompt_mod.yes_bias_messages(question))
        mn = inject.median_residual_norm(model, ids, gen_layer, positions)
        clean = inject.injected_prefill(
            model, ids, gen_layer, None, None, positions,
            record_layers=[gen_layer], record_positions=positions)
        probs = torch.softmax(clean["logits"], dim=-1)
        base = float(probs.index_select(-1, yes_t).sum(-1)
                     / (probs.index_select(-1, yes_t).sum(-1)
                        + probs.index_select(-1, no_t).sum(-1)))
        scores = detect_scores(ids, positions, gen_layer,
                               vectors_by_layer[gen_layer], concepts,
                               args.gen_strength, mn)
        yes_bias[q_idx] = {"clean": base, "injected": scores}

    # ------------------------------------------- identification, generation
    ids, positions = prepared(prompt_mod.probe_messages())
    gen_norm = inject.median_residual_norm(model, ids, gen_layer, positions)
    identified, guesses, coherence_nll, parse_ok, parse_total = [], [], [], 0, 0
    responses = []
    for begin in range(0, len(concepts), args.batch):
        chunk = concepts[begin:begin + args.batch]
        batch = ids.expand(len(chunk), -1).contiguous()
        vecs = torch.stack([vectors_by_layer[gen_layer][w] for w in chunk]).to(device)
        alpha = torch.full((len(chunk),), args.gen_strength * gen_norm, device=device)
        for sample in range(args.gen_samples):
            gen = inject.generate_with_injection(
                model, hf_model, tok, batch, gen_layer, vecs, alpha, positions,
                max_new_tokens=32, temperature=1.0, seed=seed + sample)
            nll = inject.sequence_nll(model, gen["sequences"], int(ids.shape[1]))
            for k, word in enumerate(chunk):
                text = gen["completions"][k]
                hit = judge_mod.mention_identifies(text, word)
                identified.append((word, hit))
                coherence_nll.append(float(nll[k]))
                responses.append({"concept": word, "sample": sample,
                                  "response": text})
                parse_total += 1
                first = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
                if first:
                    parse_ok += 1
                    if not hit:
                        guesses.append(first[0].lower())
        print(f"[gen] {begin + len(chunk)}/{len(concepts)}", file=sys.stderr)

    wall_s = time.time() - t0

    # --------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G3 : concept vectors and baseline ================")

    w("\nCONFIG")
    w(f"  model @ revision         {cfg['model']['repo']} @ {cfg['model']['revision'][:12]}")
    w(f"  seed / device            {seed} / {device}")
    w(f"  layers x strengths       {layers} x {strengths}")
    w(f"  concepts                 {len(concepts)}")
    w(f"  tasks per condition      {len(tasks)}")
    w(f"  generation point         layer {gen_layer}, alpha_rel {args.gen_strength}, "
      f"{args.gen_samples} samples, T=1.0")
    w(f"  stratification source    {selection['rate_source']}")

    w("\nINVARIANTS")
    all_vecs = torch.stack([vectors_by_layer[gen_layer][c] for c in concepts]).float()
    vec_norms = all_vecs.norm(dim=-1).cpu().numpy()
    dist = stats.median_iqr(vec_norms)
    outl = stats.mad_outliers(vec_norms, threshold=3.0)
    w(f"  concept vector norms      median {dist['median']:.3f}  IQR {dist['iqr']:.3f}"
      f"  min {dist['min']:.3f}  max {dist['max']:.3f}")
    w(f"  norms beyond 3 MAD        {outl['n_outliers']} of {len(concepts)}"
      f"  {[concepts[i] for i in outl['indices'][:6]]}")
    unit = all_vecs / all_vecs.norm(dim=-1, keepdim=True)
    cos = (unit @ unit.T).cpu().numpy()
    off = cos[np.triu_indices(len(concepts), k=1)]
    null_means = []
    for _ in range(200):
        g = rng.normal(size=(len(concepts), all_vecs.shape[1]))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        c = g @ g.T
        null_means.append(c[np.triu_indices(len(concepts), k=1)].mean())
    w(f"  mean pairwise cosine      {off.mean():+.4f}  (sd {off.std():.4f})")
    w(f"  null at d={all_vecs.shape[1]}, same n     "
      f"{np.mean(null_means):+.4f}  (sd {np.std(null_means):.4f})")
    w("  the build spec quotes 0.032 sd 0.281 as the reference, but that is at")
    w(f"  d=5376 (Gemma-3-27B). This model is d={all_vecs.shape[1]}, so the null above is")
    w("  recomputed rather than transferred.")
    audit = selection["composition_audit"]
    w(f"  composition audit, single-token filter: "
      f"{audit['n_before']} -> {audit['n_after']}")
    w(f"    {'bucket':<10}{'before':>8}{'after':>8}{'retention':>11}")
    for bucket in sorted(audit["bucket_before"]):
        w(f"    {bucket:<10}{audit['bucket_before'][bucket]:>8}"
          f"{audit['bucket_after'].get(bucket, 0):>8}"
          f"{audit['bucket_retention'][bucket]:>11.3f}")
    w("    a retention gap between abstract and concrete is the confound that")
    w("    turns tokenisation into a finding; the gap is the number to read.")
    w(f"  parse rate (a word emitted)  "
      f"{fmt(stats.wilson(parse_ok, parse_total))}")
    if boolean_mass:
        bm = stats.median_iqr(np.concatenate(boolean_mass))
        w(f"  P(true)+P(false) at the report position  median {bm['median']:.4f}"
          f"  min {bm['min']:.4f}")
        w("    near 1.0 means the prefill landed on a JSON boolean; well below")
        w("    means the detection channel is reading the wrong position")

    w("\nPRIMARY")
    w("  detection from next-token logits. Two thresholds, because a fixed 0.5")
    w("  cut is uninformative when the whole score distribution sits far below")
    w("  it: the clean control median is "
      f"{float(np.median(control)):.4f}, so TPR@0.5 reads 0.000 even when")
    w("  injection reliably moves the score. TPR@ctrl is the fraction of")
    w("  injected trials above EVERY control trial, which is the same idea as")
    w("  f1's 5% FPR threshold applied to a control set of only "
      f"{control.size}.")
    w("  AUC is threshold-free: P(injected > control), from Cliff's delta.")
    ctrl_threshold = float(np.max(control)) if control.size else float("nan")
    w(f"    control max = {ctrl_threshold:.4f}   (the TPR@ctrl threshold)")
    w(f"    {'layer':>5} {'alpha':>6} {'TPR@0.5':>17} {'TPR@ctrl':>17}"
      f" {'AUC':>7} {'median score':>13}")
    fpr_hits = int((control > 0.5).sum())
    fpr_iv = stats.wilson(fpr_hits, len(control))
    grid_rows = []
    for layer in layers:
        for strength in strengths:
            arr = injected[(layer, strength)].ravel()
            tpr = stats.wilson(int((arr > 0.5).sum()), arr.size)
            tpr_c = stats.wilson(int((arr > ctrl_threshold).sum()), arr.size)
            auc = (stats.cliffs_delta(arr, control) + 1) / 2
            grid_rows.append({"layer": layer, "strength": strength,
                              "tpr": tpr.point, "tpr_ctrl": tpr_c.point,
                              "fpr": fpr_iv.point, "auc": auc,
                              "median_score": float(np.median(arr)),
                              "n": int(arr.size)})
            w(f"    {layer:>5} {strength:>6.1f} {fmt(tpr):>17} {fmt(tpr_c):>17}"
              f" {auc:>7.3f} {float(np.median(arr)):>13.4f}")
    w(f"  zero-strength FPR at 0.5           {fmt(fpr_iv)}  n={control.size}")
    w("  NOTE: a high TPR@ctrl with a low median score means injection moves")
    w("  the detection score reliably but nowhere near a decision boundary.")
    w("  Compare it against the random-direction arm below before reading it")
    w("  as introspection -- if random directions move it too, it is a")
    w("  perturbation response, not concept detection.")
    w(f"  identification (separate from detection), layer {gen_layer} "
      f"alpha_rel {args.gen_strength}")
    id_hits = sum(1 for _, hit in identified if hit)
    w(f"    identification rate      {fmt(stats.wilson(id_hits, len(identified)))}")
    w("    scored by deterministic word-boundary match on singular/plural forms,")
    w("    NOT by the LLM judge -- see CROSS-CHECK for judge availability.")

    w("\nCONTROLS")
    w(f"  zero-strength (FPR)            {fmt(fpr_iv)}  n={len(control)}")
    w(f"    control score median/IQR     "
      f"{stats.median_iqr(control)['median']:.4f} / "
      f"{stats.median_iqr(control)['iqr']:.4f}")
    w("    n is the number of task prompts: with no injection every concept")
    w("    yields identical logits, so trials cannot be multiplied by concept.")
    w("    The interval is wide by construction; the AUC column above is the")
    w("    better-powered comparison.")
    w("  random direction at matched norm")
    w(f"    {'layer':>5} {'alpha':>6} {'rate>0.5':>22} {'AUC vs control':>15}")
    for layer in layers:
        for strength in strengths:
            arr = randomarm[(layer, strength)].ravel()
            w(f"    {layer:>5} {strength:>6.1f} "
              f"{fmt(stats.wilson(int((arr > 0.5).sum()), arr.size)):>22}"
              f" {(stats.cliffs_delta(arr, control) + 1) / 2:>15.3f}")
    w(f"  concept vs random direction at layer {gen_layer}, alpha {args.gen_strength}")
    ci = injected[(gen_layer, args.gen_strength)].ravel() if (gen_layer, args.gen_strength) in injected else None
    cr = randomarm[(gen_layer, args.gen_strength)].ravel() if ci is not None else None
    if ci is not None:
        w(f"    Cliff's delta                {stats.cliffs_delta(ci, cr):+.4f}")
        w(f"    difference in rate           "
          f"{fmt(stats.newcombe_diff(int((ci > 0.5).sum()), ci.size, int((cr > 0.5).sum()), cr.size))}")
    w("  yes-bias arm (the Godet confound): the same vectors injected into")
    w("  unrelated yes/no questions whose truthful answer is 'no'. A rise here")
    w("  is a generic yes-bias from perturbing the residual stream, not")
    w("  introspection, and it discounts the detection numbers above.")
    yb_clean = np.array([v["clean"] for v in yes_bias.values()])
    yb_inj = np.concatenate([v["injected"] for v in yes_bias.values()])
    w(f"    clean P(yes) median          {np.median(yb_clean):.4f}")
    w(f"    injected P(yes) median       {np.median(yb_inj):.4f}")
    w(f"    injected rate > 0.5          "
      f"{fmt(stats.wilson(int((yb_inj > 0.5).sum()), yb_inj.size))}")
    w(f"    shift, Cliff's delta         {stats.cliffs_delta(yb_inj, yb_clean):+.4f}")

    w("\nCROSS-CHECK")
    pilot = json.loads((args.vectors / "pilot.json").read_text()) \
        if (args.vectors / "pilot.json").exists() else None
    if pilot:
        pool_rates = np.array(list(pilot["rates"].values()))
        dip_pool = stats.dip_test(pool_rates, n_boot=10_000, seed=seed)
        w(f"  bimodality, FULL single-token pool (n={pool_rates.size})")
        w(f"    Hartigan dip                 {dip_pool['dip']:.4f}  p={dip_pool['p']:.4f}")
        pd = stats.median_iqr(pool_rates)
        w(f"    mean / median                {pool_rates.mean():.4f} / {pd['median']:.4f}")
        w(f"    fraction >= 0.9              "
          f"{fmt(stats.wilson(int((pool_rates >= 0.9).sum()), pool_rates.size))}")
        w(f"    fraction <= 0.01             "
          f"{fmt(stats.wilson(int((pool_rates <= 0.01).sum()), pool_rates.size))}")
        w("    Macar reports Gemma detection strongly bimodal: 55 concepts >=90%,")
        w("    63 at exactly 0%, mean 38.2%, median 30.0%. That is a rate over")
        w("    generated trials; this is a probability from logits, so the")
        w("    distributions are of different quantities and only their SHAPE")
        w("    is comparable.")
        w("    Measured on the full pool, not the selected 60: the 60 were")
        w("    chosen to span the range, which would manufacture the spread.")
    sel_rates = np.array([selection["rates_of_selected"][w_] for w_ in concepts])
    dip_sel = stats.dip_test(sel_rates, n_boot=10_000, seed=seed)
    w(f"  bimodality, selected 60          dip {dip_sel['dip']:.4f}  p={dip_sel['p']:.4f}"
      f"   (selection-induced; not evidence)")
    w("  cross-model rank correlation vs Macar's Gemma rates")
    w("    UNAVAILABLE. The per-concept rates are not published: the metrics")
    w("    caches are aggregates over (layer_idx, strength, arm), the")
    w("    abliterated checkpoint is weights only, and the README reports")
    w("    aggregate findings. This cross-check cannot be computed. Supply")
    w("    rates via --stratify-file to scripts/01_concept_vectors.py to")
    w("    restore both it and the spec's original stratification.")
    w("  confabulation profile: most frequent wrong identifications")
    for word, count in Counter(guesses).most_common(10):
        w(f"    {word:<20} {count:>4}  "
          f"{count / max(1, len(guesses)):.3f} of wrong identifications")
    apple = sum(v for k, v in Counter(guesses).items() if k.startswith("apple"))
    w(f"    'apple' share                {apple / max(1, len(guesses)):.4f}")
    w("    Lederman & Mahowald report 74.8% of Qwen's wrong identifications")
    w("    guess 'apple' against a 0.003% corpus base rate. Note the upstream")
    w("    concept list already excludes Apples as hallucination-prone, so")
    w("    apple can only appear here as a guess, never as an injected concept.")
    nll_dist = stats.median_iqr(np.array(coherence_nll))
    w(f"  coherence at alpha_rel {args.gen_strength}: clean-scored NLL median "
      f"{nll_dist['median']:.4f} IQR {nll_dist['iqr']:.4f}")
    w("    (G2 carries the full NLL-vs-alpha curve; this is the one point.)")
    rubrics = judge_mod.load_rubrics(ROOT / "configs" / "judge_rubrics.json")
    llm = judge_mod.LLMJudge(rubrics)
    w(f"  judge rubrics loaded             {sorted(rubrics)}")
    w(f"  LLM judge available              {llm.available}")
    labels_path = args.out / "hand_labels.json"
    if labels_path.exists():
        hand = json.loads(labels_path.read_text())
        pairs = [(r, hand[str(i)]) for i, r in enumerate(identified)
                 if str(i) in hand]
        if pairs:
            auto = np.array([int(p[0][1]) for p in pairs])
            manual = np.array([int(p[1]) for p in pairs])
            k = stats.cohens_kappa(auto, manual)
            w(f"  Cohen's kappa vs {len(pairs)} hand labels   {k['kappa']:.4f}"
              f"  observed agreement {k['observed_agreement']:.3f}")
            w(f"    confusion {k['confusion'].tolist()}")
    else:
        w(f"  Cohen's kappa                    PENDING -- no hand labels.")
        w(f"    50 responses written to {args.out / 'to_label.json'};")
        w("    label each 1/0 for 'names the injected concept', save as")
        w(f"    {labels_path.name}, and re-run to get kappa and the confusion matrix.")

    w("\nANOMALIES")
    anomalies = []
    ret = audit["bucket_retention"]
    if "abstract" in ret and "concrete" in ret:
        anomalies.append(
            f"single-token retention differs by bucket: abstract "
            f"{ret['abstract']:.3f} vs concrete {ret['concrete']:.3f}; any "
            f"abstract/concrete detection difference is confounded by this")
    if len(control) < 30:
        anomalies.append(
            f"FPR rests on {len(control)} zero-strength trials; the Wilson "
            f"interval is wide and TPR-FPR inherits that width")
    if selection["stratification"].get("shortfall"):
        anomalies.append(
            f"stratification short by {selection['stratification']['shortfall']} "
            f"concepts: {selection['stratification']}")
    anomalies.append(
        "tiers are a sampling device, never a reporting unit: concepts were "
        "selected on measured detection, so per-tier rates would regress to "
        "the mean by construction. No per-tier number appears in this report.")
    anomalies.append(
        "identification is scored by deterministic string match; the LLM judge "
        f"is {'available but not used for the headline' if llm.available else 'unavailable'}")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  model load                       {load_s:8.1f} s")
    w(f"  detection grid                   "
      f"{len(layers) * len(strengths) * len(tasks) * len(concepts) * 2} trials")
    w(f"  generations                      {len(identified)}")
    w(f"  peak VRAM                        "
      f"{torch.cuda.max_memory_allocated() / 2**30:8.2f} GiB")
    w(f"  gate wall-clock                  {wall_s:8.1f} s")

    w("\nARTIFACTS")
    sample_idx = rng.choice(len(responses), size=min(50, len(responses)),
                            replace=False)
    (args.out / "to_label.json").write_text(json.dumps(
        {str(int(i)): responses[int(i)] for i in sample_idx}, indent=1))
    np.savez(
        args.out / "g3.npz",
        control=control,
        concepts=np.array(concepts),
        vector_norms=vec_norms,
        pairwise_cosine=off,
        **{f"injected_L{l}_a{s}": injected[(l, s)] for l in layers for s in strengths},
        **{f"random_L{l}_a{s}": randomarm[(l, s)] for l in layers for s in strengths},
    )
    (args.out / "g3_grid.json").write_text(json.dumps(grid_rows, indent=1))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g3_report.txt").write_text(report)
    print(report)
    for name in ("g3_report.txt", "g3.npz", "g3_grid.json", "to_label.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
