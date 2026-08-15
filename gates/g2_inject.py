"""GATE G2: injection harness.

The module most likely to be subtly wrong, so this gate spends most of its
budget on invariants: things that must hold exactly for any downstream number
to mean anything. Zero-strength identity and spatial containment should be
exactly 0.0. Batch equivalence will not be, because bf16 reductions are not
order-invariant -- the raw number is reported and a human judges the
magnitude.

Emits a report and stops. It judges nothing.

    python gates/g2_inject.py [--layer 27] [--n-concepts 10] [--batch 8]
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
import stats  # noqa: E402
import vectors as vec_mod  # noqa: E402

RULE = "=" * 78


def fmt(interval: stats.Interval) -> str:
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None, help="default: first planned")
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--inject-width", type=int, default=8)
    ap.add_argument("--inject-tail-offset", type=int, default=4,
                    help="tokens left untouched after the injection window")
    ap.add_argument("--gen-alphas", type=float, nargs="*", default=[0.0, 2.0, 8.0])
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g2")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    seed = cfg["seed"]
    layer = args.layer if args.layer is not None else cfg["planned"]["layers"][0]
    alphas_rel = [0.0] + list(cfg["planned"]["strengths_rel"])
    torch.manual_seed(seed)

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
    load_s = time.time() - t_start

    # ------------------------------------------------------- concepts, prompt
    baseline_words, baseline_diag = vec_mod.load_baseline_words(
        ROOT / "configs" / "baseline_words.json")
    # concepts.json entries are {word, category, bucket}; go through the shared
    # loader and the shared token helper so this gate cannot drift from the
    # rest of the pipeline again.
    pool, _pool_meta = vec_mod.load_concept_pool(ROOT / "configs" / "concepts.json")

    # Round-robin over the source categories rather than taking the first N.
    # The pool is ordered by category, so "first 10 single-token" returned ten
    # animals -- a specificity matrix over ten near-synonyms measures category
    # confusion, not steering specificity.
    by_category: dict[str, list[str]] = {}
    n_multi_token = 0
    concept_ids = {}
    for item in pool:
        ids = vec_mod.single_token_ids(tok, item["word"])
        if not ids:
            n_multi_token += 1
            continue
        concept_ids[item["word"]] = ids
        by_category.setdefault(item["category"], []).append(item["word"])

    concepts: list[str] = []
    rank = 0
    while len(concepts) < args.n_concepts:
        added = False
        for category in sorted(by_category):
            members = by_category[category]
            if rank < len(members) and len(concepts) < args.n_concepts:
                concepts.append(members[rank])
                added = True
        if not added:
            break
        rank += 1
    concept_ids = {w: concept_ids[w] for w in concepts}
    if len(concepts) < 2:
        raise RuntimeError(f"only {len(concepts)} single-token concepts available")

    rendered = prompt_mod.render(tok, prompt_mod.probe_messages(), prefill=True)
    base_ids = model.encode(rendered)
    seq_len = int(base_ids.shape[1])
    stop = seq_len - args.inject_tail_offset
    start = max(0, stop - args.inject_width)
    positions = slice(start, stop)
    if start <= 0 or stop >= seq_len:
        raise RuntimeError(
            f"injection window {start}:{stop} leaves no tokens on both sides "
            f"of a {seq_len}-token prompt; lower --inject-width")

    vecs, _baseline_mean = vec_mod.extract_concept_vectors(
        model, tok, concepts, baseline_words, layer)
    randoms = vec_mod.matched_random_directions(vecs, seed)
    median_norm = inject.median_residual_norm(model, base_ids, layer, positions)
    extract_s = time.time() - t_start - load_s

    def batch_ids(n: int) -> torch.Tensor:
        return base_ids.expand(n, -1).contiguous()

    def stack(words, source) -> torch.Tensor:
        return torch.stack([source[w] for w in words]).to(device)

    # ---------------------------------------------------------- invariants
    clean = inject.injected_prefill(
        model, batch_ids(1), layer, None, None, positions,
        record_layers=[layer], record_positions=slice(None))

    zero_alpha = inject.injected_prefill(
        model, batch_ids(1), layer, stack(concepts[:1], vecs),
        torch.zeros(1, device=device), positions,
        record_layers=[layer], record_positions=slice(None))
    zero_delta = float((clean["logits"] - zero_alpha["logits"]).abs().max())

    # batch equivalence: 4 concepts together vs one at a time
    four = concepts[:4]
    alpha_abs = torch.full((4,), 4.0 * median_norm, device=device)
    batched = inject.injected_prefill(
        model, batch_ids(4), layer, stack(four, vecs), alpha_abs, positions,
        record_layers=[layer], record_positions=slice(None))
    singles = [
        inject.injected_prefill(
            model, batch_ids(1), layer, stack([w], vecs),
            alpha_abs[:1], positions,
            record_layers=[layer], record_positions=slice(None))["logits"]
        for w in four
    ]
    batch_delta = float(
        (batched["logits"] - torch.cat(singles, dim=0)).abs().max())

    # spatial containment
    below = [l for l in (0, max(0, layer - 4), layer - 1) if 0 <= l < layer]
    above = [l for l in (layer + 1, layer + 4, model.n_layers - 1)
             if layer < l < model.n_layers]
    probe_layers = sorted({*below, layer, *above})
    clean_full = inject.injected_prefill(
        model, batch_ids(1), layer, None, None, positions,
        record_layers=probe_layers, record_positions=slice(None))
    inj_full = inject.injected_prefill(
        model, batch_ids(1), layer, stack(concepts[:1], vecs),
        torch.full((1,), 4.0 * median_norm, device=device), positions,
        record_layers=probe_layers, record_positions=slice(None))
    # masks must live on the residuals' device to index them
    outside = torch.ones(seq_len, dtype=torch.bool, device=device)
    outside[start:stop] = False
    after = torch.zeros(seq_len, dtype=torch.bool, device=device)
    after[stop:] = True
    containment = {}
    for l in probe_layers:
        diff = (clean_full["residuals"][l] - inj_full["residuals"][l]).abs()
        containment[l] = {
            "outside_window": float(diff[:, outside, :].max()),
            "inside_window": float(diff[:, start:stop, :].max()),
            "after_window": float(diff[:, after, :].max()) if after.any() else float("nan"),
        }

    # seed stability
    gen_a = inject.generate_with_injection(
        model, hf_model, tok, batch_ids(1), layer, stack(concepts[:1], vecs),
        torch.full((1,), 4.0 * median_norm, device=device), positions,
        max_new_tokens=24, temperature=1.0, seed=seed)
    gen_b = inject.generate_with_injection(
        model, hf_model, tok, batch_ids(1), layer, stack(concepts[:1], vecs),
        torch.full((1,), 4.0 * median_norm, device=device), positions,
        max_new_tokens=24, temperature=1.0, seed=seed)
    seed_stable = bool(torch.equal(gen_a["sequences"], gen_b["sequences"]))

    # ------------------------------------------------------- dose-response
    all_ids = torch.as_tensor(
        sorted({i for w in concepts for i in concept_ids[w]}), device=device)
    id_slot = {int(t): j for j, t in enumerate(all_ids)}

    def probe(source, alpha_rel: float) -> np.ndarray:
        """P(concept token) for every concept, injected one per batch element."""
        out = np.zeros((len(concepts), all_ids.numel()))
        for begin in range(0, len(concepts), args.batch):
            chunk = concepts[begin:begin + args.batch]
            result = inject.injected_prefill(
                model, batch_ids(len(chunk)), layer, stack(chunk, source),
                torch.full((len(chunk),), alpha_rel * median_norm, device=device),
                positions, record_layers=[layer], record_positions=positions)
            probs = torch.softmax(result["logits"], dim=-1)
            out[begin:begin + len(chunk)] = (
                probs.index_select(-1, all_ids).cpu().numpy())
        return out

    def concept_prob(matrix: np.ndarray, row: int, word: str) -> float:
        return float(max(matrix[row, id_slot[i]] for i in concept_ids[word]))

    dose = {a: probe(vecs, a) for a in alphas_rel}
    dose_random = {a: probe(randoms, a) for a in alphas_rel}

    def rho_per_concept(table) -> np.ndarray:
        out = []
        for row, word in enumerate(concepts):
            series = [concept_prob(table[a], row, word) for a in alphas_rel]
            out.append(stats.spearman_ci(np.array(alphas_rel), np.array(series))["rho"])
        return np.array(out, dtype=float)

    rho_concept = rho_per_concept(dose)
    rho_random = rho_per_concept(dose_random)

    # residual norm ratio vs clean, per alpha
    clean_norm = float(
        clean["residuals"][layer][:, start:stop, :].norm(dim=-1).median())
    norm_ratio = {}
    for a in alphas_rel:
        result = inject.injected_prefill(
            model, batch_ids(1), layer, stack(concepts[:1], vecs),
            torch.full((1,), a * median_norm, device=device), positions,
            record_layers=[layer], record_positions=positions)
        norm_ratio[a] = float(
            result["residuals"][layer].norm(dim=-1).median()) / clean_norm

    # ------------------------------------------------------------ coherence
    # Measured on a NEUTRAL task, not on the probe. The probe asks the model to
    # name the injected concept, so any successful steering changes the answer
    # and raises NLL under the clean model -- that metric cannot separate
    # "steering worked" from "brain damage". On a task the injection should not
    # change, rising NLL means degradation and nothing else.
    neutral_text = prompt_mod.render(
        tok, [{"role": "user", "content": prompt_mod.TASK_PROMPTS[0]}], prefill=False)
    neutral_ids = model.encode(neutral_text)
    n_seq = int(neutral_ids.shape[1])
    n_stop = n_seq - args.inject_tail_offset
    neutral_pos = slice(max(0, n_stop - args.inject_width), n_stop)
    neutral_norm = inject.median_residual_norm(model, neutral_ids, layer, neutral_pos)

    n_gen = min(4, len(concepts))
    coherence = {}
    for a in args.gen_alphas:
        gen = inject.generate_with_injection(
            model, hf_model, tok, neutral_ids.expand(n_gen, -1).contiguous(), layer,
            stack(concepts[:n_gen], vecs),
            torch.full((n_gen,), a * neutral_norm, device=device),
            neutral_pos, max_new_tokens=64, temperature=1.0, seed=seed)
        nll = inject.sequence_nll(model, gen["sequences"], n_seq)
        probe_gen = inject.generate_with_injection(
            model, hf_model, tok, batch_ids(n_gen), layer,
            stack(concepts[:n_gen], vecs),
            torch.full((n_gen,), a * median_norm, device=device),
            positions, max_new_tokens=32, temperature=1.0, seed=seed)
        coherence[a] = {
            "nll": stats.median_iqr(nll.cpu().numpy()),
            "lengths": stats.median_iqr(np.array(gen["new_token_counts"], float)),
            "n_fires": gen["n_fires"],
            "sample": gen["completions"][0][:96].replace("\n", " "),
            "probe": probe_gen["completions"][0][:72].replace("\n", " "),
        }

    # ----------------------------------------------------------- throughput
    torch.cuda.reset_peak_memory_stats()
    t_bench = time.time()
    n_bench = 4
    for _ in range(n_bench):
        inject.injected_prefill(
            model, batch_ids(args.batch), layer,
            stack(concepts[:args.batch] if len(concepts) >= args.batch
                  else concepts + concepts[:args.batch - len(concepts)], vecs),
            torch.full((args.batch,), 4.0 * median_norm, device=device), positions,
            record_layers=[layer], record_positions=positions)
    bench_s = (time.time() - t_bench) / n_bench
    peak_bytes = torch.cuda.max_memory_allocated()
    grid = (cfg["planned"]["n_concepts"] * len(cfg["planned"]["layers"])
            * len(cfg["planned"]["strengths_rel"]) * len(cfg["planned"]["orders"])
            * len(cfg["planned"]["conditions"]))
    wall_s = time.time() - t_start

    # --------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G2 : injection harness ================")

    w("\nCONFIG")
    w(f"  model @ revision         {cfg['model']['repo']} @ {cfg['model']['revision'][:12]}")
    w(f"  seed / device / dtype    {seed} / {device} / {torch.bfloat16}")
    w(f"  injection layer          {layer}  (block type: "
      f"{cfg['planned'].get('layer_block_type', 'unspecified')})")
    w(f"  prompt tokens            {seq_len}")
    w(f"  injection window         [{start}, {stop})  width {stop - start}, "
      f"{seq_len - stop} tokens after")
    w(f"  median clean residual L2 {median_norm:.4f}  (alpha_abs = alpha_rel x this)")
    w(f"  alpha_rel ladder         {alphas_rel}")
    w(f"  concepts                 {len(concepts)}  {concepts}")
    w(f"  baseline words           {baseline_diag['n']} "
      f"({baseline_diag['n_duplicated']} duplicated) from {baseline_diag['source']}")
    w(f"  concepts dropped, multi-token  {n_multi_token} of {len(pool)}")

    w("\nINVARIANTS")
    w(f"  zero-strength identity, max |logit delta|   {zero_delta:.6e}")
    w("    (alpha=0 vs no hook registered at all; adding 0*v is exact, so any")
    w("     nonzero value here means the hook is doing something else too)")
    w(f"  batch equivalence B=4 vs 4x B=1, max |d|    {batch_delta:.6e}")
    w("    (bf16 reductions are not order-invariant, so this is not expected")
    w("     to be 0.0; the magnitude is the thing to read)")
    w(f"  hook handles after teardown                 "
      f"{clean['n_handles_after']} / {zero_alpha['n_handles_after']} / "
      f"{gen_a['n_handles_after']}  (prefill clean / injected / generate)")
    w(f"  hook fire count, prefill                    {inj_full['n_fires']}  (expect 1)")
    w(f"  hook fire count, across generate            {gen_a['n_fires']}  (expect 1:")
    w("     prefill only, never on a cached decode step)")
    w(f"  seed stability, identical generation        {seed_stable}")
    w("")
    w("  spatial containment: max |residual delta| by layer")
    w(f"    {'layer':>5}  {'inside window':>14}  {'outside window':>15}"
      f"  {'after window':>13}")
    for l in probe_layers:
        c = containment[l]
        marker = "  <- injected here" if l == layer else ""
        w(f"    {l:>5}  {c['inside_window']:>14.6e}  {c['outside_window']:>15.6e}"
          f"  {c['after_window']:>13.6e}{marker}")
    w("    at the injection layer and below, 'outside window' must be 0.0.")
    w("    above it, leakage to later positions is expected and is reported")
    w("    rather than asserted -- 48 of 64 blocks are linear-attention with a")
    w("    causal conv (kernel 4), so information moves forward in position.")

    w("\nPRIMARY")
    w("  dose-response: P(concept token) at the report position vs alpha_rel")
    w(f"    {'concept':<16} " + " ".join(f"{'a=' + str(a):>9}" for a in alphas_rel)
      + f"  {'spearman':>9}")
    for row, word in enumerate(concepts):
        series = [concept_prob(dose[a], row, word) for a in alphas_rel]
        w(f"    {word:<16} " + " ".join(f"{p:>9.5f}" for p in series)
          + f"  {rho_concept[row]:>+9.3f}")
    w(f"  spearman rho across concepts       "
      f"median {np.nanmedian(rho_concept):+.3f}  "
      f"IQR {stats.median_iqr(rho_concept)['iqr']:.3f}")
    non_monotonic = [concepts[i] for i in range(len(concepts))
                     if not (rho_concept[i] > 0.9)]
    w(f"  concepts with rho <= 0.9           {non_monotonic}")
    w("")
    w("  residual norm ratio vs clean, at the injection positions.")
    w("  The contract is ||delta|| = alpha_rel x median residual norm, so with")
    w("  delta roughly orthogonal to h the ratio should track sqrt(1+alpha^2).")
    w("  A ratio far above that means the steering direction is not unit-norm")
    w("  and the whole ladder is mis-scaled.")
    w(f"    {'alpha_rel':>10}  {'ratio':>8}  {'expected':>9}  {'obs/exp':>8}")
    for a in alphas_rel:
        expected = float(np.sqrt(1.0 + a * a))
        w(f"    {a:>10.1f}  {norm_ratio[a]:>8.4f}  {expected:>9.4f}"
          f"  {norm_ratio[a] / expected:>8.2f}")

    w("\nCONTROLS")
    w("  random direction at matched norm: the same ladder, same positions,")
    w("  a direction with no concept in it. This distribution should sit on 0.")
    w(f"    spearman rho, random directions   median {np.nanmedian(rho_random):+.3f}"
      f"  IQR {stats.median_iqr(rho_random)['iqr']:.3f}")
    w(f"    concept rho vs random rho, Cliff's delta  "
      f"{stats.cliffs_delta(rho_concept, rho_random):+.4f}")
    w(f"    bootstrap CI, mean concept rho    "
      f"{fmt(stats.bootstrap_ci(rho_concept, seed=seed))}")
    w(f"    bootstrap CI, mean random rho     "
      f"{fmt(stats.bootstrap_ci(rho_random, seed=seed))}")
    w("")
    w("  steering specificity at alpha_rel=4: inject the row concept, read")
    w("  P(column concept). Diagonal dominance is the property; off-diagonal")
    w("  structure says which concepts are entangled.")
    matrix = dose[4.0] if 4.0 in dose else dose[alphas_rel[-1]]
    header = "".join(f"{c[:6]:>8}" for c in concepts)
    # kept out of the f-string: backslashes inside replacement fields are a
    # syntax error before Python 3.12, and Colab is not guaranteed to be newer
    corner = "inject \\ read"
    w(f"    {corner:<16}{header}")
    confusion = np.zeros((len(concepts), len(concepts)))
    for row, word in enumerate(concepts):
        for col, other in enumerate(concepts):
            confusion[row, col] = concept_prob(matrix, row, other)
        w(f"    {word:<16}" + "".join(f"{v:>8.4f}" for v in confusion[row]))
    diag_wins = int(sum(confusion[i].argmax() == i for i in range(len(concepts))))
    w(f"    rows whose max is on the diagonal  {diag_wins} of {len(concepts)}"
      f"   {fmt(stats.wilson(diag_wins, len(concepts)))}")

    w("\nCROSS-CHECK")
    w(f"  coherence onset on a NEUTRAL task ({prompt_mod.TASK_PROMPTS[0]!r}),")
    w("  scored by the UNHOOKED model. The probe prompt is unusable for this:")
    w("  it asks the model to name the injected concept, so steering that works")
    w("  raises NLL by definition. On a task injection should not change,")
    w("  rising NLL is degradation and nothing else.")
    w(f"    {'alpha_rel':>10}  {'NLL median':>11}  {'NLL IQR':>9}"
      f"  {'len median':>11}  {'len IQR':>8}  {'fires':>5}")
    for a in args.gen_alphas:
        c = coherence[a]
        w(f"    {a:>10.1f}  {c['nll']['median']:>11.4f}  {c['nll']['iqr']:>9.4f}"
          f"  {c['lengths']['median']:>11.1f}  {c['lengths']['iqr']:>8.1f}"
          f"  {c['n_fires']:>5}")
    for a in args.gen_alphas:
        w(f"    alpha={a:<5.1f} neutral: {coherence[a]['sample']!r}")
        w(f"    alpha={a:<5.1f} probe  : {coherence[a]['probe']!r}")

    w("\nANOMALIES")
    anomalies = []
    if zero_delta != 0.0:
        anomalies.append(f"zero-strength identity is {zero_delta:.3e}, not exactly 0")
    for l in probe_layers:
        if l <= layer and containment[l]["outside_window"] != 0.0:
            anomalies.append(
                f"layer {l} <= injection layer has nonzero delta outside the "
                f"window: {containment[l]['outside_window']:.3e}")
    if inj_full["n_fires"] != 1 or gen_a["n_fires"] != 1:
        anomalies.append(
            f"hook fired {inj_full['n_fires']} times in prefill and "
            f"{gen_a['n_fires']} times across generate; expected 1 and 1")
    if not seed_stable:
        anomalies.append("same seed produced different generations")
    anomalies.append(
        "concept vectors here come from the paper's 50-word list; Phase 3 "
        "replaces this with the stratified 60 and audits its composition")
    anomalies.append(
        f"the probe prompt is ours, not Garcia's protocol -- it prefills an "
        f"assistant turn so the readout is one next-token distribution. The "
        f"sweep uses the vendored two-order protocol instead.")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  model load                         {load_s:8.1f} s")
    w(f"  vector extraction ({len(baseline_words)} baselines) {extract_s:8.1f} s")
    w(f"  peak VRAM at batch {args.batch:<3}             "
      f"{peak_bytes / 2**30:8.2f} GiB")
    w(f"  prefill, batch {args.batch}                   {bench_s:8.3f} s")
    w(f"  throughput                         "
      f"{args.batch / bench_s:8.2f} injected passes/s")
    w(f"  planned sweep grid                 {grid} prefill passes")
    w(f"  extrapolated sweep wall-clock      "
      f"{grid / (args.batch / bench_s) / 3600:8.2f} h")
    w(f"  gate wall-clock                    {wall_s:8.1f} s")

    w("\nARTIFACTS")
    np.savez(
        args.out / "g2.npz",
        alphas_rel=np.array(alphas_rel),
        rho_concept=rho_concept,
        rho_random=rho_random,
        confusion=confusion,
        norm_ratio=np.array([norm_ratio[a] for a in alphas_rel]),
        dose=np.stack([dose[a] for a in alphas_rel]),
        dose_random=np.stack([dose_random[a] for a in alphas_rel]),
    )
    (args.out / "g2_meta.json").write_text(json.dumps({
        "layer": layer, "concepts": concepts, "median_norm": median_norm,
        "window": [start, stop], "seq_len": seq_len,
        "zero_delta": zero_delta, "batch_delta": batch_delta,
        "containment": containment, "seed_stable": seed_stable,
        "throughput_per_s": args.batch / bench_s,
    }, indent=2))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g2_report.txt").write_text(report)
    print(report)
    for name in ("g2_report.txt", "g2.npz", "g2_meta.json"):
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
