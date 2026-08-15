"""GATE G2b: the band, both arms, and the operating point.

The 30-minute gate that stands between two invalid ladders and the sweep. Run
1 injected raw vectors at 11-88x the residual norm (overwrite, not steering).
Run 2 fixed that and then injected Garcia's PER-LAYER strengths at a SINGLE
layer, ~17x too weak in cumulative displacement. Neither run tested the regime
between them, which is the regime Garcia actually operates in, so every
downstream number so far describes a misconfiguration.

This gate measures that regime directly:

  * dose-response  P(target token) and its rank vs alpha_rel, per arm
  * coherence      NLL on a NEUTRAL task vs alpha_rel, scored by the unhooked
                   model -- where steering becomes damage
  * specificity    inject row concept, read column concept
  * displacement   realised ||delta_l|| / base_l at every band layer, and the
                   band-vs-single-layer ratio that motivated the whole change
  * arm B check    cosine between the analytic J-lens row and the same
                   direction obtained by differentiating the readout

A HUMAN reads this and picks the ladder and the operating point, then writes
`planned.operating_strength` back into configs/sprint.yaml. Nothing downstream
runs until they do -- src/config.py refuses to guess.

Emits a report and stops. It judges nothing.

    python gates/g2b_band.py [--n-concepts 10] [--batch 8]
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

import band_inject  # noqa: E402
import config as cfg_mod  # noqa: E402
import inject  # noqa: E402
import lens as lens_mod  # noqa: E402
import prompts as prompt_mod  # noqa: E402
import stats  # noqa: E402
import vectors as vec_mod  # noqa: E402

RULE = "=" * 78


def fmt(interval: stats.Interval) -> str:
    return f"{interval.point:.3f} [{interval.lo:.3f},{interval.hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--strengths", type=float, nargs="*", default=None,
                    help="default: strengths_rel + strengths_extension")
    ap.add_argument("--gen-alphas", type=float, nargs="*", default=None,
                    help="alphas to generate at for coherence; default the "
                         "lowest, middle and highest rung")
    ap.add_argument("--specificity-alpha", type=float, default=None,
                    help="alpha for the confusion matrix; default the "
                         "highest rung of the base ladder")
    ap.add_argument("--check-layers", type=int, nargs="*", default=None,
                    help="band layers for the J-lens row cosine check; "
                         "default first, middle, last")
    ap.add_argument("--check-tokens", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g2b")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    cfg = cfg_mod.load(ROOT)
    seed = cfg["seed"]
    policy = cfg_mod.injection_policy(cfg)
    band = cfg_mod.injection_layers(cfg)
    norm_mode = cfg_mod.norm_mode(cfg)
    arms = cfg_mod.vector_arms(cfg)
    alphas = ([0.0] + sorted(args.strengths) if args.strengths
              else [0.0] + cfg_mod.strengths(cfg, include_extension=True))
    base_ladder = cfg_mod.strengths(cfg)
    spec_alpha = (args.specificity_alpha if args.specificity_alpha is not None
                  else base_ladder[-1])
    gen_alphas = (args.gen_alphas if args.gen_alphas is not None
                  else [alphas[1], alphas[len(alphas) // 2], alphas[-1]])
    check_layers = (args.check_layers if args.check_layers
                    else [band[0], band[len(band) // 2], band[-1]])
    single_layer = band[len(band) // 2]
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
    pool, _pool_meta = vec_mod.load_concept_pool(ROOT / "configs" / "concepts.json")

    # Round-robin over the source categories rather than taking the first N:
    # the pool is ordered by category, so "first 10 single-token" returns ten
    # animals and a specificity matrix over ten near-synonyms measures category
    # confusion, not steering specificity.
    by_category: dict[str, list[str]] = {}
    concept_ids, n_multi_token = {}, 0
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

    base_ids, positions, _rendered = prompt_mod.prepare(
        model, tok, prompt_mod.probe_messages(), prefill=True,
        enable_thinking=cfg["planned"]["enable_thinking"])
    seq_len = int(base_ids.shape[1])
    clean_norms = band_inject.band_median_norms(model, base_ids, band, positions)

    # ------------------------------------------------------------ the arms
    vectors_by_arm: dict[str, dict] = {}
    arm_errors: dict[str, str] = {}
    concept_vecs, _means = vec_mod.extract_concept_vectors_band(
        model, tok, concepts, baseline_words, band)
    vectors_by_arm["concept"] = concept_vecs
    lens = None
    if "jlens_row" in arms:
        try:
            lens = lens_mod.load_lens(
                cfg["lens"]["repo"], cfg["lens"]["filename"],
                cfg["lens"]["revision"], device=device, layers=band)
            vectors_by_arm["jlens_row"] = band_inject.rows_by_layer(
                lens, model, concept_ids, band)
        except Exception as exc:  # noqa: BLE001
            arm_errors["jlens_row"] = f"{type(exc).__name__}: {exc}"
            print(f"[arm] jlens_row unavailable: {arm_errors['jlens_row']}",
                  file=sys.stderr)
    randoms = {l: vec_mod.matched_random_directions(concept_vecs[l], seed + l)
               for l in band}
    live_arms = list(vectors_by_arm)
    extract_s = time.time() - t_start - load_s

    def batch_ids(n: int) -> torch.Tensor:
        return base_ids.expand(n, -1).contiguous()

    def stack(words, source_by_layer, layers=None):
        return {l: torch.stack([source_by_layer[l][w] for w in words]).to(device)
                for l in (layers if layers is not None else band)}

    # ---------------------------------------------------------- invariants
    clean = band_inject.injected_prefill_band(
        model, batch_ids(1), band, None, None, positions,
        record_layers=[band[0], band[-1], model.n_layers - 1],
        record_positions=slice(None))
    zero_alpha = band_inject.injected_prefill_band(
        model, batch_ids(1), band, stack(concepts[:1], concept_vecs),
        torch.zeros(1, device=device), positions,
        record_layers=[band[0], band[-1], model.n_layers - 1],
        record_positions=slice(None), norm_mode=norm_mode,
        clean_norms=clean_norms)
    zero_delta = float((clean["logits"] - zero_alpha["logits"]).abs().max())

    injected_one = band_inject.injected_prefill_band(
        model, batch_ids(1), band, stack(concepts[:1], concept_vecs),
        torch.full((1,), spec_alpha, device=device), positions,
        record_layers=[band[0], band[-1], model.n_layers - 1],
        record_positions=slice(None), norm_mode=norm_mode,
        clean_norms=clean_norms)
    realised = band_inject.realised_displacement(injected_one)

    # spatial containment: the band's own layers and the blocks around it
    probe_layers = sorted({max(0, band[0] - 1), band[0], band[-1],
                           min(model.n_layers - 1, band[-1] + 1),
                           model.n_layers - 1})
    clean_full = band_inject.injected_prefill_band(
        model, batch_ids(1), band, None, None, positions,
        record_layers=probe_layers, record_positions=slice(None))
    inj_full = band_inject.injected_prefill_band(
        model, batch_ids(1), band, stack(concepts[:1], concept_vecs),
        torch.full((1,), spec_alpha, device=device), positions,
        record_layers=probe_layers, record_positions=slice(None),
        norm_mode=norm_mode, clean_norms=clean_norms)
    pos_index = torch.as_tensor(list(positions), device=device)
    outside = torch.ones(seq_len, dtype=torch.bool, device=device)
    outside[pos_index] = False
    containment = {}
    for l in probe_layers:
        diff = (clean_full["residuals"][l] - inj_full["residuals"][l]).abs()
        containment[l] = {
            "inside_window": float(diff[:, pos_index, :].max()),
            "outside_window": float(diff[:, outside, :].max()),
        }

    # band vs single layer: the 17x claim, measured on this model
    single = band_inject.injected_prefill_band(
        model, batch_ids(1), [single_layer],
        {single_layer: stack(concepts[:1], concept_vecs, [single_layer])[single_layer]},
        torch.full((1,), spec_alpha, device=device), positions,
        record_layers=[band[-1], model.n_layers - 1],
        record_positions=slice(None), norm_mode=norm_mode,
        clean_norms=clean_norms)

    def displacement(result, layer):
        diff = (result["residuals"][layer] - clean_full["residuals"][layer]
                if layer in clean_full["residuals"]
                else result["residuals"][layer] - clean["residuals"][layer])
        return float(diff[:, pos_index, :].norm(dim=-1).median())

    disp_band_end = displacement(inj_full, band[-1])
    disp_single_end = displacement(single, band[-1])
    disp_band_final = displacement(inj_full, model.n_layers - 1)
    disp_single_final = displacement(single, model.n_layers - 1)

    # generate-time fire counts and seed stability
    gen_a = band_inject.generate_with_injection_band(
        model, hf_model, tok, batch_ids(1), band,
        stack(concepts[:1], concept_vecs),
        torch.full((1,), spec_alpha, device=device), positions,
        max_new_tokens=24, temperature=1.0, seed=seed,
        norm_mode=norm_mode, clean_norms=clean_norms)
    gen_b = band_inject.generate_with_injection_band(
        model, hf_model, tok, batch_ids(1), band,
        stack(concepts[:1], concept_vecs),
        torch.full((1,), spec_alpha, device=device), positions,
        max_new_tokens=24, temperature=1.0, seed=seed,
        norm_mode=norm_mode, clean_norms=clean_norms)
    seed_stable = bool(torch.equal(gen_a["sequences"], gen_b["sequences"]))

    # ------------------------------------------------------- dose-response
    all_ids = torch.as_tensor(
        sorted({i for w in concepts for i in concept_ids[w]}), device=device)
    id_slot = {int(t): j for j, t in enumerate(all_ids)}

    def probe(source_by_layer, alpha_rel: float):
        """P(concept token) and its rank, one concept per batch element."""
        probs_out = np.zeros((len(concepts), all_ids.numel()))
        rank_out = np.zeros((len(concepts), all_ids.numel()), dtype=np.int64)
        for begin in range(0, len(concepts), args.batch):
            chunk = concepts[begin:begin + args.batch]
            result = band_inject.injected_prefill_band(
                model, batch_ids(len(chunk)), band,
                stack(chunk, source_by_layer),
                torch.full((len(chunk),), alpha_rel, device=device), positions,
                record_layers=[band[-1]], record_positions=[-1],
                norm_mode=norm_mode, clean_norms=clean_norms)
            logits = result["logits"].float()
            probs_out[begin:begin + len(chunk)] = (
                torch.softmax(logits, dim=-1).index_select(-1, all_ids)
                .cpu().numpy())
            rank_out[begin:begin + len(chunk)] = lens_mod.ranks_of(
                logits, all_ids).cpu().numpy()
        return probs_out, rank_out

    def own(matrix, row, word, reduce=max):
        return reduce(matrix[row, id_slot[i]] for i in concept_ids[word])

    dose = {arm: {a: probe(vectors_by_arm[arm], a) for a in alphas}
            for arm in live_arms}
    dose_random = {a: probe(randoms, a) for a in alphas}

    def rho_series(table) -> np.ndarray:
        out = []
        for row, word in enumerate(concepts):
            series = [own(table[a][0], row, word) for a in alphas]
            out.append(stats.spearman_ci(np.array(alphas),
                                         np.array(series))["rho"])
        return np.array(out, dtype=float)

    rho_by_arm = {arm: rho_series(dose[arm]) for arm in live_arms}
    rho_random = rho_series(dose_random)

    # realised per-layer displacement, per alpha, from one batch
    ratio_by_alpha = {}
    for a in alphas:
        result = band_inject.injected_prefill_band(
            model, batch_ids(1), band, stack(concepts[:1], concept_vecs),
            torch.full((1,), a, device=device), positions,
            record_layers=[band[-1]], record_positions=[-1],
            norm_mode=norm_mode, clean_norms=clean_norms)
        ratio_by_alpha[a] = band_inject.realised_displacement(result)

    # ------------------------------------------------------------ coherence
    neutral_ids, neutral_pos, _ = prompt_mod.prepare(
        model, tok, [{"role": "user", "content": prompt_mod.TASK_PROMPTS[0]}],
        prefill=False, enable_thinking=cfg["planned"]["enable_thinking"])
    n_seq = int(neutral_ids.shape[1])
    neutral_norms = band_inject.band_median_norms(
        model, neutral_ids, band, neutral_pos)
    n_gen = min(4, len(concepts))
    coherence = {}
    for arm in live_arms:
        for a in gen_alphas:
            gen = band_inject.generate_with_injection_band(
                model, hf_model, tok, neutral_ids.expand(n_gen, -1).contiguous(),
                band, stack(concepts[:n_gen], vectors_by_arm[arm]),
                torch.full((n_gen,), a, device=device), neutral_pos,
                max_new_tokens=64, temperature=1.0, seed=seed,
                norm_mode=norm_mode, clean_norms=neutral_norms)
            nll = inject.sequence_nll(model, gen["sequences"], n_seq)
            probe_gen = band_inject.generate_with_injection_band(
                model, hf_model, tok, batch_ids(n_gen), band,
                stack(concepts[:n_gen], vectors_by_arm[arm]),
                torch.full((n_gen,), a, device=device), positions,
                max_new_tokens=32, temperature=1.0, seed=seed,
                norm_mode=norm_mode, clean_norms=clean_norms)
            coherence[(arm, a)] = {
                "nll": stats.median_iqr(nll.cpu().numpy()),
                "lengths": stats.median_iqr(
                    np.array(gen["new_token_counts"], float)),
                "layers_fired": gen["n_layers_fired"],
                "max_fires": gen["max_fires_per_layer"],
                "neutral": gen["completions"][0][:96].replace("\n", " "),
                "probe": probe_gen["completions"][0][:72].replace("\n", " "),
            }
        print(f"[coherence] arm {arm} done", file=sys.stderr)

    # ------------------------------------------- arm B construction check
    # The analytic row is J_l^T (g * W_U[t]); the numeric one is the gradient
    # of the readout logit with respect to the residual, which includes
    # whatever the final norm actually does. Reported as a cosine, not
    # asserted -- a value below 1 localises the disagreement to a layer rather
    # than aborting the gate.
    row_check = {}
    jspace_loading = {}
    if lens is not None:
        check_tokens = [concept_ids[w][0] for w in concepts][:args.check_tokens]
        with torch.no_grad():
            with jlens.ActivationRecorder(model.layers, at=check_layers) as rec:
                model.forward(base_ids)
                check_residuals = {
                    l: rec.activations[l][0, -1, :].detach().float().clone()
                    for l in check_layers
                }
        for layer in check_layers:
            analytic = band_inject.jlens_row_vectors(
                lens, model, check_tokens, layer)
            numeric = band_inject.jlens_row_vectors_numeric(
                model, lens, check_tokens, layer, check_residuals[layer])
            cosines = torch.nn.functional.cosine_similarity(
                analytic.float().cpu(), numeric.float().cpu(), dim=-1)
            raw = band_inject.jlens_row_vectors(
                lens, model, check_tokens, layer, fold_final_norm=False)
            raw_cos = torch.nn.functional.cosine_similarity(
                raw.float().cpu(), numeric.float().cpu(), dim=-1)
            row_check[layer] = {
                "median_cosine": float(cosines.median()),
                "min_cosine": float(cosines.min()),
                "median_cosine_unfolded": float(raw_cos.median()),
                "n_tokens": len(check_tokens),
            }
        # how much of a concept vector lives in the J-lens row direction
        for layer in check_layers:
            cos = [float(torch.nn.functional.cosine_similarity(
                concept_vecs[layer][w].float().cpu().unsqueeze(0),
                vectors_by_arm["jlens_row"][layer][w].float().cpu().unsqueeze(0),
                dim=-1)) for w in concepts]
            jspace_loading[layer] = {
                "median_cosine": float(np.median(cos)),
                "median_cos2": float(np.median(np.square(cos))),
            }

    # ----------------------------------------------------------- throughput
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_bench = time.time()
    n_bench = 4
    bench_words = (concepts * args.batch)[:args.batch]
    for _ in range(n_bench):
        band_inject.injected_prefill_band(
            model, batch_ids(args.batch), band, stack(bench_words, concept_vecs),
            torch.full((args.batch,), spec_alpha, device=device), positions,
            record_layers=[band[-1]], record_positions=[-1],
            norm_mode=norm_mode, clean_norms=clean_norms)
    bench_s = (time.time() - t_bench) / n_bench
    peak_bytes = (torch.cuda.max_memory_allocated()
                  if torch.cuda.is_available() else 0)
    n_cells = (cfg["planned"]["n_concepts"] * len(base_ladder)
               * len(cfg["planned"]["orders"]) * (len(arms) + 2))
    wall_s = time.time() - t_start

    # --------------------------------------------------------------- report
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G2b : band injection, both arms ================")

    w("\nCONFIG")
    w(f"  model @ revision         {cfg['model']['repo']} @ "
      f"{cfg['model']['revision'][:12]}")
    w(f"  seed / device / dtype    {seed} / {device} / {torch.bfloat16}")
    w(f"  injection policy         {policy}   norm_mode {norm_mode}")
    w(f"  band                     {len(band)} layers {band[0]}..{band[-1]}")
    w(f"  arms built               {live_arms}"
      f"{'  (failed: ' + json.dumps(arm_errors) + ')' if arm_errors else ''}")
    w(f"  alpha_rel ladder         {alphas}   (PER LAYER)")
    w(f"  base ladder from config  {base_ladder}"
      f"   extension {cfg['planned'].get('strengths_extension')}")
    w(f"  prompt tokens            {seq_len}")
    w(f"  injection positions      all_user: {len(positions)} tokens "
      f"[{positions[0]}, {positions[-1] + 1}) of {seq_len}")
    w(f"  concepts                 {len(concepts)}  {concepts}")
    w(f"  baseline words           {baseline_diag['n']} "
      f"({baseline_diag['n_duplicated']} duplicated)")
    w(f"  concepts dropped, multi-token  {n_multi_token} of {len(pool)}")
    w("  clean median residual L2 at the injection positions, per band layer:")
    w("    " + "  ".join(f"L{l}:{clean_norms[l]:.1f}" for l in band))

    w("\nINVARIANTS")
    w(f"  zero-strength identity, max |logit delta|   {zero_delta:.6e}")
    w("    (alpha=0 through all 17 hooks vs no hooks at all; adding 0*v is")
    w("     exact, so any nonzero value here means a hook does something else)")
    w(f"  layers fired, prefill                       "
      f"{injected_one['n_layers_fired']} of {len(band)}  (expect {len(band)})")
    w(f"  total fires, prefill                        "
      f"{injected_one['n_fires_total']}  (expect {len(band)}: one per layer)")
    w(f"  layers fired, across generate               "
      f"{gen_a['n_layers_fired']} of {len(band)}, max "
      f"{gen_a['max_fires_per_layer']} per layer")
    w("    (expect every layer exactly once: prefill only, never on a cached")
    w("     decode step -- a hook that fired while decoding would edit the")
    w("     model's own output and look like introspection)")
    w(f"  hook handles after teardown                 "
      f"{clean['n_handles_after']} / {injected_one['n_handles_after']} / "
      f"{gen_a['n_handles_after']}  (clean / injected / generate)")
    w(f"  seed stability, identical generation        {seed_stable}")
    w("")
    w(f"  realised ||delta_l|| / base_l at alpha_rel={spec_alpha}.")
    w("  The contract fixes this at alpha_rel EXACTLY at every layer, because")
    w("  the hook unit-normalises the direction before scaling. A layer that")
    w("  departs is a mis-scaled layer, named.")
    w(f"    {'layer':>6}{'realised':>12}{'error':>12}")
    for l in band:
        value = realised.get(l, float("nan"))
        w(f"    {l:>6}{value:>12.6f}{abs(value - spec_alpha):>12.3e}")
    w("")
    w("  spatial containment: max |residual delta| by layer")
    w(f"    {'layer':>6}{'inside window':>16}{'outside window':>17}")
    for l in probe_layers:
        c = containment[l]
        marker = ("  <- band" if band[0] <= l <= band[-1] else "")
        w(f"    {l:>6}{c['inside_window']:>16.6e}{c['outside_window']:>17.6e}{marker}")
    w("    below the band's first layer, 'outside window' must be 0.0. Inside")
    w("    and above it, leakage to later positions is expected and reported")
    w("    rather than asserted -- 48 of 64 blocks are linear-attention with a")
    w("    causal conv (kernel 4), so information moves forward in position.")

    w("\nPRIMARY -- the number this whole change order turns on")
    w(f"  cumulative displacement at alpha_rel={spec_alpha}, band vs a single")
    w(f"  layer ({single_layer}) at the SAME per-layer alpha. Run 2 injected the")
    w("  single-layer version while citing the band's strengths.")
    w(f"    median ||h_inj - h_clean|| at L{band[-1]}   band {disp_band_end:.4f}"
      f"   single {disp_single_end:.4f}"
      f"   ratio {disp_band_end / disp_single_end if disp_single_end else float('nan'):.2f}x")
    w(f"    median ||h_inj - h_clean|| at L{model.n_layers - 1}   "
      f"band {disp_band_final:.4f}   single {disp_single_final:.4f}"
      f"   ratio {disp_band_final / disp_single_final if disp_single_final else float('nan'):.2f}x")
    w("    a ratio near 1 would mean the band buys nothing and the diagnosis")
    w("    was wrong; the toy-stack estimate was ~23x, above 17x because live")
    w("    norms grow as earlier injections stack.")

    for arm in live_arms:
        w("")
        w(f"  dose-response, arm {arm}: P(target token) at the report position")
        w(f"    {'concept':<16} " + " ".join(f"{'a=' + str(a):>10}" for a in alphas)
          + f"  {'spearman':>9}")
        for row, word in enumerate(concepts):
            series = [own(dose[arm][a][0], row, word) for a in alphas]
            w(f"    {word:<16} " + " ".join(f"{p:>10.6f}" for p in series)
              + f"  {rho_by_arm[arm][row]:>+9.3f}")
        w(f"    {'-- rank --':<16} " + " ".join(f"{'a=' + str(a):>10}" for a in alphas))
        for row, word in enumerate(concepts):
            series = [own(dose[arm][a][1], row, word, reduce=min) for a in alphas]
            w(f"    {word:<16} " + " ".join(f"{int(r):>10d}" for r in series))
        w(f"    spearman rho across concepts   median "
          f"{np.nanmedian(rho_by_arm[arm]):+.3f}  "
          f"IQR {stats.median_iqr(rho_by_arm[arm])['iqr']:.3f}")
        w(f"    bootstrap CI, mean rho         "
          f"{fmt(stats.bootstrap_ci(rho_by_arm[arm], seed=seed))}")
    w("")
    w("  probabilities near 1e-5 are the floor; rank is the informative")
    w("  quantity there. The decision this gate exists for: pick the lowest")
    w("  alpha at which the target token's rank is small AND the coherence NLL")
    w("  below has not moved, and write it to planned.operating_strength.")

    w("")
    w(f"  realised displacement by alpha (median over the {len(band)} band layers)")
    w(f"    {'alpha_rel':>10}{'median realised':>18}{'min':>12}{'max':>12}")
    for a in alphas:
        values = np.array([ratio_by_alpha[a].get(l, np.nan) for l in band])
        w(f"    {a:>10.3f}{np.nanmedian(values):>18.6f}"
          f"{np.nanmin(values):>12.6f}{np.nanmax(values):>12.6f}")

    w("\nCONTROLS")
    w("  random directions, same ladder, same positions, no concept in them.")
    w("  This distribution should sit on 0.")
    w(f"    spearman rho, random    median {np.nanmedian(rho_random):+.3f}"
      f"  IQR {stats.median_iqr(rho_random)['iqr']:.3f}")
    for arm in live_arms:
        w(f"    Cliff's delta, {arm} vs random   "
          f"{stats.cliffs_delta(rho_by_arm[arm], rho_random):+.4f}")
    w(f"    bootstrap CI, mean random rho     "
      f"{fmt(stats.bootstrap_ci(rho_random, seed=seed))}")
    w("")
    w(f"  steering specificity at alpha_rel={spec_alpha}: inject the row")
    w("  concept, read P(column concept). Diagonal dominance is the property;")
    w("  off-diagonal structure says which concepts are entangled.")
    confusion_by_arm = {}
    for arm in live_arms:
        matrix = dose[arm][spec_alpha][0]
        confusion = np.zeros((len(concepts), len(concepts)))
        for row, word in enumerate(concepts):
            for col, other in enumerate(concepts):
                confusion[row, col] = own(matrix, row, other)
        confusion_by_arm[arm] = confusion
        w(f"    arm {arm}")
        corner = "inject \\ read"
        w(f"    {corner:<16}" + "".join(f"{c[:6]:>8}" for c in concepts))
        for row, word in enumerate(concepts):
            w(f"    {word:<16}" + "".join(f"{v:>8.4f}" for v in confusion[row]))
        diag_wins = int(sum(confusion[i].argmax() == i
                            for i in range(len(concepts))))
        w(f"    rows whose max is on the diagonal  {diag_wins} of "
          f"{len(concepts)}   {fmt(stats.wilson(diag_wins, len(concepts)))}")

    w("\nCROSS-CHECK")
    w(f"  coherence on a NEUTRAL task ({prompt_mod.TASK_PROMPTS[0]!r}), scored by")
    w("  the UNHOOKED model. The probe prompt is unusable for this: it asks the")
    w("  model to name the injected concept, so steering that works raises NLL")
    w("  by definition. On a task the injection should not change, rising NLL")
    w("  is degradation and nothing else.")
    w(f"    {'arm':<11}{'alpha_rel':>10}{'NLL median':>12}{'NLL IQR':>10}"
      f"{'len median':>12}{'fired':>8}")
    for arm in live_arms:
        for a in gen_alphas:
            c = coherence[(arm, a)]
            w(f"    {arm:<11}{a:>10.3f}{c['nll']['median']:>12.4f}"
              f"{c['nll']['iqr']:>10.4f}{c['lengths']['median']:>12.1f}"
              f"{c['layers_fired']:>8}")
    for arm in live_arms:
        for a in gen_alphas:
            w(f"    {arm} a={a:<6.3f} neutral: {coherence[(arm, a)]['neutral']!r}")
            w(f"    {arm} a={a:<6.3f} probe  : {coherence[(arm, a)]['probe']!r}")

    if row_check:
        w("")
        w("  arm B construction. The analytic row is J_l^T (g * W_U[t]); the")
        w("  numeric one differentiates the readout logit with respect to the")
        w("  residual, so it includes whatever the final norm actually does.")
        w("  Cosine 1.0 means the arm injects the direction the lens reads.")
        w("  'unfolded' drops the final-norm gain, which is the version a")
        w("  reading of the lens that ignores the norm would produce.")
        w(f"    {'layer':>6}{'median cos':>13}{'min cos':>10}"
          f"{'unfolded cos':>15}{'tokens':>8}")
        for layer in check_layers:
            r = row_check[layer]
            w(f"    {layer:>6}{r['median_cosine']:>13.6f}{r['min_cosine']:>10.6f}"
              f"{r['median_cosine_unfolded']:>15.6f}{r['n_tokens']:>8}")
        w("    Garcia's own construction should match this up to normalisation;")
        w("    verify against his repo before the sweep, not after.")
    if jspace_loading:
        w("")
        w("  J-space loading of the concept vector: cosine between the arm-A")
        w("  vector and the arm-B row for the same concept at the same layer.")
        w("  cos^2 is the share of the concept vector's variance lying along")
        w("  the direction the lens reads -- the quantity the A-vs-B contrast")
        w("  manipulates.")
        w(f"    {'layer':>6}{'median cos':>13}{'median cos^2':>15}")
        for layer in check_layers:
            r = jspace_loading[layer]
            w(f"    {layer:>6}{r['median_cosine']:>13.4f}{r['median_cos2']:>15.4f}")

    w("\nANOMALIES")
    anomalies = []
    if zero_delta != 0.0:
        anomalies.append(
            f"zero-strength identity is {zero_delta:.3e}, not exactly 0")
    if injected_one["n_layers_fired"] != len(band):
        anomalies.append(
            f"the band fired on {injected_one['n_layers_fired']} of "
            f"{len(band)} layers in prefill")
    if gen_a["max_fires_per_layer"] != 1:
        anomalies.append(
            f"a hook fired {gen_a['max_fires_per_layer']} times across one "
            f"generate call; expected 1 (prefill only)")
    below = [l for l in probe_layers if l < band[0]]
    for l in below:
        if containment[l]["outside_window"] != 0.0:
            anomalies.append(
                f"layer {l} is below the band and has nonzero delta outside "
                f"the window: {containment[l]['outside_window']:.3e}")
    worst = max((abs(v - spec_alpha) for v in realised.values()), default=0.0)
    if worst > 1e-3:
        anomalies.append(
            f"realised displacement departs from alpha_rel by up to {worst:.3e}")
    if not seed_stable:
        anomalies.append("same seed produced different generations")
    for arm, error in arm_errors.items():
        anomalies.append(f"arm {arm} could not be built: {error}")
    anomalies.append(
        "the median that scales each layer is taken per batch ELEMENT over "
        "positions; Garcia's is a scalar over batch x positions. They coincide "
        "at B=1, and this gate runs the invariants at B=1 -- the sweep does "
        "not, so the fidelity note belongs in the report.")
    anomalies.append(
        "concept vectors here come from a category round-robin over the pool, "
        "not the stratified 60; this gate sizes the ladder, it does not "
        "estimate any rate.")
    anomalies.append(
        "the probe prompt is ours, not Garcia's protocol -- it prefills an "
        "assistant turn so the readout is one next-token distribution. The "
        "sweep uses the vendored two-order protocol instead.")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  model load                         {load_s:8.1f} s")
    w(f"  arms built ({len(baseline_words)} baselines, {len(band)} layers) "
      f"{extract_s:8.1f} s")
    w(f"  peak VRAM at batch {args.batch:<3}             "
      f"{peak_bytes / 2**30:8.2f} GiB")
    w(f"  band prefill, batch {args.batch}              {bench_s:8.3f} s")
    w(f"  throughput                         "
      f"{args.batch / bench_s:8.2f} injected passes/s")
    w(f"  planned sweep grid                 {n_cells} cells "
      f"({cfg['planned']['n_concepts']} concepts x {len(base_ladder)} strengths "
      f"x {len(cfg['planned']['orders'])} orders x {len(arms) + 2} arms/controls)")
    w(f"  extrapolated sweep wall-clock      "
      f"{n_cells / (args.batch / bench_s) / 3600:8.2f} h")
    w(f"  gate wall-clock                    {wall_s:8.1f} s")

    w("\nARTIFACTS")
    np.savez(
        args.out / "g2b.npz",
        alphas=np.array(alphas),
        band=np.array(band),
        rho_random=rho_random,
        **{f"rho_{arm}": rho_by_arm[arm] for arm in live_arms},
        **{f"dose_{arm}": np.stack([dose[arm][a][0] for a in alphas])
           for arm in live_arms},
        **{f"rank_{arm}": np.stack([dose[arm][a][1] for a in alphas])
           for arm in live_arms},
        **{f"confusion_{arm}": confusion_by_arm[arm] for arm in live_arms},
        dose_random=np.stack([dose_random[a][0] for a in alphas]),
        realised=np.array([[ratio_by_alpha[a].get(l, np.nan) for l in band]
                           for a in alphas]),
    )
    (args.out / "g2b_meta.json").write_text(json.dumps({
        "injection_policy": policy, "band_layers": band,
        "norm_mode": norm_mode, "arms": live_arms, "arm_errors": arm_errors,
        "alphas": alphas, "specificity_alpha": spec_alpha,
        "concepts": concepts, "clean_norms": clean_norms,
        "seq_len": seq_len, "n_injected_positions": len(positions),
        "zero_delta": zero_delta, "containment": containment,
        "realised_at_specificity_alpha": realised,
        "displacement": {
            "single_layer": single_layer,
            "band_at_band_end": disp_band_end,
            "single_at_band_end": disp_single_end,
            "band_at_final": disp_band_final,
            "single_at_final": disp_single_final,
        },
        "row_check": row_check, "jspace_loading": jspace_loading,
        "seed_stable": seed_stable,
        "throughput_per_s": args.batch / bench_s,
        "wall_s": wall_s,
    }, indent=2))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g2b_report.txt").write_text(report)
    print(report)
    for name in ("g2b_report.txt", "g2b.npz", "g2b_meta.json"):
        print(f"  wrote {args.out / name}")
    print("\nNEXT: a human picks the ladder and the operating point from the")
    print("dose-response and coherence tables above, and writes it into")
    print("configs/sprint.yaml as planned.operating_strength. Nothing")
    print("downstream runs until they do.")


if __name__ == "__main__":
    main()
