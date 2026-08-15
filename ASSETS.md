# ASSETS — what resolved, what drifted

Resolution date: **2026-08-14**. Every row was checked against the live asset,
not from memory.

**Probe validity.** Existence was checked by HTTP status. That probe was itself
controlled first: a nonsense GitHub path returns 404 and a nonsense HF path
returns 401, while `torvalds/linux` returns 200 — so the 200s below are real
and not a proxy answering everything. This mattered: `Qwen/Qwen3.6-27B` looked
implausible (27B is a Gemma size, not a Qwen one) but is real, with 6.9M
downloads and a release postdating the assistant's May 2026 knowledge cutoff.

Statuses are **RESOLVED** (exists and its contents were inspected),
**EXISTS-UNINSPECTED** (verified present, contents not yet read — these are
Phase 3+ assets), or **MISSING-with-substitute**.

---

## Code

| Asset | Status | Pinned |
|---|---|---|
| `anthropics/jacobian-lens` | RESOLVED | `581d398` |
| `safety-research/introspection-mechanisms` | RESOLVED (concept list read; judge rubrics not yet read) | `5d5d9b4` |
| `e-m-garcia/j-lens-verbalized-awareness` | RESOLVED | `f92218c` |
| `tao-hpu/jspace-replication` | EXISTS-UNINSPECTED (reference only) | `e03ee97` |

`anthropics/jacobian-lens` is the companion code to *Verbalizable
Representations Form a Global Workspace in Language Models*
(transformer-circuits.pub/2026/workspace). 1,763 stars, last pushed
2026-08-04. It supplies everything the sprint needs from the lens and is
imported, not reimplemented.

Garcia's repo ships `configs/qwen3.6-27b-*` for exactly this model, including
`workspace-first-half` / `workspace-second-half` variants — the half-band
comparison the build spec's trap table warns about is already parameterised
there. It also ships `data/frozen_suite.json` + `SHA256SUMS` and
`scripts/verify_published_results.py`, so its numbers are reproducible before
we depend on them.

## Weights and lenses

| Asset | Status | Note |
|---|---|---|
| `Qwen/Qwen3.6-27B` | RESOLVED | rev `6a9e13b`, 55.6 GB bf16 across 15 shards |
| `neuronpedia/jacobian-lens` → `qwen3.6-27b/.../n1000.pt` | RESOLVED | 3,303,032,772 B, LFS sha256 `1718c8c5…` |
| `camilablank/workspace-lenses` | RESOLVED | **has an r-lens for qwen3.6-27b** |
| `uzaymacar/gemma-3-27b-abliterated` | EXISTS-UNINSPECTED | not on the critical path |

---

## Divergences from the build spec

These are the reasons Phase 0 exists. Each changes code that was specified.

### 1. The lens is `J_l`, not `W_U J_l` — and it is 50× smaller than budgeted

The spec's Phase 1 signature reads:

```python
def load_lens(repo, filename, device) -> Tensor:
    """W_U @ J_ell, [n_layers, n_vocab, d_model], fp16."""
```

That object would be `64 × 248,320 × 5,120 × 2 B` = **162.7 GB**. The actual
file is **3.30 GB**. The stored object is `J_l` itself, `[d_model, d_model]`
per layer; the README confirms the readout is `lens_l(h) = unembed(J_l @ h)`,
with the vocabulary entering through the model's own unembedding.

Arithmetic, confirmed on a second model:

| model | d_model | fitted layers | predicted | actual | overhead |
|---|---|---|---|---|---|
| Qwen3.6-27B | 5120 | 63 | 3,302,993,920 | 3,303,032,772 | 38,852 B |
| gemma-3-27b | 5376 | 61 | 3,525,967,872 | 3,525,986,731 | 18,859 B |

Consequences, all favourable:

- The **whole** lens fits on the GPU. The spec's Tier A plan to "keep the lens
  on GPU for the active layer only and page other layers from CPU" is
  unnecessary. One layer is 105 MB fp32, not 1.5 GB.
- `JacobianLens.__init__` casts to fp32, so the resident CPU copy is 6.6 GB.
  Only swept layers need to move to GPU.
- Revised budget: 55.6 GB weights + ~0.3 GB for three fp32 layers +
  activations. Headroom on 80 GB is real but the spec's "~62 GB total" was
  optimistic on weights (55.6, not 54) and pessimistic on the lens.

### 2. `readout` / `token_rank` should be built on `JacobianLens.transport`

`transport(residual, layer)` is documented as "the bare `J_l @ h` for callers
that already have residuals" — exactly the injection harness's need, since it
must read out *modified* residuals rather than re-running from text. The
spec's `readout(h, lens, layer, k)` and `token_rank(...)` become thin wrappers
over `model.unembed(lens.transport(h, layer))`.

`apply(..., use_jacobian=False)` gives the vanilla logit-lens baseline that G1
needs as a control, already implemented. The **random-rotation null lens** is
not published and must be written.

### 3. The target model is a VLM with hybrid attention

`Qwen3_5ForConditionalGeneration`, not a `*ForCausalLM`. Two consequences:

- **Loading.** `AutoModelForCausalLM` may not accept it; `g0_assets.py` tries
  `AutoModelForImageTextToText` first and reports which class actually loaded.
- **Layout.** Weights sit at `model.language_model.{layers,norm,embed_tokens}`
  with `lm_head` at top level — jlens's second known `Layout`, so `from_hf`
  auto-detects it and hooks land on the 64 text blocks only. The 333-key
  `model.visual` tower and the separate `mtp.*` multi-token-prediction head
  are never touched. Verified against `model.safetensors.index.json`.
- **`layer_types` alternates three `linear_attention` blocks to one
  `full_attention`** (16 of 64 are full attention). This is the one open
  question that bears on a spec decision — see below.

### 4. The lens covers 63 of 64 layers

Implied by the file-size arithmetic; `g0_assets.py` confirms `source_layers`
at runtime and reports which block has no lens. Any "final layer" cross-check
must use the deepest *fitted* layer, not layer 63.

### 5. `d_model` is 5120, so the published cosine null does not transfer

The spec quotes "≈0.032 with sd 0.281 at d=5,376" as the null expectation for
mean pairwise cosine across concept vectors. 5,376 is Gemma-3-27B's width.
This model is 5,120. G3 must recompute the null at d=5120 rather than
comparing against the quoted figure.

### 6. This lens ships without its fitting metadata

Sibling directories (`qwen3.5-27b`, `qwen2.5-7b-it`) carry `config.yaml` and
`*_convergence.csv`. The `qwen3.6-27b` directory carries neither — only the
`.pt` and a `CREDIT.md` ("trained by @mntss, Mateusz Piotrowski, Anthropic
Interpretability"). Fitting hyperparameters and the convergence trace are
therefore unavailable for the exact lens we use. `n_prompts` is recoverable
from the checkpoint itself and is reported in G0.

### 7. A second, independent J-lens exists for this model

`camilablank/workspace-lenses/qwen3.6-27b/` holds both a `j-lens/lens.pt`
(3,303,028,503 B) and an `r-lens/lens.pt` (3,303,028,567 B). The j-lens there
is **not** byte-identical to Neuronpedia's (3,303,032,772 B), so it is a
different fit of the same object.

- The sprint pins the Neuronpedia lens (the spec's choice) as primary.
- The r-lens makes Phase 5's "R-lens comparison if available" **available**,
  not optional.
- The second j-lens is a free robustness arm: the same pipeline under two
  independent fits bounds how much of f₂ is lens-fit noise.

### 8. One dependency added beyond the spec's list: `diptest`

Hartigan's dip is not in scipy and G3 reads bimodality as a headline. A
hand-rolled implementation was written first, from the tube characterisation:
a unimodal `G` within sup-distance `d` of `Fn` with mode at `m` exists iff
`2d ≥ max(Fn − GCM Fn)` on the left and `2d ≥ max(LCM Fn − Fn)` on the right.

**That derivation is wrong.** It treats the two segments as independent, but a
single `G` takes one value at the mode, and Hartigan minimises over a modal
*interval*, not a point. Checked against an independent linear program
(minimise tube half-width subject to convex-left/concave-right constraints,
scanning the mode) over 21 samples:

- agreement to machine precision (≤4e-17) on 15 samples
- understated the dip by up to **1.7e-2** on the other six
- the error was one-directional — the hand-rolled version was never too large

`diptest` (a port of Hartigan's AS 217) reproduces the LP exactly on all 21.
An understated dip understates bimodality, which is precisely where G3 reads a
result. `scripts/00_verify_dip.py` reproduces the whole comparison.

### 9. The published eval sets are not uniform in schema (Phase 1)

`jacobian-lens/data/evaluations/` supplies six sets; the sprint uses the
build spec's four. 398 items, but they do not share a shape:

| set | items | has `target` | intermediates per item |
|---|---|---|---|
| multihop | 93 | 93 | 84 items have exactly 1 |
| multilingual | 107 | 107 | **every item has >1** |
| association | 102 | **0** | 1 |
| typo | 96 | **0** | 1 |

Three consequences, each of which would otherwise have produced a wrong
number or a crash:

- **`target` is absent from association and typo.** Reading `item["target"]`
  raises `KeyError` on 198 of 398 items. It is optional, and where present it
  fixes the readout position rather than being scored.
- **The build spec's "filter to questions the model answers correctly with no
  chain of thought" is undefined for association and typo** — there is no
  answer to check, only a vignette evoking a concept and a misspelling. G1
  applies the competence filter where a `target` exists and reports the rest
  as filter-inapplicable rather than pretending it ran.
- **The published pass@k is a mean over items of the *fraction* of that
  item's intermediates that hit**, not the best one. Since every multilingual
  item carries several (`Spanish`, `opposite`, `big`, `small`), scoring the
  best would inflate the number and make it comparable to nothing. G1 scores
  each intermediate separately and reports both the item-level fraction
  (published, bootstrap CI over items) and the word-level proportion (a true
  proportion, so Wilson applies).

### 10. `concepts_list.py` already excludes the confabulation attractor

The 500-concept list is 450 new + 50 baseline. Its header excludes, verbatim,
`Apples, Bicycles, Ocean` as "hallucination-prone". The spec's G3 cross-check
expects apple-dominance among *wrong* identifications (Lederman & Mahowald:
74.8% against a 0.003% corpus base rate). That check remains meaningful —
apple can dominate wrong guesses without ever being an injected concept — but
the exclusion should be stated when the number is reported, since the upstream
authors already treated it as a known artifact.

### 11. Concept-vector extraction resolved to published code (Phase 2)

`introspection-mechanisms` publishes `src/vector_utils.py`, which the huge
`experiments/*.py` files import. The method is stated in its own output note:

    vector = activation(concept_word) - mean(baseline_activations)

with template `"Tell me about {word}"` rendered through the chat template with
a generation prompt, read at the last token, and 100 baseline words
(`DEFAULT_BASELINE_WORDS`, `get_baseline_words(n=100)`).

Vendored into the repo so the sprint is reproducible without a network call:

- `configs/baseline_words.json` — the 100 baseline words. **The published list
  contains one duplicate** (100 entries, 99 unique), so the baseline mean is
  very slightly weighted toward that word. Kept rather than deduplicated —
  removing it would move the mean away from the published one — and the count
  is reported in G2 so the reweighting is visible.
- `configs/concepts.json` — the paper's 50 baseline concepts
  (`DEFAULT_TEST_CONCEPTS`). Phase 3 adds the 450 in `concepts_list.py` and
  stratifies.

One deliberate deviation: the published extractor batches prompts for speed.
`src/vectors.py` runs one prompt per forward instead, because
`jlens.HFLensModel.forward` passes only `input_ids` to the text module — with
no attention mask, pad tokens would be attended to and every activation in a
padded batch would be wrong. These prompts are ~20 tokens, so the loop costs
seconds and removes the trap entirely.

### 12. MISSING: Macar's per-concept Gemma detection rates (Phase 3)

**Status: MISSING-with-substitute.** The build spec uses these for two things —
stratifying the 60 concepts into 20 high / 20 mid / 20 zero, and the
cross-model rank correlation in G3. They are not published. Checked:

| where | what is actually there |
|---|---|
| `plotting/data/fig1_metrics_cache.parquet` | aggregates over `(layer_idx, strength)`, columns `detection_hit_rate`, `forced_identification_accuracy`, `combined_...`, each with `_ci_lower/_ci_upper` |
| `plotting/data/fig2_sweep_metrics_cache.parquet` | same, plus `injection_layer` / `meta_layer` |
| `plotting/data/fig3_metrics_cache.parquet` | same, plus `arm` |
| `uzaymacar/gemma-3-27b-abliterated` | weights only — 12 safetensors shards, `ablation_config.json`, no results |
| repo `README.md` | aggregate findings prose |

Schema confirmed from `plotting/plot_figures.py`, which consumes all three by
`df[df["strength"] == s].sort_values("layer_idx")`. No per-concept axis exists
in any of them.

**Substitute.** `scripts/01_concept_vectors.py` measures its own per-concept
detection across the full single-token pool from next-token logits (Macar's
other prior, ~10x cheaper than generating) and stratifies on that.

Net effect, all surfaced in the G3 report:

- **Better:** no cross-model transfer, so no regression to the mean from
  another model's noisy labels — the exact trap the spec's §7 table names.
- **Better:** bimodality is tested on the **full pool**, not on 60 concepts
  chosen to span the range. Testing the dip on a set selected for spread would
  have manufactured the result. G3 reports the pool dip as the finding and the
  selected-60 dip explicitly labelled selection-induced.
- **Worse:** the cross-model Spearman against Gemma cannot be computed at all,
  and G3 says so rather than substituting a different correlation.
- **Constraint:** selecting on a measured outcome means tier membership is not
  independent of detection. Tiers are a sampling device only; no per-tier rate
  appears anywhere in the report.

`--stratify-file rates.json` (`{"word": rate}`) restores the spec's original
design if the numbers ever turn up.

### 13. Judge rubrics and concept pool resolved (Phase 3)

- `configs/judge_rubrics.json` — 8 `EvaluationCriteria` copied **verbatim**
  from `src/eval_utils.py`, with a sha256 of the source file. The three G3
  uses are `claims_detection`, `correct_concept_identification`,
  `coherency_score`. Rubric text is data, never reworded in code.
- `configs/concepts.json` — the full 500 (450 `NEW_CONCEPTS` + the paper's 50),
  each tagged with the source file's own category. Every category was validated
  against the count its own header declares; all 20 match, summing to 450. One
  header (`ACTIONS/VERBS as nouns`) contains lowercase and was initially missed
  by the parser, silently folding 28 concepts into `PROFESSIONS` — caught by
  that count check.
- The abstract/concrete mapping is **ours**, stated explicitly in the file.
  Families that are neither cleanly (professions, science terms, sports,
  music/art, weather, actions-as-nouns) are a third `mixed` bucket rather than
  forced into a binary: 230 concrete / 94 abstract / 176 mixed.

---

## Decisions taken (2026-08-14)

**D1 — swept layers are `[27, 31, 35]`, all `full_attention`.** The spec
inherits Garcia's "hooks during prefill only, never during cached decode",
justified as "correct semantics **and** KV-cache reuse". 48 of 64 blocks here
are `linear_attention` and carry recurrent state rather than a KV cache, so
the cache-reuse half of that justification does not transfer unchanged. Hook
removal before decode remains correct either way — the injection is a
prefill-time edit — but restricting the sweep to `full_attention` blocks makes
the inherited argument hold literally.

`layer_types` places `full_attention` at indices ≡ 3 (mod 4), so Garcia's
24–40 band contains 27, 31, 35, 39. Taking the first three preserves the band
prior at no cost and leaves 39 as a spare. G2's spatial-containment arm should
still be read with the causal conv (`linear_conv_kernel_dim: 4`) in mind for
any downstream linear-attention block.

**D2 — Neuronpedia primary, `camilablank` as a robustness arm.** All headline
numbers use the pinned Neuronpedia lens. The second fit re-runs the f₂ readout
stage only, bounding how much of f₂ is lens-fit noise; no extra generation.
The r-lens from the same repo supplies Phase 5's side-by-side comparison.

---

## Not yet inspected

Deferred to the phase that needs them, listed so they are not mistaken for
resolved: judge rubrics and the injected-thought prompt template in
`introspection-mechanisms`; Macar's per-concept Gemma detection rates (the
stratification prior); the abliterated Gemma checkpoint; the eval sets in
`jacobian-lens/data/evaluations/` for G1.
