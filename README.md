# Introspection Factorization

A model can hold a concept in its residual stream, and the Jacobian lens can read
that concept off, while the model says nothing about it. This repo measures where
in that path the signal is lost, by factoring one end-to-end rate into three:

```
P(report) = P(represented) x P(verbalizable | represented) x P(reported | verbalizable)
             f1                f2                             f3
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shagunhegde/digital-minds-apart-research-aug-2026/blob/main/notebooks/demo.ipynb)

---

## Status: built, not yet run

**Every gate is implemented and no gate has been executed.** The machine this was
written on has no CUDA and no installable `torch`, `matplotlib` or `pyarrow`
wheel, so all GPU code is syntax-checked and reviewed but unrun.

What *was* verified, and how, is listed under [Verification](#verification). The
headline number below is a slot, not a result. Nothing in this repo reports a
finding that has not been measured.

| headline | value |
|---|---|
| f₁ · f₂ · f₃ | *not yet measured* |
| `cascade_residual` | *not yet measured* |

Fill it by running the gate chain (below); `scripts/05_figures.py` generates
[`RESULTS.md`](RESULTS.md) with the real numbers, and this table quotes from
there rather than being typed by hand.

---

## Method, in three sentences

A steering vector is added to the residual stream during **prefill only**, at
every layer of the 24–40 workspace band at once (Garcia's `workspace_band`
policy), each addition scaled by the median *live* residual norm at that layer
so strengths are comparable across layers and across papers. Two objects are
injected at matched norm: the **concept vector** — the residual for "Tell me
about *X*" minus the mean over 100 baseline words, extracted at each band layer
— and the **J-lens row** `(W_U[t] @ J_ℓ)`, which is the object Garcia injects and
serves as the positive control. For each trial we record three things about the
*same* prompt: a
cross-validated mass-mean probe score at the readout position (**f₁**, thresholded
at 5% FPR on matched-norm random-direction controls), whether the concept token
falls inside the top-k of the Jacobian-lens readout at the report position
(**f₂**), and whether the model's own generation names it (**f₃**). Because the
three denominators nest, `f₁·f₂·f₃` must equal the end-to-end rate exactly — so
`cascade_residual` is a bug detector that fires whenever a stage silently drops
trials or changes its denominator.

---

## Three honest limitations

**1. The lens is applied off its fitting distribution.** The Jacobian lens was
fitted on WikiText and is applied here to chat-formatted introspection prompts.
Nothing corrects for that shift; it is a standing caveat on every f₂ number.

**2. The stratification prior does not exist.** The build spec stratifies concepts
by Macar's per-concept Gemma detection rates. Those are not published anywhere
reachable — the released caches are aggregates over `(layer, strength, arm)`, and
the abliterated checkpoint is weights only. Concepts are therefore stratified on
detection measured by *this* project. That avoids cross-model regression to the
mean, but it forfeits the cross-model rank correlation entirely, and selecting on
a measured outcome means **tiers are a sampling device and no per-tier rate is
reported anywhere**. Details in [`ASSETS.md`](ASSETS.md) §12.

**3. Ten tasks, one operating point, and f₂ is a curve.** Concept *i* takes task
*i* mod 10 in all of its cells, so the controls finally carry task-level
variation — but the task is fixed *within* a concept (that is what pairs its
injected and control cells), so intervals are over concepts and tasks jointly and
no per-task rate rests on more than six concepts. Generation runs at one
strength, chosen by a human from the G2b report. And f₂ rises monotonically with
k by construction, so a scalar f₂ is not a reportable quantity — it is only
meaningful against the matched-norm random-vector null band, because cosine has
no absolute scale at d=5120.

**4. Two ladders were run before this one, and both were invalid.** `[1, 2, 4, 8]`
ran before unit-normalisation, so realised ‖δ‖ was 11–88× the residual norm —
overwrite, not steering. `[0.02, 0.05, 0.09, 0.14]` was Garcia's *per-layer*
ladder applied at a *single* layer, ~17× weaker in cumulative displacement (23×
measured on a toy stack, because live norms compound). Every number from those
runs describes a misconfiguration and is superseded, not compared against. The
methods point generalises: **published strength conventions are incommensurable,
so report ‖δ‖ relative to the residual norm × the number of intervened layers,
always.**

Further caveats live in each gate's `ANOMALIES` block, which is where they belong:
FPR rests on as many trials as there are task prompts (zero-strength injections are
identical across concepts by construction); identification is scored by a
deterministic word-boundary matcher unless `ANTHROPIC_API_KEY` is set, and the
report always says which was used.

---

## Running it

Gates are strictly sequential. Each **emits a report and stops** — no gate prints
PASS or FAIL. A human reads the numbers and decides whether to continue.

```bash
pip install -r requirements.txt
pip install "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
```

```bash
python gates/g0_assets.py          # assets, lens shape, identity-cosine profile
python gates/g1_lens.py            # J-lens vs logit lens vs random-rotation null
python gates/g2_inject.py          # single-layer harness invariants
python gates/g2b_band.py           # the band, both arms, the ladder  <- DECISION POINT
```

G2b is where a human reads the dose-response and coherence tables and writes
`planned.operating_strength` into `configs/sprint.yaml`. `src/config.py` refuses
to guess, so everything below stops with an explanation until that is done.

```bash
python scripts/01_concept_vectors.py   # band re-pilot, selection, both arms
python gates/g3_baseline.py            # (arm, strength) grid, yes-bias, bimodality
python scripts/02_sweep.py             # 1,920 cells, resumable, one shard per concept
python scripts/03_generate.py          # T=1, 4 samples, operating point only
python gates/g4a_sweep.py              # sweep integrity      (no GPU)
```

The cascade is a cascade *of something*, so the last four stages run once per
vector arm:

```bash
for arm in concept jlens_row; do
  python scripts/04_factors.py --vector-arm "$arm"   # lens readouts + position control
  python gates/g4_factors.py  --arm "$arm"           # the cascade    (no GPU)
  python scripts/05_figures.py --arm "$arm"          # three figures  (no GPU)
  python gates/g5_controls.py  --arm "$arm"          # controls, multiplicity, power, prereg
done
```

`notebooks/gates_a100.ipynb` runs the whole chain on an A100 80 GB with the
expected wall-clock per stage. `notebooks/demo.ipynb` is the six-cell submission:
cells 1–2 and 6 need no GPU and regenerate every figure and number from
`artifacts/`.

To mirror artifacts so the Colab can pull them without a GPU:

```bash
huggingface-cli upload <you>/introspection-factorization artifacts --repo-type=dataset
```

Then set `ARTIFACTS_REPO` in cell 1 of the demo.

---

## What is reused

Almost everything. New code was written only where nothing was published.

| asset | used for | pinned |
|---|---|---|
| [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) | lens loading, `transport`, the logit-lens baseline, the six eval sets | `581d398` |
| [`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms) | 500 concepts, 100 baseline words, 8 judge rubrics (verbatim) | `5d5d9b4` |
| [`e-m-garcia/j-lens-verbalized-awareness`](https://github.com/e-m-garcia/j-lens-verbalized-awareness) | the two-order protocol and three-key JSON schema (verbatim) | `f92218c` |
| `neuronpedia/jacobian-lens` | the fitted J-lens for Qwen3.6-27B | `a4114d7` |
| `camilablank/workspace-lenses` | R-lens comparison + a second J-lens fit as a robustness arm | — |

Written from scratch: the random-rotation null lens, the injection harness, the
sweep, the cascade, and the gates.

### Provenance and licensing of vendored content

Three files under `configs/` contain material copied verbatim from upstream, so
that this sprint reproduces without a network call. Each records its source repo,
commit SHA and file path; `judge_rubrics.json` also records a sha256 of the source
file it was extracted from.

| file | from | upstream licence |
|---|---|---|
| `configs/judge_rubrics.json` — 8 rubrics, ~12 KB of prompt text | `safety-research/introspection-mechanisms` `src/eval_utils.py` | **no LICENSE file**; its README states the project is "released for research purposes" |
| `configs/concepts.json` — 500 concept words | same repo, `concepts_list.py` + `01_concept_injection.py` | as above |
| `configs/baseline_words.json` — 100 baseline words | same repo, `src/vector_utils.py` | as above |
| the two protocol instructions in `src/prompts.py` | `e-m-garcia/j-lens-verbalized-awareness` | Apache-2.0 |
| `jlens` (imported, not vendored) and the six eval sets (fetched at run time) | `anthropics/jacobian-lens` | Apache-2.0 |

The judge rubrics are reproduced **verbatim and unmodified** because comparability
with the published numbers depends on the exact wording. This use is research, with
attribution, consistent with the upstream's stated intent — but note it is not a
formal licence grant. If you are reusing this repo commercially, or if you are an
upstream author who would rather these were fetched at run time than vendored, open
an issue and they will be replaced with a fetch-and-verify step like the one already
used for the eval sets.

This repo's own code is MIT — see [`LICENSE`](LICENSE). That covers the code
written here; it does not relicense the vendored material above, which stays
under its own terms.

---

## Verification

Unrun code is labelled as such. What was checked, and how:

- **`stats.dip_statistic`** — a hand-rolled Hartigan dip was written first and was
  **wrong**: it decoupled the two segments at the mode, agreeing with an
  independent linear program to 4e-17 on 15 of 21 samples and understating the dip
  by up to 1.7e-2 on the rest, always downward. `diptest` reproduces the LP
  exactly, so it is imported instead. `scripts/00_verify_dip.py` reproduces the
  whole comparison.
- **`factors.cascade`** — residual max **5.6e-17** over 200 randomized nested-set
  cases, and **0.075** on a deliberately mismatched denominator. It detects rather
  than assumes.
- **The f₁ probe** — grouped folds leak zero concepts; shuffled labels collapse f₁
  from 0.233 to **0.042**, which is the 0.05 FPR the threshold was set at.
- **The null lens** — `h·Jᵀ·Qᵀ == h·(QJ)ᵀ` to 1.2e-14, with `QJ` preserving `J`'s
  singular values to 7e-15: the null breaks alignment with the unembedding while
  leaving the spectrum intact.
- **Rank counting** — matches a naive argsort exactly; `argmax` ranks 1.
- **Figure 1 arithmetic** — the four segments partition 100% to 2.8e-14 across
  1e5 random triples and every degenerate case.
- **Grid arithmetic** — 4,320 cells exactly as specified; 432 actual forwards
  (8× from batching, then an exact zero-strength dedupe).
- **Eval schema** — checked against the real JSON, which is what caught that
  `association` and `typo` carry no `target` field at all (198 of 398 items would
  have raised `KeyError`) and that the published pass@k is a mean of per-item
  *fractions*, not a best-of.

Not verified: anything requiring a GPU, and the R-lens checkpoint format — that
arm is guarded and reports itself unavailable rather than crashing.

---

## Layout

```
configs/     sprint.yaml (single source of truth), vendored concepts,
             baseline words, judge rubrics, prereg record
src/         config, stats, lens, vectors, prompts, inject, band_inject,
             sweep, factors, judge, plots
gates/       g0, g1, g2, g2b, g3, g4a, g4, g5 -- each emits a report and stops
scripts/     00 dip + position verification, 01 vectors, 02 sweep,
             03 generate, 04 factors, 05 figures
notebooks/   gates_a100.ipynb (full chain), demo.ipynb (six cells)
artifacts/   gitignored; mirror to HF
```

`src/inject.py` is the single-layer harness and still backs G2 and the layer
scan; `src/band_inject.py` is the `workspace_band` policy everything downstream
runs under. `src/config.py` is the one reader for `sprint.yaml`, so nine scripts
cannot drift apart on the operating point.

`ASSETS.md` records what resolved, what drifted, and what is missing, with a
resolution date. `configs/prereg.json` records which analyses were specified
before any data existed and which were added during the build — G5 prints both.
