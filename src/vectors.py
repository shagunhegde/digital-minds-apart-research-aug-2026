"""Concept vectors.

Started in Phase 2 because G2 cannot measure a dose-response without vectors.
Phase 3 extends this module with the stratified 60-concept selection, the
single-token filter and the composition audit; what is here is the extraction
itself.

Method is the published one, reproduced from
github.com/safety-research/introspection-mechanisms @ 5d5d9b4
(src/vector_utils.py::extract_concept_vector_with_baseline), whose own output
note states it as:

    vector = activation(concept_word) - mean(baseline_activations)

with template "Tell me about {word}", rendered through the chat template with
a generation prompt, read at the last token of the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import jlens

TEMPLATE = "Tell me about {word}"


def load_baseline_words(path: Path) -> tuple[list[str], dict]:
    """Load the vendored baseline list, with the diagnostics G2 reports.

    The published list contains one duplicate. It is kept rather than
    deduplicated -- removing it would change the baseline mean away from the
    published one -- but the count is reported so the very slight reweighting
    is visible.
    """
    payload = json.loads(Path(path).read_text())
    words = payload["words"]
    return words, {
        "n": len(words),
        "n_unique": len(set(words)),
        "n_duplicated": len(words) - len(set(words)),
        "source": payload["source"],
        "commit": payload["commit"],
    }


def word_residual(model, tokenizer, word: str, layer: int) -> torch.Tensor:
    """Residual at the final prompt token for one word. Shape [d_model].

    One prompt per forward. The published extractor batches for speed, but
    batching needs padding, and `jlens.HFLensModel.forward` passes only
    `input_ids` to the text module -- with no attention mask, pad tokens would
    be attended to and every activation in the batch would be wrong. These
    prompts are ~20 tokens, so the loop costs seconds and removes the trap.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEMPLATE.format(word=word)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = model.encode(rendered)
    with jlens.ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(input_ids)
        return recorder.activations[layer][0, -1, :].detach().float().clone()


def extract_concept_vectors(
    model, tokenizer, concepts: list[str], baseline_words: list[str], layer: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Concept vectors at `layer`, plus the baseline mean they were built from.

    Returns ({concept: [d_model]}, baseline_mean [d_model]).
    """
    vectors, means = extract_concept_vectors_band(
        model, tokenizer, concepts, baseline_words, [layer])
    return vectors[layer], means[layer]


def word_residuals(model, tokenizer, word: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Residual at the final prompt token for one word, at several layers.

    One forward pass for the whole band. `word_residual` is this at a single
    layer and is kept because the single-layer path is still what G2 and the
    layer scan use.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEMPLATE.format(word=word)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = model.encode(rendered)
    with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        return {
            layer: recorder.activations[layer][0, -1, :].detach().float().clone()
            for layer in layers
        }


def extract_concept_vectors_band(
    model, tokenizer, concepts: list[str], baseline_words: list[str],
    layers: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor]]:
    """Concept vectors at EVERY layer in `layers`, from one pass per word.

    Returns ({layer: {concept: [d_model]}}, {layer: baseline_mean}).

    The band injects at 17 layers, and a concept vector extracted at layer 27
    is not the layer-38 representation of that concept -- injecting it there is
    a cross-layer mismatch that shows up as a weaker, blurrier intervention
    with no error anywhere. Extracting per layer removes it, and costs nothing
    extra: the 17 layers come off the SAME forward the single-layer version
    already paid for, so 100 baselines + n concepts is 100 + n passes whether
    the band is 1 layer wide or 17.
    """
    layers = sorted(set(int(l) for l in layers))
    with torch.inference_mode():
        baseline_sum = {layer: None for layer in layers}
        for word in baseline_words:
            residuals = word_residuals(model, tokenizer, word, layers)
            for layer in layers:
                baseline_sum[layer] = (
                    residuals[layer] if baseline_sum[layer] is None
                    else baseline_sum[layer] + residuals[layer])
        n = len(baseline_words)
        baseline_mean = {layer: baseline_sum[layer] / n for layer in layers}

        vectors: dict[int, dict[str, torch.Tensor]] = {l: {} for l in layers}
        for concept in concepts:
            residuals = word_residuals(model, tokenizer, concept, layers)
            for layer in layers:
                vectors[layer][concept] = residuals[layer] - baseline_mean[layer]

    for layer in layers:
        for concept, vector in vectors[layer].items():
            if not torch.isfinite(vector).all():
                raise ValueError(
                    f"non-finite concept vector for {concept!r} at layer {layer}")
    return vectors, baseline_mean


def load_concept_pool(path: Path) -> tuple[list[dict], dict]:
    """The 500-concept pool with the source's own category for each word."""
    payload = json.loads(Path(path).read_text())
    return payload["concepts"], {
        "n": payload["n"],
        "source": payload["source"],
        "commit": payload["commit"],
        "category_counts_validated": payload.get("category_counts_validated"),
        "excluded_by_source": payload.get(
            "excluded_by_source_as_hallucination_prone", []),
    }


def single_token_ids(tokenizer, word: str) -> list[int]:
    """Ids of every casing / leading-space variant that is a single token."""
    out: list[int] = []
    bare = word.strip()
    for variant in (bare, bare.lower(), bare.capitalize(),
                    f" {bare}", f" {bare.lower()}", f" {bare.capitalize()}"):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in out:
            out.append(enc[0])
    return out


def composition_audit(pool: list[dict], surviving: set[str]) -> dict:
    """Bucket and category composition before and after the single-token filter.

    The confound most likely to manufacture a finding: abstract concepts are
    more often multi-token, so filtering to single-token names can silently
    turn "abstract concepts are less detectable" into an artifact of
    tokenisation. This reports the shift instead of assuming there isn't one.
    """
    def counts(words: list[dict], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in words:
            out[item[key]] = out.get(item[key], 0) + 1
        return out

    after = [c for c in pool if c["word"] in surviving]
    before_b, after_b = counts(pool, "bucket"), counts(after, "bucket")
    retention = {
        bucket: (after_b.get(bucket, 0) / before_b[bucket]) if before_b.get(bucket) else float("nan")
        for bucket in sorted(before_b)
    }
    return {
        "n_before": len(pool),
        "n_after": len(after),
        "bucket_before": before_b,
        "bucket_after": after_b,
        "bucket_retention": retention,
        "category_before": counts(pool, "category"),
        "category_after": counts(after, "category"),
    }


def stratify_by_rate(
    words: list[str], rates: dict[str, float], per_tier: int, seed: int
) -> tuple[list[str], dict]:
    """Pick `per_tier` concepts each from the top, middle and zero of `rates`.

    IMPORTANT, and the build spec says it too: tiers are a sampling device to
    cover the range, never a reporting unit. Selecting on an outcome and then
    reporting that outcome per tier guarantees regression to the mean -- the
    high tier will look worse and the zero tier better than they "should",
    purely from selection. G3 therefore reports per-concept distributions and
    never "high-tier detected at X%".
    """
    rng = np.random.default_rng(seed)
    ranked = sorted(words, key=lambda w: (-rates[w], w))
    zero = [w for w in ranked if rates[w] == 0.0]

    # The build spec's bottom tier is "20 at exactly 0%", which is a RATE over
    # generated trials and can be exactly zero. Our pilot score is
    # P(true)/(P(true)+P(false)) from next-token logits -- continuous, and
    # never exactly zero. Requiring exact zeros silently returned 40 concepts
    # instead of 60. Exact zeros are still preferred when the score does
    # produce them; otherwise the bottom tier is the bottom of the ranking,
    # which is what "lowest detection" means for a continuous score.
    if len(zero) >= per_tier:
        low = [str(w) for w in rng.permutation(zero)[:per_tier]]
        low_mode = "exact-zero"
        remaining = [w for w in ranked if rates[w] > 0.0]
    else:
        low = ranked[-per_tier:]
        low_mode = "bottom-ranked"
        remaining = ranked[:-per_tier]

    high = remaining[:per_tier]
    middle_pool = remaining[per_tier:]
    if middle_pool:
        centre = len(middle_pool) // 2
        start = max(0, centre - per_tier // 2)
        middle = middle_pool[start:start + per_tier]
    else:
        middle = []

    selected = [*high, *middle, *low]
    tier_rates = {
        "high": [rates[w] for w in high],
        "middle": [rates[w] for w in middle],
        "low": [rates[w] for w in low],
    }
    return selected, {
        "low_tier_mode": low_mode,
        "n_zero_available": len(zero),
        "n_available": len(words),
        "n_high": len(high),
        "n_middle": len(middle),
        "n_low": len(low),
        "shortfall": max(0, 3 * per_tier - len(selected)),
        "tier_rate_range": {
            name: [float(min(v)), float(max(v))] if v else None
            for name, v in tier_rates.items()
        },
    }


def matched_random_directions(
    vectors: dict[str, torch.Tensor], seed: int
) -> dict[str, torch.Tensor]:
    """A random direction per concept, matched to that concept's norm.

    The control arm for every steering claim: cosine has no absolute scale at
    d=5120, so a random direction is only a fair null if its norm matches.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for concept, vector in vectors.items():
        noise = torch.randn(vector.shape[0], generator=generator, dtype=torch.float32)
        noise = noise / noise.norm() * vector.norm().cpu()
        out[concept] = noise.to(vector.device)
    return out
