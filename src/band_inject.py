"""Band injection: Garcia's `workspace_band` policy for this harness.

Why this module exists. The sweep injected at ONE layer with per-layer
alpha_rel in [0.02, 0.14], citing Garcia's config. But Garcia's
`intervention_layer_policy: workspace_band` registers one intervention per
layer across `workspace_layers: [24..40]` -- SEVENTEEN simultaneous per-layer
additions, each of magnitude strength x additive_norm_multiplier(2.0) x
median live residual norm (interventions.py L161-168 of his repo, @ f92218c).
A single-layer injection at the same per-layer alpha is roughly 17x weaker in
cumulative residual displacement, which is why G2's dose-response
probabilities sat at 1e-4 and G3 detection was 0.000 in every cell. The
[1, 2, 4, 8] ladder that preceded it was ALSO invalid, for the opposite
reason: it ran before unit-normalisation, so ||delta|| was 11-88x the residual
norm. The regime between those two mistakes -- Garcia's actual operating
regime -- has never been run on this harness.

Fidelity notes vs Garcia (state these in the report):
  * He scales by the LIVE median norm inside the hook (post earlier-layer
    injections); so does this module. `norm_mode="clean"` is available for an
    ablation but is not his protocol.
  * His median is a scalar over (batch x positions). This module defaults to
    the median per batch ELEMENT over positions, because our batch mixes
    concepts whose earlier-band injections differ. For B=1 the two coincide,
    and `median_scope="global"` reproduces his reduction exactly.
  * He injects per-layer J-lens rows. We inject per-layer objects supplied by
    the caller: concept vectors for the headline arm, J-lens rows for the
    Garcia-replication / ceiling arm. Same-norm, different-object is itself
    the f2 contrast.

Everything `inject.py` promises still holds here, per layer: the direction is
unit-normalised inside the hook so ||delta_l|| == alpha_rel x base_l exactly,
the hook refuses to fire on a cached decode step, and handles are removed in a
`finally`. What changes is that there are now 17 of them and the base norm is
read live rather than cached, so the displacement compounds down the band the
way Garcia's does.
"""

from __future__ import annotations

import torch

import jlens

#: Garcia's `additive_norm_multiplier` for this checkpoint. His `strength` is
#: multiplied by it before it meets the residual, so a config strength of 0.07
#: is an alpha_rel of 0.14. Every strength in sprint.yaml is ALREADY an
#: alpha_rel with this folded in; the constant is here so the arithmetic is
#: recorded somewhere other than a comment.
GARCIA_ADDITIVE_NORM_MULTIPLIER = 2.0


def band_median_norms(
    model, input_ids: torch.Tensor, layers: list[int], positions
) -> dict[int, float]:
    """Clean median residual L2 norm at `positions`, for every band layer.

    One forward pass for the whole band, where `inject.median_residual_norm`
    costs one per layer. These are NOT what the live hook uses -- they are the
    `norm_mode="clean"` ablation's scale, and the denominator the gate divides
    realised ||delta_l|| by when it reports how far the band moved the stack.
    """
    layers = sorted(set(int(l) for l in layers))
    with torch.inference_mode():
        with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            return {
                layer: float(
                    recorder.activations[layer][:, positions, :]
                    .detach().float().norm(dim=-1).median())
                for layer in layers
            }


def make_band_hook(
    vectors: torch.Tensor,        # [B, d_model] unit-or-not; normalised here
    strength_rel,                 # [B] or scalar (alpha_rel, x2.0 folded in)
    positions,                    # index list over the prompt
    fired: dict,                  # {layer: [records]} mutated on every firing
    layer: int,
    norm_mode: str = "live",      # "live" (Garcia) | "clean"
    clean_norm: float | None = None,
    prefill_only: bool = True,
    median_scope: str = "per_element",   # "per_element" | "global"
):
    """One layer's additive hook. Registered once per band layer.

    `fired[layer]` gains one record per firing, carrying the sequence length,
    the base norm the delta was scaled by and the realised ||delta|| per batch
    element. The fire count is `len(fired[layer])` and must be 1 for a prefill
    or a whole `generate` call; the norms are what G2b divides to report the
    realised per-layer displacement rather than the intended one.
    """
    if norm_mode not in ("live", "clean"):
        raise ValueError(f"unknown norm_mode {norm_mode!r}")
    if norm_mode == "clean" and clean_norm is None:
        raise ValueError(f"clean_norm required at L{layer} for norm_mode='clean'")
    if median_scope not in ("per_element", "global"):
        raise ValueError(f"unknown median_scope {median_scope!r}")
    if vectors.ndim != 2:
        raise ValueError(f"vectors at L{layer} must be [B, d_model], got "
                         f"{tuple(vectors.shape)}")
    unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    alpha_rel = torch.as_tensor(strength_rel, dtype=torch.float32)
    if alpha_rel.ndim == 0:
        alpha_rel = alpha_rel.expand(vectors.shape[0])
    if alpha_rel.shape[0] != vectors.shape[0]:
        raise ValueError(f"strength_rel {tuple(alpha_rel.shape)} does not match "
                         f"vectors {tuple(vectors.shape)} at L{layer}")

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if prefill_only and h.shape[1] == 1:      # cached decode step
            return output
        if h.shape[0] != unit.shape[0]:
            raise ValueError(f"batch mismatch at L{layer}: "
                             f"{h.shape[0]} vs {unit.shape[0]}")
        if h.shape[-1] != unit.shape[-1]:
            raise ValueError(f"d_model mismatch at L{layer}: "
                             f"{h.shape[-1]} vs {unit.shape[-1]}")
        # Snapshot BEFORE the add: advanced indexing copies, so `base` is the
        # incoming residual even though the add below is in-place.
        sel = h[:, positions, :].float()
        if norm_mode == "live":
            norms = sel.norm(dim=-1)                              # [B, P]
            if median_scope == "global":
                base = norms.median().expand(h.shape[0]).clone()  # Garcia
            else:
                base = norms.median(dim=1).values                 # [B]
        else:
            base = torch.full((h.shape[0],), float(clean_norm),
                              dtype=torch.float32, device=h.device)
        alpha_abs = alpha_rel.to(h.device) * base                 # [B]
        delta = (alpha_abs[:, None, None]
                 * unit[:, None, :].to(h.device).float()).to(h.dtype)
        h[:, positions, :] += delta
        fired.setdefault(layer, []).append({
            "seq_len": int(h.shape[1]),
            "base": base.detach().cpu().tolist(),
            "delta_norm": delta.float().norm(dim=-1)[:, 0].detach().cpu().tolist(),
        })
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


def _register_band(
    model, band_layers, vectors_by_layer, strength_rel, positions, fired,
    norm_mode, clean_norms, median_scope,
):
    """Register one hook per band layer, in layer order. Returns the handles.

    Order matters and is not cosmetic: torch fires forward hooks in
    registration order per module, and the modules themselves run in depth
    order, so layer 24's edit is inside layer 25's live median. That
    compounding IS the band policy -- it is why 17 layers at alpha_rel 0.09 is
    not the same as one layer at 17 x 0.09.
    """
    missing = [l for l in band_layers if l not in vectors_by_layer]
    if missing:
        raise KeyError(
            f"no vectors for band layers {missing}; the band is "
            f"{list(band_layers)} and vectors cover "
            f"{sorted(vectors_by_layer)}")
    handles = []
    for layer in sorted(band_layers):
        handles.append(model.layers[layer].register_forward_hook(
            make_band_hook(
                vectors_by_layer[layer], strength_rel, positions, fired, layer,
                norm_mode=norm_mode,
                clean_norm=(clean_norms or {}).get(layer),
                median_scope=median_scope)))
    return handles


def injected_prefill_band(
    model,
    input_ids: torch.Tensor,
    band_layers: list[int],
    vectors_by_layer: dict[int, torch.Tensor] | None,   # {L: [B, d_model]}
    strength_rel,                                       # [B] or scalar
    positions,
    record_layers: list[int],
    record_positions,
    report_position: int = -1,
    norm_mode: str = "live",
    clean_norms: dict[int, float] | None = None,
    median_scope: str = "per_element",
) -> dict:
    """One prefill with the band intervention. Mirrors inject.injected_prefill.

    `vectors_by_layer=None` runs clean (no hooks) for the zero control.
    Returns residuals at record_layers x record_positions, logits at the
    report position, per-layer fire counts and realised displacements, and the
    handle count after teardown.
    """
    fired: dict[int, list[dict]] = {}
    handles = []
    if vectors_by_layer is not None:
        if strength_rel is None:
            raise ValueError("strength_rel required when vectors are given")
        handles = _register_band(
            model, band_layers, vectors_by_layer, strength_rel, positions,
            fired, norm_mode, clean_norms, median_scope)
    try:
        final_layer = model.n_layers - 1
        at = sorted({*record_layers, final_layer})
        with torch.inference_mode():
            with jlens.ActivationRecorder(model.layers, at=at) as recorder:
                model.forward(input_ids)
                residuals = {
                    l: recorder.activations[l][:, record_positions, :]
                        .detach().float()
                    for l in record_layers
                }
                final = recorder.activations[final_layer][:, report_position, :]
                logits = model.unembed(final.detach()).float()
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()

    return {
        "residuals": residuals,
        "logits": logits,
        "fired": fired,
        "n_fires_per_layer": {l: len(v) for l, v in fired.items()},
        "n_fires_total": sum(len(v) for v in fired.values()),
        "n_layers_fired": len(fired),
        "n_handles_after": len(handles),
    }


def generate_with_injection_band(
    model,
    hf_model,
    tokenizer,
    input_ids: torch.Tensor,
    band_layers: list[int],
    vectors_by_layer: dict[int, torch.Tensor] | None,
    strength_rel,
    positions,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    seed: int = 0,
    norm_mode: str = "live",
    clean_norms: dict[int, float] | None = None,
    median_scope: str = "per_element",
) -> dict:
    """Generate with the band applied during prefill only.

    Same argument as the single-layer version: the hooks stay registered
    across the whole `generate` call but refuse to fire on any single-token
    step, so they edit the prompt's residuals and nothing the model then
    writes. Each band layer must fire exactly once; the count is reported per
    layer rather than trusted, because a hook that silently fired on decode
    steps would look like introspection.
    """
    fired: dict[int, list[dict]] = {}
    handles = []
    if vectors_by_layer is not None:
        if strength_rel is None:
            raise ValueError("strength_rel required when vectors are given")
        handles = _register_band(
            model, band_layers, vectors_by_layer, strength_rel, positions,
            fired, norm_mode, clean_norms, median_scope)
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = hf_model.generate(
                input_ids=input_ids,
                # every row is the same prompt with no padding, so the mask is
                # all ones; passing it explicitly silences a transformers
                # warning and stops it guessing from a pad==eos tokenizer
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()

    completions = [
        tokenizer.decode(row[input_ids.shape[1]:], skip_special_tokens=True)
        for row in out
    ]
    lengths = [
        int((row[input_ids.shape[1]:] != tokenizer.eos_token_id).sum()) for row in out
    ]
    fires = {l: len(v) for l, v in fired.items()}
    return {
        "sequences": out,
        "completions": completions,
        "new_token_counts": lengths,
        "fired": fired,
        "n_fires_per_layer": fires,
        "n_fires_total": sum(fires.values()),
        "n_layers_fired": len(fires),
        "max_fires_per_layer": max(fires.values()) if fires else 0,
        "n_handles_after": len(handles),
    }


def realised_displacement(result: dict) -> dict[int, float]:
    """Median realised ||delta_l|| / base_l per layer, from a band result.

    The contract is ||delta_l|| == alpha_rel x base_l exactly, so this returns
    alpha_rel at every layer when the hook is correct. It is reported rather
    than asserted because a wrong value localises the fault to a layer.
    """
    out = {}
    for layer, records in result.get("fired", {}).items():
        ratios = [d / b if b else float("nan")
                  for record in records
                  for d, b in zip(record["delta_norm"], record["base"])]
        ratios = sorted(r for r in ratios if r == r)
        out[layer] = ratios[len(ratios) // 2] if ratios else float("nan")
    return out


# --------------------------------------------------------------------------
# arm B: the object Garcia actually injects
# --------------------------------------------------------------------------


def _unembed_parts(model):
    """(W_U weight [vocab, d_model], final-norm gain [d_model] or None).

    `jlens.HFLensModel` keeps these private, so reach for them once, here,
    with a legible failure -- rather than in three call sites with an
    AttributeError.
    """
    head = getattr(model, "_lm_head", None)
    norm = getattr(model, "_final_norm", None)
    if head is None or not hasattr(head, "weight"):
        raise AttributeError(
            "cannot find the unembedding on this model wrapper; expected "
            "jlens.HFLensModel._lm_head with a .weight")
    gain = getattr(norm, "weight", None) if norm is not None else None
    if gain is not None and "gemma" in type(norm).__name__.lower():
        # Gemma-family RMSNorm scales by (1 + weight); Qwen's is bare weight.
        gain = gain + 1.0
    return head.weight, gain


def unembed_rows(model, token_ids, fold_final_norm: bool = True) -> torch.Tensor:
    """Rows of W_U for `token_ids`, optionally through the final norm's gain.

    The readout is `unembed(J_l h) = W_U @ norm(J_l h)`, and this model's
    final norm is an RMSNorm: `norm(x) = g * x / rms(x)`. So

        logit_t = (g * W_U[t]) . (J_l h) / rms(J_l h)

    and the direction in layer-l space that raises logit_t is
    `J_l^T (g * W_U[t])`, up to a positive scalar. Folding `g` in is therefore
    the correct row; `fold_final_norm=False` gives the raw W_U row, which is
    what a reading of the lens that ignores the norm would produce. G2b
    reports the cosine between the two so the choice is visible.
    """
    weight, gain = _unembed_parts(model)
    ids = torch.as_tensor(list(token_ids), device=weight.device, dtype=torch.long)
    rows = weight.index_select(0, ids).float()
    if fold_final_norm and gain is not None:
        rows = rows * gain.to(rows.device).float()[None, :]
    return rows


def jlens_row_vectors(
    lens, model, token_ids, layer: int, fold_final_norm: bool = True
) -> torch.Tensor:
    """The J-lens direction for each token at `layer`: rows of (W_U J_l).

    readout(h) = W_U(J_l h), so logit_t responds to h along J_l^T (g * W_U[t]).
    That is the direction Garcia injects. Returned [B, d_model], unnormalised
    -- the hook normalises.

    `lens.jacobians[layer]` is the stored J_l itself (ASSETS.md 1), and
    `lens.transport` computes `h @ J^T`, i.e. `J h`; the row is therefore
    `W_U[t] @ J`, which is `(J^T W_U[t])^T`.
    """
    if layer not in lens.jacobians:
        raise KeyError(f"layer {layer} not fitted; source_layers = "
                       f"{lens.source_layers}")
    rows = unembed_rows(model, token_ids, fold_final_norm=fold_final_norm)
    jacobian = lens.jacobians[layer]
    return rows.to(jacobian.device).float() @ jacobian.float()


def jlens_row_vectors_numeric(
    model, lens, token_ids, layer: int, residual: torch.Tensor
) -> torch.Tensor:
    """The same direction, differentiated instead of derived. [B, d_model].

    d/dh of `unembed(J_l h)[t]` at `h = residual`, one row per token. This is
    the ground truth the analytic row is checked against in G2b, and unlike a
    hand-computed row it includes whatever the final norm actually does rather
    than what its docstring says it does. Slow (one backward per token), so it
    is a ten-token check, not a code path.
    """
    if residual.ndim != 1:
        raise ValueError(f"residual must be [d_model], got {tuple(residual.shape)}")
    out = []
    for token_id in token_ids:
        h = residual.detach().float().clone().requires_grad_(True)
        logit = model.unembed(lens.transport(h, layer))[int(token_id)]
        grad, = torch.autograd.grad(logit, h)
        out.append(grad.detach())
    return torch.stack(out)


def rows_by_layer(
    lens, model, concept_token_ids: dict[str, list[int]], layers: list[int],
    fold_final_norm: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """{layer: {concept: row}} for arm B, one J matmul per layer.

    A concept with several single-token variants (casing, leading space) gets
    the mean of their rows, normalised by the hook anyway -- the alternative,
    picking one variant, would make the arm depend on the tokenizer's
    preferred casing rather than on the concept.
    """
    words = [w for w, ids in concept_token_ids.items() if ids]
    flat, spans = [], {}
    for word in words:
        spans[word] = (len(flat), len(flat) + len(concept_token_ids[word]))
        flat.extend(concept_token_ids[word])
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        rows = jlens_row_vectors(lens, model, flat, layer,
                                 fold_final_norm=fold_final_norm)
        out[layer] = {w: rows[a:b].mean(dim=0) for w, (a, b) in spans.items()}
    return out
