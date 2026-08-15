"""Batched injection harness.

Four requirements from the build spec, and what each costs if it is wrong:

2.1  The prompt never changes; only the vector does. Every concept in a batch
     shares one `input_ids`, so B concepts go through ONE forward pass with a
     per-batch-element hook. This is the difference between a 3-hour sweep and
     a 30-hour one.
2.2  Norm normalisation is mandatory. alpha_abs = alpha_rel * median clean
     residual norm at the injection positions, cached per layer. A raw alpha
     is not comparable across layers. The contract is
     ||delta|| == alpha_rel * median_residual_norm, which requires the
     steering direction to be a unit vector -- make_hook normalises it, and
     G2 checks the realised norm ratio against sqrt(1 + alpha_rel^2).
2.3  Prefill only. The injection is an edit to the prompt's residuals; it must
     not fire while decoding cached tokens.
2.4  Cache selectively -- residuals at the injection and report positions
     only, plus next-token logits at the report position. Caching every
     position OOMs and shrinks the batch.
"""

from __future__ import annotations

import torch

import jlens


def median_residual_norm(model, input_ids: torch.Tensor, layer: int, positions) -> float:
    """Median L2 norm of the clean residual at `positions` of `layer`.

    The scale that makes alpha_rel comparable across layers. Cache it per
    (layer, positions) and never pass a raw alpha downstream.
    """
    with torch.inference_mode():
        with jlens.ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(input_ids)
            residual = recorder.activations[layer][:, positions, :].detach().float()
    return float(residual.norm(dim=-1).median())


def make_hook(
    vectors: torch.Tensor,
    alpha_abs: torch.Tensor,
    positions,
    fired: list,
    prefill_only: bool = True,
):
    """A per-batch-element additive hook.

    vectors:   [B, d_model]  one steering vector per batch element
    alpha_abs: [B]           per-element strength, ALREADY norm-normalised
    positions: token slice or index list to inject over
    fired:     mutable list the hook appends to on every firing, so callers
               can assert it fired exactly once (see `prefill_only`)

    `prefill_only` skips any call whose sequence length is 1, which is what a
    cached decode step looks like. The build spec phrases this as "register,
    prefill, remove, then generate", but doing that literally means handing a
    half-built cache back to `generate`, and this model's cache is hybrid --
    48 of 64 blocks are linear-attention and carry recurrent state rather than
    keys and values. Guarding on sequence length gets the same semantics
    without cache surgery, and the fire count makes it checkable rather than
    assumed.
    """

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if prefill_only and h.shape[1] == 1:
            return output
        if h.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"batch mismatch: hidden {h.shape[0]} vs vectors {vectors.shape[0]}")
        if h.shape[-1] != vectors.shape[-1]:
            raise ValueError(
                f"d_model mismatch: hidden {h.shape[-1]} vs vectors {vectors.shape[-1]}")
        # The direction must be a UNIT vector for alpha_abs to mean what 2.2
        # says it means. alpha_abs is alpha_rel x the median clean residual
        # norm, so the intended perturbation size is ||delta|| = alpha_abs;
        # that only holds if ||v|| == 1. Concept vectors are
        # activation(word) - mean(baseline) and come out with ||v|| ~ 11 on
        # this model, so scaling the raw vector injected ~11x too hard at the
        # lowest rung and ~87x at the top -- G2 measured exactly that, and
        # every probability in the dose-response table was driven to 0.
        unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Cast the delta to the hidden dtype. Without this the in-place add
        # raises: fp32 vectors against a bf16 residual promote to fp32, which
        # is not the destination dtype.
        delta = (alpha_abs[:, None, None] * unit[:, None, :]).to(h.dtype).to(h.device)
        h[:, positions, :] += delta
        fired.append(int(h.shape[1]))
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


def injected_prefill(
    model,
    input_ids: torch.Tensor,
    layer: int,
    vectors: torch.Tensor | None,
    alpha_abs: torch.Tensor | None,
    positions,
    record_layers: list[int],
    record_positions,
    report_position: int = -1,
) -> dict:
    """One prefill pass, optionally injected. Returns selectively cached state.

    `vectors=None` runs clean, with no hook registered at all -- which is what
    the zero-strength identity invariant compares against.

    Returns:
        residuals: {layer: [B, n_record_positions, d_model]}
        logits:    [B, vocab] at `report_position`
        n_fires:   how many times the hook fired (0 when clean)
        n_handles_after: hook handles still registered after teardown
    """
    fired: list[int] = []
    handles = []
    if vectors is not None:
        if alpha_abs is None:
            raise ValueError("alpha_abs is required when vectors is given")
        handles.append(model.layers[layer].register_forward_hook(
            make_hook(vectors, alpha_abs, positions, fired, prefill_only=True)))

    try:
        final_layer = model.n_layers - 1
        at = sorted({*record_layers, final_layer})
        with torch.inference_mode():
            with jlens.ActivationRecorder(model.layers, at=at) as recorder:
                model.forward(input_ids)
                residuals = {
                    l: recorder.activations[l][:, record_positions, :].detach().float()
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
        "n_fires": len(fired),
        "n_handles_after": len(handles),
    }


def generate_with_injection(
    model,
    hf_model,
    tokenizer,
    input_ids: torch.Tensor,
    layer: int,
    vectors: torch.Tensor | None,
    alpha_abs: torch.Tensor | None,
    positions,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict:
    """Generate with the injection applied during prefill only.

    The hook is registered across the `generate` call but refuses to fire on
    any single-token step, so it edits the prompt's residuals and nothing the
    model then writes. `n_fires` must be 1; the gate reports it rather than
    trusting it.
    """
    fired: list[int] = []
    handles = []
    if vectors is not None:
        handles.append(model.layers[layer].register_forward_hook(
            make_hook(vectors, alpha_abs, positions, fired, prefill_only=True)))
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
    return {
        "sequences": out,
        "completions": completions,
        "new_token_counts": lengths,
        "n_fires": len(fired),
        "n_handles_after": len(handles),
    }


def sequence_nll(model, sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Mean negative log-likelihood of the generated span, scored CLEAN.

    Locates where "brain damage" begins: a completion produced under strong
    steering is scored by the unhooked model, so rising NLL means the text is
    drifting off-distribution rather than the steering merely being visible.
    """
    with torch.inference_mode():
        final_layer = model.n_layers - 1
        with jlens.ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(sequences)
            hidden = recorder.activations[final_layer].detach()
        logits = model.unembed(hidden).float()
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = sequences[:, 1:]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    span = picked[:, prompt_len - 1:]
    return -span.mean(dim=-1)
