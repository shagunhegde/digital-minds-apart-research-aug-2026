"""Lens loading and readout.

Thin layer over `jlens`. The build spec sketched these signatures against a
lens shaped [n_layers, n_vocab, d_model]; the real stored object is J_l,
[d_model, d_model], and the vocabulary enters through the model's own
unembedding (ASSETS.md §1). So a readout is:

    unembed(J_l @ h)  ==  model.unembed(lens.transport(h, layer))

and `token_rank` is the same thing scored at one vocabulary entry. That is the
f2 primitive, and it is why the harness can read *modified* residuals:
`JacobianLens.transport` takes a tensor, not a prompt.

Everything here is a flat function taking explicit arguments. Nothing caches
global state.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import jlens

# Categories whose readout position is the final prompt token. The published
# convention (data/evaluations/README.md) reads multihop / multilingual /
# order-ops at "the token immediately preceding `target`" -- and since `target`
# never appears in the prompt for these sets, that is the last prompt token.
# association reads the closing period and typo the last fragment of the
# misspelling, which are also the last prompt token.
#
# poetry (last newline, not the final token) and order-ops (intermediates are
# synonym sets, scored as a min over synonyms) need handling this module does
# not implement; they are excluded rather than silently mis-scored.
FINAL_TOKEN_CATEGORIES = ("multihop", "multilingual", "association", "typo")


def load_eval_items(
    data_dir: Path, categories: tuple[str, ...] = FINAL_TOKEN_CATEGORIES
) -> list[dict]:
    """Load the published lens-eval sets, tagged with their category.

    `data_dir` is `data/evaluations` from a checkout of
    github.com/anthropics/jacobian-lens. Items keep their published fields:
    `name`, `prompt`, `target`, `intermediates`.
    """
    items: list[dict] = []
    for category in categories:
        path = data_dir / f"lens-eval-{category}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing eval set: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            items.append({**item, "category": category})
    return items


def load_lens(
    repo: str,
    filename: str,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    layers: list[int] | None = None,
) -> jlens.JacobianLens:
    """Load a fitted lens and place the requested layers on `device`.

    `JacobianLens.__init__` casts to fp32, so all 63 layers resident is
    ~6.6 GB. `transport` moves J to the residual's device on every call, which
    at 105 MB per layer is a transfer per readout; placing the swept layers
    once avoids that.
    """
    lens = jlens.JacobianLens.from_pretrained(repo, filename=filename, revision=revision)
    for layer in layers if layers is not None else lens.source_layers:
        if layer not in lens.jacobians:
            raise KeyError(
                f"layer {layer} not fitted; source_layers = {lens.source_layers}"
            )
        lens.jacobians[layer] = lens.jacobians[layer].to(device)
    return lens


def random_rotation(d_model: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """A Haar-ish random orthogonal matrix, from the QR of a Gaussian."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(d_model, d_model, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian)
    # Sign-fix the diagonal so Q is distributed uniformly over O(d).
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def transport(
    lens: jlens.JacobianLens,
    residual: torch.Tensor,
    layer: int,
    mode: str = "jacobian",
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a residual into the final-layer basis under one of three lenses.

    mode:
      "jacobian" -- J_l @ h, the lens under test
      "logit"    -- h unchanged, the vanilla logit-lens baseline (what
                    `JacobianLens.apply(use_jacobian=False)` does)
      "null"     -- Q @ (J_l @ h) for a fixed random orthogonal Q

    The null is applied *after* transport rather than by rotating J itself:
    Q @ (J_l h) == (Q J_l) h exactly, and this way one [d, d] matrix stands in
    for 63 rotated copies of J. Left-multiplying preserves the singular
    spectrum of J_l exactly while destroying its alignment with the
    unembedding basis, which is the thing the null is supposed to break. If
    readouts survive it, the code is reading through W_U alone and the lens is
    contributing nothing.
    """
    if residual.shape[-1] != lens.d_model:
        raise ValueError(
            f"residual d_model {residual.shape[-1]} != lens d_model {lens.d_model}"
        )
    if mode == "logit":
        return residual
    j = lens.jacobians[layer]
    transported = residual.to(j.dtype).to(j.device) @ j.T
    if mode == "jacobian":
        return transported
    if mode == "null":
        if rotation is None:
            raise ValueError("mode='null' needs a rotation matrix")
        return transported @ rotation.T
    raise ValueError(f"unknown mode {mode!r}")


def readout(
    model,
    lens: jlens.JacobianLens,
    residual: torch.Tensor,
    layer: int,
    k: int = 10,
    mode: str = "jacobian",
    rotation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k vocabulary entries for a residual. Returns (ids, scores)."""
    if residual.shape[-1] != lens.d_model:
        raise ValueError(
            f"residual d_model {residual.shape[-1]} != lens d_model {lens.d_model}"
        )
    logits = model.unembed(transport(lens, residual, layer, mode, rotation)).float()
    scores, ids = logits.topk(k, dim=-1)
    return ids, scores


def ranks_of(
    logits: torch.Tensor, token_ids: torch.Tensor, chunk: int = 16
) -> torch.Tensor:
    """1-indexed full-vocabulary rank of each id. [..., vocab] -> [..., T].

    Counting strictly-greater logits gives every member of a tied block the
    same best rank, which is deterministic; an argsort-based rank (as in the
    private `jlens.vis._ranks_of`) breaks ties by position.

    Chunked over targets because the comparison broadcasts to
    [..., vocab, T]: at 63 layers, a 248k vocabulary and 200 random probe
    tokens that single tensor would be 3.1 GB.
    """
    pieces = []
    for start in range(0, int(token_ids.numel()), chunk):
        picked = logits.index_select(-1, token_ids[start:start + chunk])
        pieces.append((logits.unsqueeze(-1) > picked.unsqueeze(-2)).sum(-2))
    return torch.cat(pieces, dim=-1) + 1


def token_rank(
    model,
    lens: jlens.JacobianLens,
    residual: torch.Tensor,
    layer: int,
    token_ids: torch.Tensor,
    mode: str = "jacobian",
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-vocabulary rank of each token id at one layer. The f2 primitive.

    Ranks carry far more information than a binary hit, and are the metric
    Garcia's argument turns on.
    """
    logits = model.unembed(transport(lens, residual, layer, mode, rotation)).float()
    return ranks_of(logits, token_ids)


def residuals_at(model, input_ids: torch.Tensor, layers: list[int], position: int):
    """Residuals at `position` for each requested block: {layer: [B, d_model]}.

    One forward pass. `position` is a Python index into the sequence.
    """
    with jlens.ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        return {
            layer: recorder.activations[layer][:, position, :].detach().float()
            for layer in layers
        }


def token_variants(word: str) -> list[str]:
    """The casing / leading-space variants a concept name can tokenize as."""
    bare = word.strip()
    lowered = bare[:1].lower() + bare[1:]
    titled = bare[:1].upper() + bare[1:]
    seen, out = set(), []
    for variant in (bare, lowered, titled, f" {bare}", f" {lowered}", f" {titled}"):
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def single_token_ids(tokenizer, word: str) -> list[int]:
    """Ids of every variant of `word` that is a single token.

    Returns [] when no variant is single-token. Callers score a word by the
    best rank over its variants; the count of words with no single-token
    variant is an invariant worth reporting, not a silent drop.
    """
    ids: list[int] = []
    for variant in token_variants(word):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def is_trash_token(tokenizer, token_id: int, special_ids: set[int]) -> bool:
    """Whether a readout entry is a special token, byte fragment, or non-word.

    The R-lens paper's qualitative complaint, made countable. A token counts
    as trash when it is a special/added token, decodes to nothing printable,
    contains the U+FFFD replacement character (an incomplete UTF-8 byte
    fragment from byte-level BPE), or contains no alphanumeric character at
    all.
    """
    if token_id in special_ids:
        return True
    decoded = tokenizer.decode([token_id])
    if not decoded.strip():
        return True
    if "�" in decoded:
        return True
    return not any(ch.isalnum() for ch in decoded)
