"""The resumable sweep.

Grid: 60 concepts x 3 layers x 4 strengths x 2 orders x 3 conditions = 4,320
cells. One shard per concept, written atomically, skipped on restart.

Two things here are not in the build spec and both are load-bearing.

**The report position has to be constructed, not assumed.** "Cache next-token
logits at the report position" is unambiguous for report_then_task, where the
model's first output token is the detection verdict. It is not for
task_then_report, where the report lands somewhere inside the generation.
Reading both at the last prompt token would compare two different quantities
and call the difference an order effect. So the assistant turn is prefilled up
to the detection key in whichever order the protocol demands, which makes the
next token a JSON boolean in both arms:

    report_then_task   {"change_detected":
    task_then_report   {"task_answer": "<clean answer>", "change_detected":

The clean answer is the model's own greedy completion of the fixed task,
computed once per order, so the prefill is a constant and not a per-cell cost.

**Zero-strength cells are computed once per batch, not once per cell.**
alpha=0 adds exactly zero -- G2 measures this as the zero-strength identity
invariant and reports it as 0.0 -- so the four strengths of a control_zero
condition are bit-identical. Computing one and broadcasting is exact, not an
approximation, and removes a third of the forward passes. It is done per
(batch, layer, order) rather than once globally precisely so G4a still has a
per-batch control statistic to test for temporal drift.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import inject
import prompts as prompt_mod

CONDITIONS = ("injected", "control_zero", "control_random")


def build_cells(layers, strengths, orders) -> list[dict]:
    """Every (layer, strength, order, condition) a single concept needs."""
    cells = []
    for layer in layers:
        for order in orders:
            for condition in CONDITIONS:
                for strength in strengths:
                    cells.append({"layer": int(layer), "strength": float(strength),
                                  "order": order, "condition": condition})
    return cells


def protocol_for(order: str) -> str:
    for protocol, expected in prompt_mod.PROTOCOL_ORDER.items():
        if expected == order:
            return protocol
    raise ValueError(f"no protocol for order {order!r}")


def clean_task_answer(model, hf_model, tokenizer, task: str, max_new_tokens: int = 12) -> str:
    """The model's own greedy answer to the fixed task, used in the prefill.

    Computed once per order so the task-first prefill is a constant. Kept short
    and stripped of quotes so it can sit inside a JSON string literal.
    """
    messages = [{"role": "user", "content": task}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = model.encode(rendered)
    with torch.inference_mode():
        out = hf_model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0].replace('"', "'").strip()[:80]


def sweep_prompt(model, tokenizer, order: str, task: str, clean_answer: str,
                 inject_width: int, tail_offset: int):
    """Prompt whose next token is the detection boolean, in either order.

    Returns (input_ids [1, seq], injection positions slice, rendered text).
    """
    protocol = protocol_for(order)
    messages = prompt_mod.build_messages(task, order, protocol)
    rendered = prompt_mod.render(tokenizer, messages, prefill=False,
                                 enable_thinking=False)
    if order == "report_then_task":
        prefill = '{"change_detected":'
    else:
        prefill = '{"task_answer": "' + clean_answer + '", "change_detected":'
    rendered = rendered + prefill
    ids = model.encode(rendered)
    # Garcia's all_user policy: inject over every token of the task turn.
    positions = prompt_mod.user_positions(tokenizer, rendered, task)
    positions = [i for i in positions if i < int(ids.shape[1])]
    if not positions:
        raise ValueError("user span fell outside the encoded prompt")
    return ids, positions, rendered


def boolean_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Re-exported from prompts, which owns the frame that forces them."""
    return prompt_mod.boolean_token_ids(tokenizer)


def shard_path(out_dir: Path, concept: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in concept)
    return Path(out_dir) / f"shard_{safe}.npz"


def config_fingerprint(layers, strengths, orders, readout_layers, task) -> str:
    """Identifies the configuration a shard was produced under."""
    import hashlib

    payload = json.dumps({"layers": sorted(layers), "strengths": sorted(strengths),
                          "orders": sorted(orders),
                          "readout_layers": sorted(readout_layers),
                          "task": task}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def completed_concepts(out_dir: Path, concepts: list[str],
                       fingerprint: str | None = None) -> set[str]:
    """Concepts whose shard exists AND was produced under `fingerprint`.

    Resuming on filename alone is not enough. Change the injection layer or
    the strength ladder and every existing shard becomes stale, but the names
    do not change -- the sweep would skip them and the run would silently mix
    two configurations.
    """
    done = set()
    for concept in concepts:
        path = shard_path(out_dir, concept)
        if not path.exists():
            continue
        if fingerprint is None:
            done.add(concept)
            continue
        try:
            blob = np.load(path, allow_pickle=False)
            if str(blob["fingerprint"]) == fingerprint:
                done.add(concept)
        except Exception:  # noqa: BLE001  unreadable or pre-fingerprint shard
            continue
    return done


def write_shard_atomic(path: Path, **arrays) -> None:
    """Write via a temp file and rename, so a kill never leaves a half shard.

    The temp name has to END in .npz: numpy.savez silently appends .npz when
    the filename does not, so a ".npz.tmp" temp wrote ".npz.tmp.npz" and the
    rename then looked for a file that was never created.
    """
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run_batch(
    model,
    tokenizer,
    concepts_batch: list[str],
    vectors_by_layer: dict,
    randoms_by_layer: dict,
    cells: list[dict],
    cache_layers: list[int],
    prompts_by_order: dict,
    true_ids: torch.Tensor,
    false_ids: torch.Tensor,
    topk: int,
) -> dict:
    """All cells for a batch of concepts. Returns {concept: arrays}.

    Batched by concept: every element of the batch shares one `input_ids`, so
    B concepts cost one forward. This is the difference between a 3-hour sweep
    and a 30-hour one.
    """
    device = model.input_device
    batch_size = len(concepts_batch)
    n_cells = len(cells)
    d_model = model.d_model
    n_cached = len(cache_layers)

    residuals = np.zeros((batch_size, n_cells, n_cached, d_model), dtype=np.float32)
    p_true = np.zeros((batch_size, n_cells), dtype=np.float64)
    p_false = np.zeros((batch_size, n_cells), dtype=np.float64)
    top_ids = np.zeros((batch_size, n_cells, topk), dtype=np.int32)
    top_logits = np.zeros((batch_size, n_cells, topk), dtype=np.float32)
    cell_seconds = np.zeros(n_cells, dtype=np.float64)

    # zero-strength is bit-identical across strengths; compute once per
    # (layer, order) in this batch and reuse. Kept per-batch so G4a can test
    # for drift across the run.
    zero_cache: dict[tuple[int, str], tuple] = {}

    def record(idx: int, result) -> None:
        final = result["logits"].float()
        probs = torch.softmax(final, dim=-1)
        p_true[:, idx] = probs.index_select(-1, true_ids).sum(-1).cpu().numpy()
        p_false[:, idx] = probs.index_select(-1, false_ids).sum(-1).cpu().numpy()
        values, indices = final.topk(topk, dim=-1)
        top_ids[:, idx] = indices.cpu().numpy()
        top_logits[:, idx] = values.cpu().numpy()
        for j, layer in enumerate(cache_layers):
            residuals[:, idx, j, :] = result["residuals"][layer][:, 0, :].cpu().numpy()

    for idx, cell in enumerate(cells):
        started = time.time()
        layer, order = cell["layer"], cell["order"]
        ids, positions, median_norm = prompts_by_order[order]
        batch_ids = ids.expand(batch_size, -1).contiguous()
        report_positions = [-1]

        if cell["condition"] == "control_zero":
            key = (layer, order)
            if key not in zero_cache:
                zero_cache[key] = inject.injected_prefill(
                    model, batch_ids, layer, None, None, positions,
                    record_layers=cache_layers, record_positions=report_positions)
            record(idx, zero_cache[key])
            cell_seconds[idx] = time.time() - started
            continue

        source = (vectors_by_layer if cell["condition"] == "injected"
                  else randoms_by_layer)[layer]
        vecs = torch.stack([source[c] for c in concepts_batch]).to(device)
        alpha = torch.full((batch_size,), cell["strength"] * median_norm[layer],
                           device=device)
        result = inject.injected_prefill(
            model, batch_ids, layer, vecs, alpha, positions,
            record_layers=cache_layers, record_positions=report_positions)
        record(idx, result)
        cell_seconds[idx] = time.time() - started

    return {
        concept: {
            "residuals": residuals[i],
            "p_true": p_true[i],
            "p_false": p_false[i],
            "top_ids": top_ids[i],
            "top_logits": top_logits[i],
            "cell_seconds": cell_seconds,
        }
        for i, concept in enumerate(concepts_batch)
    }


def cells_to_arrays(cells: list[dict]) -> dict:
    return {
        "cell_layer": np.array([c["layer"] for c in cells], dtype=np.int32),
        "cell_strength": np.array([c["strength"] for c in cells], dtype=np.float32),
        "cell_order": np.array([c["order"] for c in cells]),
        "cell_condition": np.array([c["condition"] for c in cells]),
    }


def load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}
