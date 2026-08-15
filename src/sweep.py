"""The resumable sweep.

Grid: 60 concepts x 4 strengths x 2 orders x (2 arms + zero + random) = 1,920
cells. One shard per concept, written atomically, skipped on restart.

The layer dimension is gone: under `injection_policy: workspace_band` there is
one intervention per layer across all 17 band layers at once, so "which layer"
is no longer a coordinate of the grid. Two dimensions replaced it -- the
vector arm (concept vs J-lens row, on `injected` cells only; the controls are
shared) and the task, which is not a grid coordinate but a per-concept
assignment.

Four things here are not in the build spec and all four are load-bearing.

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
condition are bit-identical, and so are the two arms of it. Computing one and
broadcasting is exact, not an approximation. It is done per (batch, order)
rather than once globally precisely so G4a still has a per-batch control
statistic to test for temporal drift.

**A batch is a task group, not an arbitrary slice of the concept list.**
Every element of a batch shares one `input_ids`, which is what makes B
concepts cost one forward. Once concept i takes task i % n_tasks, concepts
that share a task are the only ones that can share a forward -- so the batches
are the task groups (6 concepts each at 60 concepts and 10 tasks), and the
prompt cache is keyed by (order, task): 20 entries rather than 2.

**The strength passed to the band hook is alpha_REL, not alpha_abs.** The
single-layer path multiplies by a cached clean median norm before the hook
sees it. The band reads its own median live, inside each layer, after earlier
band layers have fired -- that compounding is the policy. Passing an alpha_abs
here would silently reinstate clean-norm scaling at 17 layers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import band_inject
import prompts as prompt_mod

CONDITIONS = ("injected", "control_zero", "control_random")

#: The arm label on cells that have no arm. Both controls are shared between
#: the two vector arms -- alpha=0 adds exactly zero whatever the vector is,
#: and one matched-norm random set is the null for both -- so they are run
#: once and read by either arm's cascade.
SHARED_ARM = "shared"


def build_cells(strengths, orders, arms) -> list[dict]:
    """Every (strength, order, condition, arm) a single concept needs.

    `injected` cells are per arm; the controls are shared. At 4 strengths, 2
    orders and 2 arms that is 4 x 2 x (2 + 1 + 1) = 32 cells per concept.
    """
    cells = []
    for order in orders:
        for condition in CONDITIONS:
            cell_arms = list(arms) if condition == "injected" else [SHARED_ARM]
            for arm in cell_arms:
                for strength in strengths:
                    cells.append({"strength": float(strength), "order": order,
                                  "condition": condition, "arm": arm})
    return cells


def assign_tasks(concepts: list[str], tasks: list[str]) -> dict[str, str]:
    """Concept i takes task i % len(tasks), in every one of its cells.

    Round-robin rather than random so the map is reproducible from the concept
    list alone, and so each task carries the same number of concepts. Holding
    the task fixed WITHIN a concept is what keeps the injected and control
    cells of that concept paired: the contrast is the injection, not the task.
    """
    if not tasks:
        raise ValueError("no task prompts to assign")
    return {c: tasks[i % len(tasks)] for i, c in enumerate(concepts)}


def task_groups(concepts: list[str], task_of: dict[str, str],
                batch: int) -> list[tuple[str, list[str]]]:
    """[(task, [concepts])], chunked to `batch`. A batch shares one prompt."""
    grouped: dict[str, list[str]] = {}
    for concept in concepts:
        grouped.setdefault(task_of[concept], []).append(concept)
    out = []
    for task in sorted(grouped):
        members = grouped[task]
        for begin in range(0, len(members), batch):
            out.append((task, members[begin:begin + batch]))
    return out


def protocol_for(order: str) -> str:
    for protocol, expected in prompt_mod.PROTOCOL_ORDER.items():
        if expected == order:
            return protocol
    raise ValueError(f"no protocol for order {order!r}")


def clean_task_answer(model, hf_model, tokenizer, task: str, max_new_tokens: int = 12) -> str:
    """The model's own greedy answer to one task, used in the prefill.

    Computed once per task -- ten greedy 12-token generations for the whole
    run -- so the task-first prefill is a constant per task and not a per-cell
    cost. Kept short and stripped of quotes so it can sit inside a JSON string
    literal.
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


def report_prefill(order: str, clean_answer: str,
                   task_first_prefill: str = "clean") -> str:
    """The assistant prefill that puts the next token where we read it.

    report_then_task stops at the detection key, so the next token is a JSON
    boolean and the model still writes its own task answer afterwards.

    task_then_report has two modes, and the difference is a whole experimental
    condition:

      clean    quote the model's own CLEAN greedy answer, then stop at the
               detection key. The next token is a boolean in this order too,
               which is what makes the order contrast a contrast of one
               quantity. But the model never writes a steered answer -- so
               this is NOT Garcia's task-first condition, it is the
               clean-substitution cell of the output-substitution 2x2, and
               comparing its report rate to his 322 is comparing two different
               experiments.
      natural  prefill only the opening of the answer and let the model write
               it, steered or not, before it reaches the detection keys. This
               IS Garcia's condition. The report position is then wherever the
               model puts it, so detection must be read from the parsed JSON
               rather than from a next-token logit.
    """
    if order == "report_then_task":
        return '{"change_detected":'
    if task_first_prefill == "natural":
        return '{"task_answer": "'
    if task_first_prefill != "clean":
        raise ValueError(f"unknown task_first_prefill {task_first_prefill!r}")
    return '{"task_answer": "' + clean_answer + '", "change_detected":'


def prompt_with_prefill(model, tokenizer, order: str, task: str, prefill: str):
    """The two-order protocol prompt, continued by `prefill`.

    Returns (input_ids [1, seq], injection positions, rendered text). The
    injection positions are located by character offset inside the USER turn,
    so they do not move when the assistant prefill changes length -- which is
    what lets the boolean slot and the naming slot be read off the same
    intervention.
    """
    protocol = protocol_for(order)
    messages = prompt_mod.build_messages(task, order, protocol)
    rendered = prompt_mod.render(tokenizer, messages, prefill=False,
                                 enable_thinking=False)
    rendered = rendered + prefill
    ids = model.encode(rendered)
    # Garcia's all_user policy: inject over every token of the task turn.
    positions = prompt_mod.user_positions(tokenizer, rendered, task)
    positions = [i for i in positions if i < int(ids.shape[1])]
    if not positions:
        raise ValueError("user span fell outside the encoded prompt")
    return ids, positions, rendered


def sweep_prompt(model, tokenizer, order: str, task: str, clean_answer: str,
                 task_first_prefill: str = "clean"):
    """Prompt whose next token is the detection BOOLEAN, in either order."""
    return prompt_with_prefill(
        model, tokenizer, order, task,
        report_prefill(order, clean_answer, task_first_prefill))


def naming_prefill(order: str, clean_answer: str) -> str:
    """Assistant prefill whose next token is the model's NAME for the concept.

    f2 asks whether the concept is verbalizable at the report position. Read
    at the boolean slot -- the token after `"change_detected":` -- the answer
    is fixed by the grammar: the next token there must be `true` or `false`,
    and a concept word has no business in any top-k. f2 = 0 at k=50 is then
    close to guaranteed by the position, not a finding about the model.

    This prefill walks one key further, to the slot where a word is what the
    grammar demands. It asserts `change_detected: true` on the way, which is
    deliberate: the counterfactual verbalizability asks "IF it tried to name
    the concept, could it", so the detection frame is conditioned on rather
    than measured. That is Macar's forced-identification read from logits
    (arXiv 2603.21396), not a new construct -- cite it, do not claim it.

    G2b already validated the operationalisation: at the naming-style probe
    slot P(target) reached 0.31-0.41 for the jlens arm at alpha 0.05-0.09, so
    the machinery is known to light up when the position is right.
    """
    if order == "report_then_task":
        return '{"change_detected": true, "detected_concept": "'
    return ('{"task_answer": "' + clean_answer + '", '
            '"change_detected": true, "detected_concept": "')


def naming_prompt(model, tokenizer, order: str, task: str, clean_answer: str):
    """Prompt whose next token is the model's name for the injected concept."""
    return prompt_with_prefill(
        model, tokenizer, order, task, naming_prefill(order, clean_answer))


def build_prompt_cache(model, hf_model, tokenizer, orders, tasks, band_layers,
                       task_first_prefill: str = "clean"):
    """{(order, task): (ids, positions, clean_norms)}, plus {task: answer}, meta.

    Twenty entries at 2 orders and 10 tasks, built once and shared by the
    sweep, the generation pass and the factor pass -- if those three built
    their own the "same trial" that f1, f2 and f3 are read off would not be
    the same prompt, and cascade_residual would be measuring the mismatch.

    `clean_norms` is the per-band-layer median clean residual norm at the
    injection positions. Under Garcia's live scaling the hook does not use it;
    it is the denominator the reports divide realised displacement by, and the
    scale the `norm_mode="clean"` ablation runs at.
    """
    answers = {task: clean_task_answer(model, hf_model, tokenizer, task)
               for task in tasks}
    cache, meta = {}, {}
    for task in tasks:
        for order in orders:
            ids, positions, rendered = sweep_prompt(
                model, tokenizer, order, task, answers[task], task_first_prefill)
            norms = band_inject.band_median_norms(
                model, ids, band_layers, positions)
            cache[(order, task)] = (ids, positions, norms)
            meta[f"{order}|{task}"] = {
                "task": task,
                "order": order,
                "task_first_prefill": task_first_prefill,
                "clean_task_answer": answers[task],
                "seq_len": int(ids.shape[1]),
                "n_injected_positions": len(positions),
                "position_span": [int(positions[0]), int(positions[-1]) + 1],
                "clean_band_norms": norms,
                "rendered_tail": rendered[-220:],
            }
    return cache, answers, meta


def build_naming_cache(model, tokenizer, orders, tasks, answers, band_layers):
    """{(order, task): (ids, positions, clean_norms)} for the NAMING slot.

    Same intervention, same injected positions, one key further into the JSON
    object -- see `naming_prefill`. Takes the answers the report cache already
    computed rather than regenerating them, so the two frames quote the same
    clean answer and differ only in where they stop.
    """
    cache = {}
    for task in tasks:
        for order in orders:
            ids, positions, _rendered = naming_prompt(
                model, tokenizer, order, task, answers[task])
            norms = band_inject.band_median_norms(
                model, ids, band_layers, positions)
            cache[(order, task)] = (ids, positions, norms)
    return cache


def boolean_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Re-exported from prompts, which owns the frame that forces them."""
    return prompt_mod.boolean_token_ids(tokenizer)


def load_arm(vectors_dir: Path, arm: str, layers: list[int], device):
    """{layer: {concept: [d_model]}} for one arm, as 01_concept_vectors wrote it.

    Arm A lives in `vectors_layer{L}.pt` and arm B in `jlens_rows_layer{L}.pt`;
    both carry the same {arm, layer, concepts, vectors} shape, so everything
    downstream treats them interchangeably and the arm is a label rather than a
    code path.
    """
    stem = "vectors" if arm == "concept" else "jlens_rows"
    out = {}
    for layer in layers:
        path = Path(vectors_dir) / f"{stem}_layer{layer}.pt"
        if not path.exists():
            raise SystemExit(
                f"{path} not found -- run scripts/01_concept_vectors.py "
                f"(arm {arm!r} needs one file per band layer)")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out[layer] = {w: v.to(device) for w, v in blob["vectors"].items()}
    return out


def shard_path(out_dir: Path, concept: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in concept)
    return Path(out_dir) / f"shard_{safe}.npz"


def config_fingerprint(injection_policy, band_layers, strengths, orders,
                       readout_layers, tasks, vector_arms, norm_mode) -> str:
    """Identifies the configuration a shard was produced under.

    Everything that changes what a cell MEANS belongs in here. The policy,
    band and norm_mode are in because the sprint's central error was two runs
    whose per-layer alpha matched and whose effective strength differed ~20x:
    shards from those two runs are not comparable and their filenames are
    identical. The arms are in because an `injected` cell is a different
    object per arm, and the task list is in because the concept->task map is
    derived from it.
    """
    import hashlib

    payload = json.dumps({"injection_policy": injection_policy,
                          "band_layers": sorted(int(l) for l in band_layers),
                          "strengths": sorted(strengths),
                          "orders": sorted(orders),
                          "readout_layers": sorted(readout_layers),
                          "tasks": list(tasks),
                          "vector_arms": sorted(vector_arms),
                          "norm_mode": norm_mode}, sort_keys=True)
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
    concepts_batch: list[str],
    vectors_by_arm: dict,
    randoms_by_layer: dict,
    cells: list[dict],
    band_layers: list[int],
    cache_layers: list[int],
    prompts_by_order: dict,
    true_ids: torch.Tensor,
    false_ids: torch.Tensor,
    topk: int,
    norm_mode: str = "live",
    median_scope: str = "per_element",
) -> dict:
    """All cells for a batch of concepts. Returns {concept: arrays}.

    Batched by concept: every element of the batch shares one `input_ids`, so
    B concepts cost one forward. This is the difference between a 3-hour sweep
    and a 30-hour one. Since the task diversity landed, a batch is one task
    group -- concepts on different tasks have different prompts and cannot
    share a forward.

    `vectors_by_arm` is {arm: {layer: {concept: [d_model]}}} and
    `randoms_by_layer` is {layer: {concept: [d_model]}}, shared by both arms.
    `prompts_by_order` is {order: (ids, positions, clean_norms_by_layer)} for
    THIS batch's task.
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
    # realised displacement per cell: median over band layers of
    # ||delta_l|| / base_l, which the hook's contract fixes at alpha_rel.
    # Stored so G4a can check the band actually fired at the strength the cell
    # claims, rather than trusting the label.
    cell_fires = np.zeros(n_cells, dtype=np.int32)
    cell_delta_ratio = np.full(n_cells, np.nan, dtype=np.float32)

    # Stack each arm's vectors once per batch rather than once per cell: at 17
    # band layers a per-cell stack is 32 x 17 copies of the same tensor.
    stacked_by_arm = {
        arm: {layer: torch.stack([layers[layer][c] for c in concepts_batch]).to(device)
              for layer in band_layers}
        for arm, layers in vectors_by_arm.items()
    }
    stacked_random = {
        layer: torch.stack([randoms_by_layer[layer][c] for c in concepts_batch]).to(device)
        for layer in band_layers
    }

    # zero-strength is bit-identical across strengths AND arms; compute once
    # per order in this batch and reuse. Kept per-batch so G4a can test for
    # drift across the run.
    zero_cache: dict[str, dict] = {}

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
        cell_fires[idx] = result.get("n_fires_total", 0)
        ratios = sorted(band_inject.realised_displacement(result).values())
        if ratios:
            cell_delta_ratio[idx] = ratios[len(ratios) // 2]

    for idx, cell in enumerate(cells):
        started = time.time()
        order = cell["order"]
        ids, positions, clean_norms = prompts_by_order[order]
        batch_ids = ids.expand(batch_size, -1).contiguous()
        report_positions = [-1]

        if cell["condition"] == "control_zero":
            if order not in zero_cache:
                zero_cache[order] = band_inject.injected_prefill_band(
                    model, batch_ids, band_layers, None, None, positions,
                    record_layers=cache_layers, record_positions=report_positions)
            record(idx, zero_cache[order])
            cell_seconds[idx] = time.time() - started
            continue

        source = (stacked_by_arm[cell["arm"]] if cell["condition"] == "injected"
                  else stacked_random)
        # alpha_REL, not alpha_abs: the band hook reads its own median live.
        alpha_rel = torch.full((batch_size,), float(cell["strength"]), device=device)
        result = band_inject.injected_prefill_band(
            model, batch_ids, band_layers, source, alpha_rel, positions,
            record_layers=cache_layers, record_positions=report_positions,
            norm_mode=norm_mode, clean_norms=clean_norms,
            median_scope=median_scope)
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
            "cell_fires": cell_fires,
            "cell_delta_ratio": cell_delta_ratio,
        }
        for i, concept in enumerate(concepts_batch)
    }


def cells_to_arrays(cells: list[dict]) -> dict:
    return {
        "cell_strength": np.array([c["strength"] for c in cells], dtype=np.float32),
        "cell_order": np.array([c["order"] for c in cells]),
        "cell_condition": np.array([c["condition"] for c in cells]),
        "cell_arm": np.array([c["arm"] for c in cells]),
    }


def load_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}
