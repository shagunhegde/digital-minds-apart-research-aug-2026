"""Exercise the injection-position plumbing without a GPU.

Not a test suite -- a type-and-bounds check on the one thing that changed
shape. `positions` moved from a `slice` to a `list[int]` when Garcia's
all_user policy landed, and three consumers still called `.start` / `.stop`.
Each surfaced only when the stage that used it finally ran, one crash at a
time. This walks every prompt builder and asserts what consumers rely on.

    python scripts/00_verify_positions.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FakeIds:
    def __init__(self, n):
        self.shape = (1, n)


class FakeTok:
    bos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, **kw):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if kw.get("add_generation_prompt"):
            out += "<|im_start|>assistant\n"
        return out

    def __call__(self, text, return_offsets_mapping=False, truncation=True,
                 max_length=512):
        offs, i = [], 0
        for w in re.findall(r"\S+|\s+", text):
            if not w.isspace():
                offs.append((i, i + len(w)))
            i += len(w)
        return {"offset_mapping": offs}


class FakeModel:
    def __init__(self, tok):
        self.tokenizer = tok

    def encode(self, text):
        return FakeIds(len(self.tokenizer(text)["offset_mapping"]))


def check(name, ids, positions):
    assert isinstance(positions, list), f"{name}: positions is {type(positions)}"
    assert positions, f"{name}: empty positions"
    assert all(isinstance(i, int) for i in positions), f"{name}: non-int index"
    assert all(0 <= i < ids.shape[1] for i in positions), f"{name}: out of range"
    print(f"  {name:<18} seq={ids.shape[1]:>4}  n_positions={len(positions):>4}"
          f"  span=[{positions[0]}..{positions[-1]}]")


def main():
    import prompts
    import sweep as sweep_mod

    tok, model = FakeTok(), FakeModel(FakeTok())
    print("prompts.prepare")
    for name, messages, prefill in (
        ("probe", prompts.probe_messages(), True),
        ("detect", prompts.detect_messages(prompts.TASK_PROMPTS[0]), True),
        ("neutral", [{"role": "user", "content": prompts.TASK_PROMPTS[0]}], False),
        ("yes_bias", prompts.yes_bias_messages(prompts.YES_BIAS_QUESTIONS[0]), True),
    ):
        ids, positions, _ = prompts.prepare(model, tok, messages, prefill=prefill)
        check(name, ids, positions)

    print("sweep.sweep_prompt")
    for order in ("task_then_report", "report_then_task"):
        ids, positions, _ = sweep_mod.sweep_prompt(
            model, tok, order, prompts.TASK_PROMPTS[0], "Blue")
        check(order, ids, positions)

    # The band assigns a task per concept, so the prompt builders are called
    # once per (order, task) rather than once per order. A task whose text
    # collides with the protocol instruction, or whose user span falls outside
    # the encoding, would surface as one crash deep in the sweep otherwise.
    print("sweep.sweep_prompt over all tasks")
    for task in prompts.TASK_PROMPTS:
        for order in ("task_then_report", "report_then_task"):
            ids, positions, _ = sweep_mod.sweep_prompt(
                model, tok, order, task, "Blue")
            assert isinstance(positions, list) and positions
            assert all(0 <= i < ids.shape[1] for i in positions)
    print(f"  {len(prompts.TASK_PROMPTS)} tasks x 2 orders  all in bounds")

    # And the task assignment itself: every concept gets exactly one task, the
    # groups partition the concept list, and no group exceeds the batch.
    print("sweep.assign_tasks / task_groups")
    concepts = [f"c{i}" for i in range(60)]
    task_of = sweep_mod.assign_tasks(concepts, prompts.TASK_PROMPTS)
    groups = sweep_mod.task_groups(concepts, task_of, batch=8)
    flat = [c for _, members in groups for c in members]
    assert sorted(flat) == sorted(concepts), "task groups lost or duplicated a concept"
    assert all(len(m) <= 8 for _, m in groups), "a group exceeds the batch"
    assert all(len({task_of[c] for c in m}) == 1 for _, m in groups), \
        "a group mixes tasks, so its elements cannot share one forward"
    sizes = sorted({len(m) for _, m in groups})
    print(f"  {len(concepts)} concepts -> {len(groups)} groups, sizes {sizes}")

    print("\nall position builders return list[int] within bounds")


if __name__ == "__main__":
    main()
