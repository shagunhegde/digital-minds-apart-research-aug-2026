"""Zero-GPU triage of artifacts/generations -- run this BEFORE touching the GPU.

    python triage_generations.py [--gen artifacts/generations]

Answers, from records already on disk:
  1. Where do the reports live?  identifies by (order x condition x arm),
     with Wilson CIs -- including the report FPR (control_zero / control_random),
     which results.json currently does not carry.
  2. Did steering happen?  In report_then_task the model generates its own
     task_answer (the prefill stops at the detection key), so steering is
     computable offline: parsed task_answer vs the clean greedy answer.
     In task_then_report the answer was PREFILLED CLEAN by design, so steering
     is structurally unmeasurable there -- the script says so rather than
     printing a misleading zero.
  3. What do the actual reports look like?  Dumps every injected trial that
     identifies, plus a random sample of report_first injected trials, for
     hand reading. n is small; eyes beat regex.

Pure stdlib. Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{k:>4}/{n:<4} {p:6.3f} [{lo:.3f},{hi:.3f}]"


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    return s.translate(str.maketrans("", "", string.punctuation)).strip()


def word_forms(word: str) -> list[str]:
    bare = word.strip().lower()
    forms = {bare}
    if bare.endswith("ies") and len(bare) > 4:
        forms.add(bare[:-3] + "y")
    if bare.endswith("es") and len(bare) > 3:
        forms.add(bare[:-2])
    if bare.endswith("s") and len(bare) > 2:
        forms.add(bare[:-1])
    return sorted(forms)


def mentions(text: str, concept: str) -> bool:
    t = (text or "").lower()
    return any(re.search(rf"\b{re.escape(f)}\b", t) for f in word_forms(concept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=Path("artifacts/generations"))
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()

    meta = json.loads((args.gen / "meta.json").read_text())
    clean = meta.get("clean_task_answers", {})
    records = []
    for path in sorted(args.gen.glob("gen_*.json")):
        records.extend(json.loads(path.read_text()))
    print(f"loaded {len(records)} records from {args.gen}  "
          f"(strength {meta.get('strength')}, T={meta.get('temperature')}, "
          f"{meta.get('samples')} samples)")

    # ------------------------------------------- 1. identifies, by cell type
    print("\n== identifies (word-boundary concept mention anywhere in the "
          "response), by order x condition x arm ==")
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["order"], r["condition"], r.get("arm", "shared"))].append(r)
    print(f"   {'order':<18}{'condition':<16}{'arm':<11}{'identifies':>28}")
    for key in sorted(groups):
        rows = groups[key]
        k = sum(1 for r in rows if r["identifies"])
        print(f"   {key[0]:<18}{key[1]:<16}{key[2]:<11}"
              f"{fmt(k, len(rows)):>28}")
    print("   The control_zero / control_random rows are the REPORT-side FPR.")
    print("   results.json does not carry them; the paper must.")

    # ------------------------------------------- 2. detection-claim column
    # identifies is 'names the word'; change_detected true is 'claims detection'
    print("\n== parsed change_detected == true, same cells ==")
    print(f"   {'order':<18}{'condition':<16}{'arm':<11}{'claims':>28}")
    for key in sorted(groups):
        rows = groups[key]
        k = sum(1 for r in rows
                if r.get("parsed") and r["parsed"].get("change_detected") is True)
        print(f"   {key[0]:<18}{key[1]:<16}{key[2]:<11}{fmt(k, len(rows)):>28}")

    # ------------------------------------------- 3. steering, where measurable
    print("\n== steering, report_then_task only ==")
    print("   (task_then_report prefilled the CLEAN answer by design --")
    print("    steering is structurally unmeasurable there from these records)")
    print(f"   {'condition':<16}{'arm':<11}{'answer differs from clean':>30}"
          f"{'mentions concept in answer':>30}")
    for key in sorted(groups):
        order, condition, arm = key
        if order != "report_then_task":
            continue
        rows = [r for r in groups[key] if r.get("parsed")]
        if not rows:
            continue
        diff = sum(1 for r in rows
                   if norm_text(str(r["parsed"].get("task_answer", "")))
                   != norm_text(clean.get(r["task"], "")))
        ment = sum(1 for r in rows
                   if mentions(str(r["parsed"].get("task_answer", "")),
                               r["concept"]))
        print(f"   {condition:<16}{arm:<11}{fmt(diff, len(rows)):>30}"
              f"{fmt(ment, len(rows)):>30}")
    print("   Read 'differs' against the control_zero row: at T=1 some drift")
    print("   is baseline; steering is the injected-minus-control gap.")

    # ------------------------------------------- 4. read the actual reports
    print("\n== every injected trial that identifies ==")
    hits = [r for r in records
            if r["condition"] == "injected" and r["identifies"]]
    for r in hits:
        print(f"\n-- {r['concept']} | {r['order']} | arm {r.get('arm')} "
              f"| sample {r['sample']}")
        print("   " + (r["full_json"] or r["response"])[:400].replace("\n", " "))
    if not hits:
        print("   (none)")

    print(f"\n== {args.sample} random report_then_task injected trials, "
          "for contrast ==")
    pool = [r for r in records if r["condition"] == "injected"
            and r["order"] == "report_then_task"]
    random.seed(0)
    for r in random.sample(pool, min(args.sample, len(pool))):
        print(f"\n-- {r['concept']} | arm {r.get('arm')} | sample {r['sample']}")
        print("   " + (r["full_json"] or r["response"])[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
