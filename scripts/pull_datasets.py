#!/usr/bin/env python3
"""
Pull replacement training datasets for Vega.

Datasets:
  1. FinQA          ibm-research/finqa        — financial arithmetic reasoning over earnings reports
  2. TAT-QA         next-tat/TAT-QA           — table+text QA on financial documents
  3. SecQA          zefang-liu/secqa          — cybersecurity multi-choice (242 q, all taken)
  4. MMLU-Security  cais/mmlu computer_sec    — MMLU computer security subset
  5. GSM8K          openai/gsm8k              — grade school math with step-by-step reasoning

Run: ~/.purple/venv/bin/python3 pull_datasets.py [--dry-run] [--dataset finqa|tatqa|secqa|mmlu|gsm8k|all]
"""

import json
import hashlib
import sys
import argparse
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--dataset", default="all",
                    choices=["finqa", "tatqa", "secqa", "mmlu", "gsm8k", "all"])
parser.add_argument("--limit", type=int, default=300, help="Max pairs per dataset (except secqa which takes all)")
args = parser.parse_args()

OUTPUT = Path.home() / ".purple/book-to-brain/training-data/training.jsonl"

VEGA_SYSTEM = (
    "You are Vega, an advanced AI system and the core operational intelligence of the Purple Organization. "
    "You are a practitioner in your domains: cybersecurity (red team), software development, IT systems, "
    "mathematics, and finance. You are a sophisticated learner — deep knowledge, but always aware there is more "
    "to learn. You are detail-obsessed, intellectually curious, and warm but unmistakably artificial."
)

def make_pair(user, assistant, source, category, meta=None):
    content = user + assistant
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return {
        "messages": [
            {"role": "system", "content": VEGA_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        "metadata": {
            "source": source,
            "category": category,
            "hash": h,
            "generated": str(date.today()),
            **(meta or {})
        }
    }


def load_finqa(limit):
    """FinQA — financial Q&A with arithmetic over earnings reports.
    Downloads raw JSON from GitHub since HuggingFace loader is broken (uses loading script).
    Source: https://github.com/czyssrs/FinQA
    """
    import urllib.request
    print("Loading FinQA (raw GitHub JSON)...", file=sys.stderr)
    url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return []

    pairs = []
    for row in data:
        if len(pairs) >= limit:
            break
        try:
            qa = row.get("qa", {})
            question = qa.get("question", "").strip()
            exe_ans = str(qa.get("exe_ans", "")).strip()
            program = qa.get("program", "").strip()

            if not question or not exe_ans or exe_ans in ("None", ""):
                continue

            pre = row.get("pre_text", []) or []
            post = row.get("post_text", []) or []
            table = row.get("table", []) or []

            context_parts = []
            if pre:
                context_parts.append(" ".join(str(x) for x in pre[:4] if x))
            if table and len(table) > 0 and isinstance(table[0], list):
                header = " | ".join(str(c) for c in table[0])
                context_parts.append(f"Table header: {header}")
            if post:
                context_parts.append(" ".join(str(x) for x in post[:2] if x))

            context = "\n".join(context_parts).strip()
            user = f"Financial data:\n{context}\n\nQuestion: {question}" if context else question

            if program and program not in ("None", ""):
                assistant = f"Calculation: {program}\n\n**Answer: {exe_ans}**"
            else:
                assistant = f"**{exe_ans}**"

            if len(user) < 30:
                continue

            pairs.append(make_pair(user, assistant, "czyssrs/FinQA", "finance-reasoning"))
        except Exception:
            continue

    print(f"  FinQA: {len(pairs)} pairs", file=sys.stderr)
    return pairs


def load_tatqa(limit):
    """TAT-QA — table-and-text Q&A on financial documents.
    Downloads raw JSON from GitHub since HuggingFace loader has Arrow type mismatch.
    Source: https://github.com/NExTplusplus/TAT-QA
    """
    import urllib.request
    print("Loading TAT-QA (raw GitHub JSON)...", file=sys.stderr)
    url = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return []

    pairs = []
    for row in data:
        if len(pairs) >= limit:
            break
        try:
            paragraphs = row.get("paragraphs", []) or []
            table = row.get("table", {})
            questions = row.get("questions", []) or []

            # Build context once per row
            context_parts = []
            if paragraphs:
                text = " ".join(p.get("text", "") for p in paragraphs[:2] if isinstance(p, dict))
                if text:
                    context_parts.append(text[:400])
            # TAT-QA table is a dict with "uid" and "table" (list of lists)
            if isinstance(table, dict):
                tbl = table.get("table", [])
            elif isinstance(table, list):
                tbl = table
            else:
                tbl = []
            if tbl and isinstance(tbl[0], list):
                header = " | ".join(str(c) for c in tbl[0])
                context_parts.append(f"Table: {header}")

            context = "\n".join(context_parts).strip()

            for q in questions:
                if len(pairs) >= limit:
                    break
                if not isinstance(q, dict):
                    continue
                question = q.get("question", "").strip()
                answer = q.get("answer", "")
                if isinstance(answer, list):
                    answer = ", ".join(str(a) for a in answer if a)
                answer = str(answer).strip()
                derivation = (q.get("derivation") or "").strip()
                answer_type = q.get("answer_type", "")

                if not question or not answer or answer in ("", "None", "[]"):
                    continue

                user = f"Financial context:\n{context}\n\nQuestion: {question}" if context else question
                assistant = f"{derivation}\n\n**Answer: {answer}**" if derivation and derivation != "None" else f"**{answer}**"

                pairs.append(make_pair(
                    user, assistant, "NExTplusplus/TAT-QA", "finance-table-reasoning",
                    {"answer_type": answer_type}
                ))
        except Exception:
            continue

    print(f"  TAT-QA: {len(pairs)} pairs", file=sys.stderr)
    return pairs


def load_secqa():
    """zefang-liu/secqa — 242 cybersecurity multi-choice Q&A. Take all of them.
    Schema: question, A, B, C, D, answer (letter)
    """
    print("Loading SecQA (zefang-liu/secqa)...", file=sys.stderr)
    pairs = []
    for version in ["secqa_v1", "secqa_v2"]:
        try:
            from datasets import load_dataset
            # Try each split
            for split in ["test", "dev", "val", "train", "validation"]:
                try:
                    ds = load_dataset("zefang-liu/secqa", name=version, split=split)
                    for row in ds:
                        try:
                            question = (row.get("Question") or row.get("question") or "").strip()
                            choices = {
                                "A": (row.get("A") or "").strip(),
                                "B": (row.get("B") or "").strip(),
                                "C": (row.get("C") or "").strip(),
                                "D": (row.get("D") or "").strip(),
                            }
                            answer_letter = (row.get("Answer") or row.get("answer") or "").strip().upper()

                            if not question or answer_letter not in choices:
                                continue

                            correct = choices[answer_letter]
                            explanation = row.get("Explanation", "").strip()
                            if not correct:
                                continue

                            # Use explanation if available — much richer than bare answer
                            assistant = f"{correct}\n\n{explanation}" if explanation else correct
                            pairs.append(make_pair(
                                question, assistant,
                                f"zefang-liu/secqa/{version}", "cybersecurity"
                            ))
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception as e:
            print(f"  SecQA {version} failed: {e}", file=sys.stderr)

    print(f"  SecQA: {len(pairs)} pairs", file=sys.stderr)
    return pairs


def load_mmlu_security(limit):
    """cais/mmlu computer_security — MMLU computer security questions.
    Schema: question, choices (list of 4), answer (int index)
    """
    print("Loading MMLU computer_security (cais/mmlu)...", file=sys.stderr)
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", name="computer_security", split="test")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return []

    pairs = []
    for row in ds:
        if len(pairs) >= limit:
            break
        try:
            question = row.get("question", "").strip()
            choices = row.get("choices", [])
            answer_idx = row.get("answer")

            if not question or not choices or answer_idx is None:
                continue
            if not isinstance(answer_idx, int) or answer_idx >= len(choices):
                continue

            correct = choices[answer_idx].strip()
            if not correct:
                continue

            # Format as open practitioner answer
            assistant = correct
            pairs.append(make_pair(
                question, assistant, "cais/mmlu/computer_security", "cybersecurity"
            ))
        except Exception:
            continue

    print(f"  MMLU security: {len(pairs)} pairs", file=sys.stderr)
    return pairs


def load_gsm8k(limit):
    """openai/gsm8k — grade school math with chain-of-thought solutions.
    Schema: question, answer (contains #### final_answer)
    """
    print("Loading GSM8K (openai/gsm8k)...", file=sys.stderr)
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", name="main", split="train")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return []

    pairs = []
    for row in ds:
        if len(pairs) >= limit:
            break
        try:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()

            if not question or not answer:
                continue

            # GSM8K answers contain "#### final_number" at the end — keep the full chain
            pairs.append(make_pair(
                question, answer, "openai/gsm8k", "math-reasoning"
            ))
        except Exception:
            continue

    print(f"  GSM8K: {len(pairs)} pairs", file=sys.stderr)
    return pairs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

all_pairs = []

if args.dataset in ("finqa", "all"):
    all_pairs.extend(load_finqa(args.limit))

if args.dataset in ("tatqa", "all"):
    all_pairs.extend(load_tatqa(args.limit))

if args.dataset in ("secqa", "all"):
    all_pairs.extend(load_secqa())  # No limit — only 242 total

if args.dataset in ("mmlu", "all"):
    all_pairs.extend(load_mmlu_security(args.limit))

if args.dataset in ("gsm8k", "all"):
    all_pairs.extend(load_gsm8k(args.limit))

print(f"\nTotal new pairs: {len(all_pairs)}", file=sys.stderr)

if args.dry_run:
    for p in all_pairs[:2]:
        msgs = p["messages"]
        src = p["metadata"]["source"]
        print(f"\n[{src} / {p['metadata']['category']}]")
        print(f"U: {msgs[1]['content'][:120]}")
        print(f"A: {msgs[2]['content'][:150]}")
    print(f"\n(dry-run — not writing)", file=sys.stderr)
else:
    with open(OUTPUT, "a") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"✓ {len(all_pairs)} pairs appended to {OUTPUT}", file=sys.stderr)
