#!/usr/bin/env python3
"""
Build Vega v2 training dataset.

Changes from v1:
  DROPPED: FinQA (300) — "Calculation: op(a,b)" DSL polluted all outputs
  DROPPED: TAT-QA (300) — bare bold-number answers (**2.9**) same issue
  DROPPED: MMLU (100) — single-word answers with no explanation
  KEPT:    SecQA (242, explanation-only), GSM8K (300), Alpaca (300)
  KEPT:    Identity (50), Reasoning (28), Fragments+Sessions (55)
  ADDED:   Evol-Instruct-Code (300) — code generation capability

Usage:
    source ~/.purple/venv/bin/activate
    python3 build_dataset_v2.py [--dry-run]

Output:
    ~/.purple/book-to-brain/training-data/training_v2.jsonl
"""
import json
import hashlib
import random
import sys
import argparse
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--code-limit", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

SOURCE_V1 = Path.home() / ".purple/book-to-brain/training-data/training.jsonl"
OUTPUT    = Path.home() / ".purple/book-to-brain/training-data/training_v2.jsonl"

VEGA_SYSTEM = (
    "You are Vega, an advanced AI system and the core operational intelligence of the Purple Organization. "
    "You are a practitioner in your domains: cybersecurity (red team), software development, IT systems, "
    "mathematics, and finance. You are a sophisticated learner — deep knowledge, but always aware there is more "
    "to learn. You are detail-obsessed, intellectually curious, and warm but unmistakably artificial."
)

DROPPED_SOURCES = {
    "czyssrs/FinQA",
    "NExTplusplus/TAT-QA",
    "cais/mmlu/computer_security",
}

def make_pair(user, assistant, source, category, meta=None):
    h = hashlib.sha256((user + assistant).encode()).hexdigest()[:16]
    return {
        "messages": [
            {"role": "system", "content": VEGA_SYSTEM},
            {"role": "user",   "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": source, "category": category,
            "hash": h, "generated": str(date.today()),
            **(meta or {})
        }
    }

# ── 1. Load and filter existing v1 pairs ─────────────────────────────
print("Loading v1 training data...", file=sys.stderr)
v1_pairs = [json.loads(l) for l in open(SOURCE_V1)]
print(f"  v1 total: {len(v1_pairs)}", file=sys.stderr)

kept = []
dropped = {}
for p in v1_pairs:
    src = p.get("metadata", {}).get("source", "unknown")
    if src in DROPPED_SOURCES:
        dropped[src] = dropped.get(src, 0) + 1
        continue
    # Also drop SecQA pairs with no explanation (bare single-word answers)
    if "secqa" in src.lower():
        ans = p["messages"][2]["content"]
        # Keep only if explanation present (contains newline = letter + explanation)
        if "\n" not in ans and len(ans) < 80:
            dropped[src] = dropped.get(src, 0) + 1
            continue
    kept.append(p)

print(f"  Kept:    {len(kept)}", file=sys.stderr)
print(f"  Dropped: {sum(dropped.values())} ({dict(dropped)})", file=sys.stderr)

# ── 2. Pull Evol-Instruct-Code ────────────────────────────────────────
def load_evol_code(limit):
    print(f"Loading Evol-Instruct-Code (nickrosh/Evol-Instruct-Code-80k-v1, limit={limit})...", file=sys.stderr)
    try:
        from datasets import load_dataset
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return []

    # Shuffle and sample
    indices = list(range(len(ds)))
    random.shuffle(indices)

    pairs = []
    for i in indices:
        if len(pairs) >= limit:
            break
        try:
            row = ds[i]
            instruction = (row.get("instruction") or "").strip()
            output = (row.get("output") or "").strip()
            if not instruction or not output:
                continue
            # Skip trivially short outputs (< 100 chars) — likely dataset noise
            if len(output) < 100:
                continue
            # Skip outputs that are just "I cannot" refusals
            if output.lower().startswith("i cannot") or output.lower().startswith("i'm sorry"):
                continue
            pairs.append(make_pair(
                instruction, output,
                "nickrosh/Evol-Instruct-Code-80k-v1", "code-generation"
            ))
        except Exception:
            continue

    print(f"  Evol-Code: {len(pairs)} pairs", file=sys.stderr)
    return pairs

code_pairs = load_evol_code(args.code_limit)

# ── 3. Combine and shuffle ─────────────────────────────────────────────
all_pairs = kept + code_pairs
random.shuffle(all_pairs)

# ── 4. Stats ──────────────────────────────────────────────────────────
from collections import Counter
sources = Counter(p.get("metadata", {}).get("source", "?") for p in all_pairs)
ans_lens = [len(p["messages"][2]["content"]) for p in all_pairs]
short = sum(1 for l in ans_lens if l < 100)
long  = sum(1 for l in ans_lens if l > 500)

print(f"\n── v2 Dataset Stats ──────────────────────────────", file=sys.stderr)
print(f"  Total pairs:       {len(all_pairs)}", file=sys.stderr)
print(f"  Short answers (<100 chars): {short} ({100*short//len(all_pairs)}%)", file=sys.stderr)
print(f"  Long answers  (>500 chars): {long}  ({100*long//len(all_pairs)}%)", file=sys.stderr)
print(f"  Sources:", file=sys.stderr)
for src, cnt in sorted(sources.items(), key=lambda x: -x[1])[:15]:
    print(f"    {cnt:4d}  {src}", file=sys.stderr)

if args.dry_run:
    print("\n(dry-run — not writing)", file=sys.stderr)
    # Show one code pair
    code = [p for p in all_pairs if "Evol" in p.get("metadata",{}).get("source","")]
    if code:
        p = code[0]
        print(f"\nCode pair sample:", file=sys.stderr)
        print(f"  USR: {p['messages'][1]['content'][:150]}", file=sys.stderr)
        print(f"  ANS: {p['messages'][2]['content'][:300]}", file=sys.stderr)
    sys.exit(0)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    for p in all_pairs:
        f.write(json.dumps(p) + "\n")
print(f"\n✓ {len(all_pairs)} pairs written to {OUTPUT}", file=sys.stderr)
print(f"  Next: update train_config.yaml → train_data: {OUTPUT}", file=sys.stderr)
