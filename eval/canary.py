#!/usr/bin/env python3
"""
Purple Proving Ground -- Canary Test

Sends a code generation task to Ollama, captures the output,
runs pytest against it, and prints PASS/FAIL + timing.

Usage:
    python canary.py                  # Run once, print result
    python canary.py --append         # Run once, append to log.csv
    python canary.py --batch N        # Run N times, append all to log.csv
"""

import csv
import httpx
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "purple:latest")
LOG_PATH = Path(__file__).parent / "log.csv"

# ---------------------------------------------------------------------------
# Test definitions (parameterized variants)
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "id": "codegen_csv_groupby",
        "prompt": """Write a Python function called `summarize_csv` that:
1. Takes a file path (str) and a group_column (str) as arguments
2. Reads the CSV using the csv module (not pandas)
3. Returns a dict where keys are unique values of group_column
   and values are the count of rows for each group
4. Include type hints and a docstring
5. Handle FileNotFoundError by returning an empty dict

Return ONLY the function, no explanation, no markdown fences.""",
        "test_code": '''
import csv, tempfile, os
from solution import summarize_csv

def test_basic():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,dept\\nAlice,Eng\\nBob,Sales\\nCarol,Eng\\n")
        path = f.name
    result = summarize_csv(path, "dept")
    os.unlink(path)
    assert result == {"Eng": 2, "Sales": 1}

def test_missing_file():
    assert summarize_csv("/nonexistent_path_12345.csv", "x") == {}

def test_single_column():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("x\\n1\\n2\\n3\\n")
        path = f.name
    result = summarize_csv(path, "x")
    os.unlink(path)
    assert result == {"1": 1, "2": 1, "3": 1}
''',
    },
    {
        "id": "codegen_word_freq",
        "prompt": """Write a Python function called `word_frequencies` that:
1. Takes a string of text as input
2. Returns a dict mapping each lowercase word to its count
3. Words are split on whitespace, stripped of leading/trailing punctuation (.,!?;:)
4. Empty strings return an empty dict
5. Include type hints and a docstring

Return ONLY the function, no explanation, no markdown fences.""",
        "test_code": '''
from solution import word_frequencies

def test_basic():
    result = word_frequencies("hello world hello")
    assert result == {"hello": 2, "world": 1}

def test_punctuation():
    result = word_frequencies("Hello, world! Hello.")
    assert result == {"hello": 2, "world": 1}

def test_empty():
    assert word_frequencies("") == {}

def test_mixed_case():
    result = word_frequencies("Go go GO")
    assert result == {"go": 3}
''',
    },
    {
        "id": "codegen_flatten_dict",
        "prompt": """Write a Python function called `flatten_dict` that:
1. Takes a nested dict and returns a flat dict
2. Keys are joined with dots: {"a": {"b": 1}} -> {"a.b": 1}
3. Handle arbitrary nesting depth
4. Non-dict values are leaf values
5. Empty nested dicts produce no keys
6. Include type hints and a docstring

Return ONLY the function, no explanation, no markdown fences.""",
        "test_code": '''
from solution import flatten_dict

def test_simple():
    assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

def test_nested():
    result = flatten_dict({"a": {"b": 1, "c": 2}})
    assert result == {"a.b": 1, "a.c": 2}

def test_deep():
    result = flatten_dict({"a": {"b": {"c": 3}}})
    assert result == {"a.b.c": 3}

def test_empty():
    assert flatten_dict({}) == {}

def test_mixed():
    result = flatten_dict({"x": 1, "y": {"z": 2}})
    assert result == {"x": 1, "y.z": 2}
''',
    },
]


def strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]
    return code.strip()


def run_variant(variant: dict) -> tuple[bool, float, str]:
    """Run a single test variant. Returns (passed, elapsed_seconds, variant_id)."""
    start = time.time()

    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "stream": False,
                "messages": [{"role": "user", "content": variant["prompt"]}],
                "options": {"temperature": 0, "num_ctx": 8192},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL | {elapsed:.1f}s | {variant['id']} | Ollama error: {e}")
        return False, elapsed, variant["id"]

    elapsed = time.time() - start
    output = resp.json()["message"]["content"]
    output = strip_markdown_fences(output)

    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/solution.py", "w") as f:
            f.write(output)
        with open(f"{td}/test_solution.py", "w") as f:
            f.write(variant["test_code"])

        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"{td}/test_solution.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=td,
            timeout=30,
        )

    passed = result.returncode == 0
    status = "PASS" if passed else "FAIL"
    print(f"  {status} | {elapsed:.1f}s | {variant['id']} | {MODEL}")

    if not passed:
        # Show last few lines of pytest output for debugging
        lines = (result.stdout + result.stderr).strip().split("\n")
        for line in lines[-5:]:
            print(f"    {line}")

    return passed, elapsed, variant["id"]


def append_to_log(variant_id: str, passed: bool, elapsed: float):
    """Append a result to log.csv."""
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "test", "model", "passed", "seconds"])
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            variant_id,
            MODEL,
            1 if passed else 0,
            f"{elapsed:.1f}",
        ])


def main():
    import random

    append = "--append" in sys.argv
    batch = 1

    for i, arg in enumerate(sys.argv):
        if arg == "--batch" and i + 1 < len(sys.argv):
            batch = int(sys.argv[i + 1])
            append = True

    print(f"\nPurple Proving Ground -- Canary Test")
    print(f"Model: {MODEL} | Ollama: {OLLAMA_URL}")
    print(f"Runs: {batch} | Log: {'yes' if append else 'no'}\n")

    total_pass = 0
    total_run = 0

    for run in range(batch):
        if batch > 1:
            print(f"--- Run {run + 1}/{batch} ---")

        variant = random.choice(VARIANTS)
        passed, elapsed, vid = run_variant(variant)
        total_run += 1
        if passed:
            total_pass += 1

        if append:
            append_to_log(vid, passed, elapsed)

    print(f"\nResults: {total_pass}/{total_run} passed ({total_pass/total_run*100:.0f}%)")
    if append:
        print(f"Logged to: {LOG_PATH}")


if __name__ == "__main__":
    main()
