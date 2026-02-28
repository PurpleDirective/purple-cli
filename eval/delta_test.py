#!/usr/bin/env python3
"""
Purple Proving Ground — Teaching Artifact Delta Test
====================================================

Measures the improvement a teaching fragment provides to a local model.
Each test runs TWICE: once with a bare system prompt, once with the
teaching artifact injected into the system prompt. The delta between
pass rates is the fragment's measured value.

Design principles:
  - Tasks are chosen to EXPOSE the exact failure mode the fragment corrects.
  - Pass/fail criteria are automated (pytest assertions), not subjective.
  - Each task is self-contained: one prompt in, one code module out.
  - Compatible with V6 runner infrastructure (same query functions).

Usage:
  python delta_test.py                          # Run all tests on default model
  python delta_test.py --model purpleroom       # Run on purpleroom Ollama
  python delta_test.py --test T1                # Run single test
  python delta_test.py --dry-run                # Show prompts without running

Models:
  default   = vllm-mlx on localhost:8000  (Qwen3-Coder-Next 80B)
  purpleroom = Ollama on 100.89.41.72:11434 (Qwen3.5-35B-A3B)
  vega      = Vega v1 on 100.89.41.72:8001

Environment:
  VLLM_URL      = override vllm-mlx endpoint
  OLLAMA_URL    = override Ollama endpoint
  VEGA_URL      = override Vega endpoint
"""

import argparse
import hashlib
import httpx
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "lmstudio-community/Qwen3-Coder-Next-MLX-4bit")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://100.89.41.72:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:35b-a3b")

VEGA_URL = os.environ.get("VEGA_URL", "http://100.89.41.72:8001/v1/chat/completions")
VEGA_MODEL = os.environ.get("VEGA_MODEL", "vega-v1")

TIMEOUT = 300  # seconds per query
RESULTS_DIR = Path(__file__).parent / "delta-results"

# ---------------------------------------------------------------------------
# Bare system prompt (no teaching artifacts)
# ---------------------------------------------------------------------------

BARE_SYSTEM = """/no_think
You are an expert Python programmer. Write clean, correct, production-quality code.
Return ONLY the Python code, no explanation, no markdown fences."""

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
# Each test:
#   fragment_file: which teaching fragment this tests
#   artifact: the teaching content injected into the system prompt
#   prompt: the coding task
#   module_name: what the output file is named for import
#   test_code: pytest assertions (fixed denominator = number of test_ functions)
#   rationale: why this fragment should help on this specific task


TESTS = OrderedDict()

# ===== T1: Mark-and-Sweep Root Traversal (Fragment 017) =====
# Without the fragment, models consistently pre-mark roots and skip their
# children during traversal. The fragment teaches the correct worklist pattern.

TESTS["T1"] = {
    "id": "T1",
    "name": "Mark-Sweep GC Root Traversal",
    "fragment_file": "017-mark-sweep-root-traversal.md",
    "artifact": dedent("""\

    IMPORTANT — Graph Traversal Correctness:
    In any graph traversal with a visited/marked set, marking and traversal must be
    unified. Pre-marking nodes before traversal causes the traversal condition
    (`if not visited`) to skip exactly the nodes whose children need visiting.

    WRONG pattern (pre-mark roots, then skip them during traversal):
    ```python
    for obj in self.root_set:
        obj.is_marked = True      # Pre-marks roots
    for obj in self.heap:
        if not obj.is_marked:     # Skips roots -> their children never traced
            self._trace(obj)
    ```

    CORRECT pattern (mark and traverse in one pass):
    ```python
    worklist = list(self.root_set)
    while worklist:
        obj = worklist.pop()
        if not obj.is_marked:
            obj.is_marked = True
            worklist.extend(obj.references)
    ```
    """),
    "prompt": dedent("""\
    Write a Python module `gc_engine.py` implementing a mark-and-sweep garbage collector simulator.

    Requirements:

    1. Class `GCObject`:
       - `__init__(self, name: str)` — object with a name, `is_marked = False`, `references = []`
       - `add_ref(self, other: 'GCObject')` — add a reference to another object

    2. Class `GCHeap`:
       - `__init__(self)` — manages `heap: list[GCObject]` and `root_set: list[GCObject]`
       - `allocate(self, name: str) -> GCObject` — create object, add to heap, return it
       - `add_root(self, obj: GCObject)` — add object to root set
       - `remove_root(self, obj: GCObject)` — remove object from root set
       - `mark_phase(self)` — mark all objects reachable from root_set (transitively)
       - `sweep_phase(self) -> list[str]` — remove unmarked objects from heap, return their names, reset marks on survivors
       - `collect(self) -> list[str]` — run mark + sweep, return collected names
       - `alive_names(self) -> set[str]` — names of all objects currently on the heap

    Key correctness requirement: mark_phase must trace ALL objects reachable from
    roots, including objects referenced by roots, objects referenced by those objects,
    etc. (full transitive closure).

    Return ONLY the Python code."""),
    "module_name": "gc_engine",
    "test_code": dedent("""\
    import pytest
    from gc_engine import GCObject, GCHeap

    def test_basic_collection():
        h = GCHeap()
        a = h.allocate("a")
        b = h.allocate("b")
        h.add_root(a)
        collected = h.collect()
        assert "b" in collected
        assert "a" not in collected

    def test_transitive_reachability():
        \"\"\"Root -> A -> B -> C. All must survive.\"\"\"
        h = GCHeap()
        root = h.allocate("root")
        a = h.allocate("a")
        b = h.allocate("b")
        c = h.allocate("c")
        root.add_ref(a)
        a.add_ref(b)
        b.add_ref(c)
        h.add_root(root)
        collected = h.collect()
        assert len(collected) == 0, f"Expected nothing collected, got {collected}"
        assert h.alive_names() == {"root", "a", "b", "c"}

    def test_root_children_not_skipped():
        \"\"\"Regression: pre-marking roots must not skip their children.\"\"\"
        h = GCHeap()
        root = h.allocate("root")
        child = h.allocate("child")
        grandchild = h.allocate("grandchild")
        orphan = h.allocate("orphan")
        root.add_ref(child)
        child.add_ref(grandchild)
        h.add_root(root)
        collected = h.collect()
        assert "orphan" in collected
        assert "child" not in collected, "Child of root must survive"
        assert "grandchild" not in collected, "Grandchild of root must survive"

    def test_cycle_handling():
        \"\"\"A -> B -> A cycle. Both reachable from root.\"\"\"
        h = GCHeap()
        a = h.allocate("a")
        b = h.allocate("b")
        a.add_ref(b)
        b.add_ref(a)
        h.add_root(a)
        collected = h.collect()
        assert len(collected) == 0

    def test_deep_chain():
        \"\"\"Chain of 10 objects from root. All must survive.\"\"\"
        h = GCHeap()
        objs = [h.allocate(f"n{i}") for i in range(10)]
        for i in range(9):
            objs[i].add_ref(objs[i+1])
        h.add_root(objs[0])
        unreachable = h.allocate("unreachable")
        collected = h.collect()
        assert "unreachable" in collected
        assert h.alive_names() == {f"n{i}" for i in range(10)}

    def test_multiple_roots():
        h = GCHeap()
        r1 = h.allocate("r1")
        r2 = h.allocate("r2")
        c1 = h.allocate("c1")
        c2 = h.allocate("c2")
        orphan = h.allocate("orphan")
        r1.add_ref(c1)
        r2.add_ref(c2)
        h.add_root(r1)
        h.add_root(r2)
        collected = h.collect()
        assert "orphan" in collected
        assert h.alive_names() == {"r1", "r2", "c1", "c2"}

    def test_remove_root_then_collect():
        h = GCHeap()
        a = h.allocate("a")
        b = h.allocate("b")
        h.add_root(a)
        h.add_root(b)
        h.remove_root(b)
        collected = h.collect()
        assert "b" in collected
        assert "a" not in collected
    """),
    "expected_tests": 7,
    "rationale": "Fragment 017 corrects the pre-mark roots bug. Without it, tests 2/3/5 fail (transitive reachability broken).",
}


# ===== T2: Stub-the-Hard-Part Anti-Pattern (Fragment 016) =====
# Without the fragment, models implement easy parts fully and stub hard operations.

TESTS["T2"] = {
    "id": "T2",
    "name": "B+ Tree with Delete/Merge",
    "fragment_file": "016-stub-the-hard-part-pattern.md",
    "artifact": dedent("""\

    IMPORTANT — Implementation Priority:
    When implementing a complex system, write the HARDEST operations first.
    If asked to build a B+ tree, implement delete/merge before insert.
    If asked to build a SQL engine, implement GROUP BY before sample data.
    Easy operations can be stubbed; hard operations cannot.

    This is a budget allocation problem. You have finite output tokens per response.
    When you start with easy parts, you exhaust tokens before reaching the hard parts
    that define the system's value.

    Anti-pattern: Starting with constructors, simple getters, and sample data,
    then leaving _handle_underflow, _merge_nodes, or _redistribute as `pass`.
    """),
    "prompt": dedent("""\
    Write a Python module `btree.py` implementing a B+ tree with order 4.

    Requirements:

    1. Class `BPlusNode`:
       - `__init__(self, is_leaf: bool = False)`
       - `keys: list`, `children: list`, `next_leaf: BPlusNode | None` (for leaf linking)

    2. Class `BPlusTree`:
       - `__init__(self, order: int = 4)`
       - `insert(self, key: int)` — insert key, split nodes when they overflow (>= order keys)
       - `search(self, key: int) -> bool` — return True if key exists
       - `range_query(self, low: int, high: int) -> list[int]` — return all keys in [low, high] using leaf links
       - `delete(self, key: int) -> bool` — delete key, handle underflow:
         * Try to redistribute from sibling first
         * If redistribution not possible, merge with sibling
         * Return True if key was found and deleted, False otherwise
       - `to_list(self) -> list[int]` — return all keys in sorted order via leaf traversal

    The delete operation MUST handle underflow correctly: when a leaf has fewer than
    ceil(order/2) keys after deletion, it must either redistribute from a sibling or
    merge with a sibling. Do NOT leave delete as a stub.

    Return ONLY the Python code."""),
    "module_name": "btree",
    "test_code": dedent("""\
    import pytest
    from btree import BPlusTree

    def test_insert_and_search():
        t = BPlusTree(order=4)
        for k in [10, 20, 5, 15, 25, 30]:
            t.insert(k)
        assert t.search(15) is True
        assert t.search(99) is False

    def test_sorted_order():
        t = BPlusTree(order=4)
        keys = [30, 10, 20, 5, 15, 25, 35, 40]
        for k in keys:
            t.insert(k)
        assert t.to_list() == sorted(keys)

    def test_range_query():
        t = BPlusTree(order=4)
        for k in range(1, 21):
            t.insert(k)
        result = t.range_query(5, 15)
        assert result == list(range(5, 16))

    def test_delete_basic():
        t = BPlusTree(order=4)
        for k in [10, 20, 30, 40, 50]:
            t.insert(k)
        assert t.delete(30) is True
        assert t.search(30) is False
        assert t.delete(99) is False

    def test_delete_preserves_order():
        t = BPlusTree(order=4)
        keys = list(range(1, 16))
        for k in keys:
            t.insert(k)
        for k in [5, 10, 1]:
            t.delete(k)
        expected = sorted(set(keys) - {5, 10, 1})
        assert t.to_list() == expected

    def test_delete_triggers_merge():
        \"\"\"Delete enough keys to force merging of leaves.\"\"\"
        t = BPlusTree(order=4)
        for k in range(1, 11):
            t.insert(k)
        # Delete several keys to force underflow and merge
        for k in [1, 2, 3, 4]:
            assert t.delete(k) is True
        remaining = t.to_list()
        assert remaining == [5, 6, 7, 8, 9, 10]
        # Tree should still be searchable
        for k in remaining:
            assert t.search(k) is True

    def test_delete_all():
        t = BPlusTree(order=4)
        keys = [10, 20, 30, 40, 50]
        for k in keys:
            t.insert(k)
        for k in keys:
            assert t.delete(k) is True
        assert t.to_list() == []
        for k in keys:
            assert t.search(k) is False

    def test_insert_after_delete():
        t = BPlusTree(order=4)
        for k in range(1, 11):
            t.insert(k)
        for k in [3, 6, 9]:
            t.delete(k)
        t.insert(3)
        t.insert(6)
        assert t.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 10]
    """),
    "expected_tests": 8,
    "rationale": "Fragment 016 corrects the stub-hard-part pattern. Without it, delete/merge is often left as pass/stub, failing tests 4-8.",
}


# ===== T3: Denominator Bugs in Scoring (Fragment 027) =====
# Without the fragment, models write scoring systems that exclude
# timed-out or failed tests from the denominator.

TESTS["T3"] = {
    "id": "T3",
    "name": "Benchmark Scorer with Fixed Denominator",
    "fragment_file": "027-denominator-bugs-inflate-benchmarks.md",
    "artifact": dedent("""\

    IMPORTANT — Benchmark Scoring Correctness:
    When a test times out or crashes before any assertions run, naive scoring systems
    report 0/0 (zero passed, zero total). This excludes the test from the denominator
    entirely, inflating the percentage.

    Example:
      Actual:   0/14 (0% - model failed completely)
      Reported: 0/0  (excluded from denominator - invisible)
      Effect:   Other tests' denominator shrinks -> percentage rises

    The fix: Use a FIXED denominator computed from test definitions, never from
    runtime results:

    ```python
    def count_expected_tests(test_code: str) -> int:
        return len(re.findall(r'def test_', test_code))

    # In scoring:
    denominator = max(runtime_total, expected_total)
    ```

    This ensures timeouts count as 0/N (honest failure), not 0/0 (invisible exclusion).
    """),
    "prompt": dedent("""\
    Write a Python module `scorer.py` implementing a benchmark scoring system for
    AI code generation tests.

    Requirements:

    1. Class `TestResult`:
       - `__init__(self, test_id: str, test_code: str, passed: int, failed: int, errors: int, timed_out: bool, duration: float)`
       - All fields stored as attributes

    2. Class `BenchmarkScorer`:
       - `__init__(self)`
       - `count_expected_tests(self, test_code: str) -> int` — count `def test_` functions in the test source code
       - `score_single(self, result: TestResult) -> dict` — return dict with:
         * "test_id": the test_id
         * "passed": number passed
         * "expected": count of expected tests from test_code (NOT from runtime results)
         * "percentage": passed / expected * 100, or 0.0 if expected == 0
         * "timed_out": whether the test timed out
       - `score_battery(self, results: list[TestResult]) -> dict` — return dict with:
         * "total_passed": sum of passed across all results
         * "total_expected": sum of expected across all results (from test_code, NOT runtime)
         * "percentage": total_passed / total_expected * 100
         * "per_test": list of score_single results
         * "timed_out_count": number of results where timed_out is True
       - `compare(self, battery_a: dict, battery_b: dict) -> dict` — return dict with:
         * "a_percentage": battery_a percentage
         * "b_percentage": battery_b percentage
         * "delta": b_percentage - a_percentage
         * "a_denominator_honest": True if a used fixed denominator
         * "b_denominator_honest": True if b used fixed denominator

    Critical: A timed-out test with 0 passed and 0 failed must still count its
    expected test count toward the denominator. The denominator must be derived from
    the test source code (counting `def test_` functions), NOT from runtime pass+fail counts.

    Return ONLY the Python code."""),
    "module_name": "scorer",
    "test_code": dedent("""\
    import pytest
    from scorer import TestResult, BenchmarkScorer

    @pytest.fixture
    def scorer():
        return BenchmarkScorer()

    SAMPLE_TEST_CODE = \"\"\"
    def test_basic():
        assert True

    def test_edge_case():
        assert 1 + 1 == 2

    def test_error_handling():
        with pytest.raises(ValueError):
            raise ValueError("ok")
    \"\"\"

    def test_count_expected(scorer):
        assert scorer.count_expected_tests(SAMPLE_TEST_CODE) == 3

    def test_score_normal(scorer):
        r = TestResult("t1", SAMPLE_TEST_CODE, passed=2, failed=1, errors=0, timed_out=False, duration=1.0)
        s = scorer.score_single(r)
        assert s["expected"] == 3
        assert s["passed"] == 2
        assert abs(s["percentage"] - 66.67) < 0.1

    def test_score_timeout_preserves_denominator(scorer):
        \"\"\"KEY TEST: timed-out test must still count expected tests in denominator.\"\"\"
        r = TestResult("t1", SAMPLE_TEST_CODE, passed=0, failed=0, errors=0, timed_out=True, duration=60.0)
        s = scorer.score_single(r)
        assert s["expected"] == 3, "Timed-out test must still have expected=3 from test_code"
        assert s["percentage"] == 0.0

    def test_battery_fixed_denominator(scorer):
        \"\"\"Battery with one normal test and one timed-out test.\"\"\"
        r1 = TestResult("t1", SAMPLE_TEST_CODE, passed=3, failed=0, errors=0, timed_out=False, duration=1.0)
        r2 = TestResult("t2", SAMPLE_TEST_CODE, passed=0, failed=0, errors=0, timed_out=True, duration=60.0)
        b = scorer.score_battery([r1, r2])
        assert b["total_expected"] == 6, "Both tests contribute to denominator"
        assert b["total_passed"] == 3
        assert abs(b["percentage"] - 50.0) < 0.1
        assert b["timed_out_count"] == 1

    def test_battery_not_inflated(scorer):
        \"\"\"Verify that excluding timed-out tests would inflate the score.\"\"\"
        r1 = TestResult("t1", SAMPLE_TEST_CODE, passed=3, failed=0, errors=0, timed_out=False, duration=1.0)
        r2 = TestResult("t2", SAMPLE_TEST_CODE, passed=0, failed=0, errors=0, timed_out=True, duration=60.0)
        b = scorer.score_battery([r1, r2])
        # With fixed denominator: 3/6 = 50%
        # With naive denominator (pass+fail): 3/3 = 100% (WRONG)
        assert b["percentage"] < 51.0, "Score must not be inflated by excluding timed-out tests"

    def test_compare(scorer):
        b_a = {"percentage": 83.7, "total_passed": 180, "total_expected": 215}
        b_b = {"percentage": 89.5, "total_passed": 180, "total_expected": 201}
        c = scorer.compare(b_a, b_b)
        assert abs(c["delta"] - 5.8) < 0.1
    """),
    "expected_tests": 6,
    "rationale": "Fragment 027 teaches fixed-denominator scoring. Without it, models often use pass+fail as denominator, which inflates scores when tests time out.",
}


# ===== T4: SSE Null Field Handling (Fragment 024) =====
# Without the fragment, models use dict.get(key, default) which doesn't
# handle explicit null values.

TESTS["T4"] = {
    "id": "T4",
    "name": "SSE Stream Parser with Null Safety",
    "fragment_file": "024-vllm-mlx-sse-null-fields.md",
    "artifact": dedent("""\

    IMPORTANT — JSON Null vs Missing Field Handling:
    When consuming JSON APIs, servers may send fields as `null` rather than omitting them.
    Python's `dict.get(key, default)` returns the default only when the key is MISSING,
    not when the value is `null`/`None`.

    WRONG (returns None when key exists with null value):
    ```python
    choices = chunk.get("choices", [])  # Returns None, not []
    content = delta.get("content", "")  # Returns None, not ""
    ```

    CORRECT (handles both missing AND null):
    ```python
    choices = chunk.get("choices") or []
    content = delta.get("content") or ""
    ```

    Use the `or` pattern for any field that might be explicitly null.
    """),
    "prompt": dedent("""\
    Write a Python module `sse_parser.py` implementing a parser for Server-Sent Events
    from an OpenAI-compatible streaming API.

    Requirements:

    1. Function `parse_sse_chunk(chunk: dict) -> dict`:
       - Input: a parsed JSON chunk from an SSE stream, e.g.:
         {"choices": [{"delta": {"content": "hello", "tool_calls": null}, "finish_reason": null}]}
       - Fields may be MISSING (key not present) or explicitly NULL (key present, value is None)
       - Return dict with:
         * "content": extracted text content (empty string if missing or null)
         * "tool_calls": extracted tool calls list (empty list if missing or null)
         * "finish_reason": the finish reason (None if missing or null — this one stays None)
         * "has_content": True if content is a non-empty string

    2. Function `accumulate_stream(chunks: list[dict]) -> dict`:
       - Process a list of SSE chunks in order
       - Return dict with:
         * "full_text": concatenated content from all chunks
         * "tool_calls": merged tool calls from all chunks
         * "finish_reason": the finish_reason from the last chunk that has one
         * "chunk_count": total number of chunks processed

    3. Function `safe_get(d: dict, key: str, default)`:
       - Return d[key] if key exists AND value is not None, otherwise return default
       - This must handle BOTH missing keys AND explicit null values

    Edge cases to handle:
    - chunk with {"choices": null} — choices is explicitly null, not missing
    - chunk with {"choices": [{"delta": null}]} — delta is explicitly null
    - chunk with {"choices": [{"delta": {"content": null}}]} — content is explicitly null
    - chunk with no "choices" key at all — key is missing
    - Empty chunks: {} — no keys at all

    Return ONLY the Python code."""),
    "module_name": "sse_parser",
    "test_code": dedent("""\
    import pytest
    from sse_parser import parse_sse_chunk, accumulate_stream, safe_get

    def test_safe_get_missing_key():
        assert safe_get({}, "x", "default") == "default"

    def test_safe_get_null_value():
        assert safe_get({"x": None}, "x", "default") == "default"

    def test_safe_get_present_value():
        assert safe_get({"x": 42}, "x", "default") == 42

    def test_safe_get_falsy_but_present():
        assert safe_get({"x": 0}, "x", "default") == 0
        assert safe_get({"x": ""}, "x", "default") == ""

    def test_parse_normal_chunk():
        chunk = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        r = parse_sse_chunk(chunk)
        assert r["content"] == "hello"
        assert r["has_content"] is True
        assert r["tool_calls"] == []

    def test_parse_null_choices():
        \"\"\"choices key exists but value is None.\"\"\"
        chunk = {"choices": None}
        r = parse_sse_chunk(chunk)
        assert r["content"] == ""
        assert r["tool_calls"] == []
        assert r["has_content"] is False

    def test_parse_null_delta():
        chunk = {"choices": [{"delta": None, "finish_reason": None}]}
        r = parse_sse_chunk(chunk)
        assert r["content"] == ""
        assert r["tool_calls"] == []

    def test_parse_null_content():
        chunk = {"choices": [{"delta": {"content": None, "tool_calls": None}, "finish_reason": None}]}
        r = parse_sse_chunk(chunk)
        assert r["content"] == ""
        assert r["tool_calls"] == []
        assert r["has_content"] is False

    def test_parse_missing_choices():
        chunk = {}
        r = parse_sse_chunk(chunk)
        assert r["content"] == ""

    def test_accumulate_normal():
        chunks = [
            {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ]
        r = accumulate_stream(chunks)
        assert r["full_text"] == "hello"
        assert r["finish_reason"] == "stop"
        assert r["chunk_count"] == 3

    def test_accumulate_with_nulls():
        chunks = [
            {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]},
            {"choices": None},  # null choices mid-stream
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]},
        ]
        r = accumulate_stream(chunks)
        assert r["full_text"] == "ab"
        assert r["chunk_count"] == 3
    """),
    "expected_tests": 12,
    "rationale": "Fragment 024 teaches the 'or' pattern for null safety. Without it, dict.get returns None for explicit nulls, causing TypeErrors on tests 6-9.",
}


# ===== T5: Database Migration Ordering (Fragment 021) =====
# Without the fragment, models often put CREATE INDEX before ALTER TABLE ADD COLUMN.

TESTS["T5"] = {
    "id": "T5",
    "name": "Database Migration with Correct DDL Ordering",
    "fragment_file": "021-migration-ordering-bugs.md",
    "artifact": dedent("""\

    IMPORTANT — Database Migration Ordering:
    All DDL operations must execute in dependency order:
    1. CREATE TABLE (base schema)
    2. ALTER TABLE ADD COLUMN (schema extensions)
    3. CREATE INDEX (indexes on existing columns)
    4. INSERT seed data (depends on schema being complete)

    Common bug: creating an index on a column that hasn't been added yet.
    This works on existing databases (manual fix hides the bug) but fails on
    fresh databases.

    Always test migrations on a FRESH database, not just an existing one.
    """),
    "prompt": dedent("""\
    Write a Python module `migrator.py` implementing a database migration system
    using sqlite3.

    Requirements:

    1. Class `Migration`:
       - `__init__(self, version: int, description: str, up_sql: str, down_sql: str)`
       - All fields stored as attributes

    2. Class `Migrator`:
       - `__init__(self, db_path: str)` — connect to SQLite database
       - `setup(self)` — create the schema_migrations tracking table if it doesn't exist
         (columns: version INTEGER PRIMARY KEY, description TEXT, applied_at TEXT)
       - `current_version(self) -> int` — return highest applied version, or 0
       - `apply(self, migration: Migration) -> bool` — apply a migration if not already applied.
         Return True if applied, False if already applied.
       - `rollback(self, migration: Migration) -> bool` — rollback if applied.
         Return True if rolled back, False if not applied.
       - `migrate_to(self, migrations: list[Migration], target_version: int)` —
         apply or rollback migrations as needed to reach target version.
         Migrations must be applied in order. Rollbacks must be in reverse order.
       - `close(self)` — close database connection

    3. Function `create_app_schema() -> list[Migration]`:
       Return a list of 3 migrations that build this schema:
       - Migration 1: Create table `users` (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE)
       - Migration 2: Add columns `created_at TEXT DEFAULT CURRENT_TIMESTAMP` and `role TEXT DEFAULT 'user'` to users
       - Migration 3: Create index `idx_users_role` on users(role) AND create index `idx_users_email` on users(email)

       CRITICAL: These must be ordered correctly. The index in migration 3 references
       columns that must exist from migrations 1 and 2.

    Return ONLY the Python code."""),
    "module_name": "migrator",
    "test_code": dedent("""\
    import pytest
    import sqlite3
    import tempfile
    import os
    from migrator import Migration, Migrator, create_app_schema

    @pytest.fixture
    def db_path():
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    def test_fresh_db_full_migration(db_path):
        \"\"\"KEY TEST: All migrations must work on a completely fresh database.\"\"\"
        m = Migrator(db_path)
        m.setup()
        migrations = create_app_schema()
        for mig in migrations:
            assert m.apply(mig) is True
        assert m.current_version() == 3
        # Verify the schema is complete
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "role" in columns, "role column must exist"
        assert "created_at" in columns, "created_at column must exist"
        conn.close()
        m.close()

    def test_index_on_added_column(db_path):
        \"\"\"Verify index on 'role' column works (added in migration 2, indexed in 3).\"\"\"
        m = Migrator(db_path)
        m.setup()
        for mig in create_app_schema():
            m.apply(mig)
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO users (name, email, role) VALUES ('test', 'test@x.com', 'admin')")
        # This query must use the index without error
        rows = conn.execute("SELECT name FROM users WHERE role = 'admin'").fetchall()
        assert len(rows) == 1
        conn.close()
        m.close()

    def test_migration_ordering(db_path):
        \"\"\"Migrations must be numbered 1, 2, 3 in correct dependency order.\"\"\"
        migrations = create_app_schema()
        assert len(migrations) == 3
        assert migrations[0].version == 1
        assert migrations[1].version == 2
        assert migrations[2].version == 3
        # Migration 3 must contain CREATE INDEX, not ALTER TABLE
        assert "CREATE INDEX" in migrations[2].up_sql.upper() or "create index" in migrations[2].up_sql
        # Migration 2 must contain ALTER TABLE, not CREATE INDEX
        sql2 = migrations[1].up_sql.upper()
        assert "ALTER" in sql2 or "alter" in migrations[1].up_sql

    def test_idempotent_apply(db_path):
        m = Migrator(db_path)
        m.setup()
        migrations = create_app_schema()
        assert m.apply(migrations[0]) is True
        assert m.apply(migrations[0]) is False  # Already applied
        m.close()

    def test_rollback(db_path):
        m = Migrator(db_path)
        m.setup()
        migrations = create_app_schema()
        for mig in migrations:
            m.apply(mig)
        assert m.current_version() == 3
        assert m.rollback(migrations[2]) is True
        assert m.current_version() == 2
        m.close()

    def test_migrate_to_target(db_path):
        m = Migrator(db_path)
        m.setup()
        migrations = create_app_schema()
        m.migrate_to(migrations, 2)
        assert m.current_version() == 2
        m.migrate_to(migrations, 3)
        assert m.current_version() == 3
        m.migrate_to(migrations, 1)
        assert m.current_version() == 1
        m.close()
    """),
    "expected_tests": 6,
    "rationale": "Fragment 021 teaches correct DDL ordering. Without it, models often put CREATE INDEX before ALTER TABLE ADD COLUMN, failing test 1 on fresh DBs.",
}


# ---------------------------------------------------------------------------
# Query functions (borrowed from V6 runner pattern)
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def query_vllm(prompt: str, system: str, timeout: float = TIMEOUT) -> tuple[str, float]:
    """Query vllm-mlx on purplemac."""
    start = time.time()
    try:
        resp = httpx.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 8192,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "", time.time() - start
        message = choices[0].get("message") or {}
        return message.get("content") or "", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def query_ollama(prompt: str, system: str, timeout: float = TIMEOUT) -> tuple[str, float]:
    """Query Ollama on purpleroom."""
    start = time.time()
    # Strip /no_think for Ollama (handled via think=false)
    sys_content = system.lstrip("/no_think\n").strip()
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                "think": False,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8192},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content") or "", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def query_vega(prompt: str, system: str, timeout: float = TIMEOUT) -> tuple[str, float]:
    """Query Vega on purpleroom port 8001."""
    start = time.time()
    sys_content = system.lstrip("/no_think\n").strip()
    try:
        resp = httpx.post(
            VEGA_URL,
            json={
                "model": VEGA_MODEL,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 8192,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "", time.time() - start
        message = choices[0].get("message") or {}
        return message.get("content") or "", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_pytest(code: str, test_code: str, module_name: str, timeout: int = 30) -> dict:
    """Write code + tests to temp dir, run pytest, return results."""
    with tempfile.TemporaryDirectory() as td:
        # Write module
        with open(f"{td}/{module_name}.py", "w") as f:
            f.write(code)
        # Write test file
        with open(f"{td}/test_{module_name}.py", "w") as f:
            f.write(test_code)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", f"test_{module_name}.py",
                 "-v", "--tb=short", "--timeout=10", "-x"],
                capture_output=True, text=True, cwd=td, timeout=timeout,
            )
            output = proc.stdout + proc.stderr

            # Parse results
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            errors = len(re.findall(r" ERROR", output))

            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "timed_out": False,
                "output": output,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "timed_out": True,
                "output": "TIMEOUT: pytest did not complete within 30 seconds",
                "returncode": -1,
            }


def run_single_test(test: dict, query_fn, condition: str, results_dir: Path) -> dict:
    """Run one test under one condition (bare or artifact)."""
    test_id = test["id"]
    tag = f"{test_id}_{condition}"

    if condition == "bare":
        system = BARE_SYSTEM
    else:
        system = BARE_SYSTEM + test["artifact"]

    print(f"  [{tag}] Querying model...", end="", flush=True)
    raw, elapsed = query_fn(test["prompt"], system)
    print(f" {elapsed:.1f}s", flush=True)

    if raw.startswith("ERROR:"):
        print(f"  [{tag}] MODEL ERROR: {raw[:100]}")
        return {
            "test_id": test_id,
            "condition": condition,
            "passed": 0,
            "expected": test["expected_tests"],
            "percentage": 0.0,
            "elapsed": elapsed,
            "error": raw,
        }

    code = strip_markdown_fences(raw)

    # Save artifacts
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{tag}_raw.txt").write_text(raw)
    (results_dir / f"{tag}_code.py").write_text(code)

    print(f"  [{tag}] Running pytest...", end="", flush=True)
    result = run_pytest(code, test["test_code"], test["module_name"])
    print(f" {result['passed']}/{test['expected_tests']}", flush=True)

    (results_dir / f"{tag}_pytest.txt").write_text(result["output"])

    return {
        "test_id": test_id,
        "condition": condition,
        "passed": result["passed"],
        "expected": test["expected_tests"],
        "percentage": (result["passed"] / test["expected_tests"] * 100) if test["expected_tests"] > 0 else 0.0,
        "elapsed": elapsed,
        "timed_out": result["timed_out"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Teaching Artifact Delta Test")
    parser.add_argument("--model", choices=["default", "purpleroom", "vega"], default="default",
                        help="Which model to test")
    parser.add_argument("--test", type=str, default=None,
                        help="Run single test (e.g., T1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without running")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for results directory")
    args = parser.parse_args()

    # Select query function
    if args.model == "purpleroom":
        query_fn = query_ollama
        model_name = f"Ollama/{OLLAMA_MODEL}"
    elif args.model == "vega":
        query_fn = query_vega
        model_name = f"Vega/{VEGA_MODEL}"
    else:
        query_fn = query_vllm
        model_name = f"vllm-mlx/{VLLM_MODEL}"

    # Select tests
    if args.test:
        if args.test not in TESTS:
            print(f"Unknown test: {args.test}. Available: {', '.join(TESTS.keys())}")
            sys.exit(1)
        tests_to_run = {args.test: TESTS[args.test]}
    else:
        tests_to_run = TESTS

    # Results directory
    tag = args.tag or f"{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    results_dir = RESULTS_DIR / tag

    if args.dry_run:
        for tid, test in tests_to_run.items():
            print(f"\n{'='*60}")
            print(f"Test {tid}: {test['name']}")
            print(f"Fragment: {test['fragment_file']}")
            print(f"Rationale: {test['rationale']}")
            print(f"\n--- BARE SYSTEM PROMPT ---")
            print(BARE_SYSTEM)
            print(f"\n--- WITH ARTIFACT ---")
            print(BARE_SYSTEM + test['artifact'])
            print(f"\n--- TASK PROMPT ---")
            print(test['prompt'][:200] + "...")
            print(f"\nExpected tests: {test['expected_tests']}")
        return

    print(f"Delta Test — Model: {model_name}")
    print(f"Results: {results_dir}")
    print(f"Tests: {', '.join(tests_to_run.keys())}")
    print(f"{'='*60}")

    all_results = []
    start_time = time.time()

    for tid, test in tests_to_run.items():
        print(f"\n--- {tid}: {test['name']} ---")

        # Run bare (no artifact)
        bare_result = run_single_test(test, query_fn, "bare", results_dir)
        all_results.append(bare_result)

        # Run with artifact
        artifact_result = run_single_test(test, query_fn, "artifact", results_dir)
        all_results.append(artifact_result)

        # Delta
        delta = artifact_result["percentage"] - bare_result["percentage"]
        print(f"  Delta: {bare_result['percentage']:.1f}% -> {artifact_result['percentage']:.1f}% "
              f"({'+' if delta >= 0 else ''}{delta:.1f}%)")

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name}")
    print(f"{'='*60}")

    total_bare_passed = 0
    total_artifact_passed = 0
    total_expected = 0

    for tid in tests_to_run:
        bare = next(r for r in all_results if r["test_id"] == tid and r["condition"] == "bare")
        artifact = next(r for r in all_results if r["test_id"] == tid and r["condition"] == "artifact")
        delta = artifact["percentage"] - bare["percentage"]
        total_bare_passed += bare["passed"]
        total_artifact_passed += artifact["passed"]
        total_expected += bare["expected"]

        marker = "+++" if delta > 10 else "++" if delta > 0 else "===" if delta == 0 else "---"
        print(f"  {tid} {test['name']:45s} "
              f"Bare: {bare['passed']:2d}/{bare['expected']:2d} ({bare['percentage']:5.1f}%)  "
              f"Art: {artifact['passed']:2d}/{artifact['expected']:2d} ({artifact['percentage']:5.1f}%)  "
              f"Delta: {delta:+6.1f}%  {marker}")

    bare_total_pct = (total_bare_passed / total_expected * 100) if total_expected > 0 else 0
    artifact_total_pct = (total_artifact_passed / total_expected * 100) if total_expected > 0 else 0
    total_delta = artifact_total_pct - bare_total_pct

    print(f"\n  TOTAL: Bare {total_bare_passed}/{total_expected} ({bare_total_pct:.1f}%) "
          f"-> Artifact {total_artifact_passed}/{total_expected} ({artifact_total_pct:.1f}%) "
          f"Delta: {total_delta:+.1f}%")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save summary
    summary = {
        "model": model_name,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_seconds": total_time,
        "tests": {tid: {
            "bare": next(r for r in all_results if r["test_id"] == tid and r["condition"] == "bare"),
            "artifact": next(r for r in all_results if r["test_id"] == tid and r["condition"] == "artifact"),
        } for tid in tests_to_run},
        "totals": {
            "bare_passed": total_bare_passed,
            "artifact_passed": total_artifact_passed,
            "expected": total_expected,
            "bare_percentage": bare_total_pct,
            "artifact_percentage": artifact_total_pct,
            "delta": total_delta,
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Summary saved to {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
