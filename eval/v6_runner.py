#!/usr/bin/env python3
"""
Purple Proving Ground — V6 Contamination-Proof Battery Runner
Dual-backend: local (vllm-mlx) + Claude (Anthropic API).
Inherits V5 layer system. Adds edit-accuracy and reasoning metrics.

Usage:
  python v6_runner.py --backend local --tag local-v1
  python v6_runner.py --backend claude --tag claude-v1
  python v6_runner.py --backend both --tag h2h-v1
  python v6_runner.py --compare local-v1 claude-v1
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_API_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
LOCAL_MODEL = os.environ.get("VLLM_MODEL", "lmstudio-community/Qwen3-Coder-Next-MLX-4bit")
LOCAL_TIMEOUT = float(os.environ.get("LOCAL_TIMEOUT", "300"))
# Set OLLAMA_NATIVE=1 to use Ollama's /api/chat endpoint directly
# Set OLLAMA_THINK=1 to enable thinking mode (think=true), default is think=false
_USE_OLLAMA_NATIVE = os.environ.get("OLLAMA_NATIVE", "0") == "1"
_OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "0") == "1"
OLLAMA_NATIVE_URL = os.environ.get("OLLAMA_NATIVE_URL", "http://localhost:11434/api/chat")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EVAL_DIR = Path(__file__).parent
RESULTS_BASE = EVAL_DIR

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM = """You are an expert Python programmer. Write clean, correct, production-quality code.

CRITICAL SELF-CHECK BEFORE RETURNING CODE:
1. Verify every method name you DEFINE matches every method name you CALL.
2. If a method is decorated @staticmethod/@classmethod, use class name, not self.xxx.
3. Implement ALL listed features. Do not stub with `pass`.
4. Ensure constructor (__init__) calls match the method names you defined.

Return ONLY the Python code, no explanation, no markdown fences."""

# Qwen3-Coder-Next uses /no_think prefix to disable thinking mode.
# Qwen3.5 (Vega) does NOT — /no_think triggers prose-before-code.
# Detect via VLLM_MODEL env var: if it contains "vega", skip /no_think.
_is_vega = "vega" in LOCAL_MODEL.lower()
LOCAL_SYSTEM = _BASE_SYSTEM if _is_vega else f"/no_think\n{_BASE_SYSTEM}"
CLAUDE_SYSTEM = _BASE_SYSTEM

_BASE_CORRECTION = """You are an expert Python programmer fixing a bug. Return ONLY the corrected complete Python module, no explanation, no markdown fences."""

CORRECTION_SYSTEM_LOCAL = _BASE_CORRECTION if _is_vega else f"/no_think\n{_BASE_CORRECTION}"
CORRECTION_SYSTEM_CLAUDE = _BASE_CORRECTION


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

sys.path.insert(0, str(EVAL_DIR))
from v6_tests import TESTS


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    text = text.strip()
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_code_from_split(raw: str) -> str:
    """Extract code from a split response (JSON + ---SPLIT--- + code)."""
    if "---SPLIT---" in raw:
        parts = raw.split("---SPLIT---", 1)
        return strip_markdown_fences(parts[1])
    return strip_markdown_fences(raw)


def count_expected_tests(test: dict) -> int:
    """Count expected test functions from test definition (fixed denominator)."""
    if test.get("turns"):
        return sum(
            len(re.findall(r'def test_', turn.get("test_code", "")))
            for turn in test["turns"]
        )
    return len(re.findall(r'def test_', test.get("test_code", "")))


# ---------------------------------------------------------------------------
# Backend: Local (vllm-mlx)
# ---------------------------------------------------------------------------

def query_local(prompt: str, system: str = LOCAL_SYSTEM,
                context: list | None = None, timeout: float = LOCAL_TIMEOUT) -> tuple[str, float]:
    """Send prompt to local model. Returns (response_text, elapsed).
    If OLLAMA_NATIVE=1: uses Ollama /api/chat endpoint directly.
      OLLAMA_THINK=0 (default): think=false, fast, no degeneration guard
      OLLAMA_THINK=1: think=true, slower but prevents degeneration on complex tasks
    Otherwise: uses OpenAI-compat endpoint (vllm-mlx default).
    """
    start = time.time()
    sys_content = system.lstrip("/no_think\n").strip() if _USE_OLLAMA_NATIVE else system
    messages = [{"role": "system", "content": sys_content}]
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": prompt})

    try:
        if _USE_OLLAMA_NATIVE:
            resp = httpx.post(
                OLLAMA_NATIVE_URL,
                json={"model": LOCAL_MODEL, "messages": messages, "think": _OLLAMA_THINK,
                      "stream": False, "options": {"temperature": 0, "num_predict": 8192}},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content") or "", time.time() - start
        else:
            resp = httpx.post(
                LOCAL_API_URL,
                json={"model": LOCAL_MODEL, "messages": messages, "temperature": 0, "max_tokens": 8192},
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
# Backend: Claude (Anthropic API)
# ---------------------------------------------------------------------------

def query_claude(prompt: str, system: str = CLAUDE_SYSTEM,
                 context: list | None = None, timeout: float = 300.0) -> tuple[str, float]:
    """Send prompt to Claude API. Returns (response_text, elapsed)."""
    if not CLAUDE_API_KEY:
        return "ERROR: ANTHROPIC_API_KEY not set", 0.0

    start = time.time()
    messages = []
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": prompt})

    try:
        resp = httpx.post(
            CLAUDE_API_URL,
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "system": system,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        return text, time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


# ---------------------------------------------------------------------------
# Layer functions (inherited from V5)
# ---------------------------------------------------------------------------

MAX_REASONABLE_LENGTH = 25000
REPETITION_THRESHOLD = 5


def check_repetition_degeneration(code: str) -> str | None:
    """Layer 0: Detect repetition degeneration."""
    if len(code) > MAX_REASONABLE_LENGTH:
        lines = code.split("\n")
        line_counts = Counter(line.strip() for line in lines if line.strip())
        if line_counts:
            most_common_line, most_common_count = line_counts.most_common(1)[0]
            if most_common_count >= REPETITION_THRESHOLD:
                return f"Repetition: {len(code)} chars, '{most_common_line[:50]}' x{most_common_count}"

    lines = code.split("\n")
    if len(lines) > 50:
        line_counts = Counter(line.strip() for line in lines if line.strip())
        if line_counts:
            most_common_line, most_common_count = line_counts.most_common(1)[0]
            non_empty = sum(1 for l in lines if l.strip())
            if most_common_count > max(REPETITION_THRESHOLD, non_empty * 0.2):
                return f"Repetition: '{most_common_line[:50]}' {most_common_count}/{non_empty}"
    return None


def check_import(code: str, module: str) -> str | None:
    """Layer 1: Try to import the module."""
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/{module}.py", "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                capture_output=True, text=True, cwd=td, timeout=10,
            )
            if proc.returncode != 0:
                return (proc.stderr or proc.stdout).strip()
        except subprocess.TimeoutExpired:
            return "Import timed out"
        except Exception as e:
            return str(e)
    return None


def extract_expected_names(test_code: str, module: str) -> list[str]:
    """Parse test code for imported names."""
    names = []
    pattern = rf"from\s+{re.escape(module)}\s+import\s+(.+)"
    for match in re.finditer(pattern, test_code):
        for name in re.split(r"[,\s]+", match.group(1)):
            name = name.strip().rstrip("\\()")
            if name and name.isidentifier() and name != "as":
                names.append(name)
    return names


def check_names_exist(code: str, module: str, test_code: str) -> list[str]:
    """Layer 1.5: Check expected names exist."""
    expected = extract_expected_names(test_code, module)
    if not expected:
        return []
    missing = []
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/{module}.py", "w") as f:
            f.write(code)
        for name in expected:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", f"import {module}\nassert hasattr({module}, '{name}')"],
                    capture_output=True, text=True, cwd=td, timeout=5,
                )
                if proc.returncode != 0:
                    missing.append(name)
            except Exception:
                missing.append(name)
    return missing


def run_pytest(code: str, module: str, test_code: str,
               timeout: int = 60) -> tuple[str, int, int]:
    """Run pytest. Returns (output, passed, failed)."""
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/{module}.py", "w") as f:
            f.write(code)
        with open(f"{td}/test_{module}.py", "w") as f:
            f.write(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", f"{td}/test_{module}.py",
                 "-v", "--tb=short", "--timeout=10"],
                capture_output=True, text=True, cwd=td, timeout=timeout,
            )
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: Tests exceeded timeout"
        except Exception as e:
            output = f"RUN ERROR: {e}"
    passed = output.count(" PASSED")
    failed = output.count(" FAILED") + output.count(" ERROR")
    return output, passed, failed


def extract_failures(pytest_output: str, limit: int = 5) -> str:
    """Extract failure details from pytest output."""
    lines = pytest_output.split("\n")
    failures = []
    for i, line in enumerate(lines):
        if "FAILED" in line or ("AssertionError" in line) or ("Error" in line and "E " in line):
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            failures.append("\n".join(lines[start:end]))
    return "\n\n".join(failures[:limit]) if failures else ""


# ---------------------------------------------------------------------------
# Test runner with layers
# ---------------------------------------------------------------------------

def run_test(test: dict, backend: str, results_dir: Path) -> dict:
    """Run a single test with all layers."""
    test_id = test["id"]
    module = test["module"]
    test_code = test["test_code"]
    category = test.get("category", "A")

    query_fn = query_local if backend == "local" else query_claude
    system = LOCAL_SYSTEM if backend == "local" else CLAUDE_SYSTEM
    correction_sys = CORRECTION_SYSTEM_LOCAL if backend == "local" else CORRECTION_SYSTEM_CLAUDE

    print(f"\n{'='*60}")
    print(f"[{backend.upper()}] TEST {test_id}: {test['name']}")
    print(f"{'='*60}")

    # Handle multi-turn tests (Category E1)
    if test.get("turns"):
        return run_multiturn_test(test, backend, results_dir)

    # Handle response_parser=split (Category C1, C2, C3, E2)
    parser = test.get("response_parser")
    expected_total = count_expected_tests(test)

    # ---- Generate code ----
    print(f"  [Gen] Querying {backend}...")
    raw_response, gen_time = query_fn(test["prompt"], system)

    if raw_response.startswith("ERROR:"):
        print(f"  MODEL ERROR: {raw_response}")
        return _save_result(test, results_dir, gen_time, "", raw_response,
                            0, expected_total, 0, expected_total,
                            backend=backend)

    # Handle split responses (diagnosis + code)
    if parser == "split":
        code = extract_code_from_split(raw_response)
    else:
        code = strip_markdown_fences(raw_response)

    print(f"  [Gen] {gen_time:.1f}s, {len(code)} chars")

    correction_time = 0.0
    layers_triggered = {"L0": False, "L1": False, "L1.5": False, "L2": False}

    # ---- Layer 0: Repetition guard (local only) ----
    if backend == "local":
        degen = check_repetition_degeneration(code)
        if degen:
            layers_triggered["L0"] = True
            print(f"  [L0] {degen}")
            regen_prompt = f"{test['prompt']}\n\nIMPORTANT: Keep code concise. Avoid repeating lines."
            regen_raw, regen_time = query_fn(regen_prompt, system)
            correction_time += regen_time
            if not regen_raw.startswith("ERROR:"):
                regen_code = strip_markdown_fences(regen_raw)
                if check_repetition_degeneration(regen_code) is None:
                    code = regen_code
                    print(f"  [L0] Fixed ({regen_time:.1f}s)")

    # ---- Layer 1: Import check ----
    import_err = check_import(code, module)
    if import_err:
        layers_triggered["L1"] = True
        print(f"  [L1] Import failed: {import_err[:100]}")
        fix_prompt = f"Import error:\n{import_err}\n\nOriginal task:\n{test['prompt']}\n\nFix and return ONLY the code."
        fix_raw, fix_time = query_fn(fix_prompt, correction_sys,
                                      context=[{"role": "assistant", "content": raw_response}])
        correction_time += fix_time
        fix_code = strip_markdown_fences(fix_raw)
        if fix_code and not fix_code.startswith("ERROR:") and check_import(fix_code, module) is None:
            code = fix_code
            print(f"  [L1] Fixed ({fix_time:.1f}s)")

    # ---- Layer 1.5: Name check ----
    missing = check_names_exist(code, module, test_code)
    if missing:
        layers_triggered["L1.5"] = True
        print(f"  [L1.5] Missing: {', '.join(missing)}")
        name_prompt = f"Missing names: {', '.join(missing)}\n\nTask:\n{test['prompt']}\n\nFix and return ONLY the code."
        name_raw, name_time = query_fn(name_prompt, correction_sys,
                                        context=[{"role": "assistant", "content": code}])
        correction_time += name_time
        name_code = strip_markdown_fences(name_raw)
        if name_code and not name_code.startswith("ERROR:"):
            still_missing = check_names_exist(name_code, module, test_code)
            if len(still_missing) < len(missing) and check_import(name_code, module) is None:
                code = name_code
                print(f"  [L1.5] Fixed ({name_time:.1f}s)")

    # ---- FTA: First-try accuracy ----
    print(f"  [FTA] Running tests...")
    fta_output, fta_passed, fta_failed = run_pytest(code, module, test_code)
    fta_total = fta_passed + fta_failed
    print(f"  [FTA] {fta_passed}/{fta_total} ({fta_passed*100//fta_total if fta_total else 0}%)")

    aca_passed, aca_failed, aca_total, aca_output = fta_passed, fta_failed, fta_total, fta_output
    final_code = code

    # ---- Layer 2: Pytest feedback ----
    if fta_failed > 0 or (fta_total == 0 and expected_total > 0):
        layers_triggered["L2"] = True
        failures = extract_failures(fta_output)
        if not failures:
            for line in fta_output.split("\n"):
                if "FAILED" in line or "ERROR" in line or "SyntaxError" in line:
                    failures += line.strip() + "\n"
        num_failures = fta_failed if fta_failed > 0 else expected_total
        print(f"  [L2] {num_failures} failures, requesting fix...")
        fb_prompt = f"Test failures:\n{failures}\n\nTask:\n{test['prompt']}\n\nFix ALL bugs. Return ONLY the Python code, no JSON, no explanation."
        fb_raw, fb_time = query_fn(fb_prompt, correction_sys,
                                    context=[{"role": "assistant", "content": code}])
        correction_time += fb_time
        # Apply split parser if needed
        if parser == "split":
            fb_code = extract_code_from_split(fb_raw)
        else:
            fb_code = strip_markdown_fences(fb_raw)
        if fb_code and not fb_code.startswith("ERROR:"):
            aca_output, aca_passed, aca_failed = run_pytest(fb_code, module, test_code)
            aca_total = aca_passed + aca_failed
            # Accept L2 fix only if it improves AND doesn't lose test coverage
            if aca_passed > fta_passed or (aca_passed == fta_passed and aca_total >= fta_total):
                final_code = fb_code
                print(f"  [L2] Fixed: {aca_passed}/{aca_total} ({fb_time:.1f}s)")
            else:
                aca_passed, aca_failed, aca_total, aca_output = fta_passed, fta_failed, fta_total, fta_output
                final_code = code
                print(f"  [L2] Worse, keeping original")

    # ---- Edit accuracy (Category B) ----
    edit_accuracy = None
    if category == "B" and "provided_code" in test:
        orig_lines = test["provided_code"].strip().split("\n")
        new_lines = final_code.strip().split("\n")
        # Count how many original lines are preserved
        orig_set = set(line.rstrip() for line in orig_lines)
        preserved = sum(1 for line in new_lines if line.rstrip() in orig_set)
        edit_accuracy = preserved / len(orig_lines) if orig_lines else 0

    return _save_result(
        test, results_dir, gen_time + correction_time, final_code, aca_output,
        fta_passed, fta_total, aca_passed, aca_total,
        backend=backend, layers=layers_triggered,
        raw_response=raw_response, edit_accuracy=edit_accuracy,
    )


def run_multiturn_test(test: dict, backend: str, results_dir: Path) -> dict:
    """Run a multi-turn test (Category E)."""
    test_id = test["id"]
    module = test["module"]
    turns = test["turns"]

    query_fn = query_local if backend == "local" else query_claude
    system = LOCAL_SYSTEM if backend == "local" else CLAUDE_SYSTEM

    print(f"\n{'='*60}")
    print(f"[{backend.upper()}] TEST {test_id}: {test['name']} ({len(turns)} turns)")
    print(f"{'='*60}")

    total_time = 0.0
    current_code = ""
    all_passed = 0
    all_failed = 0
    all_total = 0
    fta_passed_total = 0
    fta_total_total = 0
    all_outputs = []
    layers_triggered = {"L0": False, "L1": False, "L1.5": False, "L2": False}

    for turn_idx, turn in enumerate(turns):
        turn_num = turn_idx + 1
        print(f"\n  --- Turn {turn_num}/{len(turns)} ---")

        # Format prompt with previous code if needed
        prompt = turn["prompt"]
        if "{previous_code}" in prompt and current_code:
            prompt = prompt.format(previous_code=current_code)

        # Query
        print(f"  [T{turn_num}] Querying {backend}...")
        raw, gen_time = query_fn(prompt, system)
        total_time += gen_time

        if raw.startswith("ERROR:"):
            print(f"  [T{turn_num}] ERROR: {raw}")
            all_outputs.append(f"Turn {turn_num}: {raw}")
            continue

        code = strip_markdown_fences(raw)
        print(f"  [T{turn_num}] {gen_time:.1f}s, {len(code)} chars")

        # L0: Repetition check (local only)
        if backend == "local":
            degen = check_repetition_degeneration(code)
            if degen:
                layers_triggered["L0"] = True
                print(f"  [T{turn_num}/L0] {degen}")
                regen, rt = query_fn(prompt + "\n\nKeep code concise.", system)
                total_time += rt
                regen_code = strip_markdown_fences(regen)
                if check_repetition_degeneration(regen_code) is None:
                    code = regen_code

        # L1: Import check
        import_err = check_import(code, module)
        if import_err:
            layers_triggered["L1"] = True
            print(f"  [T{turn_num}/L1] Import failed, requesting fix...")
            fix_prompt = f"Import error:\n{import_err}\n\nOriginal:\n{prompt}\n\nFix and return ONLY the code."
            fix_raw, ft = query_fn(fix_prompt,
                                    CORRECTION_SYSTEM_LOCAL if backend == "local" else CORRECTION_SYSTEM_CLAUDE,
                                    context=[{"role": "assistant", "content": raw}])
            total_time += ft
            fix_code = strip_markdown_fences(fix_raw)
            if fix_code and not fix_code.startswith("ERROR:") and check_import(fix_code, module) is None:
                code = fix_code

        # Run this turn's tests
        turn_test_code = turn["test_code"]
        print(f"  [T{turn_num}] Running turn tests...")
        output, passed, failed = run_pytest(code, module, turn_test_code)
        total = passed + failed
        print(f"  [T{turn_num}] {passed}/{total}")

        # FTA tracking
        if turn_idx == 0:
            fta_passed_total += passed
            fta_total_total += total
        else:
            fta_passed_total += passed
            fta_total_total += total

        # L2: Fix if failures
        if failed > 0:
            layers_triggered["L2"] = True
            failures = extract_failures(output)
            fb_prompt = f"Test failures:\n{failures}\n\nTask:\n{prompt}\n\nFix ALL bugs. Return ONLY the code."
            fb_raw, fb_time = query_fn(fb_prompt,
                                        CORRECTION_SYSTEM_LOCAL if backend == "local" else CORRECTION_SYSTEM_CLAUDE,
                                        context=[{"role": "assistant", "content": code}])
            total_time += fb_time
            fb_code = strip_markdown_fences(fb_raw)
            if fb_code and not fb_code.startswith("ERROR:"):
                out2, p2, f2 = run_pytest(fb_code, module, turn_test_code)
                if p2 >= passed:
                    code = fb_code
                    passed = p2
                    failed = f2
                    total = p2 + f2
                    output = out2
                    print(f"  [T{turn_num}/L2] Fixed: {passed}/{total}")

        all_passed += passed
        all_total += total
        all_failed += failed
        all_outputs.append(f"=== Turn {turn_num} ===\n{output}")
        current_code = code

        # Regression: also run ALL previous turns' tests with current code
        if turn_idx > 0:
            for prev_idx in range(turn_idx):
                prev_test = turns[prev_idx]["test_code"]
                _, pp, pf = run_pytest(code, module, prev_test)
                if pf > 0:
                    print(f"  [T{turn_num}] REGRESSION: Turn {prev_idx+1} tests: {pp}/{pp+pf}")

    # Save
    combined_output = "\n\n".join(all_outputs)
    results_dir.mkdir(parents=True, exist_ok=True)
    if current_code:
        (results_dir / f"{test_id}_code.py").write_text(current_code)
    (results_dir / f"{test_id}_pytest.txt").write_text(combined_output[-5000:])

    return _save_result(
        test, results_dir, total_time, current_code, combined_output,
        fta_passed_total, fta_total_total, all_passed, all_total,
        backend=backend, layers=layers_triggered,
    )


def _save_result(test, results_dir, elapsed, code, output,
                 fta_passed, fta_total, aca_passed, aca_total,
                 backend="local", layers=None, raw_response="",
                 edit_accuracy=None):
    """Save result files and return result dict."""
    test_id = test["id"]
    expected = count_expected_tests(test)
    results_dir.mkdir(parents=True, exist_ok=True)

    if code:
        (results_dir / f"{test_id}_code.py").write_text(code)
    if raw_response:
        (results_dir / f"{test_id}_raw.txt").write_text(raw_response)
    if output:
        (results_dir / f"{test_id}_pytest.txt").write_text(output[-5000:])

    # Use expected test count as denominator floor — prevents timeouts/errors
    # from silently shrinking the denominator
    fta_denom = max(fta_total, expected)
    aca_denom = max(aca_total, expected)

    result = {
        "id": test_id,
        "name": test["name"],
        "category": test.get("category", "A"),
        "backend": backend,
        "elapsed": round(elapsed, 2),
        "code_length": len(code) if code else 0,
        "fta_passed": fta_passed,
        "fta_total": fta_denom,
        "fta_rate": round(fta_passed / fta_denom, 4) if fta_denom > 0 else 0,
        "aca_passed": aca_passed,
        "aca_total": aca_denom,
        "aca_rate": round(aca_passed / aca_denom, 4) if aca_denom > 0 else 0,
        "expected_total": expected,
        "layers_triggered": layers or {},
        "edit_accuracy": round(edit_accuracy, 4) if edit_accuracy is not None else None,
        "output": output[-3000:] if output else "",
    }
    with open(results_dir / f"{test_id}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

def write_summary(results: list[dict], results_dir: Path, backend: str, tag: str):
    """Write aggregate summary."""
    fta_sum = sum(r["fta_passed"] for r in results)
    aca_sum = sum(r["aca_passed"] for r in results)
    # Use expected_total (fixed denominator from test defs) — never shrinks on timeout/error
    total = sum(r.get("expected_total", r["fta_total"]) for r in results)
    total_time = sum(r["elapsed"] for r in results)

    # Per-category scores
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"fta_passed": 0, "aca_passed": 0, "total": 0}
        categories[cat]["fta_passed"] += r["fta_passed"]
        categories[cat]["aca_passed"] += r["aca_passed"]
        categories[cat]["total"] += r.get("expected_total", r["fta_total"])

    cat_scores = {}
    for cat, data in categories.items():
        cat_scores[cat] = {
            "fta_rate": round(data["fta_passed"] / data["total"], 4) if data["total"] else 0,
            "aca_rate": round(data["aca_passed"] / data["total"], 4) if data["total"] else 0,
        }

    layer_counts = {"L0": 0, "L1": 0, "L1.5": 0, "L2": 0}
    for r in results:
        for layer, triggered in r.get("layers_triggered", {}).items():
            if triggered:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

    summary = {
        "backend": backend,
        "model": LOCAL_MODEL if backend == "local" else CLAUDE_MODEL,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": total,
        "fta_total_passed": fta_sum,
        "aca_total_passed": aca_sum,
        "fta_rate": round(fta_sum / total, 4) if total else 0,
        "aca_rate": round(aca_sum / total, 4) if total else 0,
        "total_time": round(total_time, 1),
        "category_scores": cat_scores,
        "layer_counts": layer_counts,
        "results": results,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def compare_results(local_dir: Path, claude_dir: Path, output_dir: Path):
    """Generate gap report comparing local vs Claude results."""
    local_summary = json.loads((local_dir / "summary.json").read_text())
    claude_summary = json.loads((claude_dir / "summary.json").read_text())

    # Per-category comparison
    cat_comparison = {}
    all_cats = set(local_summary.get("category_scores", {}).keys()) | set(claude_summary.get("category_scores", {}).keys())
    for cat in sorted(all_cats):
        local_score = local_summary.get("category_scores", {}).get(cat, {}).get("aca_rate", 0)
        claude_score = claude_summary.get("category_scores", {}).get(cat, {}).get("aca_rate", 0)
        cat_comparison[cat] = {
            "local": local_score,
            "claude": claude_score,
            "gap": round(local_score - claude_score, 4),
        }

    # Per-test comparison
    local_by_id = {r["id"]: r for r in local_summary.get("results", [])}
    claude_by_id = {r["id"]: r for r in claude_summary.get("results", [])}
    per_test = []
    capability_gaps = []

    for test_id in sorted(set(local_by_id.keys()) | set(claude_by_id.keys())):
        lr = local_by_id.get(test_id, {})
        cr = claude_by_id.get(test_id, {})
        gap = (lr.get("aca_rate", 0) - cr.get("aca_rate", 0))
        entry = {
            "test_id": test_id,
            "name": lr.get("name") or cr.get("name", ""),
            "category": lr.get("category") or cr.get("category", ""),
            "local_aca": f"{lr.get('aca_passed',0)}/{lr.get('aca_total',0)}",
            "claude_aca": f"{cr.get('aca_passed',0)}/{cr.get('aca_total',0)}",
            "local_rate": lr.get("aca_rate", 0),
            "claude_rate": cr.get("aca_rate", 0),
            "gap": round(gap, 4),
            "local_layers": lr.get("layers_triggered", {}),
            "local_time": lr.get("elapsed", 0),
            "claude_time": cr.get("elapsed", 0),
        }
        per_test.append(entry)

        if gap < -0.2:  # Local significantly worse
            severity = "critical" if gap < -0.5 else "high" if gap < -0.3 else "moderate"
            # Determine if improvable via scaffolding
            layers = lr.get("layers_triggered", {})
            if any(layers.values()):
                improvable = "scaffolding (layers already helping)"
            elif lr.get("category") == "B":
                improvable = "scaffolding (add diff-format layer)"
            elif lr.get("category") == "C":
                improvable = "thinking_mode (enable chain-of-thought)"
            elif lr.get("category") == "E":
                improvable = "orchestration (multi-turn management)"
            else:
                improvable = "model_upgrade (base capability gap)"
            capability_gaps.append({
                "test_id": test_id,
                "dimension": entry["category"],
                "severity": severity,
                "gap": round(gap, 4),
                "improvable_via": improvable,
            })

    report = {
        "local_model": local_summary.get("model", ""),
        "claude_model": claude_summary.get("model", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "local_aca": local_summary.get("aca_rate", 0),
            "claude_aca": claude_summary.get("aca_rate", 0),
            "gap": round(local_summary.get("aca_rate", 0) - claude_summary.get("aca_rate", 0), 4),
        },
        "category_scores": cat_comparison,
        "per_test_comparison": per_test,
        "capability_gaps": capability_gaps,
        "scaffolding_recommendations": _generate_recommendations(capability_gaps),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gap_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'#'*60}")
    print(f"  V6 GAP REPORT: {report['local_model']} vs {report['claude_model']}")
    print(f"{'#'*60}")
    print(f"  Overall — Local: {report['overall']['local_aca']:.1%}  Claude: {report['overall']['claude_aca']:.1%}  Gap: {report['overall']['gap']:+.1%}")
    print()
    for cat, scores in sorted(cat_comparison.items()):
        bar_l = "█" * int(scores["local"] * 20)
        bar_c = "█" * int(scores["claude"] * 20)
        print(f"  {cat:12s}  Local {bar_l:<20s} {scores['local']:.0%}")
        print(f"  {' ':12s}  Claude {bar_c:<20s} {scores['claude']:.0%}")
        print()
    if capability_gaps:
        print(f"  Critical Gaps:")
        for g in capability_gaps:
            print(f"    {g['test_id']:20s} {g['severity']:10s} {g['gap']:+.0%}  → {g['improvable_via']}")
    print(f"\n  Report: {output_dir / 'gap_report.json'}")
    return report


def _generate_recommendations(gaps: list[dict]) -> list[str]:
    """Generate scaffolding recommendations from gaps."""
    recs = []
    categories_hit = {g["dimension"] for g in gaps}

    if "B" in categories_hit:
        recs.append("Add L4 (diff-format enforcement): prompt model to preserve unchanged code sections explicitly")
    if "C" in categories_hit:
        recs.append("Enable selective thinking mode for reasoning tasks: test with /think prefix on complex problems")
    if "E" in categories_hit:
        recs.append("Improve multi-turn orchestration: pass full conversation context, add regression checks between turns")
    if any(g["severity"] == "critical" for g in gaps):
        recs.append("Critical gaps may require model upgrade (larger model or newer generation)")
    if not gaps:
        recs.append("No significant gaps detected. Local model is competitive with Claude on this battery.")
    return recs


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results(results: list[dict], backend: str):
    """Print results table."""
    print(f"\n{'#'*60}")
    print(f"  V6 BATTERY — {backend.upper()} RESULTS")
    print(f"{'#'*60}")
    print(f"{'Test':<40} {'Cat':>3} {'FTA':>8} {'ACA':>8} {'L0':>3} {'L1':>3} {'L2':>3} {'Time':>7}")
    print(f"{'-'*40} {'-'*3} {'-'*8} {'-'*8} {'-'*3} {'-'*3} {'-'*3} {'-'*7}")

    fta_sum = aca_sum = total_n = 0
    total_time = 0.0

    for r in results:
        ft = r["fta_total"]
        fp = r["fta_passed"]
        ap = r["aca_passed"]
        fta_str = f"{fp}/{ft}" if ft else "ERR"
        aca_str = f"{ap}/{ft}" if ft else "ERR"
        lt = r.get("layers_triggered", {})
        print(f"{r['name']:<40} {r['category']:>3} {fta_str:>8} {aca_str:>8} "
              f"{'Y' if lt.get('L0') else '':>3} {'Y' if lt.get('L1') else '':>3} "
              f"{'Y' if lt.get('L2') else '':>3} {r['elapsed']:>6.1f}s")
        fta_sum += fp
        aca_sum += ap
        total_n += ft
        total_time += r["elapsed"]

    print(f"{'-'*40} {'-'*3} {'-'*8} {'-'*8} {'-'*3} {'-'*3} {'-'*3} {'-'*7}")
    fta_pct = f"{fta_sum*100//total_n}%" if total_n else "N/A"
    aca_pct = f"{aca_sum*100//total_n}%" if total_n else "N/A"
    print(f"{'TOTAL':<40} {'':>3} {fta_pct:>8} {aca_pct:>8} {'':>3} {'':>3} {'':>3} {total_time:>6.1f}s")
    print(f"\n  FTA: {fta_sum}/{total_n} | ACA: {aca_sum}/{total_n} | Time: {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V6 Proving Ground Runner")
    parser.add_argument("--backend", choices=["local", "claude", "both"], default="local")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--category", help="Run only this category (A, B, C, D, E)")
    parser.add_argument("--compare", nargs=2, metavar=("LOCAL_TAG", "CLAUDE_TAG"),
                        help="Compare two existing result sets")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        local_dir = RESULTS_BASE / f"v6-results-{args.compare[0]}"
        claude_dir = RESULTS_BASE / f"v6-results-{args.compare[1]}"
        output_dir = RESULTS_BASE / f"v6-comparison-{args.compare[0]}-vs-{args.compare[1]}"
        compare_results(local_dir, claude_dir, output_dir)
        return

    # Filter tests by category
    tests = TESTS
    if args.category:
        tests = [t for t in TESTS if t.get("category") == args.category]
        if not tests:
            print(f"No tests in category {args.category}")
            sys.exit(1)

    backends = [args.backend] if args.backend != "both" else ["local", "claude"]

    for backend in backends:
        tag = f"{args.tag}-{backend}" if args.backend == "both" else args.tag
        results_dir = RESULTS_BASE / f"v6-results-{tag}"

        print(f"\n{'#'*60}")
        print(f"  PURPLE PROVING GROUND — V6 BATTERY")
        print(f"  Backend: {backend}")
        print(f"  Model: {LOCAL_MODEL if backend == 'local' else CLAUDE_MODEL}")
        print(f"  Tests: {len(tests)}")
        print(f"  Tag: {tag}")
        print(f"  Output: {results_dir}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        # Verify backend
        if backend == "local":
            try:
                resp = httpx.get("http://localhost:8000/v1/models", timeout=5)
                resp.raise_for_status()
                print(f"  Local backend: OK")
            except Exception as e:
                print(f"  Local backend ERROR: {e}")
                sys.exit(1)
        elif backend == "claude":
            if not CLAUDE_API_KEY:
                print("  ERROR: ANTHROPIC_API_KEY not set")
                sys.exit(1)
            print(f"  Claude backend: OK (key set)")

        results = []
        for test in tests:
            result = run_test(test, backend, results_dir)
            results.append(result)

        print_results(results, backend)
        summary = write_summary(results, results_dir, backend, tag)
        print(f"\n  Results saved: {results_dir}")

    # Auto-compare if running both
    if args.backend == "both":
        local_dir = RESULTS_BASE / f"v6-results-{args.tag}-local"
        claude_dir = RESULTS_BASE / f"v6-results-{args.tag}-claude"
        output_dir = RESULTS_BASE / f"v6-comparison-{args.tag}"
        compare_results(local_dir, claude_dir, output_dir)


if __name__ == "__main__":
    main()
