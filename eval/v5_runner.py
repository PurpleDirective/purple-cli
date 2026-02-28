#!/usr/bin/env python3
"""
Purple Proving Ground — V5 Sovereign Battery Runner
Five layers of defense + quantization comparison support.

Layer 0: Output length guard — detects repetition degeneration, auto-regenerates
Layer 1: Import check + one correction turn
Layer 1.5: Name existence check — verifies expected classes/functions exist after import
Layer 2: Pytest feedback + one resubmission
Layer 3: Enhanced system prompt with self-review directives

Tracks FTA (First-Try Accuracy) and ACA (After-Correction Accuracy) separately.
"""

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

API_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "lmstudio-community/Qwen3-Coder-Next-MLX-4bit")
RESULTS_DIR = Path(__file__).parent / "v5-results"

# ---------------------------------------------------------------------------
# Layer 3: Enhanced system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think
You are an expert Python programmer. Write clean, correct, production-quality code.

CRITICAL SELF-CHECK BEFORE RETURNING CODE:
1. Verify every method name you DEFINE matches every method name you CALL. Check underscores, capitalization, and spelling.
2. If a method is decorated @staticmethod or @classmethod, do NOT call instance methods (self.xxx) from within it. Use the class name or make them @staticmethod too.
3. Implement ALL listed features. Do not stub any operation with `pass` or placeholder. If a feature is hard, implement your best attempt.
4. Ensure constructor (__init__) calls match the method names you actually defined.

Return ONLY the Python code, no explanation, no markdown fences."""

CORRECTION_SYSTEM = """/no_think
You are an expert Python programmer fixing a bug. Return ONLY the corrected complete Python module, no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# Import test definitions from v4 (reuse all 10 tests)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from v4_runner import TESTS as _V4_TESTS, strip_markdown_fences

# Deep copy and fix Test 10 spec bug
TESTS = []
for t in _V4_TESTS:
    t_copy = dict(t)
    if t_copy["id"] == "10_ratelimiter":
        # Fix: add timestamp parameter to remaining() in the prompt
        t_copy["prompt"] = t_copy["prompt"].replace(
            "- `remaining(client_id: str) -> int` — how many requests remain in current window",
            "- `remaining(client_id: str, timestamp: float | None = None) -> int` — how many requests remain in current window. If timestamp is None, use time.time()"
        )
        # Fix: add timestamp to test call
        t_copy["test_code"] = t_copy["test_code"].replace(
            '    assert lim.remaining("u") == 3',
            '    assert lim.remaining("u", timestamp=2.0) == 3'
        )
    TESTS.append(t_copy)


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------

def query_model(prompt: str, system: str = SYSTEM_PROMPT,
                context: list | None = None, timeout: float = 300.0) -> tuple[str, float]:
    """Send prompt to vllm-mlx. Returns (response_text, elapsed_seconds)."""
    start = time.time()
    messages = [{"role": "system", "content": system}]
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": prompt})

    try:
        resp = httpx.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": messages,
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
        content = message.get("content") or ""
        return content, time.time() - start
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def run_pytest(code: str, module: str, test_code: str,
               timeout: int = 60) -> tuple[str, int, int]:
    """Run pytest on code. Returns (output, passed, failed)."""
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/{module}.py", "w") as f:
            f.write(code)
        with open(f"{td}/test_{module}.py", "w") as f:
            f.write(test_code)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest",
                 f"{td}/test_{module}.py", "-v", "--tb=short"],
                capture_output=True, text=True, cwd=td, timeout=timeout,
            )
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: Tests exceeded 60 seconds"
        except Exception as e:
            output = f"RUN ERROR: {e}"

    passed = output.count(" PASSED")
    failed = output.count(" FAILED") + output.count(" ERROR")
    return output, passed, failed


def check_import(code: str, module: str) -> str | None:
    """Layer 1: Try to import the module. Returns error message or None."""
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
            return "Import timed out (possible infinite loop in module-level code)"
        except Exception as e:
            return str(e)
    return None


def extract_failures(pytest_output: str, limit: int = 5) -> str:
    """Extract the first N failure details from pytest output."""
    lines = pytest_output.split("\n")
    failures = []
    in_failure = False
    current = []

    for line in lines:
        if line.startswith("_") and "FAILED" not in line and "ERROR" not in line:
            # Separator line — start of a failure block
            if current:
                failures.append("\n".join(current))
                current = []
            in_failure = True
            current.append(line)
        elif in_failure:
            current.append(line)
            if line.startswith("E ") or "assert" in line.lower():
                pass  # keep collecting
            elif line.strip() == "" and len(current) > 3:
                failures.append("\n".join(current))
                current = []
                in_failure = False

    if current:
        failures.append("\n".join(current))

    # Fallback: just grab FAILED lines with context
    if not failures:
        for i, line in enumerate(lines):
            if "FAILED" in line or ("AssertionError" in line) or ("Error" in line and "E " in line):
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                failures.append("\n".join(lines[start:end]))

    return "\n\n".join(failures[:limit])


# ---------------------------------------------------------------------------
# Layer 0: Output length guard — detects repetition degeneration
# ---------------------------------------------------------------------------

MAX_REASONABLE_LENGTH = 25000  # chars — longest clean output was ~14K
REPETITION_THRESHOLD = 5      # same line repeated this many times = degenerate

def check_repetition_degeneration(code: str) -> str | None:
    """Layer 0: Detect repetition degeneration in generated code.

    Returns a description of the problem if degenerate, None if OK.
    Known failure mode: 4-bit models can enter infinite loops generating
    the same attribute initialization lines until max_tokens truncates.
    """
    # Check 1: Absolute length — clean sovereign code is rarely >15K chars
    if len(code) > MAX_REASONABLE_LENGTH:
        # Check if it's genuinely long vs degenerate
        lines = code.split("\n")
        line_counts = Counter(line.strip() for line in lines if line.strip())
        most_common_line, most_common_count = line_counts.most_common(1)[0]

        if most_common_count >= REPETITION_THRESHOLD:
            return (f"Repetition degeneration detected: "
                    f"{len(code)} chars, line '{most_common_line[:60]}...' "
                    f"repeated {most_common_count} times")

    # Check 2: Even under the length limit, check for extreme repetition
    lines = code.split("\n")
    if len(lines) > 50:
        line_counts = Counter(line.strip() for line in lines if line.strip())
        most_common_line, most_common_count = line_counts.most_common(1)[0]
        # If >20% of non-empty lines are the same line, that's suspicious
        non_empty = sum(1 for l in lines if l.strip())
        if most_common_count > max(REPETITION_THRESHOLD, non_empty * 0.2):
            return (f"Repetition pattern detected: "
                    f"'{most_common_line[:60]}...' appears {most_common_count}/{non_empty} times")

    # Check 3: Truncation — if code ends mid-token (no closing newline or ends mid-word)
    if code and not code.rstrip().endswith((")", ":", "'", '"', "]", "}", "#")):
        last_line = code.rstrip().split("\n")[-1].strip()
        if last_line and not last_line.startswith("#") and len(last_line) < 3:
            return f"Possible truncation: code ends with '{last_line}'"

    return None


# ---------------------------------------------------------------------------
# Layer 1.5: Name existence check — verifies expected names after import
# ---------------------------------------------------------------------------

def extract_expected_names(test_code: str, module: str) -> list[str]:
    """Parse test code to find what names are imported from the module."""
    names = []

    # Match: from module import Name1, Name2, ...
    pattern = rf"from\s+{re.escape(module)}\s+import\s+(.+)"
    for match in re.finditer(pattern, test_code):
        imports = match.group(1)
        # Handle multi-line imports with backslash or parentheses
        for name in re.split(r"[,\s]+", imports):
            name = name.strip().rstrip("\\()")
            if name and name.isidentifier() and name != "as":
                names.append(name)

    # Match: import module (then module.Name usage)
    if not names:
        pattern = rf"{re.escape(module)}\.(\w+)"
        for match in re.finditer(pattern, test_code):
            name = match.group(1)
            if name.isidentifier() and name not in names:
                names.append(name)

    return names


def check_names_exist(code: str, module: str, test_code: str) -> list[str]:
    """Layer 1.5: Check that expected names exist in the module after import.

    Returns list of missing names, or empty list if all present.
    """
    expected = extract_expected_names(test_code, module)
    if not expected:
        return []

    missing = []
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/{module}.py", "w") as f:
            f.write(code)

        # Check each expected name
        check_code = f"import {module}\n"
        for name in expected:
            check_code += f"assert hasattr({module}, '{name}'), 'Missing: {name}'\n"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True, text=True, cwd=td, timeout=10,
            )
            if proc.returncode != 0:
                # Parse which names are missing
                for line in proc.stderr.split("\n"):
                    if "Missing:" in line:
                        name = line.split("Missing:")[-1].strip().rstrip("'\"")
                        if name:
                            missing.append(name)
                # If we couldn't parse, just report the error
                if not missing and proc.stderr.strip():
                    # Try to find the first assertion failure
                    for name in expected:
                        check_single = f"import {module}\nassert hasattr({module}, '{name}')"
                        p = subprocess.run(
                            [sys.executable, "-c", check_single],
                            capture_output=True, text=True, cwd=td, timeout=5,
                        )
                        if p.returncode != 0:
                            missing.append(name)
        except Exception:
            pass

    return missing


# ---------------------------------------------------------------------------
# Main test runner with 5 layers
# ---------------------------------------------------------------------------

def run_test(test: dict) -> dict:
    """Run a single test with all 3 layers of defense."""
    test_id = test["id"]
    module = test["module"]
    test_code = test["test_code"]

    print(f"\n{'='*60}")
    print(f"TEST {test_id}: {test['name']}")
    print(f"{'='*60}")

    # ---- First attempt: generate code ----
    print(f"  [Gen] Querying model...")
    raw_response, gen_time = query_model(test["prompt"])

    if raw_response.startswith("ERROR:"):
        print(f"  MODEL ERROR: {raw_response}")
        return save_result(test, gen_time, "", raw_response, 0, 0, 0, 0)

    code = strip_markdown_fences(raw_response)
    print(f"  [Gen] {gen_time:.1f}s, {len(code)} chars")

    correction_time = 0.0
    layer0_triggered = False
    layer1_triggered = False
    layer15_triggered = False
    layer2_triggered = False

    # ---- Layer 0: Output length guard ----
    degen = check_repetition_degeneration(code)
    if degen:
        layer0_triggered = True
        print(f"  [L0] {degen}")
        print(f"  [L0] Regenerating with anti-repetition prompt...")

        regen_prompt = f"""{test['prompt']}

IMPORTANT: Your previous attempt produced degenerate repetitive output. Keep your code concise and avoid repeating similar lines. If you need to initialize many attributes, use a loop or dict."""

        regen_raw, regen_time = query_model(regen_prompt)
        correction_time += regen_time

        if not regen_raw.startswith("ERROR:"):
            regen_code = strip_markdown_fences(regen_raw)
            regen_degen = check_repetition_degeneration(regen_code)
            if regen_degen is None:
                print(f"  [L0] Regeneration OK ({regen_time:.1f}s, {len(regen_code)} chars)")
                code = regen_code
                raw_response = regen_raw
            else:
                print(f"  [L0] Regeneration still degenerate: {regen_degen[:80]}")
        else:
            print(f"  [L0] Regeneration failed: {regen_raw[:80]}")

    # ---- Layer 1: Import check ----
    import_err = check_import(code, module)

    if import_err:
        layer1_triggered = True
        print(f"  [L1] Import failed: {import_err[:120]}...")
        print(f"  [L1] Requesting correction...")

        correction_prompt = f"""Your code has a bug that prevents it from importing. Here is the error:

{import_err}

Here is the original task:
{test['prompt']}

Fix the code and return ONLY the corrected complete Python module, no explanation."""

        corrected_raw, correction_time = query_model(
            correction_prompt,
            system=CORRECTION_SYSTEM,
            context=[
                {"role": "assistant", "content": raw_response},
            ],
        )
        corrected_code = strip_markdown_fences(corrected_raw)

        if corrected_code and not corrected_code.startswith("ERROR:"):
            import_err2 = check_import(corrected_code, module)
            if import_err2 is None:
                print(f"  [L1] Correction fixed the import ({correction_time:.1f}s)")
                code = corrected_code
            else:
                print(f"  [L1] Correction did NOT fix import: {import_err2[:80]}...")
        else:
            print(f"  [L1] Correction attempt failed")

    # ---- Layer 1.5: Name existence check ----
    missing_names = check_names_exist(code, module, test_code)
    if missing_names:
        layer15_triggered = True
        print(f"  [L1.5] Missing names: {', '.join(missing_names)}")
        print(f"  [L1.5] Requesting correction...")

        name_fix_prompt = f"""Your code imports and runs, but is missing these expected names: {', '.join(missing_names)}

The test file expects to import these from your module. Check for typos, wrong class names, or missing definitions.

Here is the original task:
{test['prompt']}

Fix the code and return ONLY the corrected complete Python module, no explanation."""

        name_fix_raw, l15_time = query_model(
            name_fix_prompt,
            system=CORRECTION_SYSTEM,
            context=[{"role": "assistant", "content": code}],
        )
        correction_time += l15_time
        name_fix_code = strip_markdown_fences(name_fix_raw)

        if name_fix_code and not name_fix_code.startswith("ERROR:"):
            # Verify the fix actually added the missing names
            still_missing = check_names_exist(name_fix_code, module, test_code)
            import_err2 = check_import(name_fix_code, module)
            if len(still_missing) < len(missing_names) and import_err2 is None:
                print(f"  [L1.5] Fixed ({l15_time:.1f}s): {len(missing_names) - len(still_missing)}/{len(missing_names)} names resolved")
                code = name_fix_code
            else:
                print(f"  [L1.5] Fix did not help, keeping original")
        else:
            print(f"  [L1.5] Correction attempt failed")

    # ---- Run FTA (First-Try Accuracy) tests ----
    print(f"  [FTA] Running tests...")
    fta_output, fta_passed, fta_failed = run_pytest(code, module, test_code)
    fta_total = fta_passed + fta_failed
    fta_rate = fta_passed / fta_total if fta_total > 0 else 0
    print(f"  [FTA] {fta_passed}/{fta_total} passed ({fta_rate*100:.0f}%)")

    # Track the FTA code and scores
    aca_passed = fta_passed
    aca_failed = fta_failed
    aca_total = fta_total
    aca_output = fta_output
    final_code = code

    # ---- Layer 2: Pytest feedback (only if tests failed) ----
    if fta_failed > 0 and fta_total > 0:
        layer2_triggered = True
        failures = extract_failures(fta_output)
        if not failures:
            # Fallback: get short summary
            for line in fta_output.split("\n"):
                if "FAILED" in line or "ERROR" in line:
                    failures += line.strip() + "\n"

        print(f"  [L2] {fta_failed} failures detected, requesting correction...")

        feedback_prompt = f"""Your code compiles and imports correctly but fails some tests. Here are the failures:

{failures}

Here is the original task:
{test['prompt']}

Fix ALL the bugs and return ONLY the corrected complete Python module, no explanation."""

        corrected_raw2, l2_time = query_model(
            feedback_prompt,
            system=CORRECTION_SYSTEM,
            context=[
                {"role": "assistant", "content": code},
            ],
        )
        correction_time += l2_time
        corrected_code2 = strip_markdown_fences(corrected_raw2)

        if corrected_code2 and not corrected_code2.startswith("ERROR:"):
            print(f"  [L2] Running tests on corrected code ({l2_time:.1f}s)...")
            aca_output, aca_passed, aca_failed = run_pytest(
                corrected_code2, module, test_code
            )
            aca_total = aca_passed + aca_failed

            if aca_passed >= fta_passed:
                print(f"  [L2] Corrected: {aca_passed}/{aca_total} passed ({aca_passed*100//aca_total if aca_total else 0}%)")
                final_code = corrected_code2
            else:
                print(f"  [L2] Correction was WORSE ({aca_passed}/{aca_total}), keeping original")
                aca_passed = fta_passed
                aca_failed = fta_failed
                aca_total = fta_total
                aca_output = fta_output
                final_code = code
        else:
            print(f"  [L2] Correction attempt failed")

    # ---- Save everything ----
    return save_result(
        test, gen_time + correction_time, final_code, aca_output,
        fta_passed, fta_total, aca_passed, aca_total,
        layer0_triggered=layer0_triggered,
        layer1_triggered=layer1_triggered,
        layer15_triggered=layer15_triggered,
        layer2_triggered=layer2_triggered,
        raw_response=raw_response,
    )


def save_result(test, elapsed, code, output, fta_passed, fta_total,
                aca_passed, aca_total, layer0_triggered=False,
                layer1_triggered=False, layer15_triggered=False,
                layer2_triggered=False, raw_response=""):
    """Save all result files and return result dict."""
    test_id = test["id"]

    (RESULTS_DIR / f"{test_id}_code.py").write_text(code)
    (RESULTS_DIR / f"{test_id}_pytest.txt").write_text(output[-5000:])
    if raw_response:
        (RESULTS_DIR / f"{test_id}_raw.txt").write_text(raw_response)

    result = {
        "id": test_id,
        "name": test["name"],
        "elapsed": elapsed,
        "code_length": len(code),
        "fta_passed": fta_passed,
        "fta_total": fta_total,
        "fta_rate": fta_passed / fta_total if fta_total > 0 else 0,
        "aca_passed": aca_passed,
        "aca_total": aca_total,
        "aca_rate": aca_passed / aca_total if aca_total > 0 else 0,
        "layer0_triggered": layer0_triggered,
        "layer1_triggered": layer1_triggered,
        "layer15_triggered": layer15_triggered,
        "layer2_triggered": layer2_triggered,
        "output": output[-3000:] if output else "",
    }

    with open(RESULTS_DIR / f"{test_id}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global RESULTS_DIR

    # Parse args
    tag = ""
    for i, arg in enumerate(sys.argv):
        if arg == "--tag" and i + 1 < len(sys.argv):
            tag = sys.argv[i + 1]

    if tag:
        RESULTS_DIR = Path(__file__).parent / f"v5-results-{tag}"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  PURPLE PROVING GROUND — V5 SOVEREIGN BATTERY")
    print(f"  Model: {MODEL}")
    print(f"  API: {API_URL}")
    print(f"  Tests: {len(TESTS)}")
    print(f"  Layers: L0 (length guard) + L1 (import) + L1.5 (names) + L2 (pytest) + L3 (prompt)")
    print(f"  Tag: {tag or 'default'}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # Check API
    try:
        resp = httpx.get("http://localhost:8000/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in (data.get("data") or [])]
        print(f"  Backend: OK ({', '.join(models)})")
    except Exception as e:
        print(f"  Backend ERROR: {e}")
        sys.exit(1)

    results = []
    for test in TESTS:
        result = run_test(test)
        results.append(result)

    # ---- Summary ----
    print(f"\n\n{'#'*60}")
    print(f"  V5 SOVEREIGN BATTERY — RESULTS")
    print(f"  Model: {MODEL}")
    print(f"{'#'*60}")
    print(f"{'Test':<45} {'FTA':>8} {'ACA':>8} {'L0':>3} {'L1':>3} {'1.5':>3} {'L2':>3} {'Time':>7}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*7}")

    fta_sum = 0
    aca_sum = 0
    total_tests_n = 0
    total_time = 0
    l0_count = 0
    l1_count = 0
    l15_count = 0
    l2_count = 0

    for r in results:
        ft = r.get("fta_total", 0)
        fp = r.get("fta_passed", 0)
        at = r.get("aca_total", 0)
        ap = r.get("aca_passed", 0)
        fta_str = f"{fp}/{ft}" if ft > 0 else "ERR"
        aca_str = f"{ap}/{at}" if at > 0 else "ERR"
        l0 = "Y" if r.get("layer0_triggered") else ""
        l1 = "Y" if r.get("layer1_triggered") else ""
        l15 = "Y" if r.get("layer15_triggered") else ""
        l2 = "Y" if r.get("layer2_triggered") else ""
        if r.get("layer0_triggered"):
            l0_count += 1
        if r.get("layer1_triggered"):
            l1_count += 1
        if r.get("layer15_triggered"):
            l15_count += 1
        if r.get("layer2_triggered"):
            l2_count += 1
        print(f"{r['name']:<45} {fta_str:>8} {aca_str:>8} {l0:>3} {l1:>3} {l15:>3} {l2:>3} {r['elapsed']:>6.1f}s")
        fta_sum += fp
        aca_sum += ap
        total_tests_n += ft
        total_time += r["elapsed"]

    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*7}")
    fta_pct = f"{fta_sum*100//total_tests_n}%" if total_tests_n > 0 else "N/A"
    aca_pct = f"{aca_sum*100//total_tests_n}%" if total_tests_n > 0 else "N/A"
    print(f"{'TOTAL':<45} {fta_pct:>8} {aca_pct:>8} {l0_count:>3} {l1_count:>3} {l15_count:>3} {l2_count:>3} {total_time:>6.1f}s")

    # Compute grade-style scores
    print(f"\n  FTA Overall: {fta_sum}/{total_tests_n} ({fta_pct})")
    print(f"  ACA Overall: {aca_sum}/{total_tests_n} ({aca_pct})")
    print(f"  Layer 0 triggered: {l0_count}/{len(TESTS)} tests")
    print(f"  Layer 1 triggered: {l1_count}/{len(TESTS)} tests")
    print(f"  Layer 1.5 triggered: {l15_count}/{len(TESTS)} tests")
    print(f"  Layer 2 triggered: {l2_count}/{len(TESTS)} tests")
    print(f"  Total time: {total_time:.1f}s")

    # Save summary
    summary = {
        "model": MODEL,
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": SYSTEM_PROMPT[:200] + "...",
        "layers": ["L0_length_guard", "L1_import_check", "L1.5_name_check", "L2_pytest_feedback", "L3_enhanced_prompt"],
        "fta_total_passed": fta_sum,
        "aca_total_passed": aca_sum,
        "total_tests": total_tests_n,
        "fta_rate": fta_sum / total_tests_n if total_tests_n > 0 else 0,
        "aca_rate": aca_sum / total_tests_n if total_tests_n > 0 else 0,
        "total_time": total_time,
        "l0_triggered": l0_count,
        "l1_triggered": l1_count,
        "l15_triggered": l15_count,
        "l2_triggered": l2_count,
        "results": results,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
