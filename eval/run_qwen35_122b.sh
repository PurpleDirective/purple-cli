#!/usr/bin/env bash
# Run V6 battery + frontier challenges for Qwen3.5-122B-A10B via Ollama
# Model: qwen3.5:122b-a10b (81GB, 10B active params, thinking enabled)
# Backend: http://localhost:11434/v1 (Ollama OpenAI-compat)
# Started: $(date)

set -e
EVAL_DIR="$(dirname "$0")"
RESULTS_DIR="$EVAL_DIR/v6-results-qwen35-122b-v1"
FRONTIER_DIR="$EVAL_DIR/frontier"
LOG="$EVAL_DIR/qwen35-122b-run.log"

echo "========================================"
echo "  Qwen3.5-122B-A10B Benchmark Run"
echo "  $(date)"
echo "========================================"
echo ""

# Verify model is available
echo "[check] Verifying Ollama model..."
ollama list | grep qwen3.5:122b-a10b || { echo "ERROR: qwen3.5:122b-a10b not found in Ollama"; exit 1; }

# Quick smoke test
echo "[check] Smoke test..."
SMOKE=$(curl -s --max-time 60 http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:122b-a10b","messages":[{"role":"user","content":"Reply READY"}],"max_tokens":500,"stream":false}')
echo "$SMOKE" | python3 -c "
import sys,json
r=json.load(sys.stdin)
c=r['choices'][0]['message'].get('content','').strip()
print(f'  Smoke test: {repr(c[:50])}')
" 2>/dev/null || { echo "ERROR: Smoke test failed"; exit 1; }
echo ""

# Phase 1: V6 Battery
echo "========================================"
echo "  Phase 1: V6 Battery (215 assertions)"
echo "  Tag: qwen35-122b-v1"
echo "  Estimated time: ~35-45 min"
echo "========================================"
export VLLM_URL="http://localhost:11434/v1/chat/completions"
export VLLM_MODEL="qwen3.5:122b-a10b"
export LOCAL_TIMEOUT="600"

cd "$EVAL_DIR"
python3 v6_runner.py --backend local --tag qwen35-122b-v1 2>&1 | tee -a "$LOG"

echo ""
echo "========================================"
echo "  Phase 2: Frontier Challenges"
echo "========================================"

# CTL Model Checker
echo ""
echo "[frontier] CTL Model Checker (43 tests)..."
cd "$FRONTIER_DIR"
python3 run_local_test.py 2>&1 | tee -a "$LOG"

# Cascade Supply Networks
echo ""
echo "[frontier] Cascade Gated Supply Networks (38 tests)..."
python3 run_cascade_test.py 2>&1 | tee -a "$LOG"

echo ""
echo "========================================"
echo "  All phases complete: $(date)"
echo "  Log: $LOG"
echo "========================================"
