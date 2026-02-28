#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Vega Training Stack — install on purpleroom (RTX 5090, CUDA 12.8)
# Run: bash setup_training.sh
# ─────────────────────────────────────────────────────────────────
set -e

VENV="$HOME/.purple/train-env"
echo "==> Creating training venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo ""
echo "==> Installing PyTorch 2.6 + CUDA 12.8 (Blackwell/sm_100 support)"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "==> Installing HuggingFace stack"
pip install \
    transformers==4.51.0 \
    datasets==4.0.0 \
    accelerate==1.5.0 \
    peft==0.15.0 \
    bitsandbytes==0.45.3 \
    trl==0.17.0 \
    sentencepiece \
    protobuf \
    einops

echo ""
echo "==> Attempting Unsloth install (fastest LoRA; may not support DeltaNet yet)"
pip install "unsloth[cu128-torch260] @ git+https://github.com/unslothai/unsloth.git" \
    || echo "  [WARN] Unsloth install failed — training will use vanilla PEFT (slower but correct)"

echo ""
echo "==> Verifying install"
python3 - <<'EOF'
import torch
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.version.cuda}")
print(f"Device:   {torch.cuda.get_device_name(0)}")
print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import transformers, peft, bitsandbytes
print(f"transformers: {transformers.__version__}")
print(f"peft:         {peft.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")

try:
    import unsloth
    print(f"unsloth:  {unsloth.__version__} ✓")
except ImportError:
    print("unsloth:  not available (using vanilla PEFT)")
EOF

echo ""
echo "==> Done. Activate with: source $VENV/bin/activate"
