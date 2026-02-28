#!/usr/bin/env python3
"""
Vega v1 Training Script
QLoRA fine-tuning of Qwen3.5-27B with DeltaNet-aware LoRA target modules.

Usage:
    source ~/.purple/train-env/bin/activate
    python3 vega_train.py [--config train_config.yaml] [--dry-run]

Outputs:
    LoRA adapter saved to output_dir from config
    Training logs to output_dir/logs/
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(Path(__file__).parent / "train_config.yaml"))
parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
args = parser.parse_args()

# ── Load config ──────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

log.info(f"Config loaded: {args.config}")
log.info(f"Model: {cfg['model_id']}")
log.info(f"Output: {cfg['output_dir']}")

# ── Verify CUDA ───────────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    log.error("CUDA not available. This script requires a GPU.")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
log.info(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")

# ── Try Unsloth first (2x faster), fall back to vanilla PEFT ─────
# Set FORCE_VANILLA_PEFT=1 to skip Unsloth (needed for DeltaNet hybrid models
# like Qwen3.5-27B where Unsloth's Triton kernel JIT compilation hangs indefinitely).
USE_UNSLOTH = False
if not os.environ.get("FORCE_VANILLA_PEFT"):
    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
        log.info("Unsloth available — using optimized LoRA kernels")
    except ImportError:
        log.info("Unsloth not available — using vanilla PEFT (correct, ~40% slower)")
else:
    log.info("FORCE_VANILLA_PEFT=1 — using vanilla PEFT (Unsloth skipped)")

# ── Load and prepare data ─────────────────────────────────────────
log.info(f"Loading data from {cfg['train_data']}")

with open(cfg["train_data"]) as f:
    all_pairs = [json.loads(line) for line in f if line.strip()]

random.seed(cfg.get("shuffle_seed", 42))
random.shuffle(all_pairs)

val_n = max(1, int(len(all_pairs) * cfg.get("val_split", 0.05)))
val_pairs = all_pairs[:val_n]
train_pairs = all_pairs[val_n:]

log.info(f"Dataset: {len(train_pairs)} train / {len(val_pairs)} val")

if args.dry_run:
    log.info("Dry run — validating data format")
    for i, p in enumerate(train_pairs[:5]):
        msgs = p.get("messages", [])
        assert len(msgs) == 3, f"Pair {i}: expected 3 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
    log.info(f"✓ Data format valid. Sample pair roles: {[m['role'] for m in train_pairs[0]['messages']]}")
    log.info("✓ Dry run complete — no training performed")
    sys.exit(0)

# ── Load model and tokenizer ──────────────────────────────────────
from transformers import AutoTokenizer, BitsAndBytesConfig

log.info(f"Loading tokenizer: {cfg['model_id']}")
tokenizer = AutoTokenizer.from_pretrained(
    cfg["model_id"],
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg.get("load_in_4bit", True),
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
    bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
)

if USE_UNSLOTH:
    model, _unsloth_processor = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=cfg.get("max_seq_length", 2048),
        dtype=torch.bfloat16,
        load_in_4bit=cfg.get("load_in_4bit", True),
    )
    # Qwen3.5-27B is a VL model — Unsloth returns a multimodal processor.
    # Extract the text tokenizer for text-only SFT training.
    if hasattr(_unsloth_processor, 'tokenizer'):
        tokenizer = _unsloth_processor.tokenizer
        log.info(f"Extracted text tokenizer from VL processor: {type(tokenizer).__name__}")
    else:
        tokenizer = _unsloth_processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["target_modules"],
        bias=cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
    )
else:
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    log.info(f"Loading model: {cfg['model_id']} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",    # flash_attn_2 may not support DeltaNet
    )

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["target_modules"],
        bias=cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ── Tokenize dataset ──────────────────────────────────────────────
MAX_LEN = cfg.get("max_seq_length", 2048)

def format_and_tokenize(pair):
    """
    Apply chat template and tokenize.
    Loss is masked on system+user tokens — only train on assistant output.
    """
    messages = pair["messages"]

    # Build input_ids using the model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors=None,
        padding=False,
    )
    input_ids = tokens["input_ids"]

    # Build labels — mask everything up to and including the last user turn
    # Find the assistant response start by re-encoding just the prompt
    prompt_messages = messages[:-1]   # system + user
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors=None, padding=False)
    prompt_len = len(prompt_tokens["input_ids"])

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:MAX_LEN]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }

log.info("Tokenizing training data...")
from datasets import Dataset

def pairs_to_dataset(pairs):
    records = []
    for p in pairs:
        try:
            record = format_and_tokenize(p)
            if len(record["input_ids"]) > 10:  # skip degenerate pairs
                records.append(record)
        except Exception as e:
            log.debug(f"Skipped pair: {e}")
    return Dataset.from_list(records)

train_dataset = pairs_to_dataset(train_pairs)
val_dataset = pairs_to_dataset(val_pairs)
log.info(f"Tokenized: {len(train_dataset)} train / {len(val_dataset)} val")

# ── Data collator (pads batches dynamically) ──────────────────────
from transformers import DataCollatorForSeq2Seq
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    # model= intentionally omitted — passing a CUDA model to multi-worker DataLoader
    # causes futex deadlock on Linux (fork inherits CUDA state, workers hang).
    # label_pad_token_id=-100 makes model reference unnecessary.
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# ── Training arguments ────────────────────────────────────────────
from transformers import TrainingArguments, Trainer

output_dir = cfg["output_dir"]
Path(output_dir).mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=cfg.get("num_train_epochs", 2),
    per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
    per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
    gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
    learning_rate=cfg.get("learning_rate", 2e-4),
    lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
    warmup_ratio=cfg.get("warmup_ratio", 0.05),
    weight_decay=cfg.get("weight_decay", 0.01),
    max_grad_norm=cfg.get("max_grad_norm", 1.0),
    optim=cfg.get("optim", "adamw_8bit"),
    bf16=cfg.get("bf16", True),
    tf32=cfg.get("tf32", True),
    logging_dir=f"{output_dir}/logs",
    logging_steps=cfg.get("logging_steps", 10),
    eval_strategy="steps",
    eval_steps=cfg.get("eval_steps", 100),
    save_strategy="steps",
    save_steps=cfg.get("save_steps", 200),
    save_total_limit=cfg.get("save_total_limit", 3),
    load_best_model_at_end=cfg.get("load_best_model_at_end", True),
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
    seed=cfg.get("seed", 42),
    report_to="none",           # disable wandb/tensorboard unless you want it
    resume_from_checkpoint=args.resume,
)

# ── Train ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    processing_class=tokenizer,
)

log.info("="*60)
log.info("Starting Vega v1 training run")
log.info(f"  Pairs:    {len(train_dataset)} train / {len(val_dataset)} val")
log.info(f"  Epochs:   {cfg.get('num_train_epochs', 2)}")
log.info(f"  Batch:    {cfg.get('per_device_train_batch_size', 2)} × {cfg.get('gradient_accumulation_steps', 8)} grad_accum = {cfg.get('per_device_train_batch_size', 2) * cfg.get('gradient_accumulation_steps', 8)} eff. batch")
log.info(f"  LR:       {cfg.get('learning_rate', 2e-4)}")
log.info(f"  Adapter:  {cfg['adapter_name']}")
log.info("="*60)

trainer.train(resume_from_checkpoint=args.resume)

# ── Save final adapter ────────────────────────────────────────────
final_path = f"{output_dir}/final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
log.info(f"✓ Adapter saved to {final_path}")

# Save config alongside adapter for reproducibility
import shutil
shutil.copy(args.config, f"{output_dir}/train_config.yaml")

log.info("Training complete.")
log.info(f"To load the adapter: peft.PeftModel.from_pretrained(base_model, '{final_path}')")
