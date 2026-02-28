#!/usr/bin/env python3
"""
Vega v1 inference server — OpenAI-compatible /v1/chat/completions
Loads Qwen3.5-27B + vega-v1 adapter in 4-bit, serves on port 8001.

Usage:
    source ~/.purple/train-env/bin/activate
    python3 vega_serve.py

Then from purplemac:
    VLLM_URL=http://100.89.41.72:8001/v1/chat/completions \
    VLLM_MODEL=vega-v1 \
    python3 ~/.purple/eval/v6_runner.py --backend local --tag vega-v1
"""
import json
import time
import torch
from threading import Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen3.5-27B"
ADAPTER_PATH = "/home/purple/.purple/adapters/vega-v2/final"
PORT         = 8001

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
print("Loading base model (4-bit)…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
print("Applying vega-v1 adapter…")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
_lock = Lock()
print(f"Model ready. Serving on port {PORT}.")

def chat(messages, max_tokens=8192, temperature=0.0):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            do_sample=(temperature > 0.01),
            repetition_penalty=1.05,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "vega-v1", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        messages  = req.get("messages", [])
        max_tok   = req.get("max_tokens", 8192)
        temp      = req.get("temperature", 0.0)

        with _lock:
            t0 = time.time()
            content = chat(messages, max_tok, temp)
            elapsed = time.time() - t0

        resp = {
            "id": f"chatcmpl-vega",
            "object": "chat.completion",
            "model": "vega-v1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"  → {len(content)} chars in {elapsed:.1f}s")

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Listening on 0.0.0.0:{PORT}")
    server.serve_forever()
