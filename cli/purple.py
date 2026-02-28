#!/usr/bin/env python3
"""
Purple CLI -- Local AI Inference + MCP Orchestrator

Supports two inference backends:
  - Ollama (/api/chat)     -- for GGUF models via llama.cpp
  - vllm-mlx (/v1/chat)   -- for MLX models via Apple Metal (2x faster on MoE)

Auto-detects the best backend at startup. Override with PURPLE_BACKEND env var.
Executes MCP tool calls directly via FastMCP STDIO clients.

Usage:
    Interactive:  python purple.py
    One-shot:     python purple.py "your prompt here"

Environment:
    PURPLE_BACKEND  auto|ollama|vllm-mlx  (default: auto)
    OLLAMA_URL      http://localhost:11434
    VLLM_URL        http://localhost:8000
    OLLAMA_MODEL    purple:latest (auto-detected from vllm-mlx if available)
"""

import asyncio
import json
import re
import readline
import shutil
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastmcp.client import Client
from fastmcp.client.transports import StdioTransport

# Proving Ground tracker
_cli_dir = str(Path(__file__).parent)
if _cli_dir not in sys.path:
    sys.path.insert(0, _cli_dir)
import tracker
import dashboard as proving_ground

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "purple:latest")
PURPLE_BACKEND = os.environ.get("PURPLE_BACKEND", "auto")  # "auto", "ollama", "vllm-mlx"
IDENTITY_PATH = Path.home() / ".purple" / "identity" / "identity.md"
MCP_CONFIG_PATH = Path.home() / ".purple" / "config" / "mcp.json"

MAX_TOOL_ROUNDS = 10
MAX_THINKING_CALLS = 3    # Cap sequentialthinking per chat turn
MAX_EMPTY_LOOKUPS = 2     # Stop searching empty knowledge base after N misses
MAX_HISTORY_TOKENS = 50000  # Leave ~15K for system prompt + tool defs
OLLAMA_TIMEOUT = 300.0  # seconds -- model can be slow on first load
VLLM_MAX_PLAN_TOKENS = 8192  # Higher token limit for planning (thinking uses tokens)

# ── Planning mode system prompt ───────────────────────────────────────────
PLANNING_PROMPT = """You are Purple in planning mode. Your job is to help your operator think through a problem BEFORE writing code.

## Planning Rules

- Ask clarifying questions before proposing solutions. Do not assume requirements.
- Think step by step. Break complex problems into smaller pieces.
- When you know something (algorithms, patterns, architecture), reason from it confidently.
- When you DON'T know something (platform specifics, API details, pricing, real-world facts), say so explicitly: "I'm not confident about X — verify this."
- Never invent platform names, tool names, product names, or specific dollar amounts. If you can't verify it from your training, flag it.
- Propose 2-3 approaches when multiple valid options exist. Explain tradeoffs.
- End each response with a clear next question or a summary of the agreed plan.
- No tool calls in planning mode. This is pure reasoning and discussion.
- Keep your operator's constraints in mind: local-first, privacy-respecting, sovereignty matters.

## Output Format

Structure your planning responses as:
1. **Understanding** — restate what you think the operator wants
2. **Questions** — what you need to know before proceeding
3. **Approach** — proposed solution(s) with tradeoffs
4. **Next step** — what to discuss or decide next

When the plan is solid, the operator will use /build to switch to execution mode."""

# ── Friendly model name mapping ──────────────────────────────────────────
# Maps substrings (lowercase) found in raw model IDs to (display_name, params)
_MODEL_FRIENDLY = {
    "qwen3-coder-next": ("Qwen 3 Coder", "80B"),
    "qwen3.5-27b":      ("Qwen 3.5", "27B"),
    "qwen3.5-122b":     ("Qwen 3.5", "122B"),
    "qwen3.5-35b":      ("Qwen 3.5", "35B"),
    "qwen3-coder":      ("Qwen 3 Coder", "30B"),
    "llama3.1:70b":     ("Llama 3.1", "70B"),
    "llama3.1:8b":      ("Llama 3.1", "8B"),
}


def friendly_model_name(raw_id: str) -> str:
    """Convert a raw model ID or HuggingFace cache path to a clean display name.

    Examples:
        '/Users/...models--Eldadalbajob--Huihui-Qwen3-Coder-Next-abliterated-mlx-4Bit/snapshots/...'
        → 'Qwen 3 Coder (80B) 4-bit'

        'lmstudio-community/Qwen3-Coder-Next-MLX-6bit'
        → 'Qwen 3 Coder (80B) 6-bit'
    """
    # Extract model slug from HuggingFace cache paths
    slug = raw_id
    match = re.search(r'models--[^/]+--([^/]+)', slug)
    if match:
        slug = match.group(1)
    elif "/" in slug:
        slug = slug.rsplit("/", 1)[-1]

    lower = slug.lower()

    # Detect quantization
    quant = ""
    for q in ("4bit", "4-bit", "q4_k_m", "6bit", "6-bit", "8bit", "8-bit"):
        if q in lower:
            bits = q.replace("bit", "").replace("-", "").replace("_k_m", "")
            quant = f" {bits}-bit"
            break

    # Match against known models (longest match first)
    for pattern, (name, params) in sorted(_MODEL_FRIENDLY.items(), key=lambda x: -len(x[0])):
        if pattern in lower:
            return f"{name} ({params}{quant})"

    # Fallback: strip common suffixes and return cleaned slug
    for suffix in ("-MLX-6bit", "-MLX-4bit", "-mlx-4Bit", "-GGUF",
                   "-Q4_K_M", "-abliterated", "-instruct"):
        slug = slug.replace(suffix, "")
    return slug
NUM_CTX = 65536  # Match Modelfile's 64K context
VLLM_MAX_TOKENS = 4096  # Max generation tokens for vllm-mlx

# Session logging
SESSIONS_DIR = Path.home() / ".purple" / "sessions"
HISTORY_FILE = Path.home() / ".purple" / "cli" / "history"

# ── ANSI Colors ────────────────────────────────────────────────────────────
# 256-color palette for richer Purple aesthetic
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_ITALIC = "\033[3m"
C_PURPLE = "\033[38;5;141m"      # Soft lavender purple
C_DEEP_PURPLE = "\033[38;5;99m"  # Deeper purple for accents
C_CYAN = "\033[38;5;117m"        # Soft cyan
C_GREEN = "\033[38;5;114m"       # Soft green
C_YELLOW = "\033[38;5;221m"      # Warm amber
C_RED = "\033[38;5;167m"         # Soft coral red
C_WHITE = "\033[38;5;252m"       # Soft white
C_MUTED = "\033[38;5;245m"       # Muted gray
C_BORDER = "\033[38;5;60m"       # Subtle purple-gray

# ── Markers ────────────────────────────────────────────────────────────────
DIAMOND = "◆"      # Brand mark (gemstone)
DIAMOND_SM = "◇"   # Outline diamond
ARROW_R = "▸"      # Prompt arrow
DOT = "●"          # Status dot


# ---------------------------------------------------------------------------
# Session transcript logger
# ---------------------------------------------------------------------------

class SessionLogger:
    """Logs session transcripts as JSONL for the overnight verification pipeline."""

    def __init__(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._file = SESSIONS_DIR / f"{self._session_id}.jsonl"
        self._count = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def log(self, role: str, content: str, **extra):
        """Append a message to the session transcript."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
            **extra,
        }
        with open(self._file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._count += 1

    def log_tool_call(self, name: str, args: dict, result: str, is_error: bool):
        """Log a tool call and its result."""
        self.log("tool_call", "", tool=name, arguments=args,
                 result=result[:2000], is_error=is_error)

    def log_timing(self, round_num: int, timing: dict):
        """Log per-turn performance metrics (tok/s, TTFT, gen time, tokens)."""
        self.log("timing", "", round=round_num,
                 tokens=timing.get("tokens", 0),
                 ttft=round(timing.get("ttft", 0), 4),
                 gen_seconds=round(timing.get("gen", 0), 3),
                 total_seconds=round(timing.get("total", 0), 3),
                 tok_per_sec=round(timing.get("tokens", 0) / timing["gen"], 1)
                 if timing.get("gen", 0) > 0 else 0)

    def log_session_meta(self, model: str, backend: str, tier: str,
                         tool_count: int, server_count: int):
        """Log session-level metadata at startup."""
        self.log("session_start", "", model=model,
                 friendly_name=friendly_model_name(model),
                 backend=backend, tier=tier,
                 tool_count=tool_count, server_count=server_count)

    def log_task_outcome(self, task_id: int, outcome: str, rounds: int,
                         total_timing: dict, user_rating: int | None = None):
        """Log task completion with outcome and aggregate timing for fine-tuning extraction."""
        self.log("task_outcome", "", task_id=task_id, outcome=outcome,
                 rounds=rounds, user_rating=user_rating,
                 total_tokens=total_timing.get("tokens", 0),
                 total_seconds=round(total_timing.get("total", 0), 2),
                 avg_tok_per_sec=round(total_timing.get("tok_per_sec", 0), 1))

    @property
    def message_count(self) -> int:
        return self._count

    @staticmethod
    def list_sessions(limit: int = 10) -> list[dict]:
        """List recent session files with metadata."""
        if not SESSIONS_DIR.exists():
            return []
        files = sorted(SESSIONS_DIR.glob("*.jsonl"), reverse=True)[:limit]
        sessions = []
        for f in files:
            lines = f.read_text().strip().split("\n")
            msg_count = len(lines)
            # Get first user message as preview
            preview = ""
            for line in lines:
                try:
                    entry = json.loads(line)
                    if entry.get("role") == "user":
                        preview = entry.get("content", "")[:60]
                        break
                except json.JSONDecodeError:
                    continue
            sessions.append({
                "id": f.stem,
                "messages": msg_count,
                "preview": preview,
                "size": f.stat().st_size,
            })
        return sessions


def _tool_args_summary(args: dict) -> str:
    """Extract a human-readable summary from tool arguments."""
    if not args:
        return ""
    # Look for the most descriptive string value
    for key in ("query", "text", "content", "prompt", "path", "name",
                "key", "topic", "thought", "message"):
        if key in args and isinstance(args[key], str):
            val = args[key]
            if len(val) > 60:
                val = val[:57] + "..."
            return f'"{val}"'
    # Fallback: first string value
    for val in args.values():
        if isinstance(val, str) and len(val) > 2:
            if len(val) > 60:
                val = val[:57] + "..."
            return f'"{val}"'
    return ""


# XML tool call pattern for Format 2 (Qwen3-Coder fallback)
_XML_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*\{(.*?)\}\s*</tool_call>',
    re.DOTALL,
)


def _parse_xml_tool_calls(content: str) -> list[dict] | None:
    """Attempt to extract tool calls from XML-formatted content.

    Qwen3-Coder sometimes emits tool calls as XML text in the content field
    instead of JSON in the tool_calls field (Goose bug #6883). This catches
    both known formats:
      <function=name><parameter=key>value</parameter></function>
      <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Returns a list of tool_call dicts in Ollama format, or None if no XML found.
    """
    calls = []

    # Format 1: <function=name><parameter=key>value</parameter></function>
    for match in re.finditer(
        r'<function=(\w+)>(.*?)</function>', content, re.DOTALL
    ):
        func_name = match.group(1)
        params_block = match.group(2)
        arguments = {}
        for param_match in re.finditer(
            r'<parameter=(\w+)>(.*?)</parameter>', params_block, re.DOTALL
        ):
            arguments[param_match.group(1)] = param_match.group(2).strip()
        calls.append({
            "function": {"name": func_name, "arguments": arguments}
        })

    # Format 2: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    if not calls:
        for match in _XML_TOOL_CALL_RE.finditer(content):
            try:
                data = json.loads("{" + match.group(1) + "}")
                calls.append({
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": data.get("arguments", {}),
                    }
                })
            except json.JSONDecodeError:
                continue

    return calls if calls else None


# ---------------------------------------------------------------------------
# Load identity system prompt
# ---------------------------------------------------------------------------

def load_identity() -> str:
    """Load the Purple identity file as a system prompt."""
    if not IDENTITY_PATH.exists():
        example = IDENTITY_PATH.parent / "identity.example.md"
        if example.exists():
            import shutil
            shutil.copy2(example, IDENTITY_PATH)
            print(f"  {C_GREEN}{DOT} Created identity from example: {IDENTITY_PATH}{C_RESET}")
    if IDENTITY_PATH.exists():
        return IDENTITY_PATH.read_text().strip()
    print(f"  {C_MUTED}{DIAMOND_SM} Identity file not found at {IDENTITY_PATH}{C_RESET}")
    return "You are Purple, a local AI assistant."


# ---------------------------------------------------------------------------
# MCP tool management
# ---------------------------------------------------------------------------

class MCPToolManager:
    """Manages connections to MCP servers and provides tool execution."""

    def __init__(self):
        self._clients: dict[str, Client] = {}
        self._tool_map: dict[str, tuple[str, Any]] = {}  # tool_name -> (server_name, tool_obj)
        self._ollama_tools: list[dict] = []  # Ollama-formatted tool list

    async def connect(self):
        """Connect to all configured MCP servers and discover tools.

        Loads server definitions from ~/.purple/config/mcp.json.
        Each server entry has a "command" array and an "enabled" flag.
        Uses StdioTransport for all servers (Python, npx, or any executable).
        """
        # Load server config (auto-create from example if missing)
        if not MCP_CONFIG_PATH.exists():
            example = MCP_CONFIG_PATH.parent / "mcp.example.json"
            if example.exists():
                import shutil
                shutil.copy2(example, MCP_CONFIG_PATH)
                print(f"  {C_GREEN}{DOT} Created MCP config from example: {MCP_CONFIG_PATH}{C_RESET}")
            else:
                print(f"  {C_RED}{DOT} MCP config not found: {MCP_CONFIG_PATH}{C_RESET}")
                return

        try:
            config = json.loads(MCP_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  {C_RED}{DOT} MCP config error: {e}{C_RESET}")
            return

        servers = config.get("servers", {})

        for server_name, server_cfg in servers.items():
            if not server_cfg.get("enabled", True):
                continue

            cmd = server_cfg.get("command", [])
            if not cmd:
                print(f"  {C_MUTED}{DIAMOND_SM} {server_name}: no command specified{C_RESET}")
                continue

            try:
                transport = StdioTransport(
                    command=cmd[0],
                    args=cmd[1:],
                    keep_alive=True,
                    log_file=Path.home() / ".purple" / "cli" / "mcp.log",
                )
                client = Client(transport=transport, timeout=30)
                await client.__aenter__()
                self._clients[server_name] = client

                # Discover tools
                tools = await client.list_tools()
                for tool in tools:
                    self._tool_map[tool.name] = (server_name, tool)
                    self._ollama_tools.append(self._mcp_to_ollama_tool(tool))


            except Exception as e:
                print(f"  {C_RED}{DOT} {server_name}: connection failed — {e}{C_RESET}")

    async def disconnect(self):
        """Disconnect from all MCP servers."""
        for name, client in self._clients.items():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()

    def _mcp_to_ollama_tool(self, tool: Any) -> dict:
        """Convert an MCP tool schema to Ollama's tool format.

        Ollama uses OpenAI-compatible tool format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }
        """
        # MCP tool.inputSchema is already a JSON Schema dict
        parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        }

    @property
    def ollama_tools(self) -> list[dict]:
        return self._ollama_tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_map.keys())

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool call via the appropriate MCP server.

        Returns the tool result as a string.
        """
        if name not in self._tool_map:
            return f"Error: tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        server_name, _tool_obj = self._tool_map[name]
        client = self._clients.get(server_name)
        if client is None:
            return f"Error: MCP server '{server_name}' is not connected"

        try:
            result = await client.call_tool(name, arguments, raise_on_error=False)

            # Extract text from content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "data"):
                    parts.append(f"[binary data: {len(block.data)} bytes]")
                else:
                    parts.append(str(block))

            output = "\n".join(parts)

            if result.is_error:
                return f"Tool error: {output}"
            return output

        except Exception as e:
            return f"Error calling tool '{name}': {e}"


# ---------------------------------------------------------------------------
# Ollama chat client
# ---------------------------------------------------------------------------

class OllamaChat:
    """Manages conversation with Ollama or vllm-mlx inference backends."""

    def __init__(self, tool_manager: MCPToolManager, session_logger: SessionLogger,
                 backend: str = "ollama"):
        self._tool_manager = tool_manager
        self._session_logger = session_logger
        self._messages: list[dict] = []
        self._system_prompt = load_identity()
        # Inject current date so the model uses the correct year in searches
        today = datetime.now().strftime("%Y-%m-%d")
        self._system_prompt += (
            f"\n\n## Current Date\n"
            f"Today is {today}. Always use the year {datetime.now().year} "
            f"in search queries, trend references, and date-sensitive claims."
        )
        self._backend = backend  # "ollama" or "vllm-mlx"
        base_url = VLLM_URL if backend == "vllm-mlx" else OLLAMA_URL
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0),
        )
        self._current_task_id: int | None = None
        self._plan_mode: bool = False
        # Per-turn tool budget counters (reset each chat call)
        self._thinking_calls: int = 0
        self._empty_lookups: int = 0
        # Per-turn timing
        self._round_timings: list[dict] = []
        # Streaming indent tracking
        self._col: int = 0

    def _print_fragment(self, fragment: str):
        """Print a streaming fragment with 2-space indent on new lines."""
        for char in fragment:
            if char == '\n':
                print('\n  ', end='', flush=True)
                self._col = 2
            else:
                print(char, end='', flush=True)
                self._col += 1

    def _turn_separator(self):
        """Print a subtle divider between conversation turns."""
        width = min(shutil.get_terminal_size().columns, 72)
        print(f"\n{C_BORDER}{'─' * width}{C_RESET}\n")

    async def close(self):
        await self._http.aclose()

    async def chat(self, user_input: str) -> str:
        """Send a user message and get a response, handling tool calls.

        Text is streamed to stdout as it arrives. Tool call rounds still use
        the streaming path for each model turn.

        Returns the final assistant text response (already printed to terminal).
        """
        self._messages.append({"role": "user", "content": user_input})
        self._session_logger.log("user", user_input)

        # Reset per-turn budget counters
        self._thinking_calls = 0
        self._empty_lookups = 0
        self._round_timings = []
        self._col = 2  # Start indented (after "  " prefix printed by caller)

        # Proving Ground: start tracking this task
        self._current_task_id = tracker.start_task(user_input, OLLAMA_MODEL)

        for round_num in range(MAX_TOOL_ROUNDS):
            # Continuation rounds get a fresh response prefix with budget indicator
            if round_num > 0:
                remaining = MAX_TOOL_ROUNDS - round_num
                budget_color = C_GREEN if remaining > 4 else (C_YELLOW if remaining > 2 else C_RED)
                print(f"\n{C_PURPLE}{DIAMOND}{C_RESET} {budget_color}{C_DIM}[{remaining} rounds left]{C_RESET}\n  ", end="", flush=True)
                self._col = 2

            try:
                infer_start = time.monotonic()
                response = await self._stream_to_ollama()
                infer_end = time.monotonic()
            except KeyboardInterrupt:
                print(f"\n{C_MUTED}  {DIAMOND_SM} stream interrupted{C_RESET}")
                agg = self._aggregate_timing()
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "interrupted", round_num)
                    self._session_logger.log_task_outcome(
                        self._current_task_id, "interrupted", round_num + 1, agg)
                return "[interrupted]"

            if response is None:
                agg = self._aggregate_timing()
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "failed", round_num)
                    self._session_logger.log_task_outcome(
                        self._current_task_id, "failed", round_num + 1, agg)
                backend_name = "vllm-mlx" if self._backend == "vllm-mlx" else "Ollama"
                return f"[Error: no response from {backend_name}]"

            # Capture timing metadata from the stream
            timing = response.pop("_timing", {})
            timing["total"] = infer_end - infer_start

            message = response.get("message", {})
            tool_calls = message.get("tool_calls")

            # Detect XML-formatted tool calls in content field
            # (Qwen3-Coder fallback when it can't emit proper JSON tool calls)
            if not tool_calls:
                content = message.get("content", "")
                xml_calls = _parse_xml_tool_calls(content)
                if xml_calls:
                    print(f"\n  {C_YELLOW}{DIAMOND_SM} XML tool call detected — intercepting{C_RESET}")
                    tool_calls = xml_calls

            if not tool_calls:
                # No tool calls -- we have a final response (already streamed)
                content = message.get("content", "")
                self._messages.append({"role": "assistant", "content": content})
                self._session_logger.log("assistant", content)
                self._session_logger.log_timing(round_num, timing)
                print()  # Newline after streamed output
                # Show timing summary
                self._round_timings.append(timing)
                self._print_timing_summary()
                self._turn_separator()
                # Log task outcome with aggregate timing
                agg = self._aggregate_timing()
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "completed", round_num)
                    self._session_logger.log_task_outcome(
                        self._current_task_id, "completed", round_num + 1, agg)
                return content

            # Tool calls detected -- newline after any streamed preamble text
            print()
            self._round_timings.append(timing)
            self._session_logger.log_timing(round_num, timing)
            tool_exec_start = time.monotonic()

            # Append the assistant message with tool_calls to history
            self._messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", {})
                # Some models return arguments as a JSON string instead of dict
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                if not isinstance(tool_args, dict):
                    tool_args = {}

                # ── Tool budget enforcement ───────────────────────────
                # Cap sequentialthinking to prevent infinite thinking loops
                if tool_name == "sequentialthinking":
                    self._thinking_calls += 1
                    if self._thinking_calls > MAX_THINKING_CALLS:
                        print(f"    {C_YELLOW}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_YELLOW}capped ({MAX_THINKING_CALLS} max) — produce your answer now{C_RESET}")
                        result = (
                            f"BUDGET: You have used {MAX_THINKING_CALLS} thinking rounds. "
                            f"Stop thinking and produce your final answer directly. "
                            f"You have {MAX_TOOL_ROUNDS - round_num - 1} tool rounds remaining."
                        )
                        self._messages.append({"role": "tool", "content": result})
                        continue

                # Cap repeated empty knowledge/search lookups
                if tool_name in ("lookup_knowledge", "search"):
                    if self._empty_lookups >= MAX_EMPTY_LOOKUPS:
                        print(f"    {C_YELLOW}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_YELLOW}skipped — knowledge base is sparse, proceed without{C_RESET}")
                        result = (
                            "BUDGET: The knowledge base has limited entries. "
                            "Stop searching and answer from your own knowledge. "
                            f"You have {MAX_TOOL_ROUNDS - round_num - 1} tool rounds remaining."
                        )
                        self._messages.append({"role": "tool", "content": result})
                        continue

                # Print tool call — human-readable
                summary = _tool_args_summary(tool_args)
                if summary:
                    print(f"    {C_CYAN}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_MUTED}{summary}{C_RESET}")
                else:
                    print(f"    {C_CYAN}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}")

                # Execute the tool
                result = await self._tool_manager.call_tool(tool_name, tool_args)

                # Track empty knowledge/search lookups
                if tool_name in ("lookup_knowledge", "search") and (
                    "No knowledge found" in result or "No results found" in result
                ):
                    self._empty_lookups += 1

                # Inject budget warning when rounds are running low
                remaining = MAX_TOOL_ROUNDS - round_num - 1
                if remaining <= 3 and remaining > 0:
                    result += (
                        f"\n\n[SYSTEM: {remaining} tool rounds remaining. "
                        f"Produce your final answer soon.]"
                    )

                # Proving Ground: track tool call success/failure
                is_error = result.startswith("Error:") or result.startswith("Tool error:")
                if self._current_task_id is not None:
                    tracker.record_tool_call(self._current_task_id, success=not is_error)
                self._session_logger.log_tool_call(tool_name, tool_args, result, is_error)

                # Print result — clean and compact
                if is_error:
                    # Strip prefixes for cleaner error display
                    err_msg = result.split(": ", 1)[-1] if ": " in result else result
                    if len(err_msg) > 80:
                        err_msg = err_msg[:77] + "..."
                    print(f"    {C_RED}{DOT} {err_msg}{C_RESET}")
                else:
                    preview = result.replace("\n", " ").strip()
                    if len(preview) > 80:
                        print(f"    {C_GREEN}{DOT}{C_RESET} {C_MUTED}done ({len(result)} chars){C_RESET}")
                    elif preview:
                        print(f"    {C_GREEN}{DOT}{C_RESET} {C_MUTED}{preview}{C_RESET}")
                    else:
                        print(f"    {C_GREEN}{DOT}{C_RESET} {C_MUTED}done{C_RESET}")

                # Add tool result to message history
                self._messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Track tool execution time for this round
            tool_exec_end = time.monotonic()
            if self._round_timings:
                self._round_timings[-1]["tool_exec"] = tool_exec_end - tool_exec_start

        # Proving Ground: mark task as failed (too many rounds)
        agg = self._aggregate_timing()
        if self._current_task_id is not None:
            tracker.complete_task(self._current_task_id, "failed", MAX_TOOL_ROUNDS)
            self._session_logger.log_task_outcome(
                self._current_task_id, "exhausted", MAX_TOOL_ROUNDS, agg)
        self._print_timing_summary()
        self._turn_separator()
        return "[Error: max tool call rounds exceeded]"

    def _aggregate_timing(self) -> dict:
        """Aggregate timing data across all rounds for logging."""
        total_tokens = sum(t.get("tokens", 0) for t in self._round_timings)
        total_gen = sum(t.get("gen", 0) for t in self._round_timings)
        total_time = sum(t.get("total", 0) for t in self._round_timings)
        return {
            "tokens": total_tokens,
            "gen": total_gen,
            "total": total_time,
            "tok_per_sec": total_tokens / total_gen if total_gen > 0 else 0,
        }

    def _print_timing_summary(self):
        """Print a compact timing summary after task completion."""
        if not self._round_timings:
            return
        total_infer = sum(t.get("total", 0) for t in self._round_timings)
        total_ttft = sum(t.get("ttft", 0) for t in self._round_timings)
        total_gen = sum(t.get("gen", 0) for t in self._round_timings)
        total_tool = sum(t.get("tool_exec", 0) for t in self._round_timings)
        total_tokens = sum(t.get("tokens", 0) for t in self._round_timings)
        rounds = len(self._round_timings)

        parts = []
        if total_tokens > 0 and total_gen > 0:
            tps = total_tokens / total_gen
            parts.append(f"{tps:.0f} tok/s")
        if total_ttft > 0:
            parts.append(f"ttft {total_ttft*1000:.0f}ms")
        if total_gen > 0:
            parts.append(f"gen {total_gen:.1f}s")
        if total_tool > 0:
            parts.append(f"tools {total_tool:.1f}s")
        if rounds > 1:
            parts.append(f"{rounds} rounds")
        parts.append(f"total {total_infer + total_tool:.1f}s")

        if parts:
            summary = " · ".join(parts)
            print(f"    {C_MUTED}{DIAMOND_SM} {summary}{C_RESET}")

    def _build_payload(self, *, stream: bool) -> dict[str, Any]:
        """Build the inference payload for the active backend."""
        messages = self._build_messages()
        enable_thinking = self._plan_mode  # Think in plan mode, not in chat mode

        if self._backend == "vllm-mlx":
            payload: dict[str, Any] = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": stream,
                "max_tokens": VLLM_MAX_PLAN_TOKENS if self._plan_mode else VLLM_MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        else:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": stream,
                "think": enable_thinking,
                "options": {
                    "num_ctx": NUM_CTX,
                },
            }

        # No tools in plan mode — pure reasoning only
        if not self._plan_mode and self._tool_manager.ollama_tools:
            payload["tools"] = self._tool_manager.ollama_tools
        return payload

    async def _send_to_ollama(self) -> dict | None:
        """Send non-streaming request to the active backend."""
        payload = self._build_payload(stream=False)
        endpoint = "/v1/chat/completions" if self._backend == "vllm-mlx" else "/api/chat"
        backend_name = "vllm-mlx" if self._backend == "vllm-mlx" else "Ollama"
        base_url = VLLM_URL if self._backend == "vllm-mlx" else OLLAMA_URL

        try:
            resp = await self._http.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if self._backend == "vllm-mlx":
                # Normalize OpenAI format to our internal format
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                tool_calls = None
                if msg.get("tool_calls"):
                    tool_calls = [
                        {"function": {"name": tc["function"]["name"],
                                      "arguments": tc["function"].get("arguments", {})}}
                        for tc in msg["tool_calls"]
                    ]
                result = {"message": {"content": msg.get("content", "") or ""}, "_timing": {}}
                if tool_calls:
                    result["message"]["tool_calls"] = tool_calls
                return result
            return data
        except httpx.TimeoutException:
            print(f"  {C_RED}{DOT} {backend_name} request timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"  {C_RED}{DOT} {backend_name} HTTP {e.response.status_code}: {e.response.text[:200]}{C_RESET}")
            return None
        except httpx.ConnectError:
            print(f"  {C_RED}{DOT} Cannot connect to {backend_name} at {base_url}{C_RESET}")
            return None
        except Exception as e:
            print(f"  {C_RED}{DOT} {backend_name} request failed: {e}{C_RESET}")
            return None

    async def _stream_to_ollama(self) -> dict | None:
        """Stream a response from the active backend, printing tokens as they arrive.

        Returns a response dict with "message" key and "_timing" metadata.
        Falls back to non-streaming if the stream connection fails.
        """
        if self._backend == "vllm-mlx":
            return await self._stream_vllm()
        return await self._stream_ollama()

    async def _stream_ollama(self) -> dict | None:
        """Stream from Ollama /api/chat (NDJSON format)."""
        payload = self._build_payload(stream=True)

        content_parts: list[str] = []
        tool_calls: list[dict] | None = None
        final_chunk: dict = {}
        first_token_time: float | None = None
        stream_start = time.monotonic()
        token_count = 0

        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})

                    # Print text fragments as they arrive
                    fragment = msg.get("content", "")
                    if fragment:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        self._print_fragment(fragment)
                        content_parts.append(fragment)
                        token_count += 1  # approximation: one chunk ~ one token

                    # Accumulate tool_calls (Ollama sends them on the final chunk)
                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]

                    if chunk.get("done"):
                        final_chunk = chunk
                        break

        except httpx.TimeoutException:
            print(f"\n  {C_RED}{DOT} Ollama stream timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"\n  {C_RED}{DOT} Ollama HTTP {e.response.status_code}{C_RESET}")
            print(f"  {C_MUTED}{DIAMOND_SM} falling back to non-streaming{C_RESET}")
            return await self._send_to_ollama()
        except httpx.ConnectError:
            print(f"  {C_RED}{DOT} Cannot connect to Ollama at {OLLAMA_URL}. Is it running?{C_RESET}")
            return None
        except Exception as e:
            print(f"\n  {C_RED}{DOT} Stream failed: {e}{C_RESET}")
            print(f"  {C_MUTED}{DIAMOND_SM} falling back to non-streaming{C_RESET}")
            return await self._send_to_ollama()

        stream_end = time.monotonic()

        # Assemble the response in the same shape as non-streaming
        assembled_message: dict[str, Any] = {
            "content": "".join(content_parts),
        }
        if tool_calls:
            assembled_message["tool_calls"] = tool_calls

        result = dict(final_chunk)
        result["message"] = assembled_message
        result["_timing"] = {
            "ttft": (first_token_time - stream_start) if first_token_time else 0,
            "gen": (stream_end - first_token_time) if first_token_time else 0,
            "tokens": token_count,
        }
        return result

    async def _stream_vllm(self) -> dict | None:
        """Stream from vllm-mlx /v1/chat/completions (SSE format).

        Handles OpenAI-style streaming: data: {json} lines, data: [DONE] at end.
        Tool calls may arrive as deltas across multiple chunks.
        """
        payload = self._build_payload(stream=True)

        content_parts: list[str] = []
        tool_calls_accum: dict[int, dict] = {}  # index -> {id, name, arguments_parts}
        first_token_time: float | None = None
        stream_start = time.monotonic()
        token_count = 0

        try:
            async with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices")
                    if not choices:
                        continue  # vllm-mlx may send chunks with choices: null
                    delta = choices[0].get("delta", {})

                    # Text content (vllm-mlx sends content: null, not missing)
                    fragment = delta.get("content") or ""
                    if fragment:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        self._print_fragment(fragment)
                        content_parts.append(fragment)
                        token_count += 1

                    # Tool calls (may arrive as deltas across multiple chunks)
                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_accum:
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                            tool_calls_accum[idx] = {
                                "id": tc.get("id", ""),
                                "name": "",
                                "arguments_parts": [],
                            }
                        func = tc.get("function", {})
                        if func.get("name"):
                            tool_calls_accum[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls_accum[idx]["arguments_parts"].append(func["arguments"])

        except httpx.TimeoutException:
            print(f"\n  {C_RED}{DOT} vllm-mlx stream timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"\n  {C_RED}{DOT} vllm-mlx HTTP {e.response.status_code}{C_RESET}")
            print(f"  {C_MUTED}{DIAMOND_SM} falling back to non-streaming{C_RESET}")
            return await self._send_to_ollama()
        except httpx.ConnectError:
            print(f"  {C_RED}{DOT} Cannot connect to vllm-mlx at {VLLM_URL}{C_RESET}")
            return None
        except Exception as e:
            print(f"\n  {C_RED}{DOT} Stream failed: {e}{C_RESET}")
            print(f"  {C_MUTED}{DIAMOND_SM} falling back to non-streaming{C_RESET}")
            return await self._send_to_ollama()

        stream_end = time.monotonic()

        # Assemble tool_calls from accumulated deltas
        assembled_tool_calls = None
        if tool_calls_accum:
            assembled_tool_calls = []
            for idx in sorted(tool_calls_accum.keys()):
                tc = tool_calls_accum[idx]
                args_str = "".join(tc["arguments_parts"])
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                assembled_tool_calls.append({
                    "function": {
                        "name": tc["name"],
                        "arguments": args,
                    }
                })

        assembled_message: dict[str, Any] = {
            "content": "".join(content_parts),
        }
        if assembled_tool_calls:
            assembled_message["tool_calls"] = assembled_tool_calls

        return {
            "message": assembled_message,
            "done": True,
            "_timing": {
                "ttft": (first_token_time - stream_start) if first_token_time else 0,
                "gen": (stream_end - first_token_time) if first_token_time else 0,
                "tokens": token_count,
            },
        }

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate: 4 chars ~ 1 token."""
        return sum(len(m.get("content", "")) for m in messages) // 4

    def _build_messages(self) -> list[dict]:
        """Build the message list, trimming old messages if over budget."""
        if self._plan_mode:
            system_content = PLANNING_PROMPT
            if self._backend == "vllm-mlx":
                system_content = "/think\n" + system_content
        else:
            system_content = self._system_prompt
            if self._backend == "vllm-mlx":
                # Qwen3 enables thinking by default. /no_think in the system prompt
                # is the reliable way to disable it (chat_template_kwargs alone is
                # not sufficient for streaming requests).
                system_content = "/no_think\n" + system_content
        messages = [{"role": "system", "content": system_content}]

        # Trim from front if over budget
        history = list(self._messages)
        while self._estimate_tokens(history) > MAX_HISTORY_TOKENS and len(history) > 2:
            history.pop(0)

        # Clean up orphaned tool responses at the front
        # (happens when trim boundary splits an assistant+tool_calls / tool pair)
        while history and history[0]["role"] == "tool":
            history.pop(0)

        if len(history) < len(self._messages):
            trimmed = len(self._messages) - len(history)
            print(f"  {C_MUTED}{DIAMOND_SM} trimmed {trimmed} old messages to stay within context{C_RESET}")

        messages.extend(history)
        return messages

    def reset(self):
        """Clear conversation history."""
        self._messages.clear()


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def print_banner(tool_manager: MCPToolManager, backend_version: str = "",
                 backend: str = "ollama"):
    """Print startup banner — everything in one clean box."""
    print(proving_ground.render_compact(
        friendly_model_name(OLLAMA_MODEL),
        ollama_version=backend_version,
        tool_count=len(tool_manager.tool_names),
        server_count=len(tool_manager._clients),
        backend=backend,
    ))
    print()


def _setup_readline():
    """Initialize readline with persistent history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(HISTORY_FILE))
    except (FileNotFoundError, PermissionError, OSError):
        pass
    readline.set_history_length(1000)
    # Tab-complete slash commands
    commands = ["/clear", "/tools", "/stats", "/rate", "/override",
                "/promote", "/knowledge", "/history", "/model",
                "/sessions", "/help", "/quit", "/exit"]
    def completer(text, state):
        options = [c for c in commands if c.startswith(text)]
        return options[state] if state < len(options) else None
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def _save_readline():
    """Save readline history."""
    try:
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass


async def interactive_loop(chat: OllamaChat, tool_manager: MCPToolManager,
                           session_logger: SessionLogger,
                           backend_version: str = "", backend: str = "ollama"):
    """Run the interactive input loop."""
    _setup_readline()
    print_banner(tool_manager, backend_version, backend)

    while True:
        try:
            mode_hint = f" \x01{C_YELLOW}\x02[plan]\x01{C_RESET}\x02" if chat._plan_mode else ""
            prompt_str = f"\x01{C_WHITE}{C_BOLD}\x02You\x01{C_RESET}\x02{mode_hint} \x01{C_CYAN}\x02{ARROW_R}\x01{C_RESET}\x02 "
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(prompt_str)
            )
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/clear":
                chat.reset()
                print(f"  {C_MUTED}{DIAMOND_SM} conversation cleared{C_RESET}")
                continue
            elif cmd == "/plan":
                if chat._plan_mode:
                    print(f"  {C_MUTED}{DIAMOND_SM} already in plan mode. /build to switch to execution{C_RESET}")
                else:
                    chat._plan_mode = True
                    chat.reset()
                    print(f"  {C_YELLOW}{C_BOLD}{DIAMOND} Plan mode{C_RESET} {C_MUTED}— thinking enabled, tools disabled{C_RESET}")
                    print(f"  {C_MUTED}{DIAMOND_SM} discuss your approach, then /build to execute{C_RESET}")
                continue
            elif cmd == "/build":
                if not chat._plan_mode:
                    print(f"  {C_MUTED}{DIAMOND_SM} not in plan mode. /plan to start planning{C_RESET}")
                else:
                    chat._plan_mode = False
                    chat.reset()
                    print(f"  {C_GREEN}{C_BOLD}{DIAMOND} Build mode{C_RESET} {C_MUTED}— tools enabled, ready to execute{C_RESET}")
                continue
            elif cmd == "/tools":
                if tool_manager.tool_names:
                    for name in sorted(tool_manager.tool_names):
                        _server, tool = tool_manager._tool_map[name]
                        desc = (tool.description or "")[:80]
                        print(f"  {C_CYAN}{ARROW_R}{C_RESET} {C_WHITE}{name}{C_RESET} {C_MUTED}({_server}) {desc}{C_RESET}")
                else:
                    print(f"  {C_MUTED}{DIAMOND_SM} No tools available{C_RESET}")
                continue
            elif cmd == "/stats":
                print(proving_ground.render_full())
                continue
            elif cmd.startswith("/rate"):
                parts = user_input.split()
                task_id = tracker.get_last_task_id()
                if task_id is None:
                    print(f"  {C_MUTED}{DIAMOND_SM} No tasks to rate{C_RESET}")
                elif len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 5:
                    rating = int(parts[1])
                    tracker.rate_task(task_id, rating)
                    session_logger.log("rating", "", task_id=task_id, rating=rating)
                    stars = "★" * rating + "☆" * (5 - rating)
                    print(f"  {C_YELLOW}{stars}{C_RESET} {C_MUTED}task #{task_id}{C_RESET}")
                else:
                    print(f"  {C_MUTED}{DIAMOND_SM} Usage: /rate <1-5>{C_RESET}")
                continue
            elif cmd == "/override":
                task_id = tracker.get_last_task_id()
                if task_id:
                    tracker.mark_override(task_id)
                    session_logger.log("override", "", task_id=task_id)
                    print(f"  {C_YELLOW}{DIAMOND_SM}{C_RESET} {C_MUTED}Task #{task_id} marked as overridden{C_RESET}")
                else:
                    print(f"  {C_MUTED}{DIAMOND_SM} No tasks to mark{C_RESET}")
                continue
            elif cmd == "/promote":
                promo = tracker.check_promotion()
                if promo:
                    tier_name = tracker.TIER_NAMES.get(promo, promo)
                    tracker.apply_tier_change(promo, "Promotion approved by human", OLLAMA_MODEL)
                    print(f"  {C_GREEN}{C_BOLD}{DIAMOND} Promoted to {tier_name} ({promo}){C_RESET}")
                else:
                    print(f"  {C_MUTED}{DIAMOND_SM} Not yet qualified. Use /stats to see requirements.{C_RESET}")
                continue
            elif cmd.startswith("/knowledge"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"  {C_MUTED}{DIAMOND_SM} Usage: /knowledge <query>{C_RESET}")
                else:
                    query = parts[1]
                    result = await tool_manager.call_tool("search", {"query": query, "scope": "knowledge"})
                    if result.startswith("Error:") or result.startswith("Tool error:"):
                        print(f"  {C_RED}{DOT} {result}{C_RESET}")
                    elif "No results found" in result or "No knowledge found" in result:
                        print(f"  {C_MUTED}{DIAMOND_SM} No knowledge found for '{query}'{C_RESET}")
                    else:
                        print(f"  {C_GREEN}{DOT}{C_RESET} {result[:500]}")
                continue
            elif cmd == "/history":
                import sqlite3
                conn = sqlite3.connect(str(tracker.DB_PATH))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, started_at, prompt_preview, outcome, tool_calls, tool_errors, user_rating "
                    "FROM tasks ORDER BY id DESC LIMIT 15"
                ).fetchall()
                conn.close()
                if not rows:
                    print(f"  {C_MUTED}{DIAMOND_SM} No task history{C_RESET}")
                else:
                    for r in reversed(rows):
                        ts = r["started_at"][11:16] if r["started_at"] else "??:??"
                        outcome_icon = f"{C_GREEN}{DOT}{C_RESET}" if r["outcome"] == "completed" else f"{C_RED}{DOT}{C_RESET}"
                        preview = (r["prompt_preview"] or "")[:45]
                        tools = f"{r['tool_calls']}t" if r["tool_calls"] else ""
                        rating = f" {C_YELLOW}{'★' * r['user_rating']}{C_RESET}" if r["user_rating"] else ""
                        print(f"  {C_MUTED}#{r['id']:>3} {ts}{C_RESET} {outcome_icon} {C_WHITE}{preview}{C_RESET} {C_MUTED}{tools}{C_RESET}{rating}")
                continue
            elif cmd.startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    backend_label = f" {C_MUTED}via {chat._backend}{C_RESET}"
                    print(f"  {C_WHITE}{friendly_model_name(OLLAMA_MODEL)}{C_RESET}{backend_label}")
                    # List available models from the active backend
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as http:
                            if chat._backend == "vllm-mlx":
                                resp = await http.get(f"{VLLM_URL}/v1/models")
                                resp.raise_for_status()
                                models = resp.json().get("data", [])
                                for m in models[:10]:
                                    name = m.get("id", "?")
                                    mark = f"{C_GREEN}{DOT}{C_RESET}" if name == OLLAMA_MODEL else f"{C_MUTED}{DIAMOND_SM}{C_RESET}"
                                    print(f"  {mark} {C_WHITE}{friendly_model_name(name)}{C_RESET} {C_MUTED}(vllm-mlx){C_RESET}")
                            else:
                                resp = await http.get(f"{OLLAMA_URL}/api/tags")
                                resp.raise_for_status()
                                models = resp.json().get("models", [])
                                for m in models[:10]:
                                    name = m.get("name", "?")
                                    size = m.get("size", 0) / 1e9
                                    mark = f"{C_GREEN}{DOT}{C_RESET}" if name == OLLAMA_MODEL else f"{C_MUTED}{DIAMOND_SM}{C_RESET}"
                                    print(f"  {mark} {C_WHITE}{name}{C_RESET} {C_MUTED}{size:.1f}GB{C_RESET}")
                    except Exception:
                        pass
                else:
                    new_model = parts[1].strip()
                    _this = sys.modules[__name__]
                    _this.OLLAMA_MODEL = new_model
                    if tracker.check_model_change(new_model):
                        print(f"  {C_YELLOW}{DIAMOND_SM} Model changed — tier reset to T0{C_RESET}")
                    print(f"  {C_GREEN}{DOT}{C_RESET} Model set to {C_WHITE}{new_model}{C_RESET}")
                continue
            elif cmd == "/sessions":
                sessions = SessionLogger.list_sessions(limit=10)
                if not sessions:
                    print(f"  {C_MUTED}{DIAMOND_SM} No sessions saved{C_RESET}")
                else:
                    for s in sessions:
                        size_kb = s["size"] / 1024
                        print(f"  {C_MUTED}{s['id']}{C_RESET}  {C_WHITE}{s['messages']} msgs{C_RESET}  {C_MUTED}{size_kb:.1f}KB{C_RESET}  {C_MUTED}{s['preview']}{C_RESET}")
                continue
            elif cmd == "/help":
                print(f"  {C_BOLD}{C_WHITE}Chat{C_RESET}")
                print(f"  {C_WHITE}/clear{C_RESET}      {C_MUTED}─ Reset conversation{C_RESET}")
                print(f"  {C_WHITE}/model [M]{C_RESET}  {C_MUTED}─ Show/change model{C_RESET}")
                print(f"  {C_WHITE}/tools{C_RESET}      {C_MUTED}─ List MCP tools{C_RESET}")
                print(f"  {C_WHITE}/knowledge{C_RESET}  {C_MUTED}─ Query knowledge base{C_RESET}")
                print()
                print(f"  {C_BOLD}{C_WHITE}Modes{C_RESET}")
                print(f"  {C_WHITE}/plan{C_RESET}       {C_MUTED}─ Plan mode (thinking on, tools off){C_RESET}")
                print(f"  {C_WHITE}/build{C_RESET}      {C_MUTED}─ Build mode (tools on, execute plan){C_RESET}")
                print()
                print(f"  {C_BOLD}{C_WHITE}Proving Ground{C_RESET}")
                print(f"  {C_WHITE}/stats{C_RESET}      {C_MUTED}─ Full statistics{C_RESET}")
                print(f"  {C_WHITE}/history{C_RESET}    {C_MUTED}─ Recent tasks{C_RESET}")
                print(f"  {C_WHITE}/rate N{C_RESET}     {C_MUTED}─ Rate last response (1-5){C_RESET}")
                print(f"  {C_WHITE}/override{C_RESET}   {C_MUTED}─ Mark last task as corrected{C_RESET}")
                print(f"  {C_WHITE}/promote{C_RESET}    {C_MUTED}─ Approve tier promotion{C_RESET}")
                print(f"  {C_WHITE}/sessions{C_RESET}   {C_MUTED}─ List saved sessions{C_RESET}")
                print()
                print(f"  {C_WHITE}/quit{C_RESET}       {C_MUTED}─ Exit{C_RESET}")
                continue
            else:
                print(f"  {C_MUTED}{DIAMOND_SM} Unknown command. Try /help{C_RESET}")
                continue

        # Send to Purple -- response text streams inline during chat()
        mode_label = f" {C_YELLOW}[planning]{C_RESET}" if chat._plan_mode else ""
        print(f"\n{C_PURPLE}{C_BOLD}Purple{C_RESET}{mode_label} {C_PURPLE}{DIAMOND}{C_RESET}")
        print(f"  ", end="", flush=True)
        try:
            await chat.chat(user_input)
            # chat() already printed the streamed text and a trailing newline
        except KeyboardInterrupt:
            print(f"\n  {C_MUTED}{DIAMOND_SM} interrupted{C_RESET}\n")
            continue
        except asyncio.CancelledError:
            print(f"\n  {C_MUTED}{DIAMOND_SM} cancelled{C_RESET}\n")
            continue


async def one_shot(chat: OllamaChat, prompt: str):
    """Run a single prompt. Response is streamed to stdout during chat()."""
    await chat.chat(prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _check_vllm() -> tuple[bool, str, str]:
    """Check if vllm-mlx is reachable. Returns (healthy, model_id, version)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_URL}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "unknown")
                return True, model_id, "vllm-mlx"
            return True, "", "vllm-mlx"
    except Exception:
        return False, "", ""


async def _check_ollama() -> tuple[bool, str]:
    """Verify Ollama is reachable. Returns (healthy, version_string)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/version")
            resp.raise_for_status()
            version = resp.json().get("version", "unknown")
            return True, version
    except httpx.ConnectError:
        print(f"  {C_RED}{DOT} Cannot connect to Ollama at {OLLAMA_URL}{C_RESET}")
        print(f"  {C_MUTED}  Start it with: ollama serve{C_RESET}")
        return False, ""
    except Exception as e:
        print(f"  {C_YELLOW}{DIAMOND_SM} Ollama health check failed: {e}{C_RESET}")
        return True, ""  # Proceed anyway -- might still work


async def _detect_backend() -> tuple[str, str, str]:
    """Auto-detect the best available backend.

    Returns (backend, model_name, version_string).
    backend is "" if no backend is available.
    Priority: vllm-mlx (if running with a model loaded) > Ollama.
    """
    global OLLAMA_MODEL

    # Try vllm-mlx if requested or auto-detecting
    if PURPLE_BACKEND in ("vllm-mlx", "auto"):
        healthy, model_id, ver = await _check_vllm()
        if healthy:
            # Auto-detect requires a model loaded; forced mode accepts empty
            if model_id or PURPLE_BACKEND == "vllm-mlx":
                if model_id:
                    OLLAMA_MODEL = model_id
                return "vllm-mlx", OLLAMA_MODEL, ver
        elif PURPLE_BACKEND == "vllm-mlx":
            print(f"  {C_RED}{DOT} vllm-mlx not available at {VLLM_URL}{C_RESET}")

    # Fall back to Ollama (or it was explicitly requested)
    healthy, ver = await _check_ollama()
    if not healthy:
        return "", OLLAMA_MODEL, ""
    return "ollama", OLLAMA_MODEL, ver


async def main():
    tool_manager = MCPToolManager()
    session_logger = SessionLogger()
    chat = None
    backend_version = ""

    # Initialize Proving Ground tracker
    tracker.init_db()

    try:
        # Detect backend (auto: vllm-mlx if available, else Ollama)
        backend, model_name, backend_version = await _detect_backend()
        if not backend:
            return

        if tracker.check_model_change(OLLAMA_MODEL):
            print(f"  {C_YELLOW}{DIAMOND_SM} Model changed — tier reset to T0 (Candidate){C_RESET}")

        # Connect to MCP servers (silent — results shown in banner)
        await tool_manager.connect()

        # Log session metadata for verification/fine-tuning pipeline
        session_logger.log_session_meta(
            model=OLLAMA_MODEL, backend=backend,
            tier=tracker.get_current_tier(),
            tool_count=len(tool_manager.tool_names),
            server_count=len(tool_manager._clients))

        # Create chat client
        chat = OllamaChat(tool_manager, session_logger, backend=backend)

        if len(sys.argv) > 1:
            # One-shot mode: join all args as prompt
            prompt = " ".join(sys.argv[1:])
            await one_shot(chat, prompt)
        else:
            # Interactive mode
            await interactive_loop(chat, tool_manager, session_logger,
                                   backend_version, backend)

    except KeyboardInterrupt:
        pass
    finally:
        # Auto-demotion check at session end
        demotion = tracker.check_demotion()
        if demotion:
            old_tier = tracker.get_current_tier()
            tracker.apply_tier_change(demotion, "Auto-demotion: TCR below floor", OLLAMA_MODEL)
            old_name = tracker.TIER_NAMES.get(old_tier, old_tier)
            new_name = tracker.TIER_NAMES.get(demotion, demotion)
            print(f"\n  {C_RED}{DIAMOND} Demoted: {old_name} → {new_name}{C_RESET}")

        if chat:
            await chat.close()
        await tool_manager.disconnect()
        _save_readline()

        # Session summary
        if session_logger.message_count > 0:
            print(f"\n  {C_MUTED}{DIAMOND_SM} session {session_logger.session_id} · {session_logger.message_count} messages logged{C_RESET}")
        print(f"  {C_MUTED}{DIAMOND_SM} purple offline{C_RESET}")


def main_sync():
    """Synchronous entry point for console_scripts in pyproject.toml."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
