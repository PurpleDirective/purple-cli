#!/usr/bin/env python3
"""
Purple CLI -- Native Ollama + MCP Orchestrator

Talks to Ollama via native /api/chat (NOT the broken /v1 endpoint).
Executes MCP tool calls directly via FastMCP STDIO clients.
Bypasses the lossy AI SDK translation layer entirely.

Usage:
    Interactive:  python purple.py
    One-shot:     python purple.py "your prompt here"
"""

import asyncio
import json
import re
import readline
import sys
import os
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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "purple:latest")
IDENTITY_PATH = Path.home() / ".purple" / "identity" / "identity.md"
MCP_CONFIG_PATH = Path.home() / ".purple" / "config" / "mcp.json"

MAX_TOOL_ROUNDS = 10
MAX_THINKING_CALLS = 3    # Cap sequentialthinking per chat turn
MAX_EMPTY_LOOKUPS = 2     # Stop searching empty knowledge base after N misses
MAX_HISTORY_TOKENS = 50000  # Leave ~15K for system prompt + tool defs
OLLAMA_TIMEOUT = 300.0  # seconds -- model can be slow on first load
NUM_CTX = 65536  # Match Modelfile's 64K context

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
    """Manages conversation with Ollama via native /api/chat."""

    def __init__(self, tool_manager: MCPToolManager, session_logger: SessionLogger):
        self._tool_manager = tool_manager
        self._session_logger = session_logger
        self._messages: list[dict] = []
        self._system_prompt = load_identity()
        self._http = httpx.AsyncClient(
            base_url=OLLAMA_URL,
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0),
        )
        self._current_task_id: int | None = None
        # Per-turn tool budget counters (reset each chat call)
        self._thinking_calls: int = 0
        self._empty_lookups: int = 0

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

        # Proving Ground: start tracking this task
        self._current_task_id = tracker.start_task(user_input, OLLAMA_MODEL)

        for round_num in range(MAX_TOOL_ROUNDS):
            # Continuation rounds get a fresh response prefix with budget indicator
            if round_num > 0:
                remaining = MAX_TOOL_ROUNDS - round_num
                budget_color = C_GREEN if remaining > 4 else (C_YELLOW if remaining > 2 else C_RED)
                print(f"\n{C_PURPLE}{DIAMOND}{C_RESET} {budget_color}{C_DIM}[{remaining} rounds left]{C_RESET} ", end="", flush=True)

            try:
                response = await self._stream_to_ollama()
            except KeyboardInterrupt:
                print(f"\n{C_MUTED}  {DIAMOND_SM} stream interrupted{C_RESET}")
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "interrupted", round_num)
                return "[interrupted]"

            if response is None:
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "failed", round_num)
                return "[Error: no response from Ollama]"

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
                print()  # Newline after streamed output
                # Proving Ground: mark task complete
                if self._current_task_id is not None:
                    tracker.complete_task(self._current_task_id, "completed", round_num)
                return content

            # Tool calls detected -- newline after any streamed preamble text
            print()

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
                        print(f"  {C_YELLOW}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_YELLOW}capped ({MAX_THINKING_CALLS} max) — produce your answer now{C_RESET}")
                        result = (
                            f"BUDGET: You have used {MAX_THINKING_CALLS} thinking rounds. "
                            f"Stop thinking and produce your final answer directly. "
                            f"You have {MAX_TOOL_ROUNDS - round_num - 1} tool rounds remaining."
                        )
                        self._messages.append({"role": "tool", "content": result})
                        continue

                # Cap repeated empty knowledge lookups
                if tool_name == "lookup_knowledge":
                    if self._empty_lookups >= MAX_EMPTY_LOOKUPS:
                        print(f"  {C_YELLOW}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_YELLOW}skipped — knowledge base is sparse, proceed without{C_RESET}")
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
                    print(f"  {C_CYAN}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}  {C_MUTED}{summary}{C_RESET}")
                else:
                    print(f"  {C_CYAN}{ARROW_R} {C_WHITE}{tool_name}{C_RESET}")

                # Execute the tool
                result = await self._tool_manager.call_tool(tool_name, tool_args)

                # Track empty knowledge lookups
                if tool_name == "lookup_knowledge" and "No knowledge found" in result:
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

        # Proving Ground: mark task as failed (too many rounds)
        if self._current_task_id is not None:
            tracker.complete_task(self._current_task_id, "failed", MAX_TOOL_ROUNDS)
        return "[Error: max tool call rounds exceeded]"

    def _build_payload(self, *, stream: bool) -> dict[str, Any]:
        """Build the Ollama /api/chat payload."""
        payload: dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "messages": self._build_messages(),
            "stream": stream,
            "options": {
                "num_ctx": NUM_CTX,
            },
        }
        if self._tool_manager.ollama_tools:
            payload["tools"] = self._tool_manager.ollama_tools
        return payload

    async def _send_to_ollama(self) -> dict | None:
        """Send the current conversation to Ollama /api/chat (non-streaming)."""
        payload = self._build_payload(stream=False)

        try:
            resp = await self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            print(f"  {C_RED}{DOT} Ollama request timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"  {C_RED}{DOT} Ollama HTTP {e.response.status_code}: {e.response.text[:200]}{C_RESET}")
            return None
        except httpx.ConnectError:
            print(f"  {C_RED}{DOT} Cannot connect to Ollama at {OLLAMA_URL}. Is it running?{C_RESET}")
            return None
        except Exception as e:
            print(f"  {C_RED}{DOT} Ollama request failed: {e}{C_RESET}")
            return None

    async def _stream_to_ollama(self) -> dict | None:
        """Stream a response from Ollama /api/chat, printing tokens as they arrive.

        Ollama streaming sends one JSON object per line. Each chunk has:
          - message.content: text fragment (may be empty)
          - message.tool_calls: present only on the final chunk when tools are invoked
          - done: true on the final chunk

        Returns a response dict in the same format as non-streaming (with a
        synthesized "message" key), or None on failure.
        Falls back to non-streaming if the stream connection fails.
        """
        payload = self._build_payload(stream=True)

        content_parts: list[str] = []
        tool_calls: list[dict] | None = None
        final_chunk: dict = {}

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
                        print(fragment, end="", flush=True)
                        content_parts.append(fragment)

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

        # Assemble the response in the same shape as non-streaming
        assembled_message: dict[str, Any] = {
            "content": "".join(content_parts),
        }
        if tool_calls:
            assembled_message["tool_calls"] = tool_calls

        result = dict(final_chunk)
        result["message"] = assembled_message
        return result

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate: 4 chars ~ 1 token."""
        return sum(len(m.get("content", "")) for m in messages) // 4

    def _build_messages(self) -> list[dict]:
        """Build the message list, trimming old messages if over budget."""
        messages = [{"role": "system", "content": self._system_prompt}]

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

def print_banner(tool_manager: MCPToolManager, ollama_version: str = ""):
    """Print startup banner — everything in one clean box."""
    print(proving_ground.render_compact(
        OLLAMA_MODEL,
        ollama_version=ollama_version,
        tool_count=len(tool_manager.tool_names),
        server_count=len(tool_manager._clients),
    ))
    print()


def _setup_readline():
    """Initialize readline with persistent history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(HISTORY_FILE))
    except FileNotFoundError:
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
                           ollama_version: str = ""):
    """Run the interactive input loop."""
    _setup_readline()
    print_banner(tool_manager, ollama_version)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(f"{C_CYAN}{ARROW_R}{C_RESET} ")
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
                elif len(parts) == 2 and parts[1].isdigit():
                    rating = int(parts[1])
                    tracker.rate_task(task_id, rating)
                    stars = "★" * rating + "☆" * (5 - rating)
                    print(f"  {C_YELLOW}{stars}{C_RESET} {C_MUTED}task #{task_id}{C_RESET}")
                else:
                    print(f"  {C_MUTED}{DIAMOND_SM} Usage: /rate <1-5>{C_RESET}")
                continue
            elif cmd == "/override":
                task_id = tracker.get_last_task_id()
                if task_id:
                    tracker.mark_override(task_id)
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
                    result = await tool_manager.call_tool("lookup_knowledge", {"query": query})
                    if result.startswith("Error:") or result.startswith("Tool error:"):
                        print(f"  {C_RED}{DOT} {result}{C_RESET}")
                    elif "No knowledge found" in result:
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
                    print(f"  {C_WHITE}{OLLAMA_MODEL}{C_RESET}")
                    # List available models
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as http:
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
        print(f"{C_PURPLE}{DIAMOND}{C_RESET} ", end="", flush=True)
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


async def main():
    tool_manager = MCPToolManager()
    session_logger = SessionLogger()
    chat = None
    ollama_version = ""

    # Initialize Proving Ground tracker
    tracker.init_db()
    if tracker.check_model_change(OLLAMA_MODEL):
        print(f"  {C_YELLOW}{DIAMOND_SM} Model changed — tier reset to T0 (Candidate){C_RESET}")

    try:
        # Verify Ollama is reachable
        healthy, ollama_version = await _check_ollama()
        if not healthy:
            return

        # Connect to MCP servers (silent — results shown in banner)
        await tool_manager.connect()

        # Create chat client
        chat = OllamaChat(tool_manager, session_logger)

        if len(sys.argv) > 1:
            # One-shot mode: join all args as prompt
            prompt = " ".join(sys.argv[1:])
            await one_shot(chat, prompt)
        else:
            # Interactive mode
            await interactive_loop(chat, tool_manager, session_logger, ollama_version)

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
