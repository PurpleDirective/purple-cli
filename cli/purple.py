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
import sys
import os
from pathlib import Path
from typing import Any

import httpx
from fastmcp.client import Client
from fastmcp.client.transports import StdioTransport

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "purple:latest")
IDENTITY_PATH = Path.home() / ".purple" / "identity" / "identity.md"
MCP_CONFIG_PATH = Path.home() / ".purple" / "config" / "mcp.json"

MAX_TOOL_ROUNDS = 10
MAX_HISTORY_TOKENS = 50000  # Leave ~15K for system prompt + tool defs
OLLAMA_TIMEOUT = 300.0  # seconds -- model can be slow on first load
NUM_CTX = 65536  # Match Modelfile's 64K context

# ANSI colors for terminal output
C_RESET = "\033[0m"
C_PURPLE = "\033[35m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_BOLD = "\033[1m"
C_YELLOW = "\033[33m"

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
    if IDENTITY_PATH.exists():
        return IDENTITY_PATH.read_text().strip()
    print(f"{C_DIM}[warn] Identity file not found at {IDENTITY_PATH}{C_RESET}")
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
        # Load server config
        if not MCP_CONFIG_PATH.exists():
            print(f"{C_RED}[mcp] Config not found: {MCP_CONFIG_PATH}{C_RESET}")
            return

        try:
            config = json.loads(MCP_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"{C_RED}[mcp] Failed to read config: {e}{C_RESET}")
            return

        servers = config.get("servers", {})

        for server_name, server_cfg in servers.items():
            if not server_cfg.get("enabled", True):
                print(f"{C_DIM}[mcp] {server_name}: disabled, skipping{C_RESET}")
                continue

            cmd = server_cfg.get("command", [])
            if not cmd:
                print(f"{C_DIM}[warn] {server_name}: no command specified{C_RESET}")
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

                print(f"{C_DIM}[mcp] {server_name}: {len(tools)} tools loaded{C_RESET}")

            except Exception as e:
                print(f"{C_RED}[mcp] Failed to connect to {server_name}: {e}{C_RESET}")

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

    def __init__(self, tool_manager: MCPToolManager):
        self._tool_manager = tool_manager
        self._messages: list[dict] = []
        self._system_prompt = load_identity()
        self._http = httpx.AsyncClient(
            base_url=OLLAMA_URL,
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0),
        )

    async def close(self):
        await self._http.aclose()

    async def chat(self, user_input: str) -> str:
        """Send a user message and get a response, handling tool calls.

        Text is streamed to stdout as it arrives. Tool call rounds still use
        the streaming path for each model turn.

        Returns the final assistant text response (already printed to terminal).
        """
        self._messages.append({"role": "user", "content": user_input})

        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                response = await self._stream_to_ollama()
            except KeyboardInterrupt:
                print(f"\n{C_DIM}[stream interrupted]{C_RESET}")
                return "[interrupted]"

            if response is None:
                return "[Error: no response from Ollama]"

            message = response.get("message", {})
            tool_calls = message.get("tool_calls")

            # Detect XML-formatted tool calls in content field
            # (Qwen3-Coder fallback when it can't emit proper JSON tool calls)
            if not tool_calls:
                content = message.get("content", "")
                xml_calls = _parse_xml_tool_calls(content)
                if xml_calls:
                    print(f"\n{C_YELLOW}  [detected XML tool call in content -- intercepting]{C_RESET}")
                    tool_calls = xml_calls

            if not tool_calls:
                # No tool calls -- we have a final response (already streamed)
                content = message.get("content", "")
                self._messages.append({"role": "assistant", "content": content})
                print()  # Newline after streamed output
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

                # Print tool call info
                args_preview = json.dumps(tool_args, ensure_ascii=False)
                if len(args_preview) > 120:
                    args_preview = args_preview[:117] + "..."
                print(f"{C_CYAN}  -> {tool_name}({args_preview}){C_RESET}")

                # Execute the tool
                result = await self._tool_manager.call_tool(tool_name, tool_args)

                # Print brief result
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"{C_DIM}  <- {result_preview}{C_RESET}")

                # Add tool result to message history
                self._messages.append({
                    "role": "tool",
                    "content": result,
                })

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
            print(f"{C_RED}[error] Ollama request timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"{C_RED}[error] Ollama HTTP {e.response.status_code}: {e.response.text[:200]}{C_RESET}")
            return None
        except httpx.ConnectError:
            print(f"{C_RED}[error] Cannot connect to Ollama at {OLLAMA_URL}. Is it running?{C_RESET}")
            return None
        except Exception as e:
            print(f"{C_RED}[error] Ollama request failed: {e}{C_RESET}")
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
            print(f"\n{C_RED}[error] Ollama stream timed out after {OLLAMA_TIMEOUT}s{C_RESET}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"\n{C_RED}[error] Ollama HTTP {e.response.status_code}{C_RESET}")
            # Fall back to non-streaming
            print(f"{C_DIM}[falling back to non-streaming]{C_RESET}")
            return await self._send_to_ollama()
        except httpx.ConnectError:
            print(f"{C_RED}[error] Cannot connect to Ollama at {OLLAMA_URL}. Is it running?{C_RESET}")
            return None
        except Exception as e:
            print(f"\n{C_RED}[error] Stream failed: {e}{C_RESET}")
            print(f"{C_DIM}[falling back to non-streaming]{C_RESET}")
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
            print(f"{C_DIM}[trimmed {trimmed} old messages to stay within context]{C_RESET}")

        messages.extend(history)
        return messages

    def reset(self):
        """Clear conversation history."""
        self._messages.clear()


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def print_banner(tool_manager: MCPToolManager):
    """Print startup banner."""
    tools = tool_manager.tool_names
    print(f"\n{C_BOLD}{C_PURPLE}Purple CLI{C_RESET} -- {OLLAMA_MODEL} via {OLLAMA_URL}")
    print(f"{C_DIM}Tools: {', '.join(tools) if tools else 'none'}{C_RESET}")
    print(f"{C_DIM}Commands: /clear (reset), /tools (list), /quit (exit){C_RESET}")
    print()


async def interactive_loop(chat: OllamaChat, tool_manager: MCPToolManager):
    """Run the interactive input loop."""
    print_banner(tool_manager)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(f"{C_GREEN}you>{C_RESET} ")
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
                print(f"{C_DIM}[conversation cleared]{C_RESET}")
                continue
            elif cmd == "/tools":
                if tool_manager.tool_names:
                    for name in sorted(tool_manager.tool_names):
                        _server, tool = tool_manager._tool_map[name]
                        desc = (tool.description or "")[:80]
                        print(f"  {C_CYAN}{name}{C_RESET} ({_server}) -- {desc}")
                else:
                    print(f"{C_DIM}  No tools available{C_RESET}")
                continue
            elif cmd == "/help":
                print(f"  /clear  -- Reset conversation history")
                print(f"  /tools  -- List available MCP tools")
                print(f"  /quit   -- Exit")
                continue
            else:
                print(f"{C_DIM}  Unknown command. Try /help{C_RESET}")
                continue

        # Send to Purple -- response text streams inline during chat()
        print(f"{C_PURPLE}purple>{C_RESET} ", end="", flush=True)
        try:
            await chat.chat(user_input)
            # chat() already printed the streamed text and a trailing newline
        except KeyboardInterrupt:
            print(f"\n{C_DIM}[interrupted]{C_RESET}\n")
            continue
        except asyncio.CancelledError:
            print(f"\n{C_DIM}[cancelled]{C_RESET}\n")
            continue


async def one_shot(chat: OllamaChat, prompt: str):
    """Run a single prompt. Response is streamed to stdout during chat()."""
    await chat.chat(prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    tool_manager = MCPToolManager()
    chat = None

    try:
        # Connect to MCP servers
        await tool_manager.connect()

        # Create chat client
        chat = OllamaChat(tool_manager)

        if len(sys.argv) > 1:
            # One-shot mode: join all args as prompt
            prompt = " ".join(sys.argv[1:])
            await one_shot(chat, prompt)
        else:
            # Interactive mode
            await interactive_loop(chat, tool_manager)

    except KeyboardInterrupt:
        pass
    finally:
        if chat:
            await chat.close()
        await tool_manager.disconnect()
        print(f"\n{C_DIM}[purple offline]{C_RESET}")


def main_sync():
    """Synchronous entry point for console_scripts in pyproject.toml."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
