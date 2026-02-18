# Purple CLI

A local AI agent that connects Ollama to MCP tools. ~1200 lines of Python across 3 files. No frameworks, no OpenAI shims, no abstractions between you and the model.

## What It Does

- Talks to Ollama via native `/api/chat` (not the OpenAI-compatible `/v1` endpoint)
- Calls MCP tools through FastMCP STDIO transport
- Loads MCP servers from a config file -- add your own in one line
- Streams output token-by-token as it generates
- Catches XML-formatted tool calls when models fall back from JSON (Qwen3-Coder workaround)
- Manages context with a sliding window that trims old messages automatically
- Injects a system prompt from a Markdown identity file
- Runs interactive or one-shot (`python purple.py "your prompt"`)

## Batteries Included

Ships with 2 MCP servers:

**purple-memory** -- SQLite persistent memory (4 tools)
- `store_memory` -- Save facts, preferences, experiences, corrections
- `recall_memories` -- Keyword search with type filtering
- `forget_memory` -- Delete by ID with reason tracking
- `list_recent` -- Show memories from the last N hours

**purple-docs** -- Document handling (6 tools)
- `read_pdf` -- Extract text and tables from PDFs (page ranges supported)
- `read_excel` -- Read spreadsheets (all sheets or specific)
- `create_excel` -- Build styled workbooks from JSON
- `create_word` -- Generate Word documents with headings, lists, tables
- `create_powerpoint` -- Create presentations with multiple layouts
- `list_directory` -- Browse files with size and type info

Both servers enforce path validation -- file operations are restricted to the home directory and `/tmp`.

## Quick Start

```bash
git clone https://github.com/PurpleDirective/purple-cli.git ~/.purple
cd ~/.purple
pip install -r requirements.txt
python cli/purple.py
```

You need Ollama running with a model that supports tool calling. Create a model with the included Modelfile, or point to your own:

```bash
ollama create purple -f config/Modelfile
python cli/purple.py
```

Or use any tool-calling model directly:

```bash
OLLAMA_MODEL=qwen3-coder:30b python cli/purple.py "summarize this PDF"
```

## Architecture

```
you --> Purple CLI --> Ollama /api/chat --> your model
            |
            +--> MCP Servers (STDIO) --> tools
                    |
                    +-- purple-memory (SQLite)
                    +-- purple-docs (PDF, Excel, Word, PowerPoint)
                    +-- [your servers here]
```

The CLI is the orchestrator. It sends your message to Ollama with tool definitions, gets back either text or tool calls, executes the tools via MCP, sends results back to the model, and repeats until the model responds with text. Up to 10 tool-call rounds per turn.

## Configuration

**`config/mcp.json`** -- Declares MCP servers. Each entry has a command array and an enabled flag:

```json
{
  "servers": {
    "purple-memory": {
      "command": ["python", "memory/server.py"],
      "enabled": true
    }
  }
}
```

**`config/Modelfile`** -- Ollama model definition. Sets context window, temperature, and a system prompt with tool-use rules.

**`identity/identity.md`** -- System prompt loaded at startup. This is your model's personality. Write whatever you want here.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `purple:latest` | Model to use for chat |

## Comparison

| | Purple CLI | ollmcp | mcp-client-cli |
|---|---|---|---|
| Ollama integration | Native `/api/chat` | Ollama SDK | LangChain |
| MCP transport | STDIO (FastMCP) | STDIO | STDIO |
| Tool call fallback | XML parsing for broken JSON | None | None |
| Streaming | Yes | Yes | No |
| Context management | Sliding window trim | None | None |
| Config format | JSON | TOML | JSON |
| Lines of code | ~1200 | ~800 | ~400 |
| Document tools | Included (PDF, Excel, Word, PPTX) | None | None |
| Memory | Included (SQLite) | None | None |

These are solid projects. Purple's angle: it ships with useful tools, handles the XML fallback that open-weight models need, and uses Ollama's native API instead of the lossy OpenAI translation layer.

## Requirements

- Python 3.11+
- Ollama with a tool-calling model (Qwen3-Coder, Llama 3.x, Mistral, etc.)
- macOS or Linux

## Project Structure

```
.purple/
  cli/purple.py          # CLI + Ollama client + MCP orchestrator (600 lines)
  memory/server.py       # SQLite memory MCP server (150 lines)
  docs/server.py         # Document handling MCP server (445 lines)
  config/mcp.json        # MCP server declarations
  config/Modelfile       # Ollama model definition
  identity/identity.md   # System prompt (your config, not committed)
  requirements.txt       # Python dependencies
  pyproject.toml         # Package metadata
```

## License

MIT

## Built With

Built with [Claude Code](https://claude.ai/claude-code).
