# Purple CLI

A local AI agent with native inference + MCP tool calling. ~3800 lines of Python across 5 core files. No frameworks, no OpenAI shims, no abstractions between you and the model.

![Purple CLI demo](demo.gif)

## What It Does

- **Dual inference backends**: Ollama (`/api/chat`) for GGUF models, vllm-mlx (`/v1/chat`) for Apple Metal (auto-detected at startup)
- Calls MCP tools through FastMCP STDIO transport
- Loads MCP servers from a config file -- add your own in one line
- Streams output token-by-token as it generates
- Catches XML-formatted tool calls when models fall back from JSON (Qwen3-Coder workaround)
- Manages context with a sliding window that trims old messages automatically
- Injects current date + system prompt from a Markdown identity file
- Planning mode (`/plan`, `/build`) for structured problem-solving before coding
- Runs interactive or one-shot (`python purple.py "your prompt"`)

### New in v0.3

- **vllm-mlx backend** -- Apple Metal inference via MLX, 2x faster on MoE architectures. Auto-detected at startup.
- **Unified brain server** -- Memory, knowledge, web search, and teaching in one MCP server (8 tools). Replaces separate memory + knowledge servers.
- **Date-aware search** -- Current date injected into system prompt so web searches use the correct year.
- **Planning mode** -- `/plan` to think through a problem, `/build` to switch to execution.
- **Eval framework** -- V6 test battery (15 tests, 215 assertions) + delta tests for teaching artifacts.
- **Vega fine-tuning pipeline** -- Dataset builders, QLoRA training, abliteration, serving scripts.
- **SearXNG integration** -- Self-hosted multi-engine search (Docker on purpleroom).
- **Session hooks** -- Pre-compact context saver, secret scanner, session-end transcript archiver.

### New in v0.2

- **Session transcript logging** -- Every conversation saved as JSONL
- **Terminal dashboard** -- 256-color ANSI display with rounded boxes, progress bars, metrics
- **Proving Ground tier system** -- Track model performance (TCR, FTA, UOR), auto-promote/demote
- **Tool budget enforcement** -- Caps on sequential thinking and empty lookups
- **New commands:** `/knowledge`, `/history`, `/model`, `/sessions`

## Batteries Included

Ships with 3 MCP servers (15 tools):

**purple-brain** -- Unified memory + knowledge + search (8 tools)
- `store_memory` / `search` / `forget_memory` -- Persistent memory with hybrid search (FTS5 + vector similarity)
- `store_knowledge` / `log_outcome` -- Teaching knowledge base with effectiveness tracking
- `store_training_example` -- Accumulate fine-tuning data
- `web_search` / `fetch_page` -- Web search via SearXNG (multi-engine) with fallback to DuckDuckGo

**purple-docs** -- Document handling (6 tools)
- `read_pdf` -- Extract text and tables from PDFs (page ranges supported)
- `read_excel` -- Read spreadsheets (all sheets or specific)
- `create_excel` -- Build styled workbooks from JSON
- `create_word` -- Generate Word documents with headings, lists, tables
- `create_powerpoint` -- Create presentations with multiple layouts
- `list_directory` -- Browse files with size and type info

**sequential-thinking** -- Extended reasoning (1 tool, via NPX)
- Multi-step reasoning for complex problems with budget enforcement

All servers enforce path validation -- file operations are restricted to the home directory and `/tmp`.

## Prerequisites

- **Python 3.11+** -- check with `python3 --version`
- **[Ollama](https://ollama.com)** -- install and start with `ollama serve`
- **A tool-calling model** -- e.g. `ollama pull qwen3-coder:30b` (or any model that supports function calling)
- **Optional:** [vllm-mlx](https://github.com/vllm-project/vllm-mlx) for faster Apple Metal inference

## Quick Start

### One-command setup

```bash
git clone https://github.com/PurpleDirective/purple-cli.git ~/.purple
cd ~/.purple
bash setup.sh
source venv/bin/activate
python cli/purple.py
```

### Manual setup

```bash
# Clone
git clone https://github.com/PurpleDirective/purple-cli.git ~/.purple
cd ~/.purple

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up config files
cp config/mcp.example.json config/mcp.json
cp identity/identity.example.md identity/identity.md

# Create the default model (or use your own -- see below)
ollama create purple -f config/Modelfile

# Run
python cli/purple.py
```

> **Note:** The project lives at `~/.purple` (a hidden directory). Use `cd ~/.purple` to access it after cloning.

Or skip the custom model and use any tool-calling model directly:

```bash
OLLAMA_MODEL=qwen3-coder:30b python cli/purple.py "summarize this PDF"
```

## How Memory Works

Memories are stored in a SQLite database at `~/.purple/brain/purple_brain.db`. The database is created automatically on first use.

- Memories persist across sessions -- close Purple, reopen it, and your memories are still there
- Each memory has a type (`fact`, `preference`, `experience`, `correction`) and optional tags
- Search uses hybrid matching (FTS5 full-text + vector similarity via sentence embeddings)
- The database uses WAL mode for safe concurrent access
- File permissions are set to owner-only (chmod 600) automatically
- The database is excluded from git (in `.gitignore`) -- your memories stay on your machine

## Architecture

```
you --> Purple CLI --> Ollama /api/chat  --> your model (GGUF)
            |     \-> vllm-mlx /v1/chat --> your model (MLX)
            |
            +--> MCP Servers (STDIO) --> tools
                    |
                    +-- purple-brain (SQLite: memory + knowledge + search)
                    +-- purple-docs (PDF, Excel, Word, PowerPoint)
                    +-- sequential-thinking (extended reasoning)
                    +-- [your servers here]
```

The CLI is the orchestrator. It sends your message to the inference backend with tool definitions, gets back either text or tool calls, executes the tools via MCP, sends results back to the model, and repeats until the model responds with text. Up to 10 tool-call rounds per turn.

## Configuration

**`config/mcp.json`** -- Declares MCP servers. Each entry has a command array and an enabled flag. Created from `config/mcp.example.json` during setup (or auto-created on first run):

```json
{
  "servers": {
    "purple-brain": {
      "command": ["python3", "brain/server.py"],
      "enabled": true
    }
  }
}
```

Server commands run from the `~/.purple` directory. If you installed into a virtual environment, update the commands to use your venv's Python:

```json
"command": ["/home/you/.purple/venv/bin/python3", "brain/server.py"]
```

**`config/Modelfile`** -- Ollama model definition. Sets context window, temperature, and a system prompt with tool-use rules.

**`identity/identity.md`** -- System prompt loaded at startup. This is your model's personality. Write whatever you want here. Created from `identity/identity.example.md` during setup (or auto-created on first run).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PURPLE_BACKEND` | `auto` | Inference backend: `auto`, `ollama`, or `vllm-mlx` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `VLLM_URL` | `http://localhost:8000` | vllm-mlx server address |
| `OLLAMA_MODEL` | `purple:latest` | Model to use (auto-detected from vllm-mlx if available) |

## Comparison

| | Purple CLI | [ollmcp](https://github.com/jonigl/mcp-client-for-ollama) | [mcp-client-cli](https://github.com/adhikasp/mcp-client-cli) |
|---|---|---|---|
| Inference | Ollama + vllm-mlx (auto-detect) | Ollama SDK | Multi-provider (OpenAI, Groq, llama.cpp) |
| MCP transport | STDIO | STDIO, SSE, Streamable HTTP | STDIO, HTTP |
| Tool call fallback | XML parsing for broken JSON | None | None |
| Streaming | Yes | Yes | Yes |
| Context management | Sliding window trim | Keep Tokens + history export | Token tracking + session persistence |
| Bundled tools | 15 tools (brain, docs, thinking) | None (bring your own) | None (bring your own) |
| Web search | Built-in (SearXNG + DuckDuckGo) | None | None |
| Eval framework | V6 battery (15 tests, 215 assertions) | None | None |
| Config format | JSON | TOML | JSON |
| Codebase | ~3800 lines, 5 core files | Larger, full TUI | Smaller, multi-LLM focus |

These are mature, actively developed projects with larger communities. Purple's angle is different: it ships with useful tools out of the box, handles the XML fallback that open-weight models need, and keeps the entire codebase small enough to read in one sitting. If you want a full-featured TUI, use ollmcp. If you want multi-provider support, use mcp-client-cli. If you want something you can understand and hack on in an afternoon, use Purple.

## Troubleshooting

**"Could not open requirements file"** -- Make sure you're in the `~/.purple` directory: `cd ~/.purple` then `pip install -r requirements.txt`.

**"No module named 'fastmcp'"** -- Dependencies aren't installed. Run `pip install -r requirements.txt` inside your virtual environment.

**"Cannot connect to Ollama" on startup** -- Ollama isn't running. Start it with `ollama serve` or check that it's listening on `http://localhost:11434`.

**"Config not found: mcp.json"** -- Run `cp config/mcp.example.json config/mcp.json`, or let the CLI auto-create it on next run.

**MCP server won't start** -- If you installed in a venv, update `config/mcp.json` to use the full path to your venv's Python (e.g., `"/home/you/.purple/venv/bin/python3"`).

**"model 'purple:latest' not found"** -- Either create it with `ollama create purple -f config/Modelfile`, or set a different model: `OLLAMA_MODEL=qwen3-coder:30b python cli/purple.py`.

## Project Structure

```
~/.purple/
  cli/
    purple.py                # CLI + inference client + MCP orchestrator (~1498 lines)
    dashboard.py             # 256-color terminal display renderer (~295 lines)
    tracker.py               # SQLite tier tracker + metrics (~369 lines)
  brain/server.py            # Unified memory + knowledge + search MCP server (~1167 lines)
  docs/server.py             # Document handling MCP server (~459 lines)
  eval/
    v6_tests.py              # V6 eval battery (15 tests, 215 assertions)
    v6_runner.py             # V6 eval runner (sends tasks to model, runs pytest)
    delta_test.py            # Teaching artifact delta test (5 tests, 39 assertions)
    v5_runner.py             # V5 eval runner
    v4_runner.py             # V4 eval runner
  scripts/                   # Vega fine-tuning pipeline (dataset builders, training, serving)
  hooks/                     # Session hooks (pre-compact, scan-secrets, session-end)
  search/                    # SearXNG config (docker-compose + settings)
  config/mcp.json            # MCP server declarations (created from example)
  config/mcp.example.json    # Example config (committed to repo)
  config/Modelfile           # Ollama model definition
  identity/identity.md       # System prompt (created from example, not committed)
  identity/identity.example.md  # Example identity (committed to repo)
  sessions/                  # JSONL conversation transcripts (not committed)
  teaching/                  # Teaching fragment queue + compiled data (not committed)
  requirements.txt           # Python dependencies
  pyproject.toml             # Package metadata
  setup.sh                   # One-command setup script
```

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All tests run without Ollama -- external dependencies are mocked.

## License

MIT

## Built With

Built with [Claude Code](https://claude.ai/claude-code).
