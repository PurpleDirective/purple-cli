# Purple CLI

A local AI agent that connects Ollama to MCP tools. ~3000 lines of Python across 6 files. No frameworks, no OpenAI shims, no abstractions between you and the model.

![Purple CLI demo](demo.gif)

## What It Does

- Talks to Ollama via native `/api/chat` (not the OpenAI-compatible `/v1` endpoint)
- Calls MCP tools through FastMCP STDIO transport
- Loads MCP servers from a config file -- add your own in one line
- Streams output token-by-token as it generates
- Catches XML-formatted tool calls when models fall back from JSON (Qwen3-Coder workaround)
- Manages context with a sliding window that trims old messages automatically
- Injects a system prompt from a Markdown identity file
- Runs interactive or one-shot (`python purple.py "your prompt"`)

### New in v0.2

- **Session transcript logging** -- Every conversation saved as JSONL for review and verification
- **Persistent readline history** -- Up-arrow recall across sessions, tab-complete for `/commands`
- **Terminal dashboard** -- 256-color ANSI display with rounded boxes, progress bars, metrics
- **Proving Ground tier system** -- Track model performance (TCR, FTA, UOR), auto-promote/demote
- **Teaching knowledge server** -- Cloud-to-local knowledge transfer via MCP (9 tools)
- **Tool budget enforcement** -- Caps on sequential thinking and empty lookups to prevent tool loops
- **Code gen benchmarks** -- Canary tests that send tasks to Ollama, run pytest, log pass/fail
- **New commands:** `/knowledge`, `/history`, `/model`, `/sessions`

## Batteries Included

Ships with 3 MCP servers:

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

**purple-knowledge** -- Teaching knowledge base (9 tools)
- `lookup_knowledge` / `store_knowledge` -- Query and build a verified knowledge base
- `validate_teaching` / `import_queue` -- Process teaching fragments from cloud AI sessions
- `log_outcome` / `teaching_effectiveness` -- Track which knowledge actually helps
- `store_training_example` / `export_training_data` -- Accumulate fine-tuning data
- `list_domains` -- Browse knowledge categories

All servers enforce path validation -- file operations are restricted to the home directory and `/tmp`.

## Prerequisites

- **Python 3.11+** -- check with `python3 --version`
- **[Ollama](https://ollama.com)** -- install and start with `ollama serve`
- **A tool-calling model** -- e.g. `ollama pull qwen3-coder:30b` (or any model that supports function calling)

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

Memories are stored in a SQLite database at `~/.purple/memory/purple.db`. The database is created automatically on first use.

- Memories persist across sessions -- close Purple, reopen it, and your memories are still there
- Each memory has a type (`fact`, `preference`, `experience`, `correction`) and optional tags
- The database uses WAL mode for safe concurrent access
- File permissions are set to owner-only (chmod 600) automatically
- Search uses keyword matching (`recall_memories "search term"`)
- The database is excluded from git (in `.gitignore`) -- your memories stay on your machine

To back up your memories: `cp ~/.purple/memory/purple.db ~/.purple/memory/purple.db.backup`

To start fresh: delete `~/.purple/memory/purple.db` and it will be recreated on next run.

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

**`config/mcp.json`** -- Declares MCP servers. Each entry has a command array and an enabled flag. Created from `config/mcp.example.json` during setup (or auto-created on first run):

```json
{
  "servers": {
    "purple-memory": {
      "command": ["python3", "memory/server.py"],
      "enabled": true
    }
  }
}
```

Server commands run from the `~/.purple` directory. If you installed into a virtual environment, update the commands to use your venv's Python:

```json
"command": ["/home/you/.purple/venv/bin/python3", "memory/server.py"]
```

**`config/Modelfile`** -- Ollama model definition. Sets context window, temperature, and a system prompt with tool-use rules.

**`identity/identity.md`** -- System prompt loaded at startup. This is your model's personality. Write whatever you want here. Created from `identity/identity.example.md` during setup (or auto-created on first run).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `purple:latest` | Model to use for chat |

## Comparison

| | Purple CLI | [ollmcp](https://github.com/jonigl/mcp-client-for-ollama) | [mcp-client-cli](https://github.com/adhikasp/mcp-client-cli) |
|---|---|---|---|
| Ollama integration | Native `/api/chat` | Ollama SDK | Multi-provider (OpenAI, Groq, llama.cpp) |
| MCP transport | STDIO | STDIO, SSE, Streamable HTTP | STDIO, HTTP |
| Tool call fallback | XML parsing for broken JSON | None | None |
| Streaming | Yes | Yes | Yes |
| Context management | Sliding window trim | Keep Tokens + history export | Token tracking + session persistence |
| Bundled tools | 19 tools (memory, docs, knowledge) | None (bring your own) | None (bring your own) |
| Config format | JSON | TOML | JSON |
| Codebase | ~3000 lines, 6 files | Larger, full TUI | Smaller, multi-LLM focus |

These are mature, actively developed projects with larger communities. Purple's angle is different: it ships with useful tools out of the box, handles the XML fallback that open-weight models need, and keeps the entire codebase small enough to read in one sitting. If you want a full-featured TUI, use ollmcp. If you want multi-provider support, use mcp-client-cli. If you want something you can understand and hack on in an afternoon, use Purple.

## Troubleshooting

**"Could not open requirements file"** -- Make sure you're in the `~/.purple` directory: `cd ~/.purple` then `pip install -r requirements.txt`.

**"No module named 'fastmcp'"** -- Dependencies aren't installed. Run `pip install -r requirements.txt` inside your virtual environment.

**"Cannot connect to Ollama" on startup** -- Ollama isn't running. Start it with `ollama serve` or check that it's listening on `http://localhost:11434`.

**"Config not found: mcp.json"** -- Run `cp config/mcp.example.json config/mcp.json`, or let the CLI auto-create it on next run.

**MCP server won't start** -- If you installed in a venv, update `config/mcp.json` to use the full path to your venv's Python (e.g., `"/home/you/.purple/venv/bin/python3"`).

**"model 'purple:latest' not found"** -- Either create it with `ollama create purple -f config/Modelfile`, or set a different model: `OLLAMA_MODEL=qwen3-coder:30b python cli/purple.py`.

## Requirements

- Python 3.11+
- Ollama with a tool-calling model (Qwen3-Coder, Llama 3.x, Mistral, etc.)
- macOS or Linux

## Project Structure

```
~/.purple/
  cli/
    purple.py                # CLI + Ollama client + MCP orchestrator (~1008 lines)
    dashboard.py             # 256-color terminal display renderer (~289 lines)
    tracker.py               # SQLite tier tracker + metrics (~369 lines)
  memory/server.py           # SQLite memory MCP server (~556 lines)
  docs/server.py             # Document handling MCP server (~460 lines)
  knowledge/server.py        # Teaching knowledge base MCP server (~578 lines)
  eval/canary.py             # Code gen benchmarks (~248 lines)
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
