#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Setting up Purple CLI..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
source venv/bin/activate
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Copy example configs if real configs don't exist
if [ ! -f "config/mcp.json" ]; then
    cp config/mcp.example.json config/mcp.json
    echo "Created config/mcp.json from example"
fi

if [ ! -f "identity/identity.md" ]; then
    cp identity/identity.example.md identity/identity.md
    echo "Created identity/identity.md from example"
fi

echo ""
echo "Setup complete. To run Purple CLI:"
echo ""
echo "  source ~/.purple/venv/bin/activate"
echo "  python ~/.purple/cli/purple.py"
echo ""
echo "Make sure Ollama is running with a tool-calling model."
echo "Default model: purple:latest"
echo "  Create it with:  ollama create purple -f ~/.purple/config/Modelfile"
echo "  Or use any model: OLLAMA_MODEL=qwen3-coder:30b python ~/.purple/cli/purple.py"
