#!/bin/bash
# SessionEnd hook: commit shared memory to git + trim HANDOFF.md
# Called by Claude Code at session end. No output expected.

SHARED_DIR="$HOME/.claude/agent-memory/purple-shared"
HANDOFF="$SHARED_DIR/HANDOFF.md"
MAX_HANDOFF_LINES=200
LOG="$HOME/.purple/hooks/hook.log"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [session-end] $1" >> "$LOG"; }

# 1. Git commit shared memory
if [ -d "$SHARED_DIR/.git" ]; then
    cd "$SHARED_DIR"
    if [ -n "$(git status --porcelain)" ]; then
        git add -A
        git commit -m "auto: session end $(date +%Y-%m-%d-%H%M)" --quiet 2>/dev/null
        log "committed shared memory changes"
    else
        log "no changes to commit"
    fi
fi

# 2. Trim HANDOFF.md if over limit
if [ -f "$HANDOFF" ]; then
    LINES=$(wc -l < "$HANDOFF")
    if [ "$LINES" -gt "$MAX_HANDOFF_LINES" ]; then
        ARCHIVE="$SHARED_DIR/HANDOFF-ARCHIVE-$(date +%Y-%m-%d).md"
        cp "$HANDOFF" "$ARCHIVE"
        # Keep header (first 5 lines) + last 150 lines
        { head -5 "$HANDOFF"; echo ""; echo "**Trimmed $(date +%Y-%m-%d): entries archived to $(basename $ARCHIVE)**"; echo ""; tail -150 "$HANDOFF"; } > "$HANDOFF.tmp"
        mv "$HANDOFF.tmp" "$HANDOFF"
        log "trimmed HANDOFF.md from $LINES to ~158 lines"
    fi
fi
