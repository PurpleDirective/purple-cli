#!/bin/bash
# PreCompact hook: backup transcript before compaction destroys granularity
# Called by Claude Code before auto/manual compaction.
# Input on stdin: JSON with transcript_path, session_id, trigger

LOG="$HOME/.purple/hooks/hook.log"
TRANSCRIPT_DIR="$HOME/.purple/teaching/transcripts"
mkdir -p "$TRANSCRIPT_DIR"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [pre-compact] $1" >> "$LOG"; }

# Read stdin for transcript_path
INPUT=$(cat)
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null)
SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','unknown'))" 2>/dev/null)
TRIGGER=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('trigger','unknown'))" 2>/dev/null)

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
    DEST="$TRANSCRIPT_DIR/$(date +%Y%m%d-%H%M%S)-${SESSION_ID:0:8}-${TRIGGER}.jsonl"
    cp "$TRANSCRIPT_PATH" "$DEST"
    SIZE=$(wc -c < "$DEST")
    LINES=$(wc -l < "$DEST")
    log "saved transcript: $DEST ($LINES messages, $SIZE bytes, trigger=$TRIGGER)"
else
    log "no transcript_path or file not found (trigger=$TRIGGER, path=$TRANSCRIPT_PATH)"
fi
