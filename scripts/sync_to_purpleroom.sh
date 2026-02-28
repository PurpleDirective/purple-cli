#!/bin/bash
# sync_to_purpleroom.sh — Sync teaching data from purplemac to purpleroom
# Runs nightly at 23:45 (before purpleroom's midnight verification pipeline)

LOG=~/.purple/scripts/sync.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting sync..." >> "$LOG"

# Brain maintenance (backfill embeddings, import queue, export training data)
/Users/purple/.purple/venv/bin/python /Users/purple/.purple/brain/server.py --maintain \
  >> "$LOG" 2>&1 && echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Brain maintenance complete" >> "$LOG"

rsync -az ~/.purple/teaching/queue/ purpleroom:~/.purple/teaching/queue/ \
  && echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Fragments synced" >> "$LOG"

rsync -az ~/.purple/sessions/ purpleroom:~/.purple/sessions/ \
  && echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sessions synced" >> "$LOG"

ssh purpleroom "python3 ~/.purple/scripts/absorb_fragments.py >> ~/.purple/scripts/absorb.log 2>&1"
ssh purpleroom "python3 ~/.purple/scripts/absorb_sessions.py >> ~/.purple/scripts/absorb.log 2>&1"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync complete" >> "$LOG"
