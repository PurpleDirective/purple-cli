"""
Purple Memory Server
SQLite-backed persistent memory for Purple.
MCP server with 4 tools: store, recall, forget, list_recent.
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastmcp import FastMCP

DB_PATH = Path.home() / ".purple" / "memory" / "purple.db"

mcp = FastMCP("purple-memory")

_initialized = False


def get_db() -> sqlite3.Connection:
    global _initialized
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not _initialized:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('fact', 'preference', 'experience', 'correction')),
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        conn.commit()
        # Ensure DB file is owner-only (fix for world-readable 644 default)
        import stat
        db_stat = DB_PATH.stat()
        if db_stat.st_mode & 0o077:  # Has group or world permissions
            DB_PATH.chmod(0o600)
        _initialized = True
    return conn


def _format_row(row: sqlite3.Row, truncate: int = 200) -> str:
    tags = json.loads(row["tags"])
    tag_str = f" [{', '.join(tags)}]" if tags else ""
    return f"#{row['id']} [{row['type']}]{tag_str} {row['content'][:truncate]} ({row['updated_at'][:10]})"


# --- Core functions (callable directly for testing) ---

def _store_memory(content: str, type: str, tags: list[str] | None = None) -> str:
    if type not in ("fact", "preference", "experience", "correction"):
        return f"Error: type must be fact, preference, experience, or correction. Got: {type}"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO memories (content, type, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (content, type, json.dumps(tags or []), now, now),
        )
        conn.commit()
        return f"Stored memory #{cursor.lastrowid}: [{type}] {content[:80]}"
    finally:
        conn.close()


def _recall_memories(query: str, type: str | None = None, limit: int = 20) -> str:
    conn = get_db()
    try:
        params: list = [f"%{query}%"]
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        if type:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return f"No memories found matching '{query}'"
        results = [_format_row(r, truncate=500) for r in rows]
        return f"Found {len(results)} memories:\n" + "\n".join(results)
    finally:
        conn.close()


def _forget_memory(id: int, reason: str) -> str:
    conn = get_db()
    try:
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return f"Error: memory #{id} not found"
        conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        conn.commit()
        return f"Deleted memory #{id}: {row['content'][:80]}. Reason: {reason}"
    finally:
        conn.close()


def _list_recent(hours: int = 24, type: str | None = None) -> str:
    conn = get_db()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        if type:
            sql = "SELECT * FROM memories WHERE type = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 50"
            params: list = [type, cutoff]
        else:
            sql = "SELECT * FROM memories WHERE created_at >= ? ORDER BY created_at DESC LIMIT 50"
            params = [cutoff]
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return f"No memories in the last {hours} hours"
        return "\n".join(_format_row(r) for r in rows)
    finally:
        conn.close()


# --- MCP tool wrappers ---

@mcp.tool()
def store_memory(content: str, type: str, tags: list[str] | None = None) -> str:
    """Store a new memory. Type must be: fact, preference, experience, or correction. Never store passwords, account numbers, or sensitive personal information."""
    return _store_memory(content, type, tags)


@mcp.tool()
def recall_memories(query: str, type: str | None = None, limit: int = 20) -> str:
    """Search memories by keyword. Optionally filter by type. Returns up to limit results with full context."""
    return _recall_memories(query, type, limit)


@mcp.tool()
def forget_memory(id: int, reason: str) -> str:
    """Delete a memory by ID. Requires a reason for the deletion."""
    return _forget_memory(id, reason)


@mcp.tool()
def list_recent(hours: int = 24, type: str | None = None) -> str:
    """List memories created in the last N hours. Optionally filter by type."""
    return _list_recent(hours, type)


if __name__ == "__main__":
    mcp.run()
