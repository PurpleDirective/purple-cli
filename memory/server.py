"""
Purple Memory Kernel v2.0
SQLite + FTS5 + sqlite-vec hybrid memory system.

MCP server with tools: store, recall, search, forget, list_recent, stats.

Upgrade from v1.0:
- FTS5 full-text search replaces LIKE queries
- sqlite-vec for semantic vector search via nomic-embed-text embeddings
- Reciprocal Rank Fusion (RRF) combining text + vector results
- Agent attribution (who stored what)
- Expanded memory types
- Lazy embedding backfill for existing memories
"""

import sqlite3
import struct
import json
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path.home() / ".purple" / "memory" / "purple.db"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
RRF_K = 60  # Reciprocal Rank Fusion constant

VALID_TYPES = ("fact", "preference", "experience", "correction", "procedural", "pattern", "agent_output")

mcp = FastMCP("purple-memory")
_initialized = False
_vec_available = False


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def _try_load_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension. Returns True if successful."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except (ImportError, Exception):
        return False


def get_db() -> sqlite3.Connection:
    global _initialized, _vec_available
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if not _initialized:
        _vec_available = _try_load_vec(conn)

        # ----- v1.0 → v2.0 migration (runs before anything that needs new columns) -----

        # Step 1: Add columns missing from v1.0 schema
        for _col_sql in [
            "ALTER TABLE memories ADD COLUMN agent TEXT DEFAULT 'unknown'",
            "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'manual'",
        ]:
            try:
                conn.execute(_col_sql)
            except sqlite3.OperationalError:
                pass  # Column already exists, or table doesn't exist yet

        # Step 2: Remove restrictive CHECK constraint from v1.0 schema
        # v1.0 had: CHECK(type IN ('fact','preference','experience','correction'))
        # v2.0 validates types in Python and supports 7 types
        _table_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if _table_row and _table_row[0] and "CHECK" in _table_row[0].upper():
            conn.execute("""
                CREATE TABLE memories_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    agent TEXT DEFAULT 'unknown',
                    source TEXT DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT INTO memories_v2 (id, content, type, tags, agent, source, created_at, updated_at)
                SELECT id, content, type, tags,
                       COALESCE(agent, 'unknown'),
                       COALESCE(source, 'manual'),
                       created_at, updated_at
                FROM memories
            """)
            conn.execute("DROP TABLE memories")
            conn.execute("ALTER TABLE memories_v2 RENAME TO memories")

        # ----- End migration -----

        # Core memories table (created fresh only on brand-new installs)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                agent TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Indexes (safe now — agent column guaranteed to exist)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)")

        # FTS5 full-text search index
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                tags,
                content='memories',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS5 in sync with memories table
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
            END
        """)

        # Backfill FTS5 for any existing memories not yet indexed
        conn.execute("""
            INSERT OR IGNORE INTO memories_fts(rowid, content, tags)
            SELECT id, content, tags FROM memories
            WHERE id NOT IN (SELECT rowid FROM memories_fts)
        """)

        # Vector embedding table (only if sqlite-vec is available)
        if _vec_available:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(embedding float[{EMBED_DIM}])
            """)

        conn.commit()

        # Ensure DB file is owner-only
        if DB_PATH.exists():
            db_stat = DB_PATH.stat()
            if db_stat.st_mode & 0o077:
                DB_PATH.chmod(0o600)

        _initialized = True

    elif _vec_available:
        # Re-load extension on each connection (required for sqlite-vec)
        _try_load_vec(conn)

    return conn


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> Optional[list[float]]:
    """Generate embedding via Ollama's nomic-embed-text model."""
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if embeddings and len(embeddings[0]) == EMBED_DIM:
            return embeddings[0]
    except Exception:
        pass
    return None


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _embed_and_store(conn: sqlite3.Connection, memory_id: int, content: str):
    """Generate embedding and store in vector table. Fails silently."""
    if not _vec_available:
        return
    embedding = _get_embedding(content)
    if embedding:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                (memory_id, _serialize_f32(embedding)),
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def _fts_search(conn: sqlite3.Connection, query: str, type: Optional[str], limit: int) -> list[tuple[int, float]]:
    """Full-text search via FTS5. Returns list of (id, rank) tuples."""
    try:
        sql = """
            SELECT m.id, fts.rank
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [query]
        if type:
            sql += " AND m.type = ?"
            params.append(type)
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit * 2)  # Fetch extra for RRF merge
        return [(row["id"], row["rank"]) for row in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def _vec_search(conn: sqlite3.Connection, query: str, type: Optional[str], limit: int) -> list[tuple[int, float]]:
    """Vector similarity search via sqlite-vec. Returns list of (id, distance) tuples."""
    if not _vec_available:
        return []
    embedding = _get_embedding(query)
    if not embedding:
        return []
    try:
        sql = """
            SELECT v.rowid as id, v.distance
            FROM memories_vec v
            WHERE v.embedding MATCH ?
            AND k = ?
        """
        rows = conn.execute(sql, [_serialize_f32(embedding), limit * 2]).fetchall()
        if type:
            # Post-filter by type (vec0 doesn't support JOIN in MATCH queries)
            type_ids = {row["id"] for row in conn.execute(
                "SELECT id FROM memories WHERE type = ?", (type,)
            ).fetchall()}
            rows = [r for r in rows if r["id"] in type_ids]
        return [(row["id"], row["distance"]) for row in rows]
    except Exception:
        return []


def _hybrid_search(conn: sqlite3.Connection, query: str, type: Optional[str] = None, limit: int = 20) -> list[int]:
    """Reciprocal Rank Fusion combining FTS5 + vector search results."""
    fts_results = _fts_search(conn, query, type, limit)
    vec_results = _vec_search(conn, query, type, limit)

    # If only one source has results, use it directly
    if not fts_results and not vec_results:
        return []
    if not vec_results:
        return [id for id, _ in fts_results[:limit]]
    if not fts_results:
        return [id for id, _ in vec_results[:limit]]

    # RRF: score = sum(1 / (k + rank)) across sources
    scores: dict[int, float] = {}
    for rank, (id, _) in enumerate(fts_results):
        scores[id] = scores.get(id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, (id, _) in enumerate(vec_results):
        scores[id] = scores.get(id, 0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return sorted_ids[:limit]


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _format_row(row: sqlite3.Row, truncate: int = 500) -> str:
    try:
        tags = json.loads(row["tags"])
        tag_str = f" [{', '.join(str(t) for t in tags)}]" if isinstance(tags, list) and tags else ""
    except (json.JSONDecodeError, TypeError):
        tag_str = ""
    agent_str = f" @{row['agent']}" if row["agent"] and row["agent"] != "unknown" else ""
    return f"#{row['id']} [{row['type']}]{tag_str}{agent_str} {row['content'][:truncate]} ({row['updated_at'][:10]})"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _store_memory(content: str, type: str, tags: list[str] | None = None,
                  agent: str = "unknown", source: str = "manual") -> str:
    if type not in VALID_TYPES:
        return f"Error: type must be one of {VALID_TYPES}. Got: {type}"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO memories (content, type, tags, agent, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, type, json.dumps(tags or []), agent, source, now, now),
        )
        memory_id = cursor.lastrowid
        conn.commit()

        # Generate and store embedding (async-friendly, non-blocking)
        _embed_and_store(conn, memory_id, content)
        conn.commit()

        return f"Stored memory #{memory_id}: [{type}] {content[:80]}"
    finally:
        conn.close()


def _search_memories(query: str, type: str | None = None, limit: int = 20) -> str:
    """Hybrid search: FTS5 + vector similarity with RRF fusion."""
    conn = get_db()
    try:
        result_ids = _hybrid_search(conn, query, type, limit)
        if not result_ids:
            # Fallback to simple LIKE search
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params: list = [f"%{escaped}%"]
            sql = "SELECT * FROM memories WHERE content LIKE ? ESCAPE '\\'"
            if type:
                sql += " AND type = ?"
                params.append(type)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return f"No memories found matching '{query}'"
            return f"Found {len(rows)} memories (keyword search):\n" + "\n".join(_format_row(r) for r in rows)

        # Fetch full rows in RRF order
        placeholders = ",".join("?" * len(result_ids))
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", result_ids
        ).fetchall()
        row_map = {row["id"]: row for row in rows}
        ordered = [row_map[id] for id in result_ids if id in row_map]

        search_type = "hybrid" if _vec_available else "full-text"
        return f"Found {len(ordered)} memories ({search_type} search):\n" + "\n".join(_format_row(r) for r in ordered)
    finally:
        conn.close()


def _recall_memories(query: str, type: str | None = None, limit: int = 20) -> str:
    """Backward-compatible recall — now uses hybrid search internally."""
    return _search_memories(query, type, limit)


def _forget_memory(id: int, reason: str) -> str:
    conn = get_db()
    try:
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return f"Error: memory #{id} not found"
        conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        # Clean up vector embedding
        if _vec_available:
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (id,))
            except Exception:
                pass
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


def _get_stats() -> str:
    """Return memory kernel statistics."""
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        by_type = conn.execute(
            "SELECT type, COUNT(*) as c FROM memories GROUP BY type ORDER BY c DESC"
        ).fetchall()
        by_agent = conn.execute(
            "SELECT agent, COUNT(*) as c FROM memories GROUP BY agent ORDER BY c DESC"
        ).fetchall()

        embedded = 0
        if _vec_available:
            try:
                embedded = conn.execute("SELECT COUNT(*) as c FROM memories_vec").fetchone()["c"]
            except Exception:
                pass

        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        db_size_mb = db_size / (1024 * 1024)

        lines = [
            f"Purple Memory Kernel v2.0",
            f"Total memories: {total}",
            f"Embedded: {embedded}/{total} ({embedded*100//max(total,1)}%)",
            f"Database size: {db_size_mb:.1f} MB",
            f"Vector search: {'enabled' if _vec_available else 'disabled (install sqlite-vec)'}",
            f"Embedding model: {EMBED_MODEL} ({EMBED_DIM}d)",
            "",
            "By type:",
        ]
        for row in by_type:
            lines.append(f"  {row['type']}: {row['c']}")
        lines.append("")
        lines.append("By agent:")
        for row in by_agent:
            lines.append(f"  {row['agent']}: {row['c']}")

        return "\n".join(lines)
    finally:
        conn.close()


def _backfill_embeddings(batch_size: int = 50) -> str:
    """Generate embeddings for memories that don't have them yet."""
    if not _vec_available:
        return "Vector search not available (sqlite-vec not installed)"

    conn = get_db()
    try:
        existing = {row[0] for row in conn.execute("SELECT rowid FROM memories_vec").fetchall()}
        all_ids = [row["id"] for row in conn.execute("SELECT id FROM memories").fetchall()]
        missing = [id for id in all_ids if id not in existing]

        if not missing:
            return f"All {len(all_ids)} memories already have embeddings"

        batch = missing[:batch_size]
        embedded = 0
        for memory_id in batch:
            row = conn.execute("SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                _embed_and_store(conn, memory_id, row["content"])
                embedded += 1

        conn.commit()
        remaining = len(missing) - embedded
        return f"Embedded {embedded} memories. {remaining} remaining."
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

@mcp.tool()
def store_memory(content: str, type: str, tags: list[str] | None = None,
                 agent: str = "unknown", source: str = "manual") -> str:
    """Store a new memory with semantic embedding.
    Type: fact, preference, experience, correction, procedural, pattern, agent_output.
    Agent: who is storing (evolution, improvement, keenness, admin, human, etc.).
    Source: manual, compiled, agent_output.
    Never store passwords, account numbers, or sensitive personal information."""
    return _store_memory(content, type, tags, agent, source)


@mcp.tool()
def search(query: str, type: str | None = None, limit: int = 20) -> str:
    """Hybrid search: combines full-text (FTS5) and semantic (vector) search.
    Finds memories by meaning, not just keywords. Best for natural language queries."""
    return _search_memories(query, type, limit)


@mcp.tool()
def recall_memories(query: str, type: str | None = None, limit: int = 20) -> str:
    """Search memories by keyword or meaning. Backward-compatible with v1.0.
    Now powered by hybrid FTS5 + vector search internally."""
    return _recall_memories(query, type, limit)


@mcp.tool()
def forget_memory(id: int, reason: str) -> str:
    """Delete a memory by ID. Requires a reason for the deletion."""
    return _forget_memory(id, reason)


@mcp.tool()
def list_recent(hours: int = 24, type: str | None = None) -> str:
    """List memories created in the last N hours. Optionally filter by type."""
    return _list_recent(hours, type)


@mcp.tool()
def stats() -> str:
    """Show memory kernel statistics: total memories, types, agents, embedding coverage, database size."""
    return _get_stats()


@mcp.tool()
def backfill_embeddings(batch_size: int = 50) -> str:
    """Generate vector embeddings for memories that don't have them yet.
    Run this periodically to ensure all memories are searchable by meaning.
    Processes up to batch_size memories per call."""
    return _backfill_embeddings(batch_size)


if __name__ == "__main__":
    mcp.run()
