"""
Purple Brain v1.0
Unified MCP server: memory + knowledge + web search.

Merges purple-memory (v2.0), purple-knowledge (v1.1), purple-search (v1.0)
into 8 model-facing tools + CLI admin commands.

MCP tools:  store_memory, search, forget_memory, store_knowledge,
            log_outcome, store_training_example, web_search, fetch_page

CLI admin:  --stats, --recent, --domains, --validate, --backfill,
            --import-queue, --effectiveness, --export, --status, --maintain
"""

import argparse
import asyncio
import json
import re
import sqlite3
import struct
import sys
from datetime import datetime, timezone, timedelta
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import httpx
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_DB_PATH = Path.home() / ".purple" / "memory" / "purple.db"
KNOWLEDGE_DB_PATH = Path.home() / ".purple" / "knowledge" / "knowledge.db"
QUEUE_PATH = Path.home() / ".purple" / "teaching" / "queue"
TRAINING_DATA_PATH = Path.home() / ".purple" / "teaching" / "finetune" / "training.jsonl"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
RRF_K = 60

SEARXNG_URL = "http://100.89.41.72:8890"
DDG_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
HTTP_TIMEOUT = 15.0

MEMORY_TYPES = ("fact", "preference", "experience", "correction",
                "procedural", "pattern", "agent_output")
KNOWLEDGE_CONFIDENCE = ("high", "moderate", "low")
KNOWLEDGE_TYPES = ("principle", "procedure", "correction", "tool-recommendation")

mcp = FastMCP("purple-brain")

# ---------------------------------------------------------------------------
# Memory database initialization
# ---------------------------------------------------------------------------

_mem_initialized = False
_vec_available = False


def _try_load_vec(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[brain] sqlite-vec load error: {e}", file=sys.stderr)
        return False


def get_memory_db() -> sqlite3.Connection:
    global _mem_initialized, _vec_available
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if not _mem_initialized:
        _vec_available = _try_load_vec(conn)

        # v1.0 → v2.0 migration
        for col_sql in [
            "ALTER TABLE memories ADD COLUMN agent TEXT DEFAULT 'unknown'",
            "ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'manual'",
        ]:
            try:
                conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass

        table_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if table_row and table_row[0] and "CHECK" in table_row[0].upper():
            conn.execute("""
                CREATE TABLE memories_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL, type TEXT NOT NULL,
                    tags TEXT DEFAULT '[]', agent TEXT DEFAULT 'unknown',
                    source TEXT DEFAULT 'manual',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT INTO memories_v2 (id, content, type, tags, agent, source, created_at, updated_at)
                SELECT id, content, type, tags, COALESCE(agent, 'unknown'),
                       COALESCE(source, 'manual'), created_at, updated_at FROM memories
            """)
            conn.execute("DROP TABLE memories")
            conn.execute("ALTER TABLE memories_v2 RENAME TO memories")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL, type TEXT NOT NULL,
                tags TEXT DEFAULT '[]', agent TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'manual',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)")

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, content='memories', content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
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
        conn.execute("""
            INSERT OR IGNORE INTO memories_fts(rowid, content, tags)
            SELECT id, content, tags FROM memories
            WHERE id NOT IN (SELECT rowid FROM memories_fts)
        """)

        if _vec_available:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(embedding float[{EMBED_DIM}])
            """)

        conn.commit()
        if MEMORY_DB_PATH.exists() and MEMORY_DB_PATH.stat().st_mode & 0o077:
            MEMORY_DB_PATH.chmod(0o600)
        _mem_initialized = True

    elif _vec_available:
        _try_load_vec(conn)

    return conn


# ---------------------------------------------------------------------------
# Knowledge database initialization
# ---------------------------------------------------------------------------

_know_initialized = False


def get_knowledge_db() -> sqlite3.Connection:
    global _know_initialized
    KNOWLEDGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(KNOWLEDGE_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    if not _know_initialized:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
                when_to_apply TEXT DEFAULT '', anti_pattern TEXT DEFAULT '',
                confidence TEXT NOT NULL DEFAULT 'moderate',
                source_agent TEXT DEFAULT 'unknown',
                type TEXT NOT NULL DEFAULT 'principle',
                independence_test TEXT DEFAULT '',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge(source_agent)")

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                title, content, when_to_apply, domain,
                content='knowledge', content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, title, content, when_to_apply, domain)
                VALUES (new.id, new.title, new.content, new.when_to_apply, new.domain);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, when_to_apply, domain)
                VALUES ('delete', old.id, old.title, old.content, old.when_to_apply, old.domain);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, when_to_apply, domain)
                VALUES ('delete', old.id, old.title, old.content, old.when_to_apply, old.domain);
                INSERT INTO knowledge_fts(rowid, title, content, when_to_apply, domain)
                VALUES (new.id, new.title, new.content, new.when_to_apply, new.domain);
            END
        """)
        conn.execute("""
            INSERT OR IGNORE INTO knowledge_fts(rowid, title, content, when_to_apply, domain)
            SELECT id, title, content, when_to_apply, domain FROM knowledge
            WHERE id NOT IN (SELECT rowid FROM knowledge_fts)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('success', 'failure', 'partial')),
                fragments_used TEXT DEFAULT '[]', notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_result ON outcomes(result)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_prompt TEXT NOT NULL, user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL, domain TEXT DEFAULT 'general',
                quality TEXT DEFAULT 'unreviewed'
                    CHECK(quality IN ('unreviewed', 'approved', 'rejected')),
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        if KNOWLEDGE_DB_PATH.exists() and KNOWLEDGE_DB_PATH.stat().st_mode & 0o077:
            KNOWLEDGE_DB_PATH.chmod(0o600)
        _know_initialized = True

    return conn


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> Optional[list[float]]:
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
    return struct.pack(f"{len(vec)}f", *vec)


def _embed_and_store(conn: sqlite3.Connection, memory_id: int, content: str):
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
# Memory search internals
# ---------------------------------------------------------------------------

def _mem_fts_search(conn: sqlite3.Connection, query: str, mem_type: Optional[str], limit: int) -> list[tuple[int, float]]:
    try:
        sql = """
            SELECT m.id, fts.rank FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid WHERE memories_fts MATCH ?
        """
        params: list = [query]
        if mem_type:
            sql += " AND m.type = ?"
            params.append(mem_type)
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit * 2)
        return [(row["id"], row["rank"]) for row in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def _mem_vec_search(conn: sqlite3.Connection, query: str, mem_type: Optional[str], limit: int) -> list[tuple[int, float]]:
    if not _vec_available:
        return []
    embedding = _get_embedding(query)
    if not embedding:
        return []
    try:
        rows = conn.execute(
            "SELECT v.rowid as id, v.distance FROM memories_vec v WHERE v.embedding MATCH ? AND k = ?",
            [_serialize_f32(embedding), limit * 2],
        ).fetchall()
        if mem_type:
            type_ids = {row["id"] for row in conn.execute(
                "SELECT id FROM memories WHERE type = ?", (mem_type,)
            ).fetchall()}
            rows = [r for r in rows if r["id"] in type_ids]
        return [(row["id"], row["distance"]) for row in rows]
    except Exception:
        return []


def _mem_hybrid_search(conn: sqlite3.Connection, query: str, mem_type: Optional[str] = None, limit: int = 20) -> list[int]:
    fts_results = _mem_fts_search(conn, query, mem_type, limit)
    vec_results = _mem_vec_search(conn, query, mem_type, limit)
    if not fts_results and not vec_results:
        return []
    if not vec_results:
        return [id for id, _ in fts_results[:limit]]
    if not fts_results:
        return [id for id, _ in vec_results[:limit]]
    scores: dict[int, float] = {}
    for rank, (id, _) in enumerate(fts_results):
        scores[id] = scores.get(id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, (id, _) in enumerate(vec_results):
        scores[id] = scores.get(id, 0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Knowledge search internal
# ---------------------------------------------------------------------------

def _know_fts_search(conn: sqlite3.Connection, query: str, domain: Optional[str], limit: int) -> list[sqlite3.Row]:
    try:
        sql = """
            SELECT k.*, fts.rank FROM knowledge_fts fts
            JOIN knowledge k ON k.id = fts.rowid WHERE knowledge_fts MATCH ?
        """
        params: list = [query]
        if domain:
            sql += " AND k.domain = ?"
            params.append(domain)
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit)
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _know_like_search(conn: sqlite3.Connection, query: str, domain: Optional[str], limit: int) -> list[sqlite3.Row]:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    sql = "SELECT * FROM knowledge WHERE (content LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\')"
    params: list = [f"%{escaped}%", f"%{escaped}%"]
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Unified search + cross-source merge
# ---------------------------------------------------------------------------

def _search_memory_results(query: str, limit: int = 20) -> list[dict]:
    conn = get_memory_db()
    try:
        result_ids = _mem_hybrid_search(conn, query, None, limit)
        if not result_ids:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? ESCAPE '\\' ORDER BY updated_at DESC LIMIT ?",
                [f"%{escaped}%", limit],
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(result_ids))
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", result_ids
            ).fetchall()
            row_map = {row["id"]: row for row in rows}
            rows = [row_map[id] for id in result_ids if id in row_map]

        results = []
        for rank, row in enumerate(rows):
            try:
                tags = json.loads(row["tags"])
                tag_str = f" [{', '.join(str(t) for t in tags)}]" if isinstance(tags, list) and tags else ""
            except (json.JSONDecodeError, TypeError):
                tag_str = ""
            agent_str = f" @{row['agent']}" if row["agent"] and row["agent"] != "unknown" else ""
            results.append({
                "source": "memory", "rank": rank, "id": row["id"],
                "text": f"#{row['id']} [{row['type']}]{tag_str}{agent_str} {row['content'][:500]} ({row['updated_at'][:10]})",
            })
        return results
    finally:
        conn.close()


def _search_knowledge_results(query: str, domain: Optional[str] = None, limit: int = 20) -> list[dict]:
    conn = get_knowledge_db()
    try:
        rows = _know_fts_search(conn, query, domain, limit)
        if not rows:
            rows = _know_like_search(conn, query, domain, limit)

        results = []
        for rank, row in enumerate(rows):
            results.append({
                "source": "knowledge", "rank": rank, "id": row["id"],
                "text": (
                    f"#{row['id']} [{row['domain']}] {row['title']}\n"
                    f"  Confidence: {row['confidence']} | Type: {row['type']} | Agent: {row['source_agent']}\n"
                    f"  {row['content'][:300]}{'...' if len(row['content']) > 300 else ''}\n"
                    f"  When: {row['when_to_apply'][:150]}{'...' if len(row['when_to_apply']) > 150 else ''}"
                ),
            })
        return results
    finally:
        conn.close()


def _merge_results(mem_results: list[dict], know_results: list[dict], limit: int) -> list[dict]:
    scores: dict[tuple, float] = {}
    all_items: dict[tuple, dict] = {}
    for r in mem_results:
        key = ("memory", r["id"])
        scores[key] = 1.0 / (RRF_K + r["rank"] + 1)
        all_items[key] = r
    for r in know_results:
        key = ("knowledge", r["id"])
        scores[key] = 1.0 / (RRF_K + r["rank"] + 1)
        all_items[key] = r
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [all_items[k] for k in sorted_keys[:limit]]


def _unified_search(query: str, scope: str = "all", limit: int = 20) -> str:
    mem_results = []
    know_results = []

    if scope in ("all", "memory"):
        mem_results = _search_memory_results(query, limit)
    if scope in ("all", "knowledge"):
        know_results = _search_knowledge_results(query, None, limit)

    if not mem_results and not know_results:
        return f"No results found matching '{query}'"

    if scope == "all" and mem_results and know_results:
        merged = _merge_results(mem_results, know_results, limit)
    else:
        merged = (mem_results + know_results)[:limit]

    mc = sum(1 for r in merged if r["source"] == "memory")
    kc = sum(1 for r in merged if r["source"] == "knowledge")
    parts = []
    if mc:
        parts.append(f"{mc} memories")
    if kc:
        parts.append(f"{kc} knowledge fragments")
    header = f"Found {len(merged)} results ({', '.join(parts)}):\n"

    lines = [header]
    for r in merged:
        label = "[MEMORY]" if r["source"] == "memory" else "[KNOWLEDGE]"
        lines.append(f"{label} {r['text']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory core functions
# ---------------------------------------------------------------------------

def _store_memory(content: str, type: str, tags: list[str] | None = None,
                  agent: str = "unknown", source: str = "manual") -> str:
    if type not in MEMORY_TYPES:
        return f"Error: type must be one of {MEMORY_TYPES}. Got: {type}"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_memory_db()
    try:
        cursor = conn.execute(
            "INSERT INTO memories (content, type, tags, agent, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, type, json.dumps(tags or []), agent, source, now, now),
        )
        memory_id = cursor.lastrowid
        conn.commit()
        _embed_and_store(conn, memory_id, content)
        conn.commit()
        return f"Stored memory #{memory_id}: [{type}] {content[:80]}"
    finally:
        conn.close()


def _forget_memory(id: int, reason: str) -> str:
    conn = get_memory_db()
    try:
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (id,)).fetchone()
        if not row:
            return f"Error: memory #{id} not found"
        conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        if _vec_available:
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (id,))
            except Exception:
                pass
        conn.commit()
        return f"Deleted memory #{id}: {row['content'][:80]}. Reason: {reason}"
    finally:
        conn.close()


def _list_recent(hours: int = 24, mem_type: Optional[str] = None) -> str:
    conn = get_memory_db()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        if mem_type:
            rows = conn.execute(
                "SELECT * FROM memories WHERE type = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 50",
                [mem_type, cutoff],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories WHERE created_at >= ? ORDER BY created_at DESC LIMIT 50",
                [cutoff],
            ).fetchall()
        if not rows:
            return f"No memories in the last {hours} hours"
        lines = []
        for row in rows:
            try:
                tags = json.loads(row["tags"])
                tag_str = f" [{', '.join(str(t) for t in tags)}]" if isinstance(tags, list) and tags else ""
            except (json.JSONDecodeError, TypeError):
                tag_str = ""
            agent_str = f" @{row['agent']}" if row["agent"] and row["agent"] != "unknown" else ""
            lines.append(f"#{row['id']} [{row['type']}]{tag_str}{agent_str} {row['content'][:500]} ({row['updated_at'][:10]})")
        return "\n".join(lines)
    finally:
        conn.close()


def _get_memory_stats() -> str:
    conn = get_memory_db()
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
        db_size = MEMORY_DB_PATH.stat().st_size if MEMORY_DB_PATH.exists() else 0
        lines = [
            f"Total memories: {total}",
            f"Embedded: {embedded}/{total} ({embedded * 100 // max(total, 1)}%)",
            f"Database: {db_size / (1024 * 1024):.1f} MB",
            f"Vector search: {'enabled' if _vec_available else 'disabled'}",
            "", "By type:",
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
    conn = get_memory_db()
    if not _vec_available:
        conn.close()
        return "Vector search not available (sqlite-vec not installed)"
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
# Knowledge core functions
# ---------------------------------------------------------------------------

def _parse_fragment_file(path: Path) -> dict | None:
    text = path.read_text()
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
    if not match:
        return None
    meta = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    sections = {}
    current_section = None
    current_content: list[str] = []
    for line in match.group(2).split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    title = content = None
    when_to_apply = anti_pattern = ""
    for key, value in sections.items():
        lower = key.lower()
        if lower == 'when to apply':
            when_to_apply = value
        elif lower == 'anti-pattern':
            anti_pattern = value
        elif not title:
            title = key
            content = value
    if not title or not content:
        return None
    return {
        'domain': meta.get('domain', 'general'), 'title': title,
        'content': content, 'when_to_apply': when_to_apply,
        'anti_pattern': anti_pattern,
        'confidence': meta.get('confidence', 'moderate'),
        'source_agent': meta.get('source_agent', 'unknown'),
        'ktype': meta.get('type', 'principle'),
        'independence_test': meta.get('independence_test', ''),
    }


def _store_knowledge(domain: str, title: str, content: str,
                     when_to_apply: str = "", anti_pattern: str = "",
                     confidence: str = "moderate", source_agent: str = "unknown",
                     ktype: str = "principle", independence_test: str = "") -> str:
    if confidence not in KNOWLEDGE_CONFIDENCE:
        return f"Error: confidence must be one of {KNOWLEDGE_CONFIDENCE}. Got: {confidence}"
    if ktype not in KNOWLEDGE_TYPES:
        return f"Error: type must be one of {KNOWLEDGE_TYPES}. Got: {ktype}"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_knowledge_db()
    try:
        cursor = conn.execute(
            """INSERT INTO knowledge
               (domain, title, content, when_to_apply, anti_pattern,
                confidence, source_agent, type, independence_test, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (domain, title, content, when_to_apply, anti_pattern,
             confidence, source_agent, ktype, independence_test, now, now)
        )
        conn.commit()
        return f"Stored knowledge #{cursor.lastrowid}: [{domain}] {title}"
    finally:
        conn.close()


def _list_domains() -> str:
    conn = get_knowledge_db()
    try:
        rows = conn.execute(
            "SELECT domain, COUNT(*) as count, GROUP_CONCAT(DISTINCT confidence) as confidences "
            "FROM knowledge GROUP BY domain ORDER BY count DESC"
        ).fetchall()
        if not rows:
            return "No knowledge domains yet."
        total = sum(r['count'] for r in rows)
        lines = [f"Knowledge base: {total} fragments across {len(rows)} domains\n"]
        for r in rows:
            lines.append(f"  {r['domain']}: {r['count']} fragments ({r['confidences']})")
        return "\n".join(lines)
    finally:
        conn.close()


def _validate_teaching(id: int) -> str:
    conn = get_knowledge_db()
    try:
        row = conn.execute("SELECT * FROM knowledge WHERE id = ?", (id,)).fetchone()
        if not row:
            return f"Error: knowledge #{id} not found"
        issues = []
        if not row['content'].strip():
            issues.append("Empty content")
        if not row['when_to_apply'].strip():
            issues.append("Missing 'when to apply' section")
        if not row['anti_pattern'].strip():
            issues.append("Missing anti-pattern section")
        if not row['independence_test'].strip():
            issues.append("Missing independence test")
        if len(row['content']) > 2000:
            issues.append(f"Content too long ({len(row['content'])} chars, recommend <2000)")
        if issues:
            return f"#{id} '{row['title']}' has {len(issues)} issue(s):\n" + "\n".join(f"  - {i}" for i in issues)
        return f"#{id} '{row['title']}' passes validation."
    finally:
        conn.close()


def _import_queue() -> str:
    if not QUEUE_PATH.exists():
        return "Queue directory does not exist."
    files = sorted(QUEUE_PATH.glob("*.md"))
    if not files:
        return "No fragments in queue."
    results = []
    for f in files:
        parsed = _parse_fragment_file(f)
        if parsed:
            result = _store_knowledge(**parsed)
            results.append(f"  {f.name}: {result}")
        else:
            results.append(f"  {f.name}: Error — could not parse fragment")
    return f"Processed {len(files)} fragments:\n" + "\n".join(results)


def _log_outcome(task_description: str, result: str,
                 fragments_used: list[int] | None = None, notes: str = "") -> str:
    if result not in ("success", "failure", "partial"):
        return f"Error: result must be success, failure, or partial. Got: {result}"
    now = datetime.now(timezone.utc).isoformat()
    conn = get_knowledge_db()
    try:
        conn.execute(
            "INSERT INTO outcomes (task_description, result, fragments_used, notes, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_description, result, json.dumps(fragments_used or []), notes, now)
        )
        conn.commit()
        return f"Logged outcome: [{result}] {task_description[:80]}"
    finally:
        conn.close()


def _teaching_effectiveness() -> str:
    conn = get_knowledge_db()
    try:
        total = conn.execute("SELECT COUNT(*) as c FROM outcomes").fetchone()["c"]
        if total == 0:
            return "No outcomes logged yet."
        by_result = conn.execute(
            "SELECT result, COUNT(*) as c FROM outcomes GROUP BY result"
        ).fetchall()
        result_map = {r["result"]: r["c"] for r in by_result}
        success = result_map.get("success", 0)
        failure = result_map.get("failure", 0)
        partial = result_map.get("partial", 0)
        success_rate = success * 100 // max(total, 1)

        all_outcomes = conn.execute("SELECT * FROM outcomes").fetchall()
        fragment_success: dict[int, list[int]] = {}
        for outcome in all_outcomes:
            try:
                fids = json.loads(outcome["fragments_used"])
            except (json.JSONDecodeError, TypeError):
                fids = []
            for fid in fids:
                if fid not in fragment_success:
                    fragment_success[fid] = [0, 0]
                fragment_success[fid][1] += 1
                if outcome["result"] == "success":
                    fragment_success[fid][0] += 1

        lines = [
            "Teaching Effectiveness Report",
            f"Total outcomes: {total}",
            f"  Success: {success} ({success_rate}%)",
            f"  Partial: {partial}",
            f"  Failure: {failure}", "",
        ]
        if fragment_success:
            lines.append("Fragment effectiveness (2+ tasks):")
            for fid, (s, t) in sorted(fragment_success.items(), key=lambda x: x[1][1], reverse=True):
                if t >= 2:
                    row = conn.execute("SELECT title FROM knowledge WHERE id = ?", (fid,)).fetchone()
                    title = row["title"] if row else f"(deleted #{fid})"
                    lines.append(f"  #{fid} {title}: {s}/{t} ({s * 100 // max(t, 1)}%)")

        training_count = conn.execute("SELECT COUNT(*) as c FROM training_data").fetchone()["c"]
        approved = conn.execute("SELECT COUNT(*) as c FROM training_data WHERE quality = 'approved'").fetchone()["c"]
        lines.append(f"\nTraining data: {training_count} examples ({approved} approved)")
        lines.append(f"Fine-tuning threshold: 1,000 (current: {approved})")
        return "\n".join(lines)
    finally:
        conn.close()


def _store_training_example(system_prompt: str, user_message: str,
                            assistant_response: str, domain: str = "general") -> str:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_knowledge_db()
    try:
        cursor = conn.execute(
            "INSERT INTO training_data (system_prompt, user_message, assistant_response, domain, created_at) VALUES (?, ?, ?, ?, ?)",
            (system_prompt, user_message, assistant_response, domain, now)
        )
        conn.commit()
        return f"Stored training example #{cursor.lastrowid} [{domain}]"
    finally:
        conn.close()


def _export_training_data() -> str:
    conn = get_knowledge_db()
    try:
        rows = conn.execute(
            "SELECT * FROM training_data WHERE quality = 'approved' ORDER BY id"
        ).fetchall()
        if not rows:
            rows = conn.execute("SELECT * FROM training_data ORDER BY id").fetchall()
        if not rows:
            return "No training data to export."
        TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_DATA_PATH, "w") as f:
            for row in rows:
                entry = {"messages": [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_message"]},
                    {"role": "assistant", "content": row["assistant_response"]},
                ]}
                f.write(json.dumps(entry) + "\n")
        return f"Exported {len(rows)} examples to {TRAINING_DATA_PATH}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Web search functions
# ---------------------------------------------------------------------------

async def _searxng_search(query: str, num_results: int = 10) -> list[dict] | None:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as http:
            resp = await http.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json",
                        "engines": "google,bing,duckduckgo,brave", "language": "en"},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for r in data.get("results", [])[:num_results]:
                results.append({
                    "title": r.get("title", ""), "url": r.get("url", ""),
                    "snippet": r.get("content", ""), "source": "searxng",
                })
            return results if results else None
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return None


def _extract_ddg_results(html: str) -> list[dict]:
    results = []
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
    blocks = re.split(r'<div[^>]+class="[^"]*result[^"]*results_links[^"]*"', html)
    for block in blocks[1:]:
        link_match = link_pattern.search(block)
        snippet_match = snippet_pattern.search(block)
        if link_match:
            url = link_match.group(1)
            if "uddg=" in url:
                real_url = re.search(r'uddg=([^&]+)', url)
                if real_url:
                    url = unquote(real_url.group(1))
            title = re.sub(r'<[^>]+>', '', link_match.group(2)).strip()
            snippet = ""
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
            title = unescape(title)
            snippet = unescape(snippet)
            if url.startswith("http"):
                results.append({"title": title, "url": url,
                                "snippet": snippet, "source": "duckduckgo"})
    return results


async def _ddg_search(query: str, num_results: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as http:
        resp = await http.post(
            DDG_URL, data={"q": query, "b": ""},
            headers={"User-Agent": USER_AGENT, "Referer": "https://duckduckgo.com/"},
        )
        resp.raise_for_status()
        results = _extract_ddg_results(resp.text)
        return results[:num_results]


async def _fetch_page_internal(url: str, max_chars: int = 8000) -> str:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as http:
        resp = await http.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        html = resp.text
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL)
    html = re.sub(r'<br\s*/?\s*>', '\n', html)
    html = re.sub(r'</(p|div|h[1-6]|li|tr)>', '\n', html)
    text = re.sub(r'<[^>]+>', '', html)
    text = unescape(text)
    lines = [line.strip() for line in text.split('\n')]
    lines = [l for l in lines if l]
    text = '\n'.join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"
    return text


async def _check_search_status() -> str:
    status = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(f"{SEARXNG_URL}/healthz")
            if resp.status_code == 200:
                status.append(f"SearXNG: ONLINE ({SEARXNG_URL})")
            else:
                status.append(f"SearXNG: ERROR (HTTP {resp.status_code})")
    except (httpx.ConnectError, httpx.TimeoutException):
        status.append("SearXNG: OFFLINE")
    status.append("DuckDuckGo: AVAILABLE (fallback)")
    return "\n".join(status)


# ---------------------------------------------------------------------------
# MCP tool wrappers (8 tools)
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
def search(query: str, scope: str = "all", limit: int = 20) -> str:
    """Search across memory and knowledge bases.
    Finds relevant memories and teaching fragments by meaning and keywords.
    Scope: 'all' (default, searches both), 'memory', or 'knowledge'.
    Results are ranked by relevance using hybrid search (FTS5 + vector similarity)."""
    if scope not in ("all", "memory", "knowledge"):
        return f"Error: scope must be 'all', 'memory', or 'knowledge'. Got: {scope}"
    return _unified_search(query, scope, limit)


@mcp.tool()
def forget_memory(id: int, reason: str) -> str:
    """Delete a memory by ID. Requires a reason for the deletion."""
    return _forget_memory(id, reason)


@mcp.tool()
def store_knowledge(domain: str, title: str, content: str,
                    when_to_apply: str = "", anti_pattern: str = "",
                    confidence: str = "moderate", source_agent: str = "unknown",
                    type: str = "principle", independence_test: str = "") -> str:
    """Store a teaching fragment in the knowledge base.
    Confidence: high, moderate, low. Type: principle, procedure, correction, tool-recommendation."""
    return _store_knowledge(domain, title, content, when_to_apply, anti_pattern,
                            confidence, source_agent, type, independence_test)


@mcp.tool()
def log_outcome(task_description: str, result: str,
                fragments_used: list[int] | None = None, notes: str = "") -> str:
    """Log a task outcome for teaching effectiveness tracking.
    Result: success, failure, or partial.
    fragments_used: list of knowledge fragment IDs consulted during the task."""
    return _log_outcome(task_description, result, fragments_used, notes)


@mcp.tool()
def store_training_example(system_prompt: str, user_message: str,
                           assistant_response: str, domain: str = "general") -> str:
    """Store a training example for future LoRA fine-tuning.
    Format: system + user + assistant messages (SFT format)."""
    return _store_training_example(system_prompt, user_message, assistant_response, domain)


@mcp.tool()
async def web_search(query: str, num_results: int = 8) -> str:
    """Search the web for real-time information.
    Uses self-hosted SearXNG when available (multi-engine: Google, Bing,
    DuckDuckGo, Brave). Falls back to DuckDuckGo if SearXNG is not running.

    Args:
        query: The search query.
        num_results: Max number of results to return (default 8)."""
    results = await _searxng_search(query, num_results)
    backend = "searxng"
    if results is None:
        results = await _ddg_search(query, num_results)
        backend = "duckduckgo"
    if not results:
        return f"No results found for: {query}"
    lines = [f"Web search results for: {query}  [via {backend}]\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r['snippet']:
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 8000) -> str:
    """Fetch and extract readable text from a web page.
    Strips HTML, scripts, navigation, and returns clean text content.
    Use this after web_search to read full articles or documentation."""
    try:
        text = await _fetch_page_internal(url, max_chars)
        if not text.strip():
            return f"Page at {url} returned no readable content."
        return f"Content from: {url}\n\n{text}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error fetching {url}: {e.response.status_code}"
    except httpx.ConnectError:
        return f"Could not connect to {url}"
    except httpx.TimeoutException:
        return f"Timeout fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ---------------------------------------------------------------------------
# CLI admin interface
# ---------------------------------------------------------------------------

def _combined_stats() -> str:
    lines = ["Purple Brain v1.0 — Combined Statistics\n"]
    lines.append("=== Memory Kernel ===")
    lines.append(_get_memory_stats())
    lines.append("\n=== Knowledge Base ===")
    lines.append(_list_domains())
    lines.append("\n=== Teaching ===")
    lines.append(_teaching_effectiveness())
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purple Brain — Unified MCP Server")
    parser.add_argument("--stats", action="store_true", help="Combined memory + knowledge stats")
    parser.add_argument("--recent", nargs="?", const=24, type=int, metavar="HOURS",
                        help="List recent memories (default: 24h)")
    parser.add_argument("--domains", action="store_true", help="List knowledge domains")
    parser.add_argument("--validate", type=int, metavar="ID", help="Validate a teaching fragment")
    parser.add_argument("--import-queue", action="store_true", help="Import fragments from queue")
    parser.add_argument("--effectiveness", action="store_true", help="Teaching effectiveness report")
    parser.add_argument("--export", action="store_true", help="Export training data as JSONL")
    parser.add_argument("--backfill", nargs="?", const=50, type=int, metavar="BATCH",
                        help="Backfill embeddings (default batch: 50)")
    parser.add_argument("--status", action="store_true", help="Search backend health check")
    parser.add_argument("--maintain", action="store_true",
                        help="Run all maintenance: backfill + import-queue + export")

    args = parser.parse_args()

    cli_flags = [args.stats, args.recent is not None, args.domains,
                 args.validate is not None, args.import_queue,
                 args.effectiveness, args.export,
                 args.backfill is not None, args.status, args.maintain]

    if any(cli_flags):
        if args.stats:
            print(_combined_stats())
        if args.recent is not None:
            print(_list_recent(args.recent))
        if args.domains:
            print(_list_domains())
        if args.validate is not None:
            print(_validate_teaching(args.validate))
        if args.import_queue:
            print(_import_queue())
        if args.effectiveness:
            print(_teaching_effectiveness())
        if args.export:
            print(_export_training_data())
        if args.backfill is not None:
            print(_backfill_embeddings(args.backfill))
        if args.status:
            print(asyncio.run(_check_search_status()))
        if args.maintain:
            print("=== Backfill Embeddings ===")
            print(_backfill_embeddings(100))
            print("\n=== Import Queue ===")
            print(_import_queue())
            print("\n=== Export Training Data ===")
            print(_export_training_data())
    else:
        mcp.run()
