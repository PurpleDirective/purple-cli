"""
Purple Knowledge Server v1.1
Teaching knowledge base for cloud-to-local AI knowledge transfer.

MCP server with tools: lookup_knowledge, store_knowledge, list_domains,
validate_teaching, import_queue, log_outcome, teaching_effectiveness,
store_training_example, export_training_data.

Part of the Teaching Protocol (v1.1, APPROVED 2026-02-26).
Phases 2 + 3: Knowledge Server + Feedback Loop.
"""

import sqlite3
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path.home() / ".purple" / "knowledge" / "knowledge.db"
QUEUE_PATH = Path.home() / ".purple" / "teaching" / "queue"

VALID_CONFIDENCE = ("high", "moderate", "low")
VALID_TYPES = ("principle", "procedure", "correction", "tool-recommendation")

mcp = FastMCP("purple-knowledge")
_initialized = False


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    if not _initialized:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                when_to_apply TEXT DEFAULT '',
                anti_pattern TEXT DEFAULT '',
                confidence TEXT NOT NULL DEFAULT 'moderate',
                source_agent TEXT DEFAULT 'unknown',
                type TEXT NOT NULL DEFAULT 'principle',
                independence_test TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge(source_agent)")

        # FTS5 full-text search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                title, content, when_to_apply, domain,
                content='knowledge', content_rowid='id',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS5 in sync
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

        # Backfill FTS5 for any existing knowledge not yet indexed
        conn.execute("""
            INSERT OR IGNORE INTO knowledge_fts(rowid, title, content, when_to_apply, domain)
            SELECT id, title, content, when_to_apply, domain FROM knowledge
            WHERE id NOT IN (SELECT rowid FROM knowledge_fts)
        """)

        # Phase 3: Outcome tracking for teaching effectiveness
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT NOT NULL,
                result TEXT NOT NULL CHECK(result IN ('success', 'failure', 'partial')),
                fragments_used TEXT DEFAULT '[]',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_result ON outcomes(result)")

        # Phase 3: Fine-tuning data accumulator
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_prompt TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                domain TEXT DEFAULT 'general',
                quality TEXT DEFAULT 'unreviewed'
                    CHECK(quality IN ('unreviewed', 'approved', 'rejected')),
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()

        if DB_PATH.exists():
            if DB_PATH.stat().st_mode & 0o077:
                DB_PATH.chmod(0o600)

        _initialized = True

    return conn


# ---------------------------------------------------------------------------
# Fragment parsing
# ---------------------------------------------------------------------------

def _parse_fragment_file(path: Path) -> dict | None:
    """Parse a Teaching Fragment markdown file into a dict."""
    text = path.read_text()

    # Parse YAML-like frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
    if not match:
        return None

    meta = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()

    # Parse body sections
    sections = {}
    current_section = None
    current_content = []

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

    # Map sections to fields
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
        'domain': meta.get('domain', 'general'),
        'title': title,
        'content': content,
        'when_to_apply': when_to_apply,
        'anti_pattern': anti_pattern,
        'confidence': meta.get('confidence', 'moderate'),
        'source_agent': meta.get('source_agent', 'unknown'),
        'type': meta.get('type', 'principle'),
        'independence_test': meta.get('independence_test', ''),
    }


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _format_fragment(row: sqlite3.Row) -> str:
    return (
        f"#{row['id']} [{row['domain']}] {row['title']}\n"
        f"  Confidence: {row['confidence']} | Type: {row['type']} | Agent: {row['source_agent']}\n"
        f"  {row['content'][:300]}{'...' if len(row['content']) > 300 else ''}\n"
        f"  When: {row['when_to_apply'][:150]}{'...' if len(row['when_to_apply']) > 150 else ''}"
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _lookup(query: str, domain: Optional[str] = None, limit: int = 10) -> str:
    conn = get_db()
    try:
        rows = []
        try:
            sql = """
                SELECT k.*, fts.rank
                FROM knowledge_fts fts
                JOIN knowledge k ON k.id = fts.rowid
                WHERE knowledge_fts MATCH ?
            """
            params: list = [query]
            if domain:
                sql += " AND k.domain = ?"
                params.append(domain)
            sql += " ORDER BY fts.rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            pass

        if not rows:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql = "SELECT * FROM knowledge WHERE (content LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\')"
            params = [f"%{escaped}%", f"%{escaped}%"]
            if domain:
                sql += " AND domain = ?"
                params.append(domain)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return f"No knowledge found for '{query}'" + (f" in domain '{domain}'" if domain else "")

        return f"Found {len(rows)} fragments:\n\n" + "\n\n".join(_format_fragment(r) for r in rows)
    finally:
        conn.close()


def _store(domain: str, title: str, content: str, when_to_apply: str = "",
           anti_pattern: str = "", confidence: str = "moderate",
           source_agent: str = "unknown", type: str = "principle",
           independence_test: str = "") -> str:
    if confidence not in VALID_CONFIDENCE:
        return f"Error: confidence must be one of {VALID_CONFIDENCE}. Got: {confidence}"
    if type not in VALID_TYPES:
        return f"Error: type must be one of {VALID_TYPES}. Got: {type}"

    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO knowledge
               (domain, title, content, when_to_apply, anti_pattern,
                confidence, source_agent, type, independence_test, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (domain, title, content, when_to_apply, anti_pattern,
             confidence, source_agent, type, independence_test, now, now)
        )
        conn.commit()
        return f"Stored knowledge #{cursor.lastrowid}: [{domain}] {title}"
    finally:
        conn.close()


def _list_domains() -> str:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT domain, COUNT(*) as count, "
            "GROUP_CONCAT(DISTINCT confidence) as confidences "
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


def _validate(id: int) -> str:
    conn = get_db()
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
    """Import all Teaching Fragments from the queue directory."""
    if not QUEUE_PATH.exists():
        return "Queue directory does not exist."

    files = sorted(QUEUE_PATH.glob("*.md"))
    if not files:
        return "No fragments in queue."

    results = []
    for f in files:
        parsed = _parse_fragment_file(f)
        if parsed:
            result = _store(**parsed)
            results.append(f"  {f.name}: {result}")
        else:
            results.append(f"  {f.name}: Error — could not parse fragment")

    return f"Processed {len(files)} fragments:\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Phase 3: Feedback loop — outcome tracking & training data
# ---------------------------------------------------------------------------

TRAINING_DATA_PATH = Path.home() / ".purple" / "teaching" / "finetune" / "training.jsonl"


def _log_outcome(task_description: str, result: str,
                 fragments_used: list[int] | None = None,
                 notes: str = "") -> str:
    if result not in ("success", "failure", "partial"):
        return f"Error: result must be success, failure, or partial. Got: {result}"

    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO outcomes (task_description, result, fragments_used, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_description, result, json.dumps(fragments_used or []), notes, now)
        )
        conn.commit()
        return f"Logged outcome: [{result}] {task_description[:80]}"
    finally:
        conn.close()


def _teaching_effectiveness() -> str:
    conn = get_db()
    try:
        # Overall outcome stats
        total = conn.execute("SELECT COUNT(*) as c FROM outcomes").fetchone()["c"]
        if total == 0:
            return "No outcomes logged yet. Use log_outcome() after tasks to track effectiveness."

        by_result = conn.execute(
            "SELECT result, COUNT(*) as c FROM outcomes GROUP BY result"
        ).fetchall()
        result_map = {r["result"]: r["c"] for r in by_result}

        success = result_map.get("success", 0)
        failure = result_map.get("failure", 0)
        partial = result_map.get("partial", 0)
        success_rate = success * 100 // max(total, 1)

        # Fragment usage analysis
        all_outcomes = conn.execute("SELECT * FROM outcomes").fetchall()
        fragment_success = {}  # fragment_id -> [success_count, total_count]

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
            f"Teaching Effectiveness Report",
            f"Total outcomes: {total}",
            f"  Success: {success} ({success_rate}%)",
            f"  Partial: {partial}",
            f"  Failure: {failure}",
            "",
        ]

        if fragment_success:
            lines.append("Fragment effectiveness (fragments used in 2+ tasks):")
            for fid, (s, t) in sorted(fragment_success.items(), key=lambda x: x[1][1], reverse=True):
                if t >= 2:
                    row = conn.execute("SELECT title FROM knowledge WHERE id = ?", (fid,)).fetchone()
                    title = row["title"] if row else f"(deleted #{fid})"
                    lines.append(f"  #{fid} {title}: {s}/{t} success ({s * 100 // max(t, 1)}%)")

        # Training data stats
        training_count = conn.execute("SELECT COUNT(*) as c FROM training_data").fetchone()["c"]
        approved = conn.execute("SELECT COUNT(*) as c FROM training_data WHERE quality = 'approved'").fetchone()["c"]
        lines.append(f"\nTraining data: {training_count} examples ({approved} approved)")
        lines.append(f"Fine-tuning threshold: 1,000 examples (current: {approved})")

        return "\n".join(lines)
    finally:
        conn.close()


def _store_training_example(system_prompt: str, user_message: str,
                            assistant_response: str, domain: str = "general") -> str:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO training_data (system_prompt, user_message, assistant_response, domain, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (system_prompt, user_message, assistant_response, domain, now)
        )
        conn.commit()
        return f"Stored training example #{cursor.lastrowid} [{domain}]"
    finally:
        conn.close()


def _export_training_data() -> str:
    """Export approved training examples as JSONL for fine-tuning."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM training_data WHERE quality = 'approved' ORDER BY id"
        ).fetchall()

        if not rows:
            # Fall back to all unreviewed if none approved yet
            rows = conn.execute(
                "SELECT * FROM training_data ORDER BY id"
            ).fetchall()

        if not rows:
            return "No training data to export."

        TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_DATA_PATH, "w") as f:
            for row in rows:
                entry = {
                    "messages": [
                        {"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": row["user_message"]},
                        {"role": "assistant", "content": row["assistant_response"]},
                    ]
                }
                f.write(json.dumps(entry) + "\n")

        return f"Exported {len(rows)} examples to {TRAINING_DATA_PATH}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

@mcp.tool()
def lookup_knowledge(query: str, domain: str | None = None, limit: int = 10) -> str:
    """Search the teaching knowledge base for relevant fragments.
    Use when facing unfamiliar problems or needing decision frameworks.
    Optionally filter by domain: architecture, python-debugging, security, devops, etc."""
    return _lookup(query, domain, limit)


@mcp.tool()
def store_knowledge(domain: str, title: str, content: str,
                    when_to_apply: str = "", anti_pattern: str = "",
                    confidence: str = "moderate", source_agent: str = "unknown",
                    type: str = "principle", independence_test: str = "") -> str:
    """Store a teaching fragment in the knowledge base.
    Confidence: high, moderate, low. Type: principle, procedure, correction, tool-recommendation."""
    return _store(domain, title, content, when_to_apply, anti_pattern,
                  confidence, source_agent, type, independence_test)


@mcp.tool()
def list_domains() -> str:
    """List all knowledge domains with fragment counts and confidence levels."""
    return _list_domains()


@mcp.tool()
def validate_teaching(id: int) -> str:
    """Validate a teaching fragment for completeness.
    Checks: content, when_to_apply, anti_pattern, independence_test, length."""
    return _validate(id)


@mcp.tool()
def import_queue() -> str:
    """Import all Teaching Fragment markdown files from ~/.purple/teaching/queue/.
    Use this to batch-import fragments generated during cloud AI sessions."""
    return _import_queue()


# --- Phase 3: Feedback loop tools ---

@mcp.tool()
def log_outcome(task_description: str, result: str,
                fragments_used: list[int] | None = None,
                notes: str = "") -> str:
    """Log a task outcome for teaching effectiveness tracking.
    Result: success, failure, or partial.
    fragments_used: list of knowledge fragment IDs that were consulted during the task."""
    return _log_outcome(task_description, result, fragments_used, notes)


@mcp.tool()
def teaching_effectiveness() -> str:
    """Show teaching effectiveness report: outcome stats, per-fragment success rates,
    and training data accumulation progress toward the 1,000-example fine-tuning threshold."""
    return _teaching_effectiveness()


@mcp.tool()
def store_training_example(system_prompt: str, user_message: str,
                           assistant_response: str, domain: str = "general") -> str:
    """Store a training example for future LoRA fine-tuning.
    Format: system + user + assistant messages (SFT format).
    Examples accumulate toward the 1,000-example fine-tuning threshold."""
    return _store_training_example(system_prompt, user_message, assistant_response, domain)


@mcp.tool()
def export_training_data() -> str:
    """Export training examples as JSONL for LoRA fine-tuning.
    Exports approved examples (or all if none approved yet) to
    ~/.purple/teaching/finetune/training.jsonl"""
    return _export_training_data()


if __name__ == "__main__":
    mcp.run()
