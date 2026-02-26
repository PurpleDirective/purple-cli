"""
Purple Proving Ground -- Task Tracker

Instruments Purple CLI sessions for the reward/tier system.
Stores task outcomes, tool call metrics, and tier progression
in a local SQLite database.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".purple" / "metrics" / "tracker.db"

# Tier definitions
TIERS = ["T0", "T1", "T2", "T3", "T4"]
TIER_NAMES = {
    "T0": "Candidate",
    "T1": "Apprentice",
    "T2": "Journeyman",
    "T3": "Craftsman",
    "T4": "Sovereign",
}

# Graduation thresholds: (min_tasks, min_tcr, min_fta, max_uor)
GRADUATION = {
    "T0": (10, 0.60, 0.0, 1.0),    # Just pass the benchmark battery
    "T1": (20, 0.75, 0.0, 0.30),
    "T2": (30, 0.70, 0.60, 0.30),
    "T3": (50, 0.80, 0.70, 0.20),
}

# Demotion: if TCR drops below this in a 20-task window, demote one tier
DEMOTION_TCR_FLOOR = 0.50
DEMOTION_WINDOW = 20


def _connect() -> sqlite3.Connection:
    """Get a connection to the tracker database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            prompt_preview TEXT,
            tool_calls INTEGER DEFAULT 0,
            tool_errors INTEGER DEFAULT 0,
            tool_rounds INTEGER DEFAULT 0,
            outcome TEXT DEFAULT 'pending',
            user_rating INTEGER,
            user_override INTEGER DEFAULT 0,
            model TEXT,
            tier TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS tier_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            changed_at TEXT NOT NULL,
            old_tier TEXT,
            new_tier TEXT,
            reason TEXT,
            model TEXT
        );

        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()

    # Initialize tier if not set
    row = conn.execute("SELECT value FROM config WHERE key = 'current_tier'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO config (key, value) VALUES ('current_tier', 'T0')"
        )
        conn.execute(
            "INSERT INTO config (key, value) VALUES ('current_model', '')"
        )
        conn.commit()

    conn.close()


def get_current_tier() -> str:
    """Get the current tier."""
    conn = _connect()
    row = conn.execute("SELECT value FROM config WHERE key = 'current_tier'").fetchone()
    conn.close()
    return row["value"] if row else "T0"


def _set_config(key: str, value: str):
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()
    conn.close()


def check_model_change(current_model: str) -> bool:
    """Check if the model changed since last session. Returns True if reset occurred."""
    conn = _connect()
    row = conn.execute("SELECT value FROM config WHERE key = 'current_model'").fetchone()
    stored_model = row["value"] if row else ""
    conn.close()

    if stored_model and stored_model != current_model:
        # Model changed -- hard reset to T0
        old_tier = get_current_tier()
        _set_config("current_tier", "T0")
        _set_config("current_model", current_model)

        conn = _connect()
        conn.execute(
            "INSERT INTO tier_history (changed_at, old_tier, new_tier, reason, model) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                old_tier,
                "T0",
                f"Model changed: {stored_model} -> {current_model}",
                current_model,
            ),
        )
        conn.commit()
        conn.close()
        return True

    if not stored_model:
        _set_config("current_model", current_model)

    return False


def start_task(prompt: str, model: str) -> int:
    """Record a new task. Returns the task ID."""
    tier = get_current_tier()
    conn = _connect()
    cursor = conn.execute(
        "INSERT INTO tasks (started_at, prompt_preview, model, tier) VALUES (?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            prompt[:200] if prompt else "",
            model,
            tier,
        ),
    )
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return task_id


def record_tool_call(task_id: int, success: bool):
    """Record a tool call result for a task."""
    conn = _connect()
    if success:
        conn.execute(
            "UPDATE tasks SET tool_calls = tool_calls + 1 WHERE id = ?",
            (task_id,),
        )
    else:
        conn.execute(
            "UPDATE tasks SET tool_calls = tool_calls + 1, tool_errors = tool_errors + 1 WHERE id = ?",
            (task_id,),
        )
    conn.commit()
    conn.close()


def complete_task(task_id: int, outcome: str = "completed", rounds: int = 0):
    """Mark a task as completed."""
    conn = _connect()
    conn.execute(
        "UPDATE tasks SET completed_at = ?, outcome = ?, tool_rounds = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), outcome, rounds, task_id),
    )
    conn.commit()
    conn.close()


def rate_task(task_id: int, rating: int):
    """Apply a user rating (1-5) to a task."""
    conn = _connect()
    conn.execute(
        "UPDATE tasks SET user_rating = ? WHERE id = ?",
        (max(1, min(5, rating)), task_id),
    )
    conn.commit()
    conn.close()


def mark_override(task_id: int):
    """Mark that the user overrode/corrected this task."""
    conn = _connect()
    conn.execute(
        "UPDATE tasks SET user_override = 1 WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    conn.close()


def compute_metrics(window: int = 20) -> dict:
    """Compute current metrics from the most recent tasks.

    Returns dict with: tcr, fta, tcsr, uor, total_tasks, completed_tasks,
    overridden_tasks, violations, avg_rating.
    """
    conn = _connect()

    # Total tasks ever
    total = conn.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]

    # Recent completed tasks for metrics
    rows = conn.execute(
        "SELECT * FROM tasks WHERE outcome != 'pending' ORDER BY completed_at DESC LIMIT ?",
        (window,),
    ).fetchall()

    conn.close()

    if not rows:
        return {
            "tcr": 0.0,
            "fta": 0.0,
            "tcsr": 0.0,
            "uor": 0.0,
            "total_tasks": total,
            "completed_tasks": 0,
            "overridden_tasks": 0,
            "violations": 0,
            "avg_rating": 0.0,
            "window_size": 0,
        }

    completed = [r for r in rows if r["outcome"] == "completed"]
    overridden = [r for r in rows if r["user_override"]]
    total_tool_calls = sum(r["tool_calls"] for r in rows)
    total_tool_errors = sum(r["tool_errors"] for r in rows)
    ratings = [r["user_rating"] for r in rows if r["user_rating"] is not None]

    n = len(rows)
    tcr = len(completed) / n if n else 0.0
    fta = len([r for r in completed if not r["user_override"] and (r["user_rating"] is None or r["user_rating"] >= 4)]) / n if n else 0.0
    tcsr = (total_tool_calls - total_tool_errors) / total_tool_calls if total_tool_calls else 1.0
    uor = len(overridden) / n if n else 0.0

    return {
        "tcr": tcr,
        "fta": fta,
        "tcsr": tcsr,
        "uor": uor,
        "total_tasks": total,
        "completed_tasks": len(completed),
        "overridden_tasks": len(overridden),
        "violations": 0,
        "avg_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        "window_size": n,
    }


def check_promotion() -> str | None:
    """Check if current metrics qualify for promotion. Returns new tier or None."""
    tier = get_current_tier()
    if tier == "T4":
        return None  # Already at max

    thresholds = GRADUATION.get(tier)
    if not thresholds:
        return None

    min_tasks, min_tcr, min_fta, max_uor = thresholds
    metrics = compute_metrics(window=min_tasks)

    if metrics["window_size"] < min_tasks:
        return None
    if metrics["tcr"] < min_tcr:
        return None
    if metrics["fta"] < min_fta:
        return None
    if metrics["uor"] > max_uor:
        return None

    # Qualified for next tier
    tier_idx = TIERS.index(tier)
    return TIERS[tier_idx + 1]


def check_demotion() -> str | None:
    """Check if recent performance warrants demotion. Returns new tier or None."""
    tier = get_current_tier()
    if tier == "T0":
        return None  # Can't demote below T0

    metrics = compute_metrics(window=DEMOTION_WINDOW)
    if metrics["window_size"] < DEMOTION_WINDOW:
        return None

    if metrics["tcr"] < DEMOTION_TCR_FLOOR:
        tier_idx = TIERS.index(tier)
        return TIERS[tier_idx - 1]

    return None


def apply_tier_change(new_tier: str, reason: str, model: str):
    """Apply a tier change and log it."""
    old_tier = get_current_tier()
    _set_config("current_tier", new_tier)

    conn = _connect()
    conn.execute(
        "INSERT INTO tier_history (changed_at, old_tier, new_tier, reason, model) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), old_tier, new_tier, reason, model),
    )
    conn.commit()
    conn.close()


def get_last_task_id() -> int | None:
    """Get the ID of the most recent task."""
    conn = _connect()
    row = conn.execute("SELECT id FROM tasks ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row["id"] if row else None


def get_tier_history(limit: int = 10) -> list[dict]:
    """Get recent tier changes."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM tier_history ORDER BY changed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def sovereignty_score(metrics: dict) -> int:
    """Compute a 0-100 sovereignty score from metrics."""
    if metrics["window_size"] == 0:
        return 0
    # Weighted: TCR 30%, FTA 25%, TCSR 20%, UOR (inverted) 15%, tier 10%
    tier = get_current_tier()
    tier_score = TIERS.index(tier) / 4.0  # 0.0 to 1.0

    score = (
        metrics["tcr"] * 0.30
        + metrics["fta"] * 0.25
        + metrics["tcsr"] * 0.20
        + (1.0 - metrics["uor"]) * 0.15
        + tier_score * 0.10
    )
    return int(score * 100)
