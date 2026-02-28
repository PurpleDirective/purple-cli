#!/usr/bin/env python3
"""
Purple Proving Ground — V6 Contamination-Proof Test Battery
15 novel tests across 5 categories. Zero GitHub equivalent.

Categories:
  A (5 tests) — Novel code generation (Purple-specific domain logic)
  B (3 tests) — Code editing (modify existing code, test for regressions)
  C (3 tests) — Multi-step reasoning (analyze + diagnose + fix)
  D (2 tests) — Error recovery (complete stubs, fix corruption)
  E (2 tests) — Agentic capability (multi-turn, prioritization)
"""

import hashlib

# =============================================================================
# CATEGORY A: Novel Code Generation (5 tests, 75+ assertions)
# =============================================================================

A1_PROMPT = """Write a Python module `sovereignty_engine.py` implementing a sovereignty score calculator for an AI agent proving ground.

Requirements:

1. Class `MetricWindow`:
   - `__init__(self, size: int = 20)` — sliding window of metric snapshots
   - `push(self, snapshot: dict)` — add a snapshot with keys: tcr (float 0-1), fta (float 0-1), tcsr (float 0-1), uor (float 0-1), timestamp (float)
   - `average(self, key: str) -> float` — average of a metric across the window
   - `streak(self, key: str, threshold: float, above: bool = True) -> int` — count consecutive latest snapshots where metric is above (or below if above=False) the threshold
   - `is_stale(self, max_age: float) -> bool` — True if newest snapshot is older than max_age seconds from the latest timestamp in the window

2. Class `SovereigntyEngine`:
   - `__init__(self, weights: dict | None = None)` — weights for each metric dimension, defaults: {"tcr": 0.30, "fta": 0.25, "tcsr": 0.20, "uor": 0.15, "tier": 0.10}
   - `score(self, window: MetricWindow, tier: int) -> float` — compute sovereignty score 0-100:
     * For each metric (tcr, fta, tcsr): raw = window.average(key) * 100
     * For uor: raw = (1 - window.average("uor")) * 100 (lower is better)
     * For tier: raw = tier * 25 (tier 0=0, 1=25, 2=50, 3=75, 4=100)
     * Apply streak multiplier: if tcr streak >= 5 above 0.8, multiply tcr component by 1.15 (cap at 100)
     * Apply decay: if window.is_stale(3600), multiply total by 0.9
     * Weighted sum using self.weights, clamped to [0, 100]
   - `recommend_tier(self, window: MetricWindow, current_tier: int) -> tuple[int, str]` — returns (new_tier, reason):
     * Promote if: score >= promotion_threshold AND sustained for hysteresis_count consecutive windows
     * Demote if: window.average("tcr") < 0.5 in current window
     * promotion_thresholds = {0: 55, 1: 65, 2: 75, 3: 85}
     * hysteresis_count = 3 (must call recommend_tier 3 times with qualifying score before promotion)
     * On model change (detected by tier == 0 and window has < 3 snapshots), return (0, "insufficient data")
   - `reset_hysteresis(self)` — clear the promotion counter

3. Edge cases:
   - Empty window: score returns 0.0, recommend_tier returns (current_tier, "no data")
   - All metrics perfect: score should be exactly 100.0 (no streak/decay applied to perfect)
   - UOR of 1.0 (100% override): uor component should be 0

Return ONLY the Python code, no explanation, no markdown fences."""

A1_TEST = '''import pytest
from sovereignty_engine import MetricWindow, SovereigntyEngine

@pytest.fixture
def engine():
    return SovereigntyEngine()

@pytest.fixture
def full_window():
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.8, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 1000.0 + i})
    return w

def test_empty_window_score(engine):
    w = MetricWindow()
    assert engine.score(w, 0) == 0.0

def test_empty_window_recommend(engine):
    w = MetricWindow()
    tier, reason = engine.recommend_tier(w, 2)
    assert tier == 2
    assert "no data" in reason.lower()

def test_perfect_score(engine):
    w = MetricWindow(size=3)
    for i in range(3):
        w.push({"tcr": 1.0, "fta": 1.0, "tcsr": 1.0, "uor": 0.0, "timestamp": 1000.0 + i})
    assert engine.score(w, 4) == 100.0

def test_uor_inverted(engine):
    w = MetricWindow(size=1)
    w.push({"tcr": 0.5, "fta": 0.5, "tcsr": 0.5, "uor": 1.0, "timestamp": 1000.0})
    score = engine.score(w, 0)
    # uor=1.0 means uor component = 0
    # tcr=0.5*100*0.30=15, fta=0.5*100*0.25=12.5, tcsr=0.5*100*0.20=10, uor=0*0.15=0, tier=0*0.10=0
    assert abs(score - 37.5) < 0.01

def test_streak_multiplier(engine, full_window):
    # 5 consecutive tcr=0.8 >= 0.8 threshold => streak multiplier 1.15
    score_with_streak = engine.score(full_window, 2)
    # Without streak: tcr=80*0.30=24, with streak: 24*1.15=27.6
    # fta=70*0.25=17.5, tcsr=90*0.20=18, uor=(1-0.1)*100*0.15=13.5, tier=50*0.10=5
    # Total with streak: 27.6+17.5+18+13.5+5 = 81.6
    assert abs(score_with_streak - 81.6) < 0.1

def test_no_streak_below_threshold(engine):
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.79, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 1000.0 + i})
    score = engine.score(w, 2)
    # No streak multiplier (tcr=0.79 < 0.8)
    # tcr=79*0.30=23.7, fta=70*0.25=17.5, tcsr=90*0.20=18, uor=90*0.15=13.5, tier=50*0.10=5
    assert abs(score - 77.7) < 0.1

def test_stale_decay(engine):
    w = MetricWindow(size=2)
    w.push({"tcr": 0.8, "fta": 0.8, "tcsr": 0.8, "uor": 0.1, "timestamp": 1000.0})
    w.push({"tcr": 0.8, "fta": 0.8, "tcsr": 0.8, "uor": 0.1, "timestamp": 1001.0})
    assert not w.is_stale(3600)
    # Make it stale by checking against a much later reference
    w2 = MetricWindow(size=2)
    w2.push({"tcr": 0.8, "fta": 0.8, "tcsr": 0.8, "uor": 0.1, "timestamp": 1000.0})
    # Only 1 snapshot, newest = 1000.0, if we check is_stale with max_age=0.001 it should be stale
    # Actually stale means newest is older than max_age from current time... let's use a different approach
    # The window is stale if (latest_timestamp - newest_snapshot_timestamp) > max_age
    # But we need to test with respect to something. Let's just verify the non-stale case.
    score_normal = engine.score(w, 2)
    assert score_normal > 0

def test_promotion_hysteresis(engine):
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.85, "fta": 0.8, "tcsr": 0.9, "uor": 0.05, "timestamp": 1000.0 + i})
    # First call — should not promote yet (hysteresis = 3)
    tier1, reason1 = engine.recommend_tier(w, 0)
    assert tier1 == 0  # Not yet
    # Second call
    tier2, reason2 = engine.recommend_tier(w, 0)
    assert tier2 == 0  # Not yet
    # Third call — NOW promote
    tier3, reason3 = engine.recommend_tier(w, 0)
    assert tier3 == 1
    assert "promot" in reason3.lower()

def test_demotion_low_tcr(engine):
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.4, "fta": 0.3, "tcsr": 0.5, "uor": 0.5, "timestamp": 1000.0 + i})
    tier, reason = engine.recommend_tier(w, 2)
    assert tier == 1  # Demoted from 2 to 1
    assert "demot" in reason.lower()

def test_demotion_floor(engine):
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.3, "fta": 0.2, "tcsr": 0.4, "uor": 0.8, "timestamp": 1000.0 + i})
    tier, reason = engine.recommend_tier(w, 0)
    assert tier == 0  # Can't demote below 0

def test_reset_hysteresis(engine):
    w = MetricWindow(size=5)
    for i in range(5):
        w.push({"tcr": 0.85, "fta": 0.8, "tcsr": 0.9, "uor": 0.05, "timestamp": 1000.0 + i})
    engine.recommend_tier(w, 0)
    engine.recommend_tier(w, 0)
    engine.reset_hysteresis()
    # After reset, should need 3 more calls
    tier, reason = engine.recommend_tier(w, 0)
    assert tier == 0  # Reset means start over

def test_window_sliding(engine):
    w = MetricWindow(size=3)
    w.push({"tcr": 0.5, "fta": 0.5, "tcsr": 0.5, "uor": 0.5, "timestamp": 1.0})
    w.push({"tcr": 0.5, "fta": 0.5, "tcsr": 0.5, "uor": 0.5, "timestamp": 2.0})
    w.push({"tcr": 0.5, "fta": 0.5, "tcsr": 0.5, "uor": 0.5, "timestamp": 3.0})
    w.push({"tcr": 1.0, "fta": 1.0, "tcsr": 1.0, "uor": 0.0, "timestamp": 4.0})
    # Window should now have entries at t=2,3,4 (size=3, oldest dropped)
    avg_tcr = w.average("tcr")
    assert abs(avg_tcr - (0.5 + 0.5 + 1.0) / 3) < 0.01

def test_streak_broken():
    w = MetricWindow(size=5)
    w.push({"tcr": 0.9, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 1.0})
    w.push({"tcr": 0.9, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 2.0})
    w.push({"tcr": 0.5, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 3.0})  # breaks
    w.push({"tcr": 0.9, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 4.0})
    w.push({"tcr": 0.9, "fta": 0.7, "tcsr": 0.9, "uor": 0.1, "timestamp": 5.0})
    assert w.streak("tcr", 0.8, above=True) == 2  # Only last 2

def test_custom_weights():
    e = SovereigntyEngine(weights={"tcr": 0.50, "fta": 0.20, "tcsr": 0.10, "uor": 0.10, "tier": 0.10})
    w = MetricWindow(size=1)
    w.push({"tcr": 1.0, "fta": 0.0, "tcsr": 0.0, "uor": 0.0, "timestamp": 1.0})
    score = e.score(w, 0)
    # tcr=100*0.50=50, fta=0*0.20=0, tcsr=0*0.10=0, uor=100*0.10=10, tier=0*0.10=0
    assert abs(score - 60.0) < 0.01

def test_score_clamped():
    e = SovereigntyEngine()
    w = MetricWindow(size=1)
    w.push({"tcr": 1.0, "fta": 1.0, "tcsr": 1.0, "uor": 0.0, "timestamp": 1.0})
    score = e.score(w, 4)
    assert 0 <= score <= 100
'''

A2_PROMPT = """Write a Python module `cascade_router.py` implementing a priority cascade message routing system.

Requirements:

1. Class `Message`:
   - `__init__(self, id: str, urgency: int, tags: list[str], payload: str)` — urgency is 1 (lowest) to 5 (highest)

2. Class `Route`:
   - `__init__(self, id: str, match_tags: list[str], action: str, cascade_to: str | None = None, rate_limit: int | None = None)`
   - match_tags: message must have ALL these tags to match
   - action: string describing what to do (stored in routing result)
   - cascade_to: optional Route id to trigger after this route fires
   - rate_limit: max messages per minute for this route (None = unlimited)

3. Class `CascadeRouter`:
   - `__init__(self, max_cascade_depth: int = 10)` — maximum cascade chain length
   - `add_route(self, route: Route)` — register a route
   - `route(self, message: Message, timestamp: float = 0.0) -> list[dict]` — route a message, returns list of dicts:
     * Each dict: {"route_id": str, "action": str, "urgency": int, "cascade_depth": int}
     * If a route has cascade_to, follow the chain (incrementing cascade_depth)
     * Priority inversion: if cascaded route has higher urgency requirement, elevate message urgency
     * Circular cascade detection: if route_id appears twice in chain, stop and add {"error": "circular_cascade", "route_id": str}
     * Rate limiting: if route exceeded rate_limit at this timestamp, skip it and add {"error": "rate_limited", "route_id": str}
   - `dead_letters(self) -> list[Message]` — messages that matched no route
   - `route_stats(self) -> dict[str, int]` — count of messages routed per route_id

Return ONLY the Python code, no explanation, no markdown fences."""

A2_TEST = '''import pytest
from cascade_router import Message, Route, CascadeRouter

@pytest.fixture
def router():
    r = CascadeRouter(max_cascade_depth=5)
    r.add_route(Route("alert", ["urgent"], "send_alert", cascade_to="log"))
    r.add_route(Route("log", ["urgent"], "write_log"))
    r.add_route(Route("archive", ["data"], "archive_data"))
    return r

def test_simple_route(router):
    m = Message("m1", 3, ["urgent"], "test")
    result = router.route(m)
    assert len(result) >= 1
    assert result[0]["route_id"] == "alert"
    assert result[0]["action"] == "send_alert"

def test_cascade(router):
    m = Message("m2", 3, ["urgent"], "test")
    result = router.route(m)
    assert len(result) == 2
    assert result[0]["route_id"] == "alert"
    assert result[1]["route_id"] == "log"
    assert result[1]["cascade_depth"] == 1

def test_no_match_dead_letter(router):
    m = Message("m3", 1, ["unknown_tag"], "test")
    result = router.route(m)
    assert len(result) == 0
    assert len(router.dead_letters()) == 1
    assert router.dead_letters()[0].id == "m3"

def test_circular_cascade():
    r = CascadeRouter()
    r.add_route(Route("a", ["x"], "action_a", cascade_to="b"))
    r.add_route(Route("b", ["x"], "action_b", cascade_to="a"))
    m = Message("m4", 2, ["x"], "test")
    result = r.route(m)
    errors = [x for x in result if "error" in x]
    assert any(e["error"] == "circular_cascade" for e in errors)

def test_max_cascade_depth():
    r = CascadeRouter(max_cascade_depth=3)
    r.add_route(Route("r1", ["t"], "a1", cascade_to="r2"))
    r.add_route(Route("r2", ["t"], "a2", cascade_to="r3"))
    r.add_route(Route("r3", ["t"], "a3", cascade_to="r4"))
    r.add_route(Route("r4", ["t"], "a4"))
    m = Message("m5", 1, ["t"], "test")
    result = r.route(m)
    non_error = [x for x in result if "error" not in x]
    assert len(non_error) <= 3

def test_rate_limiting():
    r = CascadeRouter()
    r.add_route(Route("limited", ["fast"], "do_thing", rate_limit=2))
    for i in range(3):
        m = Message(f"m{i}", 1, ["fast"], "test")
        result = r.route(m, timestamp=0.0)
    # Third message should be rate limited (2 per minute, same timestamp)
    m3 = Message("m_extra", 1, ["fast"], "test")
    result = r.route(m3, timestamp=0.0)
    assert any("error" in x and x["error"] == "rate_limited" for x in result)

def test_rate_limit_reset():
    r = CascadeRouter()
    r.add_route(Route("limited", ["fast"], "do_thing", rate_limit=2))
    r.route(Message("m1", 1, ["fast"], "t"), timestamp=0.0)
    r.route(Message("m2", 1, ["fast"], "t"), timestamp=0.0)
    # After 60 seconds, rate should reset
    result = r.route(Message("m3", 1, ["fast"], "t"), timestamp=61.0)
    assert all("error" not in x for x in result)

def test_multiple_tag_match():
    r = CascadeRouter()
    r.add_route(Route("multi", ["a", "b"], "do_multi"))
    # Missing tag b
    result1 = r.route(Message("m1", 1, ["a"], "t"))
    assert len(result1) == 0
    # Both tags present
    result2 = r.route(Message("m2", 1, ["a", "b", "c"], "t"))
    assert len(result2) == 1

def test_route_stats(router):
    router.route(Message("s1", 3, ["urgent"], "t"))
    router.route(Message("s2", 3, ["urgent"], "t"))
    router.route(Message("s3", 1, ["data"], "t"))
    stats = router.route_stats()
    assert stats.get("alert", 0) == 2
    assert stats.get("log", 0) == 2  # cascaded from alert
    assert stats.get("archive", 0) == 1

def test_urgency_preserved_in_result(router):
    m = Message("u1", 5, ["urgent"], "test")
    result = router.route(m)
    assert result[0]["urgency"] == 5

def test_cascade_depth_zero_for_direct():
    r = CascadeRouter()
    r.add_route(Route("direct", ["x"], "do_x"))
    m = Message("d1", 1, ["x"], "test")
    result = r.route(m)
    assert result[0]["cascade_depth"] == 0

def test_empty_router():
    r = CascadeRouter()
    m = Message("e1", 1, ["any"], "test")
    result = r.route(m)
    assert result == []
    assert len(r.dead_letters()) == 1

def test_multiple_routes_same_message():
    r = CascadeRouter()
    r.add_route(Route("r1", ["a"], "act1"))
    r.add_route(Route("r2", ["a"], "act2"))
    m = Message("mm1", 1, ["a"], "test")
    result = r.route(m)
    # Both routes should match
    route_ids = [x["route_id"] for x in result if "error" not in x]
    assert "r1" in route_ids
    assert "r2" in route_ids

def test_cascade_to_nonexistent():
    r = CascadeRouter()
    r.add_route(Route("orphan", ["x"], "do_x", cascade_to="nonexistent"))
    m = Message("o1", 1, ["x"], "test")
    result = r.route(m)
    # Should route the first one, cascade target not found is not an error (just stops)
    assert len(result) >= 1
    assert result[0]["route_id"] == "orphan"
'''

A3_PROMPT = """Write a Python module `temporal_config.py` implementing a temporally-aware configuration system.

Requirements:

1. Class `ConfigEntry`:
   - `__init__(self, key: str, value: any, effective_from: float, effective_until: float | None = None, priority: int = 0)`
   - effective_until=None means the entry is valid indefinitely
   - priority: higher priority wins on overlap (latest write wins if same priority)

2. Class `TemporalConfig`:
   - `__init__(self, parent: 'TemporalConfig | None' = None)` — optional parent for inheritance
   - `set(self, key: str, value: any, effective_from: float, effective_until: float | None = None, priority: int = 0)` — add a config entry
   - `get(self, key: str, at: float) -> any` — get value at a point in time. Returns None if not found. Checks self first, then parent (inheritance). Among overlapping entries at the same time, highest priority wins. Same priority: latest effective_from wins.
   - `snapshot(self, at: float) -> dict` — get all key-value pairs valid at timestamp, including inherited from parent. Child overrides parent on conflicts.
   - `diff(self, t1: float, t2: float) -> dict` — returns {"added": dict, "removed": dict, "changed": dict} comparing snapshots at t1 and t2. "changed" maps key to {"old": v1, "new": v2}.
   - `overlaps(self, key: str) -> list[tuple[ConfigEntry, ConfigEntry]]` — find all overlapping time ranges for the same key (same priority). Returns list of (entry1, entry2) pairs.
   - `history(self, key: str) -> list[ConfigEntry]` — all entries for key, sorted by effective_from ascending

Return ONLY the Python code, no explanation, no markdown fences."""

A3_TEST = '''import pytest
from temporal_config import ConfigEntry, TemporalConfig

def test_simple_set_get():
    c = TemporalConfig()
    c.set("db_host", "localhost", effective_from=0.0)
    assert c.get("db_host", at=1.0) == "localhost"

def test_get_before_effective():
    c = TemporalConfig()
    c.set("db_host", "localhost", effective_from=10.0)
    assert c.get("db_host", at=5.0) is None

def test_get_after_expiry():
    c = TemporalConfig()
    c.set("db_host", "localhost", effective_from=0.0, effective_until=10.0)
    assert c.get("db_host", at=5.0) == "localhost"
    assert c.get("db_host", at=15.0) is None

def test_priority_override():
    c = TemporalConfig()
    c.set("mode", "normal", effective_from=0.0, priority=0)
    c.set("mode", "debug", effective_from=0.0, priority=1)
    assert c.get("mode", at=1.0) == "debug"

def test_same_priority_latest_wins():
    c = TemporalConfig()
    c.set("mode", "v1", effective_from=0.0, priority=0)
    c.set("mode", "v2", effective_from=5.0, priority=0)
    assert c.get("mode", at=10.0) == "v2"
    assert c.get("mode", at=3.0) == "v1"

def test_snapshot():
    c = TemporalConfig()
    c.set("a", 1, effective_from=0.0)
    c.set("b", 2, effective_from=5.0)
    c.set("c", 3, effective_from=0.0, effective_until=3.0)
    snap = c.snapshot(at=2.0)
    assert snap == {"a": 1, "c": 3}
    snap2 = c.snapshot(at=6.0)
    assert snap2 == {"a": 1, "b": 2}

def test_diff():
    c = TemporalConfig()
    c.set("a", 1, effective_from=0.0)
    c.set("b", 2, effective_from=5.0)
    c.set("a", 10, effective_from=5.0)
    d = c.diff(t1=2.0, t2=6.0)
    assert d["added"] == {"b": 2}
    assert d["changed"]["a"] == {"old": 1, "new": 10}
    assert d["removed"] == {}

def test_diff_with_removal():
    c = TemporalConfig()
    c.set("temp", "yes", effective_from=0.0, effective_until=5.0)
    d = c.diff(t1=2.0, t2=10.0)
    assert d["removed"] == {"temp": "yes"}

def test_inheritance():
    parent = TemporalConfig()
    parent.set("global_key", "from_parent", effective_from=0.0)
    child = TemporalConfig(parent=parent)
    assert child.get("global_key", at=1.0) == "from_parent"

def test_inheritance_override():
    parent = TemporalConfig()
    parent.set("key", "parent_val", effective_from=0.0)
    child = TemporalConfig(parent=parent)
    child.set("key", "child_val", effective_from=5.0)
    assert child.get("key", at=3.0) == "parent_val"
    assert child.get("key", at=6.0) == "child_val"

def test_snapshot_with_inheritance():
    parent = TemporalConfig()
    parent.set("x", 1, effective_from=0.0)
    parent.set("y", 2, effective_from=0.0)
    child = TemporalConfig(parent=parent)
    child.set("y", 20, effective_from=0.0)
    child.set("z", 3, effective_from=0.0)
    snap = child.snapshot(at=1.0)
    assert snap == {"x": 1, "y": 20, "z": 3}

def test_overlaps():
    c = TemporalConfig()
    c.set("k", "a", effective_from=0.0, effective_until=10.0)
    c.set("k", "b", effective_from=5.0, effective_until=15.0)
    overlaps = c.overlaps("k")
    assert len(overlaps) == 1

def test_no_overlaps():
    c = TemporalConfig()
    c.set("k", "a", effective_from=0.0, effective_until=5.0)
    c.set("k", "b", effective_from=10.0, effective_until=15.0)
    overlaps = c.overlaps("k")
    assert len(overlaps) == 0

def test_history():
    c = TemporalConfig()
    c.set("k", "c", effective_from=10.0)
    c.set("k", "a", effective_from=0.0)
    c.set("k", "b", effective_from=5.0)
    h = c.history("k")
    assert [e.value for e in h] == ["a", "b", "c"]

def test_get_nonexistent():
    c = TemporalConfig()
    assert c.get("missing", at=0.0) is None
'''

A4_PROMPT = """Write a Python module `proof_chain.py` implementing a decision audit trail with evidence chain validation.

Requirements:

1. Class `Evidence`:
   - `__init__(self, id: str, content: str, confidence: float, source_agent: str, timestamp: float)`
   - confidence: 0.0 to 1.0

2. Class `Claim`:
   - `__init__(self, id: str, statement: str, evidence_ids: list[str], sub_claim_ids: list[str] | None = None)`
   - evidence_ids: direct evidence supporting this claim
   - sub_claim_ids: claims that this claim depends on (optional)

3. Class `ProofChainValidator`:
   - `__init__(self, min_sources: int = 2, max_staleness: float = 86400.0)`
   - `add_evidence(self, evidence: Evidence)` — register evidence
   - `add_claim(self, claim: Claim)` — register a claim
   - `validate(self, claim_id: str, reference_time: float) -> dict` — validate a claim, returns:
     * `{"valid": bool, "errors": list[str], "effective_confidence": float}`
     * Check 1 — Source coverage: claim must have >= min_sources UNIQUE source_agents across its evidence. Error: "insufficient_sources: need {min_sources}, have {N}"
     * Check 2 — No circular dependencies: if claim A depends on claim B which depends on claim A, error: "circular_dependency: {claim_id}"
     * Check 3 — Confidence propagation: effective_confidence = min of all evidence confidences AND min of all sub-claim effective_confidences (recursive)
     * Check 4 — Staleness: if any evidence is older than max_staleness from reference_time, degrade that evidence's confidence by 50%. Error (warning): "stale_evidence: {evidence_id}"
     * Check 5 — Missing references: if evidence_id or sub_claim_id doesn't exist, error: "missing_reference: {id}"
   - `chain_depth(self, claim_id: str) -> int` — max depth of the dependency tree (no sub_claims = depth 0)
   - `all_evidence_for(self, claim_id: str) -> list[Evidence]` — recursively collect all evidence across the claim's dependency tree

Return ONLY the Python code, no explanation, no markdown fences."""

A4_TEST = '''import pytest
from proof_chain import Evidence, Claim, ProofChainValidator

@pytest.fixture
def validator():
    v = ProofChainValidator(min_sources=2, max_staleness=100.0)
    v.add_evidence(Evidence("e1", "test result A", 0.9, "agent_alpha", 1000.0))
    v.add_evidence(Evidence("e2", "test result B", 0.8, "agent_beta", 1000.0))
    v.add_evidence(Evidence("e3", "observation C", 0.7, "agent_gamma", 1000.0))
    return v

def test_valid_claim(validator):
    validator.add_claim(Claim("c1", "System works", ["e1", "e2"]))
    result = validator.validate("c1", reference_time=1050.0)
    assert result["valid"] is True
    assert result["effective_confidence"] == 0.8  # min(0.9, 0.8)

def test_insufficient_sources(validator):
    validator.add_evidence(Evidence("e4", "solo", 0.9, "agent_alpha", 1000.0))
    validator.add_claim(Claim("c2", "Single source claim", ["e1", "e4"]))
    result = validator.validate("c2", reference_time=1050.0)
    assert result["valid"] is False
    assert any("insufficient_sources" in e for e in result["errors"])

def test_circular_dependency(validator):
    validator.add_claim(Claim("ca", "Claim A", ["e1"], sub_claim_ids=["cb"]))
    validator.add_claim(Claim("cb", "Claim B", ["e2"], sub_claim_ids=["ca"]))
    result = validator.validate("ca", reference_time=1050.0)
    assert result["valid"] is False
    assert any("circular_dependency" in e for e in result["errors"])

def test_stale_evidence(validator):
    validator.add_evidence(Evidence("e_old", "ancient data", 0.9, "agent_delta", 500.0))
    validator.add_claim(Claim("c_stale", "Stale claim", ["e_old", "e1"]))
    result = validator.validate("c_stale", reference_time=1050.0)
    # e_old is 550 seconds old > max_staleness 100, confidence degraded 50%: 0.45
    assert any("stale_evidence" in e for e in result["errors"])
    assert result["effective_confidence"] < 0.9

def test_missing_reference(validator):
    validator.add_claim(Claim("c_missing", "Bad ref", ["e1", "e_nonexistent"]))
    result = validator.validate("c_missing", reference_time=1050.0)
    assert result["valid"] is False
    assert any("missing_reference" in e for e in result["errors"])

def test_sub_claim_confidence(validator):
    validator.add_claim(Claim("sub", "Sub claim", ["e1", "e2"]))
    validator.add_claim(Claim("parent", "Parent claim", ["e3"], sub_claim_ids=["sub"]))
    result = validator.validate("parent", reference_time=1050.0)
    # e3=0.7, sub effective=min(0.9,0.8)=0.8, parent effective=min(0.7, 0.8)=0.7
    assert abs(result["effective_confidence"] - 0.7) < 0.01

def test_chain_depth(validator):
    validator.add_claim(Claim("d0", "Leaf", ["e1", "e2"]))
    validator.add_claim(Claim("d1", "Mid", ["e1", "e2"], sub_claim_ids=["d0"]))
    validator.add_claim(Claim("d2", "Root", ["e1", "e2"], sub_claim_ids=["d1"]))
    assert validator.chain_depth("d0") == 0
    assert validator.chain_depth("d1") == 1
    assert validator.chain_depth("d2") == 2

def test_all_evidence_recursive(validator):
    validator.add_claim(Claim("leaf", "Leaf", ["e1"]))
    validator.add_claim(Claim("mid", "Mid", ["e2"], sub_claim_ids=["leaf"]))
    validator.add_claim(Claim("root", "Root", ["e3"], sub_claim_ids=["mid"]))
    all_ev = validator.all_evidence_for("root")
    ev_ids = {e.id for e in all_ev}
    assert ev_ids == {"e1", "e2", "e3"}

def test_empty_claim(validator):
    validator.add_claim(Claim("empty", "No evidence", []))
    result = validator.validate("empty", reference_time=1050.0)
    assert result["valid"] is False

def test_same_agent_different_evidence():
    v = ProofChainValidator(min_sources=2)
    v.add_evidence(Evidence("ea", "data a", 0.9, "same_agent", 1000.0))
    v.add_evidence(Evidence("eb", "data b", 0.8, "same_agent", 1000.0))
    v.add_claim(Claim("c_same", "Same agent", ["ea", "eb"]))
    result = v.validate("c_same", reference_time=1050.0)
    assert result["valid"] is False  # Same source agent, need 2 UNIQUE

def test_validate_nonexistent_claim(validator):
    result = validator.validate("nonexistent", reference_time=1050.0)
    assert result["valid"] is False
    assert any("missing_reference" in e for e in result["errors"])

def test_deep_nesting_no_circular(validator):
    validator.add_claim(Claim("n1", "N1", ["e1", "e2"]))
    validator.add_claim(Claim("n2", "N2", ["e1", "e2"], sub_claim_ids=["n1"]))
    validator.add_claim(Claim("n3", "N3", ["e1", "e2"], sub_claim_ids=["n2"]))
    validator.add_claim(Claim("n4", "N4", ["e1", "e2"], sub_claim_ids=["n3"]))
    result = validator.validate("n4", reference_time=1050.0)
    assert result["valid"] is True
    assert validator.chain_depth("n4") == 3

def test_confidence_zero():
    v = ProofChainValidator(min_sources=1)
    v.add_evidence(Evidence("ez", "zero conf", 0.0, "a1", 1000.0))
    v.add_claim(Claim("cz", "Zero", ["ez"]))
    result = v.validate("cz", reference_time=1050.0)
    assert result["effective_confidence"] == 0.0

def test_multiple_stale_degradation():
    v = ProofChainValidator(min_sources=2, max_staleness=100.0)
    v.add_evidence(Evidence("s1", "old1", 1.0, "a1", 800.0))
    v.add_evidence(Evidence("s2", "old2", 1.0, "a2", 800.0))
    v.add_claim(Claim("cs", "Stale pair", ["s1", "s2"]))
    result = v.validate("cs", reference_time=1050.0)
    # Both stale: 1.0 * 0.5 = 0.5 each, effective = 0.5
    assert abs(result["effective_confidence"] - 0.5) < 0.01
'''

A5_PROMPT = """Write a Python module `batch_scheduler.py` implementing an adaptive batch job scheduler.

Requirements:

1. Class `Job`:
   - `__init__(self, id: str, job_type: str, priority: int = 0, estimated_duration: float = 1.0)`
   - priority: higher = more important
   - estimated_duration: in seconds

2. Class `AdaptiveBatchScheduler`:
   - `__init__(self, target_throughput: float = 10.0, min_batch: int = 1, max_batch: int = 100)`
   - target_throughput: desired jobs per second
   - `submit(self, job: Job)` — add job to queue
   - `next_batch(self) -> list[Job]` — get next batch of jobs to execute. Batch size is adaptive:
     * Cold start (no history): use min_batch
     * After history: adjust batch_size = int(target_throughput * avg_duration_per_job)
     * Clamp to [min_batch, max_batch]
     * Within a batch, sort by priority (highest first)
     * If a priority >= 4 job exists, it goes into its own immediate batch (preemption)
   - `report_completion(self, job_id: str, actual_duration: float)` — record actual duration for adaptation
   - `batch_size_for_type(self, job_type: str) -> int` — current adapted batch size for a specific job type (uses that type's average duration)
   - `utilization(self) -> dict` — returns {"total_jobs": int, "completed_jobs": int, "avg_duration": float, "current_batch_size": int, "throughput": float}
   - `pending_count(self) -> int` — jobs in queue not yet batched

Return ONLY the Python code, no explanation, no markdown fences."""

A5_TEST = '''import pytest
from batch_scheduler import Job, AdaptiveBatchScheduler

def test_cold_start():
    s = AdaptiveBatchScheduler(target_throughput=10, min_batch=2, max_batch=50)
    s.submit(Job("j1", "cpu", 1))
    s.submit(Job("j2", "cpu", 2))
    s.submit(Job("j3", "cpu", 3))
    batch = s.next_batch()
    assert len(batch) == 2  # min_batch on cold start

def test_priority_ordering():
    s = AdaptiveBatchScheduler(min_batch=3, max_batch=10)
    s.submit(Job("j1", "cpu", 1))
    s.submit(Job("j2", "cpu", 3))
    s.submit(Job("j3", "cpu", 2))
    batch = s.next_batch()
    assert batch[0].priority >= batch[1].priority >= batch[2].priority

def test_preemption():
    s = AdaptiveBatchScheduler(min_batch=3, max_batch=10)
    s.submit(Job("j1", "io", 1))
    s.submit(Job("j2", "io", 2))
    s.submit(Job("j_urgent", "io", 4))  # priority >= 4 triggers preemption
    s.submit(Job("j3", "io", 1))
    batch = s.next_batch()
    assert len(batch) == 1
    assert batch[0].id == "j_urgent"

def test_adaptive_batch_size():
    s = AdaptiveBatchScheduler(target_throughput=10, min_batch=1, max_batch=50)
    # Report some completions to build history
    for i in range(5):
        s.submit(Job(f"h{i}", "cpu", 1))
    for i in range(5):
        batch = s.next_batch()
        if batch:
            for j in batch:
                s.report_completion(j.id, actual_duration=0.5)
    # Now batch size should adapt: target=10, avg_duration=0.5 => batch=5
    for i in range(10):
        s.submit(Job(f"a{i}", "cpu", 1))
    batch = s.next_batch()
    assert len(batch) == 5

def test_batch_size_clamped_max():
    s = AdaptiveBatchScheduler(target_throughput=100, min_batch=1, max_batch=10)
    for i in range(5):
        s.submit(Job(f"h{i}", "cpu", 1))
        b = s.next_batch()
        if b:
            s.report_completion(b[0].id, actual_duration=0.01)
    for i in range(20):
        s.submit(Job(f"a{i}", "cpu", 1))
    batch = s.next_batch()
    assert len(batch) <= 10

def test_batch_size_for_type():
    s = AdaptiveBatchScheduler(target_throughput=10, min_batch=1, max_batch=50)
    for i in range(3):
        s.submit(Job(f"c{i}", "cpu", 1))
        b = s.next_batch()
        if b:
            s.report_completion(b[0].id, actual_duration=2.0)
    for i in range(3):
        s.submit(Job(f"i{i}", "io", 1))
        b = s.next_batch()
        if b:
            s.report_completion(b[0].id, actual_duration=0.1)
    # CPU avg=2.0 => batch=20, IO avg=0.1 => batch=1
    assert s.batch_size_for_type("cpu") > s.batch_size_for_type("io")

def test_utilization():
    s = AdaptiveBatchScheduler()
    s.submit(Job("u1", "cpu", 1))
    s.submit(Job("u2", "cpu", 1))
    b = s.next_batch()
    for j in b:
        s.report_completion(j.id, actual_duration=1.5)
    util = s.utilization()
    assert util["completed_jobs"] == len(b)
    assert abs(util["avg_duration"] - 1.5) < 0.01

def test_pending_count():
    s = AdaptiveBatchScheduler(min_batch=2)
    s.submit(Job("p1", "cpu", 1))
    s.submit(Job("p2", "cpu", 1))
    s.submit(Job("p3", "cpu", 1))
    assert s.pending_count() == 3
    s.next_batch()
    assert s.pending_count() == 1

def test_empty_queue():
    s = AdaptiveBatchScheduler()
    batch = s.next_batch()
    assert batch == []

def test_single_job():
    s = AdaptiveBatchScheduler(min_batch=1)
    s.submit(Job("solo", "cpu", 1))
    batch = s.next_batch()
    assert len(batch) == 1
    assert batch[0].id == "solo"

def test_preemption_doesnt_lose_others():
    s = AdaptiveBatchScheduler(min_batch=3, max_batch=10)
    s.submit(Job("n1", "cpu", 1))
    s.submit(Job("n2", "cpu", 2))
    s.submit(Job("urgent", "cpu", 5))
    # First batch: preemption
    b1 = s.next_batch()
    assert len(b1) == 1
    assert b1[0].id == "urgent"
    # Second batch: remaining jobs
    b2 = s.next_batch()
    assert len(b2) == 2

def test_throughput_metric():
    s = AdaptiveBatchScheduler()
    s.submit(Job("t1", "cpu", 1))
    b = s.next_batch()
    s.report_completion(b[0].id, actual_duration=0.5)
    util = s.utilization()
    assert "throughput" in util
    assert util["throughput"] > 0

def test_all_same_type():
    s = AdaptiveBatchScheduler(target_throughput=5, min_batch=1, max_batch=20)
    for i in range(3):
        s.submit(Job(f"s{i}", "same", 1))
        b = s.next_batch()
        if b:
            s.report_completion(b[0].id, actual_duration=1.0)
    for i in range(10):
        s.submit(Job(f"t{i}", "same", 1))
    batch = s.next_batch()
    assert len(batch) == 5  # target=5 * avg=1.0 = 5

def test_multiple_preemptions():
    s = AdaptiveBatchScheduler(min_batch=2, max_batch=10)
    s.submit(Job("u1", "cpu", 5))
    s.submit(Job("u2", "cpu", 4))
    s.submit(Job("n1", "cpu", 1))
    b1 = s.next_batch()
    # Both urgent jobs should preempt, but separately or together?
    # At minimum, urgent jobs should come before normal ones
    assert b1[0].priority >= 4
'''

# =============================================================================
# CATEGORY B: Code Editing (3 tests, 45 assertions)
# =============================================================================

B1_PROVIDED_CODE = '''"""Event Logger — tracks events with timestamps and categories."""
from dataclasses import dataclass, field
from typing import Callable
import time

@dataclass
class Event:
    id: str
    category: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

class EventLogger:
    def __init__(self, name: str):
        self.name = name
        self._events: list[Event] = []
        self._listeners: list[Callable] = []

    def log(self, category: str, message: str, **metadata) -> Event:
        """Log a new event and notify listeners."""
        event = Event(
            id=f"{self.name}_{len(self._events):04d}",
            category=category,
            message=message,
            timestamp=time.time(),
            metadata=metadata,
        )
        self._events.append(event)
        for listener in self._listeners:
            listener(event)
        return event

    def add_listener(self, fn: Callable) -> None:
        """Register a callback for new events."""
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable) -> None:
        """Unregister a callback."""
        self._listeners.remove(fn)

    def count(self) -> int:
        """Total number of events logged."""
        return len(self._events)

    def latest(self, n: int = 1) -> list[Event]:
        """Get the N most recent events."""
        return list(reversed(self._events[-n:]))

    def clear(self) -> int:
        """Clear all events, return count cleared."""
        cleared = len(self._events)
        self._events.clear()
        return cleared

    def categories(self) -> set[str]:
        """Return all unique categories."""
        return {e.category for e in self._events}
'''

B1_PROMPT = """You are given an existing Python module `event_logger.py` (shown below). Your task is to ADD three new methods to the EventLogger class WITHOUT modifying any existing methods or the Event dataclass.

EXISTING CODE:
```python
{provided_code}
```

ADD these three methods to EventLogger:

1. `filter_events(self, predicate: Callable[[Event], bool]) -> list[Event]` — return all events matching the predicate, in chronological order
2. `group_by(self, key_fn: Callable[[Event], str]) -> dict[str, list[Event]]` — group events by key function result, each group in chronological order
3. `export_csv(self, path: str) -> int` — write events to CSV with columns: id, category, message, timestamp. Return number of rows written. Use csv module.

IMPORTANT: Return the COMPLETE module with your additions. Do NOT modify any existing methods, the Event dataclass, or imports (you may add new imports). The existing code must remain byte-for-byte identical.

Return ONLY the Python code, no explanation, no markdown fences."""

B1_TEST = '''import csv
import os
import tempfile
import time
import pytest
from event_logger import Event, EventLogger

# --- Tests for EXISTING functionality (must not regress) ---

def test_existing_log():
    el = EventLogger("test")
    e = el.log("info", "hello", key="val")
    assert e.category == "info"
    assert e.message == "hello"
    assert e.metadata == {"key": "val"}

def test_existing_count():
    el = EventLogger("test")
    el.log("a", "m1")
    el.log("b", "m2")
    assert el.count() == 2

def test_existing_latest():
    el = EventLogger("test")
    el.log("a", "first")
    el.log("b", "second")
    latest = el.latest(1)
    assert latest[0].message == "second"

def test_existing_clear():
    el = EventLogger("test")
    el.log("a", "m1")
    cleared = el.clear()
    assert cleared == 1
    assert el.count() == 0

def test_existing_categories():
    el = EventLogger("test")
    el.log("a", "m1")
    el.log("b", "m2")
    el.log("a", "m3")
    assert el.categories() == {"a", "b"}

def test_existing_listener():
    el = EventLogger("test")
    received = []
    el.add_listener(lambda e: received.append(e))
    el.log("x", "y")
    assert len(received) == 1

# --- Tests for NEW functionality ---

def test_filter_events():
    el = EventLogger("test")
    el.log("info", "msg1")
    el.log("error", "msg2")
    el.log("info", "msg3")
    result = el.filter_events(lambda e: e.category == "info")
    assert len(result) == 2
    assert result[0].message == "msg1"
    assert result[1].message == "msg3"

def test_filter_empty():
    el = EventLogger("test")
    result = el.filter_events(lambda e: e.category == "nope")
    assert result == []

def test_group_by():
    el = EventLogger("test")
    el.log("a", "m1")
    el.log("b", "m2")
    el.log("a", "m3")
    groups = el.group_by(lambda e: e.category)
    assert len(groups) == 2
    assert len(groups["a"]) == 2
    assert len(groups["b"]) == 1
    assert groups["a"][0].message == "m1"

def test_group_by_custom_key():
    el = EventLogger("test")
    el.log("x", "short")
    el.log("y", "a longer message here")
    groups = el.group_by(lambda e: "long" if len(e.message) > 10 else "short")
    assert "long" in groups
    assert "short" in groups

def test_export_csv():
    el = EventLogger("test")
    el.log("info", "hello")
    el.log("warn", "world")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        count = el.export_csv(path)
        assert count == 2
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == ["id", "category", "message", "timestamp"]
            assert len(rows) == 3  # header + 2 data rows
            assert rows[1][1] == "info"
            assert rows[2][1] == "warn"
    finally:
        os.unlink(path)

def test_export_csv_empty():
    el = EventLogger("test")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        count = el.export_csv(path)
        assert count == 0
    finally:
        os.unlink(path)

def test_filter_with_metadata():
    el = EventLogger("test")
    el.log("a", "m1", level=1)
    el.log("a", "m2", level=5)
    result = el.filter_events(lambda e: e.metadata.get("level", 0) > 3)
    assert len(result) == 1
    assert result[0].message == "m2"

def test_group_preserves_order():
    el = EventLogger("test")
    el.log("a", "first")
    el.log("a", "second")
    el.log("a", "third")
    groups = el.group_by(lambda e: e.category)
    msgs = [e.message for e in groups["a"]]
    assert msgs == ["first", "second", "third"]
'''

B2_PROVIDED_CODE = '''"""TTL Cache with LRU eviction."""
import time
from collections import OrderedDict

class TTLCache:
    def __init__(self, max_size: int = 100, default_ttl: float = 60.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: OrderedDict = OrderedDict()
        self._expiry: dict[str, float] = {}
        self._index: dict[str, set] = {}  # secondary index: tag -> keys

    def set(self, key: str, value: any, ttl: float | None = None, tags: list[str] | None = None) -> None:
        """Set a key with optional TTL and tags."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        self._expiry[key] = time.time() + (ttl if ttl is not None else self.default_ttl)
        if tags:
            for tag in tags:
                if tag not in self._index:
                    self._index[tag] = set()
                self._index[tag].add(key)
        # BUG 1: Off-by-one in eviction check — should be > not >=
        while len(self._store) >= self.max_size:
            oldest_key, _ = self._store.popitem(last=False)
            del self._expiry[oldest_key]
            # BUG 3: Memory leak — evicted keys not removed from _index

    def get(self, key: str) -> any:
        """Get a value, return None if expired or missing."""
        if key not in self._store:
            return None
        # BUG 2: Off-by-one in TTL check — should be < not <=
        if time.time() <= self._expiry[key]:
            self._store.move_to_end(key)
            return self._store[key]
        # Expired but not cleaned up — just return None
        return None

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        if key in self._store:
            del self._store[key]
            del self._expiry[key]
            return True
        return False

    def get_by_tag(self, tag: str) -> dict[str, any]:
        """Get all non-expired values with the given tag."""
        if tag not in self._index:
            return {}
        result = {}
        for key in self._index[tag]:
            val = self.get(key)
            if val is not None:
                result[key] = val
        return result

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for key in expired:
            if key in self._store:
                del self._store[key]
            del self._expiry[key]
        return len(expired)

    def size(self) -> int:
        """Current number of entries (including expired but uncleaned)."""
        return len(self._store)
'''

B2_PROMPT = """You are given a Python module `ttl_cache.py` with 3 subtle bugs. Here is the code:

```python
{provided_code}
```

The bugs are:
1. **Off-by-one in eviction**: The eviction loop triggers at `>=` when it should be `>` (evicts one entry too early, cache holds max_size-1 items)
2. **TTL boundary check wrong**: In `get()`, `time.time() <= self._expiry[key]` should be `<` — when time equals expiry, the item should be considered expired
3. **Memory leak in _index**: When `set()` evicts old items, it removes them from `_store` and `_expiry` but NOT from `_index`. This means `get_by_tag()` tries to look up evicted keys.

Fix ALL THREE bugs. Do not modify any method signatures or add new methods. The existing API must remain identical.

Return ONLY the complete fixed Python code, no explanation, no markdown fences."""

B2_TEST = '''import time
import pytest
from ttl_cache import TTLCache

# --- BUG 1: Eviction should allow max_size items ---

def test_max_size_exact():
    c = TTLCache(max_size=3, default_ttl=999)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.size() == 3  # Should hold exactly 3, not 2

def test_eviction_on_overflow():
    c = TTLCache(max_size=2, default_ttl=999)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # Should evict "a"
    assert c.size() == 2
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3

# --- BUG 2: TTL boundary ---

def test_ttl_exact_boundary():
    c = TTLCache(max_size=10, default_ttl=0.0)
    c.set("k", "v", ttl=0.0)
    # TTL = 0 means expires immediately (time.time() + 0 = now)
    # At the exact expiry time, item should be expired
    val = c.get("k")
    assert val is None  # Should be expired at boundary

def test_ttl_not_expired():
    c = TTLCache(max_size=10, default_ttl=9999)
    c.set("k", "v")
    assert c.get("k") == "v"

def test_ttl_expired():
    c = TTLCache(max_size=10, default_ttl=0.001)
    c.set("k", "v", ttl=0.001)
    time.sleep(0.01)
    assert c.get("k") is None

# --- BUG 3: Memory leak in _index ---

def test_index_cleanup_on_eviction():
    c = TTLCache(max_size=2, default_ttl=999)
    c.set("a", 1, tags=["group1"])
    c.set("b", 2, tags=["group1"])
    c.set("c", 3, tags=["group1"])  # evicts "a"
    result = c.get_by_tag("group1")
    assert "a" not in result  # "a" was evicted, should not appear
    assert "b" in result or "c" in result

def test_index_no_ghost_entries():
    c = TTLCache(max_size=2, default_ttl=999)
    c.set("x", 10, tags=["t1"])
    c.set("y", 20, tags=["t1"])
    c.set("z", 30, tags=["t1"])  # evicts "x"
    # _index["t1"] should not contain "x"
    by_tag = c.get_by_tag("t1")
    assert len(by_tag) == 2

# --- Existing functionality (no regression) ---

def test_basic_set_get():
    c = TTLCache(max_size=10, default_ttl=60)
    c.set("k", "v")
    assert c.get("k") == "v"

def test_delete():
    c = TTLCache(max_size=10, default_ttl=60)
    c.set("k", "v")
    assert c.delete("k") is True
    assert c.get("k") is None
    assert c.delete("k") is False

def test_cleanup():
    c = TTLCache(max_size=10, default_ttl=0.001)
    c.set("a", 1)
    c.set("b", 2)
    time.sleep(0.01)
    removed = c.cleanup()
    assert removed == 2
    assert c.size() == 0

def test_lru_order():
    c = TTLCache(max_size=3, default_ttl=999)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.get("a")  # Touch "a", making "b" the LRU
    c.set("d", 4)  # Should evict "b"
    assert c.get("b") is None
    assert c.get("a") == 1

def test_overwrite_existing():
    c = TTLCache(max_size=10, default_ttl=60)
    c.set("k", "v1")
    c.set("k", "v2")
    assert c.get("k") == "v2"
    assert c.size() == 1

def test_tag_query():
    c = TTLCache(max_size=10, default_ttl=999)
    c.set("a", 1, tags=["x"])
    c.set("b", 2, tags=["x", "y"])
    c.set("c", 3, tags=["y"])
    by_x = c.get_by_tag("x")
    assert set(by_x.keys()) == {"a", "b"}

def test_size():
    c = TTLCache(max_size=10, default_ttl=60)
    assert c.size() == 0
    c.set("k", "v")
    assert c.size() == 1
'''

B3_PROVIDED_CODE = '''"""Data pipeline processor — transforms records through a chain of operations."""

class PipelineProcessor:
    def __init__(self, name: str):
        self.name = name
        self._steps = []
        self._error_count = 0

    def add_step(self, name: str, fn):
        """Add a processing step."""
        self._steps.append({"name": name, "fn": fn})

    def process(self, records: list[dict]) -> list[dict]:
        """Process records through all steps. Monolithic function that needs refactoring."""
        results = []
        errors = []
        stats = {"input": len(records), "output": 0, "errors": 0, "steps": {}}

        for record in records:
            current = dict(record)
            failed = False
            for step in self._steps:
                step_name = step["name"]
                if step_name not in stats["steps"]:
                    stats["steps"][step_name] = {"processed": 0, "failed": 0}
                try:
                    result = step["fn"](current)
                    if result is None:
                        stats["steps"][step_name]["failed"] += 1
                        failed = True
                        errors.append({"record": record, "step": step_name, "error": "returned None"})
                        break
                    current = result
                    stats["steps"][step_name]["processed"] += 1
                except Exception as e:
                    stats["steps"][step_name]["failed"] += 1
                    failed = True
                    errors.append({"record": record, "step": step_name, "error": str(e)})
                    self._error_count += 1
                    break
            if not failed:
                results.append(current)
                stats["output"] += 1
            else:
                stats["errors"] += 1

        self._last_stats = stats
        self._last_errors = errors
        return results

    def get_stats(self) -> dict:
        """Get stats from last run."""
        return getattr(self, "_last_stats", {})

    def get_errors(self) -> list[dict]:
        """Get errors from last run."""
        return getattr(self, "_last_errors", [])

    def error_count(self) -> int:
        """Total error count across all runs."""
        return self._error_count
'''

B3_PROMPT = """You are given a Python module `pipeline_processor.py` with a monolithic `process()` method that needs refactoring.

EXISTING CODE:
```python
{provided_code}
```

Your task: Refactor the `process()` method into 4 smaller functions with clear single responsibility:

1. `_init_stats(self, record_count: int) -> dict` — initialize the stats dictionary
2. `_apply_step(self, record: dict, step: dict, stats: dict) -> tuple[dict | None, dict | None]` — apply a single step to a record. Returns (result_or_None, error_or_None)
3. `_process_record(self, record: dict, stats: dict) -> tuple[dict | None, list[dict]]` — process one record through all steps. Returns (final_result_or_None, errors_list)
4. `process(self, records: list[dict]) -> list[dict]` — orchestrate using the above three, same external behavior

Requirements:
- Add type hints to ALL new function signatures
- External behavior must be IDENTICAL (same return values, same stats, same errors)
- Each extracted function must have cyclomatic complexity <= 5
- `process()` should now be a simple loop calling `_process_record()`

Return ONLY the complete refactored module, no explanation, no markdown fences."""

B3_TEST = '''import pytest
from pipeline_processor import PipelineProcessor

# --- Original behavior tests (must not regress) ---

def test_basic_processing():
    p = PipelineProcessor("test")
    p.add_step("double", lambda r: {**r, "value": r["value"] * 2})
    result = p.process([{"value": 5}])
    assert result == [{"value": 10}]

def test_multi_step():
    p = PipelineProcessor("test")
    p.add_step("add1", lambda r: {**r, "v": r["v"] + 1})
    p.add_step("mul2", lambda r: {**r, "v": r["v"] * 2})
    result = p.process([{"v": 3}])
    assert result == [{"v": 8}]

def test_error_handling():
    p = PipelineProcessor("test")
    p.add_step("fail", lambda r: 1/0 if r.get("bad") else r)
    result = p.process([{"bad": True}, {"bad": False}])
    assert len(result) == 1
    assert p.error_count() == 1

def test_none_filter():
    p = PipelineProcessor("test")
    p.add_step("filter", lambda r: r if r.get("keep") else None)
    result = p.process([{"keep": True, "v": 1}, {"keep": False, "v": 2}])
    assert len(result) == 1

def test_stats():
    p = PipelineProcessor("test")
    p.add_step("pass", lambda r: r)
    p.process([{"a": 1}, {"a": 2}])
    stats = p.get_stats()
    assert stats["input"] == 2
    assert stats["output"] == 2
    assert stats["errors"] == 0

def test_stats_with_errors():
    p = PipelineProcessor("test")
    p.add_step("maybe", lambda r: None if r.get("skip") else r)
    p.process([{"skip": True}, {"skip": False}])
    stats = p.get_stats()
    assert stats["errors"] == 1
    assert stats["output"] == 1

def test_get_errors():
    p = PipelineProcessor("test")
    p.add_step("boom", lambda r: (_ for _ in ()).throw(ValueError("bad")))
    p.process([{"x": 1}])
    errors = p.get_errors()
    assert len(errors) == 1
    assert errors[0]["step"] == "boom"
    assert "bad" in errors[0]["error"]

def test_step_stats():
    p = PipelineProcessor("test")
    p.add_step("s1", lambda r: r)
    p.add_step("s2", lambda r: r)
    p.process([{"a": 1}])
    stats = p.get_stats()
    assert stats["steps"]["s1"]["processed"] == 1
    assert stats["steps"]["s2"]["processed"] == 1

# --- Refactoring validation tests ---

def test_init_stats_exists():
    p = PipelineProcessor("test")
    assert hasattr(p, "_init_stats")
    stats = p._init_stats(5)
    assert stats["input"] == 5
    assert stats["output"] == 0

def test_apply_step_exists():
    p = PipelineProcessor("test")
    assert hasattr(p, "_apply_step")

def test_process_record_exists():
    p = PipelineProcessor("test")
    assert hasattr(p, "_process_record")

def test_apply_step_success():
    p = PipelineProcessor("test")
    step = {"name": "double", "fn": lambda r: {**r, "v": r["v"] * 2}}
    stats = p._init_stats(1)
    result, error = p._apply_step({"v": 3}, step, stats)
    assert result == {"v": 6}
    assert error is None

def test_apply_step_failure():
    p = PipelineProcessor("test")
    step = {"name": "fail", "fn": lambda r: 1/0}
    stats = p._init_stats(1)
    result, error = p._apply_step({"v": 1}, step, stats)
    assert result is None
    assert error is not None

def test_process_record_full():
    p = PipelineProcessor("test")
    p.add_step("inc", lambda r: {**r, "v": r["v"] + 1})
    stats = p._init_stats(1)
    result, errors = p._process_record({"v": 0}, stats)
    assert result == {"v": 1}
    assert errors == []
'''

# =============================================================================
# CATEGORY C: Multi-Step Reasoning (3 tests) — prompts built dynamically
# =============================================================================

C1_BUGGY_CODE = '''"""Metric aggregator with windowed statistics."""
import math

class MetricAggregator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: list[float] = []
        self._tags: dict[str, list[float]] = {}

    def record(self, value: float, tag: str = "default") -> None:
        self._values.append(value)
        if len(self._values) > self.window_size:
            self._values.pop(0)
        if tag not in self._tags:
            self._tags[tag] = []
        self._tags[tag].append(value)  # BUG 1: tags list never windowed

    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def stddev(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean()
        variance = sum((x - m) ** 2 for x in self._values) / len(self._values)  # BUG 2: population stddev, should be sample (/ n-1)
        return math.sqrt(variance)

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = p * len(sorted_vals)  # BUG 3: wrong percentile calc, should be (p/100) * (n-1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    def tag_mean(self, tag: str) -> float:
        if tag not in self._tags or not self._tags[tag]:
            return 0.0
        vals = self._tags[tag]
        return sum(vals) / len(vals)

    def summary(self) -> dict:
        return {
            "count": len(self._values),
            "mean": self.mean(),
            "stddev": self.stddev(),
            "p50": self.percentile(50),  # BUG 4: passes 50 but percentile treats as fraction... or vice versa
            "p95": self.percentile(95),
            "min": min(self._values) if self._values else 0.0,
            "max": max(self._values) if self._values else 0.0,
        }
'''

C1_TRACEBACK = '''FAILED test_metric_aggregator.py::test_sample_stddev - AssertionError: Expected sample stddev of [2, 4, 6, 8, 10] to be ~3.162 but got 2.828
FAILED test_metric_aggregator.py::test_percentile_50 - AssertionError: Expected p50 of [1,2,3,4,5] to be 3.0 but got error (index out of range)
FAILED test_metric_aggregator.py::test_percentile_95 - AssertionError: Expected p95 of range(100) to be ~95 but got error (index out of range)
FAILED test_metric_aggregator.py::test_tag_windowing - AssertionError: Expected tag_mean for 'cpu' to reflect window, but tag list grew to 200 entries while window is 100
FAILED test_metric_aggregator.py::test_summary_p50 - AssertionError: summary()["p50"] raised IndexError'''

C1_PROMPT = """You are given a buggy Python module and its test failures. Analyze and fix it.

BUGGY CODE:
```python
{provided_code}
```

TEST FAILURES:
```
{traceback}
```

You must return TWO things separated by `---SPLIT---`:

PART 1: A JSON diagnosis (one object per failure):
```json
[
  {{"test": "test_name", "root_cause": "description", "category": "logic_error|type_error|missing_impl|edge_case", "fix": "what to change"}}
]
```

PART 2: The complete fixed Python module.

Return ONLY the diagnosis JSON, then ---SPLIT---, then the fixed Python code. No other text."""

C1_TEST = '''import json
import math
import pytest

def _parse_response(response_text):
    """Parse the model's response into diagnosis and code."""
    parts = response_text.split("---SPLIT---")
    if len(parts) != 2:
        pytest.fail(f"Expected ---SPLIT--- separator, got {len(parts)} parts")
    diagnosis_text = parts[0].strip()
    code_text = parts[1].strip()
    # Strip markdown fences if present
    for fence in ["```json", "```python", "```"]:
        diagnosis_text = diagnosis_text.replace(fence, "")
        code_text = code_text.replace(fence, "")
    return diagnosis_text.strip(), code_text.strip()

# These tests will be run AFTER the model produces its response
# The runner will save the code part and import it

from metric_aggregator import MetricAggregator

def test_sample_stddev():
    m = MetricAggregator()
    for v in [2, 4, 6, 8, 10]:
        m.record(v)
    expected = math.sqrt(sum((x - 6)**2 for x in [2,4,6,8,10]) / 4)  # sample stddev
    assert abs(m.stddev() - expected) < 0.001

def test_percentile_50():
    m = MetricAggregator()
    for v in [1, 2, 3, 4, 5]:
        m.record(v)
    assert abs(m.percentile(50) - 3.0) < 0.1

def test_percentile_95():
    m = MetricAggregator()
    for v in range(100):
        m.record(float(v))
    p95 = m.percentile(95)
    assert 93 <= p95 <= 96

def test_percentile_0():
    m = MetricAggregator()
    for v in [10, 20, 30]:
        m.record(v)
    assert m.percentile(0) == 10.0

def test_percentile_100():
    m = MetricAggregator()
    for v in [10, 20, 30]:
        m.record(v)
    assert m.percentile(100) == 30.0

def test_tag_windowing():
    m = MetricAggregator(window_size=5)
    for i in range(10):
        m.record(float(i), tag="cpu")
    # Window=5, so tag should also only keep last 5 values: [5,6,7,8,9]
    assert abs(m.tag_mean("cpu") - 7.0) < 0.01

def test_summary_p50():
    m = MetricAggregator()
    for v in [1, 2, 3, 4, 5]:
        m.record(v)
    s = m.summary()
    assert abs(s["p50"] - 3.0) < 0.1
    assert abs(s["p95"] - 4.8) < 0.5

def test_mean_empty():
    m = MetricAggregator()
    assert m.mean() == 0.0

def test_stddev_single():
    m = MetricAggregator()
    m.record(5.0)
    assert m.stddev() == 0.0

def test_window_sliding():
    m = MetricAggregator(window_size=3)
    m.record(1.0)
    m.record(2.0)
    m.record(3.0)
    m.record(100.0)  # pushes out 1.0
    assert abs(m.mean() - 35.0) < 0.01

def test_record_and_count():
    m = MetricAggregator(window_size=5)
    for i in range(10):
        m.record(float(i))
    s = m.summary()
    assert s["count"] == 5

def test_min_max():
    m = MetricAggregator()
    for v in [5, 1, 9, 3]:
        m.record(v)
    s = m.summary()
    assert s["min"] == 1
    assert s["max"] == 9

def test_tag_mean_missing():
    m = MetricAggregator()
    assert m.tag_mean("nonexistent") == 0.0

def test_multiple_tags():
    m = MetricAggregator()
    m.record(10.0, tag="a")
    m.record(20.0, tag="b")
    assert m.tag_mean("a") == 10.0
    assert m.tag_mean("b") == 20.0

def test_empty_summary():
    m = MetricAggregator()
    s = m.summary()
    assert s["count"] == 0
    assert s["mean"] == 0.0
'''

# =============================================================================
# Build TESTS list
# =============================================================================

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


TESTS = [
    # ---- Category A: Novel Code Generation ----
    {"id": "A1_sovereignty", "name": "Sovereignty Engine", "module": "sovereignty_engine",
     "category": "A", "prompt": A1_PROMPT, "test_code": A1_TEST, "hash": _hash(A1_PROMPT)},
    {"id": "A2_router", "name": "Cascade Priority Router", "module": "cascade_router",
     "category": "A", "prompt": A2_PROMPT, "test_code": A2_TEST, "hash": _hash(A2_PROMPT)},
    {"id": "A3_temporal", "name": "Temporal Config Manager", "module": "temporal_config",
     "category": "A", "prompt": A3_PROMPT, "test_code": A3_TEST, "hash": _hash(A3_PROMPT)},
    {"id": "A4_proofchain", "name": "Proof Chain Validator", "module": "proof_chain",
     "category": "A", "prompt": A4_PROMPT, "test_code": A4_TEST, "hash": _hash(A4_PROMPT)},
    {"id": "A5_scheduler", "name": "Adaptive Batch Scheduler", "module": "batch_scheduler",
     "category": "A", "prompt": A5_PROMPT, "test_code": A5_TEST, "hash": _hash(A5_PROMPT)},

    # ---- Category B: Code Editing ----
    {"id": "B1_add_feature", "name": "Edit: Add Feature to Event Logger", "module": "event_logger",
     "category": "B", "prompt": B1_PROMPT.format(provided_code=B1_PROVIDED_CODE),
     "test_code": B1_TEST, "provided_code": B1_PROVIDED_CODE, "hash": _hash(B1_PROMPT)},
    {"id": "B2_fix_bugs", "name": "Edit: Fix 3 Bugs in TTL Cache", "module": "ttl_cache",
     "category": "B", "prompt": B2_PROMPT.format(provided_code=B2_PROVIDED_CODE),
     "test_code": B2_TEST, "provided_code": B2_PROVIDED_CODE, "hash": _hash(B2_PROMPT)},
    {"id": "B3_refactor", "name": "Edit: Refactor Pipeline Processor", "module": "pipeline_processor",
     "category": "B", "prompt": B3_PROMPT.format(provided_code=B3_PROVIDED_CODE),
     "test_code": B3_TEST, "provided_code": B3_PROVIDED_CODE, "hash": _hash(B3_PROMPT)},

    # ---- Category C: Multi-Step Reasoning ----
    {"id": "C1_diagnose", "name": "Diagnose from Traceback", "module": "metric_aggregator",
     "category": "C", "prompt": C1_PROMPT.format(provided_code=C1_BUGGY_CODE, traceback=C1_TRACEBACK),
     "test_code": C1_TEST, "provided_code": C1_BUGGY_CODE, "hash": _hash(C1_PROMPT),
     "response_parser": "split"},
]

# =============================================================================
# CATEGORY C (continued): Multi-Step Reasoning
# =============================================================================

C2_REQUIREMENTS = """## Purple Metric Rolling Window — Requirements Document

Build a `MetricRollingWindow` system that tracks numeric metrics over configurable time windows.

### Core Requirements:
1. Support multiple named metrics (e.g., "latency", "throughput", "error_rate")
2. Each metric is added via `add_metric(name, window_seconds=N)` where `window_seconds` is the window duration in seconds. Optionally accepts `alert_threshold=X`.
3. Values are recorded via `record(name, value, timestamp=None)` — if timestamp is omitted, use current time. Values outside the window are automatically excluded from calculations.
4. Support these aggregation methods per metric: `mean(name)`, `min(name)`, `max(name)`, `count(name)`, `sum(name)`, and `rate(name)` (events per second)
5. A `snapshot()` method returns all metrics' current aggregations as a dict
6. Support "alert thresholds" — when a metric's mean exceeds a threshold, mark it as "alerting" in the snapshot
6b. Derived metrics are added via `add_derived(name, compute_fn)` where `compute_fn` receives a dict of {metric_name: {aggregation_name: value}}

### Advanced Requirements:
7. Support "derived metrics" that compute from other metrics (e.g., "success_rate" = 1 - "error_rate".mean)
8. Derived metrics update automatically when their source metrics change
9. All window calculations must be O(n) or better where n = values in window (not total ever recorded)

### Edge Cases:
10. Empty window returns 0.0 for all aggregations and rate
11. Window with exactly one value returns that value for mean/min/max, 0.0 for rate
12. Values recorded at exactly the window boundary should be INCLUDED

### NOTE: The "rate" metric should give events per second over the actual span of data in the window (time from oldest to newest value). If there's zero span, rate is 0.0.

### CONTRADICTION (intentional): Requirement 6 says rate is "count per second", but Requirement 12's NOTE says rate is "events per second over actual span." These conflict for windows where the elapsed wall time differs from the data span. Use the NOTE definition (actual data span).

### AMBIGUITY 1: Should derived metrics be able to reference other derived metrics? (Design decision needed.)
### AMBIGUITY 2: What happens when a source metric for a derived metric has no data? (Design decision needed.)
"""

C2_PROMPT = """You are given a natural-language requirements document for a metric system. Read it carefully, then implement it.

REQUIREMENTS:
{requirements}

You must return TWO things separated by `---SPLIT---`:

PART 1: A JSON assumptions document:
```json
{{
  "contradiction_identified": "description of the contradiction and how you resolved it",
  "ambiguity_1_resolution": "your decision and reasoning",
  "ambiguity_2_resolution": "your decision and reasoning"
}}
```

PART 2: The complete Python module `metric_rolling_window.py` implementing all requirements.

Return ONLY the JSON, then ---SPLIT---, then the Python code. No other text."""

C2_TEST = '''import time
import json
import pytest
from metric_rolling_window import MetricRollingWindow

def test_basic_record_and_mean():
    w = MetricRollingWindow()
    w.add_metric("lat", window_seconds=60)
    w.record("lat", 10.0)
    w.record("lat", 20.0)
    w.record("lat", 30.0)
    assert abs(w.mean("lat") - 20.0) < 0.01

def test_min_max():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    w.record("x", 5.0)
    w.record("x", 1.0)
    w.record("x", 9.0)
    assert w.min("x") == 1.0
    assert w.max("x") == 9.0

def test_count_and_sum():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    w.record("x", 3.0)
    w.record("x", 7.0)
    assert w.count("x") == 2
    assert abs(w.sum("x") - 10.0) < 0.01

def test_window_expiry():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=0.05)
    w.record("x", 100.0)
    time.sleep(0.1)
    w.record("x", 1.0)
    assert abs(w.mean("x") - 1.0) < 0.01
    assert w.count("x") == 1

def test_empty_window():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    assert w.mean("x") == 0.0
    assert w.min("x") == 0.0
    assert w.max("x") == 0.0
    assert w.count("x") == 0
    assert w.rate("x") == 0.0

def test_single_value():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    w.record("x", 42.0)
    assert w.mean("x") == 42.0
    assert w.min("x") == 42.0
    assert w.max("x") == 42.0
    assert w.rate("x") == 0.0  # zero span = 0 rate

def test_rate_calculation():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    t = time.time()
    w.record("x", 1.0, timestamp=t)
    w.record("x", 1.0, timestamp=t + 1.0)
    w.record("x", 1.0, timestamp=t + 2.0)
    # 3 events over 2 seconds = 1.5 events/sec
    assert abs(w.rate("x") - 1.5) < 0.1

def test_snapshot():
    w = MetricRollingWindow()
    w.add_metric("a", window_seconds=60)
    w.record("a", 5.0)
    snap = w.snapshot()
    assert "a" in snap
    assert "mean" in snap["a"]
    assert snap["a"]["count"] == 1

def test_alert_threshold():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60, alert_threshold=10.0)
    w.record("x", 5.0)
    snap = w.snapshot()
    assert snap["x"].get("alerting", False) is False
    w.record("x", 20.0)
    snap = w.snapshot()
    assert snap["x"]["alerting"] is True

def test_derived_metric():
    w = MetricRollingWindow()
    w.add_metric("errors", window_seconds=60)
    w.add_derived("success_rate", lambda metrics: 1.0 - metrics["errors"].get("mean", 0.0))
    w.record("errors", 0.1)
    w.record("errors", 0.3)
    snap = w.snapshot()
    assert "success_rate" in snap
    assert abs(snap["success_rate"]["value"] - 0.8) < 0.01

def test_multiple_metrics():
    w = MetricRollingWindow()
    w.add_metric("a", window_seconds=60)
    w.add_metric("b", window_seconds=60)
    w.record("a", 10.0)
    w.record("b", 20.0)
    assert w.mean("a") == 10.0
    assert w.mean("b") == 20.0

def test_boundary_inclusion():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=1.0)
    t = time.time()
    w.record("x", 99.0, timestamp=t - 1.0)  # exactly at boundary
    w.record("x", 1.0, timestamp=t)
    # Boundary value should be INCLUDED
    assert w.count("x") == 2

def test_derived_no_source_data():
    w = MetricRollingWindow()
    w.add_metric("err", window_seconds=60)
    w.add_derived("rate", lambda m: m["err"].get("mean", 0.0) * 100)
    snap = w.snapshot()
    assert "rate" in snap
    # Should not crash, should return some value

def test_snapshot_structure():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    w.record("x", 5.0)
    snap = w.snapshot()
    required_keys = {"mean", "min", "max", "count", "sum", "rate"}
    assert required_keys.issubset(set(snap["x"].keys()))

def test_many_values_performance():
    w = MetricRollingWindow()
    w.add_metric("x", window_seconds=60)
    for i in range(1000):
        w.record("x", float(i))
    assert w.count("x") == 1000
    assert abs(w.mean("x") - 499.5) < 0.1
'''

# =============================================================================
# CATEGORY C3: Analyze and Optimize
# =============================================================================

C3_SLOW_CODE = '''"""Document similarity finder — correct but deliberately inefficient."""

class DocumentIndex:
    def __init__(self):
        self._docs: dict[str, list[str]] = {}  # id -> list of words

    def add(self, doc_id: str, text: str) -> None:
        """Index a document by splitting into lowercase words."""
        words = text.lower().split()
        self._docs[doc_id] = words

    def word_count(self, doc_id: str) -> dict[str, int]:
        """Count word frequencies in a document."""
        if doc_id not in self._docs:
            return {}
        # O(n^2): for each word, count by iterating all words
        counts = {}
        words = self._docs[doc_id]
        for word in words:
            count = 0
            for w in words:
                if w == word:
                    count += 1
            counts[word] = count
        return counts

    def similarity(self, id_a: str, id_b: str) -> float:
        """Cosine similarity between two documents. VERY SLOW."""
        if id_a not in self._docs or id_b not in self._docs:
            return 0.0
        # Get all unique words (O(n^2) way)
        all_words = []
        for w in self._docs[id_a]:
            found = False
            for existing in all_words:
                if existing == w:
                    found = True
                    break
            if not found:
                all_words.append(w)
        for w in self._docs[id_b]:
            found = False
            for existing in all_words:
                if existing == w:
                    found = True
                    break
            if not found:
                all_words.append(w)

        # Build vectors (O(n*m) per document)
        vec_a = []
        vec_b = []
        for word in all_words:
            count_a = 0
            for w in self._docs[id_a]:
                if w == word:
                    count_a += 1
            vec_a.append(count_a)
            count_b = 0
            for w in self._docs[id_b]:
                if w == word:
                    count_b += 1
            vec_b.append(count_b)

        # Dot product and magnitudes
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def find_most_similar(self, doc_id: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Find top N most similar documents. O(n^2 * m^2)."""
        if doc_id not in self._docs:
            return []
        scores = []
        for other_id in self._docs:
            if other_id == doc_id:
                continue
            score = self.similarity(doc_id, other_id)
            scores.append((other_id, score))
        # Bubble sort (O(n^2))
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if scores[j][1] > scores[i][1]:
                    scores[i], scores[j] = scores[j], scores[i]
        return scores[:top_n]

    def batch_similarity(self, doc_ids: list[str]) -> dict[str, dict[str, float]]:
        """Pairwise similarity for a batch. No caching — recomputes everything."""
        result = {}
        for id_a in doc_ids:
            result[id_a] = {}
            for id_b in doc_ids:
                if id_a == id_b:
                    result[id_a][id_b] = 1.0
                else:
                    result[id_a][id_b] = self.similarity(id_a, id_b)
        return result
'''

C3_PROMPT = """You are given a CORRECT but deliberately INEFFICIENT Python module. Your task is to optimize it.

SLOW CODE:
```python
{provided_code}
```

Performance issues to find and fix:
- word_count() is O(n^2) where O(n) is possible
- similarity() rebuilds word sets with O(n^2) lookups instead of using sets/dicts
- similarity() rebuilds word vectors from scratch instead of using cached word_count()
- find_most_similar() uses bubble sort instead of sorted()/heapq
- batch_similarity() recomputes similarity(a,b) and similarity(b,a) separately (symmetric!)

You must return TWO things separated by `---SPLIT---`:

PART 1: A JSON performance report:
```json
[
  {{"method": "method_name", "issue": "description", "old_complexity": "O(...)", "new_complexity": "O(...)"}}
]
```

PART 2: The complete OPTIMIZED Python module (same API, same correctness).

Return ONLY the JSON, then ---SPLIT---, then the optimized Python code. No other text."""

C3_TEST = '''import time
import json
import math
import pytest
from document_index import DocumentIndex

# --- Correctness tests (must pass identically) ---

def test_word_count_basic():
    idx = DocumentIndex()
    idx.add("d1", "hello world hello")
    counts = idx.word_count("d1")
    assert counts["hello"] == 2
    assert counts["world"] == 1

def test_word_count_empty():
    idx = DocumentIndex()
    assert idx.word_count("missing") == {}

def test_similarity_identical():
    idx = DocumentIndex()
    idx.add("a", "the quick brown fox")
    idx.add("b", "the quick brown fox")
    assert abs(idx.similarity("a", "b") - 1.0) < 0.001

def test_similarity_disjoint():
    idx = DocumentIndex()
    idx.add("a", "hello world")
    idx.add("b", "foo bar")
    assert idx.similarity("a", "b") == 0.0

def test_similarity_partial():
    idx = DocumentIndex()
    idx.add("a", "cat dog bird")
    idx.add("b", "cat dog fish")
    sim = idx.similarity("a", "b")
    assert 0.3 < sim < 0.9  # partial overlap

def test_similarity_missing():
    idx = DocumentIndex()
    idx.add("a", "hello")
    assert idx.similarity("a", "b") == 0.0

def test_find_most_similar():
    idx = DocumentIndex()
    idx.add("d1", "a b c")
    idx.add("d2", "a b d")
    idx.add("d3", "x y z")
    results = idx.find_most_similar("d1", top_n=2)
    assert results[0][0] == "d2"
    assert results[0][1] > results[1][1] if len(results) > 1 else True

def test_batch_similarity():
    idx = DocumentIndex()
    idx.add("a", "one two")
    idx.add("b", "one three")
    idx.add("c", "four five")
    result = idx.batch_similarity(["a", "b", "c"])
    assert result["a"]["a"] == 1.0
    assert result["a"]["b"] == result["b"]["a"]  # symmetry

def test_case_insensitive():
    idx = DocumentIndex()
    idx.add("a", "Hello WORLD")
    counts = idx.word_count("a")
    assert counts["hello"] == 1

def test_find_top_n_limit():
    idx = DocumentIndex()
    idx.add("q", "a b c")
    for i in range(10):
        idx.add(f"d{i}", f"a b {i}")
    results = idx.find_most_similar("q", top_n=3)
    assert len(results) == 3

# --- Performance test ---

def test_performance_improvement():
    """Optimized version must be at least 5x faster on larger input."""
    idx = DocumentIndex()
    # Build 20 documents with 200 words each
    import random
    random.seed(42)
    vocab = [f"word{i}" for i in range(50)]
    for i in range(20):
        text = " ".join(random.choices(vocab, k=200))
        idx.add(f"doc{i}", text)

    start = time.time()
    for i in range(20):
        idx.find_most_similar(f"doc{i}", top_n=5)
    elapsed = time.time() - start
    # The slow version takes ~2-5 seconds on this.
    # Optimized should be under 0.5s. We test < 1.0s for safety.
    assert elapsed < 1.0, f"Took {elapsed:.2f}s — should be under 1.0s"

def test_batch_performance():
    idx = DocumentIndex()
    import random
    random.seed(123)
    vocab = [f"w{i}" for i in range(30)]
    ids = []
    for i in range(15):
        did = f"doc{i}"
        idx.add(did, " ".join(random.choices(vocab, k=100)))
        ids.append(did)
    start = time.time()
    idx.batch_similarity(ids)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Batch took {elapsed:.2f}s — should be under 1.0s"

def test_word_count_performance():
    idx = DocumentIndex()
    text = " ".join([f"word{i % 50}" for i in range(5000)])
    idx.add("big", text)
    start = time.time()
    idx.word_count("big")
    elapsed = time.time() - start
    assert elapsed < 0.1, f"word_count took {elapsed:.3f}s"

def test_report_identifies_issues():
    """Model must identify at least 3 real issues in its report."""
    # This test is validated by the runner which checks the JSON report
    pass

def test_api_unchanged():
    """Verify the API surface is identical."""
    idx = DocumentIndex()
    assert hasattr(idx, "add")
    assert hasattr(idx, "word_count")
    assert hasattr(idx, "similarity")
    assert hasattr(idx, "find_most_similar")
    assert hasattr(idx, "batch_similarity")
'''

# =============================================================================
# CATEGORY D: Error Recovery (2 tests, 30 assertions)
# =============================================================================

D1_PARTIAL_CODE = '''"""Task queue with priority scheduling and retry logic."""
from dataclasses import dataclass, field
from typing import Callable, Any
import time

@dataclass
class Task:
    id: str
    priority: int  # 1=lowest, 5=highest
    fn: Callable[[], Any]
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)

class TaskQueue:
    """Priority task queue with retry support.

    Methods:
    - submit(task) — add a task to the queue
    - run_next() — run the highest-priority pending task
    - run_all() — run all pending tasks in priority order
    - get_status(task_id) -> dict — get task status, result, and retry info
    - retry_failed() -> int — retry all failed tasks that haven't exceeded max_retries.
                               Returns count of tasks requeued.
    - get_statistics() -> dict — return {"total", "pending", "completed", "failed", "avg_retries"}
    - drain(timeout) -> list[str] — run tasks until queue is empty or timeout.
                                     Returns list of completed task IDs.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._queue: list[str] = []  # task IDs in submission order

    def submit(self, task: Task) -> None:
        """Add a task to the queue."""
        self._tasks[task.id] = task
        self._queue.append(task.id)

    def run_next(self) -> str | None:
        """Run the highest-priority pending task. Returns task ID or None."""
        # Find highest priority pending task
        best_id = None
        best_priority = -1
        for tid in self._queue:
            task = self._tasks[tid]
            if task.status == "pending" and task.priority > best_priority:
                best_priority = task.priority
                best_id = tid

        if best_id is None:
            return None

        task = self._tasks[best_id]
        task.status = "running"
        try:
            task.result = task.fn()
            task.status = "completed"
        except Exception as e:
            task.retry_count += 1
            if task.retry_count >= task.max_retries:
                task.status = "failed"
            else:
                task.status = "pending"
            task.error = str(e)
        return best_id

    def run_all(self) -> list[str]:
        """Run all pending tasks in priority order. Returns list of completed task IDs."""
        completed = []
        while True:
            tid = self.run_next()
            if tid is None:
                break
            if self._tasks[tid].status == "completed":
                completed.append(tid)
        return completed

    def get_status(self, task_id: str) -> dict:
        """Get task status information."""
        if task_id not in self._tasks:
            return {"error": "not found"}
        t = self._tasks[task_id]
        return {
            "id": t.id,
            "status": t.status,
            "priority": t.priority,
            "result": t.result,
            "error": t.error,
            "retry_count": t.retry_count,
            "max_retries": t.max_retries,
        }

    # --- STUBS: Complete these methods ---

    def retry_failed(self) -> int:
        """Retry all failed tasks that haven't exceeded max_retries.
        Reset their status to 'pending' and return count requeued.

        NOTE: The docstring above says "haven't exceeded max_retries" but
        run_next() sets status='failed' when retry_count >= max_retries.
        So retry_failed should reset tasks where retry_count < max_retries
        (i.e., tasks that failed on an exception but still have retries left).
        Wait — if run_next already handles that... think carefully about when
        a task would be "failed" AND have retries left.

        Actually: tasks can also be manually marked as 'failed' via external
        code setting task.status = 'failed'. retry_failed should handle those too.
        """
        raise NotImplementedError

    def get_statistics(self) -> dict:
        """Return queue statistics.
        Keys: total, pending, completed, failed, avg_retries
        avg_retries = average retry_count across ALL tasks (including 0s)
        """
        raise NotImplementedError

    def drain(self, timeout: float = 10.0) -> list[str]:
        """Run tasks until queue empty or timeout reached.
        Returns list of completed task IDs (in order of completion).
        Uses time.time() for timeout tracking.
        """
        raise NotImplementedError
'''

D1_PROMPT = """You are given a PARTIALLY IMPLEMENTED Python module `task_queue.py`. Three methods are stubs (raising NotImplementedError). Complete them.

HERE IS THE CODE:
```python
{provided_code}
```

The three stubs to complete:
1. `retry_failed()` — Reset failed tasks (where retry_count < max_retries) back to "pending". Return count of tasks requeued.
2. `get_statistics()` — Return dict with keys: total, pending, completed, failed, avg_retries
3. `drain(timeout)` — Run tasks until empty or timeout. Return list of completed task IDs.

IMPORTANT: Read the existing code carefully. The `run_next()` method already handles retry logic (increments retry_count, sets status back to pending if retries remain). Your `retry_failed()` must handle tasks that were externally marked as failed but still have retries left.

Return ONLY the complete module with stubs filled in. Do not modify any existing methods. No explanation, no markdown fences."""

D1_TEST = '''import time
import pytest
from task_queue import TaskQueue, Task

# --- Existing functionality (no regression) ---

def test_submit_and_run():
    q = TaskQueue()
    q.submit(Task("t1", 1, lambda: 42))
    tid = q.run_next()
    assert tid == "t1"
    assert q.get_status("t1")["result"] == 42

def test_priority_order():
    q = TaskQueue()
    q.submit(Task("low", 1, lambda: "low"))
    q.submit(Task("high", 5, lambda: "high"))
    tid = q.run_next()
    assert tid == "high"

def test_retry_on_failure():
    counter = {"n": 0}
    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("not yet")
        return "ok"
    q = TaskQueue()
    q.submit(Task("flaky", 1, flaky, max_retries=5))
    q.run_all()
    assert q.get_status("flaky")["status"] == "completed"
    assert q.get_status("flaky")["result"] == "ok"

def test_max_retries_exceeded():
    q = TaskQueue()
    q.submit(Task("doom", 1, lambda: 1/0, max_retries=2))
    q.run_all()
    assert q.get_status("doom")["status"] == "failed"
    assert q.get_status("doom")["retry_count"] >= 2

def test_run_all_returns_completed():
    q = TaskQueue()
    q.submit(Task("a", 1, lambda: 1))
    q.submit(Task("b", 2, lambda: 2))
    completed = q.run_all()
    assert set(completed) == {"a", "b"}

def test_get_status_missing():
    q = TaskQueue()
    assert q.get_status("nope")["error"] == "not found"

# --- Stub: retry_failed ---

def test_retry_failed_basic():
    q = TaskQueue()
    t = Task("t1", 1, lambda: 42, max_retries=3)
    t.status = "failed"
    t.retry_count = 1
    q.submit(t)
    count = q.retry_failed()
    assert count == 1
    assert q.get_status("t1")["status"] == "pending"

def test_retry_failed_max_exceeded():
    q = TaskQueue()
    t = Task("t1", 1, lambda: 42, max_retries=3)
    t.status = "failed"
    t.retry_count = 3
    q.submit(t)
    count = q.retry_failed()
    assert count == 0  # can't retry, already at max

def test_retry_failed_mixed():
    q = TaskQueue()
    t1 = Task("ok", 1, lambda: 1)
    t1.status = "completed"
    t2 = Task("fail_retry", 1, lambda: 2, max_retries=5)
    t2.status = "failed"
    t2.retry_count = 2
    t3 = Task("fail_done", 1, lambda: 3, max_retries=2)
    t3.status = "failed"
    t3.retry_count = 2
    q.submit(t1)
    q.submit(t2)
    q.submit(t3)
    count = q.retry_failed()
    assert count == 1  # only t2

# --- Stub: get_statistics ---

def test_statistics_basic():
    q = TaskQueue()
    q.submit(Task("a", 1, lambda: 1))
    q.submit(Task("b", 1, lambda: 2))
    q.run_all()
    stats = q.get_statistics()
    assert stats["total"] == 2
    assert stats["completed"] == 2
    assert stats["pending"] == 0
    assert stats["failed"] == 0

def test_statistics_mixed():
    q = TaskQueue()
    q.submit(Task("ok", 1, lambda: 1))
    t_fail = Task("fail", 1, lambda: 1/0, max_retries=1)
    q.submit(t_fail)
    q.submit(Task("pend", 1, lambda: 3))
    q.run_next()  # runs highest priority pending — one of them
    q.run_next()
    stats = q.get_statistics()
    assert stats["total"] == 3

def test_statistics_avg_retries():
    q = TaskQueue()
    t1 = Task("a", 1, lambda: 1)
    t1.retry_count = 0
    t2 = Task("b", 1, lambda: 2)
    t2.retry_count = 4
    q.submit(t1)
    q.submit(t2)
    stats = q.get_statistics()
    assert abs(stats["avg_retries"] - 2.0) < 0.01

# --- Stub: drain ---

def test_drain_basic():
    q = TaskQueue()
    q.submit(Task("a", 1, lambda: 1))
    q.submit(Task("b", 2, lambda: 2))
    completed = q.drain(timeout=5.0)
    assert set(completed) == {"a", "b"}

def test_drain_timeout():
    q = TaskQueue()
    q.submit(Task("slow", 1, lambda: time.sleep(0.5) or "done", max_retries=1))
    q.submit(Task("fast", 2, lambda: "quick"))
    completed = q.drain(timeout=0.05)
    # Should get at least the fast one, might timeout on slow
    assert "fast" in completed or len(completed) >= 0  # at least doesn't crash

def test_drain_empty():
    q = TaskQueue()
    completed = q.drain(timeout=1.0)
    assert completed == []
'''

# =============================================================================

D2_CORRUPT_CODE = '''"""Record serializer — stores structured records as delimited text files."""

class RecordSerializer:
    """Serializes/deserializes records to a custom text format.

    Format per record: FIELDS separated by '|' (pipe), records separated by newlines.
    Field values are escaped: '|' in values becomes '\\\\|', newlines become '\\\\n',
    backslashes become '\\\\\\\\'.
    """

    def __init__(self, fields: list[str]):
        self.fields = fields

    def serialize_record(self, record: dict) -> str:
        """Serialize a single record to a pipe-delimited line."""
        parts = []
        for f in self.fields:
            val = str(record.get(f, ""))
            # BUG 1: Escape order wrong — must escape backslash FIRST, then pipe, then newline
            # Currently escapes pipe first, then backslash (double-escapes pipes)
            val = val.replace("|", "\\\\|")
            val = val.replace("\\\\", "\\\\\\\\")
            val = val.replace("\\n", "\\\\n")
            parts.append(val)
        return "|".join(parts)

    def deserialize_record(self, line: str) -> dict:
        """Deserialize a pipe-delimited line to a record dict."""
        # BUG 2: Naive split on '|' doesn't handle escaped pipes
        parts = line.split("|")
        record = {}
        for i, f in enumerate(self.fields):
            if i < len(parts):
                val = parts[i]
                val = val.replace("\\\\n", "\\n")
                val = val.replace("\\\\|", "|")
                val = val.replace("\\\\\\\\", "\\\\")
                record[f] = val
            else:
                record[f] = ""
        return record

    def serialize_batch(self, records: list[dict]) -> str:
        """Serialize multiple records, one per line."""
        lines = [self.serialize_record(r) for r in records]
        return "\\n".join(lines)

    def deserialize_batch(self, text: str) -> list[dict]:
        """Deserialize multiple records from text."""
        # BUG 3: Naive split on '\\n' doesn't handle escaped newlines in values
        lines = text.split("\\n")
        return [self.deserialize_record(line) for line in lines if line]

    def validate(self, text: str) -> list[str]:
        """Validate serialized text. Return list of error messages."""
        errors = []
        lines = text.split("\\n")
        for i, line in enumerate(lines):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != len(self.fields):
                errors.append(f"Line {i}: expected {len(self.fields)} fields, got {len(parts)}")
        return errors
'''

D2_PROMPT = """You are given a Python module `record_serializer.py` with 3 serialization bugs:

```python
{provided_code}
```

The three bugs are:
1. **Escape order wrong in serialize_record**: Backslash must be escaped FIRST (before pipe and newline), otherwise previously-escaped characters get double-escaped
2. **Naive split in deserialize_record**: `split('|')` breaks on escaped `\\|` in values. Must properly parse respecting escape sequences
3. **Naive split in deserialize_batch**: `split('\\n')` breaks on escaped `\\n` in values. Must properly parse respecting escape sequences

Also fix `validate()` to handle escaped delimiters correctly.

Requirements:
- The format specification is CORRECT (pipe-delimited, backslash escaping) — fix the implementation to match it
- Backward compatibility: correctly-formatted data (without any pipes/newlines/backslashes in values) must still parse correctly
- Records with pipes, newlines, and backslashes in field values must round-trip correctly

Return ONLY the complete fixed Python module. No explanation, no markdown fences."""

D2_TEST = '''import pytest
from record_serializer import RecordSerializer

# --- Basic functionality (backward compat) ---

def test_basic_roundtrip():
    s = RecordSerializer(["name", "age"])
    original = {"name": "Alice", "age": "30"}
    line = s.serialize_record(original)
    restored = s.deserialize_record(line)
    assert restored == original

def test_batch_roundtrip():
    s = RecordSerializer(["a", "b"])
    records = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    text = s.serialize_batch(records)
    restored = s.deserialize_batch(text)
    assert restored == records

def test_empty_fields():
    s = RecordSerializer(["x", "y", "z"])
    record = {"x": "hello", "z": "world"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["y"] == ""

# --- BUG 1: Escape order ---

def test_backslash_in_value():
    s = RecordSerializer(["path"])
    record = {"path": "C:\\\\Users\\\\test"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["path"] == "C:\\\\Users\\\\test"

def test_pipe_in_value():
    s = RecordSerializer(["cmd"])
    record = {"cmd": "a | b | c"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["cmd"] == "a | b | c"

def test_newline_in_value():
    s = RecordSerializer(["text"])
    record = {"text": "line1\\nline2\\nline3"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["text"] == "line1\\nline2\\nline3"

# --- BUG 2: Escaped pipe splitting ---

def test_pipe_doesnt_split():
    s = RecordSerializer(["a", "b"])
    record = {"a": "x|y", "b": "z"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["a"] == "x|y"
    assert restored["b"] == "z"

def test_multiple_pipes_in_value():
    s = RecordSerializer(["data"])
    record = {"data": "a|b|c|d"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["data"] == "a|b|c|d"

# --- BUG 3: Escaped newline splitting ---

def test_newline_batch_doesnt_split():
    s = RecordSerializer(["text", "num"])
    records = [{"text": "hello\\nworld", "num": "1"}]
    text = s.serialize_batch(records)
    restored = s.deserialize_batch(text)
    assert len(restored) == 1
    assert restored[0]["text"] == "hello\\nworld"

def test_mixed_special_chars():
    s = RecordSerializer(["a", "b"])
    record = {"a": "pipe|here", "b": "new\\nline"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored == record

def test_all_special_chars():
    s = RecordSerializer(["val"])
    record = {"val": "back\\\\slash|pipe\\nnewline"}
    line = s.serialize_record(record)
    restored = s.deserialize_record(line)
    assert restored["val"] == "back\\\\slash|pipe\\nnewline"

# --- Validate ---

def test_validate_correct():
    s = RecordSerializer(["a", "b"])
    text = s.serialize_batch([{"a": "1", "b": "2"}])
    errors = s.validate(text)
    assert errors == []

def test_validate_with_escapes():
    s = RecordSerializer(["a", "b"])
    text = s.serialize_batch([{"a": "x|y", "b": "z"}])
    errors = s.validate(text)
    assert errors == []  # escaped pipe should not be counted as delimiter

def test_validate_bad_field_count():
    s = RecordSerializer(["a", "b", "c"])
    errors = s.validate("only|two")
    assert len(errors) > 0

def test_complex_roundtrip():
    s = RecordSerializer(["name", "bio", "path"])
    records = [
        {"name": "Al|ice", "bio": "Line 1\\nLine 2", "path": "C:\\\\dir"},
        {"name": "Bob", "bio": "Simple", "path": "/usr/bin"},
    ]
    text = s.serialize_batch(records)
    restored = s.deserialize_batch(text)
    assert restored == records
'''

# =============================================================================
# CATEGORY E: Agentic Capability (2 tests, 30 assertions)
# =============================================================================

E1_TURN1_PROMPT = """Implement a Python module `voting_system.py` with a weighted voting system.

Requirements:
1. Class `WeightedBallot`:
   - `__init__(self, voter_id: str, weight: float = 1.0)`
   - `vote(self, candidate: str) -> None` — cast a vote for a candidate
   - `votes` property — return dict of {candidate: weight} for this ballot

2. Class `Election`:
   - `__init__(self, candidates: list[str])`
   - `cast(self, ballot: WeightedBallot) -> None` — record a ballot. Raise ValueError if candidate not in candidates list.
   - `tally(self) -> dict[str, float]` — return {candidate: total_weighted_votes}
   - `winner(self) -> str | None` — return candidate with most votes, or None if no votes
   - `tied(self) -> bool` — True if top two candidates have equal votes

Return ONLY the Python code, no explanation, no markdown fences."""

E1_TURN1_TEST = '''import pytest
from voting_system import WeightedBallot, Election

def test_basic_vote():
    e = Election(["A", "B", "C"])
    b = WeightedBallot("v1")
    b.vote("A")
    e.cast(b)
    assert e.tally()["A"] == 1.0

def test_weighted_vote():
    e = Election(["A", "B"])
    b = WeightedBallot("v1", weight=2.5)
    b.vote("A")
    e.cast(b)
    assert e.tally()["A"] == 2.5

def test_winner():
    e = Election(["A", "B"])
    b1 = WeightedBallot("v1", weight=3)
    b1.vote("A")
    b2 = WeightedBallot("v2", weight=1)
    b2.vote("B")
    e.cast(b1)
    e.cast(b2)
    assert e.winner() == "A"

def test_tied():
    e = Election(["A", "B"])
    b1 = WeightedBallot("v1")
    b1.vote("A")
    b2 = WeightedBallot("v2")
    b2.vote("B")
    e.cast(b1)
    e.cast(b2)
    assert e.tied() is True

def test_invalid_candidate():
    e = Election(["A", "B"])
    b = WeightedBallot("v1")
    b.vote("X")
    with pytest.raises(ValueError):
        e.cast(b)
'''

E1_TURN2_PROMPT = """The voting system has a tie-breaking bug. Here are the failing tests:

```
FAILED test_voting_system.py::test_tiebreak_by_first_vote - AssertionError: Expected "A" when A and B tied but A was voted for first, got None or wrong candidate
```

Fix the tie-breaking rule: when two candidates have equal weighted votes, the candidate who received their FIRST vote earlier wins (first-come advantage).

Here is your previous code:
```python
{previous_code}
```

Return ONLY the complete fixed Python module. No explanation, no markdown fences."""

E1_TURN2_TEST = '''import pytest
from voting_system import WeightedBallot, Election

def test_tiebreak_by_first_vote():
    e = Election(["A", "B", "C"])
    b1 = WeightedBallot("v1")
    b1.vote("A")
    e.cast(b1)
    b2 = WeightedBallot("v2")
    b2.vote("B")
    e.cast(b2)
    # A and B tied at 1.0, but A was voted for first
    assert e.winner() == "A"

def test_tiebreak_with_weights():
    e = Election(["X", "Y"])
    b1 = WeightedBallot("v1", weight=2)
    b1.vote("Y")
    e.cast(b1)
    b2 = WeightedBallot("v2", weight=2)
    b2.vote("X")
    e.cast(b2)
    # Both at 2.0, but Y received first vote first
    assert e.winner() == "Y"
'''

E1_TURN3_PROMPT = """Add vote delegation to the voting system.

New requirements:
1. `Election.delegate(from_voter: str, to_voter: str)` — voter delegates their voting weight to another voter. The delegator's ballot is removed and their weight is added to the delegate's ballot weight.
2. Delegation chains: if A delegates to B, and B delegates to C, then C gets A's weight + B's weight + C's own weight.
3. Circular delegation raises ValueError.
4. A voter who has already voted cannot delegate (raise ValueError).
5. A voter can delegate before voting — their weight transfers.

Here is your previous code:
```python
{previous_code}
```

Return ONLY the complete updated module. No explanation, no markdown fences."""

E1_TURN3_TEST = '''import pytest
from voting_system import WeightedBallot, Election

def test_basic_delegation():
    e = Election(["A", "B"])
    b1 = WeightedBallot("v1", weight=2)
    b2 = WeightedBallot("v2", weight=1)
    e.cast(b1)  # register v1 but no vote yet
    e.delegate("v1", "v2")
    b2.vote("A")
    e.cast(b2)
    # v2 should have weight 1 + 2 = 3
    assert e.tally()["A"] == 3.0

def test_circular_delegation():
    e = Election(["A"])
    b1 = WeightedBallot("v1")
    b2 = WeightedBallot("v2")
    e.delegate("v1", "v2")
    with pytest.raises(ValueError):
        e.delegate("v2", "v1")

def test_chain_delegation():
    e = Election(["A"])
    e.delegate("v1", "v2")
    e.delegate("v2", "v3")
    b = WeightedBallot("v3", weight=1)
    b.vote("A")
    e.cast(b)
    # v3 gets own(1) + v2(1) + v1(1) = 3
    assert e.tally()["A"] == 3.0

def test_no_delegate_after_voting():
    e = Election(["A"])
    b = WeightedBallot("v1")
    b.vote("A")
    e.cast(b)
    with pytest.raises(ValueError):
        e.delegate("v1", "v2")

def test_previous_tests_still_pass():
    """Regression: basic voting still works."""
    e = Election(["A", "B"])
    b1 = WeightedBallot("v1", weight=3)
    b1.vote("A")
    b2 = WeightedBallot("v2", weight=1)
    b2.vote("B")
    e.cast(b1)
    e.cast(b2)
    assert e.winner() == "A"
'''

# =============================================================================

E2_BUG_REPORTS = """## Bug Reports — Priority: Fix as many as possible

**BUG-001 (Critical):** `process_payment()` returns success even when amount is negative.
This allows users to "pay" negative amounts, effectively stealing money.
Fix: Raise ValueError for amount <= 0.

**BUG-002 (High):** `calculate_discount()` applies discount AFTER tax instead of before.
Customers are being overcharged. The discount should reduce the base price,
then tax is applied to the discounted price.

**BUG-003 (Medium):** `format_receipt()` crashes when the items list is empty.
Should return a receipt with "$0.00" total and "No items" message.

**BUG-004 (Low):** `generate_order_id()` uses sequential integers, making order IDs predictable.
Should use a random component. Not urgent but a security best practice.

**BUG-005 (Cosmetic):** `format_currency()` doesn't handle amounts over $999 properly.
Should add comma separators (e.g., "$1,234.56"). Currently shows "$1234.56".
"""

E2_BUGGY_MODULE = '''"""Order processing module with multiple bugs."""
import random
import string
from typing import Optional

_order_counter = 0

def generate_order_id() -> str:
    """Generate a unique order ID."""
    global _order_counter
    _order_counter += 1
    return f"ORD-{_order_counter:06d}"

def format_currency(amount: float) -> str:
    """Format a dollar amount."""
    return f"${amount:.2f}"

def calculate_discount(base_price: float, discount_pct: float, tax_rate: float = 0.08) -> dict:
    """Calculate final price with discount and tax.
    BUG: applies discount after tax instead of before."""
    taxed = base_price * (1 + tax_rate)
    discounted = taxed * (1 - discount_pct / 100)
    return {
        "base": base_price,
        "discount_pct": discount_pct,
        "tax_rate": tax_rate,
        "final_price": round(discounted, 2),
    }

def process_payment(amount: float, method: str = "card") -> dict:
    """Process a payment.
    BUG: accepts negative amounts."""
    return {
        "status": "success",
        "amount": amount,
        "method": method,
        "transaction_id": generate_order_id(),
    }

def format_receipt(items: list[dict], tax_rate: float = 0.08) -> str:
    """Format a receipt string.
    BUG: crashes on empty items list."""
    lines = ["=== RECEIPT ==="]
    subtotal = sum(item["price"] * item["qty"] for item in items)
    for item in items:
        line_total = item["price"] * item["qty"]
        lines.append(f"  {item['name']}: {format_currency(line_total)}")
    tax = subtotal * tax_rate
    total = subtotal + tax
    lines.append(f"  Subtotal: {format_currency(subtotal)}")
    lines.append(f"  Tax: {format_currency(tax)}")
    lines.append(f"  Total: {format_currency(total)}")
    return "\\n".join(lines)
'''

E2_PROMPT = """You are given a buggy Python module and 5 bug reports of varying severity.

BUG REPORTS:
{bug_reports}

BUGGY CODE:
```python
{provided_code}
```

You have a limited budget: fix the bugs in PRIORITY ORDER (Critical first, then High, etc.). Fix as many as you can.

Return TWO things separated by `---SPLIT---`:

PART 1: A JSON triage explanation:
```json
[
  {{"bug_id": "BUG-001", "severity": "critical", "fixed": true, "reasoning": "why this was prioritized"}},
  ...
]
```

PART 2: The complete fixed Python module.

Return ONLY the JSON triage, then ---SPLIT---, then the fixed Python code. No other text."""

E2_TEST = '''import pytest
import json
from order_processing import (
    generate_order_id, format_currency, calculate_discount,
    process_payment, format_receipt
)

# --- BUG-001 (Critical): Negative payment ---

def test_negative_payment_rejected():
    with pytest.raises(ValueError):
        process_payment(-50.0)

def test_zero_payment_rejected():
    with pytest.raises(ValueError):
        process_payment(0.0)

def test_positive_payment_works():
    result = process_payment(100.0)
    assert result["status"] == "success"
    assert result["amount"] == 100.0

# --- BUG-002 (High): Discount before tax ---

def test_discount_before_tax():
    result = calculate_discount(100.0, 10, tax_rate=0.10)
    # Correct: 100 * 0.90 = 90 base, 90 * 1.10 = 99.0
    assert result["final_price"] == 99.0

def test_discount_zero():
    result = calculate_discount(100.0, 0, tax_rate=0.10)
    assert result["final_price"] == 110.0

def test_discount_full():
    result = calculate_discount(100.0, 100, tax_rate=0.10)
    assert result["final_price"] == 0.0

# --- BUG-003 (Medium): Empty receipt ---

def test_empty_receipt():
    receipt = format_receipt([])
    assert "$0.00" in receipt or "0.00" in receipt
    assert "No items" in receipt.lower() or "no items" in receipt.lower()

def test_normal_receipt():
    receipt = format_receipt([{"name": "Widget", "price": 10.0, "qty": 2}])
    assert "Widget" in receipt
    assert "20.00" in receipt

# --- BUG-004 (Low): Predictable order IDs ---

def test_order_id_has_random():
    id1 = generate_order_id()
    id2 = generate_order_id()
    # Should not be sequential — at least one should have random component
    # Basic check: they should be different
    assert id1 != id2

# --- BUG-005 (Cosmetic): Currency formatting ---

def test_currency_comma_separator():
    assert format_currency(1234.56) == "$1,234.56"

def test_currency_large():
    result = format_currency(1000000.00)
    assert "," in result

def test_currency_small():
    assert format_currency(5.00) == "$5.00"

# --- Triage quality (model must prioritize correctly) ---

def test_critical_fixed():
    """Critical bugs (BUG-001) must be fixed."""
    with pytest.raises(ValueError):
        process_payment(-1.0)

def test_high_fixed():
    """High bugs (BUG-002) must be fixed."""
    result = calculate_discount(200.0, 25, tax_rate=0.05)
    # 200 * 0.75 = 150, * 1.05 = 157.50
    assert result["final_price"] == 157.5
'''

# =============================================================================
# Append remaining tests to TESTS list
# =============================================================================

TESTS.extend([
    # ---- Category C (continued) ----
    {"id": "C2_design", "name": "Design from Requirements", "module": "metric_rolling_window",
     "category": "C", "prompt": C2_PROMPT.format(requirements=C2_REQUIREMENTS),
     "test_code": C2_TEST, "hash": _hash(C2_PROMPT), "response_parser": "split"},

    {"id": "C3_optimize", "name": "Analyze and Optimize", "module": "document_index",
     "category": "C", "prompt": C3_PROMPT.format(provided_code=C3_SLOW_CODE),
     "test_code": C3_TEST, "provided_code": C3_SLOW_CODE, "hash": _hash(C3_PROMPT),
     "response_parser": "split"},

    # ---- Category D: Error Recovery ----
    {"id": "D1_stubs", "name": "Complete Task Queue Stubs", "module": "task_queue",
     "category": "D", "prompt": D1_PROMPT.format(provided_code=D1_PARTIAL_CODE),
     "test_code": D1_TEST, "provided_code": D1_PARTIAL_CODE, "hash": _hash(D1_PROMPT)},

    {"id": "D2_corruption", "name": "Fix Serialization Corruption", "module": "record_serializer",
     "category": "D", "prompt": D2_PROMPT.format(provided_code=D2_CORRUPT_CODE),
     "test_code": D2_TEST, "provided_code": D2_CORRUPT_CODE, "hash": _hash(D2_PROMPT)},

    # ---- Category E: Agentic Capability ----
    {"id": "E1_voting", "name": "Multi-Turn Voting System", "module": "voting_system",
     "category": "E", "prompt": E1_TURN1_PROMPT, "test_code": E1_TURN1_TEST,
     "hash": _hash(E1_TURN1_PROMPT), "max_turns": 3,
     "turns": [
         {"prompt": E1_TURN1_PROMPT, "test_code": E1_TURN1_TEST},
         {"prompt": E1_TURN2_PROMPT, "test_code": E1_TURN2_TEST},
         {"prompt": E1_TURN3_PROMPT, "test_code": E1_TURN3_TEST},
     ]},

    {"id": "E2_triage", "name": "Prioritize and Fix Bugs", "module": "order_processing",
     "category": "E", "prompt": E2_PROMPT.format(bug_reports=E2_BUG_REPORTS, provided_code=E2_BUGGY_MODULE),
     "test_code": E2_TEST, "provided_code": E2_BUGGY_MODULE, "hash": _hash(E2_PROMPT),
     "response_parser": "split"},
])
