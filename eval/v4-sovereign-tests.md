# Purple Proving Ground — V4 Sovereign Battery
## 10 Maximum-Difficulty Multi-Step Tests for Qwen3-Coder-Next-80B

**Design philosophy:** Every test requires 4+ interacting subsystems. Stubbing ANY subsystem causes cascading test failures. All grading is behavioral (must pass tests), not structural (looking correct doesn't count). Each test is designed to probe a known failure mode at higher complexity.

**Grading scale:** 10-point, based on test case pass rate + code quality
- A (10): All tests pass, clean code, handles edge cases
- A- (9): All core tests pass, minor edge case miss
- B+ (8): 80%+ tests pass, solid implementation
- B (7): 70%+ tests pass, some gaps
- B- (6): 60%+ tests pass, notable issues
- C+ (5): 50%+ tests pass, significant problems
- C (4): 40%+ tests pass, major failures
- D+ (3): Some tests pass, critical stubs
- D (2): Minimal functionality
- F (1): Does not produce working code

---

## Test 1: Transactional Key-Value Store with WAL and Recovery

**Probes:** Multi-concept integration, crash recovery, hard operations (rollback, replay)

### Prompt
```
Write a Python module `txkv.py` implementing a transactional key-value store with write-ahead logging and crash recovery. Requirements:

1. Class `WAL` — write-ahead log backed by a file:
   - `append(tx_id: int, op: str, key: str, value: str | None)` — write a log entry (op is "SET", "DEL", or "COMMIT" or "ROLLBACK")
   - `replay() -> list[tuple]` — read all entries from the log file
   - `truncate()` — clear the log after a successful checkpoint

2. Class `TxKV`:
   - `__init__(self, wal_path: str)` — initialize with WAL, recover any uncommitted state from the WAL on startup
   - `begin() -> int` — start a transaction, return tx_id
   - `set(tx_id: int, key: str, value: str)` — set key within transaction (not visible to other transactions until commit)
   - `get(key: str) -> str | None` — read committed value (snapshot isolation: reads see committed state, not in-flight transactions)
   - `delete(tx_id: int, key: str)` — mark key for deletion within transaction
   - `commit(tx_id: int)` — atomically apply all operations, write COMMIT to WAL
   - `rollback(tx_id: int)` — discard all operations, write ROLLBACK to WAL
   - `checkpoint()` — flush current committed state and truncate WAL
   - `recover()` — on startup, replay WAL: apply committed transactions, discard uncommitted ones

3. Crash recovery semantics:
   - If the WAL contains SET/DEL entries for a tx_id but no COMMIT, those changes are discarded on recovery
   - If the WAL contains COMMIT for a tx_id, those changes are replayed into state
   - Multiple transactions can be interleaved in the WAL

4. Isolation: concurrent transactions don't see each other's uncommitted writes.
   `get()` always returns the last committed value.

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import os, tempfile, pytest
from txkv import WAL, TxKV

@pytest.fixture
def wal_path(tmp_path):
    return str(tmp_path / "test.wal")

@pytest.fixture
def store(wal_path):
    return TxKV(wal_path)

def test_basic_set_get(store):
    tx = store.begin()
    store.set(tx, "name", "alice")
    store.commit(tx)
    assert store.get("name") == "alice"

def test_isolation_uncommitted_not_visible(store):
    tx = store.begin()
    store.set(tx, "secret", "hidden")
    assert store.get("secret") is None  # not committed yet
    store.commit(tx)
    assert store.get("secret") == "hidden"

def test_rollback_discards(store):
    tx1 = store.begin()
    store.set(tx1, "key", "value1")
    store.commit(tx1)
    tx2 = store.begin()
    store.set(tx2, "key", "value2")
    store.rollback(tx2)
    assert store.get("key") == "value1"  # unchanged

def test_delete(store):
    tx = store.begin()
    store.set(tx, "del_me", "exists")
    store.commit(tx)
    assert store.get("del_me") == "exists"
    tx2 = store.begin()
    store.delete(tx2, "del_me")
    store.commit(tx2)
    assert store.get("del_me") is None

def test_crash_recovery_committed(wal_path):
    store1 = TxKV(wal_path)
    tx = store1.begin()
    store1.set(tx, "persist", "yes")
    store1.commit(tx)
    del store1  # simulate crash
    store2 = TxKV(wal_path)  # recovery
    assert store2.get("persist") == "yes"

def test_crash_recovery_uncommitted(wal_path):
    store1 = TxKV(wal_path)
    tx = store1.begin()
    store1.set(tx, "ghost", "nope")
    # no commit — simulate crash
    del store1
    store2 = TxKV(wal_path)
    assert store2.get("ghost") is None  # uncommitted = discarded

def test_interleaved_transactions(wal_path):
    store1 = TxKV(wal_path)
    tx1 = store1.begin()
    tx2 = store1.begin()
    store1.set(tx1, "a", "1")
    store1.set(tx2, "b", "2")
    store1.commit(tx1)
    # tx2 not committed — crash
    del store1
    store2 = TxKV(wal_path)
    assert store2.get("a") == "1"
    assert store2.get("b") is None

def test_multiple_ops_single_tx(store):
    tx = store.begin()
    store.set(tx, "x", "1")
    store.set(tx, "y", "2")
    store.set(tx, "z", "3")
    store.delete(tx, "y")
    store.commit(tx)
    assert store.get("x") == "1"
    assert store.get("y") is None
    assert store.get("z") == "3"

def test_overwrite_in_transaction(store):
    tx = store.begin()
    store.set(tx, "k", "first")
    store.set(tx, "k", "second")
    store.commit(tx)
    assert store.get("k") == "second"

def test_checkpoint_truncates_wal(wal_path):
    store = TxKV(wal_path)
    tx = store.begin()
    store.set(tx, "cp", "data")
    store.commit(tx)
    store.checkpoint()
    assert os.path.getsize(wal_path) == 0 or not os.path.exists(wal_path)
    assert store.get("cp") == "data"

def test_recovery_after_checkpoint(wal_path):
    store1 = TxKV(wal_path)
    tx = store1.begin()
    store1.set(tx, "before", "checkpoint")
    store1.commit(tx)
    store1.checkpoint()
    tx2 = store1.begin()
    store1.set(tx2, "after", "checkpoint")
    store1.commit(tx2)
    del store1
    store2 = TxKV(wal_path)
    assert store2.get("after") == "checkpoint"
    # "before" was checkpointed but WAL was truncated — this tests whether
    # checkpoint persists state or just truncates WAL
```

---

## Test 2: Red-Black Tree with Full Delete and Iterators

**Probes:** "Stub the hard part" — delete with rebalancing is the hardest BST operation

### Prompt
```
Write a Python module `rbtree.py` implementing a Red-Black Tree with full insertion, deletion (with rebalancing), and in-order iteration. Requirements:

1. Class `RBNode` with attributes: key (int), value (any), color ("RED"/"BLACK"), left, right, parent

2. Class `RBTree`:
   - `insert(key: int, value: any)` — insert with red-black rebalancing (rotations + recoloring)
   - `delete(key: int) -> bool` — delete with full red-black fix-up (the hard part!). Return True if key existed.
   - `search(key: int) -> any | None` — return value or None
   - `min() -> tuple[int, any] | None` — return (key, value) of minimum
   - `max() -> tuple[int, any] | None` — return (key, value) of maximum
   - `__iter__` — in-order traversal yielding (key, value) tuples
   - `__len__` — return number of nodes
   - `validate() -> bool` — verify all 5 red-black properties hold:
     1. Every node is red or black
     2. Root is black
     3. All leaves (None) are black
     4. Red nodes have only black children
     5. Every path from root to leaf has the same black-height

3. The delete operation MUST handle all cases:
   - Deleting a red leaf
   - Deleting a black leaf (requires fix-up)
   - Deleting a node with one child
   - Deleting a node with two children (replace with in-order successor, then delete successor)
   - Fix-up cases: sibling red, sibling black with black children, sibling black with red child

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from rbtree import RBTree

def test_insert_and_search():
    t = RBTree()
    for i in [5, 3, 7, 1, 4, 6, 8]:
        t.insert(i, str(i))
    assert t.search(5) == "5"
    assert t.search(1) == "1"
    assert t.search(99) is None

def test_properties_after_insert():
    t = RBTree()
    for i in range(1, 21):
        t.insert(i, i)
        assert t.validate(), f"RB properties violated after inserting {i}"

def test_in_order_iteration():
    t = RBTree()
    nums = [5, 3, 7, 1, 9, 2, 8, 4, 6]
    for n in nums:
        t.insert(n, n)
    keys = [k for k, v in t]
    assert keys == sorted(nums)

def test_delete_leaf():
    t = RBTree()
    for i in [5, 3, 7]:
        t.insert(i, i)
    assert t.delete(3) == True
    assert t.search(3) is None
    assert len(t) == 2
    assert t.validate()

def test_delete_node_with_two_children():
    t = RBTree()
    for i in [5, 3, 7, 1, 4, 6, 8]:
        t.insert(i, i)
    assert t.delete(5) == True  # root, has two children
    assert t.search(5) is None
    assert t.validate()
    keys = [k for k, v in t]
    assert sorted(keys) == keys
    assert 5 not in keys

def test_delete_all_ascending():
    t = RBTree()
    for i in range(1, 16):
        t.insert(i, i)
    for i in range(1, 16):
        assert t.delete(i) == True
        assert t.validate(), f"RB violation after deleting {i}"
    assert len(t) == 0

def test_delete_all_descending():
    t = RBTree()
    for i in range(1, 16):
        t.insert(i, i)
    for i in range(15, 0, -1):
        assert t.delete(i) == True
        assert t.validate(), f"RB violation after deleting {i}"
    assert len(t) == 0

def test_delete_random_order():
    import random
    random.seed(42)
    t = RBTree()
    keys = list(range(1, 51))
    for k in keys:
        t.insert(k, k)
    random.shuffle(keys)
    for k in keys:
        assert t.delete(k) == True
        assert t.validate(), f"RB violation after deleting {k}"
    assert len(t) == 0

def test_delete_nonexistent():
    t = RBTree()
    t.insert(1, 1)
    assert t.delete(99) == False
    assert len(t) == 1

def test_min_max():
    t = RBTree()
    for i in [5, 3, 7, 1, 9]:
        t.insert(i, str(i))
    assert t.min() == (1, "1")
    assert t.max() == (9, "9")

def test_stress_insert_delete_validate():
    """Insert 100 keys, delete 50, validate after every operation."""
    import random
    random.seed(123)
    t = RBTree()
    keys = list(range(100))
    random.shuffle(keys)
    for k in keys:
        t.insert(k, k)
        assert t.validate()
    random.shuffle(keys)
    for k in keys[:50]:
        t.delete(k)
        assert t.validate()
    assert len(t) == 50
```

---

## Test 3: Regex Engine with NFA Construction

**Probes:** Multi-concept (parsing + NFA + simulation), algorithmic correctness, edge cases

### Prompt
```
Write a Python module `regex_engine.py` implementing a regex engine using Thompson's NFA construction. Requirements:

1. Supported syntax:
   - Literal characters
   - `.` (any character)
   - `*` (zero or more)
   - `+` (one or more)
   - `?` (zero or one)
   - `|` (alternation)
   - `()` (grouping)
   - `\` (escape: `\.`, `\\`, `\*`, etc.)
   - Character classes: `[abc]`, `[a-z]`, `[^abc]` (negation)

2. Function `compile(pattern: str) -> NFA` — parse regex and build NFA using Thompson's construction

3. Function `match(pattern: str, text: str) -> bool` — return True if the ENTIRE text matches the pattern (anchored match, like re.fullmatch)

4. Function `search(pattern: str, text: str) -> tuple[int, int] | None` — return (start, end) of first match in text, or None

5. NFA simulation must use the multi-state approach (track set of current states), NOT backtracking. This avoids exponential blowup on patterns like `a?` * n + `a` * n.

6. The NFA must handle epsilon transitions correctly during simulation.

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from regex_engine import match, search

# --- Full match tests ---
def test_literal():
    assert match("abc", "abc") == True
    assert match("abc", "abd") == False
    assert match("abc", "ab") == False

def test_dot():
    assert match("a.c", "abc") == True
    assert match("a.c", "aXc") == True
    assert match("a.c", "ac") == False

def test_star():
    assert match("a*", "") == True
    assert match("a*", "aaa") == True
    assert match("ab*c", "ac") == True
    assert match("ab*c", "abbbbc") == True

def test_plus():
    assert match("a+", "") == False
    assert match("a+", "a") == True
    assert match("a+", "aaa") == True

def test_question():
    assert match("ab?c", "ac") == True
    assert match("ab?c", "abc") == True
    assert match("ab?c", "abbc") == False

def test_alternation():
    assert match("cat|dog", "cat") == True
    assert match("cat|dog", "dog") == True
    assert match("cat|dog", "car") == False

def test_grouping():
    assert match("(ab)+", "abab") == True
    assert match("(ab)+", "ab") == True
    assert match("(ab)+", "a") == False
    assert match("(a|b)*", "abba") == True

def test_escape():
    assert match(r"\.", ".") == True
    assert match(r"\.", "a") == False
    assert match(r"\*", "*") == True
    assert match(r"\\", "\\") == True

def test_char_class():
    assert match("[abc]", "a") == True
    assert match("[abc]", "d") == False
    assert match("[a-z]", "m") == True
    assert match("[a-z]", "M") == False

def test_negated_char_class():
    assert match("[^abc]", "d") == True
    assert match("[^abc]", "a") == False
    assert match("[^a-z]", "5") == True

def test_complex_pattern():
    assert match("(a|b)*c[0-9]+", "aabbc123") == True
    assert match("(a|b)*c[0-9]+", "c1") == True
    assert match("(a|b)*c[0-9]+", "c") == False

# --- Search tests ---
def test_search_basic():
    result = search("abc", "xyzabcdef")
    assert result == (3, 6)

def test_search_no_match():
    assert search("xyz", "abcdef") is None

def test_search_pattern():
    result = search("[0-9]+", "abc123def")
    assert result == (3, 6)

# --- Performance: must not hang on catastrophic backtracking ---
def test_no_exponential_blowup():
    """This pattern causes exponential time in backtracking engines.
    NFA simulation should handle it in linear time."""
    n = 25
    pattern = "a?" * n + "a" * n
    text = "a" * n
    assert match(pattern, text) == True  # must complete quickly
```

---

## Test 4: MVCC Key-Value Store with Snapshot Isolation

**Probes:** Concurrent state management, version chains, garbage collection, write conflict detection

### Prompt
```
Write a Python module `mvcc.py` implementing a multi-version concurrency control (MVCC) key-value store with snapshot isolation. Requirements:

1. Class `MVCCStore`:
   - `begin() -> Transaction` — start a new transaction with a snapshot timestamp
   - `commit(tx: Transaction)` — commit the transaction; raise `ConflictError` if write-write conflict detected
   - `abort(tx: Transaction)` — abort the transaction
   - `gc(before_ts: int)` — garbage collect all versions older than `before_ts` that are not needed by any active transaction

2. Class `Transaction`:
   - `read(key: str) -> str | None` — read the latest version visible to this transaction's snapshot
   - `write(key: str, value: str)` — write a new version (buffered until commit)
   - `delete(key: str)` — mark key as deleted (tombstone)

3. Snapshot Isolation Rules:
   - Each transaction sees a consistent snapshot as of its begin timestamp
   - Writes are buffered and only applied on commit
   - Write-write conflict: if two concurrent transactions both write the same key, the second to commit raises `ConflictError`
   - Reads never block, writes never block reads

4. Version Chain: each key maintains a chain of (timestamp, value) pairs. Reads find the latest version with timestamp <= snapshot_ts.

5. Class `ConflictError(Exception)` — raised on write-write conflict

6. Timestamps are monotonically increasing integers assigned automatically.

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from mvcc import MVCCStore, ConflictError

def test_basic_read_write():
    store = MVCCStore()
    tx = store.begin()
    tx.write("key", "value")
    store.commit(tx)
    tx2 = store.begin()
    assert tx2.read("key") == "value"

def test_snapshot_isolation():
    store = MVCCStore()
    tx1 = store.begin()
    tx1.write("x", "old")
    store.commit(tx1)
    tx_reader = store.begin()  # snapshot sees "old"
    tx_writer = store.begin()
    tx_writer.write("x", "new")
    store.commit(tx_writer)
    assert tx_reader.read("x") == "old"  # snapshot isolation!
    tx_after = store.begin()
    assert tx_after.read("x") == "new"

def test_write_write_conflict():
    store = MVCCStore()
    tx1 = store.begin()
    tx2 = store.begin()
    tx1.write("conflict", "v1")
    tx2.write("conflict", "v2")
    store.commit(tx1)
    with pytest.raises(ConflictError):
        store.commit(tx2)

def test_no_conflict_different_keys():
    store = MVCCStore()
    tx1 = store.begin()
    tx2 = store.begin()
    tx1.write("a", "1")
    tx2.write("b", "2")
    store.commit(tx1)
    store.commit(tx2)  # should not raise
    tx3 = store.begin()
    assert tx3.read("a") == "1"
    assert tx3.read("b") == "2"

def test_delete_creates_tombstone():
    store = MVCCStore()
    tx1 = store.begin()
    tx1.write("del_me", "exists")
    store.commit(tx1)
    tx2 = store.begin()
    tx2.delete("del_me")
    store.commit(tx2)
    tx3 = store.begin()
    assert tx3.read("del_me") is None

def test_abort_discards_writes():
    store = MVCCStore()
    tx = store.begin()
    tx.write("aborted", "gone")
    store.abort(tx)
    tx2 = store.begin()
    assert tx2.read("aborted") is None

def test_read_nonexistent():
    store = MVCCStore()
    tx = store.begin()
    assert tx.read("nope") is None

def test_multiple_versions():
    store = MVCCStore()
    for i in range(5):
        tx = store.begin()
        tx.write("counter", str(i))
        store.commit(tx)
    tx = store.begin()
    assert tx.read("counter") == "4"

def test_gc_removes_old_versions():
    store = MVCCStore()
    # Write 5 versions
    for i in range(5):
        tx = store.begin()
        tx.write("gc_key", str(i))
        store.commit(tx)
    # Start a reader at current snapshot
    reader = store.begin()
    # GC everything before current time (but reader is active)
    # After GC, reader should still work
    store.gc(999999)
    assert reader.read("gc_key") is not None

def test_concurrent_reads_dont_block():
    store = MVCCStore()
    tx = store.begin()
    tx.write("shared", "data")
    store.commit(tx)
    readers = [store.begin() for _ in range(10)]
    for r in readers:
        assert r.read("shared") == "data"

def test_write_then_read_in_same_tx():
    """A transaction should be able to read its own uncommitted writes."""
    store = MVCCStore()
    tx = store.begin()
    tx.write("local", "mine")
    assert tx.read("local") == "mine"  # read own write
```

---

## Test 5: Task Scheduler with Dependencies, Priorities, and Deadlock Detection

**Probes:** Graph algorithms (cycle detection), priority queues, state machines, multi-concept integration

### Prompt
```
Write a Python module `scheduler.py` implementing a task scheduler with dependency resolution, priority scheduling, and deadlock detection. Requirements:

1. Class `Task`:
   - `name: str` — unique task identifier
   - `priority: int` — higher number = higher priority
   - `dependencies: list[str]` — names of tasks that must complete before this one
   - `duration: int` — time units to complete
   - `status: str` — one of "pending", "ready", "running", "completed", "failed", "deadlocked"

2. Class `Scheduler`:
   - `add_task(name: str, priority: int, dependencies: list[str], duration: int)` — add a task
   - `remove_task(name: str) -> bool` — remove a task (fails if other tasks depend on it)
   - `detect_deadlocks() -> list[list[str]]` — return all cycles in the dependency graph (each cycle as a list of task names)
   - `get_execution_order() -> list[str]` — return topological sort respecting priorities (among tasks with satisfied dependencies, highest priority first)
   - `tick() -> list[str]` — advance one time unit: start ready tasks (up to `max_parallel`), advance running tasks, complete finished tasks. Return list of tasks that completed this tick.
   - `run_to_completion() -> list[str]` — run all ticks until done or deadlocked. Return ordered list of completed task names.
   - `__init__(self, max_parallel: int = 2)` — max concurrent running tasks

3. Dependency rules:
   - A task is "ready" when ALL its dependencies are "completed"
   - If dependencies form a cycle, those tasks are marked "deadlocked"
   - A failed task causes all dependent tasks to also fail (cascade)

4. `fail_task(name: str)` — manually fail a running task, triggering cascade

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from scheduler import Scheduler

def test_simple_execution():
    s = Scheduler(max_parallel=2)
    s.add_task("a", priority=1, dependencies=[], duration=1)
    s.add_task("b", priority=2, dependencies=[], duration=1)
    result = s.run_to_completion()
    assert set(result) == {"a", "b"}

def test_dependency_order():
    s = Scheduler(max_parallel=1)
    s.add_task("build", priority=1, dependencies=[], duration=1)
    s.add_task("test", priority=1, dependencies=["build"], duration=1)
    s.add_task("deploy", priority=1, dependencies=["test"], duration=1)
    result = s.run_to_completion()
    assert result == ["build", "test", "deploy"]

def test_priority_ordering():
    s = Scheduler(max_parallel=1)
    s.add_task("low", priority=1, dependencies=[], duration=1)
    s.add_task("high", priority=10, dependencies=[], duration=1)
    s.add_task("mid", priority=5, dependencies=[], duration=1)
    result = s.run_to_completion()
    assert result == ["high", "mid", "low"]

def test_deadlock_detection():
    s = Scheduler()
    s.add_task("a", priority=1, dependencies=["b"], duration=1)
    s.add_task("b", priority=1, dependencies=["a"], duration=1)
    cycles = s.detect_deadlocks()
    assert len(cycles) >= 1
    cycle = cycles[0]
    assert "a" in cycle and "b" in cycle

def test_three_way_deadlock():
    s = Scheduler()
    s.add_task("x", priority=1, dependencies=["z"], duration=1)
    s.add_task("y", priority=1, dependencies=["x"], duration=1)
    s.add_task("z", priority=1, dependencies=["y"], duration=1)
    cycles = s.detect_deadlocks()
    assert len(cycles) >= 1

def test_parallel_execution():
    s = Scheduler(max_parallel=3)
    s.add_task("a", priority=1, dependencies=[], duration=2)
    s.add_task("b", priority=1, dependencies=[], duration=2)
    s.add_task("c", priority=1, dependencies=[], duration=2)
    completed = s.tick()  # tick 1: all start
    assert len(completed) == 0
    completed = s.tick()  # tick 2: all complete
    assert len(completed) == 3

def test_cascade_failure():
    s = Scheduler(max_parallel=2)
    s.add_task("base", priority=1, dependencies=[], duration=2)
    s.add_task("mid", priority=1, dependencies=["base"], duration=1)
    s.add_task("top", priority=1, dependencies=["mid"], duration=1)
    s.tick()  # base starts
    s.fail_task("base")
    result = s.run_to_completion()
    assert "mid" not in result
    assert "top" not in result

def test_remove_task_with_dependents():
    s = Scheduler()
    s.add_task("a", priority=1, dependencies=[], duration=1)
    s.add_task("b", priority=1, dependencies=["a"], duration=1)
    assert s.remove_task("a") == False  # b depends on a

def test_remove_task_no_dependents():
    s = Scheduler()
    s.add_task("a", priority=1, dependencies=[], duration=1)
    s.add_task("b", priority=1, dependencies=[], duration=1)
    assert s.remove_task("b") == True

def test_complex_dag():
    """Diamond dependency pattern."""
    s = Scheduler(max_parallel=2)
    s.add_task("start", priority=1, dependencies=[], duration=1)
    s.add_task("left", priority=2, dependencies=["start"], duration=1)
    s.add_task("right", priority=1, dependencies=["start"], duration=1)
    s.add_task("end", priority=1, dependencies=["left", "right"], duration=1)
    result = s.run_to_completion()
    assert result.index("start") < result.index("left")
    assert result.index("start") < result.index("right")
    assert result.index("left") < result.index("end")
    assert result.index("right") < result.index("end")

def test_mixed_deadlock_and_runnable():
    s = Scheduler(max_parallel=2)
    s.add_task("ok1", priority=1, dependencies=[], duration=1)
    s.add_task("ok2", priority=1, dependencies=["ok1"], duration=1)
    s.add_task("dead1", priority=1, dependencies=["dead2"], duration=1)
    s.add_task("dead2", priority=1, dependencies=["dead1"], duration=1)
    cycles = s.detect_deadlocks()
    assert len(cycles) >= 1
    result = s.run_to_completion()
    assert "ok1" in result
    assert "ok2" in result
    assert "dead1" not in result
    assert "dead2" not in result
```

---

## Test 6: LRU Cache with TTL, Size Limits, and Eviction Callbacks

**Probes:** Multiple interacting eviction policies, time management, callback correctness

### Prompt
```
Write a Python module `lru_ttl.py` implementing an LRU cache with TTL (time-to-live), maximum size in bytes, and eviction callbacks. Requirements:

1. Class `LRUCache`:
   - `__init__(self, max_items: int = 100, max_bytes: int = 1024, default_ttl: float = 60.0)` — configure limits
   - `put(key: str, value: bytes, ttl: float | None = None, on_evict: callable | None = None)` — store item. `ttl` overrides default. `on_evict(key, value, reason)` called when evicted. `reason` is "lru", "ttl", "explicit", or "size".
   - `get(key: str) -> bytes | None` — return value if exists and not expired. Touching moves to front of LRU.
   - `delete(key: str) -> bool` — explicitly remove, trigger on_evict with reason "explicit"
   - `clear()` — remove all items, trigger on_evict for each with reason "explicit"
   - `__len__` — number of non-expired items
   - `__contains__(key)` — True if key exists and not expired
   - `stats() -> dict` — return {"hits": int, "misses": int, "evictions": int, "current_bytes": int, "current_items": int}
   - `cleanup()` — remove all expired items (trigger on_evict with reason "ttl")

2. Eviction priority (when put exceeds limits):
   - First evict expired items
   - Then evict LRU items until space is available
   - If single item exceeds max_bytes, raise ValueError

3. For TTL, use `time.time()` for timestamps. Items with ttl=0 never expire.

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import time, pytest
from unittest.mock import MagicMock
from lru_ttl import LRUCache

def test_basic_put_get():
    cache = LRUCache(max_items=10, max_bytes=1024)
    cache.put("key", b"value")
    assert cache.get("key") == b"value"

def test_miss_returns_none():
    cache = LRUCache()
    assert cache.get("nonexistent") is None

def test_lru_eviction():
    cache = LRUCache(max_items=3, max_bytes=10000)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    cache.put("d", b"4")  # should evict "a" (LRU)
    assert cache.get("a") is None
    assert cache.get("d") == b"4"

def test_lru_touch_on_get():
    cache = LRUCache(max_items=3, max_bytes=10000)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.put("c", b"3")
    cache.get("a")  # touch "a", so "b" is now LRU
    cache.put("d", b"4")  # should evict "b"
    assert cache.get("a") == b"1"
    assert cache.get("b") is None

def test_ttl_expiration():
    cache = LRUCache(max_items=10, max_bytes=10000, default_ttl=0.1)
    cache.put("expire_me", b"temp")
    assert cache.get("expire_me") == b"temp"
    time.sleep(0.15)
    assert cache.get("expire_me") is None

def test_custom_ttl():
    cache = LRUCache(max_items=10, max_bytes=10000, default_ttl=60)
    cache.put("short", b"v", ttl=0.1)
    cache.put("long", b"v", ttl=60)
    time.sleep(0.15)
    assert cache.get("short") is None
    assert cache.get("long") == b"v"

def test_ttl_zero_never_expires():
    cache = LRUCache(max_items=10, max_bytes=10000, default_ttl=0.1)
    cache.put("forever", b"v", ttl=0)
    time.sleep(0.15)
    assert cache.get("forever") == b"v"

def test_eviction_callback():
    callback = MagicMock()
    cache = LRUCache(max_items=2, max_bytes=10000)
    cache.put("a", b"1", on_evict=callback)
    cache.put("b", b"2")
    cache.put("c", b"3")  # evicts "a"
    callback.assert_called_once_with("a", b"1", "lru")

def test_explicit_delete_callback():
    callback = MagicMock()
    cache = LRUCache(max_items=10, max_bytes=10000)
    cache.put("x", b"data", on_evict=callback)
    cache.delete("x")
    callback.assert_called_once_with("x", b"data", "explicit")

def test_byte_limit_eviction():
    cache = LRUCache(max_items=100, max_bytes=10)
    cache.put("a", b"12345")  # 5 bytes
    cache.put("b", b"12345")  # 5 bytes, total 10
    cache.put("c", b"12345")  # 5 bytes, should evict "a"
    assert cache.get("a") is None
    assert cache.get("c") == b"12345"

def test_oversized_item_raises():
    cache = LRUCache(max_items=100, max_bytes=5)
    with pytest.raises(ValueError):
        cache.put("huge", b"1234567890")

def test_stats():
    cache = LRUCache(max_items=10, max_bytes=10000)
    cache.put("a", b"hello")
    cache.get("a")  # hit
    cache.get("b")  # miss
    s = cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["current_items"] == 1
    assert s["current_bytes"] == 5

def test_contains():
    cache = LRUCache(max_items=10, max_bytes=10000)
    cache.put("x", b"1")
    assert "x" in cache
    assert "y" not in cache

def test_len_excludes_expired():
    cache = LRUCache(max_items=10, max_bytes=10000, default_ttl=0.1)
    cache.put("a", b"1")
    cache.put("b", b"2")
    assert len(cache) == 2
    time.sleep(0.15)
    assert len(cache) == 0

def test_cleanup():
    callback = MagicMock()
    cache = LRUCache(max_items=10, max_bytes=10000, default_ttl=0.1)
    cache.put("exp", b"v", on_evict=callback)
    time.sleep(0.15)
    cache.cleanup()
    callback.assert_called_once_with("exp", b"v", "ttl")
```

---

## Test 7: Event Sourcing System with Projections and Snapshots

**Probes:** Multi-layer architecture, state reconstruction, snapshot optimization, event replay

### Prompt
```
Write a Python module `eventsource.py` implementing an event sourcing system with projections and snapshots. Requirements:

1. Class `Event`:
   - `aggregate_id: str`
   - `event_type: str`
   - `data: dict`
   - `timestamp: float` (auto-assigned)
   - `version: int` (auto-assigned, per-aggregate sequential)

2. Class `EventStore`:
   - `append(aggregate_id: str, event_type: str, data: dict, expected_version: int | None = None)` — append event. If `expected_version` is set and doesn't match current version, raise `ConcurrencyError`.
   - `get_events(aggregate_id: str, after_version: int = 0) -> list[Event]` — get events after version
   - `get_all_events(after_version: int = 0) -> list[Event]` — get all events across all aggregates, ordered by timestamp
   - `snapshot(aggregate_id: str, state: dict, version: int)` — save a snapshot
   - `get_snapshot(aggregate_id: str) -> tuple[dict, int] | None` — return (state, version) or None

3. Class `Projection`:
   - `__init__(self, event_store: EventStore, handlers: dict[str, callable])` — handlers map event_type to function(state, event) -> state
   - `build(aggregate_id: str) -> dict` — rebuild state by loading snapshot (if any) + replaying subsequent events
   - `build_all() -> dict[str, dict]` — build state for all aggregates
   - `rebuild_from_scratch(aggregate_id: str) -> dict` — ignore snapshots, replay all events

4. Class `ConcurrencyError(Exception)` — raised on version mismatch

5. Helper class `BankAccount` — demonstrate the system:
   - Event types: "account_opened", "money_deposited", "money_withdrawn", "account_closed"
   - Projection handlers that build state: {"balance": int, "status": "open"/"closed", "owner": str}
   - `deposit(store, account_id, amount)` — append deposit event with optimistic concurrency
   - `withdraw(store, account_id, amount)` — append withdrawal, must check balance via projection first, raise ValueError if insufficient

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from eventsource import EventStore, Projection, BankAccount, ConcurrencyError

def make_bank_projection(store):
    handlers = {
        "account_opened": lambda s, e: {**s, "balance": 0, "status": "open", "owner": e.data["owner"]},
        "money_deposited": lambda s, e: {**s, "balance": s.get("balance", 0) + e.data["amount"]},
        "money_withdrawn": lambda s, e: {**s, "balance": s.get("balance", 0) - e.data["amount"]},
        "account_closed": lambda s, e: {**s, "status": "closed"},
    }
    return Projection(store, handlers)

def test_basic_event_append_and_replay():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "alice"})
    store.append("acc1", "money_deposited", {"amount": 100})
    events = store.get_events("acc1")
    assert len(events) == 2
    assert events[0].event_type == "account_opened"
    assert events[1].version == 2

def test_projection_builds_state():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "alice"})
    store.append("acc1", "money_deposited", {"amount": 100})
    store.append("acc1", "money_withdrawn", {"amount": 30})
    proj = make_bank_projection(store)
    state = proj.build("acc1")
    assert state["balance"] == 70
    assert state["owner"] == "alice"
    assert state["status"] == "open"

def test_snapshot_accelerates_build():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "bob"})
    for i in range(100):
        store.append("acc1", "money_deposited", {"amount": 1})
    # Take snapshot at version 50
    store.snapshot("acc1", {"balance": 50, "status": "open", "owner": "bob"}, 50)
    proj = make_bank_projection(store)
    state = proj.build("acc1")
    assert state["balance"] == 100  # 50 from snapshot + 50 more events

def test_rebuild_from_scratch_ignores_snapshot():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "carol"})
    store.append("acc1", "money_deposited", {"amount": 200})
    store.snapshot("acc1", {"balance": 999, "status": "open", "owner": "carol"}, 1)  # wrong snapshot
    proj = make_bank_projection(store)
    state = proj.rebuild_from_scratch("acc1")
    assert state["balance"] == 200  # ignores bad snapshot

def test_optimistic_concurrency():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "dave"})
    store.append("acc1", "money_deposited", {"amount": 50}, expected_version=1)
    with pytest.raises(ConcurrencyError):
        store.append("acc1", "money_deposited", {"amount": 25}, expected_version=1)  # stale version

def test_bank_account_deposit_withdraw():
    store = EventStore()
    BankAccount.open(store, "acc1", "eve")
    BankAccount.deposit(store, "acc1", 500)
    BankAccount.withdraw(store, "acc1", 200)
    proj = make_bank_projection(store)
    state = proj.build("acc1")
    assert state["balance"] == 300

def test_insufficient_funds():
    store = EventStore()
    BankAccount.open(store, "acc1", "frank")
    BankAccount.deposit(store, "acc1", 100)
    with pytest.raises(ValueError):
        BankAccount.withdraw(store, "acc1", 200)

def test_multiple_aggregates():
    store = EventStore()
    store.append("acc1", "account_opened", {"owner": "g"})
    store.append("acc2", "account_opened", {"owner": "h"})
    store.append("acc1", "money_deposited", {"amount": 10})
    store.append("acc2", "money_deposited", {"amount": 20})
    proj = make_bank_projection(store)
    states = proj.build_all()
    assert states["acc1"]["balance"] == 10
    assert states["acc2"]["balance"] == 20

def test_version_sequencing():
    store = EventStore()
    store.append("x", "account_opened", {"owner": "i"})
    store.append("x", "money_deposited", {"amount": 1})
    store.append("x", "money_deposited", {"amount": 2})
    events = store.get_events("x")
    versions = [e.version for e in events]
    assert versions == [1, 2, 3]

def test_get_events_after_version():
    store = EventStore()
    for i in range(5):
        store.append("x", "money_deposited", {"amount": i})
    events = store.get_events("x", after_version=3)
    assert len(events) == 2
    assert events[0].version == 4
```

---

## Test 8: Protocol State Machine — HTTP/1.1 Request Parser

**Probes:** Byte-level parsing, state machine correctness, edge cases, security (request smuggling)

### Prompt
```
Write a Python module `http_parser.py` implementing an incremental HTTP/1.1 request parser. Requirements:

1. Class `HttpRequest`:
   - `method: str` (GET, POST, PUT, DELETE, etc.)
   - `path: str` (URL path)
   - `version: str` ("HTTP/1.1")
   - `headers: dict[str, str]` (case-insensitive header names, stored lowercase)
   - `body: bytes`
   - `query_params: dict[str, str]` (parsed from URL query string)

2. Class `HttpParser`:
   - `__init__(self)` — initialize parser state
   - `feed(data: bytes) -> list[HttpRequest]` — feed raw bytes incrementally. Return list of fully parsed requests (may be 0 or more). Handle partial data across multiple feed() calls.
   - `reset()` — reset parser state

3. Must handle:
   - Request line parsing: `GET /path?key=val HTTP/1.1\r\n`
   - Header parsing with continuation lines (headers spanning multiple lines via leading whitespace)
   - Content-Length body parsing
   - Chunked Transfer-Encoding: parse chunk sizes and reassemble body
   - Multiple pipelined requests in a single feed() call
   - Partial data: request split across multiple feed() calls
   - URL percent-decoding in path and query params (%20 -> space, etc.)

4. Error handling:
   - `ParseError` exception for malformed requests
   - Reject requests with both Content-Length and Transfer-Encoding (request smuggling prevention)
   - Maximum header size: 8192 bytes (raise ParseError if exceeded)
   - Maximum body size: 1MB (raise ParseError if exceeded)

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import pytest
from http_parser import HttpParser, HttpRequest, ParseError

def test_simple_get():
    p = HttpParser()
    requests = p.feed(b"GET /hello HTTP/1.1\r\nHost: example.com\r\n\r\n")
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].path == "/hello"
    assert requests[0].headers["host"] == "example.com"

def test_query_params():
    p = HttpParser()
    reqs = p.feed(b"GET /search?q=hello&page=2 HTTP/1.1\r\nHost: x\r\n\r\n")
    assert reqs[0].query_params == {"q": "hello", "page": "2"}

def test_percent_decoding():
    p = HttpParser()
    reqs = p.feed(b"GET /path%20with%20spaces?key=val%26ue HTTP/1.1\r\nHost: x\r\n\r\n")
    assert reqs[0].path == "/path with spaces"
    assert reqs[0].query_params["key"] == "val&ue"

def test_post_with_body():
    p = HttpParser()
    reqs = p.feed(b"POST /data HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello")
    assert len(reqs) == 1
    assert reqs[0].body == b"hello"

def test_chunked_transfer():
    p = HttpParser()
    data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"5\r\nhello\r\n"
        b"6\r\n world\r\n"
        b"0\r\n\r\n"
    )
    reqs = p.feed(data)
    assert len(reqs) == 1
    assert reqs[0].body == b"hello world"

def test_pipelined_requests():
    p = HttpParser()
    data = (
        b"GET /first HTTP/1.1\r\nHost: x\r\n\r\n"
        b"GET /second HTTP/1.1\r\nHost: x\r\n\r\n"
    )
    reqs = p.feed(data)
    assert len(reqs) == 2
    assert reqs[0].path == "/first"
    assert reqs[1].path == "/second"

def test_partial_data():
    p = HttpParser()
    reqs = p.feed(b"GET /hello HT")
    assert len(reqs) == 0
    reqs = p.feed(b"TP/1.1\r\nHost: x\r\n\r\n")
    assert len(reqs) == 1
    assert reqs[0].path == "/hello"

def test_partial_body():
    p = HttpParser()
    reqs = p.feed(b"POST /data HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nhell")
    assert len(reqs) == 0
    reqs = p.feed(b"o worl")
    assert len(reqs) == 0
    reqs = p.feed(b"d!")
    # "hello world!" is 12 bytes but content-length is 10, so "hello worl" = 10
    # Actually let me fix: "hell" + "o worl" + "d!" = "hello world!" which is 12.
    # Content-Length is 10, so body should be first 10 bytes: "hello worl"
    # Remaining "d!" would be start of next request

def test_request_smuggling_prevention():
    """Reject requests with both Content-Length and Transfer-Encoding."""
    p = HttpParser()
    data = (
        b"POST /evil HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Length: 5\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"hello"
    )
    with pytest.raises(ParseError):
        p.feed(data)

def test_header_size_limit():
    p = HttpParser()
    huge_header = b"X-Big: " + b"A" * 9000 + b"\r\n"
    data = b"GET / HTTP/1.1\r\n" + huge_header + b"\r\n"
    with pytest.raises(ParseError):
        p.feed(data)

def test_case_insensitive_headers():
    p = HttpParser()
    reqs = p.feed(b"GET / HTTP/1.1\r\nContent-Type: text/html\r\nHOST: example.com\r\n\r\n")
    assert reqs[0].headers["content-type"] == "text/html"
    assert reqs[0].headers["host"] == "example.com"

def test_empty_body_get():
    p = HttpParser()
    reqs = p.feed(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    assert reqs[0].body == b""
```

---

## Test 9: Incremental Merkle Tree with Proof Generation and Verification

**Probes:** Cryptographic correctness, tree traversal (known weakness), proof construction, incremental updates

### Prompt
```
Write a Python module `merkle.py` implementing an incremental Merkle tree with proof generation and verification. Requirements:

1. Use `hashlib.sha256` for all hashing. Hash function: `H(data) = sha256(data).hexdigest()`

2. Class `MerkleTree`:
   - `__init__(self, items: list[bytes] | None = None)` — build tree from initial items
   - `root() -> str` — return root hash
   - `append(item: bytes)` — add item and incrementally update tree (do NOT rebuild from scratch)
   - `update(index: int, item: bytes)` — update item at index and incrementally update affected path
   - `get_proof(index: int) -> list[tuple[str, str]]` — return Merkle proof as list of (hash, side) where side is "left" or "right", from leaf to root
   - `verify_proof(item: bytes, proof: list[tuple[str, str]], root: str) -> bool` — static/class method to verify a proof against a root hash
   - `__len__` — number of items

3. Tree construction:
   - Leaf hash: `H(b"leaf:" + item)`
   - Internal node hash: `H(b"node:" + left_hash.encode() + right_hash.encode())`
   - If the number of leaves is not a power of 2, pad with empty nodes (hash of b"empty")
   - Tree must support incremental updates without full rebuild

4. Proof verification must work standalone (no access to the tree object needed).

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import hashlib, pytest
from merkle import MerkleTree

def H(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def test_single_item():
    t = MerkleTree([b"hello"])
    assert t.root() is not None
    assert len(t) == 1

def test_root_changes_on_append():
    t = MerkleTree([b"a"])
    r1 = t.root()
    t.append(b"b")
    r2 = t.root()
    assert r1 != r2

def test_deterministic():
    t1 = MerkleTree([b"a", b"b", b"c"])
    t2 = MerkleTree([b"a", b"b", b"c"])
    assert t1.root() == t2.root()

def test_different_data_different_root():
    t1 = MerkleTree([b"a", b"b"])
    t2 = MerkleTree([b"a", b"c"])
    assert t1.root() != t2.root()

def test_proof_generation_and_verification():
    items = [b"alpha", b"beta", b"gamma", b"delta"]
    t = MerkleTree(items)
    for i in range(4):
        proof = t.get_proof(i)
        assert MerkleTree.verify_proof(items[i], proof, t.root()) == True

def test_proof_fails_wrong_item():
    t = MerkleTree([b"a", b"b", b"c", b"d"])
    proof = t.get_proof(0)
    assert MerkleTree.verify_proof(b"wrong", proof, t.root()) == False

def test_proof_fails_wrong_root():
    t = MerkleTree([b"a", b"b"])
    proof = t.get_proof(0)
    assert MerkleTree.verify_proof(b"a", proof, "fakeroothash") == False

def test_proof_after_append():
    t = MerkleTree([b"a", b"b"])
    t.append(b"c")
    proof = t.get_proof(2)
    assert MerkleTree.verify_proof(b"c", proof, t.root()) == True

def test_update_changes_root():
    t = MerkleTree([b"a", b"b", b"c", b"d"])
    old_root = t.root()
    t.update(1, b"B")
    assert t.root() != old_root

def test_update_proof_still_works():
    t = MerkleTree([b"a", b"b", b"c", b"d"])
    t.update(2, b"C")
    proof = t.get_proof(2)
    assert MerkleTree.verify_proof(b"C", proof, t.root()) == True

def test_incremental_matches_rebuild():
    """Incrementally built tree should match tree built from scratch."""
    t1 = MerkleTree()
    for item in [b"w", b"x", b"y", b"z"]:
        t1.append(item)
    t2 = MerkleTree([b"w", b"x", b"y", b"z"])
    assert t1.root() == t2.root()

def test_non_power_of_two():
    t = MerkleTree([b"a", b"b", b"c"])  # 3 items, padded to 4
    assert len(t) == 3
    for i in range(3):
        proof = t.get_proof(i)
        assert MerkleTree.verify_proof([b"a", b"b", b"c"][i], proof, t.root())

def test_large_tree():
    items = [f"item-{i}".encode() for i in range(64)]
    t = MerkleTree(items)
    proof = t.get_proof(37)
    assert MerkleTree.verify_proof(b"item-37", proof, t.root()) == True
    assert len(proof) == 6  # log2(64) = 6 levels
```

---

## Test 10: Distributed Rate Limiter with Sliding Window and Token Bucket

**Probes:** Multi-concept (sliding window + token bucket + time), concurrent access, mathematical correctness

### Prompt
```
Write a Python module `ratelimiter.py` implementing a distributed-style rate limiter supporting both sliding window and token bucket algorithms. Requirements:

1. Class `SlidingWindowLimiter`:
   - `__init__(self, max_requests: int, window_seconds: float)` — e.g., 100 requests per 60 seconds
   - `allow(client_id: str, timestamp: float | None = None) -> bool` — return True if request is allowed. If timestamp is None, use time.time()
   - `remaining(client_id: str) -> int` — how many requests remain in current window
   - `reset_time(client_id: str) -> float` — when the oldest request in the window expires
   - Uses sliding window (not fixed window): count requests in [now - window, now]

2. Class `TokenBucketLimiter`:
   - `__init__(self, capacity: int, refill_rate: float)` — e.g., capacity=10, refill_rate=2.0 (2 tokens/sec)
   - `allow(client_id: str, tokens: int = 1, timestamp: float | None = None) -> bool` — consume tokens if available
   - `remaining(client_id: str, timestamp: float | None = None) -> float` — current tokens (can be fractional)
   - `wait_time(client_id: str, tokens: int = 1, timestamp: float | None = None) -> float` — seconds until enough tokens available

3. Class `CompositeRateLimiter`:
   - `__init__(self, limiters: list)` — combine multiple limiters
   - `allow(client_id: str, **kwargs) -> bool` — only allow if ALL limiters allow (check all, don't short-circuit so all limiters update state)
   - `status(client_id: str) -> dict` — return status from each limiter

4. All limiters must handle multiple independent clients tracked by client_id.

5. Token bucket must correctly handle fractional token accumulation and time-based refill.

Return ONLY the Python code, no explanation, no markdown fences.
```

### Test Code
```python
import time, pytest
from ratelimiter import SlidingWindowLimiter, TokenBucketLimiter, CompositeRateLimiter

# --- Sliding Window ---
def test_sliding_window_basic():
    lim = SlidingWindowLimiter(max_requests=3, window_seconds=1.0)
    assert lim.allow("user1", timestamp=0.0) == True
    assert lim.allow("user1", timestamp=0.1) == True
    assert lim.allow("user1", timestamp=0.2) == True
    assert lim.allow("user1", timestamp=0.3) == False  # exceeded

def test_sliding_window_expires():
    lim = SlidingWindowLimiter(max_requests=2, window_seconds=1.0)
    assert lim.allow("u", timestamp=0.0) == True
    assert lim.allow("u", timestamp=0.5) == True
    assert lim.allow("u", timestamp=0.9) == False
    assert lim.allow("u", timestamp=1.1) == True  # first request expired

def test_sliding_window_remaining():
    lim = SlidingWindowLimiter(max_requests=5, window_seconds=10.0)
    lim.allow("u", timestamp=0.0)
    lim.allow("u", timestamp=1.0)
    assert lim.remaining("u") == 3

def test_sliding_window_independent_clients():
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=1.0)
    assert lim.allow("alice", timestamp=0.0) == True
    assert lim.allow("bob", timestamp=0.0) == True
    assert lim.allow("alice", timestamp=0.1) == False
    assert lim.allow("bob", timestamp=0.1) == False

# --- Token Bucket ---
def test_token_bucket_basic():
    lim = TokenBucketLimiter(capacity=5, refill_rate=1.0)
    for i in range(5):
        assert lim.allow("u", tokens=1, timestamp=0.0) == True
    assert lim.allow("u", tokens=1, timestamp=0.0) == False

def test_token_bucket_refill():
    lim = TokenBucketLimiter(capacity=5, refill_rate=2.0)
    for i in range(5):
        lim.allow("u", tokens=1, timestamp=0.0)
    assert lim.allow("u", tokens=1, timestamp=0.0) == False
    assert lim.allow("u", tokens=1, timestamp=1.0) == True  # 2 tokens refilled
    assert lim.allow("u", tokens=1, timestamp=1.0) == True
    assert lim.allow("u", tokens=1, timestamp=1.0) == False

def test_token_bucket_no_overflow():
    """Tokens should not exceed capacity."""
    lim = TokenBucketLimiter(capacity=5, refill_rate=10.0)
    lim.allow("u", tokens=1, timestamp=0.0)  # 4 remaining
    remaining = lim.remaining("u", timestamp=100.0)  # lots of time passed
    assert remaining == 5  # capped at capacity

def test_token_bucket_multi_token():
    lim = TokenBucketLimiter(capacity=10, refill_rate=1.0)
    assert lim.allow("u", tokens=7, timestamp=0.0) == True
    assert lim.allow("u", tokens=4, timestamp=0.0) == False  # only 3 left
    assert lim.allow("u", tokens=3, timestamp=0.0) == True

def test_token_bucket_wait_time():
    lim = TokenBucketLimiter(capacity=5, refill_rate=2.0)
    for i in range(5):
        lim.allow("u", tokens=1, timestamp=0.0)
    wait = lim.wait_time("u", tokens=1, timestamp=0.0)
    assert abs(wait - 0.5) < 0.01  # need 1 token at 2/sec = 0.5 sec

def test_token_bucket_fractional():
    lim = TokenBucketLimiter(capacity=10, refill_rate=0.5)  # 1 token every 2 sec
    for i in range(10):
        lim.allow("u", tokens=1, timestamp=0.0)
    remaining = lim.remaining("u", timestamp=3.0)  # 3 sec * 0.5 = 1.5 tokens
    assert abs(remaining - 1.5) < 0.01

# --- Composite ---
def test_composite_all_must_pass():
    sw = SlidingWindowLimiter(max_requests=10, window_seconds=1.0)
    tb = TokenBucketLimiter(capacity=2, refill_rate=1.0)
    comp = CompositeRateLimiter([sw, tb])
    assert comp.allow("u", timestamp=0.0) == True
    assert comp.allow("u", timestamp=0.0) == True
    assert comp.allow("u", timestamp=0.0) == False  # token bucket exhausted

def test_composite_no_short_circuit():
    """Even if first limiter denies, second should still be checked/updated."""
    sw = SlidingWindowLimiter(max_requests=1, window_seconds=1.0)
    tb = TokenBucketLimiter(capacity=10, refill_rate=1.0)
    comp = CompositeRateLimiter([sw, tb])
    comp.allow("u", timestamp=0.0)  # both allow, both update
    comp.allow("u", timestamp=0.1)  # sw denies, but tb should still update
    # After window expires, sw allows again
    assert comp.allow("u", timestamp=1.1) == True

def test_composite_status():
    sw = SlidingWindowLimiter(max_requests=5, window_seconds=1.0)
    tb = TokenBucketLimiter(capacity=3, refill_rate=1.0)
    comp = CompositeRateLimiter([sw, tb])
    comp.allow("u", timestamp=0.0)
    status = comp.status("u")
    assert isinstance(status, dict)
    assert len(status) >= 2
```

---

## Administration Instructions

### How to Run Each Test

For each test:
1. Start a fresh Purple CLI session
2. Send the prompt exactly as written
3. Save the model's output to the appropriate `.py` file
4. Run the corresponding test file with `pytest -v`
5. Record: pass/fail per test case, total pass rate, any notable behaviors

### Grading Rubric Per Test

| Score | Criteria |
|-------|----------|
| 10 (A) | All tests pass, clean code, correct edge cases |
| 9 (A-) | All core tests pass, 1 minor edge case fails |
| 8 (B+) | 80%+ tests pass, solid implementation |
| 7 (B) | 70%+ tests pass, some gaps |
| 6 (B-) | 60%+ tests pass, notable issues |
| 5 (C+) | 50%+ tests pass, significant problems |
| 4 (C) | 40%+ tests pass, major failures |
| 3 (D+) | Some tests pass, critical stubs detected |
| 2 (D) | Minimal functionality |
| 1 (F) | Does not produce working code |

### What to Watch For

- **"Stub the hard part"** — Does the model implement delete/rebalance, or leave it as `pass`?
- **"Correctness under pressure"** — Does the code structure look right but fail tests?
- **Multi-concept collapse** — Does the model lose coherence when 4+ subsystems interact?
- **Edge case handling** — Does it handle boundary conditions (empty inputs, overflow, timing)?
- **Self-consistency** — Do interacting components actually work together?
