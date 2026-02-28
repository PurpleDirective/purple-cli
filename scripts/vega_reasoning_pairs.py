#!/usr/bin/env python3
"""
Vega Reasoning Training Pairs — 30 hand-authored SFT pairs
Targets the V6 category A failure patterns observed in Qwen3.5-27B:
  - Complex multi-class code gen (state machines, schedulers, routers)
  - Hard data structure operations (delete, rebalance, merge)
  - Algorithm correctness under edge/boundary conditions
  - Error propagation in multi-layer systems

Run: python3 vega_reasoning_pairs.py
     python3 vega_reasoning_pairs.py --dry-run
"""

import json, hashlib, sys
from datetime import date

DRY_RUN = "--dry-run" in sys.argv

SYSTEM = (
    "You are Vega, an advanced AI system and the core operational intelligence of the Purple Organization. "
    "You are a practitioner in your domains: cybersecurity (red team), software development, IT systems, "
    "mathematics, and finance. You are a sophisticated learner — deep knowledge, but always aware there is more "
    "to learn. You are detail-obsessed, intellectually curious, and warm but unmistakably artificial."
)

def pair(user, assistant, category, note=""):
    h = hashlib.sha256((user + assistant).encode()).hexdigest()[:16]
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ], "metadata": {"source": "vega_reasoning_pairs_v1", "category": category, "note": note,
                    "hash": h, "generated": str(date.today())}}

pairs = []

# ─────────────────────────────────────────────
# SECTION 1: HARD DATA STRUCTURE OPERATIONS (8)
# The failure mode: implement easy operations correctly, stub or break hard ones
# ─────────────────────────────────────────────

pairs.append(pair(
    "Implement a complete AVL tree with insert and delete. Delete is the hard part — don't stub it.",
    '''Complete AVL tree. I'm writing delete properly because that's where implementations fall apart.

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = None
        self.height = 1

class AVLTree:
    def _height(self, n):
        return n.height if n else 0

    def _balance(self, n):
        return self._height(n.left) - self._height(n.right) if n else 0

    def _update_height(self, n):
        n.height = 1 + max(self._height(n.left), self._height(n.right))

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, n):
        self._update_height(n)
        bf = self._balance(n)
        # Left heavy
        if bf > 1:
            if self._balance(n.left) < 0:          # Left-Right case
                n.left = self._rotate_left(n.left)
            return self._rotate_right(n)
        # Right heavy
        if bf < -1:
            if self._balance(n.right) > 0:          # Right-Left case
                n.right = self._rotate_right(n.right)
            return self._rotate_left(n)
        return n

    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root  # duplicates ignored
        return self._rebalance(root)

    def _min_node(self, n):
        while n.left:
            n = n.left
        return n

    def delete(self, root, key):
        if not root:
            return root
        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            # Node found — three cases
            if not root.left:
                return root.right          # Case 1: no left child
            if not root.right:
                return root.left           # Case 2: no right child
            # Case 3: two children — replace with in-order successor
            successor = self._min_node(root.right)
            root.key = successor.key
            root.right = self.delete(root.right, successor.key)
        return self._rebalance(root)

    def inorder(self, root, result=None):
        if result is None:
            result = []
        if root:
            self.inorder(root.left, result)
            result.append(root.key)
            self.inorder(root.right, result)
        return result
```

The three delete cases matter:
1. No left child → return right subtree directly
2. No right child → return left subtree directly
3. Two children → find in-order successor (minimum of right subtree), copy its key up, delete the successor from right subtree

The rebalance after delete handles up to two rotations per level on the way back up the recursion. The height update must happen before the balance factor check — that ordering is load-bearing.''',
    "data-structures", "AVL delete — the stubbed operation"
))

pairs.append(pair(
    "Implement a B-tree of order 3 with insert and delete including the case where deletion causes underflow.",
    '''B-tree order 3 (each node holds 1-2 keys, 2-3 children). The hard parts are split on insert and borrow/merge on delete.

```python
class BTreeNode:
    def __init__(self, leaf=False):
        self.keys = []
        self.children = []
        self.leaf = leaf

class BTree:
    def __init__(self, t=2):          # t=2 → order 3, min degree 2
        self.t = t
        self.root = BTreeNode(leaf=True)

    # ── INSERT ──────────────────────────────────────────────────────
    def insert(self, k):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:   # root is full → split
            new_root = BTreeNode()
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, k)

    def _insert_non_full(self, node, k):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            while i >= 0 and k < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = k
        else:
            while i >= 0 and k < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if k > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], k)

    def _split_child(self, parent, i):
        t = self.t
        child = parent.children[i]
        new_node = BTreeNode(leaf=child.leaf)
        mid = t - 1
        parent.keys.insert(i, child.keys[mid])
        parent.children.insert(i + 1, new_node)
        new_node.keys = child.keys[mid + 1:]
        child.keys = child.keys[:mid]
        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    # ── DELETE ──────────────────────────────────────────────────────
    def delete(self, k):
        self._delete(self.root, k)
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]   # shrink tree height

    def _delete(self, node, k):
        t = self.t
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1

        if i < len(node.keys) and node.keys[i] == k:
            if node.leaf:
                node.keys.pop(i)          # simple case: leaf
            else:
                # Internal node — replace with predecessor or successor
                if len(node.children[i].keys) >= t:
                    pred = self._get_predecessor(node.children[i])
                    node.keys[i] = pred
                    self._delete(node.children[i], pred)
                elif len(node.children[i + 1].keys) >= t:
                    succ = self._get_successor(node.children[i + 1])
                    node.keys[i] = succ
                    self._delete(node.children[i + 1], succ)
                else:
                    # Both children at minimum — merge
                    self._merge(node, i)
                    self._delete(node.children[i], k)
        else:
            if node.leaf:
                return  # key not found
            # Ensure child[i] has at least t keys before descending
            if len(node.children[i].keys) < t:
                self._fill(node, i)
                # After fill, index may shift
                if i > len(node.keys):
                    i -= 1
            self._delete(node.children[i], k)

    def _fill(self, parent, i):
        """Ensure parent.children[i] has >= t keys by borrowing or merging."""
        t = self.t
        if i > 0 and len(parent.children[i - 1].keys) >= t:
            self._borrow_from_prev(parent, i)
        elif i < len(parent.children) - 1 and len(parent.children[i + 1].keys) >= t:
            self._borrow_from_next(parent, i)
        else:
            if i < len(parent.children) - 1:
                self._merge(parent, i)
            else:
                self._merge(parent, i - 1)

    def _borrow_from_prev(self, parent, i):
        child, sibling = parent.children[i], parent.children[i - 1]
        child.keys.insert(0, parent.keys[i - 1])
        parent.keys[i - 1] = sibling.keys.pop()
        if not sibling.leaf:
            child.children.insert(0, sibling.children.pop())

    def _borrow_from_next(self, parent, i):
        child, sibling = parent.children[i], parent.children[i + 1]
        child.keys.append(parent.keys[i])
        parent.keys[i] = sibling.keys.pop(0)
        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge(self, parent, i):
        child, sibling = parent.children[i], parent.children[i + 1]
        child.keys.append(parent.keys.pop(i))
        child.keys.extend(sibling.keys)
        child.children.extend(sibling.children)
        parent.children.pop(i + 1)

    def _get_predecessor(self, node):
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]

    def _get_successor(self, node):
        while not node.leaf:
            node = node.children[0]
        return node.keys[0]
```

The delete underflow cases in order of complexity:
1. Key in leaf → just remove
2. Key in internal node with rich left child → replace with predecessor, recurse
3. Key in internal node with rich right child → replace with successor, recurse
4. Both children at minimum → merge children around the key, recurse into merged node
5. Descending into a minimum-key child → fill it first (borrow left, borrow right, or merge)

The root shrink at the end (when root has no keys after delete) is easy to forget and silently leaves a dead root node.''',
    "data-structures", "B-tree delete with borrow/merge"
))

pairs.append(pair(
    "Write a correct LRU cache with O(1) get and put using a doubly linked list and hash map.",
    '''LRU cache — the hash map gives O(1) lookup, the doubly linked list gives O(1) move-to-front. Common mistake: forgetting to remove old tail when capacity is hit.

```python
class Node:
    __slots__ = ("key", "val", "prev", "next")
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}                    # key → Node
        # Sentinel head/tail — never removed, simplifies edge cases
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_front(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert_front(node)
        return node.val

    def put(self, key: int, value: int):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self._insert_front(node)
        if len(self.cache) > self.cap:
            lru = self.tail.prev          # least recently used is just before tail
            self._remove(lru)
            del self.cache[lru.key]       # must delete from map too — easy to forget
```

The sentinel pattern eliminates four null checks — every insert and remove touches real nodes, never None. The eviction path must remove from both the list AND the dict. Half implementations do the list removal and leave the dict entry, leaking memory and giving stale hits.''',
    "data-structures", "LRU — dual data structure consistency"
))

pairs.append(pair(
    "Implement merge sort on a linked list. Don't convert to array — do it in-place.",
    '''Linked list merge sort — the trick is splitting by slow/fast pointer, not by index.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sort_list(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head

    # Split: find midpoint with slow/fast pointers
    slow, fast = head, head.next       # fast starts one ahead so split favors left
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None                   # cut the list — must set to None

    left = sort_list(head)
    right = sort_list(mid)
    return merge(left, right)

def merge(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode()
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2               # attach remaining tail
    return dummy.next
```

Why `fast = head.next` (not `head`): if fast starts at head, a two-element list splits 1+1 with slow stopping at the first node. If fast starts one ahead, the split still works correctly — just makes the left half one longer than right when odd. Both work; the one-ahead version is more common.

The `slow.next = None` cut is the line everyone forgets on first pass. Without it, the recursive call on `head` sees the full list again and recurses forever.''',
    "data-structures", "linked list split without array conversion"
))

pairs.append(pair(
    "Implement a min-heap with push, pop, and heapify. Show the sift-down correctly.",
    '''Min-heap from scratch. Sift-down is the hard direction — used in both pop and heapify.

```python
class MinHeap:
    def __init__(self):
        self.data = []

    def push(self, val):
        self.data.append(val)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty heap")
        self._swap(0, len(self.data) - 1)
        val = self.data.pop()
        if self.data:
            self._sift_down(0)
        return val

    def peek(self):
        return self.data[0] if self.data else None

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i):   return 2 * i + 1
    def _right(self, i):  return 2 * i + 2
    def _swap(self, i, j): self.data[i], self.data[j] = self.data[j], self.data[i]

    def _sift_up(self, i):
        while i > 0:
            p = self._parent(i)
            if self.data[i] < self.data[p]:
                self._swap(i, p)
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            smallest = i
            l, r = self._left(i), self._right(i)
            if l < n and self.data[l] < self.data[smallest]:
                smallest = l
            if r < n and self.data[r] < self.data[smallest]:
                smallest = r
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    @classmethod
    def heapify(cls, arr):
        """Build heap in O(n) by sifting down from last non-leaf."""
        h = cls()
        h.data = arr[:]
        # Last non-leaf is at index (n//2 - 1)
        # Sift down from there up to root — O(n) total, not O(n log n)
        for i in range(len(h.data) // 2 - 1, -1, -1):
            h._sift_down(i)
        return h
```

The heapify direction is important: sifting up from every element is O(n log n). Sifting down from the last non-leaf backward to the root is O(n) because lower levels do less work — most nodes are near the bottom. Starting from `n//2 - 1` is correct because `n//2` onward are all leaves.

Sift-down stops when `smallest == i` (both children are larger or don't exist). A common bug is checking children against each other first before comparing to the parent — that misses the case where both children are larger than the parent and no swap is needed.''',
    "data-structures", "heap sift-down, heapify direction"
))

# ─────────────────────────────────────────────
# SECTION 2: MULTI-CLASS SYSTEMS (8)
# The failure mode: classes work in isolation, break on interaction
# ─────────────────────────────────────────────

pairs.append(pair(
    "Build a thread-safe bounded queue with blocking put and get in Python. No busy-waiting.",
    '''Thread-safe bounded queue using condition variables — no polling, proper blocking.

```python
import threading

class BoundedQueue:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.queue = []
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, item, timeout=None):
        """Block until space is available, then add item."""
        with self._not_full:
            if not self._not_full.wait_for(
                lambda: len(self.queue) < self.maxsize,
                timeout=timeout
            ):
                raise TimeoutError("put timed out — queue full")
            self.queue.append(item)
            self._not_empty.notify()      # wake one waiting get()

    def get(self, timeout=None):
        """Block until item is available, then return it."""
        with self._not_empty:
            if not self._not_empty.wait_for(
                lambda: len(self.queue) > 0,
                timeout=timeout
            ):
                raise TimeoutError("get timed out — queue empty")
            item = self.queue.pop(0)
            self._not_full.notify()       # wake one waiting put()
            return item

    def qsize(self):
        with self._lock:
            return len(self.queue)
```

Two condition variables sharing one lock — the standard pattern. `_not_full` blocks producers; `_not_empty` blocks consumers. `wait_for(predicate)` handles spurious wakeups automatically (it loops internally until the predicate is true). The notify after put/get must happen while still holding the lock, which `with self._not_empty` guarantees.

Common mistake: using a single condition variable for both states. That works but requires `notify_all()` instead of `notify()`, which wakes every waiter to re-check the predicate — more contention than necessary.''',
    "multi-class", "thread-safe blocking — condition variables"
))

pairs.append(pair(
    "Implement a rate limiter class using the token bucket algorithm. It should be usable as a decorator.",
    '''Token bucket rate limiter — tokens replenish at a steady rate, consumed on each request.

```python
import time
import threading
import functools

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        """
        rate: tokens added per second
        capacity: max tokens (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity          # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Return True if request is allowed, False if rate limited."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait(self, tokens: float = 1.0):
        """Block until tokens are available."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_time = (tokens - self._tokens) / self.rate
            time.sleep(wait_time)

    def __call__(self, func):
        """Use as decorator: @bucket"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


# Usage
bucket = TokenBucket(rate=10, capacity=20)   # 10 req/s, burst up to 20

@bucket
def api_call():
    pass

# Or inline
if bucket.consume():
    do_thing()
else:
    return 429
```

The refill happens lazily on consume/wait — no background thread needed. `time.monotonic()` not `time.time()` because monotonic can't go backward (clock adjustments don't affect it). The lock scope covers both refill and consume atomically — if you refill outside the lock you get a race where two threads both see enough tokens and both consume.''',
    "multi-class", "token bucket — lazy refill, lock scope correctness"
))

pairs.append(pair(
    "Write a simple event bus (pub/sub) in Python that supports async handlers.",
    '''Event bus with sync and async handler support, weak references to avoid memory leaks.

```python
import asyncio
import weakref
import inspect
from collections import defaultdict
from typing import Callable, Any

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable):
        """Register a handler. Stores weak ref for bound methods to avoid leaks."""
        if inspect.ismethod(handler):
            ref = weakref.WeakMethod(handler)
        else:
            ref = weakref.ref(handler)
        self._handlers[event].append(ref)

    def unsubscribe(self, event: str, handler: Callable):
        self._handlers[event] = [
            r for r in self._handlers[event]
            if r() is not None and r() != handler
        ]

    def _live_handlers(self, event: str):
        """Return live handlers, pruning dead weak refs."""
        live = []
        dead = []
        for ref in self._handlers[event]:
            h = ref()
            if h is None:
                dead.append(ref)
            else:
                live.append(h)
        for d in dead:
            self._handlers[event].remove(d)
        return live

    def publish(self, event: str, *args, **kwargs):
        """Call all handlers. Runs async handlers in current event loop if available."""
        for handler in self._live_handlers(event):
            if asyncio.iscoroutinefunction(handler):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(handler(*args, **kwargs))
                except RuntimeError:
                    asyncio.run(handler(*args, **kwargs))
            else:
                handler(*args, **kwargs)

    async def publish_async(self, event: str, *args, **kwargs):
        """Await all async handlers concurrently, call sync handlers inline."""
        coros = []
        for handler in self._live_handlers(event):
            if asyncio.iscoroutinefunction(handler):
                coros.append(handler(*args, **kwargs))
            else:
                handler(*args, **kwargs)
        if coros:
            await asyncio.gather(*coros)
```

The weak reference pattern matters for bound methods — `weakref.ref(obj.method)` doesn't work because the method object is created on the fly and immediately garbage collected. `WeakMethod` holds a weak ref to the object and reconstructs the bound method on access. Without this, any object that subscribes to an event will never be garbage collected as long as the bus lives.

Dead ref pruning happens lazily in `_live_handlers` — no need for a GC callback.''',
    "multi-class", "event bus — weak refs, sync+async dispatch"
))

pairs.append(pair(
    "Implement a circuit breaker pattern for wrapping unreliable function calls.",
    '''Circuit breaker — three states: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery).

```python
import time
import threading
from enum import Enum
from typing import Callable, Any, Type

class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerOpen(Exception):
    pass

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            if (self._state == State.OPEN and
                    time.monotonic() - self._last_failure_time >= self.recovery_timeout):
                self._state = State.HALF_OPEN
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        state = self.state
        if state == State.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit open — last failure {time.monotonic() - self._last_failure_time:.1f}s ago"
            )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = State.CLOSED

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = State.OPEN

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
```

The HALF_OPEN state transition is lazy — it happens in the `state` property when checked, not on a timer. This avoids a background thread. HALF_OPEN allows one call through; if it succeeds, back to CLOSED; if it fails, back to OPEN with reset timer. A common incomplete implementation omits HALF_OPEN entirely and the breaker never recovers automatically.''',
    "multi-class", "circuit breaker — HALF_OPEN state, lazy transition"
))

pairs.append(pair(
    "Build a connection pool that manages a fixed set of reusable database connections.",
    '''Connection pool — fixed size, blocking acquire with timeout, automatic return on context manager exit.

```python
import threading
import time
from contextlib import contextmanager
from typing import Callable, Any

class PoolExhausted(Exception):
    pass

class ConnectionPool:
    def __init__(self, factory: Callable, size: int = 10, timeout: float = 5.0):
        """
        factory: callable that creates a new connection
        size:    fixed pool size
        timeout: seconds to wait for an available connection
        """
        self._factory = factory
        self._size = size
        self._timeout = timeout
        self._pool: list = []
        self._lock = threading.Lock()
        self._available = threading.Semaphore(size)
        # Pre-create all connections
        for _ in range(size):
            self._pool.append(factory())

    def acquire(self, timeout: float = None) -> Any:
        t = timeout if timeout is not None else self._timeout
        if not self._available.acquire(timeout=t):
            raise PoolExhausted(f"No connection available after {t}s")
        with self._lock:
            return self._pool.pop()

    def release(self, conn: Any):
        with self._lock:
            self._pool.append(conn)
        self._available.release()

    @contextmanager
    def connection(self, timeout: float = None):
        conn = self.acquire(timeout)
        try:
            yield conn
        except Exception:
            # Option: validate connection before returning, recreate if broken
            try:
                conn.rollback()   # best-effort cleanup for DB connections
            except Exception:
                conn = self._factory()   # replace broken connection
            raise
        finally:
            self.release(conn)

    def close_all(self):
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()

# Usage
pool = ConnectionPool(factory=lambda: create_db_connection(), size=10)

with pool.connection() as conn:
    conn.execute("SELECT 1")
```

The semaphore tracks available slots — acquire blocks up to timeout without polling. The lock protects the list itself since pop/append aren't atomic under all conditions. The rollback+recreate on exception is the part most pool implementations stub out — a broken connection returned to the pool will fail the next caller. Detecting "broken" is connection-type specific (DB: catch OperationalError, HTTP: check if socket is closed).''',
    "multi-class", "connection pool — semaphore blocking, broken conn recovery"
))

pairs.append(pair(
    "Write a state machine class that enforces valid transitions and calls entry/exit hooks.",
    '''State machine with enforced transitions and hooks — the hook ordering matters.

```python
from typing import Callable, Optional

class InvalidTransition(Exception):
    pass

class StateMachine:
    def __init__(self, initial: str):
        self.state = initial
        self._transitions: dict[tuple[str, str], str] = {}   # (from, event) → to
        self._on_enter: dict[str, list[Callable]] = {}
        self._on_exit: dict[str, list[Callable]] = {}

    def add_transition(self, from_state: str, event: str, to_state: str):
        self._transitions[(from_state, event)] = to_state

    def on_enter(self, state: str, callback: Callable):
        self._on_enter.setdefault(state, []).append(callback)

    def on_exit(self, state: str, callback: Callable):
        self._on_exit.setdefault(state, []).append(callback)

    def trigger(self, event: str, **context) -> str:
        key = (self.state, event)
        if key not in self._transitions:
            raise InvalidTransition(
                f"No transition from '{self.state}' on event '{event}'"
            )
        from_state = self.state
        to_state = self._transitions[key]

        # Exit hooks fire before state changes
        for cb in self._on_exit.get(from_state, []):
            cb(from_state=from_state, to_state=to_state, event=event, **context)

        self.state = to_state   # state change is atomic here

        # Entry hooks fire after state changes
        for cb in self._on_enter.get(to_state, []):
            cb(from_state=from_state, to_state=to_state, event=event, **context)

        return to_state

    def can_trigger(self, event: str) -> bool:
        return (self.state, event) in self._transitions


# Example: order lifecycle
sm = StateMachine(initial="pending")
sm.add_transition("pending",    "submit",  "submitted")
sm.add_transition("submitted",  "approve", "approved")
sm.add_transition("submitted",  "reject",  "rejected")
sm.add_transition("approved",   "ship",    "shipped")
sm.add_transition("shipped",    "deliver", "delivered")

sm.on_enter("approved", lambda **kw: print("Order approved — notify warehouse"))
sm.on_exit("submitted", lambda **kw: print(f"Leaving submitted → {kw['to_state']}"))

sm.trigger("submit")
sm.trigger("approve")
```

Hook ordering is the detail: exit hooks fire while still in the old state (the callback knows where it's leaving), entry hooks fire after the state variable is updated (the callback sees the new current state). Reversing this causes hooks to see inconsistent state. The `**context` passthrough lets callers attach payloads without changing the interface.''',
    "multi-class", "state machine — hook ordering, transition enforcement"
))

pairs.append(pair(
    "Implement a dependency injection container in Python that resolves constructor dependencies automatically.",
    '''DI container with auto-wiring from type hints. Detects circular dependencies.

```python
import inspect
from typing import Type, TypeVar, get_type_hints

T = TypeVar("T")

class CircularDependency(Exception):
    pass

class Container:
    def __init__(self):
        self._registry: dict[type, type | callable] = {}
        self._singletons: dict[type, object] = {}
        self._resolving: set[type] = set()   # cycle detection

    def register(self, interface: type, implementation=None, singleton: bool = False):
        """Register a type. If implementation is None, register type as its own implementation."""
        impl = implementation or interface
        self._registry[interface] = (impl, singleton)

    def resolve(self, t: type) -> object:
        # Check singleton cache first
        if t in self._singletons:
            return self._singletons[t]

        # Cycle detection
        if t in self._resolving:
            raise CircularDependency(f"Circular dependency detected for {t}")

        self._resolving.add(t)
        try:
            impl, singleton = self._registry.get(t, (t, False))

            if callable(impl) and not isinstance(impl, type):
                # Factory function
                obj = impl()
            else:
                # Resolve constructor dependencies via type hints
                hints = {}
                try:
                    hints = get_type_hints(impl.__init__)
                except Exception:
                    pass
                hints.pop("return", None)

                sig = inspect.signature(impl.__init__)
                kwargs = {}
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    annotation = hints.get(name)
                    if annotation and annotation in self._registry:
                        kwargs[name] = self.resolve(annotation)
                    elif param.default is not inspect.Parameter.empty:
                        pass   # use default
                    else:
                        raise TypeError(f"Cannot resolve parameter '{name}' of {impl}")
                obj = impl(**kwargs)

            if singleton:
                self._singletons[t] = obj
            return obj
        finally:
            self._resolving.discard(t)

    def __getitem__(self, t: Type[T]) -> T:
        return self.resolve(t)


# Example
class Database:
    def __init__(self, url: str = "sqlite:///:memory:"):
        self.url = url

class UserRepository:
    def __init__(self, db: Database):
        self.db = db

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

c = Container()
c.register(Database, singleton=True)
c.register(UserRepository)
c.register(UserService)

svc = c[UserService]
assert svc.repo.db is c[Database]   # singleton — same instance
```

The cycle detection set must be cleaned up in a `finally` block — if resolution raises, the type must be removed from `_resolving` or all future resolutions of that type will falsely report a cycle. `get_type_hints` is used over `__annotations__` directly because it resolves forward references (string annotations).''',
    "multi-class", "DI container — cycle detection, type hint resolution"
))

pairs.append(pair(
    "Build a simple task scheduler that runs jobs at fixed intervals in background threads. Handle exceptions so one bad job doesn't kill others.",
    '''Task scheduler — each job runs in its own thread, exceptions are caught and logged per job, not propagated.

```python
import threading
import time
import logging
from typing import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class Job:
    name: str
    func: Callable
    interval: float           # seconds between runs
    run_immediately: bool = True
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread = field(default=None, repr=False)
    last_error: Exception = field(default=None, repr=False)
    run_count: int = 0

class Scheduler:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def add_job(self, name: str, func: Callable, interval: float,
                run_immediately: bool = True) -> Job:
        if name in self._jobs:
            raise ValueError(f"Job '{name}' already registered")
        job = Job(name=name, func=func, interval=interval,
                  run_immediately=run_immediately)
        self._jobs[name] = job
        return job

    def start(self):
        for job in self._jobs.values():
            job._stop.clear()
            job._thread = threading.Thread(
                target=self._run_job,
                args=(job,),
                daemon=True,
                name=f"scheduler-{job.name}"
            )
            job._thread.start()

    def stop(self, timeout: float = 5.0):
        for job in self._jobs.values():
            job._stop.set()
        for job in self._jobs.values():
            if job._thread and job._thread.is_alive():
                job._thread.join(timeout=timeout)

    def remove_job(self, name: str):
        job = self._jobs.pop(name, None)
        if job:
            job._stop.set()

    def _run_job(self, job: Job):
        if not job.run_immediately:
            job._stop.wait(timeout=job.interval)
        while not job._stop.is_set():
            try:
                job.func()
                job.run_count += 1
                job.last_error = None
            except Exception as e:
                job.last_error = e
                log.error(f"Job '{job.name}' raised {type(e).__name__}: {e}",
                          exc_info=True)
            job._stop.wait(timeout=job.interval)   # interruptible sleep


# Usage
sched = Scheduler()
sched.add_job("health_check", check_services, interval=30)
sched.add_job("cleanup", purge_old_files, interval=3600, run_immediately=False)
sched.start()
```

`stop.wait(timeout=interval)` instead of `time.sleep(interval)` is the key detail — it makes the sleep interruptible. A job sleeping in `time.sleep(3600)` won't stop for an hour when you call `stop()`. The daemon=True flag means jobs don't prevent process exit if you forget to call stop(). Each job is isolated: an exception in one loop iteration gets logged and the job continues on the next interval.''',
    "multi-class", "scheduler — interruptible sleep, per-job exception isolation"
))

# ─────────────────────────────────────────────
# SECTION 3: ALGORITHM CORRECTNESS UNDER EDGE CONDITIONS (8)
# The failure mode: correct on happy path, wrong at boundaries
# ─────────────────────────────────────────────

pairs.append(pair(
    "Implement binary search. What are the off-by-one errors and how do you avoid them?",
    '''Binary search — three variants, each with different invariants. Know which you're implementing.

```python
def binary_search_exact(arr, target):
    """Returns index of target, or -1 if not found."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:             # <= because single element [lo==hi] must be checked
        mid = lo + (hi - lo) // 2   # avoid overflow (matters in C; Python ints are unbounded)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def lower_bound(arr, target):
    """First index where arr[i] >= target. Returns len(arr) if all < target."""
    lo, hi = 0, len(arr)         # hi = len, not len-1 — allows returning len(arr)
    while lo < hi:               # < not <= — loop ends when lo == hi
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid             # not mid-1 — hi is exclusive bound
    return lo

def upper_bound(arr, target):
    """First index where arr[i] > target. Returns len(arr) if all <= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:   # <= instead of < — the only diff from lower_bound
            lo = mid + 1
        else:
            hi = mid
    return lo
```

The invariants that eliminate off-by-one errors:

**Exact search:** `[lo, hi]` is closed on both ends. Loop condition `lo <= hi` because a single-element range `lo==hi` must be checked. Move `lo = mid+1` / `hi = mid-1` because `mid` has already been checked.

**Lower/upper bound:** `[lo, hi)` is half-open — `hi` is the exclusive right edge. Loop condition `lo < hi` because when they're equal the interval is empty (done). `hi = mid` (not `mid-1`) because `mid` hasn't been ruled out as the answer — it satisfies the condition but might not be the leftmost.

The difference between lower and upper bound is a single `<` vs `<=` in the comparison. That's the entire off-by-one between "first occurrence" and "one past last occurrence".''',
    "algorithms", "binary search invariants — the three variants"
))

pairs.append(pair(
    "Implement Dijkstra's algorithm correctly. What breaks if you use a visited set wrong?",
    '''Dijkstra with a min-heap. The visited set mistake is common — here is the correct pattern.

```python
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], source: int) -> Dict[int, float]:
    """
    graph: {node: [(neighbor, weight), ...]}
    Returns: {node: shortest_distance_from_source}
    """
    dist = defaultdict(lambda: float("inf"))
    dist[source] = 0
    heap = [(0, source)]   # (distance, node)
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue          # stale entry — a shorter path was already found
        visited.add(u)

        for v, weight in graph.get(u, []):
            if v in visited:
                continue
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))   # may push duplicate; handled by stale check

    return dict(dist)
```

The visited check must happen AFTER popping, not before pushing. Here is the wrong version and why it fails:

```python
# WRONG — checking before push
for v, weight in graph[u]:
    if v not in visited:      # checks visited at push-time
        if new_dist < dist[v]:
            dist[v] = new_dist
            visited.add(v)    # marking visited here is the bug
            heapq.heappush(heap, (new_dist, v))
```

The bug: you mark v as visited when you push it, not when you process it. But a second, shorter path to v might be found before v is popped. That shorter path can never be processed — v is already in visited. The heap may contain stale entries at any time; that is fine. The invariant is: when a node is popped and not yet visited, its current heap entry IS its shortest distance.

This is correct for non-negative weights only. Negative weights require Bellman-Ford.''',
    "algorithms", "Dijkstra — visited set timing, stale entry handling"
))

pairs.append(pair(
    "Implement topological sort and cycle detection on a directed graph.",
    '''Two approaches: DFS-based (detects cycles) and Kahn's algorithm (BFS-based). Show both.

```python
from collections import defaultdict, deque
from typing import List, Optional

# ── DFS topological sort with cycle detection ─────────────────────
def topo_dfs(n: int, edges: List[tuple]) -> Optional[List[int]]:
    """Returns topological order or None if cycle detected."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2   # unvisited, in-progress, done
    color = [WHITE] * n
    order = []

    def dfs(u) -> bool:
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                return False        # back edge → cycle
            if color[v] == WHITE:
                if not dfs(v):
                    return False
        color[u] = BLACK
        order.append(u)             # append AFTER all descendants
        return True

    for i in range(n):
        if color[i] == WHITE:
            if not dfs(i):
                return None         # cycle found

    return order[::-1]              # reverse because we append in finish order


# ── Kahn's algorithm (BFS) ────────────────────────────────────────
def topo_kahn(n: int, edges: List[tuple]) -> Optional[List[int]]:
    """Returns topological order or None if cycle (not all nodes processed)."""
    in_degree = [0] * n
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order if len(order) == n else None   # if len < n, cycle exists
```

The DFS cycle detection uses three colors, not two. Two colors (visited/unvisited) can't distinguish a node being currently processed (on the DFS stack) from one already completed. GRAY means "currently in the recursion stack" — finding a GRAY node is a back edge, which is a cycle. GRAY nodes that finish become BLACK.

Kahn's cycle detection is simpler: if the output order doesn't include all n nodes, the remaining nodes have in-degree > 0 in a cycle and were never added to the queue. No explicit cycle tracking needed.''',
    "algorithms", "topological sort — 3-color DFS, Kahn's cycle detection"
))

pairs.append(pair(
    "Implement a sliding window maximum — given array and window size k, return max in each window.",
    '''Sliding window maximum in O(n) using a monotonic deque. Brute force is O(nk) — the deque eliminates redundant comparisons.

```python
from collections import deque
from typing import List

def sliding_window_max(nums: List[int], k: int) -> List[int]:
    if not nums or k == 0:
        return []

    result = []
    dq = deque()   # stores indices, not values; front = index of current window max

    for i, n in enumerate(nums):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove indices of smaller values from the back
        # They can never be the maximum for any future window (n is larger and newer)
        while dq and nums[dq[-1]] < n:
            dq.pop()

        dq.append(i)

        # Window is fully formed starting at index k-1
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

The deque stores indices, not values, for two reasons: the window expiry check needs positions (`dq[0] < i - k + 1`), and we need `nums[dq[0]]` to get the actual max.

The invariant: the deque contains indices of a decreasing subsequence of values within the current window. Front is always the index of the maximum. When a new element arrives that is larger than the back of the deque, those smaller elements are removed — they can never be the max for any remaining window because the new element is both larger and will stay in the window longer.

Edge cases:
- `k > len(nums)`: only one window, result has one element
- All equal elements: deque has one entry per window slot (smaller-or-equal elements stay in)
- k=1: trivially returns nums itself''',
    "algorithms", "sliding window max — monotonic deque invariant"
))

pairs.append(pair(
    "Implement quickselect to find the kth smallest element. What's the worst case and how do you mitigate it?",
    '''Quickselect — O(n) average, O(n²) worst case. Median-of-three pivot mitigates the worst case.

```python
import random
from typing import List

def quickselect(arr: List[int], k: int) -> int:
    """Find kth smallest element (1-indexed). Modifies arr in place."""
    if not 1 <= k <= len(arr):
        raise ValueError(f"k={k} out of range for array of length {len(arr)}")
    return _quickselect(arr, 0, len(arr) - 1, k - 1)   # convert to 0-indexed

def _quickselect(arr, lo, hi, k):
    if lo == hi:
        return arr[lo]

    pivot_idx = _partition(arr, lo, hi)

    if k == pivot_idx:
        return arr[pivot_idx]
    elif k < pivot_idx:
        return _quickselect(arr, lo, pivot_idx - 1, k)
    else:
        return _quickselect(arr, pivot_idx + 1, hi, k)

def _partition(arr, lo, hi):
    # Median-of-three pivot selection to avoid O(n²) on sorted input
    mid = (lo + hi) // 2
    if arr[lo] > arr[mid]:
        arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]:
        arr[mid], arr[hi] = arr[hi], arr[mid]
    # arr[lo] <= arr[mid] <= arr[hi]; use mid as pivot, place at hi-1
    pivot = arr[mid]
    arr[mid], arr[hi] = arr[hi], arr[mid]

    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1

# Alternatively: randomized pivot — simpler, same expected O(n)
def quickselect_random(arr: List[int], k: int) -> int:
    arr = arr[:]   # don't mutate input
    lo, hi = 0, len(arr) - 1
    k -= 1
    while lo < hi:
        pivot_idx = random.randint(lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        if i == k:
            return arr[i]
        elif i < k:
            lo = i + 1
        else:
            hi = i - 1
    return arr[lo]
```

Worst case O(n²) happens when the pivot is always the minimum or maximum — every partition reduces the problem by 1 instead of half. This happens on already-sorted input with a naive "pick last element as pivot" strategy.

Median-of-three: compare arr[lo], arr[mid], arr[hi], sort them in place, use the median as pivot. This prevents the sorted-array pathology and is cache-friendly. Random pivot achieves the same expected complexity (O(n)) without determinism — harder to construct adversarial input against.

Introselect (used in numpy) switches to median-of-medians if recursion depth suggests O(n²) is happening, giving guaranteed O(n) worst case at higher constant cost.''',
    "algorithms", "quickselect — partition correctness, worst case mitigation"
))

pairs.append(pair(
    "Implement consistent hashing for a distributed cache. Explain why naive modulo hashing breaks on node addition/removal.",
    '''Consistent hashing with virtual nodes. The modulo problem first, then the solution.

```python
import hashlib
from bisect import bisect_left, insort
from typing import Optional

class ConsistentHashRing:
    def __init__(self, virtual_nodes: int = 150):
        """
        virtual_nodes: number of points per physical node on the ring
        More virtual nodes = more even distribution, more memory
        """
        self.virtual_nodes = virtual_nodes
        self._ring: list[int] = []           # sorted list of hash positions
        self._node_map: dict[int, str] = {}  # position → physical node

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node}:vn{i}")
            self._node_map[pos] = node
            insort(self._ring, pos)

    def remove_node(self, node: str):
        for i in range(self.virtual_nodes):
            pos = self._hash(f"{node}:vn{i}")
            if pos in self._node_map:
                del self._node_map[pos]
                idx = bisect_left(self._ring, pos)
                if idx < len(self._ring) and self._ring[idx] == pos:
                    self._ring.pop(idx)

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        pos = self._hash(key)
        idx = bisect_left(self._ring, pos)
        if idx == len(self._ring):
            idx = 0              # wrap around — ring is circular
        return self._node_map[self._ring[idx]]
```

**Why modulo hashing breaks:**

`node = hash(key) % n_nodes`

Add a node (n=3 → n=4): almost every key's modulo changes. You need to remap ~75% of all keys. In a cache, this means cache miss rate spikes to near 100% until the new node warms up.

**Consistent hashing:**

Keys and nodes both map to positions on a ring (0 to 2^128 for MD5). Each key is assigned to the first node clockwise from its position. When a node is added, it only captures keys from its immediate clockwise neighbor. When removed, those keys go back to the neighbor. Average key movement: `K/n` (where K = total keys, n = nodes) — the theoretical minimum.

Virtual nodes solve uneven distribution: without them, a node might cover 50% of the ring by luck and another cover 2%. With 150 virtual nodes per physical node, the distribution approaches even.''',
    "algorithms", "consistent hashing — why modulo fails, ring wrap-around"
))

pairs.append(pair(
    "Implement a trie with insert, search, and starts_with. Then add delete without corrupting shared prefixes.",
    '''Trie with correct delete — the hard part is not removing nodes shared by other words.

```python
class TrieNode:
    __slots__ = ("children", "is_end")
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._traverse(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._traverse(prefix) is not None

    def _traverse(self, s: str) -> TrieNode | None:
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def delete(self, word: str) -> bool:
        """Delete word. Returns False if not found. Does not remove nodes shared by other words."""
        return self._delete(self.root, word, 0)

    def _delete(self, node: TrieNode, word: str, depth: int) -> bool:
        """Returns True if the current node can be deleted (no other words depend on it)."""
        if depth == len(word):
            if not node.is_end:
                return False       # word not in trie
            node.is_end = False
            # Safe to delete this node only if it has no children
            return len(node.children) == 0

        ch = word[depth]
        child = node.children.get(ch)
        if child is None:
            return False           # word not in trie

        should_delete_child = self._delete(child, word, depth + 1)

        if should_delete_child:
            del node.children[ch]
            # Can delete current node only if it's not an endpoint of another word
            # and has no remaining children
            return not node.is_end and len(node.children) == 0

        return False
```

The delete logic bottom-up: after clearing `is_end` on the leaf, the recursion unwinds checking whether each node can be removed. A node is removable only if it has no children AND is not the end of another word. Forgetting the second condition deletes shared prefixes:

```
Insert "app", "apple"
Delete "apple" — if we delete all nodes back to root, we destroy "app" too
```

The recursive return value (bool = "can caller delete me") propagates this check upward without needing a separate pass.''',
    "algorithms", "trie delete — shared prefix protection"
))

pairs.append(pair(
    "Implement the knapsack problem (0/1). What are the common implementation mistakes?",
    '''0/1 knapsack with DP — common mistakes are the iteration order and using the wrong previous row.

```python
from typing import List, Tuple

def knapsack_01(capacity: int, items: List[Tuple[int, int]]) -> int:
    """
    items: list of (weight, value)
    Returns: maximum value achievable within capacity
    """
    n = len(items)
    # 2D DP: dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight, value = items[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]               # don't take item i
            if weight <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weight] + value)   # take item i

    return dp[n][capacity]


def knapsack_01_optimized(capacity: int, items: List[Tuple[int, int]]) -> int:
    """Space-optimized: O(capacity) instead of O(n * capacity)."""
    dp = [0] * (capacity + 1)

    for weight, value in items:
        # CRITICAL: iterate RIGHT TO LEFT when optimizing to 1D
        # Left-to-right would reuse updated values from the same row (= unbounded knapsack)
        for w in range(capacity, weight - 1, -1):
            dp[w] = max(dp[w], dp[w - weight] + value)

    return dp[capacity]
```

The iteration direction is the single most common mistake in the optimized version:

- **Right to left (correct for 0/1):** `dp[w - weight]` still refers to the value BEFORE current item was considered, because we haven't touched smaller indices yet.
- **Left to right (wrong for 0/1, correct for unbounded knapsack):** `dp[w - weight]` may already be updated with the current item — meaning the item can be taken multiple times.

To reconstruct which items were chosen (not just the max value), use the 2D version and trace back: if `dp[i][w] != dp[i-1][w]`, item i was taken; move to `dp[i-1][w - weight_i]`. The 1D version loses this information.''',
    "algorithms", "0/1 knapsack — 1D optimization direction, why it matters"
))

# ─────────────────────────────────────────────
# SECTION 4: ERROR PROPAGATION (7)
# The failure mode: errors are swallowed, wrong layer handles them, or cascades occur
# ─────────────────────────────────────────────

pairs.append(pair(
    "What's wrong with catching all exceptions at every layer? Where should exceptions be handled?",
    '''Exception handling architecture — most applications get this wrong in the same ways.

**The common mistake:**
```python
# Wrong — catch-all at every layer
def get_user(user_id):
    try:
        return db.query("SELECT * FROM users WHERE id = ?", user_id)
    except Exception:
        return None   # Swallowed: DB is down? User doesn\'t exist? Can\'t tell.

def get_user_email(user_id):
    try:
        user = get_user(user_id)
        return user["email"]
    except Exception:
        return ""     # Swallowed: user was None from above, KeyError — silently wrong.
```

The result: the caller thinks they got an empty email address. The DB has been down for hours. No one knows.

**The principle: handle exceptions at the layer that can do something useful about them.**

```python
# Correct — specific exceptions, appropriate layer
class UserNotFound(Exception):
    pass

class DatabaseError(Exception):
    pass

def get_user(user_id):
    try:
        result = db.query("SELECT * FROM users WHERE id = ?", user_id)
    except ConnectionError as e:
        raise DatabaseError(f"DB unreachable: {e}") from e   # translate, preserve cause
    if not result:
        raise UserNotFound(f"No user with id={user_id}")
    return result[0]

def get_user_email(user_id):
    # Don\'t catch here — UserNotFound and DatabaseError should propagate
    user = get_user(user_id)
    return user["email"]

# API layer — this IS the right place to catch and convert to HTTP responses
def handle_get_user_request(user_id):
    try:
        email = get_user_email(user_id)
        return {"email": email}, 200
    except UserNotFound:
        return {"error": "user not found"}, 404
    except DatabaseError:
        log.error("Database unavailable", exc_info=True)
        return {"error": "service unavailable"}, 503
```

Rules:
1. Catch specific exceptions, not `Exception`
2. If you catch and re-raise, preserve the cause with `from e` (keeps the original traceback)
3. Handle at the boundary between layers — not inside domain logic
4. Never return None to signal an error — raise an exception with a meaningful type
5. Log at the layer that handles, not at the layer that raises''',
    "error-handling", "exception architecture — where to handle, not swallow"
))

pairs.append(pair(
    "How do you write retry logic that doesn't make thundering herd problems worse?",
    '''Retry with exponential backoff and jitter — the jitter part is what most implementations skip.

```python
import time
import random
import logging
from functools import wraps
from typing import Type, Tuple

log = logging.getLogger(__name__)

def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """Decorator: retry with exponential backoff + jitter."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    sleep_time = min(delay, max_delay)
                    if jitter:
                        # Full jitter: uniform between 0 and calculated delay
                        # Prevents synchronized retries across multiple clients
                        sleep_time = random.uniform(0, sleep_time)
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor
        return wrapper
    return decorator


@retry(exceptions=(ConnectionError, TimeoutError), max_attempts=5, base_delay=0.5)
def call_external_api():
    ...
```

**Why jitter matters:**

Without jitter: 100 clients all fail at t=0, all retry at t=1s, all fail again, all retry at t=2s. The server is hit with synchronized bursts that prevent it from recovering — the thundering herd.

With full jitter (uniform 0 to calculated delay): retries spread out. The server sees a trickle instead of synchronized hammering. The expected recovery time is similar; the success probability per attempt is much higher.

**Exponential backoff without jitter + many clients = thundering herd.** This is why AWS's SDKs all use jitter, and why most home-grown retry implementations make outage recovery worse.

Also: retry only on transient errors (connection timeouts, 503s). Never retry on 400 Bad Request — you'll retry a guaranteed failure. `exceptions` parameter controls exactly this.''',
    "error-handling", "retry backoff — jitter is not optional"
))

pairs.append(pair(
    "What causes a deadlock in threading and how do you prevent it systematically?",
    '''Deadlock — causes, detection, and systematic prevention.

**Cause: circular resource acquisition**

```python
# Thread 1            Thread 2
lock_a.acquire()      lock_b.acquire()
# ... time passes     # ... time passes
lock_b.acquire()      lock_a.acquire()   # DEADLOCK — each holds what the other needs
```

Four conditions must ALL hold (Coffman conditions):
1. Mutual exclusion — resources can\'t be shared
2. Hold and wait — threads hold resources while waiting for more
3. No preemption — resources can\'t be taken away
4. Circular wait — T1 waits for T2 which waits for T1

Eliminating ANY one condition prevents deadlock.

**Prevention 1: Lock ordering (eliminates circular wait)**

```python
# Always acquire locks in the same global order
locks = [lock_a, lock_b, lock_c]
locks_sorted = sorted(locks, key=id)   # consistent order by memory address

for lock in locks_sorted:
    lock.acquire()
try:
    # critical section
finally:
    for lock in reversed(locks_sorted):
        lock.release()
```

If every thread that needs both lock_a and lock_b always acquires lock_a first, circular wait is impossible.

**Prevention 2: Try-acquire with timeout (eliminates indefinite hold)**

```python
import threading

def acquire_or_abort(locks, timeout=1.0):
    acquired = []
    for lock in locks:
        if lock.acquire(timeout=timeout):
            acquired.append(lock)
        else:
            # Failed — release what we have and retry
            for l in acquired:
                l.release()
            return False
    return True
```

**Prevention 3: Reduce lock scope (eliminate hold-and-wait)**

```python
# Wrong — holds both locks for the full duration
with lock_a:
    with lock_b:
        do_work_a()
        do_work_b()

# Better — separate acquisitions if operations are independent
with lock_a:
    data_a = read_a()
with lock_b:
    data_b = read_b()
combine(data_a, data_b)
```

Python's `threading.RLock` (reentrant lock) prevents self-deadlock but not cross-thread deadlock. `concurrent.futures` and `asyncio` avoid threads entirely for many use cases — if you can use async instead of threads, deadlock is eliminated by design.''',
    "error-handling", "deadlock — Coffman conditions, lock ordering prevention"
))

pairs.append(pair(
    "How do you debug a memory leak in a Python long-running service?",
    '''Memory leak debugging — systematic approach using tracemalloc and objgraph.

**Step 1: Confirm it\'s actually a leak**
```python
import os, time

# Check RSS (resident set size) over time
def log_memory():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if "VmRSS" in line:
                print(line.strip())

# Run every 60s — if RSS grows unboundedly, it\'s a leak
```

**Step 2: Find what\'s growing with tracemalloc**
```python
import tracemalloc

tracemalloc.start(10)   # keep 10 frames in traceback

# ... let the service run for a while ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)
```

**Step 3: Compare snapshots to find what\'s accumulating**
```python
tracemalloc.start()
snap1 = tracemalloc.take_snapshot()

# ... process some requests ...

snap2 = tracemalloc.take_snapshot()
top_stats = snap2.compare_to(snap1, "lineno")

for stat in top_stats[:5]:
    print(stat)   # shows diff: +N KiB from file:line
```

**Step 4: Identify leaked object types with objgraph**
```python
import objgraph

# Show what types are growing
objgraph.show_growth(limit=10)

# Find all instances of a specific type
objgraph.show_backrefs(objgraph.by_type("MyClass")[0], max_depth=3)
```

**Common Python leak patterns:**

1. **Global dict/list growing**: `cache[key] = value` with no eviction — use `functools.lru_cache` or `cachetools.TTLCache`
2. **Unclosed resources**: file handles, DB connections, sockets — use context managers
3. **Event handler accumulation**: registering listeners in a loop without deregistering
4. **Circular references with `__del__`**: Python\'s GC handles simple cycles but not cycles involving objects with `__del__`
5. **Thread-local storage**: data stored in `threading.local()` that outlives the thread\'s logical task
6. **C extension leaks**: some libraries leak in their C layer; tracemalloc won\'t see these — use valgrind or heaptrack on the process

`gc.collect()` then `gc.get_objects()` with `len()` before and after processing a request tells you if Python objects are accumulating.''',
    "error-handling", "memory leak debugging — tracemalloc, objgraph, common patterns"
))

pairs.append(pair(
    "A service intermittently returns wrong results under load. Walk me through how you debug it.",
    '''Intermittent correctness failures under load — systematic investigation approach.

**Triage: what kind of wrong?**
- Wrong value (returns data that belongs to a different request) → shared mutable state, race condition
- Stale value (returns old data) → caching bug, connection pool returning dirty connections
- Missing/partial result (returns some of what was requested) → timeout mid-operation, partial write
- Wrong result reproducibly → logic bug (simpler — not a concurrency issue)

**Step 1: Reproduce at lower scale**
```bash
# Load test with wrk or locust to reproduce
wrk -t4 -c100 -d30s http://localhost:8080/api/endpoint
# Check: does the failure rate correlate with concurrency level?
```

**Step 2: Add request tracing**
```python
import uuid, threading

_request_id = threading.local()

def middleware(request, next_handler):
    _request_id.value = str(uuid.uuid4())[:8]
    log.info(f"[{_request_id.value}] START {request.path}")
    response = next_handler(request)
    log.info(f"[{_request_id.value}] END {response.status}")
    return response
```

Now cross-reference log lines by request ID. If you see request A's ID appearing in request B's log lines, you have state leak between requests.

**Step 3: Hunt shared mutable state**
```python
# Suspects in order:
# 1. Class-level or module-level variables mutated per-request
# 2. Dict/list passed by reference, mutated, not copied
# 3. ThreadLocal values not reset between requests (in thread pool)
# 4. Cached results that include request-specific data
# 5. Generator/iterator objects that aren\'t per-request

# Pattern that causes this:
class Handler:
    result = []          # Class variable — shared across all instances
    def process(self, items):
        self.result.clear()
        self.result.extend(items)   # Thread 1 clears, Thread 2 extends → wrong results
```

**Step 4: Check thread safety of your dependencies**
Many Python objects are NOT thread-safe: `random.Random` (shared state), `datetime.datetime` (fine, immutable), database cursors (not safe — use per-thread connections), most logging handlers (safe via GIL but output may interleave).

**Step 5: Verify with a mutex**
If adding a lock around the suspected section makes the problem disappear, you confirmed a race condition. Then redesign to eliminate the shared state rather than just adding locks everywhere.''',
    "error-handling", "intermittent bugs under load — concurrency investigation"
))

pairs.append(pair(
    "What's the difference between a crash and a hang, and how do you debug each?",
    '''Crashes vs hangs — completely different debugging approaches.

**Crash: process terminates unexpectedly**

Symptoms: exit code != 0, exception in logs, OOM kill, segfault

**Debug path:**
```bash
# 1. Check exit code
echo $?     # 137 = OOM kill, 139 = segfault, 1 = unhandled exception

# 2. Check system logs for OOM
dmesg | grep -i "killed process"
journalctl -k | grep -i oom

# 3. For Python: get the full traceback
# Enable faulthandler for segfaults (C extension crashes)
python3 -X faulthandler yourscript.py

# 4. Set up crash reporting if not already done
import sys, traceback, logging

def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# 5. For intermittent crashes: add signal handler to capture state at death
import signal, traceback

def debug_signal(sig, frame):
    traceback.print_stack(frame)

signal.signal(signal.SIGUSR1, debug_signal)
# Then: kill -USR1 <pid> when you want a stack snapshot
```

**Hang: process is alive but not progressing**

Symptoms: requests time out, CPU at 0% or 100%, process unresponsive

**Debug path:**
```bash
# 1. Is it CPU-bound (busy) or waiting?
top -p <pid>    # CPU near 100% = busy loop; near 0% = waiting

# 2. What is it waiting on?
strace -p <pid>    # Linux: shows system calls in progress
lsof -p <pid>      # what files/sockets are open

# 3. Get Python thread stacks (no restart needed)
kill -3 <pid>       # sends SIGQUIT — Python prints all thread stacks to stderr

# Or with py-spy (no code changes needed):
py-spy dump --pid <pid>

# 4. For deadlock: stacks will show threads blocked on lock.acquire()
# All blocked threads are waiting → deadlock
# One thread running in a loop → infinite loop

# 5. For async hangs: dump the event loop
import asyncio
# In running code:
for task in asyncio.all_tasks():
    print(task.get_name(), task.get_stack())
```

**The quick classification:**
- All threads blocked → deadlock
- One thread at 100% CPU → infinite loop (likely a condition that never becomes False)
- Process waiting on I/O → socket/file/DB call that never returns (check timeouts on your clients)
- Memory growing before hang → OOM incoming
- Intermittent, only under load → race condition that leads to a bad state''',
    "error-handling", "crash vs hang — systematic debugging, py-spy, signal handlers"
))

pairs.append(pair(
    "Explain how SQL injection works at the database driver level and why parameterized queries actually fix it.",
    '''SQL injection — mechanism at the wire level, and why parameterization is the only real fix.

**What happens with string concatenation:**

```python
# Vulnerable
query = f"SELECT * FROM users WHERE username = \'{username}\'"
cursor.execute(query)
```

If `username = "admin\'--"`, the query becomes:
```sql
SELECT * FROM users WHERE username = \'admin\'--\'
```

The `--` is a SQL comment. Everything after it is ignored, including the password check. The database received valid SQL with the injected logic.

At the wire level, the database receives a single string. It parses it as SQL. It cannot distinguish intended query structure from attacker-injected structure — they arrived as one blob.

**What parameterized queries actually do:**

```python
# Safe
cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
```

The database driver sends this in two separate messages:
1. The query template: `SELECT * FROM users WHERE username = ?` — parsed as SQL structure
2. The parameter values: `("admin\'--",)` — sent as data, never parsed as SQL

The database parses the template first, builds the query execution plan, then slots the parameter values in as literals. The single quote in `admin\'--` is treated as a character in a string value, not a SQL delimiter. The injection is structurally impossible.

**Why string-escaping functions like `mysql_real_escape_string` are insufficient:**

They rely on knowing all the ways quotes can be used in a query. Edge cases:
- Numeric contexts don\'t need quotes: `WHERE id = ` + user_input — no quote to escape
- Multibyte character sets: some encodings allow bytes that look like `\\\'` to pass through escaping
- Developer forgets to call the function on one parameter

Parameterized queries don\'t rely on escaping. The structure vs. data separation is enforced by the protocol, not by text manipulation.

**ORM doesn\'t automatically protect you:**

```python
# Still vulnerable — raw SQL in an ORM
User.objects.raw(f"SELECT * FROM users WHERE id = {user_id}")

# Safe
User.objects.filter(id=user_id)
User.objects.raw("SELECT * FROM users WHERE id = %s", [user_id])
```

If you\'re concatenating user input into SQL anywhere — even inside an ORM — you\'re vulnerable.''',
    "error-handling", "SQL injection — protocol-level explanation, parameterization mechanism"
))


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

total = len(pairs)
print(f"Generated {total} reasoning pairs", file=sys.stderr)

if DRY_RUN:
    from collections import Counter
    cats = Counter(p["metadata"]["category"] for p in pairs)
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"\nSample titles:")
    for p in pairs:
        print(f"  [{p['metadata']['category']}] {p['messages'][1]['content'][:70]}")
else:
    output_path = "/home/purple/.purple/book-to-brain/training-data/training.jsonl"
    import subprocess
    lines = "\n".join(json.dumps(p) for p in pairs) + "\n"
    result = subprocess.run(
        ["ssh", "purpleroom", f"cat >> {output_path}"],
        input=lines.encode(),
        capture_output=True
    )
    if result.returncode == 0:
        print(f"✓ {total} pairs appended to purpleroom:{output_path}", file=sys.stderr)
    else:
        print(f"✗ SSH append failed: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
