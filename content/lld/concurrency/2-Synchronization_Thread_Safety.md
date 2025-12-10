+++
title = "Synchronization & Thread Safety"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Synchronization primitives, thread safety, race conditions, deadlocks, and solutions for building thread-safe concurrent systems."
+++

# 🔒 Synchronization & Thread Safety

## Thread Safety

**Thread safety** means that a piece of code can be safely used by multiple threads simultaneously without causing data corruption or unexpected behavior.

### The Problem: Race Conditions

```python
# UNSAFE: Race condition
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # Not atomic!

# Multiple threads incrementing
t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # May not be 200000!
```

---

## Synchronization Primitives

### 1. Locks (Mutex)

**Basic Lock**:
```python
import threading

lock = threading.Lock()
shared_data = []

def safe_append(item):
    with lock:  # Acquires lock, releases on exit
        shared_data.append(item)
```

**Reentrant Lock** (same thread can acquire multiple times):
```python
rlock = threading.RLock()

def function_a():
    with rlock:
        function_b()  # Can acquire again

def function_b():
    with rlock:  # OK - same thread
        pass
```

### 2. Semaphores

Controls access to a resource with limited capacity:

```python
semaphore = threading.Semaphore(3)  # Allow 3 concurrent accesses

def access_resource():
    with semaphore:
        # Only 3 threads can execute this at once
        print("Accessing resource...")
        time.sleep(1)
```

**Use Cases**:
- Limiting concurrent database connections
- Rate limiting
- Resource pool management

### 3. Condition Variables

Allows threads to wait for a condition:

```python
condition = threading.Condition()
queue = []
MAX_SIZE = 5

def producer():
    for i in range(10):
        with condition:
            while len(queue) >= MAX_SIZE:
                condition.wait()  # Wait until space available
            queue.append(i)
            print(f"Produced: {i}")
            condition.notify()  # Notify waiting consumers

def consumer():
    for _ in range(10):
        with condition:
            while len(queue) == 0:
                condition.wait()  # Wait until item available
            item = queue.pop(0)
            print(f"Consumed: {item}")
            condition.notify()  # Notify waiting producers
```

### 4. Event Objects

Simple signaling mechanism:

```python
event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # Blocks until event is set
    print("Event occurred!")

def setter():
    time.sleep(2)
    event.set()  # Signal all waiting threads
```

### 5. Atomic Operations

Operations that complete in a single step:

```python
import threading

# Thread-safe counter
counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:
        counter += 1  # Atomic with lock
```

**Python's `queue` module** provides thread-safe operations:
```python
import queue

q = queue.Queue()  # Thread-safe queue
q.put(item)        # Thread-safe
item = q.get()     # Thread-safe
```

---

## Common Concurrency Problems

### 1. Race Conditions

**Problem**: Multiple threads access shared data without synchronization.

**Solution**: Use locks or atomic operations.

```python
# Thread-safe counter
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def get(self):
        with self._lock:
            return self._value
```

### 2. Deadlocks

**Problem**: Two or more threads waiting for each other indefinitely.

```python
# DEADLOCK EXAMPLE
lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    with lock1:
        time.sleep(0.1)
        with lock2:  # Waiting for lock2
            pass

def thread2():
    with lock2:
        time.sleep(0.1)
        with lock1:  # Waiting for lock1 - DEADLOCK!
            pass
```

**Solutions**:
1. **Lock Ordering**: Always acquire locks in same order
2. **Timeout**: Use `lock.acquire(timeout=5)`
3. **Avoid Nested Locks**: Minimize lock dependencies

```python
# FIXED: Same lock order
def thread1():
    with lock1:
        with lock2:  # Always lock1 then lock2
            pass

def thread2():
    with lock1:  # Same order
        with lock2:
            pass
```

### 3. Livelocks

**Problem**: Threads keep retrying but make no progress.

**Example**: Two threads trying to pass through a narrow door, both step back and retry.

**Solution**: Add randomization or backoff to break symmetry.

### 4. Starvation

**Problem**: Some threads never get access to resources.

**Solution**: Use fair locks or priority queues.

---

## Thread-Safe Data Structures

### Python's Thread-Safe Collections

```python
import queue

# Thread-safe queue
q = queue.Queue()
q.put(item)
item = q.get()

# Thread-safe priority queue
pq = queue.PriorityQueue()
pq.put((priority, item))

# Thread-safe LIFO queue
lifo = queue.LifoQueue()
```

### Custom Thread-Safe Dictionary

```python
class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
    
    def get(self, key):
        with self._lock:
            return self._dict.get(key)
    
    def set(self, key, value):
        with self._lock:
            self._dict[key] = value
    
    def delete(self, key):
        with self._lock:
            return self._dict.pop(key, None)
```

---

## Best Practices

1. **Minimize Shared State**: Reduce data shared between threads
2. **Use Immutable Objects**: Immutable data is naturally thread-safe
3. **Lock Granularity**: Lock only what's necessary, not entire functions
4. **Avoid Deadlocks**: Consistent lock ordering, timeouts
5. **Use Thread-Safe Collections**: Prefer `queue.Queue` over lists
6. **Document Thread Safety**: Clearly document which methods are thread-safe

---

## Key Takeaways

- Use locks to protect shared data from race conditions
- Semaphores control access to limited resources
- Condition variables enable efficient waiting
- Deadlocks can be prevented with consistent lock ordering
- Prefer thread-safe collections when possible
- Minimize shared state to reduce synchronization needs

