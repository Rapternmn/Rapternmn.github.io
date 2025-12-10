+++
title = "Concurrency Patterns"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Common concurrency patterns: Producer-Consumer, Reader-Writer, Worker Thread, Future/Promise, and thread-safe Singleton patterns."
+++

# 🎯 Concurrency Patterns

## Common Concurrency Patterns

### 1. Producer-Consumer Pattern

Separates data production from consumption using a shared buffer.

**Use Cases**: Task queues, event processing, logging systems

**Key Concept**: Producers add items to a shared buffer, consumers remove and process them. The buffer handles synchronization automatically.

> **See detailed implementation**: [Producer-Consumer Design]({{< ref "6-Producer_Consumer_Design.md" >}})

---

### 2. Reader-Writer Pattern

Allows multiple readers or a single writer.

**Use Cases**: Caches, databases, configuration management

**Key Concept**: Multiple threads can read simultaneously, but writes require exclusive access. Optimizes for read-heavy workloads.

> **See detailed implementation**: [Read-Write Lock Design]({{< ref "16-ReadWrite_Lock_Design.md" >}})

---

### 3. Worker Thread Pattern (Thread Pool)

Reuses threads to execute tasks from a queue.

**Use Cases**: Web servers, task processing, parallel computation

**Key Concept**: Maintains a pool of worker threads that process tasks from a queue, avoiding the overhead of creating new threads for each task.

> **See detailed implementation**: [Thread Pool Design]({{< ref "5-Thread_Pool_Design.md" >}})

---

### 4. Future/Promise Pattern

Represents a value that will be available in the future.

```python
import threading

class Future:
    def __init__(self):
        self._result = None
        self._exception = None
        self._done = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def set_result(self, result):
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()
    
    def set_exception(self, exception):
        with self._condition:
            self._exception = exception
            self._done = True
            self._condition.notify_all()
    
    def get(self, timeout=None):
        with self._condition:
            while not self._done:
                self._condition.wait(timeout)
            if self._exception:
                raise self._exception
            return self._result

# Usage
future = Future()

def compute():
    result = 42
    future.set_result(result)

threading.Thread(target=compute).start()
result = future.get()  # Blocks until result available
```

---

### 5. Thread-Safe Singleton

Ensures only one instance exists across threads.

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check
                    cls._instance = super().__new__(cls)
        return cls._instance

# Usage
s1 = ThreadSafeSingleton()
s2 = ThreadSafeSingleton()
print(s1 is s2)  # True
```

---

### 6. Barrier Pattern

Synchronizes multiple threads at a barrier point.

```python
import threading

class Barrier:
    def __init__(self, count):
        self.count = count
        self.waiting = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def wait(self):
        with self.condition:
            self.waiting += 1
            if self.waiting < self.count:
                self.condition.wait()
            else:
                self.condition.notify_all()
                self.waiting = 0

# Usage
barrier = Barrier(3)

def worker(name):
    print(f"{name} starting")
    barrier.wait()
    print(f"{name} passed barrier")

threading.Thread(target=worker, args=("A",)).start()
threading.Thread(target=worker, args=("B",)).start()
threading.Thread(target=worker, args=("C",)).start()
```

---

## Pattern Selection Guide

| Pattern | Use When | Detailed Implementation |
|---------|----------|------------------------|
| Producer-Consumer | Decoupling producers and consumers | [Producer-Consumer Design]({{< ref "6-Producer_Consumer_Design.md" >}}) |
| Reader-Writer | Many reads, few writes | [Read-Write Lock Design]({{< ref "16-ReadWrite_Lock_Design.md" >}}) |
| Worker Thread | Need to limit concurrent threads | [Thread Pool Design]({{< ref "5-Thread_Pool_Design.md" >}}) |
| Future/Promise | Asynchronous task execution | See Future/Promise pattern above |
| Singleton | Single shared instance needed | See Thread-Safe Singleton above |
| Barrier | Synchronize multiple threads | See Barrier pattern above |

---

## Key Takeaways

- Producer-Consumer decouples production from consumption
- Reader-Writer optimizes for read-heavy workloads
- Worker Thread pattern reuses threads efficiently
- Future/Promise enables asynchronous programming
- Detailed implementations available in dedicated design files
- Choose patterns based on your specific concurrency needs

