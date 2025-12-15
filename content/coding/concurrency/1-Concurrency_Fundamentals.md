+++
title = "Concurrency Fundamentals"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 1
description = "Fundamentals of concurrency: threads vs processes, threading basics, thread lifecycle, thread communication, and when to use concurrency in system design."
+++

# 🔄 Concurrency Fundamentals

## What is Concurrency?

**Concurrency** is the ability of a system to handle multiple tasks simultaneously, making progress on more than one task at a time. It's essential for building responsive, scalable systems.

### Concurrency vs Parallelism

- **Concurrency**: Multiple tasks making progress at the same time (may be interleaved on single CPU)
- **Parallelism**: Multiple tasks executing simultaneously on multiple CPUs/cores

**Example**:
- **Concurrent**: Web server handling multiple requests (switching between them)
- **Parallel**: Image processing on multiple CPU cores simultaneously

---

## Threads vs Processes

### Process
- Independent execution unit with its own memory space
- Isolated from other processes
- Heavyweight (more memory, slower to create)
- Inter-process communication (IPC) required

### Thread
- Lightweight execution unit within a process
- Shares memory space with other threads in same process
- Faster to create and switch
- Direct memory access (requires synchronization)

**When to Use**:
- **Processes**: When you need isolation, fault tolerance, or true parallelism
- **Threads**: When you need lightweight concurrency, shared memory, or I/O-bound tasks

---

## Threading Basics

### Creating Threads

```python
import threading
import time

def worker(name):
    print(f"Thread {name} starting")
    time.sleep(2)
    print(f"Thread {name} finished")

# Create threads
t1 = threading.Thread(target=worker, args=("Thread-1",))
t2 = threading.Thread(target=worker, args=("Thread-2",))

# Start threads
t1.start()
t2.start()

# Wait for completion
t1.join()
t2.join()
print("All threads completed")
```

### Thread Lifecycle

1. **New**: Thread object created
2. **Runnable**: Thread started, ready to run
3. **Running**: Thread executing
4. **Blocked**: Thread waiting (I/O, lock, etc.)
5. **Terminated**: Thread finished

### Thread Communication

**Shared Memory** (most common):
```python
shared_data = []
lock = threading.Lock()

def producer():
    for i in range(5):
        with lock:
            shared_data.append(i)
            print(f"Produced: {i}")

def consumer():
    while len(shared_data) < 5:
        time.sleep(0.1)
    with lock:
        print(f"Consumed: {shared_data}")
```

**Message Passing** (queues):
```python
import queue

q = queue.Queue()

def producer():
    for i in range(5):
        q.put(i)
        print(f"Produced: {i}")

def consumer():
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        q.task_done()
```

---

## When to Use Concurrency

### ✅ Good Use Cases

1. **I/O-Bound Tasks**: Network requests, file operations, database queries
2. **Multiple Independent Tasks**: Processing multiple requests simultaneously
3. **Background Tasks**: Logging, monitoring, cleanup
4. **Real-time Systems**: User interfaces, game loops, streaming

### ❌ When NOT to Use

1. **CPU-Bound Sequential Tasks**: May actually slow down due to overhead
2. **Simple Operations**: Overhead not worth it
3. **Shared State Complexity**: When synchronization becomes too complex

---

## Common Concurrency Challenges

1. **Race Conditions**: Multiple threads accessing shared data
2. **Deadlocks**: Threads waiting for each other indefinitely
3. **Livelocks**: Threads keep retrying but make no progress
4. **Starvation**: Some threads never get resources

---

## Key Takeaways

- Concurrency enables handling multiple tasks simultaneously
- Threads are lightweight, processes provide isolation
- Shared memory requires synchronization
- Use concurrency for I/O-bound tasks and independent operations
- Always consider thread safety when sharing data

