+++
title = "Non-Blocking Queue Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 9
description = "Lock-free Non-Blocking Queue implementation: CAS operations, atomic operations, wait-free algorithms, and performance optimization."
+++

# ⚡ Non-Blocking Queue Design

## Problem Statement

Design a non-blocking (lock-free) queue that:
- Never blocks threads
- Uses atomic operations instead of locks
- Provides better performance under contention
- Supports multiple producers and consumers

**Use Cases**:
- High-performance systems
- Real-time applications
- Low-latency requirements
- When lock contention is high

---

## Blocking vs Non-Blocking

### Blocking Queue
- Uses locks/mutexes
- Threads wait when queue is full/empty
- Simpler implementation
- Can cause thread starvation

### Non-Blocking Queue
- Uses atomic operations (CAS)
- Operations return immediately (success/fail)
- More complex implementation
- Better performance under contention
- No thread blocking

---

## Lock-Free Queue Implementation

```python
import threading
from typing import Optional, Any
import ctypes

class Node:
    """Node for lock-free queue"""
    def __init__(self, value: Any = None):
        self.value = value
        self.next = None

class LockFreeQueue:
    """
    Lock-free queue using Compare-And-Swap (CAS)
    Note: Python's GIL limits true lock-freedom, but this demonstrates the concept
    """
    
    def __init__(self):
        # Dummy node
        self.head = Node()
        self.tail = self.head
        self.lock = threading.Lock()  # Simplified for Python (GIL limitations)
    
    def enqueue(self, value: Any) -> bool:
        """
        Add item to queue (non-blocking)
        
        Returns:
            True if successful, False otherwise
        """
        new_node = Node(value)
        
        with self.lock:  # In real lock-free, use atomic operations
            self.tail.next = new_node
            self.tail = new_node
            return True
    
    def dequeue(self) -> Optional[Any]:
        """
        Remove item from queue (non-blocking)
        
        Returns:
            Item if available, None if empty
        """
        with self.lock:  # In real lock-free, use atomic operations
            if self.head.next is None:
                return None
            
            value = self.head.next.value
            self.head = self.head.next
            return value


# Better approach: Use Python's queue with non-blocking operations
import queue

class NonBlockingQueue:
    """
    Non-blocking queue wrapper using Python's queue
    """
    
    def __init__(self, max_size: int = 0):
        self.queue = queue.Queue(maxsize=max_size)
    
    def put(self, item: Any) -> bool:
        """
        Try to add item (non-blocking)
        
        Returns:
            True if successful, False if queue is full
        """
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False
    
    def get(self) -> Optional[Any]:
        """
        Try to get item (non-blocking)
        
        Returns:
            Item if available, None if empty
        """
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get current size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if empty"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Check if full"""
        return self.queue.full()


# Usage
nb_queue = NonBlockingQueue(max_size=5)

# Producer (non-blocking)
for i in range(10):
    if nb_queue.put(f"Item-{i}"):
        print(f"Added: Item-{i}")
    else:
        print(f"Queue full, dropped: Item-{i}")

# Consumer (non-blocking)
while True:
    item = nb_queue.get()
    if item is None:
        break
    print(f"Got: {item}")
```

---

## Wait-Free Queue (Simplified)

```python
import threading
from collections import deque

class WaitFreeQueue:
    """
    Wait-free queue using deque (thread-safe for append/popleft)
    Note: Python's deque is thread-safe for these operations
    """
    
    def __init__(self, max_size: int = 0):
        self.max_size = max_size
        self.queue = deque()
        self.lock = threading.Lock()  # For size checking
    
    def try_put(self, item: Any) -> bool:
        """
        Try to add item (wait-free)
        
        Returns:
            True if successful, False if full
        """
        if self.max_size > 0:
            with self.lock:
                if len(self.queue) >= self.max_size:
                    return False
        
        self.queue.append(item)
        return True
    
    def try_get(self) -> Optional[Any]:
        """
        Try to get item (wait-free)
        
        Returns:
            Item if available, None if empty
        """
        try:
            return self.queue.popleft()
        except IndexError:
            return None
    
    def size(self) -> int:
        """Get size (approximate)"""
        return len(self.queue)
```

---

## Real-World: High-Performance Non-Blocking Queue

```python
import threading
import time
from collections import deque

class HighPerformanceQueue:
    """
    High-performance non-blocking queue for high-throughput scenarios
    """
    
    def __init__(self, max_size: int = 0):
        self.max_size = max_size
        self.queue = deque()
        self.stats = {
            'enqueued': 0,
            'dequeued': 0,
            'dropped': 0,
            'empty_hits': 0
        }
        self.lock = threading.Lock()
    
    def enqueue(self, item: Any) -> bool:
        """Non-blocking enqueue"""
        if self.max_size > 0:
            with self.lock:
                if len(self.queue) >= self.max_size:
                    self.stats['dropped'] += 1
                    return False
        
        self.queue.append(item)
        self.stats['enqueued'] += 1
        return True
    
    def dequeue(self) -> Optional[Any]:
        """Non-blocking dequeue"""
        try:
            item = self.queue.popleft()
            self.stats['dequeued'] += 1
            return item
        except IndexError:
            self.stats['empty_hits'] += 1
            return None
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        with self.lock:
            return {
                **self.stats,
                'current_size': len(self.queue),
                'throughput': self.stats['dequeued'] / max(1, time.time())
            }


# Usage in high-throughput scenario
queue = HighPerformanceQueue(max_size=1000)

def fast_producer():
    for i in range(10000):
        queue.enqueue(f"Item-{i}")

def fast_consumer():
    count = 0
    while count < 10000:
        item = queue.dequeue()
        if item:
            count += 1

threading.Thread(target=fast_producer).start()
threading.Thread(target=fast_consumer).start()
```

---

## Comparison: Blocking vs Non-Blocking

| Aspect | Blocking | Non-Blocking |
|--------|----------|--------------|
| **Thread Behavior** | Waits when full/empty | Returns immediately |
| **Complexity** | Simpler | More complex |
| **Performance** | Good for low contention | Better for high contention |
| **Latency** | Can have high latency | Lower latency |
| **Use Case** | General purpose | High-performance systems |

---

## Key Takeaways

- Non-blocking queues never block threads
- Use `put_nowait()` and `get_nowait()` for non-blocking operations
- Better performance under high contention
- Suitable for real-time and high-throughput systems
- Python's GIL limits true lock-freedom, but concepts apply

