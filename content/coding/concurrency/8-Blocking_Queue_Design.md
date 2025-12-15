+++
title = "Blocking Queue Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Thread-safe Blocking Queue implementation: Producer-Consumer synchronization, bounded/unbounded queues, timeout support, and priority queues."
+++

# 🚧 Blocking Queue Design

## Problem Statement

Design a thread-safe blocking queue where:
- `put()` blocks when queue is full
- `get()` blocks when queue is empty
- Supports timeout for operations
- Works with multiple producers and consumers
- Thread-safe operations

**Use Cases**:
- Producer-Consumer patterns
- Task queues
- Resource pooling
- Backpressure handling

---

## Requirements

1. **Blocking Operations**: Block when queue is full/empty
2. **Thread Safety**: Safe concurrent access
3. **Timeout Support**: Optional timeout for operations
4. **Bounded/Unbounded**: Support both fixed and unlimited size
5. **Priority Support**: Optional priority-based ordering

---

## Basic Blocking Queue

```python
import threading
import time
from typing import Optional, Any

class BlockingQueue:
    """
    Thread-safe blocking queue using condition variables
    """
    
    def __init__(self, max_size: int = 0):
        """
        Args:
            max_size: Maximum queue size (0 = unbounded)
        """
        self.max_size = max_size
        self.queue = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def put(self, item: Any, timeout: Optional[float] = None):
        """
        Add item to queue, blocks if full
        
        Args:
            item: Item to add
            timeout: Maximum time to wait (None = wait indefinitely)
        
        Raises:
            queue.Full: If timeout expires
        """
        with self.not_full:
            if self.max_size > 0:
                end_time = time.time() + timeout if timeout else None
                while len(self.queue) >= self.max_size:
                    if timeout is not None:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            raise queue.Full("Queue is full")
                        self.not_full.wait(timeout=remaining)
                    else:
                        self.not_full.wait()
            
            self.queue.append(item)
            self.not_empty.notify()  # Notify waiting consumers
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """
        Remove and return item, blocks if empty
        
        Args:
            timeout: Maximum time to wait (None = wait indefinitely)
        
        Returns:
            Item from queue
        
        Raises:
            queue.Empty: If timeout expires
        """
        with self.not_empty:
            if timeout is not None:
                end_time = time.time() + timeout
                while len(self.queue) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise queue.Empty("Queue is empty")
                    self.not_empty.wait(timeout=remaining)
            else:
                while len(self.queue) == 0:
                    self.not_empty.wait()
            
            item = self.queue.pop(0)
            if self.max_size > 0:
                self.not_full.notify()  # Notify waiting producers
            return item
    
    def size(self) -> int:
        """Get current queue size"""
        with self.lock:
            return len(self.queue)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0
    
    def full(self) -> bool:
        """Check if queue is full"""
        with self.lock:
            return self.max_size > 0 and len(self.queue) >= self.max_size


# Usage Example
queue = BlockingQueue(max_size=5)

def producer():
    for i in range(10):
        queue.put(f"Item-{i}")
        print(f"Produced: Item-{i}")

def consumer():
    for _ in range(10):
        item = queue.get()
        print(f"Consumed: {item}")
        time.sleep(0.1)

threading.Thread(target=producer).start()
threading.Thread(target=consumer).start()
```

---

## Priority Blocking Queue

```python
import threading
import heapq
from typing import Optional, Any, Tuple

class PriorityBlockingQueue:
    """
    Blocking queue with priority support
    Lower priority number = higher priority
    """
    
    def __init__(self, max_size: int = 0):
        self.max_size = max_size
        self.queue = []  # Min heap
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.counter = 0  # For tie-breaking
    
    def put(self, item: Any, priority: int = 0, timeout: Optional[float] = None):
        """
        Add item with priority
        
        Args:
            item: Item to add
            priority: Priority (lower = higher priority)
            timeout: Maximum time to wait
        """
        with self.not_full:
            if self.max_size > 0:
                end_time = time.time() + timeout if timeout else None
                while len(self.queue) >= self.max_size:
                    if timeout is not None:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            raise queue.Full("Queue is full")
                        self.not_full.wait(timeout=remaining)
                    else:
                        self.not_full.wait()
            
            # Use counter for tie-breaking (FIFO for same priority)
            heapq.heappush(self.queue, (priority, self.counter, item))
            self.counter += 1
            self.not_empty.notify()
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get highest priority item"""
        with self.not_empty:
            if timeout is not None:
                end_time = time.time() + timeout
                while len(self.queue) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise queue.Empty("Queue is empty")
                    self.not_empty.wait(timeout=remaining)
            else:
                while len(self.queue) == 0:
                    self.not_empty.wait()
            
            _, _, item = heapq.heappop(self.queue)
            if self.max_size > 0:
                self.not_full.notify()
            return item


# Usage
pq = PriorityBlockingQueue(max_size=10)

pq.put("low priority", priority=3)
pq.put("high priority", priority=1)
pq.put("medium priority", priority=2)

print(pq.get())  # "high priority"
print(pq.get())  # "medium priority"
print(pq.get())  # "low priority"
```

---

## Bounded Blocking Queue with Statistics

```python
import threading
import time
from collections import deque

class BoundedBlockingQueue:
    """
    Bounded blocking queue with statistics tracking
    """
    
    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        
        # Statistics
        self.total_put = 0
        self.total_get = 0
        self.blocked_put = 0
        self.blocked_get = 0
    
    def put(self, item: Any, timeout: Optional[float] = None):
        """Add item, blocks if full"""
        with self.not_full:
            start_time = time.time()
            while len(self.queue) >= self.max_size:
                self.blocked_put += 1
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise queue.Full("Queue is full")
                    self.not_full.wait(timeout=remaining)
                else:
                    self.not_full.wait()
            
            self.queue.append(item)
            self.total_put += 1
            self.not_empty.notify()
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item, blocks if empty"""
        with self.not_empty:
            start_time = time.time()
            while len(self.queue) == 0:
                self.blocked_get += 1
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise queue.Empty("Queue is empty")
                    self.not_empty.wait(timeout=remaining)
                else:
                    self.not_empty.wait()
            
            item = self.queue.popleft()
            self.total_get += 1
            self.not_full.notify()
            return item
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        with self.lock:
            return {
                'size': len(self.queue),
                'max_size': self.max_size,
                'total_put': self.total_put,
                'total_get': self.total_get,
                'blocked_put': self.blocked_put,
                'blocked_get': self.blocked_get,
                'utilization': len(self.queue) / self.max_size
            }
```

---

## Key Takeaways

- Blocking queues use condition variables for efficient waiting
- Bounded queues prevent memory issues
- Priority queues enable priority-based processing
- Timeout support prevents indefinite blocking
- Statistics help monitor queue performance

