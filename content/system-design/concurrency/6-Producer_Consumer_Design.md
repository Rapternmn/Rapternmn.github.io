+++
title = "Producer-Consumer Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Producer-Consumer pattern implementation: Thread-safe queue, multiple producers/consumers, backpressure handling, and practical examples."
+++

# 📦 Producer-Consumer Design

## Problem Statement

Design a producer-consumer system where:
- Multiple producers generate data
- Multiple consumers process data
- Thread-safe communication between them
- Handles backpressure (when producers are faster)
- Graceful shutdown

**Use Cases**:
- Task queues
- Log processing
- Event streaming
- Data pipeline
- Message queues

---

## Requirements

1. **Thread Safety**: Safe concurrent access to shared buffer
2. **Blocking Operations**: Producers block when buffer is full
3. **Multiple Producers/Consumers**: Support multiple threads
4. **Backpressure**: Handle when producers are faster
5. **Graceful Shutdown**: Complete pending tasks

---

## Basic Implementation

```python
import threading
import queue
import time
import random

class ProducerConsumer:
    """
    Basic producer-consumer with thread-safe queue
    """
    
    def __init__(self, buffer_size: int = 10):
        """
        Args:
            buffer_size: Maximum items in buffer
        """
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_flag = threading.Event()
    
    def producer(self, producer_id: int, num_items: int):
        """Producer thread"""
        for i in range(num_items):
            if self.stop_flag.is_set():
                break
            
            item = f"Item-{producer_id}-{i}"
            self.buffer.put(item)  # Blocks if queue is full
            print(f"Producer {producer_id} produced: {item}")
            time.sleep(random.uniform(0.1, 0.3))
        
        print(f"Producer {producer_id} finished")
    
    def consumer(self, consumer_id: int):
        """Consumer thread"""
        while True:
            try:
                # Wait with timeout to check stop flag
                item = self.buffer.get(timeout=1)
                print(f"Consumer {consumer_id} consumed: {item}")
                self.buffer.task_done()
                
                # Simulate processing
                time.sleep(random.uniform(0.2, 0.4))
            except queue.Empty:
                if self.stop_flag.is_set() and self.buffer.empty():
                    break
        
        print(f"Consumer {consumer_id} finished")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.stop_flag.set()
        self.buffer.join()  # Wait for all tasks to be processed


# Usage
pc = ProducerConsumer(buffer_size=5)

# Start producers
producers = []
for i in range(2):
    p = threading.Thread(
        target=pc.producer,
        args=(i, 10),
        name=f"Producer-{i}"
    )
    p.start()
    producers.append(p)

# Start consumers
consumers = []
for i in range(3):
    c = threading.Thread(
        target=pc.consumer,
        args=(i,),
        name=f"Consumer-{i}"
    )
    c.start()
    consumers.append(c)

# Wait for producers
for p in producers:
    p.join()

# Shutdown consumers
pc.shutdown()

# Wait for consumers
for c in consumers:
    c.join()

print("All done!")
```

---

## Advanced: Priority Queue

```python
import threading
import queue
import time

class PriorityProducerConsumer:
    """
    Producer-consumer with priority queue
    """
    
    def __init__(self, buffer_size: int = 10):
        self.buffer = queue.PriorityQueue(maxsize=buffer_size)
        self.stop_flag = threading.Event()
    
    def producer(self, producer_id: int, items: list):
        """Producer with priority items"""
        for priority, item in items:
            if self.stop_flag.is_set():
                break
            
            # Lower number = higher priority
            self.buffer.put((priority, f"Item-{producer_id}-{item}"))
            print(f"Producer {producer_id} produced (priority {priority}): {item}")
            time.sleep(0.1)
    
    def consumer(self, consumer_id: int):
        """Consumer processes items by priority"""
        while True:
            try:
                priority, item = self.buffer.get(timeout=1)
                print(f"Consumer {consumer_id} consumed (priority {priority}): {item}")
                self.buffer.task_done()
                time.sleep(0.2)
            except queue.Empty:
                if self.stop_flag.is_set() and self.buffer.empty():
                    break


# Usage
pc = PriorityProducerConsumer()

# Producer with mixed priorities
items = [(3, "low"), (1, "high"), (2, "medium"), (1, "high2")]
threading.Thread(target=pc.producer, args=(1, items)).start()

# Consumer
threading.Thread(target=pc.consumer, args=(1,)).start()

time.sleep(2)
pc.shutdown()
```

---

## Advanced: Bounded Buffer with Condition Variables

```python
import threading
import time

class BoundedBuffer:
    """
    Bounded buffer using condition variables
    More control than Queue
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
    
    def put(self, item):
        """Add item to buffer, blocks if full"""
        with self.not_full:
            while len(self.buffer) >= self.capacity:
                self.not_full.wait()  # Wait until not full
            
            self.buffer.append(item)
            self.not_empty.notify()  # Notify waiting consumers
    
    def get(self):
        """Remove item from buffer, blocks if empty"""
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()  # Wait until not empty
            
            item = self.buffer.pop(0)
            self.not_full.notify()  # Notify waiting producers
            return item
    
    def size(self):
        with self.lock:
            return len(self.buffer)


# Usage
buffer = BoundedBuffer(capacity=5)

def producer(id, items):
    for item in items:
        buffer.put(f"Item-{id}-{item}")
        print(f"Producer {id} produced: {item}")
        time.sleep(0.1)

def consumer(id):
    for _ in range(5):
        item = buffer.get()
        print(f"Consumer {id} consumed: {item}")
        time.sleep(0.2)

threading.Thread(target=producer, args=(1, range(10))).start()
threading.Thread(target=consumer, args=(1,)).start()
```

---

## Real-World Example: Log Processor

```python
import threading
import queue
import time
from datetime import datetime

class LogProcessor:
    """
    Producer-consumer for log processing
    - Producers: Generate log entries
    - Consumers: Process and write logs
    """
    
    def __init__(self, buffer_size: int = 100):
        self.log_queue = queue.Queue(maxsize=buffer_size)
        self.stop_flag = threading.Event()
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def log_producer(self, source: str, num_logs: int):
        """Generate log entries"""
        for i in range(num_logs):
            if self.stop_flag.is_set():
                break
            
            log_entry = {
                'timestamp': datetime.now(),
                'source': source,
                'message': f"Log entry {i}",
                'level': 'INFO'
            }
            
            self.log_queue.put(log_entry)
            time.sleep(0.05)
    
    def log_consumer(self, consumer_id: int):
        """Process log entries"""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                
                # Process log (format, filter, etc.)
                formatted = (
                    f"[{log_entry['timestamp']}] "
                    f"{log_entry['level']} "
                    f"[{log_entry['source']}] "
                    f"{log_entry['message']}"
                )
                
                # Write to file/database (simulated)
                print(f"Consumer {consumer_id}: {formatted}")
                
                with self.lock:
                    self.processed_count += 1
                
                self.log_queue.task_done()
                time.sleep(0.1)
                
            except queue.Empty:
                if self.stop_flag.is_set() and self.log_queue.empty():
                    break
    
    def shutdown(self):
        """Graceful shutdown"""
        self.stop_flag.set()
        self.log_queue.join()
        print(f"Processed {self.processed_count} logs")


# Usage
processor = LogProcessor()

# Multiple log sources (producers)
sources = ['web-server', 'api-server', 'database']
for source in sources:
    threading.Thread(
        target=processor.log_producer,
        args=(source, 20),
        daemon=True
    ).start()

# Multiple consumers
for i in range(2):
    threading.Thread(
        target=processor.log_consumer,
        args=(i,),
        daemon=True
    ).start()

time.sleep(5)
processor.shutdown()
```

---

## Design Considerations

1. **Buffer Size**: Balance memory vs blocking
2. **Multiple Producers/Consumers**: Scale based on workload
3. **Priority**: Use PriorityQueue for priority-based processing
4. **Backpressure**: Queue size limits prevent memory issues
5. **Graceful Shutdown**: Ensure all items are processed

---

## Key Takeaways

- Producer-Consumer decouples production from consumption
- Thread-safe queues handle synchronization automatically
- Condition variables provide more control
- Priority queues enable priority-based processing
- Graceful shutdown ensures data integrity

