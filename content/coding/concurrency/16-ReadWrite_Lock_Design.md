+++
title = "Read-Write Lock Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 16
description = "Read-Write Lock implementation: Multiple readers, exclusive writer, reader-writer synchronization, and performance optimization."
+++

# 🔐 Read-Write Lock Design

## Problem Statement

Design a read-write lock that:
- Allows multiple concurrent readers
- Allows only one exclusive writer
- Prevents readers when writer is active
- Prevents writers when readers are active
- Optimizes for read-heavy workloads

**Use Cases**:
- Caches (read-heavy)
- Configuration management
- Database connections
- Shared data structures
- File systems

---

## Why Read-Write Lock?

**Regular Lock**: Only one thread at a time
**Read-Write Lock**: Multiple readers OR one writer

**Performance Benefit**: Read-heavy workloads see significant improvement

---

## Basic Read-Write Lock

```python
import threading
import time

class ReadWriteLock:
    """
    Basic read-write lock implementation
    """
    
    def __init__(self):
        self._readers = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
    
    def acquire_read(self):
        """Acquire read lock (multiple readers allowed)"""
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                # First reader acquires write lock
                self._write_lock.acquire()
    
    def release_read(self):
        """Release read lock"""
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                # Last reader releases write lock
                self._write_lock.release()
    
    def acquire_write(self):
        """Acquire write lock (exclusive)"""
        self._write_lock.acquire()
    
    def release_write(self):
        """Release write lock"""
        self._write_lock.release()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, *args):
        pass


# Usage
rw_lock = ReadWriteLock()
shared_data = {}

def reader(reader_id: int):
    rw_lock.acquire_read()
    try:
        print(f"Reader {reader_id} reading: {shared_data}")
        time.sleep(0.1)
    finally:
        rw_lock.release_read()

def writer(writer_id: int, value: dict):
    rw_lock.acquire_write()
    try:
        print(f"Writer {writer_id} writing")
        shared_data.update(value)
        time.sleep(0.1)
    finally:
        rw_lock.release_write()

# Multiple readers can read concurrently
threading.Thread(target=reader, args=(1,)).start()
threading.Thread(target=reader, args=(2,)).start()

# Writer blocks all readers
threading.Thread(target=writer, args=(1, {"key": "value"})).start()
```

---

## Advanced: Read-Write Lock with Context Managers

```python
import threading
import time

class ReadWriteLockContext:
    """
    Read-write lock with context manager support
    """
    
    def __init__(self):
        self._readers = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
    
    def acquire_read(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()
    
    def release_read(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()
    
    def acquire_write(self):
        self._write_lock.acquire()
    
    def release_write(self):
        self._write_lock.release()
    
    class ReadLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock
        
        def __enter__(self):
            self.rw_lock.acquire_read()
            return self
        
        def __exit__(self, *args):
            self.rw_lock.release_read()
    
    class WriteLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock
        
        def __enter__(self):
            self.rw_lock.acquire_write()
            return self
        
        def __exit__(self, *args):
            self.rw_lock.release_write()
    
    def read_lock(self):
        """Get read lock context manager"""
        return self.ReadLock(self)
    
    def write_lock(self):
        """Get write lock context manager"""
        return self.WriteLock(self)


# Usage
rw_lock = ReadWriteLockContext()
data = {}

# Read with context manager
with rw_lock.read_lock():
    value = data.get("key")

# Write with context manager
with rw_lock.write_lock():
    data["key"] = "value"
```

---

## Real-World: Thread-Safe Cache with Read-Write Lock

```python
import threading
from collections import OrderedDict

class ReadWriteCache:
    """
    Thread-safe cache using read-write lock
    Optimized for read-heavy workloads
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.rw_lock = ReadWriteLockContext()
    
    def get(self, key: str):
        """Get value (read operation)"""
        with self.rw_lock.read_lock():
            if key in self.cache:
                # Move to end (LRU)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value):
        """Put value (write operation)"""
        with self.rw_lock.write_lock():
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # Remove oldest
            self.cache[key] = value
    
    def size(self) -> int:
        """Get cache size"""
        with self.rw_lock.read_lock():
            return len(self.cache)


# Usage
cache = ReadWriteCache(capacity=5)

# Multiple concurrent reads
def reader(key):
    value = cache.get(key)
    print(f"Read {key}: {value}")

# Exclusive writes
def writer(key, value):
    cache.put(key, value)
    print(f"Wrote {key}: {value}")

# Concurrent reads are fast
threading.Thread(target=reader, args=("a",)).start()
threading.Thread(target=reader, args=("b",)).start()

# Writes block reads
threading.Thread(target=writer, args=("a", 1)).start()
```

---

## Performance Comparison

```python
import threading
import time

def benchmark_reads(lock_type: str, num_reads: int = 1000):
    """Benchmark read performance"""
    if lock_type == "regular":
        lock = threading.Lock()
        def read():
            with lock:
                pass
    else:  # read-write
        rw_lock = ReadWriteLockContext()
        def read():
            with rw_lock.read_lock():
                pass
    
    start = time.time()
    threads = []
    for _ in range(10):  # 10 concurrent readers
        for _ in range(num_reads // 10):
            t = threading.Thread(target=read)
            threads.append(t)
            t.start()
    
    for t in threads:
        t.join()
    
    duration = time.time() - start
    print(f"{lock_type} lock: {duration:.3f}s for {num_reads} reads")

# Compare performance
benchmark_reads("regular", 1000)
benchmark_reads("read-write", 1000)
```

---

## Key Takeaways

- Read-write locks optimize for read-heavy workloads
- Multiple readers can access concurrently
- Writers get exclusive access
- Significant performance improvement over regular locks
- Useful for caches and shared data structures

