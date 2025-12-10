+++
title = "Concurrent Cache Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Thread-safe cache implementation: LRU cache, concurrent access, eviction policies, and cache invalidation with Python implementation."
+++

# 💾 Concurrent Cache Design

## Problem Statement

Design a thread-safe cache that:
- Supports concurrent read/write operations
- Implements LRU (Least Recently Used) eviction
- Handles cache misses efficiently
- Supports TTL (Time To Live) expiration
- Is performant under high concurrency

**Use Cases**:
- API response caching
- Database query caching
- Session storage
- Configuration caching

---

## Requirements

1. **Thread Safety**: Safe concurrent access
2. **LRU Eviction**: Remove least recently used items when full
3. **TTL Support**: Optional expiration of cached items
4. **Performance**: Fast lookups and updates
5. **Memory Efficient**: Bounded size

---

## Basic Thread-Safe Cache

```python
import threading
import time
from typing import Optional, Any

class ThreadSafeCache:
    """
    Basic thread-safe cache with size limit
    """
    
    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: Maximum number of items
        """
        self.max_size = max_size
        self.cache = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest (first) item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            return self.cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)
```

---

## LRU Cache Implementation

```python
from collections import OrderedDict
import threading

class LRUCache:
    """
    Thread-safe LRU Cache
    
    Uses OrderedDict to maintain insertion order
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value and move to end (most recently used)
        
        Returns:
            Value if exists, None otherwise
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """
        Add/update value
        
        If key exists, update and move to end
        If new key and at capacity, evict LRU item
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.capacity:
                    # Evict least recently used (first item)
                    self.cache.popitem(last=False)
                
                self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            return self.cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)


# Usage
cache = LRUCache(capacity=3)

cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # Moves 'a' to end
cache.put("d", 4)      # Evicts 'b' (least recently used)
print(cache.get("b"))  # None (evicted)
```

---

## LRU Cache with TTL

```python
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, Any

class LRUCacheWithTTL:
    """
    Thread-safe LRU Cache with TTL (Time To Live)
    """
    
    def __init__(self, capacity: int, default_ttl: Optional[float] = None):
        """
        Args:
            capacity: Maximum number of items
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # key -> (value, expiry_time)
        self.lock = threading.RLock()
    
    def _is_expired(self, expiry_time: float) -> bool:
        """Check if item has expired"""
        return time.time() > expiry_time
    
    def _get_expiry(self, ttl: Optional[float] = None) -> float:
        """Get expiry timestamp"""
        ttl = ttl or self.default_ttl
        if ttl is None:
            return float('inf')  # Never expires
        return time.time() + ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value if exists and not expired
        
        Returns:
            Value if exists and valid, None otherwise
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            value, expiry_time = self.cache[key]
            
            # Check expiration
            if self._is_expired(expiry_time):
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Add/update value with optional TTL
        """
        with self.lock:
            expiry_time = self._get_expiry(ttl)
            
            if key in self.cache:
                # Update existing
                self.cache[key] = (value, expiry_time)
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.capacity:
                    # Evict expired items first
                    self._evict_expired()
                    
                    # If still at capacity, evict LRU
                    if len(self.cache) >= self.capacity:
                        self.cache.popitem(last=False)
                
                self.cache[key] = (value, expiry_time)
    
    def _evict_expired(self):
        """Remove all expired items"""
        now = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if expiry < now
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def cleanup_expired(self):
        """Manually cleanup expired items"""
        with self.lock:
            self._evict_expired()


# Usage
cache = LRUCacheWithTTL(capacity=5, default_ttl=10.0)

cache.put("key1", "value1")
cache.put("key2", "value2", ttl=5.0)  # Custom TTL

print(cache.get("key1"))  # "value1"
time.sleep(6)
print(cache.get("key2"))  # None (expired)
```

---

## Advanced: Read-Write Optimized Cache

```python
import threading
from collections import OrderedDict

class ReadWriteLRUCache:
    """
    LRU Cache optimized for read-heavy workloads
    Uses read-write lock for better concurrency
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.readers = 0
    
    def _acquire_read(self):
        """Acquire read lock"""
        with self.read_lock:
            self.readers += 1
            if self.readers == 1:
                self.write_lock.acquire()
    
    def _release_read(self):
        """Release read lock"""
        with self.read_lock:
            self.readers -= 1
            if self.readers == 0:
                self.write_lock.release()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value (read operation)"""
        self._acquire_read()
        try:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
        finally:
            self._release_read()
    
    def put(self, key: str, value: Any):
        """Put value (write operation)"""
        with self.write_lock:
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
                self.cache[key] = value
```

---

## Real-World Example: API Response Cache

```python
import threading
import time
import hashlib
import json
from collections import OrderedDict

class APIResponseCache:
    """
    Cache for API responses with automatic key generation
    """
    
    def __init__(self, capacity: int = 100, ttl: float = 300.0):
        """
        Args:
            capacity: Maximum cached responses
            ttl: Time to live in seconds (default 5 minutes)
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()  # key -> (response, expiry_time)
        self.lock = threading.RLock()
    
    def _generate_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and parameters"""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        Get cached response
        
        Returns:
            Cached response if exists and valid, None otherwise
        """
        key = self._generate_key(endpoint, params)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            response, expiry_time = self.cache[key]
            
            if time.time() > expiry_time:
                del self.cache[key]
                return None
            
            self.cache.move_to_end(key)
            return response
    
    def set(self, endpoint: str, params: dict, response: dict):
        """Cache API response"""
        key = self._generate_key(endpoint, params)
        expiry_time = time.time() + self.ttl
        
        with self.lock:
            if key in self.cache:
                self.cache[key] = (response, expiry_time)
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
                self.cache[key] = (response, expiry_time)
    
    def invalidate(self, endpoint: str, params: dict = None):
        """Invalidate cache entry"""
        if params:
            key = self._generate_key(endpoint, params)
            with self.lock:
                self.cache.pop(key, None)
        else:
            # Invalidate all entries for endpoint
            with self.lock:
                keys_to_remove = [
                    k for k in self.cache.keys()
                    if k.startswith(self._generate_key(endpoint, {}).split(':')[0])
                ]
                for k in keys_to_remove:
                    self.cache.pop(k, None)


# Usage
cache = APIResponseCache(capacity=50, ttl=300.0)

# Cache API response
response = {"data": [1, 2, 3], "status": "success"}
cache.set("/api/users", {"page": 1}, response)

# Retrieve from cache
cached = cache.get("/api/users", {"page": 1})
print(cached)  # {"data": [1, 2, 3], "status": "success"}
```

---

## Design Considerations

1. **Lock Granularity**: Fine-grained locks for better concurrency
2. **Eviction Policy**: LRU, LFU, or TTL-based
3. **Memory Management**: Bounded size prevents memory issues
4. **Expiration**: TTL support for time-sensitive data
5. **Cache Invalidation**: Manual or automatic invalidation

---

## Key Takeaways

- LRU cache evicts least recently used items
- TTL support enables time-based expiration
- Read-write locks optimize for read-heavy workloads
- Thread safety is critical for concurrent access
- OrderedDict efficiently maintains insertion order for LRU

