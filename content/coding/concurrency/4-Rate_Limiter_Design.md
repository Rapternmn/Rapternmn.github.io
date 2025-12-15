+++
title = "Rate Limiter Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Thread-safe Rate Limiter implementation: Token Bucket, Sliding Window, Fixed Window algorithms with Python implementation for API rate limiting."
+++

# 🚦 Rate Limiter Design

## Problem Statement

Design a thread-safe rate limiter that:
- Limits the number of requests per time window
- Works in a multi-threaded environment
- Supports different rate limiting algorithms
- Is efficient and scalable

**Use Cases**:
- API rate limiting
- DDoS protection
- Resource throttling
- Cost control

---

## Requirements

1. **Thread Safety**: Must work correctly with multiple concurrent requests
2. **Accuracy**: Should accurately enforce rate limits
3. **Performance**: Low overhead, fast decision making
4. **Flexibility**: Support different algorithms (Token Bucket, Sliding Window, etc.)

---

## Algorithm Options

### 1. Token Bucket Algorithm
- Maintains a bucket of tokens
- Tokens are added at a fixed rate
- Request consumes a token
- Request allowed if token available

### 2. Sliding Window Log
- Tracks timestamps of requests
- Removes old entries outside window
- Allows request if count < limit

### 3. Fixed Window Counter
- Divides time into fixed windows
- Counts requests in current window
- Resets at window boundary

---

## Implementation: Token Bucket

```python
import threading
import time
from collections import defaultdict

class TokenBucketRateLimiter:
    """
    Thread-safe Token Bucket Rate Limiter
    
    Allows requests at a fixed rate with burst capacity.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = defaultdict(lambda: capacity)  # Per-key token count
        self.last_update = defaultdict(time.time)  # Per-key last update time
        self.lock = threading.RLock()  # Per-key locking would be better, but RLock for simplicity
    
    def _refill_tokens(self, key: str, now: float):
        """Refill tokens based on elapsed time"""
        elapsed = now - self.last_update[key]
        tokens_to_add = elapsed * self.rate
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + tokens_to_add
        )
        self.last_update[key] = now
    
    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if request is allowed
        
        Args:
            key: Identifier for rate limiting (user ID, IP, etc.)
        
        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            self._refill_tokens(key, now)
            
            if self.tokens[key] >= 1:
                self.tokens[key] -= 1
                return True
            return False
    
    def get_remaining_tokens(self, key: str = "default") -> int:
        """Get remaining tokens for a key"""
        with self.lock:
            now = time.time()
            self._refill_tokens(key, now)
            return int(self.tokens[key])


# Usage Example
limiter = TokenBucketRateLimiter(rate=5.0, capacity=10)  # 5 req/sec, burst of 10

def make_request(user_id: str):
    if limiter.is_allowed(user_id):
        print(f"Request allowed for {user_id}")
        return True
    else:
        print(f"Rate limit exceeded for {user_id}")
        return False

# Multi-threaded usage
import threading

def worker(user_id):
    for i in range(5):
        make_request(user_id)
        time.sleep(0.1)

threads = []
for user_id in ["user1", "user2", "user3"]:
    t = threading.Thread(target=worker, args=(user_id,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

---

## Implementation: Sliding Window Log

```python
import threading
import time
from collections import defaultdict, deque

class SlidingWindowRateLimiter:
    """
    Thread-safe Sliding Window Log Rate Limiter
    
    Tracks request timestamps in a sliding window.
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)  # Per-key request timestamps
        self.lock = threading.RLock()
    
    def _clean_old_requests(self, key: str, now: float):
        """Remove requests outside the window"""
        window_start = now - self.window_seconds
        while (self.requests[key] and 
               self.requests[key][0] < window_start):
            self.requests[key].popleft()
    
    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if request is allowed
        
        Args:
            key: Identifier for rate limiting
        
        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            self._clean_old_requests(key, now)
            
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return True
            return False
    
    def get_remaining_requests(self, key: str = "default") -> int:
        """Get remaining requests in current window"""
        with self.lock:
            now = time.time()
            self._clean_old_requests(key, now)
            return max(0, self.max_requests - len(self.requests[key]))


# Usage
limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60.0)

def api_call(user_id: str):
    if limiter.is_allowed(user_id):
        # Process request
        return {"status": "success"}
    else:
        return {"status": "rate_limited", "retry_after": 60}
```

---

## Implementation: Fixed Window Counter

```python
import threading
import time
from collections import defaultdict

class FixedWindowRateLimiter:
    """
    Thread-safe Fixed Window Counter Rate Limiter
    
    Simple but may allow bursts at window boundaries.
    """
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.counts = defaultdict(int)  # Per-key request count
        self.window_start = defaultdict(time.time)  # Per-key window start
        self.lock = threading.RLock()
    
    def _reset_window_if_needed(self, key: str, now: float):
        """Reset counter if window has passed"""
        if now - self.window_start[key] >= self.window_seconds:
            self.counts[key] = 0
            self.window_start[key] = now
    
    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if request is allowed
        
        Args:
            key: Identifier for rate limiting
        
        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            self._reset_window_if_needed(key, now)
            
            if self.counts[key] < self.max_requests:
                self.counts[key] += 1
                return True
            return False


# Usage
limiter = FixedWindowRateLimiter(max_requests=100, window_seconds=60.0)
```

---

## Comparison of Algorithms

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Token Bucket** | Smooth rate, burst support | More complex | General purpose |
| **Sliding Window** | Accurate, no boundary bursts | Memory overhead | High accuracy needed |
| **Fixed Window** | Simple, memory efficient | Boundary bursts | Simple requirements |

---

## Distributed Rate Limiting

For distributed systems, use Redis or similar:

```python
import redis
import time

class DistributedRateLimiter:
    def __init__(self, redis_client, rate: float, capacity: int):
        self.redis = redis_client
        self.rate = rate
        self.capacity = capacity
    
    def is_allowed(self, key: str) -> bool:
        """
        Uses Redis Lua script for atomic operations
        """
        script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local elapsed = now - last_update
        tokens = math.min(capacity, tokens + elapsed * rate)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, math.ceil(capacity / rate))
            return 1
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, math.ceil(capacity / rate))
            return 0
        end
        """
        
        now = time.time()
        result = self.redis.eval(
            script, 1, key, self.rate, self.capacity, now
        )
        return bool(result)
```

---

## Design Considerations

1. **Per-Key vs Global**: Rate limit per user/IP or globally?
2. **Memory Management**: Clean up old entries (sliding window)
3. **Precision vs Performance**: Sliding window is more accurate but uses more memory
4. **Distributed Systems**: Use Redis or similar for shared state
5. **Configuration**: Make rate limits configurable

---

## Key Takeaways

- Token Bucket: Smooth rate with burst capacity
- Sliding Window: Most accurate, tracks timestamps
- Fixed Window: Simple but allows boundary bursts
- Thread safety is critical for concurrent requests
- For distributed systems, use Redis or similar shared storage

