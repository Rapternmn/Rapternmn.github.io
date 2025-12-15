+++
title = "Token Bucket Filtering"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Token Bucket Filtering implementation: Request filtering, traffic shaping, bandwidth control, and rate-based filtering with thread-safe operations."
+++

# 🪣 Token Bucket Filtering

## Problem Statement

Design a token bucket filter that:
- Filters requests based on token availability
- Shapes traffic flow
- Controls bandwidth usage
- Works in multi-threaded environment
- Supports different filtering policies

**Use Cases**:
- Network traffic shaping
- Request filtering
- Bandwidth throttling
- Resource access control
- API request filtering

---

## Token Bucket vs Rate Limiter

**Rate Limiter**: Limits requests per time window (see [Rate Limiter Design]({{< ref "4-Rate_Limiter_Design.md" >}}))
**Token Bucket Filter**: Controls flow rate and allows bursts for filtering/traffic shaping

### Key Differences:
- **Rate Limiter**: Hard limit (e.g., 100 requests/minute) - used for API rate limiting
- **Token Bucket Filter**: Soft limit with burst capacity (e.g., 10 tokens/sec, burst of 50) - used for traffic shaping and bandwidth control
- **Use Case**: Rate Limiter rejects requests, Token Bucket Filter shapes/delays traffic flow

---

## Basic Token Bucket Filter

```python
import threading
import time
from typing import Optional

class TokenBucketFilter:
    """
    Token bucket filter for request filtering
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self, now: float):
        """Refill tokens based on elapsed time"""
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, tokens_needed: int = 1) -> bool:
        """
        Check if request should be allowed
        
        Args:
            tokens_needed: Number of tokens required
        
        Returns:
            True if allowed, False if filtered
        """
        with self.lock:
            now = time.time()
            self._refill(now)
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False
    
    def get_available_tokens(self) -> int:
        """Get current available tokens"""
        with self.lock:
            now = time.time()
            self._refill(now)
            return int(self.tokens)


# Usage
filter = TokenBucketFilter(rate=5.0, capacity=10)  # 5 tokens/sec, burst of 10

def make_request(request_id: int):
    if filter.allow_request():
        print(f"Request {request_id} allowed")
        return True
    else:
        print(f"Request {request_id} filtered (rate limit)")
        return False

# Simulate requests
for i in range(20):
    make_request(i)
    time.sleep(0.1)
```

---

## Multi-Level Token Bucket Filter

```python
import threading
import time

class MultiLevelTokenBucket:
    """
    Token bucket with multiple levels (e.g., per-user and global)
    """
    
    def __init__(self, global_rate: float, global_capacity: int):
        self.global_bucket = TokenBucketFilter(global_rate, global_capacity)
        self.user_buckets = {}
        self.user_lock = threading.Lock()
        self.user_rate = global_rate * 0.5  # Per-user rate
        self.user_capacity = global_capacity // 2
    
    def allow_request(self, user_id: str, tokens_needed: int = 1) -> bool:
        """
        Check both global and per-user limits
        
        Returns:
            True if both limits allow, False otherwise
        """
        # Check global limit
        if not self.global_bucket.allow_request(tokens_needed):
            return False
        
        # Check per-user limit
        with self.user_lock:
            if user_id not in self.user_buckets:
                self.user_buckets[user_id] = TokenBucketFilter(
                    self.user_rate, self.user_capacity
                )
        
        return self.user_buckets[user_id].allow_request(tokens_needed)


# Usage
filter = MultiLevelTokenBucket(global_rate=100.0, global_capacity=200)

def user_request(user_id: str, request_id: int):
    if filter.allow_request(user_id):
        print(f"User {user_id} request {request_id} allowed")
    else:
        print(f"User {user_id} request {request_id} filtered")
```

---

## Token Bucket with Priority Filtering

```python
import threading
import time
from enum import Enum

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class PriorityTokenBucketFilter:
    """
    Token bucket that prioritizes requests
    """
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Priority multipliers (higher priority needs fewer tokens)
        self.priority_multipliers = {
            Priority.HIGH: 0.5,    # High priority uses 0.5 tokens
            Priority.MEDIUM: 1.0,  # Medium uses 1 token
            Priority.LOW: 2.0      # Low uses 2 tokens
        }
    
    def _refill(self, now: float):
        """Refill tokens"""
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, priority: Priority = Priority.MEDIUM) -> bool:
        """
        Check if request allowed based on priority
        
        Returns:
            True if allowed, False if filtered
        """
        with self.lock:
            now = time.time()
            self._refill(now)
            
            tokens_needed = self.priority_multipliers[priority]
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False


# Usage
filter = PriorityTokenBucketFilter(rate=10.0, capacity=20)

# High priority requests more likely to be allowed
for i in range(10):
    if filter.allow_request(Priority.HIGH):
        print(f"High priority request {i} allowed")
    if filter.allow_request(Priority.LOW):
        print(f"Low priority request {i} allowed")
```

---

## Token Bucket for Bandwidth Control

```python
import threading
import time

class BandwidthController:
    """
    Token bucket for bandwidth throttling
    """
    
    def __init__(self, bandwidth_bps: float):
        """
        Args:
            bandwidth_bps: Bandwidth in bytes per second
        """
        self.bandwidth_bps = bandwidth_bps
        self.tokens = bandwidth_bps  # Start with full capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self, now: float):
        """Refill tokens (bytes)"""
        elapsed = now - self.last_refill
        bytes_to_add = elapsed * self.bandwidth_bps
        self.tokens = min(self.bandwidth_bps, self.tokens + bytes_to_add)
        self.last_refill = now
    
    def consume_bandwidth(self, bytes_needed: int) -> bool:
        """
        Try to consume bandwidth
        
        Returns:
            True if bandwidth available, False if throttled
        """
        with self.lock:
            now = time.time()
            self._refill(now)
            
            if self.tokens >= bytes_needed:
                self.tokens -= bytes_needed
                return True
            return False
    
    def wait_for_bandwidth(self, bytes_needed: int):
        """Wait until bandwidth is available"""
        while not self.consume_bandwidth(bytes_needed):
            time.sleep(0.01)  # Small sleep to avoid busy waiting


# Usage
controller = BandwidthController(bandwidth_bps=1024 * 1024)  # 1 MB/s

def transfer_data(data_size: int):
    if controller.consume_bandwidth(data_size):
        print(f"Transferred {data_size} bytes")
    else:
        print(f"Bandwidth throttled for {data_size} bytes")
        controller.wait_for_bandwidth(data_size)
        print(f"Transferred {data_size} bytes after wait")
```

---

## Key Takeaways

- Token bucket allows bursts up to capacity
- Filters requests based on token availability
- Supports multi-level and priority filtering
- Useful for traffic shaping and bandwidth control
- Different from rate limiter (soft vs hard limits)

