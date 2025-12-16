+++
title = "Rate Limiter"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 3
description = "Design a rate limiter to control API request rates. Covers token bucket, sliding window, fixed window algorithms, and distributed rate limiting."
+++

---

## Problem Statement

Design a rate limiter that restricts the number of requests a client can make within a time window. The system should prevent abuse, ensure fair usage, and protect backend services from overload.

**Use Cases**: API rate limiting, DDoS protection, fair resource allocation

---

## Requirements Clarification

### Functional Requirements

1. **Rate Limiting**: Limit requests per time window
2. **Multiple Limits**: Different limits for different endpoints/users
3. **Burst Handling**: Allow burst traffic within limits
4. **Distributed**: Work across multiple servers
5. **Configurable**: Adjust limits dynamically

### Non-Functional Requirements

- **Scale**: 
  - 1M requests/second
  - 10M unique clients
  - Multiple rate limit rules
- **Latency**: < 10ms overhead per request
- **Accuracy**: Precise rate limiting
- **Availability**: 99.99% (critical component)

---

## Capacity Estimation

### Traffic Estimates

- **Requests**: 1M requests/second
- **Unique Clients**: 10M
- **Rate Limit Checks**: 1M/second (every request)
- **Storage**: 10M client records

### Storage Estimates

- **Per Client Record**: ~100 bytes (client ID, counters, timestamps)
- **Total Storage**: 10M × 100 bytes = 1 GB
- **With Replication**: 3-5 GB

---

## API Design

### Rate Limiter API

```
POST /api/v1/rate-limit/check
Headers: {
  "X-Client-ID": "user123",
  "X-Endpoint": "/api/users"
}
Response: {
  "allowed": true,
  "remaining": 95,
  "resetAt": "2025-12-15T10:01:00Z"
}
// OR
Response: {
  "allowed": false,
  "retryAfter": 5  // seconds
}
```

### Configuration API

```
POST /api/v1/rate-limit/config
Request: {
  "clientId": "user123",
  "endpoint": "/api/users",
  "limit": 100,
  "window": 60  // seconds
}
```

---

## Rate Limiting Algorithms

### 1. Fixed Window Counter

**How it works**:
- Divide time into fixed windows (e.g., 1 minute)
- Count requests in current window
- Reset counter at window boundary

**Example**: 100 requests/minute
- Window 1 (10:00-10:01): Count requests
- Window 2 (10:01-10:02): Reset counter

**Pros**:
- Simple to implement
- Memory efficient
- Easy to understand

**Cons**:
- Burst at window boundary
- Not smooth distribution

**Use Cases**: Simple rate limiting, when bursts are acceptable

---

### 2. Sliding Window Log

**How it works**:
- Store timestamp of each request
- Count requests in sliding window
- Remove old timestamps outside window

**Example**: 100 requests/minute
- Keep timestamps of last 100 requests
- Count requests in last 60 seconds

**Pros**:
- Accurate
- Smooth distribution
- No burst issues

**Cons**:
- Memory intensive (stores all timestamps)
- Slower (need to check/remove old entries)

**Use Cases**: When accuracy is critical, low traffic

---

### 3. Sliding Window Counter

**How it works**:
- Maintain multiple fixed windows
- Weighted average of overlapping windows
- Approximates sliding window

**Example**: 
- Window 1 (10:00-10:01): 50 requests
- Window 2 (10:01-10:02): 60 requests
- Current time: 10:01:30
- Count = 50 × 0.5 + 60 × 0.5 = 55 requests

**Pros**:
- More accurate than fixed window
- Memory efficient
- Good approximation

**Cons**:
- Slightly complex
- Approximation (not exact)

**Use Cases**: Balance between accuracy and efficiency

---

### 4. Token Bucket

**How it works**:
- Bucket with tokens (capacity)
- Tokens added at fixed rate
- Request consumes token
- Request rejected if no tokens

**Example**: 
- Capacity: 100 tokens
- Refill rate: 100 tokens/minute
- Request consumes 1 token

**Pros**:
- Allows bursts (up to capacity)
- Smooth rate limiting
- Flexible

**Cons**:
- More complex
- Need to track refill rate

**Use Cases**: When bursts are needed, flexible rate limiting

---

### 5. Leaky Bucket

**How it works**:
- Bucket with fixed capacity
- Requests added to bucket
- Process requests at fixed rate
- Reject if bucket full

**Example**:
- Capacity: 100 requests
- Process rate: 100 requests/minute
- Requests added as they arrive

**Pros**:
- Smooth output rate
- No bursts
- Predictable

**Cons**:
- Doesn't allow bursts
- May delay requests

**Use Cases**: When smooth, constant rate needed

---

## High-Level Design

### Architecture

![Rate Limiter Architecture](/images/system-design/rate-limiter-architecture.png)

### Components

1. **Rate Limiter Service**: Core rate limiting logic
2. **Storage (Redis)**: Store rate limit counters
3. **Configuration Service**: Manage rate limit rules
4. **API Gateway Integration**: Intercept requests

---

## Detailed Design

### Distributed Rate Limiting

**Challenge**: Multiple rate limiter instances need shared state

**Solution**: Use Redis for shared state

**Approach**:
1. Each request → Check Redis
2. Update counter in Redis
3. Return allow/deny decision

**Redis Operations**:
- **INCR**: Increment counter
- **EXPIRE**: Set expiration
- **Atomic Operations**: Ensure consistency

---

### Redis Implementation (Sliding Window Counter)

```python
def check_rate_limit(client_id, endpoint, limit, window):
    key = f"rate_limit:{client_id}:{endpoint}"
    current = redis.incr(key)
    
    if current == 1:
        redis.expire(key, window)
    
    if current > limit:
        return False, limit - current
    
    return True, limit - current
```

---

### Rate Limit Rules

**Rule Types**:
- **Per User**: Limit per user ID
- **Per IP**: Limit per IP address
- **Per Endpoint**: Different limits per API endpoint
- **Per API Key**: Limit per API key
- **Global**: System-wide limits

**Rule Priority**:
1. User-specific rules (highest priority)
2. Endpoint-specific rules
3. Default rules (lowest priority)

---

## Scalability

### Horizontal Scaling

- **Stateless Rate Limiter**: Can scale horizontally
- **Redis Cluster**: Distribute rate limit data
- **Load Balancing**: Distribute rate limit checks

### Performance Optimization

- **Redis Caching**: Fast lookups
- **Local Caching**: Cache rules locally
- **Batch Updates**: Batch counter updates

---

## Reliability

### High Availability

- **Redis Cluster**: Replication and failover
- **Multiple Rate Limiter Instances**: No single point of failure
- **Graceful Degradation**: Allow requests if Redis down (configurable)

### Data Consistency

- **Redis Atomic Operations**: Ensure counter accuracy
- **Eventual Consistency**: Acceptable for rate limiting
- **Race Conditions**: Redis handles atomically

---

## Trade-offs

### Accuracy vs Performance

- **Sliding Window Log**: Most accurate, slower
- **Fixed Window**: Less accurate, faster
- **Sliding Window Counter**: Balance

### Memory vs Accuracy

- **Fixed Window**: Low memory, less accurate
- **Sliding Window Log**: High memory, accurate
- **Token Bucket**: Medium memory, flexible

---

## Use Cases

### 1. API Rate Limiting

Limit API requests per user/key.

**Example**: 1000 requests/hour per API key

---

### 2. DDoS Protection

Protect against distributed attacks.

**Example**: 100 requests/minute per IP

---

### 3. Fair Resource Allocation

Ensure fair usage of resources.

**Example**: 10 concurrent requests per user

---

## Best Practices

1. **Choose Right Algorithm**: Based on requirements
2. **Distributed Storage**: Use Redis for shared state
3. **Graceful Degradation**: Handle Redis failures
4. **Monitoring**: Track rate limit hits
5. **Configuration**: Make limits configurable
6. **Headers**: Return rate limit info in headers

---

## Key Takeaways

- **Multiple Algorithms**: Choose based on requirements
- **Token Bucket**: Best for allowing bursts
- **Sliding Window**: Best for accuracy
- **Fixed Window**: Simplest, good for many cases
- **Distributed**: Use Redis for shared state
- **Performance**: Redis provides fast lookups

---

## Related Topics

- **[System Components - Rate Limiter]({{< ref "../system-components/4-API_Gateway.md" >}})** - Rate limiting in API Gateway
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Redis caching
- **[Concurrency - Rate Limiter Design]({{< ref "../../coding/concurrency/4-Rate_Limiter_Design.md" >}})** - Implementation details

