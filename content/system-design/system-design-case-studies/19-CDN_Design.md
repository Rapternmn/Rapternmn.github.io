+++
title = "Content Delivery Network (CDN Design)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 19
description = "Design a Content Delivery Network (CDN). Covers edge servers, cache hierarchy, origin servers, global distribution, and scaling to serve content to billions of users."
+++

---

## Problem Statement

Design a Content Delivery Network (CDN) that caches and delivers content from edge locations close to users. The system should reduce latency, offload origin servers, and scale to serve content globally.

**Examples**: Cloudflare, AWS CloudFront, Fastly, Akamai

---

## Requirements Clarification

### Functional Requirements

1. **Content Caching**: Cache content at edge locations
2. **Content Delivery**: Deliver content from edge servers
3. **Cache Invalidation**: Invalidate cached content
4. **Load Balancing**: Distribute requests across edge servers
5. **SSL/TLS**: Terminate SSL/TLS at edge
6. **DDoS Protection**: Protect against DDoS attacks

### Non-Functional Requirements

- **Scale**: 
  - 1B requests/day
  - 10M requests/hour peak
  - 1000 edge locations
  - Average 1 MB per request
- **Latency**: < 50ms for cached content
- **Availability**: 99.99% uptime
- **Cache Hit Rate**: > 90%

---

## Capacity Estimation

### Traffic Estimates

- **Requests**: 1B requests/day = ~11.6K requests/second
- **Peak Requests**: 10M/hour = ~2,800 requests/second
- **Bandwidth**: 2,800 requests/sec × 1 MB = 2.8 GB/sec
- **Daily Bandwidth**: 1B × 1 MB = 1 PB/day

### Storage Estimates

- **Cache per Edge**: 100 GB per edge location
- **Total Cache**: 100 GB × 1000 = 100 TB
- **Origin Storage**: 10 PB (source of truth)

---

## API Design

### CDN APIs

```
GET /cdn/{path}
Headers: {
  "Host": "cdn.example.com",
  "Accept-Encoding": "gzip"
}
Response: Content (cached or from origin)

POST /api/v1/cdn/invalidate
Request: {
  "paths": ["/images/logo.png", "/css/style.css"],
  "pattern": "*.jpg"  // optional
}
Response: {
  "invalidated": 100,
  "status": "success"
}

GET /api/v1/cdn/stats
Query: ?edgeLocation=us-east-1&timeRange=1h
Response: {
  "requests": 1000000,
  "cacheHitRate": 0.95,
  "bandwidth": 1000  // GB
}
```

---

## Database Design

### CDN Metadata

**Edge Locations Table** (PostgreSQL):
```
locationId (PK): UUID
region: VARCHAR
city: VARCHAR
ipAddress: VARCHAR
capacity: BIGINT
status: VARCHAR (active, maintenance)
```

**Cache Metadata** (Distributed Cache - Redis):
```
path (PK): VARCHAR
edgeLocation: VARCHAR
cachedAt: TIMESTAMP
expiresAt: TIMESTAMP
hitCount: INT
size: BIGINT
```

**Origin Servers Table** (PostgreSQL):
```
originId (PK): UUID
hostname: VARCHAR
ipAddress: VARCHAR
region: VARCHAR
status: VARCHAR (active, down)
```

---

## High-Level Design

### Architecture

```
User → [DNS] → [Edge Server] → [Cache]
                            ↓ (cache miss)
                    [Origin Server]
                            ↓
                    [Content Store]
```

### Components

1. **DNS**: Route users to nearest edge server
2. **Edge Servers**: Cache and serve content
3. **Origin Servers**: Source of truth for content
4. **Cache Hierarchy**: Multi-level cache (L1, L2, L3)
5. **Load Balancer**: Distribute requests across edge servers
6. **Origin Shield**: Cache between edge and origin

---

## Detailed Design

### Edge Server Architecture

**Edge Server Components**:
1. **HTTP Server**: Serve HTTP/HTTPS requests
2. **Cache**: In-memory cache (Redis) or disk cache
3. **Load Balancer**: Distribute requests internally
4. **SSL/TLS Termination**: Terminate SSL at edge
5. **Compression**: Compress content (gzip, brotli)

**Cache Storage**:
- **L1 Cache**: In-memory cache (fastest, smallest)
- **L2 Cache**: SSD cache (fast, larger)
- **L3 Cache**: HDD cache (slower, largest)

---

### Request Flow

**Cache Hit Flow**:
1. **User Request**: User requests content
3. **Anycast DNS**: Routes request to nearest edge IP
3. **Edge Server**: Check cache
4. **Cache Hit**: Return cached content
5. **Response**: Return content to user

**Cache Miss Flow**:
1. **User Request**: User requests content
2. **DNS Lookup**: DNS routes to nearest edge server
3. **Edge Server**: Check cache (miss)
4. **Origin Request**: Request from origin server
5. **Origin Response**: Receive content from origin
6. **Cache Content**: Cache content at edge
7. **Response**: Return content to user

---

### Cache Invalidation

**Invalidation Strategies**:

1. **TTL-Based**:
   - **TTL**: Set time-to-live for cached content
   - **Expiration**: Content expires after TTL
   - **Pros**: Simple, automatic
   - **Cons**: May serve stale content

2. **Purge-Based**:
   - **Manual Purge**: Manually invalidate content
   - **API**: API to purge specific paths/patterns
   - **Pros**: Immediate invalidation
   - **Cons**: Manual, requires API calls

3. **Version-Based**:
   - **Versioning**: Version content URLs
   - **Example**: `/images/logo-v2.png`
   - **Pros**: No invalidation needed
   - **Cons**: URL management

**Recommendation**: **TTL + Purge** (combination)

---

### Cache Hierarchy

**Multi-Level Cache**:

1. **L1 Cache (Edge)**:
   - **Location**: Edge servers
   - **Storage**: In-memory (Redis)
   - **Size**: Small (hot content)
   - **Latency**: Lowest

2. **L2 Cache (Regional)**:
   - **Location**: Regional data centers
   - **Storage**: SSD
   - **Size**: Medium
   - **Latency**: Low

3. **L3 Cache (Origin Shield)**:
   - **Location**: Between edge and origin
   - **Storage**: HDD
   - **Size**: Large
   - **Latency**: Medium

**Cache Flow**:
```
Edge (L1) → Regional (L2) → Origin Shield (L3) → Origin
```

---

### Origin Server Selection

**Selection Strategies**:

1. **Geographic Proximity**:
   - **Nearest Origin**: Select nearest origin server
   - **Latency**: Minimize latency
   - **Use Case**: Static content

2. **Load-Based**:
   - **Least Loaded**: Select least loaded origin
   - **Load Balancing**: Distribute load
   - **Use Case**: Dynamic content

3. **Health-Based**:
   - **Healthy Origins**: Select healthy origins only
   - **Health Checks**: Monitor origin health
   - **Failover**: Failover to backup origins

---

### SSL/TLS Termination

**SSL/TLS at Edge**:
- **Termination**: Terminate SSL at edge servers
- **Benefits**: Reduce origin load, faster handshake
- **Certificates**: Manage certificates at edge
- **Protocols**: Support TLS 1.2, 1.3

**Origin Communication**:
- **HTTP**: Communicate with origin via HTTP (internal)
- **HTTPS**: Or HTTPS (if needed)

---

## Scalability

### Horizontal Scaling

- **Edge Servers**: Add edge servers in new locations
- **Origin Servers**: Scale origin servers horizontally
- **Load Balancing**: Distribute requests across servers

### Performance Optimization

- **Caching**: Cache content at edge
- **Compression**: Compress content (gzip, brotli)
- **HTTP/2**: Use HTTP/2 for multiplexing
- **Pre-fetching**: Pre-fetch popular content

---

## Reliability

### High Availability

- **Multiple Edge Servers**: No single point of failure
- **Origin Redundancy**: Multiple origin servers
- **Health Checks**: Monitor server health

### Fault Tolerance

- **Edge Failures**: Route to other edge servers
- **Origin Failures**: Failover to backup origins
- **Cache Failures**: Fallback to origin

---

## Trade-offs

### Cache Size vs Hit Rate

- **Larger Cache**: Higher hit rate, higher cost
- **Smaller Cache**: Lower hit rate, lower cost

### Latency vs Cost

- **More Edge Locations**: Lower latency, higher cost
- **Fewer Edge Locations**: Higher latency, lower cost

### Consistency vs Performance

- **Long TTL**: Better performance, stale content
- **Short TTL**: Fresher content, more origin load

---

## Extensions

### Additional Features

1. **Dynamic Content**: Cache dynamic content
2. **Video Streaming**: Optimize for video streaming
3. **Image Optimization**: Optimize images (resize, format)
4. **Web Application Firewall**: WAF at edge
5. **Bot Protection**: Protect against bots
6. **Analytics**: Track CDN performance
7. **Multi-CDN**: Use multiple CDN providers
8. **Geo-Blocking**: Restrict access by region

---

## Key Takeaways

- **Edge Caching**: Cache content at edge locations
- **Cache Hierarchy**: Multi-level cache for efficiency
- **Cache Invalidation**: TTL + purge for freshness
- **Origin Shield**: Reduce origin load
- **SSL/TLS Termination**: Terminate SSL at edge
- **Scalability**: Add edge locations for global coverage

---

## Related Topics

- **[CDN & Content Delivery]({{< ref "../system-components/7-CDN_Content_Delivery.md" >}})** - CDN components and strategies
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Cache patterns
- **[Load Balancing]({{< ref "../system-components/2-Load_Balancing.md" >}})** - Request distribution
- **[Video Streaming]({{< ref "7-Video_Streaming.md" >}})** - CDN for video delivery

