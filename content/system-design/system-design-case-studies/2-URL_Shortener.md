+++
title = "URL Shortener (TinyURL)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 2
description = "Design a URL shortener service like TinyURL or bit.ly. Covers hash functions, database design, caching strategies, and scaling to handle billions of URLs."
+++

---

## Problem Statement

Design a URL shortener service that converts long URLs into short, shareable links. The system should handle millions of URL shortening requests per day and redirect users efficiently.

**Examples**: TinyURL, bit.ly, goo.gl

---

## Requirements Clarification

### Functional Requirements

1. **URL Shortening**: Convert long URL to short URL
2. **URL Redirection**: Redirect short URL to original URL
3. **Custom URLs**: Optional custom short codes
4. **URL Expiration**: Optional expiration time
5. **Analytics**: Track click counts and analytics

### Non-Functional Requirements

- **Scale**: 
  - 100M URLs created per day
  - 100:1 read/write ratio (10B reads/day)
  - 100K writes/second, 10M reads/second
- **Latency**: < 200ms for URL resolution
- **Availability**: 99.9% uptime
- **Durability**: URLs should not be lost

---

## Capacity Estimation

### Traffic Estimates

- **Writes**: 100M URLs/day = ~1,160 URLs/second
- **Peak writes**: 3x = ~3,500 URLs/second
- **Reads**: 10B reads/day = ~116K reads/second
- **Peak reads**: 3x = ~350K reads/second

### Storage Estimates

- **URL length**: Average 100 characters
- **Short code**: 7 characters (base62: 62^7 ≈ 3.5 trillion)
- **Metadata**: ~500 bytes per URL
- **Total per URL**: ~600 bytes
- **Daily storage**: 100M × 600 bytes = 60 GB/day
- **Yearly storage**: 60 GB × 365 ≈ 22 TB/year
- **10 years**: ~220 TB

### Bandwidth Estimates

- **Write**: 3,500 URLs/sec × 600 bytes = 2.1 MB/sec
- **Read**: 350K reads/sec × 600 bytes = 210 MB/sec

---

## API Design

### REST APIs

```
POST /api/v1/shorten
Request: {
  "longUrl": "https://example.com/very/long/url",
  "customCode": "optional",
  "expiresIn": 365  // days
}
Response: {
  "shortUrl": "https://short.ly/abc123",
  "expiresAt": "2026-12-15T10:00:00Z"
}

GET /api/v1/{shortCode}
Response: 302 Redirect to longUrl

GET /api/v1/analytics/{shortCode}
Response: {
  "shortCode": "abc123",
  "longUrl": "https://example.com/...",
  "createdAt": "2025-12-15T10:00:00Z",
  "clickCount": 12345,
  "lastAccessed": "2025-12-15T15:30:00Z"
}

DELETE /api/v1/{shortCode}
Response: 200 OK
```

---

## Database Design

### Schema

**URLs Table** (PostgreSQL/Cassandra):
```
shortCode (PK): VARCHAR(7)
longUrl: TEXT
createdAt: TIMESTAMP
expiresAt: TIMESTAMP (nullable)
userId: UUID (nullable)
clickCount: BIGINT
```

**Indexes**:
- Primary key on `shortCode`
- Index on `userId` (for user's URLs)
- Index on `expiresAt` (for cleanup)

### Database Selection

**Option 1: SQL Database (PostgreSQL)**
- Pros: ACID, strong consistency, easy queries
- Cons: Scaling writes is harder
- Use for: Metadata, user data, analytics

**Option 2: NoSQL Database (Cassandra/DynamoDB)**
- Pros: High write throughput, horizontal scaling
- Cons: Eventual consistency
- Use for: URL mappings (can tolerate eventual consistency)

**Recommendation**: Hybrid approach
- **Cassandra/DynamoDB** for URL mappings (high write volume)
- **PostgreSQL** for user data and analytics (strong consistency needed)

---

## High-Level Design

### Architecture

![URL Shortener Architecture](/images/system-design/url-shortener-architecture.png)

### Components

1. **CDN**: Serve static assets, cache redirects (Edge locations)
2. **Load Balancer**: Distribute traffic
3. **API Gateway**: Authentication, rate limiting, request routing
4. **URL Service**: Core shortening and redirection logic
5. **Key Generation Service (KGS)**: Pre-generates unique keys and serves them to URL Service
6. **Cache (Redis)**: Cache hot URL mappings
7. **Database**: Store URL mappings (Cassandra/DynamoDB) and Metadata
8. **Message Queue**: Decouples redirection from analytics processing (Kafka/RabbitMQ)
9. **Analytics Service**: Consumes events and aggregates data
10. **Data Warehouse**: Stores historical analytics data (ClickHouse/Redshift)

---

## Detailed Design

### Short Code Generation

#### Option 1: Hash-based (MD5/SHA256)

**Approach**:
1. Hash long URL (MD5/SHA256)
2. Take first 7 characters
3. Check for collisions
4. If collision, append counter and re-hash

**Pros**: Deterministic
**Cons**: Collision checks are expensive; longer content to hash

#### Option 2: Key Generation Service (KGS) [Recommended]

**Approach**:
1. **Pre-generation**: A dedicated service generates random 7-character strings (base62) offline.
2. **Uniqueness**: Ensures keys are unique and stores them in a "Ready" table/database.
3. **Distribution**: Loads a batch of keys (e.g., 1000) into memory of each URL Service instance.
4. **Assignment**: When a request comes, URL Service picks the next available key from memory.

**Pros**:
- **O(1) Write Latency**: No collision checks or database reads during request.
- **Concurrency**: Simple to scale; each server has its own distinct batch of keys.

**Cons**:
- Management of KGS infrastructure.
- Potential for lost keys if a server crashes (acceptable loss).

**Recommendation**: **Key Generation Service (KGS)**. This is the industry standard for high-throughput scaling.

---

### URL Redirection Flow

1. **Client requests**: `GET /abc123`
2. **Check CDN cache**: If cached, return redirect (301/302)
3. **Check Redis cache**: If cached, return redirect
4. **Query database**: Lookup shortCode
5. **Cache result**: Update Redis and CDN
6. **Publish Event**: Send `{shortCode, timestamp, userAgent}` to Message Queue (Kafka)
7. **Return 302 redirect**: To `longUrl`
8. **Async Analytics**: Analytics Service consumes event from Queue and updates DB

---

### Caching Strategy

**Multi-layer caching**:

1. **CDN Cache** (Edge):
   - Cache redirects
   - TTL: 1 hour
   - Reduces origin load

2. **Redis Cache** (Application):
   - Cache URL mappings
   - TTL: 24 hours
   - LRU eviction
   - Cache hot URLs (80/20 rule)

**Cache-Aside Pattern**:
- Check cache first
- On miss, query database
- Store in cache for future requests

---

### Database Sharding

**Sharding Strategy**: Hash-based sharding

**Shard Key**: Hash of `shortCode`

**Benefits**:
- Even distribution
- No hot spots
- Easy to add shards

**Shard Lookup**:
- Hash shortCode → Determine shard
- Route request to appropriate shard

---

## Scalability

### Horizontal Scaling

- **Stateless URL Service**: Easy horizontal scaling
- **Load Balancer**: Distribute across instances
- **Database Sharding**: Distribute data across shards
- **Cache Partitioning**: Distribute cache across Redis nodes

### Read Scaling

- **Read Replicas**: Database read replicas for analytics
- **Caching**: Multi-layer caching reduces database load
- **CDN**: Edge caching for redirects

### Write Scaling

- **Database Sharding**: Distribute writes across shards
- **Async Processing**: Analytics processing async

---

## Reliability

### High Availability

- **Multiple URL Service Instances**: No single point of failure
- **Database Replication**: Master-slave replication
- **Cache Replication**: Redis cluster with replication
- **Health Checks**: Monitor service health

### Fault Tolerance

- **Cache Failures**: Fallback to database
- **Database Failures**: Read from replicas
- **Service Failures**: Load balancer routes to healthy instances

### Data Durability

- **Database Persistence**: All URLs persisted
- **Backup Strategy**: Regular backups
- **Replication**: Multiple copies of data

---

## Trade-offs

### Consistency vs Availability

- **URL Mappings**: Eventual consistency acceptable (AP)
- **User Data**: Strong consistency needed (CP)

### Latency vs Consistency

- **Reads**: Optimize for latency (caching)
- **Writes**: Can tolerate slight delay

### Storage vs Performance

- **Caching**: Trade storage for performance
- **CDN**: Trade cost for global performance

---

## Extensions

### Additional Features

1. **User Accounts**: Track user's URLs
2. **Custom Domains**: Allow custom domains
3. **Password Protection**: Password-protected URLs
4. **QR Codes**: Generate QR codes for URLs
5. **Bulk Import**: Import multiple URLs
6. **Link Preview**: Generate link previews
7. **Geographic Analytics**: Track clicks by location
8. **Referrer Tracking**: Track where clicks come from

---

## Key Takeaways

- **Short Code Generation**: Base62 encoding is simple and collision-free
- **Caching**: Multi-layer caching critical for read-heavy workload
- **Database**: Hybrid approach (NoSQL for mappings, SQL for metadata)
- **Sharding**: Hash-based sharding for even distribution
- **Scalability**: Stateless services enable horizontal scaling
- **Trade-offs**: Accept eventual consistency for URL mappings

---

## Related Topics

- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Multi-layer caching
- **[Load Balancing]({{< ref "../system-components/2-Load_Balancing.md" >}})** - Traffic distribution
- **[Databases]({{< ref "../databases/_index.md" >}})** - Database selection and sharding
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Scaling strategies

