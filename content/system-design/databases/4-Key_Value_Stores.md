+++
title = "Key-Value Stores"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Key-Value Stores: Redis and DynamoDB - in-memory caching, fast lookups, use cases, and when to use key-value databases."
+++

---

## Introduction

Key-value stores are simple databases that store data as key-value pairs. They provide fast lookups, high performance, and are ideal for caching, session storage, and simple data models.

---

## What are Key-Value Stores?

**Key-Value Store**:
- Simple data model: key → value
- Fast lookups by key
- No query language (usually)
- High performance
- Simple operations

**Key Characteristics**:
- **Simple Model**: Key-value pairs only
- **Fast Access**: O(1) lookups typically
- **High Performance**: Optimized for speed
- **Scalable**: Horizontal scaling support
- **Flexible Values**: Can store various data types

---

## Redis

### Overview

**Redis** (Remote Dictionary Server) is an in-memory key-value store with optional persistence.

**Key Features**:
- In-memory storage
- Rich data types
- Pub/sub messaging
- Persistence options
- High performance
- Atomic operations

### Data Types

**1. Strings**
```redis
SET user:1 "John Doe"
GET user:1
```

**2. Hashes**
```redis
HSET user:1 name "John" email "john@example.com"
HGET user:1 name
```

**3. Lists**
```redis
LPUSH tasks "task1"
RPOP tasks
```

**4. Sets**
```redis
SADD tags "python" "redis" "database"
SMEMBERS tags
```

**5. Sorted Sets**
```redis
ZADD leaderboard 100 "player1"
ZRANGE leaderboard 0 -1 WITHSCORES
```

**6. Bitmaps**
```redis
SETBIT active_users:2024-01-15 1000 1
GETBIT active_users:2024-01-15 1000
```

**7. HyperLogLog**
```redis
PFADD visitors:2024-01-15 "user1" "user2"
PFCOUNT visitors:2024-01-15
```

### Use Cases

**1. Caching**
- Cache database queries
- Reduce database load
- Fast data access
- TTL support

**2. Session Storage**
- Store user sessions
- Fast session retrieval
- Automatic expiration
- Distributed sessions

**3. Real-Time Analytics**
- Counters and metrics
- Leaderboards
- Rate limiting
- Real-time dashboards

**4. Message Queues**
- Pub/sub messaging
- Task queues
- Event streaming
- Inter-service communication

**5. Rate Limiting**
- API rate limiting
- Request throttling
- Sliding window counters
- Distributed rate limiting

### Architecture

**In-Memory Storage**:
- All data in RAM
- Extremely fast
- Limited by memory size
- Persistence optional

**Persistence Options**:
- **RDB**: Point-in-time snapshots
- **AOF**: Append-only file
- **Both**: Maximum durability

**Replication**:
- Master-slave replication
- Automatic failover (with Sentinel)
- Read scaling
- High availability

**Clustering**:
- Redis Cluster
- Horizontal scaling
- Data sharding
- Automatic failover

---

## DynamoDB

### Overview

**Amazon DynamoDB** is a managed NoSQL database service with key-value and document capabilities.

**Key Features**:
- Fully managed service
- Serverless scaling
- Single-digit millisecond latency
- Built-in security
- Global tables
- On-demand or provisioned capacity

### Data Model

**Tables, Items, and Attributes**:
- **Table**: Collection of items
- **Item**: Collection of attributes (like document)
- **Attributes**: Key-value pairs
- **Primary Key**: Partition key or partition + sort key

**Example Item**:
```json
{
  "user_id": "12345",           // Partition key
  "timestamp": "2024-01-15",    // Sort key
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "New York"
  }
}
```

### Key Features

**1. Primary Keys**
- **Simple Key**: Partition key only
- **Composite Key**: Partition + sort key
- Enables range queries with sort key

**2. Secondary Indexes**
- **Global Secondary Index (GSI)**: Different partition key
- **Local Secondary Index (LSI)**: Same partition key, different sort key
- Enable query flexibility

**3. On-Demand Scaling**
- Automatic scaling
- Pay per request
- No capacity planning
- Handle traffic spikes

**4. Provisioned Capacity**
- Predictable workloads
- Cost optimization
- Read/write capacity units
- Auto-scaling available

**5. Global Tables**
- Multi-region replication
- Active-active setup
- Low latency globally
- Disaster recovery

**6. Streams**
- Change data capture
- Real-time processing
- Event-driven architecture
- Lambda integration

### Use Cases

**1. Web Applications**
- User data storage
- Session management
- Product catalogs
- High-traffic applications

**2. Gaming Applications**
- Player data
- Leaderboards
- Game state
- Real-time updates

**3. IoT Applications**
- Device data
- Time-series data
- Sensor readings
- High write throughput

**4. Mobile Applications**
- User profiles
- Offline sync
- Backend for mobile
- Serverless architecture

### Operations

**Put Item**:
```javascript
await dynamodb.putItem({
  TableName: 'Users',
  Item: {
    user_id: { S: '12345' },
    name: { S: 'John Doe' },
    email: { S: 'john@example.com' }
  }
});
```

**Get Item**:
```javascript
await dynamodb.getItem({
  TableName: 'Users',
  Key: {
    user_id: { S: '12345' }
  }
});
```

**Query** (with sort key):
```javascript
await dynamodb.query({
  TableName: 'Orders',
  KeyConditionExpression: 'user_id = :uid AND order_date BETWEEN :start AND :end',
  ExpressionAttributeValues: {
    ':uid': { S: '12345' },
    ':start': { S: '2024-01-01' },
    ':end': { S: '2024-01-31' }
  }
});
```

---

## When to Use Key-Value Stores

### Good Fit

✅ **Simple Data Model**: Key-value lookups
✅ **High Performance**: Need fast access
✅ **Caching**: Reduce database load
✅ **Session Storage**: User sessions
✅ **Simple Queries**: Lookup by key
✅ **High Throughput**: Many simple operations

### Not Ideal For

❌ **Complex Queries**: No query language
❌ **Relationships**: No joins
❌ **Analytics**: Limited aggregation
❌ **Complex Data**: Better with document DB
❌ **Strong Consistency**: Eventual consistency (DynamoDB)

---

## Comparison: Redis vs DynamoDB

### Redis

**Strengths**:
- ✅ Extremely fast (in-memory)
- ✅ Rich data types
- ✅ Pub/sub messaging
- ✅ Flexible operations
- ✅ Open source

**Limitations**:
- ❌ Memory limited
- ❌ Requires management
- ❌ Single region (without clustering)
- ❌ Persistence optional

**Best For**:
- Caching
- Real-time features
- Session storage
- Message queues

### DynamoDB

**Strengths**:
- ✅ Fully managed
- ✅ Automatic scaling
- ✅ Global tables
- ✅ Built-in security
- ✅ Serverless

**Limitations**:
- ❌ Vendor lock-in (AWS)
- ❌ Cost at scale
- ❌ Less flexible than Redis
- ❌ Eventual consistency

**Best For**:
- AWS-based applications
- Managed service needed
- Global distribution
- Serverless architectures

---

## Best Practices

### Redis

**1. Memory Management**
- Set maxmemory policy
- Use appropriate eviction policy
- Monitor memory usage
- Use data structures efficiently

**2. Persistence**
- Choose RDB or AOF based on needs
- Configure persistence frequency
- Test recovery procedures
- Consider replication for durability

**3. Performance**
- Use pipelining for multiple operations
- Use appropriate data types
- Avoid large keys/values
- Monitor slow queries

### DynamoDB

**1. Key Design**
- Choose good partition key (even distribution)
- Use sort key for range queries
- Avoid hot partitions
- Consider access patterns

**2. Indexing**
- Create GSIs for different query patterns
- Use LSIs for same partition queries
- Monitor index usage
- Consider costs

**3. Capacity Planning**
- Use on-demand for variable workloads
- Use provisioned for predictable workloads
- Enable auto-scaling
- Monitor throttling

**4. Cost Optimization**
- Right-size capacity
- Use on-demand appropriately
- Optimize item size
- Use DynamoDB Streams efficiently

---

## Common Patterns

### 1. Caching Pattern

**Redis**:
```redis
# Check cache first
GET user:123
# If miss, fetch from DB and cache
SET user:123 "data" EX 3600
```

**DynamoDB**:
- Use as primary database
- Or cache frequently accessed items
- Use TTL for automatic expiration

### 2. Session Storage

**Redis**:
```redis
SET session:abc123 "user_data" EX 1800
GET session:abc123
```

**DynamoDB**:
- Store sessions with TTL
- Automatic cleanup
- Global distribution

### 3. Rate Limiting

**Redis**:
```redis
INCR rate_limit:user:123
EXPIRE rate_limit:user:123 60
GET rate_limit:user:123
```

### 4. Leaderboards

**Redis**:
```redis
ZADD leaderboard 100 "player1"
ZREVRANGE leaderboard 0 9
```

**DynamoDB**:
- Use sort key for ranking
- Query with sort key
- Or use GSI for different views

---

## Key Takeaways

- Key-value stores provide simple, fast data access
- Redis: In-memory, rich data types, high performance
- DynamoDB: Managed service, automatic scaling, global tables
- Ideal for caching, sessions, simple lookups
- Choose Redis for in-memory needs and flexibility
- Choose DynamoDB for managed service and AWS integration
- Design keys carefully for performance
- Use appropriate patterns for common use cases

