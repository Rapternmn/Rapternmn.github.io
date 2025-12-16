+++
title = "Distributed Cache (Redis Cluster)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 13
description = "Design a distributed cache system like Redis Cluster. Covers cache sharding, replication, consistency models, eviction policies, and scaling to handle millions of requests."
+++

---

## Problem Statement

Design a distributed cache system that provides fast data access across multiple servers. The system should handle cache sharding, replication, consistency, and eviction policies while maintaining high availability and performance.

**Examples**: Redis Cluster, Memcached, Hazelcast

---

## Requirements Clarification

### Functional Requirements

1. **Cache Operations**: Get, set, delete operations
2. **Distributed Storage**: Data distributed across multiple nodes
3. **Replication**: Replicate data for availability
4. **Eviction**: Evict data when cache full
5. **Expiration**: Support TTL (time-to-live)
6. **Consistency**: Handle consistency across nodes

### Non-Functional Requirements

- **Scale**: 
  - 100M keys
  - 1M requests/second
  - 100 GB total cache size
  - 10 cache nodes
- **Latency**: < 1ms for cache operations
- **Availability**: 99.9% uptime
- **Consistency**: Eventual consistency acceptable

---

## Capacity Estimation

### Traffic Estimates

- **Requests**: 1M requests/second
- **Read/Write Ratio**: 80:20 (read-heavy)
- **Reads**: 800K reads/second
- **Writes**: 200K writes/second

### Storage Estimates

- **Total Cache**: 100 GB
- **Per Node**: 100 GB / 10 nodes = 10 GB per node
- **Keys**: 100M keys
- **Average Key Size**: 1 KB
- **Total Data**: 100 GB

---

## API Design

### Cache APIs

```
GET /api/v1/cache/{key}
Response: {
  "key": "user123",
  "value": "...",
  "ttl": 3600
}

SET /api/v1/cache/{key}
Request: {
  "value": "...",
  "ttl": 3600  // seconds
}
Response: {
  "success": true
}

DELETE /api/v1/cache/{key}
Response: {
  "success": true
}
```

---

## Database Design

### Cache Storage

**In-Memory Storage** (Redis):
- **Data Structures**: Strings, Hashes, Lists, Sets, Sorted Sets
- **Persistence**: Optional persistence (RDB, AOF)
- **Partitioning**: Hash-based sharding

**Metadata** (Coordination Service):
- **Node Information**: Node addresses, roles
- **Shard Mapping**: Key → Node mapping
- **Cluster State**: Cluster health, failures

---

## High-Level Design

### Architecture

```
Client → Load Balancer → Cache Proxy
                            ↓
                    [Hash Ring]
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
    [Node 1]           [Node 2]           [Node 3]
    (Shard 1)         (Shard 2)         (Shard 3)
        ↓                   ↓                   ↓
    [Replica]          [Replica]          [Replica]
```

### Components

1. **Cache Proxy**: Route requests to correct node
2. **Hash Ring**: Consistent hashing for sharding
3. **Cache Nodes**: Store cache data
4. **Replicas**: Replicate data for availability
5. **Coordination Service**: Manage cluster state
6. **Eviction Service**: Handle cache eviction

---

## Detailed Design

### Cache Sharding

**Sharding Strategies**:

1. **Hash-Based Sharding**:
   - Hash key → Determine node
   - Even distribution
   - Simple implementation

2. **Range-Based Sharding**:
   - Partition by key range
   - Uneven distribution possible
   - More complex

3. **Directory-Based Sharding**:
   - Centralized directory
   - Flexible, but single point of failure
   - More overhead

**Recommendation**: **Hash-Based Sharding** (consistent hashing)

---

### Consistent Hashing

**How it works**:
1. **Hash Ring**: Map nodes and keys to hash ring
2. **Key Lookup**: Find first node clockwise from key hash
3. **Node Addition**: Only remap keys from adjacent nodes
4. **Node Removal**: Remap keys to next node

**Benefits**:
- **Minimal Remapping**: Only remap ~1/N keys on node change
- **Even Distribution**: Even key distribution
- **Scalability**: Easy to add/remove nodes

**Virtual Nodes**:
- **Problem**: Uneven distribution with few nodes
- **Solution**: Multiple virtual nodes per physical node
- **Benefit**: Better distribution

---

### Replication

**Replication Strategies**:

1. **Master-Slave Replication**:
   - **Master**: Handles writes
   - **Slaves**: Replicate from master, handle reads
   - **Failover**: Promote slave to master on failure

2. **Multi-Master Replication**:
   - **Multiple Masters**: All nodes accept writes
   - **Conflict Resolution**: Resolve conflicts
   - **Complexity**: More complex

**Recommendation**: **Master-Slave Replication** (simpler)

**Replication Flow**:
1. **Write to Master**: Client writes to master
2. **Replicate to Slaves**: Master replicates to slaves
3. **Acknowledge**: Master acknowledges after replication
4. **Read from Slaves**: Read from slaves (reduce master load)

---

### Consistency Models

**Consistency Levels**:

1. **Strong Consistency**:
   - All reads see latest write
   - Higher latency
   - Use case: Critical data

2. **Eventual Consistency**:
   - Reads may see stale data
   - Lower latency
   - Use case: Cache (acceptable)

3. **Session Consistency**:
   - Consistent within session
   - Balance between strong and eventual
   - Use case: User sessions

**Cache Recommendation**: **Eventual Consistency** (acceptable for cache)

---

### Eviction Policies

**Eviction Policies**:

1. **LRU (Least Recently Used)**:
   - Evict least recently used items
   - Good for temporal locality
   - Common choice

2. **LFU (Least Frequently Used)**:
   - Evict least frequently used items
   - Good for frequency-based access
   - More complex

3. **FIFO (First In First Out)**:
   - Evict oldest items
   - Simple, but not optimal
   - Rarely used

4. **TTL (Time To Live)**:
   - Evict expired items
   - Time-based expiration
   - Common for cache

**Recommendation**: **LRU + TTL** (combination)

---

### Cache Operations

#### Get Operation

1. **Hash Key**: Hash key to determine node
2. **Route to Node**: Route request to node
3. **Check Cache**: Check if key exists
4. **Check TTL**: Check if expired
5. **Return Value**: Return value or null

#### Set Operation

1. **Hash Key**: Hash key to determine node
2. **Route to Node**: Route request to node
3. **Store Value**: Store value in cache
4. **Set TTL**: Set expiration time
5. **Replicate**: Replicate to replicas
6. **Evict if Needed**: Evict if cache full

#### Delete Operation

1. **Hash Key**: Hash key to determine node
2. **Route to Node**: Route request to node
3. **Delete Key**: Delete key from cache
4. **Replicate**: Replicate deletion to replicas

---

## Scalability

### Horizontal Scaling

- **Add Nodes**: Add nodes to cluster
- **Rebalance**: Rebalance keys across nodes
- **Consistent Hashing**: Minimal key remapping

### Performance Optimization

- **Connection Pooling**: Reuse connections
- **Pipelining**: Batch operations
- **Local Caching**: Cache frequently accessed keys locally

---

## Reliability

### High Availability

- **Replication**: Replicate data across nodes
- **Failover**: Automatic failover on node failure
- **Health Checks**: Monitor node health

### Fault Tolerance

- **Node Failures**: Continue operation with remaining nodes
- **Network Partitions**: Handle split-brain scenarios
- **Data Loss**: Minimize data loss with replication

### Cache Stampede (Thundering Herd)
**Problem**: Multiple clients request same missing key simultaneously, causing load spike on database.
**Solutions**:
- **Locking**: First client gets lock, others wait.
- **Probabilistic Early Expiration**: Refresh ahead of TTL.
- **Request Coalescing**: Combine requests for same key.

### Consistency Trade-offs

- **Availability vs Consistency**: Choose based on use case
- **Cache**: Prefer availability (eventual consistency)
- **Critical Data**: Prefer consistency (strong consistency)

---

## Trade-offs

### Consistency vs Availability

- **Strong Consistency**: Lower availability, higher latency
- **Eventual Consistency**: Higher availability, lower latency

### Memory vs Performance

- **More Memory**: Better hit rate, higher cost
- **Less Memory**: Lower hit rate, lower cost

### Replication vs Performance

- **Synchronous Replication**: Strong consistency, slower writes
- **Asynchronous Replication**: Eventual consistency, faster writes

---

## Extensions

### Additional Features

1. **Cache Warming**: Preload frequently accessed data
2. **Cache Invalidation**: Invalidate cache on data change
3. **Cache Analytics**: Track hit rate, miss rate
4. **Multi-Region**: Replicate across regions
5. **Encryption**: Encrypt cache data
6. **Compression**: Compress cache values
7. **Pub/Sub**: Publish/subscribe for cache events

---

## Key Takeaways

- **Consistent Hashing**: Efficient sharding with minimal remapping
- **Replication**: Master-slave replication for availability
- **Eviction**: LRU + TTL for optimal cache management
- **Consistency**: Eventual consistency acceptable for cache
- **Scalability**: Horizontal scaling with consistent hashing
- **Fault Tolerance**: Replication and failover for reliability

---

## Related Topics

- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Cache patterns and strategies
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed architecture
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Scaling strategies
- **[Availability & Reliability]({{< ref "../system-components/10-Availability_Reliability.md" >}})** - High availability patterns

