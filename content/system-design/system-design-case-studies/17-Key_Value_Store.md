+++
title = "Key-Value Store (DynamoDB/Redis)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 17
description = "Design a distributed key-value store like DynamoDB or Redis. Covers partitioning, replication, consistency models, and scaling to handle billions of keys."
+++

---

## Problem Statement

Design a distributed key-value store that provides fast read/write access to key-value pairs across multiple servers. The system should handle partitioning, replication, and consistency while maintaining high availability and performance.

**Examples**: DynamoDB, Redis Cluster, Cassandra, Riak

---

## Requirements Clarification

### Functional Requirements

1. **Key-Value Operations**: Get, put, delete operations
2. **Distributed Storage**: Data distributed across multiple nodes
3. **Replication**: Replicate data for availability
4. **Consistency**: Handle consistency across replicas
5. **Partitioning**: Partition data across nodes
6. **Querying**: Query by key (primary key only)

### Non-Functional Requirements

- **Scale**: 
  - 1B keys
  - 1M operations/second
  - 100 GB total data
  - 10 nodes
- **Latency**: < 10ms for operations
- **Availability**: 99.9% uptime
- **Durability**: 99.999999999% (11 nines)

---

## Capacity Estimation

### Traffic Estimates

- **Operations**: 1M operations/second
- **Read/Write Ratio**: 70:30 (read-heavy)
- **Reads**: 700K reads/second
- **Writes**: 300K writes/second

### Storage Estimates

- **Total Data**: 100 GB
- **Per Node**: 100 GB / 10 nodes = 10 GB per node
- **With Replication**: 100 GB × 3 replicas = 300 GB total
- **Keys**: 1B keys
- **Average Key Size**: 100 bytes

---

## API Design

### Key-Value APIs

```
GET /api/v1/kv/{key}
Response: {
  "key": "user123",
  "value": "...",
  "version": 5
}

PUT /api/v1/kv/{key}
Request: {
  "value": "...",
  "condition": "ifNotExists"  // optional
}
Response: {
  "success": true,
  "version": 6
}

DELETE /api/v1/kv/{key}
Response: {
  "success": true
}

BATCH_GET /api/v1/kv/batch
Request: {
  "keys": ["key1", "key2", ...]
}
Response: {
  "items": [...]
}
```

---

## Database Design

### Storage Structure

**Key-Value Storage** (In-Memory/Disk):
- **Keys**: Unique identifiers
- **Values**: Arbitrary data (strings, JSON, binary)
- **Metadata**: Version, timestamp, TTL

**Partitioning**:
- **Hash-Based**: Hash key → Determine partition
- **Range-Based**: Partition by key range
- **Consistent Hashing**: Use consistent hashing

**Replication**:
- **Replication Factor**: 3 replicas (configurable)
- **Replication Strategy**: Master-slave or multi-master

---

## High-Level Design

### Architecture

```
Client → Load Balancer → [Partition Router]
                            ↓
                    [Hash Ring]
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
    [Node 1]           [Node 2]           [Node 3]
    (Partition 1)      (Partition 2)      (Partition 3)
        ↓                   ↓                   ↓
    [Replica]          [Replica]          [Replica]
```

### Components

1. **Partition Router**: Route requests to correct partition
2. **Hash Ring**: Consistent hashing for partitioning
3. **Storage Nodes**: Store key-value pairs
4. **Replicas**: Replicate data for availability
5. **Coordination Service**: Manage cluster state
6. **Gossip Protocol**: Node discovery and state synchronization

---

## Detailed Design

### Partitioning

**Partitioning Strategies**:

1. **Hash-Based Partitioning**:
   - Hash key → Determine partition
   - Even distribution
   - Simple implementation

2. **Range-Based Partitioning**:
   - Partition by key range
   - Uneven distribution possible
   - Supports range queries (if needed)

3. **Directory-Based Partitioning**:
   - Centralized directory
   - Flexible, but single point of failure

**Recommendation**: **Hash-Based Partitioning** (consistent hashing)

---

### Consistent Hashing

**How it works**:
1. **Hash Ring**: Map nodes and keys to hash ring
2. **Key Lookup**: Find first node clockwise from key hash
3. **Virtual Nodes**: Multiple virtual nodes per physical node
4. **Replication**: Replicate to next N nodes on ring

**Benefits**:
- **Minimal Remapping**: Only remap ~1/N keys on node change
- **Even Distribution**: Even key distribution
- **Scalability**: Easy to add/remove nodes

---

### Replication

**Replication Strategies**:

1. **Master-Slave Replication**:
   - **Master**: Handles writes
   - **Slaves**: Replicate from master, handle reads
   - **Failover**: Promote slave to master on failure

2. **Multi-Master Replication**:
   - **Multiple Masters**: All nodes accept writes
   - **Conflict Resolution**: Resolve conflicts (vector clocks, last-write-wins)
   - **Complexity**: More complex

**Recommendation**: **Multi-Master Replication** (better availability)

**Replication Flow**:
1. **Write to Node**: Client writes to any node
2. **Replicate**: Node replicates to other replicas
3. **Acknowledge**: Acknowledge after replication (based on consistency level)
4. **Read**: Read from any replica (or specific replica based on consistency)

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
   - Use case: Non-critical data

3. **Read-Your-Writes Consistency**:
   - Reads see own writes
   - Balance between strong and eventual
   - Use case: User sessions

**Consistency Parameters**:
- **R (Read Quorum)**: Number of replicas to read from
- **W (Write Quorum)**: Number of replicas to write to
- **N (Replication Factor)**: Total number of replicas

**Examples**:
- **Strong**: R + W > N (e.g., R=2, W=2, N=3)
- **Eventual**: R + W ≤ N (e.g., R=1, W=1, N=3)

---

### Vector Clocks (Conflict Resolution)

**Problem**: Multiple writes to same key from different nodes

**Solution**: Vector Clocks

**How it works**:
- **Vector Clock**: [Node1: 1, Node2: 2, Node3: 1]
- **Compare Clocks**: Determine causality
- **Conflict Detection**: Detect concurrent writes
- **Resolution**: Resolve conflicts (last-write-wins, merge, user resolution)

**Implementation**:
- **Attach Clock**: Attach vector clock to each value
- **Update Clock**: Update clock on write
- **Compare**: Compare clocks on read

---

### Operations

#### Get Operation

1. **Hash Key**: Hash key to determine partition
2. **Route to Node**: Route request to node (or replicas)
3. **Read from Replicas**: Read from R replicas
4. **Resolve Conflicts**: Resolve if different versions
5. **Return Value**: Return latest value

#### Put Operation

1. **Hash Key**: Hash key to determine partition
2. **Route to Node**: Route request to node
3. **Write to Replicas**: Write to W replicas
4. **Replicate**: Replicate to remaining replicas asynchronously
5. **Acknowledge**: Acknowledge after W writes

#### Delete Operation

1. **Hash Key**: Hash key to determine partition
2. **Route to Node**: Route request to node
3. **Tombstone**: Mark as deleted (tombstone)
4. **Replicate**: Replicate deletion to replicas
5. **Garbage Collection**: Remove tombstone after TTL

---

## Scalability

### Horizontal Scaling

- **Add Nodes**: Add nodes to cluster
- **Rebalance**: Rebalance keys across nodes
- **Consistent Hashing**: Minimal key remapping

### Performance Optimization

- **Caching**: Cache frequently accessed keys
- **Connection Pooling**: Reuse connections
- **Batching**: Batch operations
- **Compression**: Compress values

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

### Data Durability

- **Replication**: Replicate data across nodes
- **Persistence**: Persist to disk (optional)
- **Backup**: Backup to object storage

---

## Trade-offs

### Consistency vs Availability

- **Strong Consistency**: Lower availability, higher latency
- **Eventual Consistency**: Higher availability, lower latency

### Durability vs Performance

- **Synchronous Writes**: Higher durability, slower writes
- **Asynchronous Writes**: Lower durability, faster writes

### Storage vs Performance

- **More Replicas**: Better availability, more storage
- **Fewer Replicas**: Less storage, lower availability

---

## Extensions

### Additional Features

1. **TTL (Time To Live)**: Expire keys automatically
2. **Conditional Writes**: Conditional put/delete
3. **Transactions**: Multi-key transactions
4. **Secondary Indexes**: Index by value attributes
5. **Streaming**: Stream changes
6. **Backup/Restore**: Backup and restore data
7. **Encryption**: Encrypt data at rest and in transit

---

## Key Takeaways

- **Consistent Hashing**: Efficient partitioning with minimal remapping
- **Multi-Master Replication**: Better availability than master-slave
- **Quorum-Based Consistency**: Balance consistency and availability
- **Vector Clocks**: Handle concurrent writes and conflicts
- **Scalability**: Horizontal scaling with consistent hashing
- **Fault Tolerance**: Replication and failover for reliability

---

## Related Topics

- **[Distributed Cache]({{< ref "13-Distributed_Cache.md" >}})** - Similar concepts, cache-specific optimizations
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - CAP theorem, consistency models
- **[Databases]({{< ref "../databases/_index.md" >}})** - NoSQL databases, DynamoDB
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Partitioning strategies

