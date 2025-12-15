+++
title = "Distributed Lock Service (Chubby/etcd)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 20
description = "Design a distributed lock service like Chubby or etcd. Covers distributed locking, leader election, consensus algorithms, and coordination in distributed systems."
+++

---

## Problem Statement

Design a distributed lock service that provides coordination primitives for distributed systems. The system should support distributed locking, leader election, and configuration management while maintaining consistency and availability.

**Examples**: Chubby (Google), etcd, Consul, ZooKeeper

---

## Requirements Clarification

### Functional Requirements

1. **Distributed Locks**: Acquire and release locks
2. **Leader Election**: Elect leader among multiple nodes
3. **Configuration Management**: Store and retrieve configuration
4. **Service Discovery**: Register and discover services
5. **Event Notifications**: Notify on changes
6. **Sessions**: Manage client sessions

### Non-Functional Requirements

- **Scale**: 
  - 10K clients
  - 100K operations/second
  - 1000 locks
  - Average lock duration: 10 seconds
- **Latency**: < 10ms for lock operations
- **Availability**: 99.9% uptime
- **Consistency**: Strong consistency (CP)

---

## Capacity Estimation

### Traffic Estimates

- **Operations**: 100K operations/second
- **Lock Operations**: 10K locks × 10 ops/sec = 100K ops/sec
- **Read/Write Ratio**: 80:20 (read-heavy)
- **Reads**: 80K reads/second
- **Writes**: 20K writes/second

### Storage Estimates

- **Locks**: 1000 locks × 1 KB = 1 MB
- **Configurations**: 10K configs × 10 KB = 100 MB
- **Sessions**: 10K sessions × 1 KB = 10 MB
- **Total Storage**: ~200 MB (small, but critical)

---

## API Design

### Lock APIs

```
POST /api/v1/locks/{lockName}/acquire
Request: {
  "timeout": 30,  // seconds
  "sessionId": "session123"
}
Response: {
  "success": true,
  "lockId": "lock456",
  "expiresAt": "2025-12-15T10:00:30Z"
}

POST /api/v1/locks/{lockName}/release
Request: {
  "lockId": "lock456"
}
Response: {
  "success": true
}

GET /api/v1/locks/{lockName}
Response: {
  "lockId": "lock456",
  "holder": "client789",
  "acquiredAt": "2025-12-15T10:00:00Z",
  "expiresAt": "2025-12-15T10:00:30Z"
}
```

### Leader Election APIs

```
POST /api/v1/leader/election/{electionName}
Request: {
  "candidateId": "node123",
  "sessionId": "session123"
}
Response: {
  "isLeader": true,
  "leaderId": "node123"
}

GET /api/v1/leader/{electionName}
Response: {
  "leaderId": "node123",
  "electedAt": "2025-12-15T10:00:00Z"
}
```

---

## Database Design

### Storage Structure

**Locks Table** (Distributed KV Store):
```
lockName (PK): VARCHAR
lockId: UUID
holder: VARCHAR (client ID)
acquiredAt: TIMESTAMP
expiresAt: TIMESTAMP
sessionId: UUID
```

**Sessions Table**:
```
sessionId (PK): UUID
clientId: VARCHAR
createdAt: TIMESTAMP
lastHeartbeat: TIMESTAMP
ttl: INT (seconds)
```

**Configurations Table**:
```
path (PK): VARCHAR (e.g., /config/database/url)
value: TEXT
version: INT
updatedAt: TIMESTAMP
```

### Storage Selection

**Storage**: **Distributed KV Store** (etcd, Consul) - strong consistency
**Consensus**: **Raft** or **Paxos** - distributed consensus

---

## High-Level Design

### Architecture

```
Clients → [Load Balancer] → [Lock Service Nodes]
                                    ↓
                            [Consensus Layer]
                                    ↓
                            [Distributed Storage]
```

### Components

1. **Lock Service Nodes**: Handle lock operations
2. **Consensus Layer**: Raft/Paxos for consensus
3. **Distributed Storage**: Store locks, configurations
4. **Session Manager**: Manage client sessions
5. **Event Notifier**: Notify on changes

---

## Detailed Design

### Distributed Locking

**Lock Types**:

1. **Exclusive Lock**:
   - **Single Holder**: Only one client can hold lock
   - **Use Case**: Critical section, resource access
   - **Implementation**: Create node, only one can exist

2. **Shared Lock**:
   - **Multiple Holders**: Multiple clients can hold lock
   - **Use Case**: Read operations
   - **Implementation**: Count-based locking

**Lock Implementation**:

**Approach 1: Ephemeral Nodes** (ZooKeeper/etcd):
- **Create Node**: Create ephemeral node for lock
- **Success**: If node created, lock acquired
- **Failure**: If node exists, lock not acquired
- **Release**: Delete node (or node expires)

**Approach 2: Compare-and-Swap**:
- **CAS**: Compare-and-swap operation
- **Atomic**: Atomic operation
- **Implementation**: Use CAS to acquire lock

**Recommendation**: **Ephemeral Nodes** (simpler)

---

### Leader Election

**Election Algorithms**:

1. **Bully Algorithm**:
   - **Highest ID Wins**: Node with highest ID becomes leader
   - **Pros**: Simple
   - **Cons**: Not fault-tolerant

2. **Ring Algorithm**:
   - **Ring Topology**: Nodes in ring
   - **Election Message**: Pass election message
   - **Pros**: Fault-tolerant
   - **Cons**: Complex

3. **Raft Consensus**:
   - **Voting**: Nodes vote for leader
   - **Majority**: Leader elected by majority
   - **Pros**: Fault-tolerant, well-understood
   - **Cons**: Requires majority

**Recommendation**: **Raft Consensus** (most common)

**Election Flow**:
1. **Start Election**: Node starts election
2. **Request Votes**: Request votes from other nodes
3. **Vote**: Nodes vote for candidate
4. **Majority**: If majority votes, become leader
5. **Heartbeat**: Leader sends heartbeats
6. **Re-election**: Re-elect if leader fails

---

### Consensus Algorithms

**Consensus Requirements**:
- **Safety**: All nodes agree on same value
- **Liveness**: System makes progress
- **Fault Tolerance**: Tolerate node failures

**Consensus Algorithms**:

1. **Raft**:
   - **Leader-Based**: Single leader
   - **Log Replication**: Replicate log entries
   - **Majority**: Requires majority for consensus
   - **Pros**: Easier to understand than Paxos
   - **Cons**: Requires majority

2. **Paxos**:
   - **Multi-Phase**: Multi-phase protocol
   - **Quorum**: Requires quorum
   - **Pros**: More flexible
   - **Cons**: Complex

**Recommendation**: **Raft** (easier to understand and implement)

---

### Session Management

**Sessions**:
- **Purpose**: Track client connections
- **Heartbeat**: Client sends heartbeat periodically
- **TTL**: Session expires if no heartbeat
- **Cleanup**: Clean up expired sessions

**Session Flow**:
1. **Create Session**: Client creates session
2. **Heartbeat**: Client sends heartbeat periodically
3. **Extend TTL**: Extend session TTL on heartbeat
4. **Expire**: Session expires if no heartbeat
5. **Cleanup**: Clean up session and associated locks

**Session Benefits**:
- **Automatic Cleanup**: Locks released on session expiry
- **Fault Detection**: Detect client failures
- **Resource Management**: Manage client resources

---

### Event Notifications

**Notification Types**:

1. **Watch**: Watch for changes
2. **Notify**: Notify on changes
3. **Subscribe**: Subscribe to events

**Use Cases**:
- **Configuration Changes**: Notify on config changes
- **Lock Releases**: Notify when lock released
- **Leader Changes**: Notify on leader change

**Implementation**:
- **Polling**: Poll for changes (simple, but inefficient)
- **Push Notifications**: Push notifications (efficient)
- **Webhooks**: Webhook callbacks

---

## Scalability

### Horizontal Scaling

- **Multiple Nodes**: Scale lock service nodes
- **Sharding**: Shard locks by name (hash-based)
- **Read Replicas**: Read replicas for reads

### Performance Optimization

- **Caching**: Cache lock states
- **Batching**: Batch operations
- **Connection Pooling**: Reuse connections

---

## Reliability

### High Availability

- **Multiple Nodes**: No single point of failure
- **Consensus**: Consensus ensures consistency
- **Replication**: Replicate data across nodes

### Fault Tolerance

- **Node Failures**: Continue operation with remaining nodes
- **Network Partitions**: Handle split-brain scenarios
- **Leader Failures**: Re-elect leader

### Consistency

- **Strong Consistency**: CP system (consistency over availability)
- **Linearizability**: All operations appear atomic
- **Sequential Consistency**: Operations appear in order

---

## Trade-offs

### Consistency vs Availability

- **CP System**: Consistency over availability
- **Requires Majority**: Requires majority for operations
- **Partition Tolerance**: Handles network partitions

### Latency vs Consistency

- **Strong Consistency**: Higher latency
- **Eventual Consistency**: Lower latency (not suitable for locks)

---

## Extensions

### Additional Features

1. **Read Locks**: Shared locks for reads
2. **Lock Queues**: Queue lock requests
3. **Lock Timeouts**: Automatic lock expiration
4. **Lock Renewal**: Renew locks before expiry
5. **Multi-Lock**: Acquire multiple locks atomically
6. **Lock Hierarchies**: Hierarchical locks
7. **Distributed Transactions**: Support transactions

---

## Key Takeaways

- **Distributed Locks**: Use ephemeral nodes or CAS
- **Leader Election**: Use Raft consensus
- **Consensus**: Raft or Paxos for consistency
- **Sessions**: Manage client sessions with heartbeats
- **CP System**: Consistency over availability
- **Fault Tolerance**: Handle node failures with consensus

---

## Related Topics

- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Consensus, CAP theorem
- **[Key-Value Store]({{< ref "17-Key_Value_Store.md" >}})** - Distributed storage
- **[Availability & Reliability]({{< ref "../system-components/10-Availability_Reliability.md" >}})** - Fault tolerance
- **[Service Discovery & Service Mesh]({{< ref "../system-components/6-Service_Discovery_Service_Mesh.md" >}})** - Service coordination

