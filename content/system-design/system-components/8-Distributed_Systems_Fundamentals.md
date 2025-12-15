+++
title = "Distributed Systems Fundamentals"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 8
description = "Core concepts of distributed systems: CAP theorem, ACID vs BASE, consensus algorithms, distributed transactions, and patterns like Saga, CQRS, and Event Sourcing."
+++

---

## Introduction

Distributed systems are collections of independent computers that appear to users as a single coherent system. Understanding distributed systems fundamentals is essential for designing scalable, reliable systems that can handle failures and maintain consistency.

---

## What are Distributed Systems?

A distributed system is a system whose components are located on different networked computers, which communicate and coordinate their actions by passing messages.

### Key Characteristics

- **Concurrency**: Multiple components execute simultaneously
- **Lack of Global Clock**: No single global time source
- **Independent Failures**: Components fail independently
- **Heterogeneity**: Different hardware, software, networks

---

## Distributed System Challenges

### 1. Network Partitions

**Definition**: Network splits that isolate parts of the system.

**Impact**: 
- Components can't communicate
- Data inconsistency possible
- Availability vs consistency trade-off

---

### 2. Partial Failures

**Definition**: Some components fail while others continue.

**Impact**: 
- System continues partially
- Harder to detect and handle
- Requires fault tolerance

---

### 3. Consistency

**Definition**: Ensuring all nodes see same data.

**Challenges**: 
- Network delays
- Concurrent updates
- Failures

---

### 4. Latency

**Definition**: Network communication takes time.

**Impact**: 
- Slower operations
- Coordination overhead
- User experience

---

## CAP Theorem

The CAP theorem states that a distributed system can guarantee at most two of three properties:

### Consistency

**Definition**: All nodes see the same data simultaneously.

**Characteristics**:
- Strong consistency
- All reads return most recent write
- Synchronous replication

---

### Availability

**Definition**: System remains operational.

**Characteristics**:
- Every request gets response
- No errors
- System continues despite failures

---

### Partition Tolerance

**Definition**: System continues despite network partitions.

**Characteristics**:
- Handles network failures
- Required for distributed systems
- Cannot be sacrificed

---

### CAP Trade-offs

**CP (Consistency + Partition Tolerance)**:
- Strong consistency
- May sacrifice availability
- Example: Traditional databases

**AP (Availability + Partition Tolerance)**:
- High availability
- May sacrifice consistency
- Example: DNS, NoSQL databases

**CA (Consistency + Availability)**:
- Not possible in distributed systems
- Requires no network partitions (single node)

---

## ACID vs BASE

### ACID Properties

**Atomicity**: All or nothing - transactions complete fully or not at all.

**Consistency**: Database remains in valid state.

**Isolation**: Concurrent transactions don't interfere.

**Durability**: Committed changes persist.

**Use Cases**: 
- Financial transactions
- Critical data
- When consistency is paramount

---

### BASE Properties

**Basically Available**: System remains available.

**Soft State**: State may change over time.

**Eventual Consistency**: System will become consistent.

**Use Cases**: 
- High availability systems
- When eventual consistency is acceptable
- Large-scale systems

---

## Distributed Consensus

### What is Consensus?

Agreement among distributed nodes on a value or decision.

### Two-Phase Commit (2PC)

**How it works**:
1. **Phase 1 (Prepare)**: Coordinator asks all participants to prepare
2. **Phase 2 (Commit/Abort)**: Coordinator commits or aborts based on responses

**Advantages**: Strong consistency

**Disadvantages**: 
- Blocking (waits for all)
- Single point of failure
- Poor performance

---

### Three-Phase Commit (3PC)

**How it works**: Adds pre-commit phase to 2PC.

**Advantages**: Non-blocking, better fault tolerance

**Disadvantages**: More complex, still has issues

---

### Raft Consensus Algorithm

**How it works**: 
- Leader election
- Log replication
- Safety guarantees

**Advantages**: 
- Understandable
- Production-ready
- Good performance

**Use Cases**: etcd, Consul

---

### Paxos Algorithm

**How it works**: 
- Proposers propose values
- Acceptors accept proposals
- Learners learn chosen values

**Advantages**: Proven correct

**Disadvantages**: Complex to understand and implement

**Use Cases**: Google Chubby, distributed databases

---

## Distributed Transactions

### Challenges

- **Atomicity**: Ensuring all-or-nothing across nodes
- **Consistency**: Maintaining consistency across nodes
- **Isolation**: Preventing interference
- **Performance**: Network overhead

### Solutions

- **Two-Phase Commit**: Coordinated commit
- **Saga Pattern**: Compensating transactions
- **Eventual Consistency**: Accept temporary inconsistency

---

## Eventual Consistency

### Definition

System will become consistent over time, but may be temporarily inconsistent.

### Types

- **Causal Consistency**: Causally related operations are consistent
- **Read-Your-Writes**: User sees own writes
- **Session Consistency**: Consistent within session
- **Monotonic Read**: Never see older data

---

## Distributed System Patterns

### 1. Saga Pattern

**Definition**: Long-running transactions broken into smaller transactions with compensating actions.

**How it works**:
- Each step has compensating action
- If step fails, execute compensations
- Eventual consistency

**Use Cases**: 
- E-commerce orders
- Multi-step processes
- When 2PC not suitable

---

### 2. CQRS (Command Query Responsibility Segregation)

**Definition**: Separate read and write models.

**How it works**:
- Commands (writes) go to write model
- Queries (reads) go to read model
- Models can be different

**Advantages**: 
- Independent scaling
- Optimized for each operation
- Flexibility

**Use Cases**: 
- High read/write ratio
- Complex queries
- Event sourcing

---

### 3. Event Sourcing

**Definition**: Store all changes as sequence of events.

**How it works**:
- Store events, not current state
- Replay events to reconstruct state
- Immutable event log

**Advantages**: 
- Complete audit trail
- Time travel
- Event replay

**Use Cases**: 
- Audit requirements
- Complex domain models
- Event-driven systems

---

## Use Cases

### Understanding System Trade-offs

CAP theorem helps understand consistency vs availability trade-offs.

---

### Designing for Consistency vs Availability

Choose based on requirements:
- **Consistency**: Financial systems, critical data
- **Availability**: Social media, content delivery

---

### Handling Distributed Failures

Patterns like Saga, circuit breakers help handle failures gracefully.

---

## Key Takeaways

- **Distributed systems** have unique challenges (partitions, failures, consistency)
- **CAP theorem** shows consistency vs availability trade-off
- **ACID** for strong consistency, **BASE** for availability
- **Consensus algorithms** enable agreement in distributed systems
- **Patterns** like Saga, CQRS, Event Sourcing solve distributed problems
- **Eventual consistency** is often acceptable for better availability

---

## Related Topics

- **[Message Queues & Message Brokers]({{< ref "5-Message_Queues_Message_Brokers.md" >}})** - Eventual consistency patterns
- **[Availability & Reliability]({{< ref "10-Availability_Reliability.md" >}})** - Handling failures
- **[Scalability Patterns]({{< ref "9-Scalability_Patterns.md" >}})** - Scaling distributed systems

