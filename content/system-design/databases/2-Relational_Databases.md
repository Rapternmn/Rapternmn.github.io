+++
title = "Relational Databases"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Relational Databases: PostgreSQL, MySQL, SQLite - features, use cases, when to use, and comparison of popular SQL databases."
+++

---

## Introduction

Relational databases store data in tables with relationships between them, using SQL for querying. They provide ACID guarantees, strong consistency, and are the foundation of most transactional systems.

---

## What are Relational Databases?

**Relational Database**:
- Data organized in tables (relations)
- Tables have rows (records) and columns (attributes)
- Relationships defined by foreign keys
- SQL for querying and manipulation
- ACID transaction guarantees

**Key Characteristics**:
- **Structured Data**: Fixed schema
- **Relationships**: Foreign keys and joins
- **ACID Properties**: Transaction guarantees
- **SQL Language**: Standard query language
- **Normalization**: Data organization principles

---

## Popular Relational Databases

### 1. PostgreSQL

**Overview**: Advanced open-source relational database

**Key Features**:
- ACID compliant
- Rich data types (JSON, arrays, custom types)
- Advanced indexing (GIN, GiST, BRIN)
- Full-text search
- Extensions ecosystem
- MVCC (Multi-Version Concurrency Control)

**Strengths**:
- ✅ Advanced features
- ✅ Excellent for complex queries
- ✅ Strong consistency
- ✅ Extensible
- ✅ JSON support
- ✅ Full-text search

**Use Cases**:
- Complex applications
- Data warehousing
- Geospatial data
- Applications needing advanced features
- Enterprise applications

**When to Choose**:
- Need advanced SQL features
- Complex queries and analytics
- JSON and document-like data
- Extensibility required
- Enterprise-grade features

---

### 2. MySQL

**Overview**: Popular open-source relational database

**Key Features**:
- ACID compliant
- Multiple storage engines (InnoDB, MyISAM)
- Replication support
- Partitioning
- Full-text search
- Widely supported

**Strengths**:
- ✅ Mature and stable
- ✅ Excellent performance
- ✅ Large ecosystem
- ✅ Easy to use
- ✅ Good documentation
- ✅ Wide tooling support

**Use Cases**:
- Web applications
- Content management systems
- E-commerce platforms
- High-traffic websites
- LAMP/LEMP stacks

**When to Choose**:
- Web applications
- Need proven stability
- Large community support
- Performance critical
- Simple to moderate complexity

**Storage Engines**:
- **InnoDB**: ACID, transactions, foreign keys (default)
- **MyISAM**: Fast reads, no transactions
- **Memory**: In-memory storage

---

### 3. SQLite

**Overview**: Lightweight, embedded SQL database

**Key Features**:
- Serverless (file-based)
- Zero configuration
- ACID compliant
- Self-contained
- Small footprint
- Cross-platform

**Strengths**:
- ✅ No server required
- ✅ Easy to deploy
- ✅ Perfect for embedded systems
- ✅ Fast for small datasets
- ✅ Simple setup
- ✅ Portable

**Limitations**:
- ❌ Single writer
- ❌ Limited concurrency
- ❌ Not for high-traffic
- ❌ No user management
- ❌ Limited data types

**Use Cases**:
- Mobile applications
- Desktop applications
- Embedded systems
- Development and testing
- Small applications
- IoT devices

**When to Choose**:
- Embedded applications
- Mobile apps
- Development/testing
- Small-scale applications
- No server infrastructure

---

## Comparison

### Feature Comparison

| Feature | PostgreSQL | MySQL | SQLite |
|---------|-----------|-------|--------|
| **Type** | Server | Server | Embedded |
| **ACID** | Yes | Yes (InnoDB) | Yes |
| **JSON Support** | Excellent | Good | Limited |
| **Full-Text Search** | Excellent | Good | Limited |
| **Performance** | Excellent | Excellent | Good (small DB) |
| **Complexity** | High | Medium | Low |
| **Scalability** | High | High | Low |
| **Extensions** | Excellent | Good | Limited |

### Use Case Comparison

**PostgreSQL**:
- Complex applications
- Data warehousing
- Advanced features needed
- Enterprise applications

**MySQL**:
- Web applications
- High-traffic sites
- Simple to moderate complexity
- LAMP/LEMP stacks

**SQLite**:
- Embedded systems
- Mobile apps
- Development/testing
- Small applications

---

## Relational Database Concepts

### 1. Tables and Relations

**Table Structure**:
- **Rows**: Records/tuples
- **Columns**: Attributes/fields
- **Primary Key**: Unique identifier
- **Foreign Key**: Reference to another table

### 2. Normalization

**Purpose**: Reduce redundancy and improve integrity

**Normal Forms**:
- **1NF**: Atomic values, no repeating groups
- **2NF**: No partial dependencies
- **3NF**: No transitive dependencies
- **BCNF**: Stronger normalization

### 3. Relationships

**Types**:
- **One-to-One**: Each record relates to one other
- **One-to-Many**: One record relates to many
- **Many-to-Many**: Many records relate to many (via junction table)

### 4. Indexes

**Types**:
- **Primary Index**: On primary key
- **Secondary Index**: On other columns
- **Composite Index**: Multiple columns
- **Unique Index**: Enforce uniqueness

---

## SQL Basics

### Data Definition Language (DDL)

```sql
-- Create table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index
CREATE INDEX idx_email ON users(email);

-- Alter table
ALTER TABLE users ADD COLUMN age INTEGER;
```

### Data Manipulation Language (DML)

```sql
-- Insert
INSERT INTO users (name, email) 
VALUES ('John', 'john@example.com');

-- Update
UPDATE users SET name = 'Jane' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;

-- Select
SELECT * FROM users WHERE email = 'john@example.com';
```

### Joins

```sql
-- Inner join
SELECT u.name, o.order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- Left join
SELECT u.name, o.order_id
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

---

## Transaction Management

### ACID Properties

**Atomicity**: All or nothing
**Consistency**: Valid state transitions
**Isolation**: Concurrent transactions don't interfere
**Durability**: Committed changes persist

### Transaction Example

```sql
BEGIN;

UPDATE accounts SET balance = balance - 100 
WHERE id = 1;

UPDATE accounts SET balance = balance + 100 
WHERE id = 2;

COMMIT; -- or ROLLBACK on error
```

### Isolation Levels

- **Read Uncommitted**: Lowest isolation
- **Read Committed**: Default in many databases
- **Repeatable Read**: Consistent reads
- **Serializable**: Highest isolation

---

## Performance Optimization

### 1. Indexing

**When to Index**:
- Frequently queried columns
- Foreign keys
- Columns in WHERE clauses
- Columns in JOIN conditions

**Index Types**:
- B-tree (default)
- Hash
- Bitmap
- Full-text

### 2. Query Optimization

**Techniques**:
- Use EXPLAIN to analyze queries
- Avoid SELECT *
- Use appropriate joins
- Filter early
- Use LIMIT when possible

### 3. Connection Pooling

**Purpose**: Reuse database connections

**Benefits**:
- Reduce connection overhead
- Better resource utilization
- Improved performance

---

## Replication

### Master-Slave Replication

**Architecture**:
- One master (writes)
- Multiple slaves (reads)
- Asynchronous replication

**Benefits**:
- Read scaling
- Backup and disaster recovery
- Load distribution

### Master-Master Replication

**Architecture**:
- Multiple masters
- Bidirectional replication
- Conflict resolution needed

**Benefits**:
- Write scaling
- High availability
- Geographic distribution

---

## Best Practices

### 1. Database Design

- Normalize appropriately
- Use appropriate data types
- Define constraints
- Index strategically

### 2. Query Optimization

- Analyze query plans
- Use indexes effectively
- Avoid N+1 queries
- Optimize joins

### 3. Security

- Use parameterized queries
- Implement access control
- Encrypt sensitive data
- Regular security updates

### 4. Backup and Recovery

- Regular backups
- Test recovery procedures
- Point-in-time recovery
- Disaster recovery plan

---

## Key Takeaways

- Relational databases provide ACID guarantees and strong consistency
- PostgreSQL: Advanced features, complex queries
- MySQL: Popular, performant, web applications
- SQLite: Embedded, lightweight, no server
- Choose based on use case, complexity, and requirements
- Design with normalization and relationships
- Optimize with indexes and query tuning
- Plan for replication and scaling

