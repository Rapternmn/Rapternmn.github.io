+++
title = "Database Design & Optimization"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Database Design & Optimization: Normalization, denormalization, indexing strategies, query optimization, and performance tuning techniques."
+++

---

## Introduction

Effective database design and optimization are crucial for performance, scalability, and maintainability. This guide covers design principles, indexing strategies, query optimization, and performance tuning techniques.

---

## Database Design Principles

### 1. Normalization

**Purpose**: Reduce data redundancy and improve integrity

**First Normal Form (1NF)**:
- Atomic values (no repeating groups)
- Each cell contains single value
- No duplicate rows

**Example**:
```
Before 1NF:
Order: {id: 1, items: "Book, Pen, Notebook"}

After 1NF:
Order: {id: 1, item: "Book"}
Order: {id: 1, item: "Pen"}
Order: {id: 1, item: "Notebook"}
```

**Second Normal Form (2NF)**:
- 1NF + no partial dependencies
- All non-key attributes depend on full primary key

**Third Normal Form (3NF)**:
- 2NF + no transitive dependencies
- Non-key attributes don't depend on other non-key attributes

**Boyce-Codd Normal Form (BCNF)**:
- Stronger than 3NF
- Every determinant is a candidate key

### 2. Denormalization

**Purpose**: Improve read performance

**When to Denormalize**:
- Read-heavy workloads
- Performance critical queries
- Acceptable trade-offs
- After normalization

**Techniques**:
- Duplicate data
- Pre-computed values
- Materialized views
- Redundant columns

**Trade-offs**:
- ✅ Faster reads
- ✅ Simpler queries
- ❌ More storage
- ❌ Update complexity
- ❌ Data redundancy

---

## Indexing Strategies

### Index Types

**1. Primary Index**
- On primary key
- Automatically created
- Unique and sorted

**2. Secondary Index**
- On non-primary key columns
- Manually created
- Improves query performance

**3. Composite Index**
- Multiple columns
- Order matters
- Left-prefix rule

**4. Unique Index**
- Enforces uniqueness
- Can be on multiple columns
- Prevents duplicates

**5. Partial Index**
- Index subset of rows
- Condition-based
- Smaller index size

### Index Design Principles

**1. Index Frequently Queried Columns**
- WHERE clause columns
- JOIN columns
- ORDER BY columns
- GROUP BY columns

**2. Consider Query Patterns**
- Analyze common queries
- Index for access patterns
- Balance read/write performance

**3. Composite Index Order**
- Most selective first
- Consider query patterns
- Left-prefix rule

**Example**:
```sql
-- Good: Most selective first
CREATE INDEX idx_user_date ON orders(user_id, order_date);

-- Query can use index
SELECT * FROM orders 
WHERE user_id = 123 AND order_date > '2024-01-01';

-- Query can use partial index (left-prefix)
SELECT * FROM orders WHERE user_id = 123;
```

**4. Avoid Over-Indexing**
- Each index adds write overhead
- Monitor index usage
- Remove unused indexes
- Balance performance

### Index Maintenance

**Monitor Index Usage**:
- Track index usage statistics
- Identify unused indexes
- Remove unnecessary indexes

**Rebuild Indexes**:
- Fragmented indexes
- After bulk operations
- Regular maintenance

---

## Query Optimization

### 1. Analyze Query Plans

**EXPLAIN Statement**:
```sql
EXPLAIN SELECT * FROM users 
WHERE email = 'john@example.com';
```

**Key Metrics**:
- Execution time
- Rows scanned
- Index usage
- Join algorithms
- Sort operations

### 2. Optimization Techniques

**Avoid SELECT ***:
```sql
-- Bad
SELECT * FROM large_table;

-- Good
SELECT id, name, email FROM large_table;
```

**Use LIMIT**:
```sql
-- Limit result set
SELECT * FROM orders 
ORDER BY order_date DESC 
LIMIT 100;
```

**Filter Early**:
```sql
-- Bad: Filter after join
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date > '2024-01-01';

-- Good: Filter before join
SELECT * FROM orders o
WHERE o.order_date > '2024-01-01'
JOIN customers c ON o.customer_id = c.id;
```

**Use Appropriate Joins**:
```sql
-- Prefer INNER JOIN when possible
SELECT * FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- Use EXISTS instead of IN for large subqueries
SELECT * FROM customers c
WHERE EXISTS (
  SELECT 1 FROM orders o 
  WHERE o.customer_id = c.id
);
```

**Avoid Functions in WHERE**:
```sql
-- Bad: Can't use index
SELECT * FROM orders 
WHERE YEAR(order_date) = 2024;

-- Good: Can use index
SELECT * FROM orders 
WHERE order_date >= '2024-01-01' 
AND order_date < '2025-01-01';
```

### 3. Subquery Optimization

**Correlated Subqueries**:
- Often slow
- Consider JOINs instead
- Use window functions when possible

**Example**:
```sql
-- Slow: Correlated subquery
SELECT * FROM orders o1
WHERE o1.amount > (
  SELECT AVG(amount) 
  FROM orders o2 
  WHERE o2.customer_id = o1.customer_id
);

-- Faster: Window function
SELECT * FROM (
  SELECT *, 
    AVG(amount) OVER (PARTITION BY customer_id) as avg_amount
  FROM orders
) WHERE amount > avg_amount;
```

---

## Performance Tuning

### 1. Connection Pooling

**Purpose**: Reuse database connections

**Benefits**:
- Reduce connection overhead
- Better resource utilization
- Improved performance

**Configuration**:
- Pool size
- Connection timeout
- Idle timeout
- Max connections

### 2. Caching Strategies

**Application-Level Caching**:
- Cache frequently accessed data
- Reduce database load
- Fast data access

**Database-Level Caching**:
- Query result cache
- Buffer pool
- Index cache

**Cache Invalidation**:
- Time-based expiration
- Event-based invalidation
- Cache-aside pattern

### 3. Partitioning

**Horizontal Partitioning (Sharding)**:
- Split table across multiple databases
- Distribute load
- Scale horizontally

**Vertical Partitioning**:
- Split columns across tables
- Reduce table size
- Improve performance

**Partition Strategies**:
- Range partitioning
- Hash partitioning
- List partitioning

### 4. Materialized Views

**Purpose**: Pre-computed query results

**Benefits**:
- Faster queries
- Reduced computation
- Aggregated data

**Trade-offs**:
- Storage overhead
- Maintenance cost
- Refresh complexity

---

## Database-Specific Optimizations

### PostgreSQL

**1. VACUUM and ANALYZE**:
```sql
VACUUM ANALYZE table_name;
```

**2. Index Types**:
- B-tree (default)
- Hash
- GiST
- GIN
- BRIN

**3. Query Planner**:
- Statistics collection
- Plan caching
- Parallel queries

### MySQL

**1. Storage Engines**:
- InnoDB: ACID, transactions
- MyISAM: Fast reads
- Memory: In-memory

**2. Query Cache**:
- Cache query results
- Fast repeated queries
- Configuration needed

**3. Index Hints**:
```sql
SELECT * FROM table_name 
USE INDEX (index_name)
WHERE condition;
```

---

## Monitoring and Profiling

### Key Metrics

**1. Performance Metrics**:
- Query execution time
- Throughput (QPS)
- Latency (p50, p95, p99)
- Resource utilization

**2. Database Metrics**:
- Connection count
- Cache hit ratio
- Index usage
- Lock contention

**3. System Metrics**:
- CPU usage
- Memory usage
- Disk I/O
- Network I/O

### Monitoring Tools

**Options**:
- Database-specific tools
- APM (Application Performance Monitoring)
- Custom dashboards
- Cloud monitoring

### Slow Query Logs

**Enable Slow Query Logging**:
- Identify slow queries
- Analyze execution plans
- Optimize problematic queries

---

## Best Practices

### 1. Design Phase

- Normalize appropriately
- Design for query patterns
- Plan for growth
- Consider access patterns

### 2. Indexing

- Index strategically
- Monitor index usage
- Avoid over-indexing
- Maintain indexes

### 3. Query Writing

- Write efficient queries
- Use EXPLAIN to analyze
- Avoid anti-patterns
- Test query performance

### 4. Performance

- Monitor continuously
- Profile regularly
- Optimize bottlenecks
- Test optimizations

### 5. Maintenance

- Regular VACUUM/ANALYZE
- Index maintenance
- Statistics updates
- Backup and recovery

---

## Common Anti-Patterns

### 1. N+1 Query Problem

**Problem**:
```sql
-- Fetch users
SELECT * FROM users;

-- Then for each user (N queries)
SELECT * FROM orders WHERE user_id = ?;
```

**Solution**: Use JOINs or batch queries

### 2. Missing Indexes

**Problem**: Queries scanning full table

**Solution**: Add appropriate indexes

### 3. Over-Normalization

**Problem**: Too many joins, slow queries

**Solution**: Strategic denormalization

### 4. Under-Normalization

**Problem**: Data redundancy, update issues

**Solution**: Normalize appropriately

---

## Key Takeaways

- Normalize to reduce redundancy, denormalize for performance
- Index strategically based on query patterns
- Optimize queries using EXPLAIN and best practices
- Monitor performance metrics continuously
- Use appropriate techniques for your database
- Balance design principles with performance needs
- Test and measure optimizations
- Regular maintenance is essential

