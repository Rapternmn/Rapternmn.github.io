+++
title = "Document Databases"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Document Databases: MongoDB - document storage, schema flexibility, use cases, and when to use document databases."
+++

---

## Introduction

Document databases store data in flexible, JSON-like documents rather than rigid tables. They provide schema flexibility, horizontal scaling, and are ideal for applications with evolving data structures.

---

## What are Document Databases?

**Document Database**:
- Store data as documents (JSON, BSON, XML)
- Schema-on-read approach
- Flexible, nested structures
- No fixed schema required
- Document-oriented queries

**Key Characteristics**:
- **Flexible Schema**: Documents can vary
- **Nested Data**: Complex structures in single document
- **Horizontal Scaling**: Sharding support
- **Rich Queries**: Query by document fields
- **No Joins**: Embed related data

---

## MongoDB

### Overview

**MongoDB** is a popular document database that stores data in BSON (Binary JSON) format.

**Key Features**:
- Document-based storage
- Flexible schema
- Horizontal scaling (sharding)
- Rich query language
- Indexing support
- Aggregation framework
- Replication

### Data Model

**Collections and Documents**:
- **Database**: Container for collections
- **Collection**: Group of documents (like table)
- **Document**: JSON-like record (like row)

**Example Document**:
```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "zip": "10001"
  },
  "hobbies": ["reading", "coding"],
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

### Key Features

**1. Flexible Schema**
- Documents can have different structures
- Add fields without migration
- Schema evolution friendly

**2. Rich Queries**
- Query by any field
- Complex conditions
- Aggregation pipeline
- Full-text search

**3. Indexing**
- Single field indexes
- Compound indexes
- Text indexes
- Geospatial indexes

**4. Aggregation Framework**
- Powerful data processing
- Pipeline-based operations
- Group, filter, transform
- Analytics capabilities

**5. Replication**
- Replica sets
- Automatic failover
- Read scaling
- High availability

**6. Sharding**
- Horizontal scaling
- Distribute data across shards
- Automatic balancing
- Scale to petabytes

---

## MongoDB Use Cases

### 1. Content Management

**Why MongoDB**:
- Flexible content structures
- Nested content hierarchies
- Easy schema evolution
- Rich media metadata

**Examples**:
- CMS platforms
- Blog systems
- Media libraries
- Documentation systems

### 2. User Profiles and Catalogs

**Why MongoDB**:
- Variable user attributes
- Product catalogs with varying fields
- Nested product information
- Easy updates

**Examples**:
- User management
- Product catalogs
- E-commerce platforms
- Social profiles

### 3. Real-Time Analytics

**Why MongoDB**:
- Fast writes
- Flexible data structures
- Aggregation framework
- Time-series data

**Examples**:
- IoT data
- Event tracking
- Log aggregation
- Real-time dashboards

### 4. Mobile Applications

**Why MongoDB**:
- JSON-native format
- Flexible schema
- Offline sync support
- Mobile SDKs

**Examples**:
- Mobile backends
- Offline-first apps
- Sync services

---

## When to Use MongoDB

### Good Fit

✅ **Schema Evolution**: Frequently changing data structures
✅ **Nested Data**: Complex, hierarchical data
✅ **Horizontal Scaling**: Need to scale out
✅ **Fast Development**: Rapid prototyping
✅ **JSON Data**: Working with JSON/API data
✅ **Content Management**: Flexible content structures

### Not Ideal For

❌ **Complex Transactions**: Multi-document transactions limited
❌ **Strong Consistency**: Eventual consistency model
❌ **Complex Joins**: No native join support
❌ **Heavy Analytics**: Better suited for OLTP
❌ **Small Datasets**: Overhead not justified

---

## MongoDB Architecture

### Components

**1. Mongod (Database Server)**
- Core database process
- Handles storage and queries
- Supports replication and sharding

**2. Mongos (Shard Router)**
- Routes queries to shards
- Handles shard balancing
- Query coordination

**3. Config Servers**
- Store cluster metadata
- Shard configuration
- Replica set configuration

### Replica Set

**Architecture**:
- Primary node (writes)
- Secondary nodes (replicas)
- Automatic failover
- Read scaling

**Benefits**:
- High availability
- Data redundancy
- Read scaling
- Automatic recovery

### Sharding

**Architecture**:
- Data distributed across shards
- Shard key determines distribution
- Automatic balancing
- Horizontal scaling

**Shard Key Selection**:
- Even distribution
- Query patterns
- Avoid hotspots
- Cardinality considerations

---

## MongoDB Operations

### CRUD Operations

**Create**:
```javascript
db.users.insertOne({
  name: "John",
  email: "john@example.com",
  age: 30
});

db.users.insertMany([...]);
```

**Read**:
```javascript
// Find one
db.users.findOne({ email: "john@example.com" });

// Find many
db.users.find({ age: { $gte: 18 } });

// Projection
db.users.find({}, { name: 1, email: 1 });
```

**Update**:
```javascript
// Update one
db.users.updateOne(
  { email: "john@example.com" },
  { $set: { age: 31 } }
);

// Update many
db.users.updateMany(
  { age: { $lt: 18 } },
  { $set: { status: "minor" } }
);
```

**Delete**:
```javascript
db.users.deleteOne({ email: "john@example.com" });
db.users.deleteMany({ status: "inactive" });
```

### Aggregation Pipeline

```javascript
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: {
      _id: "$customer_id",
      total: { $sum: "$amount" },
      count: { $sum: 1 }
    }
  },
  { $sort: { total: -1 } },
  { $limit: 10 }
]);
```

---

## Indexing Strategies

### Index Types

**1. Single Field Index**
```javascript
db.users.createIndex({ email: 1 });
```

**2. Compound Index**
```javascript
db.users.createIndex({ name: 1, age: -1 });
```

**3. Text Index**
```javascript
db.articles.createIndex({ title: "text", content: "text" });
```

**4. Geospatial Index**
```javascript
db.places.createIndex({ location: "2dsphere" });
```

### Index Best Practices

- Index frequently queried fields
- Consider query patterns
- Compound indexes for multi-field queries
- Monitor index usage
- Avoid over-indexing

---

## Data Modeling Patterns

### 1. Embedding vs Referencing

**Embedding** (One-to-Few):
- Related data in same document
- Fast reads
- Atomic updates
- Limited document size

**Referencing** (One-to-Many):
- Store references (ObjectIds)
- Separate collections
- More flexible
- Requires joins

### 2. Schema Design Patterns

**Pattern 1: Embedded Document**
```json
{
  "user": {
    "name": "John",
    "address": {
      "street": "123 Main St",
      "city": "New York"
    }
  }
}
```

**Pattern 2: Array of Embedded Documents**
```json
{
  "user": "John",
  "orders": [
    { "product": "Book", "quantity": 2 },
    { "product": "Pen", "quantity": 5 }
  ]
}
```

**Pattern 3: Reference Pattern**
```json
{
  "_id": ObjectId("..."),
  "user_id": ObjectId("..."),
  "product_id": ObjectId("..."),
  "quantity": 2
}
```

---

## Best Practices

### 1. Schema Design

- Embed for one-to-few relationships
- Reference for one-to-many
- Consider read/write patterns
- Plan for growth

### 2. Indexing

- Index frequently queried fields
- Use compound indexes wisely
- Monitor index performance
- Remove unused indexes

### 3. Performance

- Use projections to limit data
- Use aggregation for complex queries
- Monitor slow queries
- Optimize shard keys

### 4. Security

- Enable authentication
- Use role-based access control
- Encrypt data in transit
- Regular security updates

---

## Comparison with Relational Databases

### Advantages

✅ **Flexible Schema**: Easy schema evolution
✅ **Horizontal Scaling**: Better for scale-out
✅ **JSON Native**: Natural fit for APIs
✅ **Fast Development**: Rapid prototyping
✅ **Nested Data**: Store complex structures

### Disadvantages

❌ **No Joins**: Application-level joins
❌ **Weaker Consistency**: Eventual consistency
❌ **Limited Transactions**: Multi-document transactions newer
❌ **Storage Overhead**: Document structure overhead
❌ **Query Complexity**: Some queries more complex

---

## Key Takeaways

- Document databases provide flexible, schema-less storage
- MongoDB is popular document database with rich features
- Good for evolving schemas and nested data
- Horizontal scaling through sharding
- Rich query language and aggregation framework
- Choose for flexible data structures and scale-out needs
- Design with embedding vs referencing in mind
- Index strategically for performance

