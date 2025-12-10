+++
title = "Search Databases"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Search Databases: Elasticsearch - full-text search, indexing, search capabilities, and when to use search databases."
+++

---

## Introduction

Search databases are specialized for full-text search, indexing, and complex search queries. They enable fast text search, faceted search, and analytics over large volumes of text data.

---

## What are Search Databases?

**Search Database**:
- Optimized for full-text search
- Inverted index structure
- Fast text queries
- Relevance scoring
- Faceted search support

**Key Characteristics**:
- **Full-Text Search**: Search within text content
- **Inverted Index**: Fast text lookups
- **Relevance Scoring**: Rank results by relevance
- **Analytics**: Aggregations and analytics
- **Real-Time**: Near real-time indexing

---

## Elasticsearch

### Overview

**Elasticsearch** is a distributed search and analytics engine built on Apache Lucene.

**Key Features**:
- Distributed search engine
- Real-time indexing
- Full-text search
- Analytics and aggregations
- RESTful API
- Horizontal scaling

### Core Concepts

**1. Index**
- Collection of documents (like database)
- Can have multiple indices
- Sharded and replicated

**2. Document**
- JSON object (like row)
- Stored in index
- Has unique ID

**3. Type** (Deprecated in 7.x+)
- Previously used for categorization
- Now single type per index

**4. Field**
- Property of document (like column)
- Has data type
- Can be analyzed or not

**5. Mapping**
- Schema definition
- Field types and analyzers
- Index settings

### Data Model Example

**Index**: `products`
**Document**:
```json
{
  "id": 1,
  "name": "Laptop Computer",
  "description": "High-performance laptop with 16GB RAM",
  "price": 999.99,
  "category": "Electronics",
  "tags": ["laptop", "computer", "electronics"],
  "in_stock": true
}
```

---

## Elasticsearch Features

### 1. Full-Text Search

**Basic Search**:
```json
GET /products/_search
{
  "query": {
    "match": {
      "description": "laptop computer"
    }
  }
}
```

**Multi-Match**:
```json
{
  "query": {
    "multi_match": {
      "query": "laptop",
      "fields": ["name^2", "description"]
    }
  }
}
```

### 2. Filtering

**Term Filter**:
```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "description": "laptop" } }
      ],
      "filter": [
        { "term": { "in_stock": true } },
        { "range": { "price": { "gte": 500, "lte": 1500 } } }
      ]
    }
  }
}
```

### 3. Aggregations

**Terms Aggregation**:
```json
{
  "aggs": {
    "categories": {
      "terms": {
        "field": "category",
        "size": 10
      }
    },
    "price_stats": {
      "stats": {
        "field": "price"
      }
    }
  }
}
```

### 4. Faceted Search

**Facets**:
- Category filters
- Price ranges
- Brand filters
- Multiple dimensions

### 5. Relevance Scoring

**TF-IDF Scoring**:
- Term frequency
- Inverse document frequency
- Field length normalization
- Custom scoring functions

---

## Use Cases

### 1. Full-Text Search

**Why Elasticsearch**:
- Fast text search
- Relevance ranking
- Fuzzy matching
- Multi-language support

**Examples**:
- Product search
- Document search
- Content search
- Knowledge bases

### 2. Log Analytics

**Why Elasticsearch**:
- Fast log search
- Real-time indexing
- Aggregations
- Visualization (Kibana)

**Examples**:
- Application logs
- System logs
- Security logs
- Error tracking

### 3. E-Commerce Search

**Why Elasticsearch**:
- Product search
- Faceted navigation
- Autocomplete
- Recommendations

**Examples**:
- Product catalogs
- Search functionality
- Filtering
- Sorting

### 4. Analytics and Monitoring

**Why Elasticsearch**:
- Time-series data
- Aggregations
- Dashboards
- Real-time metrics

**Examples**:
- Application monitoring
- Business metrics
- User analytics
- Performance monitoring

---

## When to Use Elasticsearch

### Good Fit

✅ **Full-Text Search**: Text search requirements
✅ **Log Analytics**: Log search and analysis
✅ **Real-Time Search**: Near real-time indexing
✅ **Complex Queries**: Advanced search needs
✅ **Analytics**: Aggregations and analytics
✅ **Large Text Data**: Large volumes of text

### Not Ideal For

❌ **Simple Key-Value**: Better with key-value stores
❌ **ACID Transactions**: No transaction support
❌ **Complex Joins**: Limited relationship support
❌ **Frequent Updates**: Better for append-heavy
❌ **Small Datasets**: Overhead not justified

---

## Elasticsearch Architecture

### Cluster and Nodes

**Cluster**:
- Collection of nodes
- Shared cluster name
- Automatic node discovery

**Node Types**:
- **Master Node**: Cluster management
- **Data Node**: Store data
- **Ingest Node**: Pre-process data
- **Coordinating Node**: Route requests

### Sharding

**Primary Shards**:
- Data divided into shards
- Shards distributed across nodes
- Horizontal scaling

**Replica Shards**:
- Copies of primary shards
- High availability
- Read scaling

### Indexing Process

**1. Document Ingestion**
- Documents sent to Elasticsearch
- Assigned to shard
- Stored in index

**2. Analysis**
- Text analyzed
- Tokens created
- Inverted index updated

**3. Storage**
- Document stored
- Index updated
- Searchable immediately

---

## Query Types

### 1. Match Query

**Full-Text Search**:
```json
{
  "query": {
    "match": {
      "title": "laptop computer"
    }
  }
}
```

### 2. Term Query

**Exact Match**:
```json
{
  "query": {
    "term": {
      "status": "active"
    }
  }
}
```

### 3. Range Query

**Range Filtering**:
```json
{
  "query": {
    "range": {
      "price": {
        "gte": 100,
        "lte": 500
      }
    }
  }
}
```

### 4. Bool Query

**Complex Logic**:
```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "laptop" } }
      ],
      "should": [
        { "match": { "description": "gaming" } }
      ],
      "must_not": [
        { "term": { "status": "out_of_stock" } }
      ],
      "filter": [
        { "range": { "price": { "gte": 500 } } }
      ]
    }
  }
}
```

### 5. Aggregation Queries

**Analytics**:
```json
{
  "aggs": {
    "avg_price": {
      "avg": { "field": "price" }
    },
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 100 },
          { "from": 100, "to": 500 },
          { "from": 500 }
        ]
      }
    }
  }
}
```

---

## Indexing and Mapping

### Mapping Definition

```json
PUT /products
{
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "analyzer": "standard"
      },
      "price": {
        "type": "double"
      },
      "category": {
        "type": "keyword"
      },
      "description": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      }
    }
  }
}
```

### Field Types

**Text**:
- Full-text search
- Analyzed
- Tokenized

**Keyword**:
- Exact match
- Not analyzed
- Filtering, sorting

**Numeric**:
- long, integer, short, byte
- double, float
- Range queries

**Date**:
- Date/time values
- Range queries
- Format support

---

## Best Practices

### 1. Index Design

- Design indices for use case
- Appropriate shard count
- Replica configuration
- Index lifecycle management

### 2. Mapping

- Choose right field types
- Use text for search
- Use keyword for exact match
- Configure analyzers appropriately

### 3. Query Optimization

- Use filters for exact matches
- Use match for full-text
- Combine queries efficiently
- Use aggregations wisely

### 4. Performance

- Monitor cluster health
- Optimize shard sizes
- Use appropriate refresh interval
- Monitor query performance

---

## Elasticsearch Stack

### ELK Stack

**Elasticsearch**: Search and analytics
**Logstash**: Data processing pipeline
**Kibana**: Visualization and dashboards

### Beats

**Filebeat**: Log shipping
**Metricbeat**: Metrics collection
**Packetbeat**: Network monitoring

---

## Comparison with Other Databases

### vs Relational Databases

**Elasticsearch Advantages**:
- Full-text search
- Relevance scoring
- Horizontal scaling
- Analytics capabilities

**Relational Advantages**:
- ACID transactions
- Complex relationships
- Strong consistency
- Mature ecosystem

### vs MongoDB

**Elasticsearch Advantages**:
- Better search capabilities
- Relevance scoring
- Analytics focus
- Log analytics

**MongoDB Advantages**:
- More flexible data model
- Better for general purpose
- Easier to use
- More features

---

## Key Takeaways

- Search databases optimize for full-text search
- Elasticsearch provides powerful search and analytics
- Good for text search, log analytics, and e-commerce
- Inverted index enables fast text queries
- Relevance scoring ranks results
- Aggregations enable analytics
- Horizontal scaling built-in
- Choose for search-heavy applications

