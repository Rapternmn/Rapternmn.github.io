+++
title = "Search Engine (Google Search)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 11
description = "Design a web search engine like Google. Covers web crawling, indexing, ranking, distributed search, and scaling to billions of web pages."
+++

---

## Problem Statement

Design a web search engine that crawls the web, indexes pages, and returns relevant search results. The system should handle billions of web pages and return results in milliseconds.

**Examples**: Google Search, Bing, DuckDuckGo

---

## Requirements Clarification

### Functional Requirements

1. **Web Crawling**: Crawl and discover web pages
2. **Indexing**: Build search index from crawled pages
3. **Search**: Return relevant search results
4. **Ranking**: Rank results by relevance
5. **Caching**: Cache popular search queries
6. **Freshness**: Keep index updated

### Non-Functional Requirements

- **Scale**: 
  - 1T (trillion) web pages indexed
  - 10B searches/day
  - 100K searches/second
  - Average 10 pages per search result
- **Latency**: < 100ms for search results
- **Freshness**: Update index daily
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Searches**: 10B searches/day = ~116K searches/second
- **Peak searches**: 3x = ~350K searches/second
- **Crawling**: Assume 1B pages crawled/day = ~11.6K pages/second

### Storage Estimates

- **Index Size**: 
  - Average page: 10 KB
  - 1T pages × 10 KB = 10 PB (raw)
  - Compressed index: ~1 PB
- **Inverted Index**: 
  - Average words per page: 500
  - Total words: 1T × 500 = 500T words
  - Index size: ~100 TB (compressed)

---

## API Design

### REST APIs

```
GET /api/v1/search
Query: ?q=query&page=1&limit=10
Response: {
  "query": "query",
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com/page",
      "snippet": "Page snippet...",
      "rank": 1
    }
  ],
  "total": 1000000,
  "time": 0.05  // seconds
}

GET /api/v1/suggestions
Query: ?q=quer
Response: {
  "suggestions": ["query", "question", "queries"]
}
```

---

## Database Design

### Schema

**Web Pages Table** (Distributed Storage):
```
pageId (PK): UUID
url: VARCHAR (unique)
title: VARCHAR
content: TEXT
lastCrawled: TIMESTAMP
pageRank: DECIMAL
```

**Inverted Index** (Distributed Index):
```
word (PK): VARCHAR
pageIds: ARRAY[UUID]  // Pages containing this word
positions: MAP[UUID, ARRAY[INT]]  // Word positions in each page
```

**Crawl Queue** (Message Queue):
```
url: VARCHAR
priority: INT
lastCrawled: TIMESTAMP
```

### Database Selection

**Web Pages**: **Distributed File System** (HDFS) or **Object Storage** (S3)
**Inverted Index**: **Distributed Search Engine** (Elasticsearch, Solr)
**Crawl Queue**: **Message Queue** (Kafka)
**Metadata**: **Distributed Database** (Cassandra)

---

## High-Level Design

### Architecture

```
Web → [Crawler] → [URL Frontier] → [Content Store]
                                    ↓
                            [Indexer] → [Inverted Index]
                                    ↓
                            [Ranker] → [Search Service]
                                    ↓
                            [Cache] → Client
```

### Components

1. **Web Crawler**: Crawl web pages
2. **URL Frontier**: Queue of URLs to crawl
3. **Content Store**: Store crawled page content
4. **Indexer**: Build inverted index
5. **Inverted Index**: Search index
6. **Ranker**: Rank search results
7. **Search Service**: Handle search queries
8. **Cache**: Cache search results

---

## Detailed Design

### Web Crawling

**Crawling Process**:
1. **Seed URLs**: Start with seed URLs
2. **Fetch Page**: Download page content
3. **Extract Links**: Extract links from page
4. **Add to Queue**: Add new URLs to crawl queue
5. **Store Content**: Store page content
6. **Repeat**: Continue until queue empty or limit reached

**Crawling Challenges**:
- **Politeness**: Respect robots.txt, rate limiting
- **Duplicate Detection**: Avoid crawling same page twice
- **Distributed Crawling**: Multiple crawlers, coordinate
- **Freshness**: Re-crawl pages periodically

**URL Frontier**:
- **Queue**: Queue of URLs to crawl
- **Priority**: Prioritize important URLs
- **Deduplication**: Avoid duplicate URLs
- **Distributed**: Multiple crawlers share queue

---

### Indexing

**Inverted Index**:
- **Structure**: Word → List of pages containing word
- **Example**:
  ```
  "python" → [page1, page5, page10]
  "tutorial" → [page1, page3, page5]
  ```

**Indexing Process**:
1. **Parse Content**: Extract words from page
2. **Tokenize**: Split into words
3. **Normalize**: Lowercase, remove punctuation
4. **Build Index**: Add page to inverted index for each word
5. **Store Positions**: Store word positions in page

**Distributed Indexing**:
- **Sharding**: Shard index by word (hash-based)
- **Replication**: Replicate index for availability
- **Merge**: Merge indexes from multiple crawlers

---

### Search Process

**Search Flow**:
1. **Query Processing**: Parse search query
2. **Tokenize**: Split query into words
3. **Lookup**: Lookup each word in inverted index
4. **Intersect**: Find pages containing all words
5. **Rank**: Rank results by relevance
6. **Return**: Return top N results

**Query Types**:
- **AND Query**: Pages containing all words
- **OR Query**: Pages containing any word
- **Phrase Query**: Exact phrase matching

---

### Ranking Algorithm

**Ranking Factors**:
1. **TF-IDF**: Term frequency, inverse document frequency
2. **PageRank**: Link-based ranking
3. **Freshness**: Recent pages ranked higher
4. **User Signals**: Click-through rate, dwell time
5. **Content Quality**: Content length, structure

**PageRank Algorithm**:
```
PR(A) = (1-d) + d × Σ(PR(T)/C(T))
```
- **PR(A)**: PageRank of page A
- **d**: Damping factor (0.85)
- **T**: Pages linking to A
- **C(T)**: Total number of outgoing links from page T

**Ranking Implementation**:
- **Simple**: Score-based ranking
- **Advanced**: ML models (neural networks)

---

### Distributed Search

**Challenges**:
- **Scale**: Billions of pages
- **Latency**: Return results in milliseconds
- **Availability**: Handle failures

**Solutions**:
1. **Index Sharding**: Shard index across multiple nodes
2. **Query Distribution**: Distribute query to all shards
3. **Result Aggregation**: Aggregate results from all shards
4. **Caching**: Cache popular queries
5. **Load Balancing**: Distribute queries across nodes

**Architecture**:
```
Query → Load Balancer → Search Nodes (Shards)
                        ↓
                    Aggregate Results
                        ↓
                    Rank & Return
```

---

### Caching Strategy

**Cache Levels**:
1. **Query Cache**: Cache entire search results
   - Key: Query string
   - TTL: 1 hour
   - Cache popular queries

2. **Index Cache**: Cache index lookups
   - Key: Word
   - TTL: 24 hours
   - Cache frequent words

3. **Page Cache**: Cache page content
   - Key: URL
   - TTL: 1 day
   - Cache popular pages

---

## Scalability

### Horizontal Scaling

- **Distributed Crawling**: Multiple crawlers
- **Distributed Indexing**: Multiple indexers
- **Distributed Search**: Multiple search nodes
- **Sharding**: Shard index by word or page

### Performance Optimization

- **Caching**: Cache search results and indexes
- **Compression**: Compress index and content
- **Parallel Processing**: Process queries in parallel
- **CDN**: Serve static assets

---

## Reliability

### High Availability

- **Multiple Search Nodes**: No single point of failure
- **Index Replication**: Replicate index across nodes
- **Geographic Distribution**: Multiple data centers

### Fault Tolerance

- **Crawler Failures**: Resume crawling from checkpoint
- **Indexer Failures**: Re-index from content store
- **Search Failures**: Route to healthy nodes

### Data Consistency

- **Index Updates**: Eventual consistency acceptable
- **Search Results**: Stale results acceptable (refresh periodically)

---

## Trade-offs

### Freshness vs Performance

- **Frequent Crawling**: Fresher index, higher cost
- **Less Frequent Crawling**: Staler index, lower cost

### Accuracy vs Latency

- **Complex Ranking**: More accurate, slower
- **Simple Ranking**: Less accurate, faster

### Storage vs Performance

- **Full Index**: Higher storage, faster search
- **Compressed Index**: Lower storage, slower search

---

## Extensions

### Additional Features

1. **Image Search**: Search by image
2. **Voice Search**: Voice queries
3. **Personalization**: Personalized results
4. **Autocomplete**: Query suggestions
5. **Spell Correction**: Correct spelling errors
6. **Related Searches**: Suggest related queries
7. **Search Analytics**: Track search patterns

---

## Key Takeaways

- **Web Crawling**: Distributed crawling with URL frontier
- **Inverted Index**: Efficient data structure for search
- **Distributed Search**: Shard index, aggregate results
- **Ranking**: Combine multiple signals (TF-IDF, PageRank)
- **Caching**: Cache search results and indexes
- **Scalability**: Horizontal scaling with sharding

---

## Related Topics

- **[Databases]({{< ref "../databases/_index.md" >}})** - Distributed databases and search engines
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed architecture
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Search result caching
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Scaling search

