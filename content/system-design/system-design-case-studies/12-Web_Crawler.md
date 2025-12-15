+++
title = "Web Crawler (Googlebot)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 12
description = "Design a distributed web crawler like Googlebot. Covers URL frontier, politeness, distributed crawling, deduplication, and scaling to crawl billions of web pages."
+++

---

## Problem Statement

Design a distributed web crawler that systematically browses the web, discovers new pages, and downloads content for indexing. The system should respect robots.txt, handle politeness, and scale to crawl billions of pages.

**Examples**: Googlebot, Bingbot, web crawlers for search engines

---

## Requirements Clarification

### Functional Requirements

1. **URL Discovery**: Discover new URLs from crawled pages
2. **Page Download**: Download page content (HTML, images, etc.)
3. **Politeness**: Respect robots.txt and rate limits
4. **Deduplication**: Avoid crawling same URL multiple times
5. **Distributed Crawling**: Multiple crawlers working in parallel
6. **Priority**: Prioritize important URLs
7. **Freshness**: Re-crawl pages periodically

### Non-Functional Requirements

- **Scale**: 
  - 1B URLs to crawl
  - 100M URLs/day crawl rate
  - Average 100 links per page
  - 10K crawler instances
- **Latency**: < 5 seconds per page download
- **Politeness**: Respect rate limits (1 request/second per domain)
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Crawl Rate**: 100M URLs/day = ~1,160 URLs/second
- **Peak Rate**: 3x = ~3,500 URLs/second
- **Links Discovered**: 100M pages × 100 links = 10B links/day
- **Unique URLs**: Assume 10% unique = 1B unique URLs/day

### Storage Estimates

- **Page Content**: Average 100 KB per page
- **Daily Storage**: 100M × 100 KB = 10 TB/day
- **Yearly Storage**: ~3.6 PB/year
- **URL Frontier**: 1B URLs × 100 bytes = 100 GB

---

## API Design

### Internal APIs

```
POST /api/v1/crawler/queue-url
Request: {
  "url": "https://example.com/page",
  "priority": 5,
  "depth": 2
}

GET /api/v1/crawler/get-url
Response: {
  "url": "https://example.com/page",
  "priority": 5
}

POST /api/v1/crawler/crawled
Request: {
  "url": "https://example.com/page",
  "status": "success",
  "links": ["url1", "url2", ...],
  "content": "..."
}
```

---

## Database Design

### Schema

**URL Frontier Table** (Distributed Queue - Kafka):
```
url: VARCHAR (PK)
priority: INT
depth: INT
discoveredAt: TIMESTAMP
lastCrawled: TIMESTAMP (nullable)
```

**Crawled URLs Table** (Cassandra):
```
url (PK): VARCHAR
status: VARCHAR (success, failed, blocked)
contentHash: VARCHAR (MD5/SHA256)
lastCrawled: TIMESTAMP
nextCrawl: TIMESTAMP
links: ARRAY[VARCHAR]
```

**Robots.txt Cache** (Redis):
```
domain (PK): VARCHAR
robotsTxt: TEXT
cachedAt: TIMESTAMP
```

**Domain Rate Limits** (Redis):
```
domain (PK): VARCHAR
lastRequest: TIMESTAMP
requestCount: INT
```

### Database Selection

**URL Frontier**: **Message Queue** (Kafka) - distributed queue
**Crawled URLs**: **Cassandra** - high write volume, time-series
**Robots.txt Cache**: **Redis** - fast lookups, TTL
**Domain Rate Limits**: **Redis** - fast updates, TTL

---

## High-Level Design

### Architecture

```
Seed URLs → [URL Frontier] → [Crawler Instances]
                                    ↓
                            [Download Pages]
                                    ↓
                            [Extract Links]
                                    ↓
                            [Deduplication]
                                    ↓
                            [Content Store]
                                    ↓
                            [Indexer]
```

### Components

1. **URL Frontier**: Queue of URLs to crawl
2. **Crawler Instances**: Multiple crawler workers
3. **Downloader**: Download page content
4. **Link Extractor**: Extract links from pages
5. **Deduplicator**: Check if URL already crawled
6. **Content Store**: Store crawled content
7. **Robots.txt Parser**: Parse and cache robots.txt
8. **Rate Limiter**: Enforce politeness

---

## Detailed Design

### URL Frontier (Queue)

**Purpose**: Queue of URLs waiting to be crawled

**Structure**:
- **Priority Queue**: Prioritize important URLs
- **Distributed**: Multiple queues for parallel processing
- **Deduplication**: Avoid duplicate URLs

**Priority Factors**:
1. **PageRank**: Higher PageRank = higher priority
2. **Depth**: Lower depth = higher priority
3. **Domain**: Important domains = higher priority
4. **Freshness**: Stale pages = higher priority

**Implementation**:
- **Kafka**: Distributed message queue
- **Partitioning**: Partition by domain (same domain → same partition)
- **Priority**: Use Kafka headers for priority

---

### Distributed Crawling

**Challenges**:
- **Coordination**: Multiple crawlers, avoid duplicate work
- **Load Balancing**: Distribute URLs across crawlers
- **Fault Tolerance**: Handle crawler failures

**Solutions**:
1. **URL Frontier**: Centralized queue (Kafka)
2. **Partitioning**: Partition by domain
3. **Consumer Groups**: Kafka consumer groups for parallel processing
4. **Checkpointing**: Save progress, resume on failure

**Crawler Architecture**:
```
Crawler Instance:
  1. Fetch URL from frontier
  2. Check robots.txt
  3. Check rate limit
  4. Download page
  5. Extract links
  6. Deduplicate links
  7. Add new URLs to frontier
  8. Store content
  9. Update crawled status
```

---

### Politeness & Rate Limiting

**Politeness Rules**:
1. **Robots.txt**: Respect robots.txt rules
2. **Rate Limiting**: Limit requests per domain (e.g., 1 request/second)
3. **Crawl Delay**: Respect crawl-delay directive
4. **User-Agent**: Identify crawler in User-Agent header

**Robots.txt Parsing**:
- **Cache**: Cache robots.txt per domain (Redis)
- **TTL**: Refresh robots.txt periodically (e.g., daily)
- **Parsing**: Parse rules (Allow, Disallow, Crawl-delay)

**Rate Limiting**:
- **Per Domain**: Track requests per domain (Redis)
- **Sliding Window**: Use sliding window counter
- **Enforcement**: Wait if rate limit exceeded

---

### Deduplication

**Why Deduplicate**:
- Same URL discovered from multiple pages
- Avoid redundant crawling
- Save bandwidth and storage

**Deduplication Strategies**:
1. **URL Normalization**: Normalize URLs (lowercase, remove fragments)
2. **Content Hash**: Check if content already crawled (MD5/SHA256)
3. **Bloom Filter**: Fast duplicate detection (probabilistic)
4. **Database Lookup**: Check in crawled URLs table

**URL Normalization**:
- Lowercase domain
- Remove default ports (80, 443)
- Remove fragments (#)
- Sort query parameters
- Remove trailing slashes

**Bloom Filter**:
- **Fast**: O(1) lookup
- **Memory Efficient**: Small memory footprint
- **False Positives**: Possible (acceptable)
- **False Negatives**: Not possible

---

### Link Extraction

**Link Extraction Process**:
1. **Parse HTML**: Parse HTML content
2. **Find Links**: Extract `<a href="...">` tags
3. **Resolve URLs**: Convert relative URLs to absolute
4. **Filter**: Filter invalid URLs (javascript:, mailto:, etc.)
5. **Normalize**: Normalize URLs

**URL Resolution**:
- **Relative URLs**: Resolve relative to base URL
- **Protocol**: Handle http/https
- **Domain**: Validate domain names

**Filtering**:
- **Protocol**: Only http/https
- **File Types**: Filter binary files (images, PDFs) if needed
- **Domains**: Filter specific domains if needed

---

### Content Storage

**Storage Options**:
1. **Object Storage** (S3): Store raw HTML
2. **Distributed File System** (HDFS): Store for batch processing
3. **Database**: Store metadata only

**Content Processing**:
- **Compression**: Compress content (gzip)
- **Deduplication**: Store content hash, avoid duplicates
- **Versioning**: Store multiple versions if needed

---

## Scalability

### Horizontal Scaling

- **Multiple Crawlers**: Scale crawler instances horizontally
- **URL Frontier Partitioning**: Partition by domain
- **Content Storage**: Distributed storage (S3/HDFS)

### Performance Optimization

- **Parallel Crawling**: Multiple crawlers in parallel
- **Connection Pooling**: Reuse HTTP connections
- **Caching**: Cache robots.txt, DNS lookups
- **Compression**: Compress stored content

---

## Reliability

### High Availability

- **Multiple Crawler Instances**: No single point of failure
- **URL Frontier Replication**: Kafka replication
- **Content Storage Replication**: S3/HDFS replication

### Fault Tolerance

- **Crawler Failures**: URLs return to queue, retry
- **Network Failures**: Retry with exponential backoff
- **Checkpointing**: Save progress, resume on restart

### Data Consistency

- **URL Frontier**: Eventual consistency acceptable
- **Crawled URLs**: Strong consistency needed (avoid duplicates)

---

## Trade-offs

### Freshness vs Cost

- **Frequent Crawling**: Fresher content, higher cost
- **Less Frequent Crawling**: Staler content, lower cost

### Politeness vs Speed

- **Strict Politeness**: Slower crawling, better reputation
- **Aggressive Crawling**: Faster crawling, risk of blocking

### Storage vs Deduplication

- **Store All**: Higher storage, simpler
- **Deduplicate**: Lower storage, more complex

---

## Extensions

### Additional Features

1. **Incremental Crawling**: Only crawl changed pages
2. **Focused Crawling**: Crawl specific topics/domains
3. **Deep Web Crawling**: Crawl behind forms/logins
4. **JavaScript Rendering**: Render JavaScript pages
5. **Image Crawling**: Crawl and index images
6. **Sitemap Support**: Use sitemap.xml for discovery
7. **Crawl Analytics**: Track crawl statistics

---

## Key Takeaways

- **URL Frontier**: Centralized queue for URL management
- **Distributed Crawling**: Multiple crawlers, coordinated via queue
- **Politeness**: Respect robots.txt and rate limits
- **Deduplication**: Avoid redundant crawling
- **Scalability**: Horizontal scaling with distributed queue
- **Fault Tolerance**: Checkpointing and retry mechanisms

---

## Related Topics

- **[Search Engine]({{< ref "11-Search_Engine.md" >}})** - Using crawled content for search
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - URL frontier queue
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed architecture
- **[Rate Limiter]({{< ref "3-Rate_Limiter.md" >}})** - Rate limiting for politeness

