+++
title = "System Components Case Studies"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 13
description = "Real-world case studies combining multiple system components: URL Shortener, Chat System, Video Streaming, Social Media Feed, and E-commerce Platform."
+++

---

## Introduction

This document presents real-world case studies that demonstrate how multiple system design components work together to build scalable, reliable systems. Each case study covers requirements, component selection, architecture, and trade-offs.

---

## Case Study 1: URL Shortener (TinyURL)

### Requirements

- **Scale**: 100M URLs/day, 100:1 read/write ratio
- **Latency**: < 200ms for URL resolution
- **Availability**: 99.9%
- **Features**: Short URL generation, URL resolution, analytics

---

### Component Selection

#### Load Balancing
- **Choice**: Layer 7 load balancer (ALB/NGINX)
- **Rationale**: HTTP-based, need content-based routing
- **Features**: Health checks, session affinity not needed

#### Caching Strategy
- **Choice**: Multi-layer caching
  - CDN for static assets
  - Redis for URL mappings (hot URLs)
- **Rationale**: 
  - 100:1 read/write ratio (read-heavy)
  - Need fast URL resolution
  - Most URLs accessed infrequently

#### Database Design
- **Choice**: 
  - SQL database for metadata (PostgreSQL)
  - NoSQL for URL mappings (Cassandra/DynamoDB)
- **Rationale**: 
  - Need strong consistency for metadata
  - High write volume for URL mappings
  - Can tolerate eventual consistency for mappings

#### CDN Usage
- **Choice**: CloudFront/Cloudflare
- **Rationale**: 
  - Serve static assets (JS, CSS)
  - Global distribution
  - DDoS protection

---

### Architecture

```
Client → CDN → Load Balancer → API Gateway
                              ↓
                    [URL Service] → Redis Cache
                              ↓
                    [Database: PostgreSQL + Cassandra]
                              ↓
                    [Analytics Service] → Message Queue
```

---

### Component Interactions

1. **URL Generation**: 
   - Client → API Gateway → URL Service → Database
   - Generate short code, store mapping

2. **URL Resolution**: 
   - Client → CDN → Load Balancer → URL Service
   - Check Redis cache → Database if miss
   - Return redirect

3. **Analytics**: 
   - URL Service → Message Queue → Analytics Service
   - Async processing for click tracking

---

### Scalability Considerations

- **Horizontal Scaling**: Stateless URL service, add instances
- **Database Sharding**: Shard by short code hash
- **Caching**: Cache hot URLs in Redis
- **CDN**: Offload static assets

---

### Trade-offs Made

- **Consistency**: Eventual consistency for URL mappings acceptable
- **Caching**: Cache hit rate critical for performance
- **Database**: Dual database for different consistency needs

---

## Case Study 2: Chat System (WhatsApp/Telegram)

### Requirements

- **Scale**: 1B users, 50B messages/day
- **Latency**: < 100ms message delivery
- **Availability**: 99.9%
- **Features**: Real-time messaging, group chats, media sharing

---

### Component Selection

#### Message Queues
- **Choice**: Kafka/Pulsar for message streaming
- **Rationale**: 
  - High throughput (50B messages/day)
  - Need message ordering
  - Real-time delivery

#### Service Discovery
- **Choice**: Service mesh (Istio) or Consul
- **Rationale**: 
  - Many microservices
  - Dynamic scaling
  - Service-to-service communication

#### Caching
- **Choice**: Redis for online users, message cache
- **Rationale**: 
  - Fast user presence lookup
  - Cache recent messages
  - Real-time requirements

#### Database
- **Choice**: 
  - Cassandra for messages (write-heavy, time-series)
  - Redis for online users
  - PostgreSQL for user metadata
- **Rationale**: 
  - Messages are write-heavy
  - Need time-series storage
  - User metadata needs consistency

---

### Architecture

```
Client → Load Balancer → API Gateway
                        ↓
        [Message Service] → Kafka → [Delivery Service]
                        ↓
        [Presence Service] → Redis
                        ↓
        [Database: Cassandra + PostgreSQL]
```

---

### Component Interactions

1. **Send Message**: 
   - Client → API Gateway → Message Service
   - Store in Cassandra
   - Publish to Kafka
   - Delivery Service → Push to recipient

2. **Receive Message**: 
   - Delivery Service → Check user presence (Redis)
   - Push via WebSocket if online
   - Store for offline delivery

3. **User Presence**: 
   - User online → Update Redis
   - Heartbeat mechanism
   - Service discovery for user location

---

### Scalability Considerations

- **Message Queues**: Partition by user/conversation
- **Database Sharding**: Shard messages by conversation ID
- **Service Discovery**: Dynamic service registration
- **Caching**: Cache online users and recent messages

---

### Trade-offs Made

- **Consistency**: Eventual consistency for message delivery
- **Latency**: Prioritize low latency over strong consistency
- **Storage**: Time-series database for messages

---

## Case Study 3: Video Streaming Platform (YouTube/Netflix)

### Requirements

- **Scale**: 1B users, 1B hours watched/day
- **Latency**: < 2s video start time
- **Availability**: 99.99%
- **Features**: Video upload, transcoding, streaming, recommendations

---

### Component Selection

#### CDN Architecture
- **Choice**: Multi-tier CDN (CloudFront/Akamai)
- **Rationale**: 
  - Global video distribution
  - Reduce origin load
  - Low latency worldwide

#### Caching Strategies
- **Choice**: 
  - CDN for video files
  - Redis for metadata, recommendations
  - Application cache for user sessions
- **Rationale**: 
  - Videos are large, cache at edge
  - Metadata accessed frequently
  - Personalization data needs fast access

#### Load Balancing
- **Choice**: Layer 7 load balancer with geographic routing
- **Rationale**: 
  - Route to nearest data center
  - Content-based routing
  - Health-aware routing

#### Message Queues
- **Choice**: Kafka for video processing pipeline
- **Rationale**: 
  - Async video transcoding
  - High throughput
  - Event streaming

---

### Architecture

```
User → CDN (Edge) → Load Balancer → API Gateway
                                    ↓
                    [Video Service] → [Transcoding Queue]
                                    ↓
                    [Metadata Service] → Redis Cache
                                    ↓
                    [Storage: Object Storage + Database]
```

---

### Component Interactions

1. **Video Upload**: 
   - Client → API Gateway → Video Service
   - Store in object storage
   - Publish to transcoding queue
   - Async transcoding

2. **Video Streaming**: 
   - Client → CDN (check cache)
   - CDN → Origin if miss
   - Serve video chunks
   - Adaptive bitrate streaming

3. **Recommendations**: 
   - User → API Gateway → Recommendation Service
   - Check Redis cache
   - Generate/return recommendations

---

### Scalability Considerations

- **CDN**: Primary scaling mechanism for video delivery
- **Object Storage**: Scalable storage for video files
- **Transcoding**: Distributed transcoding workers
- **Caching**: Multi-layer caching (CDN, application, database)

---

### Trade-offs Made

- **Cost**: CDN costs vs origin bandwidth
- **Latency**: Cache hit rate critical
- **Storage**: Object storage for videos, database for metadata

---

## Case Study 4: Social Media Feed (Twitter/Facebook)

### Requirements

- **Scale**: 500M users, 500M posts/day, 100B reads/day
- **Latency**: < 200ms feed generation
- **Availability**: 99.9%
- **Features**: Post creation, feed generation, real-time updates

---

### Component Selection

#### Caching Strategies
- **Choice**: 
  - Redis for user timelines (pre-computed)
  - CDN for media
  - Application cache for user data
- **Rationale**: 
  - Read-heavy (100B reads/day)
  - Pre-compute feeds
  - Fast feed generation

#### Message Queues
- **Choice**: Kafka for event streaming
- **Rationale**: 
  - Fan-out to followers
  - Real-time updates
  - High throughput

#### Database Sharding
- **Choice**: Shard by user ID
- **Rationale**: 
  - Distribute load
  - User data isolation
  - Scalability

#### Real-time Updates
- **Choice**: WebSocket + Pub/Sub
- **Rationale**: 
  - Real-time feed updates
  - Push notifications
  - Low latency

---

### Architecture

```
Client → Load Balancer → API Gateway
                        ↓
        [Post Service] → Kafka → [Feed Service]
                        ↓
        [Feed Service] → Redis (Pre-computed Feeds)
                        ↓
        [Database: Sharded by User ID]
                        ↓
        [Real-time: WebSocket + Pub/Sub]
```

---

### Component Interactions

1. **Create Post**: 
   - Client → API Gateway → Post Service
   - Store in database
   - Publish to Kafka
   - Fan-out to followers' feeds

2. **Get Feed**: 
   - Client → API Gateway → Feed Service
   - Check Redis for pre-computed feed
   - Return feed

3. **Real-time Updates**: 
   - New post → Kafka → Feed Service
   - Update Redis feeds
   - Push via WebSocket to online followers

---

### Scalability Considerations

- **Feed Pre-computation**: Pre-compute feeds in background
- **Database Sharding**: Shard by user ID
- **Caching**: Cache pre-computed feeds
- **Message Queues**: Fan-out to followers

---

### Trade-offs Made

- **Consistency**: Eventual consistency for feeds acceptable
- **Storage**: Trade storage for speed (pre-computed feeds)
- **Complexity**: Complex fan-out logic

---

## Case Study 5: E-commerce Platform

### Requirements

- **Scale**: 10M users, 1M orders/day
- **Latency**: < 500ms for product pages
- **Availability**: 99.9%
- **Features**: Product catalog, shopping cart, checkout, inventory, payments

---

### Component Selection

#### API Gateway
- **Choice**: Kong/AWS API Gateway
- **Rationale**: 
  - Multiple microservices
  - Unified API interface
  - Authentication, rate limiting

#### Service Discovery
- **Choice**: Kubernetes service discovery or Consul
- **Rationale**: 
  - Microservices architecture
  - Dynamic service registration
  - Health-aware routing

#### Caching
- **Choice**: 
  - Redis for shopping carts, sessions
  - CDN for product images
  - Application cache for product catalog
- **Rationale**: 
  - Shopping carts need fast access
  - Product images are static
  - Catalog accessed frequently

#### Message Queues
- **Choice**: RabbitMQ/Kafka
- **Rationale**: 
  - Order processing
  - Inventory updates
  - Email notifications
  - Async processing

#### Database
- **Choice**: 
  - PostgreSQL for orders, users (ACID)
  - Redis for carts, sessions
  - Search database (Elasticsearch) for products
- **Rationale**: 
  - Orders need strong consistency
  - Carts can be eventually consistent
  - Product search needs full-text search

---

### Architecture

```
Client → CDN → Load Balancer → API Gateway
                                ↓
        [Product Service] → [Search: Elasticsearch]
        [Cart Service] → Redis
        [Order Service] → PostgreSQL
        [Payment Service] → Message Queue
        [Inventory Service] → Database
```

---

### Component Interactions

1. **Browse Products**: 
   - Client → API Gateway → Product Service
   - Search in Elasticsearch
   - Cache results
   - Serve images from CDN

2. **Add to Cart**: 
   - Client → API Gateway → Cart Service
   - Store in Redis
   - Session-based cart

3. **Checkout**: 
   - Client → API Gateway → Order Service
   - Create order in PostgreSQL
   - Publish to message queue
   - Process payment
   - Update inventory

4. **Inventory Management**: 
   - Order Service → Message Queue → Inventory Service
   - Update inventory
   - Handle race conditions

---

### Scalability Considerations

- **Microservices**: Independent scaling per service
- **Database**: Read replicas for product catalog
- **Caching**: Multi-layer caching
- **Message Queues**: Async processing for non-critical paths

---

### Trade-offs Made

- **Consistency**: Strong consistency for orders, eventual for carts
- **Performance**: Caching for product catalog
- **Complexity**: Microservices add operational complexity

---

## Key Takeaways

- **Component Selection**: Choose based on requirements and trade-offs
- **Multiple Components**: Systems use multiple components together
- **Caching**: Critical for read-heavy systems
- **Message Queues**: Enable async processing and decoupling
- **Database Choice**: Different databases for different needs
- **Scalability**: Horizontal scaling with stateless services
- **Trade-offs**: Every design involves trade-offs

---

## Common Patterns Across Case Studies

1. **Multi-Layer Caching**: CDN → Application Cache → Database Cache
2. **Load Balancing**: Distribute traffic across instances
3. **Message Queues**: Async processing and event streaming
4. **Database Sharding**: Distribute data for scalability
5. **Service Discovery**: Dynamic service location
6. **API Gateway**: Unified API interface

---

## Related Topics

- **[System Components Overview]({{< ref "1-System_Components_Overview.md" >}})** - Component selection
- **[Load Balancing]({{< ref "2-Load_Balancing.md" >}})** - Traffic distribution
- **[Caching Strategies]({{< ref "3-Caching_Strategies.md" >}})** - Caching patterns
- **[Message Queues & Message Brokers]({{< ref "5-Message_Queues_Message_Brokers.md" >}})** - Async processing
- **[Scalability Patterns]({{< ref "9-Scalability_Patterns.md" >}})** - Scaling strategies

