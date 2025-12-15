+++
title = "Social Media Feed (Twitter/Facebook)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 6
description = "Design a social media feed system like Twitter or Facebook. Covers feed generation, fan-out patterns, caching strategies, and ranking algorithms."
+++

---

## Problem Statement

Design a social media feed system that displays personalized content from users you follow. The system should generate feeds efficiently and handle millions of users posting and viewing content.

**Examples**: Twitter, Facebook News Feed, Instagram, LinkedIn

---

## Requirements Clarification

### Functional Requirements

1. **Post Creation**: Users can create posts (text, images, videos)
2. **Feed Generation**: Generate personalized feed for each user
3. **Timeline**: Show posts in chronological or ranked order
4. **Interactions**: Like, comment, share, retweet
5. **Follow/Unfollow**: Follow other users
6. **Hashtags**: Support hashtags and trending topics

### Non-Functional Requirements

- **Scale**: 
  - 1B users
  - 500M daily active users
  - 100M posts/day
  - Average 200 followers per user
  - Average 5 posts viewed per user session
- **Latency**: < 200ms to generate feed
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Posts**: 100M posts/day = ~1,160 posts/second
- **Peak posts**: 3x = ~3,500 posts/second
- **Feed Reads**: 500M DAU × 5 feeds/session × 10 sessions/day = 25B feed reads/day
- **Feed Reads/sec**: ~290K reads/second
- **Peak reads**: 3x = ~870K reads/second

### Storage Estimates

- **Post Size**: Average 1 KB (text + metadata)
- **Daily Posts**: 100M × 1 KB = 100 GB/day
- **Yearly Posts**: ~36 TB/year
- **Media**: Assume 50% posts have media (images/videos)
- **Media Storage**: Much larger (needs CDN/object storage)

### Fan-out Calculations

- **Average Followers**: 200 per user
- **Fan-out per Post**: 200 writes (to followers' feeds)
- **Total Fan-out**: 100M posts × 200 = 20B feed writes/day
- **Fan-out/sec**: ~230K writes/second

---

## API Design

### REST APIs

```
POST /api/v1/posts
Request: {
  "userId": "user123",
  "content": "Hello world!",
  "media": ["image1.jpg"],
  "hashtags": ["#hello"]
}
Response: {
  "postId": "post456",
  "timestamp": "2025-12-15T10:00:00Z"
}

GET /api/v1/feed
Query: ?limit=20&cursor=timestamp
Response: {
  "posts": [...],
  "nextCursor": "timestamp"
}

POST /api/v1/posts/{postId}/like
POST /api/v1/posts/{postId}/comment
POST /api/v1/users/{userId}/follow
```

---

## Database Design

### Schema

**Posts Table** (Cassandra - time-series):
```
postId (PK): UUID
userId: UUID
content: TEXT
mediaUrls: ARRAY[TEXT]
hashtags: ARRAY[VARCHAR]
timestamp: TIMESTAMP
likeCount: INT
commentCount: INT
```

**User Feeds Table** (Cassandra):
```
userId (PK): UUID
postId: UUID
timestamp: TIMESTAMP
```

**User Follows Table** (PostgreSQL):
```
userId (PK): UUID
followsUserId (PK): UUID
followedAt: TIMESTAMP
```

**User Timeline Table** (Cassandra - user's own posts):
```
userId (PK): UUID
postId: UUID
timestamp: TIMESTAMP
```

### Database Selection

**Posts**: **Cassandra** (time-series, high write volume)
**User Feeds**: **Cassandra** (high write volume from fan-out)
**User Relationships**: **PostgreSQL** (relational data, strong consistency)
**Timeline**: **Cassandra** (user's own posts)

---

## High-Level Design

### Architecture

```
Client → Load Balancer → Feed Service
                            ↓
                    [Cache (Redis)]
                            ↓
        ┌──────────────────┼──────────────────┐
        ↓                   ↓                   ↓
    [Fan-out]         [Feed Gen]          [Ranking]
    Service           Service             Service
        ↓                   ↓                   ↓
    [Database]        [Database]          [ML Service]
```

### Components

1. **Feed Service**: Core feed generation logic
2. **Fan-out Service**: Distribute posts to followers' feeds
3. **Ranking Service**: Rank posts for personalized feed
4. **Cache (Redis)**: Cache user feeds
5. **Database (Cassandra)**: Store posts and feeds
6. **Media Storage**: Object storage (S3) for images/videos
7. **CDN**: Serve media files

---

## Detailed Design

### Feed Generation Strategies

#### 1. Pull Model (On-Demand)

**How it works**:
- When user requests feed, fetch posts from all users they follow
- Aggregate and sort posts
- Return feed

**Pros**:
- Simple to implement
- Real-time (always fresh)
- No fan-out writes

**Cons**:
- Slow for users with many follows
- High database load
- Doesn't scale well

**Use Cases**: Small scale, users with few follows

---

#### 2. Push Model (Fan-out)

**How it works**:
- When user posts, push post to all followers' feeds
- Store in each follower's feed table
- When user requests feed, read from their feed table

**Pros**:
- Fast feed generation (pre-computed)
- Scales well for reads
- Low latency

**Cons**:
- High write volume (fan-out)
- Wasted storage for inactive users
- Doesn't work for users with millions of followers

**Use Cases**: Most users, moderate follower counts

---

#### 3. Hybrid Model (Push + Pull)

**How it works**:
- **Push** for users with < 1M followers (most users)
- **Pull** for celebrities/influencers with > 1M followers
- Combine both feeds

**Pros**:
- Best of both worlds
- Handles edge cases (celebrities)
- Efficient for most users

**Cons**:
- More complex
- Need to determine push vs pull

**Use Cases**: Large scale systems (Twitter, Facebook)

**Recommendation**: **Hybrid Model**

---

### Fan-out Process

**Flow**:
1. User creates post
2. Store post in database
3. Get list of followers
4. For each follower:
   - If follower count < threshold: Push to feed
   - If follower count > threshold: Skip (will pull)
5. Update user's own timeline

**Optimization**:
- **Async Processing**: Use message queue for fan-out
- **Batching**: Batch fan-out writes
- **Filtering**: Skip inactive users

---

### Feed Ranking

**Ranking Factors**:
1. **Recency**: Newer posts ranked higher
2. **Engagement**: Likes, comments, shares
3. **Relevance**: User interests, past interactions
4. **Relationship**: Close friends vs acquaintances
5. **Content Type**: Videos vs images vs text

**Ranking Algorithm**:
```
Score = (Recency × 0.3) + (Engagement × 0.3) + (Relevance × 0.2) + (Relationship × 0.2)
```

**Implementation**:
- **Simple**: Score-based ranking
- **Advanced**: ML models (neural networks)

---

### Caching Strategy

**Cache Layers**:
1. **Feed Cache (Redis)**: Cache generated feeds
   - Key: `feed:{userId}`
   - TTL: 5 minutes
   - Invalidate on new posts

2. **Post Cache (Redis)**: Cache individual posts
   - Key: `post:{postId}`
   - TTL: 1 hour

3. **User Cache (Redis)**: Cache user metadata
   - Key: `user:{userId}`
   - TTL: 24 hours

**Cache Invalidation**:
- On new post: Invalidate follower feeds
- On like/comment: Update post cache
- TTL-based expiration

---

## Scalability

### Horizontal Scaling

- **Stateless Services**: Feed service, fan-out service scale horizontally
- **Database Sharding**: Shard by userId
- **Cache Partitioning**: Distribute cache across Redis nodes

### Read Scaling

- **Caching**: Multi-layer caching reduces database load
- **Read Replicas**: Database read replicas
- **CDN**: Serve media files

### Write Scaling

- **Async Fan-out**: Use message queue for fan-out
- **Database Sharding**: Distribute writes across shards
- **Batching**: Batch fan-out writes

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **Database Replication**: Replicate across data centers
- **Cache Replication**: Redis cluster with replication

### Data Consistency

- **Feed Consistency**: Eventual consistency acceptable
- **Post Creation**: Strong consistency needed
- **Interactions**: Eventual consistency acceptable

---

## Trade-offs

### Consistency vs Availability

- **Feeds**: Eventual consistency acceptable (AP)
- **Posts**: Strong consistency needed (CP)

### Latency vs Freshness

- **Caching**: Lower latency, less fresh
- **Real-time**: Higher latency, always fresh

### Storage vs Performance

- **Fan-out**: Higher storage, better performance
- **Pull**: Lower storage, worse performance

---

## Extensions

### Additional Features

1. **Stories**: Ephemeral content (24 hours)
2. **Live Streaming**: Real-time video streaming
3. **Trending Topics**: Algorithm for trending hashtags
4. **Explore Feed**: Discover new content
5. **Notifications**: Notify on interactions
6. **Advertisements**: Sponsored posts in feed
7. **Content Moderation**: Filter inappropriate content

---

## Key Takeaways

- **Hybrid Model**: Push for most users, pull for celebrities
- **Fan-out**: Distribute posts to followers' feeds asynchronously
- **Caching**: Multi-layer caching for fast feed generation
- **Ranking**: Personalize feeds based on engagement and relevance
- **Scalability**: Stateless services enable horizontal scaling
- **Trade-offs**: Balance consistency, latency, and storage

---

## Related Topics

- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Multi-layer caching
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Async fan-out
- **[Databases]({{< ref "../databases/_index.md" >}})** - Time-series databases
- **[CDN & Content Delivery]({{< ref "../system-components/7-CDN_Content_Delivery.md" >}})** - Media delivery

