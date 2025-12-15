+++
title = "News Feed Ranking (Facebook/Twitter Algorithm)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 14
description = "Design a news feed ranking system like Facebook or Twitter. Covers ML-based ranking, real-time features, personalization, and scaling to rank billions of posts."
+++

---

## Problem Statement

Design a news feed ranking system that personalizes content for each user based on relevance, engagement, and user preferences. The system should rank posts in real-time and handle millions of users viewing personalized feeds.

**Examples**: Facebook News Feed, Twitter Timeline, Instagram Feed, LinkedIn Feed

---

## Requirements Clarification

### Functional Requirements

1. **Feed Generation**: Generate personalized feed for each user
2. **Ranking**: Rank posts by relevance
3. **Personalization**: Personalize based on user behavior
4. **Real-Time Updates**: Update feed in real-time
5. **Diversity**: Ensure content diversity
6. **Freshness**: Prioritize recent content

### Non-Functional Requirements

- **Scale**: 
  - 1B users
  - 500M daily active users
  - 100M posts/day
  - Average 200 posts per user feed
  - Rank 1000 candidates per user
- **Latency**: < 200ms to generate ranked feed
- **Freshness**: Update feed every few minutes
- **Accuracy**: High relevance (user engagement)

---

## Capacity Estimation

### Traffic Estimates

- **Feed Requests**: 500M DAU × 20 feeds/day = 10B feeds/day
- **Feed Requests/sec**: ~116K requests/second
- **Peak Requests**: 3x = ~350K requests/second
- **Ranking Operations**: 350K requests × 1000 candidates = 350M rankings/second

### Storage Estimates

- **User Features**: 500M users × 1 KB = 500 GB
- **Post Features**: 100M posts/day × 2 KB = 200 GB/day
- **Model Storage**: ML models ~10 GB
- **Feature Store**: Real-time features ~1 TB

---

## API Design

### REST APIs

```
GET /api/v1/feed
Query: ?userId=user123&limit=20&cursor=timestamp
Response: {
  "posts": [
    {
      "postId": "post456",
      "score": 0.95,
      "rank": 1,
      "reason": "High engagement"
    }
  ],
  "nextCursor": "timestamp"
}

POST /api/v1/feed/feedback
Request: {
  "postId": "post456",
  "action": "like",  // like, share, hide, report
  "userId": "user123"
}
```

---

## Database Design

### Schema

**Posts Table** (Cassandra):
```
postId (PK): UUID
userId: UUID
content: TEXT
timestamp: TIMESTAMP
engagementScore: DECIMAL
```

**User Features Table** (Feature Store):
```
userId (PK): UUID
features: JSON {
  "interests": [...],
  "pastEngagement": {...},
  "demographics": {...}
}
```

**Post Features Table** (Feature Store):
```
postId (PK): UUID
features: JSON {
  "content": {...},
  "engagement": {...},
  "freshness": {...}
}
```

**Ranking Results Cache** (Redis):
```
userId (PK): UUID
feed: JSON [...],
cachedAt: TIMESTAMP
```

### Database Selection

**Posts**: **Cassandra** (time-series, high write volume)
**Features**: **Feature Store** (Redis, DynamoDB) - real-time features
**Ranking Cache**: **Redis** - cache ranked feeds
**ML Models**: **Object Storage** (S3) - store models

---

## High-Level Design

### Architecture

```
Feed Request → [Candidate Generation] → [Feature Extraction]
                                            ↓
                                    [Ranking Service]
                                            ↓
                                    [ML Model] → [Re-ranking]
                                            ↓
                                    [Diversity Filter]
                                            ↓
                                    [Feed Response]
```

### Components

1. **Candidate Generation**: Generate candidate posts
2. **Feature Extraction**: Extract features for ranking
3. **Ranking Service**: Rank posts using ML model
4. **ML Model**: Trained ranking model
5. **Re-ranking**: Fine-tune ranking
6. **Diversity Filter**: Ensure content diversity
7. **Feature Store**: Store real-time features

---

## Detailed Design

### Candidate Generation

**Sources**:
1. **Followed Users**: Posts from users you follow
2. **Groups**: Posts from groups you're in
3. **Trending**: Trending posts
4. **Recommended**: Recommended content

**Generation Strategy**:
- **Pull Model**: Fetch posts from sources
- **Push Model**: Pre-compute candidate pool
- **Hybrid**: Combine both approaches

**Candidate Pool Size**: ~1000 candidates per user

---

### Feature Engineering

**User Features**:
- **Demographics**: Age, location, gender
- **Interests**: Topics, categories
- **Past Engagement**: Likes, shares, comments
- **Behavior**: Time spent, scroll depth
- **Social Graph**: Friends, connections

**Post Features**:
- **Content**: Text, images, videos
- **Engagement**: Likes, comments, shares
- **Freshness**: Time since posted
- **Author**: Author popularity, relationship
- **Media**: Type of media (image, video, text)

**Interaction Features**:
- **User-Post Interaction**: Past interactions with similar posts
- **Temporal**: Time of day, day of week
- **Context**: Device, location

---

### Ranking Algorithm

**Ranking Approaches**:

1. **Score-Based Ranking**:
   ```
   Score = w1 × Engagement + w2 × Freshness + w3 × Relevance
   ```
   - Simple, fast
   - Limited personalization

2. **Machine Learning Ranking**:
   - **Logistic Regression**: Simple ML model
   - **Gradient Boosting**: XGBoost, LightGBM
   - **Neural Networks**: Deep learning models
   - **Learning to Rank**: Specialized ranking models

**Recommendation**: **Gradient Boosting** (good balance)

---

### ML Model Training

**Training Data**:
- **Positive Examples**: Posts user engaged with (like, share, comment)
- **Negative Examples**: Posts user scrolled past
- **Features**: User features, post features, interaction features
- **Labels**: Engagement (1) or no engagement (0)

**Training Process**:
1. **Collect Data**: Collect user interactions
2. **Feature Engineering**: Extract features
3. **Train Model**: Train ranking model
4. **Evaluate**: Evaluate on test set
5. **Deploy**: Deploy model to production

**Model Updates**:
- **Online Learning**: Update model continuously
- **Batch Retraining**: Retrain daily/weekly
- **A/B Testing**: Test new models

---

### Real-Time Ranking

**Ranking Flow**:
1. **Generate Candidates**: Fetch ~1000 candidate posts
2. **Extract Features**: Extract features for each candidate
3. **Score**: Score each candidate using ML model
4. **Rank**: Sort by score
5. **Re-rank**: Apply re-ranking rules
6. **Diversity Filter**: Ensure diversity
7. **Return Top N**: Return top 20-50 posts

**Performance Optimization**:
- **Feature Caching**: Cache computed features
- **Model Caching**: Cache model predictions
- **Parallel Processing**: Process candidates in parallel
- **Early Termination**: Stop processing low-scoring candidates

---

### Diversity & Freshness

**Diversity**:
- **Content Diversity**: Mix of content types
- **Source Diversity**: Mix of sources (friends, groups, trending)
- **Temporal Diversity**: Mix of old and new posts

**Freshness**:
- **Recency Boost**: Boost recent posts
- **Time Decay**: Decay score over time
- **Breaking News**: Boost breaking news

**Implementation**:
- **Post-Ranking Filter**: Filter after ranking
- **Diversity Score**: Add diversity score to ranking
- **Freshness Multiplier**: Multiply score by freshness factor

---

## Scalability

### Horizontal Scaling

- **Stateless Ranking Service**: Scale horizontally
- **Feature Store**: Distributed feature store
- **Model Serving**: Multiple model servers

### Performance Optimization

- **Caching**: Cache ranked feeds, features
- **Pre-computation**: Pre-compute candidate pools
- **Batch Processing**: Batch feature extraction
- **Model Optimization**: Optimize model inference

---

## Reliability

### High Availability

- **Multiple Ranking Servers**: No single point of failure
- **Feature Store Replication**: Replicate features
- **Model Replication**: Replicate models

### Fault Tolerance

- **Model Failures**: Fallback to simpler model
- **Feature Failures**: Use default features
- **Service Failures**: Queue requests, process when service recovers

---

## Trade-offs

### Accuracy vs Latency

- **Complex Models**: More accurate, slower
- **Simple Models**: Less accurate, faster

### Freshness vs Relevance

- **Fresh Content**: More recent, may be less relevant
- **Relevant Content**: More relevant, may be older

### Personalization vs Diversity

- **High Personalization**: More relevant, less diverse
- **High Diversity**: More diverse, less personalized

---

## Extensions

### Additional Features

1. **Multi-Armed Bandits**: Explore vs exploit
2. **Reinforcement Learning**: Learn from user feedback
3. **Explainability**: Explain why posts are shown
4. **User Control**: Let users control feed preferences
5. **A/B Testing**: Test ranking algorithms
6. **Real-Time Learning**: Update model in real-time
7. **Contextual Ranking**: Rank based on context (time, location)

---

## Key Takeaways

- **ML-Based Ranking**: Use ML models for personalized ranking
- **Feature Engineering**: Extract relevant features
- **Candidate Generation**: Generate diverse candidate pool
- **Real-Time Ranking**: Rank in real-time for freshness
- **Diversity**: Ensure content diversity
- **Scalability**: Scale ranking service horizontally

---

## Related Topics

- **[Social Media Feed]({{< ref "6-Social_Media_Feed.md" >}})** - Feed generation and fan-out
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Cache ranked feeds
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed ranking
- **[Monitoring & Observability]({{< ref "../system-components/11-Monitoring_Observability.md" >}})** - Track ranking metrics

