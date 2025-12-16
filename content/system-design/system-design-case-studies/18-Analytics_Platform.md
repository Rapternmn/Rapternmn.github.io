+++
title = "Analytics Platform (Google Analytics)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 18
description = "Design an analytics platform like Google Analytics. Covers event collection, real-time and batch processing, data warehousing, and scaling to handle billions of events."
+++

---

## Problem Statement

Design an analytics platform that collects, processes, and analyzes user events. The system should handle high-volume event ingestion, provide real-time and batch analytics, and scale to process billions of events.

**Examples**: Google Analytics, Mixpanel, Amplitude, Segment

---

## Requirements Clarification

### Functional Requirements

1. **Event Collection**: Collect events from clients (web, mobile)
2. **Real-Time Analytics**: Provide real-time analytics
3. **Batch Analytics**: Provide historical analytics
4. **Dashboards**: Create analytics dashboards
5. **Segmentation**: Segment users by properties
6. **Funnels**: Track conversion funnels
7. **Retention**: Track user retention

### Non-Functional Requirements

- **Scale**: 
  - 10B events/day
  - 100K events/second
  - Peak: 1M events/second
  - Average 500 bytes per event
- **Latency**: < 1 second for real-time queries
- **Retention**: 2+ years of data
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Events**: 10B events/day = ~116K events/second
- **Peak Events**: 1M events/second
- **Event Size**: 10B × 500 bytes = 5 TB/day
- **Monthly Storage**: 5 TB × 30 = 150 TB/month

### Storage Estimates

- **Daily Events**: 5 TB/day
- **Monthly Storage**: 150 TB/month
- **Yearly Storage**: ~1.8 PB/year (with compression: ~500 TB)
- **Aggregated Data**: ~10% of raw data = 50 TB/year

---

## API Design

### REST APIs

```
POST /api/v1/events
Request: {
  "event": "page_view",
  "userId": "user123",
  "properties": {
    "page": "/home",
    "referrer": "google.com"
  },
  "timestamp": "2025-12-15T10:00:00Z"
}

GET /api/v1/analytics/realtime
Query: ?metric=active_users&timeRange=1h
Response: {
  "metric": "active_users",
  "value": 10000,
  "timestamp": "2025-12-15T10:00:00Z"
}

GET /api/v1/analytics/historical
Query: ?metric=page_views&startDate=2025-12-01&endDate=2025-12-15&groupBy=day
Response: {
  "metric": "page_views",
  "data": [
    {"date": "2025-12-01", "value": 1000000},
    {"date": "2025-12-02", "value": 1200000}
  ]
}
```

---

## Database Design

### Schema

**Events Table** (Time-series DB / Data Warehouse):
```
eventId (PK): UUID
eventType: VARCHAR (page_view, click, purchase)
userId: UUID
sessionId: UUID
properties: JSON
timestamp: TIMESTAMP
```

**Aggregated Metrics Table** (Data Warehouse):
```
metricId (PK): UUID
metricName: VARCHAR (page_views, active_users)
date: DATE
hour: INT (nullable)
value: BIGINT
dimensions: JSON (nullable)
```

**User Properties Table** (Data Warehouse):
```
userId (PK): UUID
properties: JSON {
  "country": "US",
  "device": "mobile",
  "browser": "Chrome"
}
```

### Database Selection

**Raw Events**: **Time-series Database** (InfluxDB, TimescaleDB) or **Data Warehouse** (BigQuery, Redshift)
**Aggregated Data**: **Data Warehouse** (BigQuery, Redshift, Snowflake)
**Real-Time Data**: **Stream Processing** (Kafka Streams, Flink) + **Redis** (cache)

---

## High-Level Design

### Architecture

```
Clients → [Event Collector] → [Message Queue] → [Stream Processor]
                                                    ↓
                                            [Real-Time Analytics]
                                                    ↓
                                            [Batch Processor]
                                                    ↓
                                            [Data Warehouse]
                                                    ↓
                                            [Analytics Service]
                                                    ↓
                                            [Dashboards]
```

### Components

1. **Event Collector**: Collect events from clients
2. **Message Queue**: Buffer events (Kafka)
3. **Stream Processor**: Process events in real-time (Kafka Streams, Flink)
4. **Batch Processor**: Process events in batches (Spark)
5. **Data Warehouse**: Store events and aggregated data
6. **Analytics Service**: Serve analytics queries
7. **Dashboard**: Visualize analytics

---

## Detailed Design

### Event Collection

**Collection Methods**:

1. **Client SDK**:
   - **JavaScript SDK**: Web tracking
   - **Mobile SDK**: iOS/Android tracking
   - **Server SDK**: Backend tracking
   - Send events via HTTP/HTTPS

2. **Event Batching**:
   - **Batch Events**: Batch events before sending
   - **Reduce Requests**: Reduce number of requests
   - **Buffer**: Buffer events locally, send periodically

3. **Event Validation**:
   - **Schema Validation**: Validate event schema
   - **Required Fields**: Check required fields
   - **Sanitization**: Sanitize event data

**Event Format**:
```json
{
  "event": "page_view",
  "userId": "user123",
  "sessionId": "session456",
  "properties": {
    "page": "/home",
    "referrer": "google.com",
    "device": "mobile"
  },
  "timestamp": "2025-12-15T10:00:00Z"
}
```

---

### Event Processing Pipeline

**Real-Time Processing**:
1. **Stream Processing**: Process events in real-time
2. **Aggregation**: Aggregate events (count, sum, average)
3. **Window**: Time windows (1 minute, 5 minutes, 1 hour)
4. **Store**: Store aggregated metrics in Redis/cache

**Batch Processing**:
1. **Batch Collection**: Collect events in batches
2. **ETL**: Extract, transform, load
3. **Aggregation**: Aggregate events by time, dimensions
4. **Store**: Store in data warehouse

**Processing Flow**:
```
Events → Kafka → [Stream Processor] → Real-Time Metrics (Redis)
                ↓
        [Batch Processor] → Aggregated Metrics (Data Warehouse)
```

---

### Real-Time Analytics

**Real-Time Metrics**:
- **Active Users**: Users active in last N minutes
- **Event Counts**: Count events in real-time
- **Conversion Rates**: Track conversions in real-time

**Implementation**:
- **Stream Processing**: Kafka Streams, Flink
- **Time Windows**: Sliding windows, tumbling windows
- **Storage**: Redis for fast access
- **TTL**: Expire old metrics

**Example**:
- **Active Users (Last 5 minutes)**: Count unique users in 5-minute window
- **Page Views (Last Hour)**: Count page views in 1-hour window

---

### Batch Analytics

**Batch Processing**:
1. **ETL Pipeline**: Extract, transform, load
2. **Aggregation**: Aggregate by time, dimensions
3. **Storage**: Store in data warehouse
4. **Query**: Query aggregated data

**Aggregations**:
- **Time Aggregations**: By hour, day, week, month
- **Dimension Aggregations**: By country, device, browser
- **Metrics**: Count, sum, average, unique count

**Data Warehouse Schema**:
- **Star Schema**: Fact table + dimension tables
- **Fact Table**: Events (grain: event)
- **Dimension Tables**: Users, Pages, Sessions

---

### Analytics Queries

**Query Types**:

1. **Time-Series Queries**:
   - **Example**: Page views over time
   - **Group By**: Day, hour, week
   - **Metrics**: Count, sum, average

2. **Segmentation Queries**:
   - **Example**: Page views by country
   - **Group By**: Country, device, browser
   - **Filter**: Filter by user properties

3. **Funnel Queries**:
   - **Example**: Conversion funnel
   - **Steps**: Step 1 → Step 2 → Step 3
   - **Conversion Rate**: % users completing funnel

4. **Retention Queries**:
   - **Example**: User retention
   - **Cohort**: Users by signup date
   - **Retention**: % users returning

---

## Scalability

### Horizontal Scaling

- **Event Collectors**: Scale collectors horizontally
- **Message Queue**: Kafka partitions for parallel processing
- **Stream Processors**: Scale processors horizontally
- **Batch Processors**: Scale Spark clusters

### Performance Optimization

- **Event Batching**: Batch events before sending
- **Compression**: Compress events
- **Caching**: Cache aggregated metrics
- **Pre-aggregation**: Pre-aggregate common queries

---

## Reliability

### High Availability

- **Multiple Collectors**: No single point of failure
- **Message Queue Replication**: Kafka replication
- **Data Warehouse Replication**: Replicate data warehouse

### Fault Tolerance

- **Event Loss**: Minimize event loss with replication
- **Processing Failures**: Retry processing
- **Data Warehouse Failures**: Replicate data warehouse

### Data Durability

- **Event Storage**: Store events in durable storage
- **Backup**: Backup data warehouse
- **Retention**: Manage data retention policies

---

## Trade-offs

### Real-Time vs Batch

- **Real-Time**: Lower latency, higher cost
- **Real-Time**: Lower latency, higher cost
- **Batch**: Higher latency, lower cost
- **Kappa Architecture**: Stream-only processing (simplifies pipeline)

### Storage vs Performance

- **Raw Events**: More storage, flexible queries
- **Aggregated Data**: Less storage, limited queries

### Accuracy vs Performance

- **Exact Counts**: More accurate, slower
- **Approximate Counts**: Less accurate, faster (HyperLogLog)

---

## Extensions

### Additional Features

1. **User Profiles**: Build user profiles from events
2. **Cohort Analysis**: Analyze user cohorts
3. **A/B Testing**: Track A/B test results
4. **Predictive Analytics**: Predict user behavior
5. **Anomaly Detection**: Detect anomalies in events
6. **Data Export**: Export data to external systems
7. **Privacy**: GDPR compliance, data anonymization

---

## Key Takeaways

- **Dual Processing**: Real-time and batch processing
- **Event Batching**: Batch events for efficiency
- **Data Warehouse**: Store aggregated data for analytics
- **Stream Processing**: Process events in real-time
- **Scalability**: Scale horizontally with partitioning
- **Caching**: Cache aggregated metrics for fast queries

---

## Related Topics

- **[Data Engineering]({{< ref "../data-engineering/_index.md" >}})** - Data pipelines and processing
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Event buffering
- **[Databases]({{< ref "../databases/_index.md" >}})** - Data warehouses
- **[Monitoring & Observability]({{< ref "../system-components/11-Monitoring_Observability.md" >}})** - Analytics and monitoring

