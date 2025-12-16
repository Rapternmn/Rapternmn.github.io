+++
title = "Distributed Logging System (Splunk/ELK Stack)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 15
description = "Design a distributed logging system like Splunk or ELK Stack. Covers log collection, aggregation, storage, search, and scaling to handle billions of log entries."
+++

---

## Problem Statement

Design a distributed logging system that collects, stores, and searches logs from multiple services. The system should handle high-volume logs, provide fast search, and scale to process billions of log entries.

**Examples**: Splunk, ELK Stack (Elasticsearch, Logstash, Kibana), Datadog, CloudWatch

---

## Requirements Clarification

### Functional Requirements

1. **Log Collection**: Collect logs from multiple sources
2. **Log Aggregation**: Aggregate logs from distributed services
3. **Log Storage**: Store logs efficiently
4. **Log Search**: Search logs by query
5. **Log Analysis**: Analyze logs (aggregations, visualizations)
6. **Alerting**: Alert on log patterns
7. **Retention**: Manage log retention policies

### Non-Functional Requirements

- **Scale**: 
  - 1B log entries/day
  - 10K log entries/second
  - Peak: 100K entries/second
  - 1000 services
  - Average 1 KB per log entry
- **Latency**: < 1 second for search queries
- **Retention**: 30-90 days retention
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Log Entries**: 1B entries/day = ~11.6K entries/second
- **Peak Entries**: 100K entries/second
- **Log Size**: 1B × 1 KB = 1 TB/day
- **Monthly Storage**: 1 TB × 30 = 30 TB/month

### Storage Estimates

- **Daily Storage**: 1 TB/day
- **Monthly Storage**: 30 TB/month
- **Yearly Storage**: ~360 TB/year (with compression: ~100 TB)
- **Index Size**: ~20% of log size = 20 TB/year

---

## API Design

### REST APIs

```
POST /api/v1/logs
Request: {
  "service": "api-service",
  "level": "error",
  "message": "Database connection failed",
  "timestamp": "2025-12-15T10:00:00Z",
  "metadata": {...}
}

GET /api/v1/logs/search
Query: ?q=error&service=api-service&from=2025-12-15&to=2025-12-16
Response: {
  "logs": [...],
  "total": 1000,
  "took": 0.5  // seconds
}

GET /api/v1/logs/aggregate
Query: ?service=api-service&groupBy=level&timeRange=1h
Response: {
  "aggregations": [
    {"level": "error", "count": 100},
    {"level": "warn", "count": 50}
  ]
}
```

---

## Database Design

### Schema

**Logs Table** (Elasticsearch/Time-series DB):
```
timestamp: TIMESTAMP (PK)
service: VARCHAR
level: VARCHAR (info, warn, error)
message: TEXT
metadata: JSON
host: VARCHAR
```

**Indexes**:
- **Time-based Index**: Index by timestamp (daily/weekly indices)
- **Service Index**: Index by service name
- **Level Index**: Index by log level

### Database Selection

**Log Storage**: **Elasticsearch** (full-text search) or **Time-series Database** (InfluxDB, TimescaleDB)
**Metadata**: **PostgreSQL** (service metadata, configurations)
**Archived Logs**: **Object Storage** (S3) - cold storage

---

## High-Level Design

### Architecture

```
Services → [Log Agents] → [Message Queue] → [Log Processor]
                                                    ↓
                                            [Log Storage]
                                                    ↓
                                            [Search Index]
                                                    ↓
                                            [Search Service]
                                                    ↓
                                            [Analytics/Visualization]
```

### Components

1. **Log Agents**: Collect logs from services (Filebeat, Fluentd)
2. **Message Queue**: Buffer logs (Kafka)
3. **Log Processor**: Parse and enrich logs (Logstash)
4. **Log Storage**: Store logs (Elasticsearch, S3)
5. **Search Index**: Index logs for search (Elasticsearch)
6. **Search Service**: Handle search queries
7. **Analytics**: Analyze and visualize logs (Kibana, Grafana)

---

## Detailed Design

### Log Collection

**Collection Methods**:

1. **Agent-Based Collection**:
   - **Filebeat**: Lightweight log shipper
   - **Fluentd**: Log collector
   - **Logstash**: Log processor
   - Deploy agents on each service host

2. **API-Based Collection**:
   - Services send logs via API
   - HTTP/HTTPS endpoints
   - Authentication required

3. **Sidecar Pattern**:
   - Sidecar container collects logs
   - Common in Kubernetes
   - Isolated from application

**Recommendation**: **Agent-Based Collection** (Filebeat/Fluentd)

---

### Log Aggregation

**Aggregation Flow**:
1. **Collect**: Agents collect logs from services
2. **Ship**: Ship logs to central location
3. **Buffer**: Buffer in message queue (Kafka)
4. **Process**: Process logs (parse, enrich, transform)
5. **Store**: Store in log storage

**Message Queue**:
- **Kafka**: High-throughput message queue
- **Partitioning**: Partition by service or log level
- **Retention**: Retain logs for processing window

---

### Log Processing

**Processing Steps**:
1. **Parse**: Parse log format (JSON, text, structured)
2. **Enrich**: Add metadata (host, service, environment)
3. **Transform**: Transform log format
4. **Filter**: Filter irrelevant logs
5. **Route**: Route to appropriate storage

**Log Formats**:
- **Structured Logs**: JSON format (preferred)
- **Unstructured Logs**: Text format (need parsing)
- **Semi-Structured**: Key-value pairs

**Enrichment**:
- **Host Information**: Hostname, IP, region
- **Service Information**: Service name, version
- **Environment**: Production, staging, dev
- **Correlation IDs**: Trace requests across services

---

### Log Storage

**Storage Strategies**:

1. **Hot Storage** (Recent Logs):
   - **Elasticsearch**: Fast search, recent logs (7-30 days)
   - **SSD Storage**: Fast access
   - **High Cost**: Expensive storage

2. **Warm Storage** (Older Logs):
   - **Elasticsearch**: Slower search, older logs (30-90 days)
   - **HDD Storage**: Slower access
   - **Medium Cost**: Moderate cost

3. **Cold Storage** (Archived Logs):
   - **Object Storage** (S3): Archive, rarely accessed
   - **Compression**: Compress logs
   - **Low Cost**: Cheap storage

**Index Lifecycle Management**:
- **Hot**: Recent indices (fast, expensive)
- **Warm**: Older indices (slower, cheaper)
- **Cold**: Archived indices (slowest, cheapest)
- **Delete**: Delete after retention period

---

### Log Search

**Search Capabilities**:
1. **Full-Text Search**: Search log content
2. **Filtered Search**: Filter by service, level, time range
3. **Aggregations**: Count, sum, average
4. **Time-Series Queries**: Query by time range

**Search Implementation**:
- **Elasticsearch**: Full-text search engine
- **Query Types**: Match, term, range, bool queries
- **Performance**: Index optimization, caching

**Search Optimization**:
- **Indexing**: Index frequently searched fields
- **Caching**: Cache common queries
- **Pagination**: Paginate large result sets
- **Time Range**: Limit search to time range

---

### Log Analysis

**Analysis Types**:
1. **Aggregations**: Count errors, group by service
2. **Time-Series Analysis**: Trends over time
3. **Pattern Detection**: Detect anomalies
4. **Correlation**: Correlate logs across services

**Visualization**:
- **Dashboards**: Create dashboards (Kibana, Grafana)
- **Charts**: Line charts, bar charts, pie charts
- **Alerts**: Set up alerts on patterns

---

## Scalability

### Horizontal Scaling

- **Log Agents**: Deploy agents on each host
- **Message Queue**: Kafka partitions for parallel processing
- **Log Processors**: Scale processors horizontally
- **Search Nodes**: Scale Elasticsearch cluster

### Performance Optimization

- **Indexing**: Optimize index settings
- **Sharding**: Shard indices by time/service
- **Caching**: Cache search results
- **Compression**: Compress stored logs

---

## Reliability

### High Availability

- **Multiple Agents**: Agents on each host (no single point of failure)
- **Message Queue Replication**: Kafka replication
- **Storage Replication**: Elasticsearch replication
- **Multiple Search Nodes**: No single point of failure

### Fault Tolerance

- **Agent Failures**: Retry sending logs (at-least-once delivery)
- **Queue Failures**: Buffer logs locally, retry
- **Storage Failures**: Replicate data, failover
- **Search Failures**: Route to healthy nodes

### Data Durability

- **Replication**: Replicate logs across nodes
- **Backup**: Backup to object storage
- **Retention**: Manage retention policies

---

## Trade-offs

### Storage vs Performance

- **Hot Storage**: Fast search, expensive
- **Cold Storage**: Slow search, cheap

### Retention vs Cost

- **Long Retention**: More storage, higher cost
- **Short Retention**: Less storage, lower cost

### Search Speed vs Storage

- **Full Indexing**: Fast search, more storage
- **Selective Indexing**: Slower search, less storage

---

## Extensions

### Additional Features

1. **Log Streaming**: Stream logs in real-time
2. **Log Correlation**: Correlate logs with traces
3. **Machine Learning**: Anomaly detection
4. **Log Sampling**: Sample logs to reduce volume
5. **Multi-Tenancy**: Support multiple tenants
6. **Compliance**: GDPR, HIPAA compliance
7. **Log Encryption**: Encrypt sensitive logs

---

## Key Takeaways

- **Agent-Based Collection**: Deploy agents for log collection
- **Message Queue**: Buffer logs for reliability
- **Tiered Storage**: Hot, warm, cold storage tiers
- **Index Lifecycle**: Manage index lifecycle
- **Search Optimization**: Optimize search performance
- **Scalability**: Scale horizontally with sharding

---

## Related Topics

- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Log buffering
- **[Databases]({{< ref "../databases/_index.md" >}})** - Time-series databases
- **[Monitoring & Observability]({{< ref "../system-components/11-Monitoring_Observability.md" >}})** - Logging and monitoring
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed architecture

