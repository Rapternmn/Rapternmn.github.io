+++
title = "Notification System"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 5
description = "Design a notification system for push notifications, emails, and SMS. Covers multi-channel delivery, queuing, batching, and scaling to millions of notifications."
+++

---

## Problem Statement

Design a notification system that delivers push notifications, emails, and SMS to users. The system should support multiple channels, handle high volume, and ensure reliable delivery.

**Use Cases**: Push notifications, email alerts, SMS notifications, in-app notifications

---

## Requirements Clarification

### Functional Requirements

1. **Multi-Channel Delivery**: Push, email, SMS
2. **User Preferences**: Users can choose notification channels
3. **Templates**: Support notification templates
4. **Scheduling**: Schedule notifications for future delivery
5. **Batching**: Batch notifications for efficiency
6. **Retry Logic**: Retry failed deliveries
7. **Analytics**: Track delivery status and open rates

### Non-Functional Requirements

- **Scale**: 
  - 1B notifications/day
  - 10M notifications/hour peak
  - Multiple channels (push, email, SMS)
- **Latency**: < 5 seconds for push, < 1 minute for email/SMS
- **Reliability**: 99.9% delivery success rate
- **Throughput**: Handle burst traffic

---

## Capacity Estimation

### Traffic Estimates

- **Notifications**: 1B notifications/day = ~11.6K notifications/second
- **Peak**: 10M/hour = ~2,800 notifications/second
- **Channels**: 
  - Push: 60% = 600M/day
  - Email: 30% = 300M/day
  - SMS: 10% = 100M/day

### Storage Estimates

- **Notification Record**: ~500 bytes
- **Daily Storage**: 1B × 500 bytes = 500 GB/day
- **Yearly Storage**: ~180 TB/year

---

## API Design

### REST APIs

```
POST /api/v1/notifications
Request: {
  "userId": "user123",
  "type": "push",  // push, email, sms
  "title": "New Message",
  "body": "You have a new message",
  "data": {...},  // custom data
  "priority": "high",
  "scheduleAt": null  // optional
}
Response: {
  "notificationId": "notif456",
  "status": "queued"
}

GET /api/v1/notifications/{notificationId}
Response: {
  "notificationId": "notif456",
  "status": "delivered",
  "deliveredAt": "2025-12-15T10:00:00Z"
}

POST /api/v1/notifications/batch
Request: {
  "userIds": ["user1", "user2", ...],
  "type": "push",
  "title": "Broadcast",
  "body": "Message"
}
```

---

## Database Design

### Schema

**Notifications Table** (PostgreSQL):
```
notificationId (PK): UUID
userId: UUID
type: VARCHAR (push, email, sms)
title: VARCHAR
body: TEXT
status: VARCHAR (queued, sent, delivered, failed)
priority: VARCHAR (low, normal, high)
createdAt: TIMESTAMP
scheduledAt: TIMESTAMP (nullable)
sentAt: TIMESTAMP (nullable)
deliveredAt: TIMESTAMP (nullable)
retryCount: INT
errorMessage: TEXT (nullable)
```

**User Preferences Table**:
```
userId (PK): UUID
pushEnabled: BOOLEAN
emailEnabled: BOOLEAN
smsEnabled: BOOLEAN
preferredChannel: VARCHAR
```

**Templates Table**:
```
templateId (PK): UUID
name: VARCHAR
type: VARCHAR (push, email, sms)
subject: VARCHAR
body: TEXT
variables: JSON
```

---

## High-Level Design

### Architecture

![Notification System Architecture](/images/system-design/notification-system-architecture.png)

### Components

1. **Notification Service**: Core notification logic
2. **Message Queue (Kafka)**: Buffer notifications, async processing
3. **Push Service**: Handle push notifications (FCM/APNS)
4. **Email Service**: Handle email delivery (SMTP)
5. **SMS Service**: Handle SMS delivery (SMS gateway)
6. **Template Service**: Manage notification templates
7. **Analytics Service**: Track delivery metrics

---

## Detailed Design

### Notification Flow

1. **Client** → Notification Service: Create notification
2. **Notification Service**:
   - Validate user preferences
   - Apply template (if template ID provided)
   - Store in database
   - Publish to message queue
3. **Message Queue** → Channel Service: Consume notification
4. **Channel Service**:
   - Format for channel (push/email/SMS)
   - Send via external service (FCM/SMTP/SMS gateway)
   - Update status in database
5. **Retry Logic**: Retry failed deliveries

---

### Multi-Channel Delivery

#### Push Notifications

**Services**:
- **FCM** (Firebase Cloud Messaging): Android
- **APNS** (Apple Push Notification Service): iOS

**Flow**:
1. Register device tokens
2. Send notification to FCM/APNS
3. FCM/APNS delivers to device

**Challenges**:
- Device token management
- Token expiration
- Platform-specific formatting

---

#### Email Delivery

**Services**:
- **SMTP**: Direct SMTP servers
- **SES** (AWS Simple Email Service): Managed service
- **SendGrid/Mailgun**: Third-party services

**Flow**:
1. Format email (HTML/text)
2. Send via SMTP/SES
3. Track delivery status

**Challenges**:
- Spam filtering
- Email deliverability
- Template rendering

---

#### SMS Delivery

**Services**:
- **Twilio**: SMS gateway
- **AWS SNS**: SMS service
- **Direct carriers**: Direct integration

**Flow**:
1. Format SMS message
2. Send via SMS gateway
3. Track delivery status

**Challenges**:
- Cost (more expensive than push/email)
- Character limits
- Carrier restrictions

---

### Batching Strategy

**Why Batch**:
- Reduce API calls to external services
- Improve throughput
- Lower costs

**Batching Approaches**:
1. **Time-based**: Batch every N seconds
2. **Size-based**: Batch when N notifications queued
3. **Hybrid**: Batch when size OR time threshold reached

**Example**: Batch push notifications
- Batch size: 100 notifications
- Batch interval: 5 seconds
- Send batch to FCM

---

### Retry Logic

**Retry Strategy**:
- **Exponential Backoff**: Increase delay between retries
- **Max Retries**: Limit retry attempts (e.g., 3-5)
- **Dead Letter Queue**: Move failed notifications after max retries

**Retry Scenarios**:
- Network failures
- Service unavailable
- Rate limiting
- Invalid tokens/addresses

---

### User Preferences

**Preference Types**:
- **Channel Selection**: Push, email, SMS
- **Notification Types**: Messages, alerts, marketing
- **Quiet Hours**: Disable notifications during certain hours
- **Frequency Limits**: Max notifications per day/hour

**Implementation**:
- Store preferences in database
- Check before sending notification
- Respect user choices

---

## Scalability

### Horizontal Scaling

- **Notification Service**: Stateless, scale horizontally
- **Channel Services**: Stateless, scale independently
- **Message Queue**: Kafka partitions for parallel processing

### Performance Optimization

- **Batching**: Reduce external API calls
- **Caching**: Cache user preferences, templates
- **Async Processing**: Queue-based async delivery
- **Connection Pooling**: Reuse connections to external services

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **Message Queue Replication**: Kafka replication
- **External Service Redundancy**: Multiple providers (failover)

### Delivery Guarantees

- **At-Least-Once**: Notifications delivered at least once
- **Idempotency**: Handle duplicate notifications
- **Retry Logic**: Automatic retry for failures
- **Dead Letter Queue**: Handle undeliverable notifications

### Fault Tolerance

- **Service Failures**: Queue notifications, process when service recovers
- **External Service Failures**: Retry with backoff, failover to backup provider
- **Database Failures**: Read from replicas

---

## Trade-offs

### Latency vs Throughput

- **Immediate Delivery**: Low latency, lower throughput
- **Batching**: Higher latency, higher throughput

### Cost vs Reliability

- **Multiple Providers**: Higher cost, better reliability
- **Single Provider**: Lower cost, single point of failure

### Storage vs Performance

- **Store All Notifications**: Higher storage, better analytics
- **Store Recent Only**: Lower storage, limited history

---

## Extensions

### Additional Features

1. **Rich Notifications**: Images, actions, deep links
2. **Notification Grouping**: Group related notifications
3. **Notification Center**: In-app notification history
4. **A/B Testing**: Test notification content
5. **Personalization**: Personalized notification content
6. **Geographic Targeting**: Location-based notifications
7. **Time Zone Handling**: Schedule based on user timezone

---

## Key Takeaways

- **Multi-Channel**: Support multiple delivery channels
- **Message Queue**: Use queues for async processing and reliability
- **Batching**: Batch notifications for efficiency
- **Retry Logic**: Implement retry with exponential backoff
- **User Preferences**: Respect user notification preferences
- **Scalability**: Stateless services enable horizontal scaling

---

## Related Topics

- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Message queue patterns
- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API design
- **[Monitoring & Observability]({{< ref "../system-components/11-Monitoring_Observability.md" >}})** - Track delivery metrics

