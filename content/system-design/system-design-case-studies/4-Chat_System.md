+++
title = "Chat System (WhatsApp/Telegram)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 4
description = "Design a real-time chat system like WhatsApp or Telegram. Covers WebSockets, message queues, presence, group chats, and scaling to billions of messages."
+++

---

## Problem Statement

Design a real-time chat system that supports one-on-one messaging, group chats, and presence indicators. The system should deliver messages instantly and handle millions of concurrent users.

**Examples**: WhatsApp, Telegram, Slack, Discord

---

## Requirements Clarification

### Functional Requirements

1. **One-on-One Chat**: Send/receive messages between two users
2. **Group Chat**: Support group conversations
3. **Presence**: Show online/offline status
4. **Message Delivery**: Deliver messages reliably
5. **Read Receipts**: Show message read status
6. **Media Messages**: Support images, videos, files
7. **Message History**: Store and retrieve chat history

### Non-Functional Requirements

- **Scale**: 
  - 1B users
  - 50B messages/day
  - 10M concurrent connections
  - Average 50 messages/user/day
- **Latency**: < 100ms message delivery
- **Availability**: 99.9% uptime
- **Durability**: Messages should not be lost

---

## Capacity Estimation

### Traffic Estimates

- **Messages**: 50B messages/day = ~580K messages/second
- **Peak messages**: 3x = ~1.7M messages/second
- **Concurrent Users**: 10M
- **Read/Write Ratio**: 1:1 (each message sent and received)

### Storage Estimates

- **Message Size**: Average 100 bytes (text)
- **Media**: Average 1 MB (images/videos)
- **Metadata**: ~200 bytes per message
- **Daily Text Messages**: 50B × 300 bytes = 15 TB/day
- **Daily Media**: Assume 10% media = 5B × 1 MB = 5 PB/day
- **Yearly Storage**: ~2 PB (text) + ~1.8 EB (media, needs optimization)

### Bandwidth Estimates

- **Message Delivery**: 1.7M msgs/sec × 300 bytes = 510 MB/sec
- **Media Delivery**: Much higher (needs CDN)

---

## API Design

### REST APIs

```
POST /api/v1/messages
Request: {
  "toUserId": "user123",
  "message": "Hello!",
  "type": "text",
  "groupId": null  // null for 1-on-1
}
Response: {
  "messageId": "msg456",
  "timestamp": "2025-12-15T10:00:00Z",
  "status": "sent"
}

GET /api/v1/messages/{chatId}
Query: ?limit=50&before=timestamp
Response: {
  "messages": [...],
  "hasMore": true
}

POST /api/v1/presence
Request: {
  "status": "online"  // online, offline, away
}
```

### WebSocket APIs

```
Connection: ws://chat.example.com/ws?userId=user123&token=...

Messages:
- Send: {"type": "message", "to": "user456", "text": "Hi"}
- Receive: {"type": "message", "from": "user456", "text": "Hi", "timestamp": "..."}
- Presence: {"type": "presence", "userId": "user456", "status": "online"}
```

---

## Database Design

### Schema

**Messages Table** (Cassandra - time-series):
```
messageId (PK): UUID
chatId: UUID (conversation ID)
fromUserId: UUID
toUserId: UUID (or groupId)
message: TEXT
messageType: VARCHAR (text, image, video, file)
timestamp: TIMESTAMP
status: VARCHAR (sent, delivered, read)
```

**Chats Table** (PostgreSQL):
```
chatId (PK): UUID
chatType: VARCHAR (1-on-1, group)
participants: ARRAY[UUID]
createdAt: TIMESTAMP
lastMessageAt: TIMESTAMP
```

**User Presence Table** (Redis):
```
userId (PK): UUID
status: VARCHAR (online, offline, away)
lastSeen: TIMESTAMP
```

### Database Selection

**Messages**: **Cassandra** (time-series, high write volume)
- Partition by chatId, cluster by timestamp
- Handles high write throughput
- Good for time-series data

**Chats/Metadata**: **PostgreSQL** (relational data)
- User relationships
- Group information
- Strong consistency needed

**Presence**: **Redis** (real-time, ephemeral)
- Fast lookups
- TTL for offline detection
- Low latency

---

## High-Level Design

### Architecture

```
Client → Load Balancer → WebSocket Server
                            ↓
                    [Message Service]
                    ↓           ↓
            [Message Queue]  [Database]
                    ↓
            [Presence Service] → Redis
                    ↓
            [Notification Service] → Push Notifications
```

### Components

1. **WebSocket Servers**: Maintain persistent connections
2. **Message Service**: Handle message routing and delivery
3. **Message Queue (Kafka)**: Async message processing
4. **Database (Cassandra)**: Store messages
5. **Presence Service**: Track user online/offline status
6. **Redis**: Cache presence, recent messages
7. **Notification Service**: Push notifications for offline users
8. **Media Storage**: Object storage (S3) for media files

---

## Detailed Design

### Message Flow

#### Sending Message

1. **Client** → WebSocket Server: Send message
2. **WebSocket Server** → Message Service: Route message
3. **Message Service**:
   - Store in database (Cassandra)
   - Publish to message queue (Kafka)
   - Check recipient presence (Redis)
4. **If Online**: Push via WebSocket
5. **If Offline**: Queue for push notification
6. **Return ACK**: Confirm message received

#### Receiving Message

1. **Message Queue** → Delivery Service: Consume message
2. **Check Presence**: Is user online?
3. **If Online**: Push via WebSocket connection
4. **If Offline**: Store for later delivery
5. **Update Status**: Mark as delivered

---

### WebSocket Connection Management

**Connection Pooling**:
- Each WebSocket server maintains connections
- User → Server mapping stored in Redis
- Load balancer routes to correct server

**Connection Lifecycle**:
1. **Connect**: User connects, store mapping
2. **Heartbeat**: Periodic ping/pong to keep alive
3. **Disconnect**: Clean up mapping, update presence

**Scaling**:
- Multiple WebSocket servers
- Sticky sessions (same user → same server)
- Or use Redis pub/sub for cross-server messaging

---

### Presence System

**Online Detection**:
- WebSocket connection = online
- Heartbeat mechanism
- TTL in Redis (e.g., 30 seconds)

**Offline Detection**:
- No heartbeat for TTL period
- Mark as offline
- Update last seen timestamp

**Status Types**:
- **Online**: Active connection
- **Offline**: No connection
- **Away**: No activity for X minutes

---

### Group Chat

**Challenges**:
- Fan-out to all group members
- Handle large groups (thousands of members)
- Message ordering

**Solutions**:
- **Small Groups (< 100)**: Fan-out immediately
- **Large Groups**: Use message queue, async delivery
- **Ordering**: Use message queue with partitioning by groupId

---

### Message Ordering

**Requirements**:
- Messages in conversation order
- Handle out-of-order delivery

**Solution**:
- **Timestamp**: Use server timestamp
- **Sequence Number**: Per-chat sequence number
- **Client Handling**: Client sorts by timestamp/sequence

---

## Scalability

### Horizontal Scaling

- **WebSocket Servers**: Stateless (except connections), scale horizontally
- **Message Service**: Stateless, scale horizontally
- **Database Sharding**: Shard by chatId or userId

### Read Scaling

- **Read Replicas**: Database read replicas for history
- **Caching**: Cache recent messages in Redis
- **CDN**: Serve media files

### Write Scaling

- **Message Queue**: Buffer writes, async processing
- **Database Sharding**: Distribute writes across shards
- **Partitioning**: Partition by chatId for even distribution

---

## Reliability

### High Availability

- **Multiple WebSocket Servers**: No single point of failure
- **Database Replication**: Replicate across data centers
- **Message Queue Replication**: Kafka replication
- **Health Checks**: Monitor all components

### Message Delivery Guarantees

- **At-Least-Once**: Messages delivered at least once
- **Idempotency**: Handle duplicate messages
- **Retry Logic**: Retry failed deliveries
- **Dead Letter Queue**: Handle undeliverable messages

### Fault Tolerance

- **WebSocket Failures**: Reconnect, resume from last message
- **Service Failures**: Queue messages, deliver when service recovers
- **Database Failures**: Read from replicas

---

## Trade-offs

### Consistency vs Availability

- **Message Delivery**: Prioritize availability (AP)
- **Presence**: Eventual consistency acceptable
- **Read Receipts**: Eventual consistency acceptable

### Latency vs Throughput

- **Online Users**: Optimize for latency (direct push)
- **Offline Users**: Optimize for throughput (batch processing)

### Storage vs Performance

- **Message History**: Store all messages (storage cost)
- **Caching**: Cache recent messages (memory cost)

---

## Extensions

### Additional Features

1. **Voice/Video Calls**: WebRTC integration
2. **File Sharing**: Large file handling
3. **Message Search**: Full-text search
4. **Message Encryption**: End-to-end encryption
5. **Message Reactions**: Emoji reactions
6. **Message Threading**: Reply threads
7. **Channels**: Public channels (like Slack)
8. **Bots**: Chatbot integration

---

## Key Takeaways

- **WebSockets**: Real-time bidirectional communication
- **Message Queue**: Handle high message volume asynchronously
- **Time-Series Database**: Cassandra for message storage
- **Presence**: Redis for fast presence lookups
- **Fan-out**: Efficiently deliver to group members
- **Scalability**: Stateless services enable horizontal scaling

---

## Related Topics

- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Message queue patterns
- **[Service Discovery & Service Mesh]({{< ref "../system-components/6-Service_Discovery_Service_Mesh.md" >}})** - Service communication
- **[Databases]({{< ref "../databases/_index.md" >}})** - Database selection for time-series data
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Caching presence and messages

