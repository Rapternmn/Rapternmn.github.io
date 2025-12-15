+++
title = "Video Streaming Platform (YouTube/Netflix)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 7
description = "Design a video streaming platform like YouTube or Netflix. Covers video upload, transcoding, CDN delivery, adaptive bitrate streaming, and scaling to billions of views."
+++

---

## Problem Statement

Design a video streaming platform that allows users to upload, store, and stream videos. The system should handle video processing, deliver videos efficiently worldwide, and support adaptive streaming.

**Examples**: YouTube, Netflix, Vimeo, Twitch

---

## Requirements Clarification

### Functional Requirements

1. **Video Upload**: Users can upload videos
2. **Video Processing**: Transcode videos to multiple formats/qualities
3. **Video Streaming**: Stream videos to viewers
4. **Video Search**: Search videos by title, description, tags
5. **Video Recommendations**: Recommend videos to users
6. **Comments & Likes**: Users can comment and like videos
7. **Live Streaming**: Support live video streaming

### Non-Functional Requirements

- **Scale**: 
  - 1B users
  - 500M daily active users
  - 1M video uploads/day
  - 1B video views/day
  - Average video size: 100 MB
  - Average video length: 10 minutes
- **Latency**: < 2 seconds to start playback
- **Availability**: 99.9% uptime
- **Bandwidth**: Handle high video streaming bandwidth

---

## Capacity Estimation

### Traffic Estimates

- **Uploads**: 1M videos/day = ~12 videos/second
- **Peak uploads**: 3x = ~36 videos/second
- **Views**: 1B views/day = ~11.6K views/second
- **Peak views**: 3x = ~35K views/second
- **Concurrent Streams**: Assume 10% of views concurrent = 3.5K concurrent streams

### Storage Estimates

- **Video Size**: Average 100 MB per video
- **Daily Uploads**: 1M × 100 MB = 100 TB/day
- **Yearly Storage**: ~36 PB/year (raw videos)
- **After Transcoding**: Multiple qualities (240p, 360p, 480p, 720p, 1080p, 4K)
- **Storage Multiplier**: ~5x (different qualities) = ~180 PB/year

### Bandwidth Estimates

- **Upload Bandwidth**: 36 videos/sec × 100 MB = 3.6 GB/sec
- **Streaming Bandwidth**: 
  - Average bitrate: 2 Mbps
  - 35K streams × 2 Mbps = 70 Gbps
  - Peak: 3x = 210 Gbps

---

## API Design

### REST APIs

```
POST /api/v1/videos/upload
Request: Multipart form data with video file
Response: {
  "videoId": "video123",
  "status": "processing",
  "uploadUrl": "https://storage.example.com/..."
}

GET /api/v1/videos/{videoId}
Response: {
  "videoId": "video123",
  "title": "Video Title",
  "description": "...",
  "thumbnailUrl": "...",
  "streamUrl": "https://cdn.example.com/...",
  "qualities": ["240p", "360p", "480p", "720p", "1080p"]
}

GET /api/v1/videos/{videoId}/stream
Query: ?quality=720p
Response: Video stream (HLS/DASH)

GET /api/v1/videos/search
Query: ?q=query&limit=20
Response: {
  "videos": [...],
  "total": 1000
}
```

---

## Database Design

### Schema

**Videos Table** (PostgreSQL):
```
videoId (PK): UUID
userId: UUID
title: VARCHAR
description: TEXT
tags: ARRAY[VARCHAR]
status: VARCHAR (uploading, processing, ready, failed)
duration: INT (seconds)
thumbnailUrl: TEXT
createdAt: TIMESTAMP
viewCount: BIGINT
likeCount: INT
```

**Video Qualities Table** (PostgreSQL):
```
videoId (PK): UUID
quality (PK): VARCHAR (240p, 360p, 480p, 720p, 1080p, 4K)
fileUrl: TEXT
bitrate: INT
fileSize: BIGINT
```

**User Watch History Table** (Cassandra):
```
userId (PK): UUID
videoId: UUID
watchedAt: TIMESTAMP
watchDuration: INT (seconds)
completed: BOOLEAN
```

### Database Selection

**Videos Metadata**: **PostgreSQL** (relational data, search)
**Watch History**: **Cassandra** (time-series, high write volume)
**Video Files**: **Object Storage** (S3) - not database

---

## High-Level Design

### Architecture

```
Upload: Client → API Gateway → Upload Service → Object Storage
                                              ↓
                                    [Transcoding Queue]
                                              ↓
                                    [Transcoding Service]
                                              ↓
                                    [Object Storage (CDN)]

Streaming: Client → CDN → Video Files
                ↓
        [Metadata Service] → Database
```

### Components

1. **Upload Service**: Handle video uploads
2. **Object Storage (S3)**: Store video files
3. **Transcoding Service**: Convert videos to multiple qualities
4. **Message Queue**: Queue transcoding jobs
5. **CDN**: Deliver videos globally
6. **Metadata Service**: Serve video metadata
7. **Search Service**: Search videos
8. **Recommendation Service**: Recommend videos

---

## Detailed Design

### Video Upload Flow

1. **Client** → Upload Service: Initiate upload
2. **Upload Service**:
   - Generate upload URL (presigned S3 URL)
   - Return to client
3. **Client** → S3: Upload video directly (multipart upload)
4. **S3** → Upload Service: Upload complete notification
5. **Upload Service**:
   - Store metadata in database
   - Queue transcoding job
   - Return video ID

---

### Video Transcoding

**Why Transcode**:
- Multiple qualities for adaptive streaming
- Different formats (HLS, DASH, MP4)
- Thumbnail generation
- Video optimization

**Transcoding Process**:
1. **Message Queue** → Transcoding Service: Consume job
2. **Transcoding Service**:
   - Download video from S3
   - Transcode to multiple qualities (240p, 360p, 480p, 720p, 1080p, 4K)
   - Generate thumbnails
   - Upload transcoded files to S3
   - Update database with file URLs
3. **Update Status**: Mark video as "ready"

**Transcoding Formats**:
- **HLS** (HTTP Live Streaming): Apple's format
- **DASH** (Dynamic Adaptive Streaming): Open standard
- **MP4**: Fallback format

---

### Adaptive Bitrate Streaming

**How it works**:
- Video encoded at multiple bitrates/qualities
- Client selects quality based on network conditions
- Automatically switches quality during playback

**Benefits**:
- Smooth playback on varying network speeds
- Better user experience
- Reduced buffering

**Implementation**:
- **HLS**: Multiple `.m3u8` playlists with different bitrates
- **DASH**: `manifest.mpd` with quality representations
- Client player selects appropriate quality

---

### CDN Strategy

**Why CDN**:
- Reduce latency (serve from edge locations)
- Reduce origin server load
- Handle high bandwidth

**CDN Architecture**:
- **Edge Servers**: Serve videos from edge locations
- **Origin Server**: S3 (source of truth)
- **Cache**: Cache popular videos at edge
- **Cache Invalidation**: Invalidate on video update

**CDN Selection**:
- **CloudFront** (AWS)
- **Cloudflare**
- **Fastly**

---

### Video Streaming Flow

1. **Client** → Metadata Service: Request video
2. **Metadata Service**:
   - Fetch video metadata from database
   - Return stream URLs (CDN URLs)
3. **Client** → CDN: Request video chunks
4. **CDN**:
   - Check cache
   - If cached: Serve from cache
   - If not cached: Fetch from origin (S3), cache, and serve
5. **Client**: Play video using adaptive streaming

---

### Search Functionality

**Search Approaches**:
1. **Full-Text Search**: PostgreSQL full-text search
2. **Elasticsearch**: Advanced search with ranking
3. **Hybrid**: Metadata in PostgreSQL, full-text in Elasticsearch

**Search Fields**:
- Title
- Description
- Tags
- Transcript (if available)

**Recommendation**: **Elasticsearch** for better search and ranking

---

## Scalability

### Horizontal Scaling

- **Stateless Services**: Upload, metadata, transcoding services scale horizontally
- **CDN**: Automatically scales
- **Object Storage**: Scales automatically (S3)

### Upload Scaling

- **Direct Upload**: Upload directly to S3 (bypasses servers)
- **Multipart Upload**: Large file uploads
- **Resumable Uploads**: Resume interrupted uploads

### Streaming Scaling

- **CDN**: Handles most streaming traffic
- **Edge Caching**: Cache popular videos
- **Origin Protection**: CDN protects origin servers

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **CDN Redundancy**: Multiple CDN providers
- **Object Storage Replication**: S3 replication across regions

### Fault Tolerance

- **Transcoding Failures**: Retry transcoding jobs
- **Upload Failures**: Resume uploads
- **CDN Failures**: Fallback to origin

### Data Durability

- **Object Storage**: S3 provides 99.999999999% durability
- **Backup**: Regular backups of metadata
- **Replication**: Replicate across regions

---

## Trade-offs

### Latency vs Cost

- **CDN**: Lower latency, higher cost
- **Origin Only**: Higher latency, lower cost

### Storage vs Quality

- **Multiple Qualities**: Higher storage, better experience
- **Single Quality**: Lower storage, worse experience

### Processing vs Storage

- **Transcoding**: Higher processing cost, lower storage (optimized)
- **No Transcoding**: Lower processing, higher storage (raw)

---

## Extensions

### Additional Features

1. **Live Streaming**: Real-time video streaming (WebRTC, HLS)
2. **Video Analytics**: Watch time, engagement metrics
3. **Video Comments**: Real-time comments
4. **Video Playlists**: Create and share playlists
5. **Video Subtitles**: Multiple language subtitles
6. **Video Chapters**: Chapter markers in videos
7. **Video Monetization**: Ads, subscriptions

---

## Key Takeaways

- **CDN**: Critical for video delivery at scale
- **Transcoding**: Multiple qualities for adaptive streaming
- **Object Storage**: Store videos in object storage (S3)
- **Adaptive Streaming**: HLS/DASH for smooth playback
- **Direct Upload**: Upload directly to S3 to reduce server load
- **Scalability**: CDN handles most traffic, services scale horizontally

---

## Related Topics

- **[CDN & Content Delivery]({{< ref "../system-components/7-CDN_Content_Delivery.md" >}})** - CDN architecture
- **[Object Storage]({{< ref "../databases/_index.md" >}})** - Object storage for videos
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Transcoding queue
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Scaling strategies

