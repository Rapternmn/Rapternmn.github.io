+++
title = "File Storage System (Dropbox/Google Drive)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 8
description = "Design a file storage and synchronization system like Dropbox or Google Drive. Covers file upload, sync, versioning, conflict resolution, and scaling to petabytes."
+++

---

## Problem Statement

Design a file storage and synchronization system that allows users to upload, store, and sync files across multiple devices. The system should handle file versioning, conflict resolution, and efficient synchronization.

**Examples**: Dropbox, Google Drive, OneDrive, iCloud

---

## Requirements Clarification

### Functional Requirements

1. **File Upload**: Upload files to cloud storage
2. **File Download**: Download files from cloud storage
3. **File Sync**: Sync files across multiple devices
4. **File Versioning**: Keep version history
5. **File Sharing**: Share files/folders with other users
6. **File Search**: Search files by name, content
7. **Collaboration**: Multiple users edit same file

### Non-Functional Requirements

- **Scale**: 
  - 500M users
  - 100M daily active users
  - 1B files stored
  - Average 10 GB per user
  - Total storage: 5 PB
- **Latency**: < 200ms for metadata operations, < 1 second for file operations
- **Availability**: 99.9% uptime
- **Durability**: 99.999999999% (11 nines)

---

## Capacity Estimation

### Traffic Estimates

- **File Uploads**: 100M DAU × 5 uploads/day = 500M uploads/day
- **Uploads/sec**: ~5,800 uploads/second
- **Peak uploads**: 3x = ~17K uploads/second
- **File Downloads**: 100M DAU × 10 downloads/day = 1B downloads/day
- **Downloads/sec**: ~11.6K downloads/second

### Storage Estimates

- **Total Storage**: 5 PB
- **Average File Size**: 1 MB
- **Total Files**: ~5B files
- **Metadata per File**: ~1 KB
- **Total Metadata**: ~5 TB
- **Version History**: Assume 10 versions per file = 50 PB

---

## API Design

### REST APIs

```
POST /api/v1/files/upload
Request: Multipart form data with file
Headers: {
  "X-Parent-Folder-Id": "folder123",
  "X-Client-Version": "v1"
}
Response: {
  "fileId": "file456",
  "version": 1,
  "size": 1024000,
  "uploadedAt": "2025-12-15T10:00:00Z"
}

GET /api/v1/files/{fileId}
Response: {
  "fileId": "file456",
  "name": "document.pdf",
  "size": 1024000,
  "version": 1,
  "downloadUrl": "https://storage.example.com/...",
  "versions": [...]
}

GET /api/v1/files/{fileId}/download
Query: ?version=1
Response: File stream

POST /api/v1/files/{fileId}/sync
Request: {
  "clientVersion": 1,
  "lastSyncTime": "2025-12-15T09:00:00Z"
}
Response: {
  "changes": [...],
  "conflicts": [...]
}

POST /api/v1/files/{fileId}/share
Request: {
  "userId": "user789",
  "permission": "read"  // read, write
}
```

---

## Database Design

### Schema

**Files Table** (PostgreSQL):
```
fileId (PK): UUID
userId: UUID
name: VARCHAR
parentFolderId: UUID (nullable)
size: BIGINT
mimeType: VARCHAR
currentVersion: INT
status: VARCHAR (uploading, ready, deleted)
createdAt: TIMESTAMP
updatedAt: TIMESTAMP
```

**File Versions Table** (PostgreSQL):
```
fileId (PK): UUID
version (PK): INT
fileUrl: TEXT
size: BIGINT
checksum: VARCHAR (MD5/SHA256)
createdAt: TIMESTAMP
createdBy: UUID (device/user)
```

**File Metadata Table** (Elasticsearch - for search):
```
fileId: UUID
name: TEXT
content: TEXT (if text file)
tags: ARRAY[VARCHAR]
userId: UUID
```

**Sync State Table** (PostgreSQL):
```
userId (PK): UUID
deviceId (PK): UUID
lastSyncTime: TIMESTAMP
syncToken: VARCHAR
```

### Database Selection

**File Metadata**: **PostgreSQL** (relational data, ACID)
**File Search**: **Elasticsearch** (full-text search)
**File Storage**: **Object Storage** (S3) - not database

---

## High-Level Design

### Architecture

```
Client → API Gateway → File Service
                        ↓
                [Object Storage (S3)]
                        ↓
                [Metadata Database]
                        ↓
                [Sync Service] → Message Queue
                        ↓
                [Versioning Service]
```

### Components

1. **File Service**: Core file operations
2. **Object Storage (S3)**: Store file contents
3. **Metadata Database**: Store file metadata
4. **Sync Service**: Handle file synchronization
5. **Versioning Service**: Manage file versions
6. **Search Service**: Search files
7. **Sharing Service**: Handle file sharing
8. **CDN**: Serve files globally

---

## Detailed Design

### File Upload Flow

1. **Client** → File Service: Initiate upload
2. **File Service**:
   - Generate upload URL (presigned S3 URL)
   - Create file metadata record
   - Return upload URL and file ID
3. **Client** → S3: Upload file directly (multipart upload for large files)
4. **S3** → File Service: Upload complete notification
5. **File Service**:
   - Calculate checksum (MD5/SHA256)
   - Store file version
   - Update metadata
   - Index for search (if text file)
   - Notify sync service

---

### File Synchronization

**Sync Challenges**:
- Multiple devices editing same file
- Network interruptions
- Conflict resolution
- Efficient sync (only changed files)

**Sync Process**:
1. **Client** → Sync Service: Request sync
   - Send: Last sync time, client file versions
2. **Sync Service**:
   - Compare client state with server state
   - Identify changes (new, modified, deleted files)
   - Detect conflicts
   - Return changes
3. **Client**: Apply changes, resolve conflicts
4. **Client** → Sync Service: Upload local changes
5. **Sync Service**: Update server state

---

### Conflict Resolution

**Conflict Scenarios**:
- Same file modified on multiple devices
- File deleted on one device, modified on another

**Resolution Strategies**:
1. **Last Write Wins**: Latest modification wins
2. **Manual Resolution**: User chooses which version
3. **Automatic Merge**: For text files, merge changes
4. **Version Both**: Keep both versions

**Recommendation**: **Version Both** (safest, user can choose later)

---

### File Versioning

**Why Version**:
- Recover deleted files
- Revert to previous versions
- Track changes
- Conflict resolution

**Version Storage**:
- Store each version in object storage
- Metadata tracks versions
- Limit versions (e.g., keep last 10 versions)
- Archive old versions to cheaper storage

**Version Cleanup**:
- Keep last N versions
- Archive versions older than X days
- Delete archived versions after Y days

---

### Delta Sync (Efficient Sync)

**Problem**: Re-upload entire file on small changes (inefficient)

**Solution**: Delta sync (only sync changes)

**How it works**:
1. **Chunking**: Divide file into chunks (e.g., 4 MB chunks)
2. **Checksums**: Calculate checksum for each chunk
3. **Compare**: Compare chunk checksums
4. **Upload**: Upload only changed chunks
5. **Reconstruct**: Reconstruct file on server

**Benefits**:
- Reduced bandwidth
- Faster sync
- Lower costs

---

### File Sharing

**Sharing Types**:
- **Public Link**: Share via public URL
- **User Sharing**: Share with specific users
- **Permission Levels**: Read-only, read-write

**Sharing Flow**:
1. **Owner** → Sharing Service: Create share
2. **Sharing Service**:
   - Generate share token
   - Store share permissions
   - Return share link
3. **Recipient**: Access file via share link
4. **Sharing Service**: Validate permissions, grant access

---

### File Search

**Search Approaches**:
1. **Metadata Search**: Search by filename, tags
2. **Content Search**: Search file contents (for text files)
3. **Hybrid**: Both metadata and content

**Implementation**:
- **Elasticsearch**: Index file metadata and content
- **OCR**: Extract text from images/PDFs
- **Full-Text Search**: Search file contents

---

## Scalability

### Horizontal Scaling

- **Stateless Services**: File service, sync service scale horizontally
- **Object Storage**: S3 scales automatically
- **Database Sharding**: Shard by userId

### Upload Scaling

- **Direct Upload**: Upload directly to S3
- **Multipart Upload**: Large file uploads
- **Resumable Uploads**: Resume interrupted uploads

### Storage Scaling

- **Object Storage**: S3 scales to exabytes
- **Tiered Storage**: Hot, warm, cold storage tiers
- **Compression**: Compress files to reduce storage

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **Object Storage Replication**: S3 replication across regions
- **Database Replication**: Master-slave replication

### Data Durability

- **Object Storage**: S3 provides 99.999999999% durability
- **Backup**: Regular backups of metadata
- **Versioning**: Multiple versions provide redundancy

### Fault Tolerance

- **Upload Failures**: Resume uploads
- **Sync Failures**: Retry sync operations
- **Service Failures**: Queue operations, process when service recovers

---

## Trade-offs

### Consistency vs Availability

- **File Metadata**: Strong consistency needed (CP)
- **Sync State**: Eventual consistency acceptable (AP)

### Storage vs Performance

- **Versioning**: Higher storage, better recovery
- **Delta Sync**: More processing, less bandwidth

### Latency vs Cost

- **CDN**: Lower latency, higher cost
- **Origin Only**: Higher latency, lower cost

---

## Extensions

### Additional Features

1. **File Collaboration**: Real-time collaborative editing
2. **File Encryption**: End-to-end encryption
3. **File Compression**: Automatic compression
4. **File Deduplication**: Store same file once
5. **File Preview**: Generate previews (images, PDFs)
6. **File Analytics**: Track file access, sharing
7. **Offline Access**: Cache files for offline access

---

## Key Takeaways

- **Object Storage**: Store files in object storage (S3)
- **Delta Sync**: Sync only changes for efficiency
- **Versioning**: Keep file versions for recovery
- **Conflict Resolution**: Handle conflicts gracefully
- **Direct Upload**: Upload directly to S3 to reduce server load
- **Scalability**: Object storage scales automatically

---

## Related Topics

- **[Object Storage]({{< ref "../databases/_index.md" >}})** - Object storage for files
- **[CDN & Content Delivery]({{< ref "../system-components/7-CDN_Content_Delivery.md" >}})** - File delivery
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Sync queue
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Scaling strategies

