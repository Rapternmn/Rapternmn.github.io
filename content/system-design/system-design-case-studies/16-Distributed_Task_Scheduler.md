+++
title = "Distributed Task Scheduler (Airflow/Cron at Scale)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 16
description = "Design a distributed task scheduler like Airflow. Covers job scheduling, dependencies, fault tolerance, and scaling to handle millions of scheduled tasks."
+++

---

## Problem Statement

Design a distributed task scheduler that schedules and executes jobs across multiple workers. The system should handle dependencies, retries, and scale to schedule millions of tasks.

**Examples**: Apache Airflow, Cron at scale, Kubernetes CronJobs, AWS Step Functions

---

## Requirements Clarification

### Functional Requirements

1. **Job Scheduling**: Schedule jobs at specific times/intervals
2. **Dependency Management**: Handle job dependencies
3. **Job Execution**: Execute jobs on workers
4. **Retry Logic**: Retry failed jobs
5. **Job Monitoring**: Monitor job status
6. **Job History**: Store job execution history
7. **Dynamic Scheduling**: Schedule jobs dynamically

### Non-Functional Requirements

- **Scale**: 
  - 1M scheduled jobs/day
  - 10K concurrent jobs
  - 1000 workers
  - Average job duration: 5 minutes
- **Latency**: < 1 second to schedule job
- **Reliability**: 99.9% job execution success rate
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Jobs Scheduled**: 1M jobs/day = ~12 jobs/second
- **Peak Scheduling**: 3x = ~36 jobs/second
- **Concurrent Jobs**: 10K jobs
- **Job Executions**: 1M executions/day

### Storage Estimates

- **Job Metadata**: 1M jobs × 1 KB = 1 GB/day
- **Job History**: 1M executions × 2 KB = 2 GB/day
- **Monthly Storage**: ~90 GB/month

---

## API Design

### REST APIs

```
POST /api/v1/jobs
Request: {
  "name": "daily-report",
  "schedule": "0 0 * * *",  // cron expression
  "command": "python generate_report.py",
  "dependencies": ["job1", "job2"],
  "retries": 3,
  "timeout": 3600
}
Response: {
  "jobId": "job123",
  "status": "scheduled"
}

GET /api/v1/jobs/{jobId}
Response: {
  "jobId": "job123",
  "status": "running",
  "startedAt": "2025-12-15T10:00:00Z",
  "progress": 50
}

POST /api/v1/jobs/{jobId}/trigger
Response: {
  "executionId": "exec456",
  "status": "queued"
}
```

---

## Database Design

### Schema

**Jobs Table** (PostgreSQL):
```
jobId (PK): UUID
name: VARCHAR
schedule: VARCHAR (cron expression)
command: TEXT
dependencies: ARRAY[UUID]
retries: INT
timeout: INT
status: VARCHAR (active, paused, deleted)
createdAt: TIMESTAMP
```

**Job Executions Table** (PostgreSQL):
```
executionId (PK): UUID
jobId: UUID
status: VARCHAR (queued, running, success, failed)
startedAt: TIMESTAMP
completedAt: TIMESTAMP (nullable)
workerId: UUID (nullable)
retryCount: INT
errorMessage: TEXT (nullable)
lastHeartbeat: TIMESTAMP
```

**Job Dependencies Table** (PostgreSQL):
```
jobId (PK): UUID
dependsOnJobId (PK): UUID
```

**Scheduled Jobs Queue** (Message Queue - Kafka):
```
jobId: UUID
scheduledTime: TIMESTAMP
priority: INT
```

### Database Selection

**Jobs, Executions**: **PostgreSQL** (relational data, ACID)
**Scheduled Queue**: **Message Queue** (Kafka) - distributed queue
**Job History**: **Time-series Database** (optional) - for analytics

---

## High-Level Design

### Architecture

```
[Scheduler] → [Job Queue] → [Workers]
                ↓
        [Dependency Resolver]
                ↓
        [Execution Engine]
                ↓
        [Result Store]
```

### Components

1. **Scheduler**: Schedule jobs based on cron expressions
2. **Job Queue**: Queue of jobs to execute (Kafka)
3. **Dependency Resolver**: Resolve job dependencies
4. **Workers**: Execute jobs
5. **Execution Engine**: Manage job execution
6. **Result Store**: Store job results
7. **Monitor**: Monitor job status

---

## Detailed Design

### Job Scheduling

**Scheduling Types**:

1. **Cron-Based Scheduling**:
   - Schedule jobs using cron expressions
   - Examples: `0 0 * * *` (daily at midnight)
   - **Pros**: Flexible, standard
   - **Cons**: Limited to time-based

2. **Interval-Based Scheduling**:
   - Schedule jobs at fixed intervals
   - Examples: Every 5 minutes, every hour
   - **Pros**: Simple
   - **Cons**: Less flexible

3. **Event-Based Scheduling**:
   - Schedule jobs on events
   - Examples: On file upload, on data arrival
   - **Pros**: Reactive
   - **Cons**: More complex

**Recommendation**: **Cron-Based Scheduling** (most common)

---

### Dependency Management

**Dependency Types**:

1. **Sequential Dependencies**:
   - Job B depends on Job A
   - Job B runs after Job A completes
   - **Example**: Process data → Generate report

2. **Parallel Dependencies**:
   - Multiple jobs depend on one job
   - All dependent jobs run in parallel
   - **Example**: Job A → [Job B, Job C, Job D]

3. **Conditional Dependencies**:
   - Job depends on condition
   - **Example**: Job B runs if Job A succeeds

**Dependency Resolution**:
1. **Build DAG**: Build directed acyclic graph (DAG)
2. **Topological Sort**: Sort jobs by dependencies
3. **Execute**: Execute jobs in order
4. **Track**: Track job completion
5. **Trigger**: Trigger dependent jobs

---

### Job Execution

**Execution Flow**:
1. **Schedule**: Scheduler schedules job
2. **Queue**: Add job to execution queue
3. **Resolve Dependencies**: Check if dependencies met
4. **Assign Worker**: Assign job to available worker
5. **Execute**: Worker executes job
6. **Monitor**: Monitor job progress
7. **Complete**: Mark job as complete
8. **Trigger Dependents**: Trigger dependent jobs

**Worker Assignment**:
- **Round-Robin**: Distribute jobs evenly
- **Load-Based**: Assign to least loaded worker
- **Affinity**: Assign to specific worker (if needed)

---

### Retry Logic

**Retry Strategies**:
1. **Fixed Retry**: Retry N times with fixed delay
2. **Exponential Backoff**: Increase delay between retries
3. **Custom Retry**: Custom retry logic per job

**Retry Conditions**:
- **Transient Failures**: Network errors, timeouts
- **Permanent Failures**: Invalid input, code errors (don't retry)

**Retry Implementation**:
- **Retry Count**: Track retry count
- **Max Retries**: Limit retry attempts
- **Backoff**: Wait before retry
- **Dead Letter Queue**: Move failed jobs after max retries

---

### Fault Tolerance

**Fault Scenarios**:

1. **Worker Failures**:
   - **Detection**: Health checks, heartbeat
   - **Recovery**: Reassign jobs to other workers
   - **Checkpointing**: Save job progress

2. **Scheduler Failures**:
   - **Multiple Schedulers**: Leader election
   - **State Replication**: Replicate scheduler state
   - **Failover**: Promote backup scheduler

3. **Job Failures**:
   - **Retry**: Retry failed jobs
   - **Alert**: Alert on repeated failures
   - **Skip**: Skip job if too many failures

---

## Scalability

### Horizontal Scaling

- **Multiple Workers**: Scale workers horizontally
- **Multiple Schedulers**: Leader election for schedulers
- **Queue Partitioning**: Partition job queue by job type

### Performance Optimization

- **Parallel Execution**: Execute independent jobs in parallel
- **Job Batching**: Batch similar jobs
- **Caching**: Cache job results
- **Connection Pooling**: Reuse connections

---

## Reliability

### High Availability

- **Multiple Schedulers**: Leader election, failover
- **Worker Redundancy**: Multiple workers, no single point of failure
- **Queue Replication**: Kafka replication

### Data Consistency

- **Job State**: Strong consistency needed
- **Execution History**: Eventual consistency acceptable
- **Dependencies**: Strong consistency needed

---

## Trade-offs

### Consistency vs Availability

- **Job Scheduling**: Strong consistency (prevent duplicate execution)
- **Execution History**: Eventual consistency acceptable

### Latency vs Throughput

- **Immediate Execution**: Lower latency, lower throughput
- **Batched Execution**: Higher latency, higher throughput

---

## Extensions

### Additional Features

1. **Dynamic Scheduling**: Schedule jobs dynamically
2. **Job Templates**: Reusable job templates
3. **Job Versioning**: Version jobs
4. **Job Rollback**: Rollback to previous version
5. **Resource Management**: Manage worker resources
6. **Job Prioritization**: Prioritize important jobs
7. **Multi-Tenancy**: Support multiple tenants

---

## Key Takeaways

- **DAG-Based Scheduling**: Use DAGs for dependency management
- **Distributed Queue**: Use message queue for job distribution
- **Worker Pool**: Pool of workers for job execution
- **Retry Logic**: Implement retry with exponential backoff
- **Fault Tolerance**: Handle worker and scheduler failures
- **Scalability**: Scale workers horizontally

---

## Related Topics

- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Job queue
- **[Distributed Systems Fundamentals]({{< ref "../system-components/8-Distributed_Systems_Fundamentals.md" >}})** - Distributed architecture
- **[Availability & Reliability]({{< ref "../system-components/10-Availability_Reliability.md" >}})** - Fault tolerance
- **[Data Engineering - Orchestration]({{< ref "../data-engineering/9-Orchestration_Workflow.md" >}})** - Workflow orchestration

