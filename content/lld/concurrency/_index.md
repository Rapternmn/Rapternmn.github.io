+++
title = "Concurrency"
description = "Concurrency in System Design: Threading, synchronization, thread safety, concurrency patterns, and practical implementations like Rate Limiter, Thread Pool, Producer-Consumer, and Concurrent Cache."
weight = 3
+++

This section covers concurrency concepts essential for building scalable, thread-safe systems. Topics include threading fundamentals, synchronization primitives, common concurrency problems, patterns, and practical implementations of real-world systems.

---

## 📖 Topics

- **[Concurrency Fundamentals]({{< ref "1-Concurrency_Fundamentals.md" >}})** - Overview, threads, processes, threading basics
- **[Synchronization & Thread Safety]({{< ref "2-Synchronization_Thread_Safety.md" >}})** - Locks, semaphores, race conditions, deadlocks
- **[Concurrency Patterns]({{< ref "3-Concurrency_Patterns.md" >}})** - Common patterns and best practices
- **[Rate Limiter Design]({{< ref "4-Rate_Limiter_Design.md" >}})** - Thread-safe rate limiting implementation
- **[Thread Pool Design]({{< ref "5-Thread_Pool_Design.md" >}})** - Efficient thread pool implementation
- **[Producer-Consumer Design]({{< ref "6-Producer_Consumer_Design.md" >}})** - Producer-consumer pattern implementation
- **[Concurrent Cache Design]({{< ref "7-Concurrent_Cache_Design.md" >}})** - Thread-safe cache with LRU eviction
- **[Blocking Queue Design]({{< ref "8-Blocking_Queue_Design.md" >}})** - Thread-safe blocking queue with priority support
- **[Non-Blocking Queue Design]({{< ref "9-NonBlocking_Queue_Design.md" >}})** - Lock-free non-blocking queue implementation
- **[Token Bucket Filtering]({{< ref "10-Token_Bucket_Filtering.md" >}})** - Request filtering and traffic shaping
- **[Deferred Callback Design]({{< ref "11-Deferred_Callback_Design.md" >}})** - Scheduled callbacks and delayed execution
- **[Message Broker Design]({{< ref "12-Message_Broker_Design.md" >}})** - Topic-based message routing and delivery
- **[Pub/Sub Design]({{< ref "13-PubSub_Design.md" >}})** - Publish-subscribe pattern with wildcard support
- **[Job Scheduler Design]({{< ref "14-Job_Scheduler_Design.md" >}})** - Task scheduling with cron-like jobs
- **[Metrics Collector Design]({{< ref "15-Metrics_Collector_Design.md" >}})** - Thread-safe metrics collection and monitoring
- **[Read-Write Lock Design]({{< ref "16-ReadWrite_Lock_Design.md" >}})** - Multiple readers, exclusive writer lock

