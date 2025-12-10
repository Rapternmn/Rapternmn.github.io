+++
title = "Job Scheduler Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 14
description = "Job Scheduler implementation: Task scheduling, cron-like jobs, priority queues, job execution, and thread pool management."
+++

# 📅 Job Scheduler Design

## Problem Statement

Design a job scheduler that:
- Schedules jobs to run at specific times
- Supports one-time and recurring jobs
- Manages job priority
- Executes jobs using thread pool
- Handles job cancellation and status tracking

**Use Cases**:
- Cron jobs
- Scheduled tasks
- Periodic data processing
- Background job execution
- Task queues

---

## Requirements

1. **Scheduling**: Schedule jobs with delay or specific time
2. **Recurring Jobs**: Support periodic execution
3. **Priority**: Execute high-priority jobs first
4. **Thread Pool**: Use thread pool for execution
5. **Job Management**: Cancel, pause, resume jobs

---

## Basic Job Scheduler

```python
import threading
import time
import heapq
from typing import Callable, Optional, Any
from enum import Enum
from dataclasses import dataclass

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    """Job definition"""
    job_id: str
    func: Callable
    args: tuple
    kwargs: dict
    scheduled_time: float
    priority: int = 0
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Exception = None

class JobScheduler:
    """
    Basic job scheduler with priority support
    """
    
    def __init__(self, num_workers: int = 3):
        """
        Args:
            num_workers: Number of worker threads
        """
        self.jobs = []  # Min heap: (scheduled_time, priority, job_id, job)
        self.job_map = {}  # job_id -> job
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.next_job_id = 0
        self.running = True
        
        # Worker threads
        self.workers = []
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, name=f"Worker-{i}", daemon=True)
            worker.start()
            self.workers.append(worker)
        
        # Scheduler thread
        self.scheduler = threading.Thread(target=self._schedule_jobs, daemon=True)
        self.scheduler.start()
    
    def _schedule_jobs(self):
        """Scheduler thread that adds ready jobs to execution queue"""
        while self.running:
            with self.condition:
                now = time.time()
                
                # Move ready jobs to execution
                ready_jobs = []
                remaining = []
                
                for item in self.jobs:
                    scheduled_time, priority, job_id, job = item
                    if scheduled_time <= now and job.status == JobStatus.PENDING:
                        ready_jobs.append((priority, job_id, job))
                    else:
                        remaining.append(item)
                
                self.jobs = remaining
                heapq.heapify(self.jobs)
                
                # Notify workers about ready jobs
                for _, job_id, job in ready_jobs:
                    self.condition.notify_all()
            
            time.sleep(0.1)  # Check every 100ms
    
    def _worker(self):
        """Worker thread that executes jobs"""
        while self.running:
            with self.condition:
                # Find ready job
                ready_job = None
                now = time.time()
                
                for item in self.jobs:
                    scheduled_time, priority, job_id, job = item
                    if scheduled_time <= now and job.status == JobStatus.PENDING:
                        ready_job = (priority, job_id, job)
                        # Remove from heap
                        self.jobs.remove(item)
                        heapq.heapify(self.jobs)
                        break
                
                if ready_job:
                    _, job_id, job = ready_job
                    job.status = JobStatus.RUNNING
                else:
                    self.condition.wait(timeout=1.0)
                    continue
            
            # Execute job (outside lock)
            if ready_job:
                _, job_id, job = ready_job
                try:
                    result = job.func(*job.args, **job.kwargs)
                    job.result = result
                    job.status = JobStatus.COMPLETED
                except Exception as e:
                    job.error = e
                    job.status = JobStatus.FAILED
                    print(f"Job {job_id} failed: {e}")
    
    def schedule(
        self,
        func: Callable,
        delay: float = 0,
        priority: int = 0,
        *args,
        **kwargs
    ) -> str:
        """
        Schedule a job
        
        Args:
            func: Function to execute
            delay: Delay in seconds
            priority: Job priority (lower = higher priority)
            *args, **kwargs: Function arguments
        
        Returns:
            Job ID
        """
        with self.condition:
            job_id = f"job_{self.next_job_id}"
            self.next_job_id += 1
            
            scheduled_time = time.time() + delay
            job = Job(
                job_id=job_id,
                func=func,
                args=args,
                kwargs=kwargs,
                scheduled_time=scheduled_time,
                priority=priority
            )
            
            heapq.heappush(self.jobs, (scheduled_time, priority, job_id, job))
            self.job_map[job_id] = job
            self.condition.notify()
            return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        with self.lock:
            job = self.job_map.get(job_id)
            return job.status if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel pending job"""
        with self.lock:
            job = self.job_map.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                # Remove from heap
                self.jobs = [(t, p, jid, j) for t, p, jid, j in self.jobs if jid != job_id]
                heapq.heapify(self.jobs)
                return True
            return False
    
    def shutdown(self):
        """Shutdown scheduler"""
        self.running = False
        with self.condition:
            self.condition.notify_all()


# Usage
scheduler = JobScheduler(num_workers=2)

def task(name: str, duration: int):
    print(f"Task {name} started")
    time.sleep(duration)
    print(f"Task {name} completed")
    return f"Result from {name}"

# Schedule jobs
job1 = scheduler.schedule(task, delay=1.0, priority=1, name="High Priority", duration=2)
job2 = scheduler.schedule(task, delay=2.0, priority=2, name="Low Priority", duration=1)

time.sleep(5)
scheduler.shutdown()
```

---

## Advanced: Cron-like Job Scheduler

```python
import threading
import time
from typing import Optional
import re

class CronScheduler:
    """
    Cron-like job scheduler with recurring jobs
    """
    
    def __init__(self, num_workers: int = 3):
        self.jobs = []
        self.job_map = {}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        
        self.workers = []
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.scheduler = threading.Thread(target=self._schedule_jobs, daemon=True)
        self.scheduler.start()
    
    def _parse_cron(self, cron_expr: str) -> dict:
        """
        Parse cron expression (simplified)
        Format: "minute hour day month weekday"
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("Invalid cron expression")
        
        return {
            'minute': parts[0],
            'hour': parts[1],
            'day': parts[2],
            'month': parts[3],
            'weekday': parts[4]
        }
    
    def _next_execution_time(self, cron_expr: str, last_time: float) -> float:
        """Calculate next execution time from cron expression"""
        # Simplified: For now, just schedule every minute if * * * * *
        # Full implementation would parse cron properly
        if cron_expr == "* * * * *":
            return last_time + 60  # Every minute
        # Add more parsing logic here
        return last_time + 60
    
    def schedule_cron(
        self,
        func: Callable,
        cron_expr: str,
        *args,
        **kwargs
    ) -> str:
        """
        Schedule recurring job with cron expression
        
        Args:
            func: Function to execute
            cron_expr: Cron expression (e.g., "* * * * *" for every minute)
            *args, **kwargs: Function arguments
        """
        with self.condition:
            job_id = f"cron_{time.time()}"
            next_time = time.time() + 60  # Start in 1 minute
            
            job = {
                'job_id': job_id,
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'cron_expr': cron_expr,
                'next_time': next_time,
                'status': 'active'
            }
            
            heapq.heappush(self.jobs, (next_time, job_id, job))
            self.job_map[job_id] = job
            return job_id
    
    def _schedule_jobs(self):
        """Schedule and reschedule jobs"""
        while self.running:
            with self.condition:
                now = time.time()
                
                # Execute ready jobs and reschedule
                ready = []
                remaining = []
                
                for item in self.jobs:
                    next_time, job_id, job = item
                    if next_time <= now and job['status'] == 'active':
                        ready.append(job)
                        # Reschedule
                        job['next_time'] = self._next_execution_time(
                            job['cron_expr'], now
                        )
                        remaining.append((job['next_time'], job_id, job))
                    else:
                        remaining.append(item)
                
                self.jobs = remaining
                heapq.heapify(self.jobs)
                
                # Notify workers
                if ready:
                    self.condition.notify_all()
            
            time.sleep(1.0)  # Check every second
    
    def _worker(self):
        """Execute jobs"""
        while self.running:
            with self.condition:
                ready_job = None
                now = time.time()
                
                for item in self.jobs:
                    next_time, job_id, job = item
                    if next_time <= now and job['status'] == 'active':
                        ready_job = job
                        break
                
                if not ready_job:
                    self.condition.wait(timeout=1.0)
                    continue
            
            # Execute
            if ready_job:
                try:
                    ready_job['func'](*ready_job['args'], **ready_job['kwargs'])
                except Exception as e:
                    print(f"Job {ready_job['job_id']} error: {e}")
    
    def cancel_cron(self, job_id: str):
        """Cancel cron job"""
        with self.lock:
            if job_id in self.job_map:
                self.job_map[job_id]['status'] = 'cancelled'


# Usage
scheduler = CronScheduler()

def periodic_task():
    print(f"Periodic task at {time.time()}")

# Schedule every minute
job_id = scheduler.schedule_cron(periodic_task, "* * * * *")

time.sleep(130)  # Run for 2+ minutes
scheduler.cancel_cron(job_id)
```

---

## Key Takeaways

- Job schedulers manage task execution timing
- Priority queues enable priority-based execution
- Cron-like scheduling supports recurring jobs
- Thread pools execute jobs efficiently
- Job status tracking enables monitoring

