+++
title = "Thread Pool Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Thread Pool implementation: Efficient thread reuse, task queue management, worker threads, and thread pool executor with Python implementation."
+++

# 🏊 Thread Pool Design

## Problem Statement

Design a thread pool that:
- Reuses threads to avoid creation overhead
- Manages a queue of tasks
- Supports configurable pool size
- Handles task submission and execution
- Gracefully shuts down

**Why Thread Pool?**
- Thread creation is expensive
- Limits resource consumption
- Better control over concurrency
- Reuses threads efficiently

---

## Requirements

1. **Thread Reuse**: Reuse existing threads instead of creating new ones
2. **Task Queue**: Queue tasks for execution
3. **Configurable Size**: Set minimum and maximum threads
4. **Graceful Shutdown**: Wait for tasks to complete before shutdown
5. **Thread Safety**: Safe concurrent task submission

---

## Basic Implementation

```python
import threading
import queue
import time
from typing import Callable, Any, Optional

class ThreadPool:
    """
    Simple thread pool implementation
    """
    
    def __init__(self, num_threads: int):
        """
        Args:
            num_threads: Number of worker threads
        """
        self.num_threads = num_threads
        self.tasks = queue.Queue()
        self.workers = []
        self.shutdown_flag = False
        self.lock = threading.Lock()
        
        # Start worker threads
        for i in range(num_threads):
            worker = threading.Thread(
                target=self._worker,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        """Worker thread that processes tasks"""
        while True:
            try:
                task = self.tasks.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Task failed: {e}")
                finally:
                    self.tasks.task_done()
            except queue.Empty:
                if self.shutdown_flag:
                    break
    
    def submit(self, func: Callable, *args, **kwargs):
        """
        Submit a task to the thread pool
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        if self.shutdown_flag:
            raise RuntimeError("ThreadPool is shutdown")
        
        self.tasks.put((func, args, kwargs))
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the thread pool
        
        Args:
            wait: If True, wait for all tasks to complete
        """
        self.shutdown_flag = True
        
        # Signal all workers to stop
        for _ in self.workers:
            self.tasks.put(None)
        
        if wait:
            # Wait for all tasks to complete
            self.tasks.join()
            # Wait for workers to finish
            for worker in self.workers:
                worker.join()


# Usage Example
def task(name, duration):
    print(f"Task {name} started")
    time.sleep(duration)
    print(f"Task {name} completed")

pool = ThreadPool(num_threads=3)

# Submit tasks
for i in range(10):
    pool.submit(task, f"Task-{i}", 1)

# Shutdown and wait
pool.shutdown(wait=True)
print("All tasks completed")
```

---

## Advanced: Dynamic Thread Pool

```python
import threading
import queue
import time
from typing import Callable, Optional

class DynamicThreadPool:
    """
    Thread pool with dynamic sizing
    - Core threads: Always active
    - Max threads: Maximum allowed
    - Tasks queue: Holds pending tasks
    """
    
    def __init__(
        self,
        core_threads: int,
        max_threads: int,
        queue_size: int = 0
    ):
        """
        Args:
            core_threads: Minimum number of threads
            max_threads: Maximum number of threads
            queue_size: Maximum queue size (0 = unlimited)
        """
        self.core_threads = core_threads
        self.max_threads = max_threads
        self.tasks = queue.Queue(maxsize=queue_size)
        self.workers = []
        self.active_workers = 0
        self.worker_lock = threading.Lock()
        self.shutdown_flag = False
        
        # Start core threads
        for i in range(core_threads):
            self._add_worker(f"Core-{i}")
    
    def _add_worker(self, name: str):
        """Add a new worker thread"""
        with self.worker_lock:
            if len(self.workers) >= self.max_threads:
                return False
            
            worker = threading.Thread(
                target=self._worker,
                name=name,
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            self.active_workers += 1
            return True
    
    def _worker(self):
        """Worker thread that processes tasks"""
        while True:
            try:
                # Try to get task with timeout
                task = self.tasks.get(timeout=1)
                
                if task is None:  # Shutdown signal
                    break
                
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Task error: {e}")
                finally:
                    self.tasks.task_done()
                
                # If we're above core threads and queue is empty, exit
                with self.worker_lock:
                    if (len(self.workers) > self.core_threads and 
                        self.tasks.empty()):
                        self.workers.remove(threading.current_thread())
                        self.active_workers -= 1
                        return
                        
            except queue.Empty:
                # Timeout - check if we should exit
                with self.worker_lock:
                    if self.shutdown_flag:
                        break
                    if (len(self.workers) > self.core_threads and 
                        self.tasks.empty()):
                        self.workers.remove(threading.current_thread())
                        self.active_workers -= 1
                        return
    
    def submit(self, func: Callable, *args, **kwargs):
        """
        Submit a task to the thread pool
        
        Raises:
            queue.Full: If queue is full
        """
        if self.shutdown_flag:
            raise RuntimeError("ThreadPool is shutdown")
        
        # Try to add worker if needed
        if (self.active_workers < self.max_threads and 
            self.tasks.qsize() > 0):
            self._add_worker(f"Dynamic-{time.time()}")
        
        self.tasks.put((func, args, kwargs))
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool"""
        self.shutdown_flag = True
        
        # Signal all workers
        for _ in self.workers:
            self.tasks.put(None)
        
        if wait:
            self.tasks.join()
            for worker in self.workers[:]:
                if worker.is_alive():
                    worker.join()


# Usage
pool = DynamicThreadPool(core_threads=2, max_threads=5)

def cpu_task(n):
    result = sum(i*i for i in range(n))
    print(f"Task {n} result: {result}")

# Submit many tasks
for i in range(20):
    pool.submit(cpu_task, 1000000)

pool.shutdown(wait=True)
```

---

## Future-Based Thread Pool

```python
import threading
import queue
from typing import Callable, TypeVar, Generic

T = TypeVar('T')

class Future:
    """Represents a result that will be available in the future"""
    
    def __init__(self):
        self._result = None
        self._exception = None
        self._done = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def set_result(self, result):
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()
    
    def set_exception(self, exception):
        with self._condition:
            self._exception = exception
            self._done = True
            self._condition.notify_all()
    
    def result(self, timeout=None):
        with self._condition:
            if not self._done:
                self._condition.wait(timeout)
            if self._exception:
                raise self._exception
            return self._result

class FutureThreadPool:
    """Thread pool that returns Future objects"""
    
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.tasks = queue.Queue()
        self.workers = []
        self.shutdown_flag = False
        
        for i in range(num_threads):
            worker = threading.Thread(
                target=self._worker,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        while True:
            try:
                task = self.tasks.get(timeout=1)
                if task is None:
                    break
                
                future, func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.tasks.task_done()
            except queue.Empty:
                if self.shutdown_flag:
                    break
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task and return Future"""
        if self.shutdown_flag:
            raise RuntimeError("ThreadPool is shutdown")
        
        future = Future()
        self.tasks.put((future, func, args, kwargs))
        return future
    
    def shutdown(self, wait: bool = True):
        self.shutdown_flag = True
        for _ in self.workers:
            self.tasks.put(None)
        if wait:
            self.tasks.join()
            for worker in self.workers:
                worker.join()


# Usage
pool = FutureThreadPool(num_threads=3)

def compute(n):
    return sum(i*i for i in range(n))

# Submit tasks and get futures
futures = []
for i in range(5):
    future = pool.submit(compute, 1000000)
    futures.append(future)

# Get results
for i, future in enumerate(futures):
    result = future.result()
    print(f"Task {i} result: {result}")

pool.shutdown()
```

---

## Design Considerations

1. **Pool Size**: Balance between resource usage and throughput
2. **Queue Size**: Limit queue to prevent memory issues
3. **Task Priority**: Use PriorityQueue for priority-based execution
4. **Error Handling**: Handle exceptions in tasks gracefully
5. **Monitoring**: Track active threads, queue size, task completion

---

## Key Takeaways

- Thread pools reuse threads to reduce overhead
- Dynamic pools scale based on workload
- Futures enable asynchronous task execution
- Graceful shutdown ensures tasks complete
- Thread safety is critical for concurrent submissions

