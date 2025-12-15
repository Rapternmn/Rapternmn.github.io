+++
title = "Deferred Callback Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Deferred Callback implementation: Scheduled callbacks, delayed execution, callback registration, and thread-safe callback management."
+++

# ⏰ Deferred Callback Design

## Problem Statement

Design a deferred callback system that:
- Schedules callbacks to execute after a delay
- Supports one-time and recurring callbacks
- Manages callback lifecycle (register, cancel, execute)
- Thread-safe callback registration and execution
- Efficient scheduling and execution

**Use Cases**:
- Scheduled tasks
- Timeout handling
- Retry mechanisms
- Periodic jobs
- Event scheduling

---

## Requirements

1. **Scheduling**: Register callbacks with delay
2. **Execution**: Execute callbacks at scheduled time
3. **Cancellation**: Cancel pending callbacks
4. **Thread Safety**: Safe concurrent registration
5. **Efficiency**: Fast scheduling and execution

---

## Basic Deferred Callback

```python
import threading
import time
import heapq
from typing import Callable, Optional, Any

class DeferredCallback:
    """
    Basic deferred callback scheduler
    """
    
    def __init__(self):
        self.callbacks = []  # Min heap: (execution_time, callback_id, callback, args, kwargs)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_id = 0
        self.running = True
        
        # Start executor thread
        self.executor = threading.Thread(target=self._execute_callbacks, daemon=True)
        self.executor.start()
    
    def _execute_callbacks(self):
        """Executor thread that runs callbacks"""
        while self.running:
            with self.condition:
                now = time.time()
                
                # Execute all due callbacks
                while self.callbacks and self.callbacks[0][0] <= now:
                    _, _, callback, args, kwargs = heapq.heappop(self.callbacks)
                    try:
                        callback(*args, **kwargs)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                # Wait until next callback or timeout
                if self.callbacks:
                    wait_time = self.callbacks[0][0] - time.time()
                    if wait_time > 0:
                        self.condition.wait(timeout=wait_time)
                else:
                    self.condition.wait(timeout=1.0)
    
    def schedule(self, callback: Callable, delay: float, *args, **kwargs) -> int:
        """
        Schedule a callback to execute after delay
        
        Args:
            callback: Function to execute
            delay: Delay in seconds
            *args, **kwargs: Arguments for callback
        
        Returns:
            Callback ID for cancellation
        """
        with self.condition:
            callback_id = self.next_id
            self.next_id += 1
            execution_time = time.time() + delay
            
            heapq.heappush(
                self.callbacks,
                (execution_time, callback_id, callback, args, kwargs)
            )
            self.condition.notify()
            return callback_id
    
    def cancel(self, callback_id: int) -> bool:
        """
        Cancel a scheduled callback
        
        Returns:
            True if cancelled, False if not found
        """
        with self.condition:
            # Find and remove callback
            for i, (_, cid, _, _, _) in enumerate(self.callbacks):
                if cid == callback_id:
                    self.callbacks.pop(i)
                    heapq.heapify(self.callbacks)  # Re-heapify
                    return True
            return False
    
    def shutdown(self):
        """Shutdown the callback executor"""
        self.running = False
        with self.condition:
            self.condition.notify()


# Usage
scheduler = DeferredCallback()

def greet(name):
    print(f"Hello, {name}!")

def goodbye(name):
    print(f"Goodbye, {name}!")

# Schedule callbacks
id1 = scheduler.schedule(greet, 2.0, "Alice")
id2 = scheduler.schedule(goodbye, 5.0, "Alice")

time.sleep(6)
scheduler.shutdown()
```

---

## Advanced: Recurring Callbacks

```python
import threading
import time
import heapq
from typing import Callable, Optional

class RecurringCallback:
    """
    Deferred callback with recurring execution support
    """
    
    def __init__(self):
        self.callbacks = []  # (execution_time, callback_id, callback, interval, args, kwargs)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_id = 0
        self.running = True
        self.executor = threading.Thread(target=self._execute_callbacks, daemon=True)
        self.executor.start()
    
    def _execute_callbacks(self):
        """Execute callbacks and reschedule recurring ones"""
        while self.running:
            with self.condition:
                now = time.time()
                
                # Execute due callbacks
                due_callbacks = []
                remaining = []
                
                for item in self.callbacks:
                    exec_time, cid, callback, interval, args, kwargs = item
                    if exec_time <= now:
                        due_callbacks.append((cid, callback, interval, args, kwargs))
                    else:
                        remaining.append(item)
                
                self.callbacks = remaining
                heapq.heapify(self.callbacks)
                
                # Execute callbacks
                for cid, callback, interval, args, kwargs in due_callbacks:
                    try:
                        callback(*args, **kwargs)
                    except Exception as e:
                        print(f"Callback {cid} error: {e}")
                    
                    # Reschedule if recurring
                    if interval is not None:
                        next_time = time.time() + interval
                        heapq.heappush(
                            self.callbacks,
                            (next_time, cid, callback, interval, args, kwargs)
                        )
                
                # Wait for next callback
                if self.callbacks:
                    wait_time = self.callbacks[0][0] - time.time()
                    if wait_time > 0:
                        self.condition.wait(timeout=wait_time)
                else:
                    self.condition.wait(timeout=1.0)
    
    def schedule(self, callback: Callable, delay: float, *args, **kwargs) -> int:
        """Schedule one-time callback"""
        return self.schedule_recurring(callback, delay, None, *args, **kwargs)
    
    def schedule_recurring(
        self,
        callback: Callable,
        initial_delay: float,
        interval: Optional[float],
        *args,
        **kwargs
    ) -> int:
        """
        Schedule recurring callback
        
        Args:
            callback: Function to execute
            initial_delay: Initial delay before first execution
            interval: Interval between executions (None = one-time)
            *args, **kwargs: Arguments for callback
        
        Returns:
            Callback ID
        """
        with self.condition:
            callback_id = self.next_id
            self.next_id += 1
            execution_time = time.time() + initial_delay
            
            heapq.heappush(
                self.callbacks,
                (execution_time, callback_id, callback, interval, args, kwargs)
            )
            self.condition.notify()
            return callback_id
    
    def cancel(self, callback_id: int) -> bool:
        """Cancel callback"""
        with self.condition:
            for i, (_, cid, _, _, _, _) in enumerate(self.callbacks):
                if cid == callback_id:
                    self.callbacks.pop(i)
                    heapq.heapify(self.callbacks)
                    return True
            return False
    
    def shutdown(self):
        """Shutdown executor"""
        self.running = False
        with self.condition:
            self.condition.notify()


# Usage
scheduler = RecurringCallback()

def periodic_task():
    print(f"Periodic task at {time.time()}")

# Schedule recurring callback (every 2 seconds)
id1 = scheduler.schedule_recurring(periodic_task, 1.0, 2.0)

time.sleep(10)
scheduler.cancel(id1)
scheduler.shutdown()
```

---

## Real-World: Timeout Handler

```python
import threading
import time

class TimeoutHandler:
    """
    Deferred callback for timeout management
    """
    
    def __init__(self):
        self.scheduler = RecurringCallback()
        self.timeouts = {}  # operation_id -> callback_id
        self.lock = threading.Lock()
    
    def set_timeout(
        self,
        operation_id: str,
        timeout_callback: Callable,
        timeout_seconds: float
    ):
        """
        Set timeout for an operation
        
        Args:
            operation_id: Unique operation identifier
            timeout_callback: Callback to execute on timeout
            timeout_seconds: Timeout duration
        """
        def timeout_handler():
            with self.lock:
                if operation_id in self.timeouts:
                    del self.timeouts[operation_id]
                    timeout_callback()
        
        with self.lock:
            # Cancel existing timeout if any
            if operation_id in self.timeouts:
                self.scheduler.cancel(self.timeouts[operation_id])
            
            # Schedule new timeout
            callback_id = self.scheduler.schedule(timeout_handler, timeout_seconds)
            self.timeouts[operation_id] = callback_id
    
    def clear_timeout(self, operation_id: str):
        """Clear timeout for operation"""
        with self.lock:
            if operation_id in self.timeouts:
                self.scheduler.cancel(self.timeouts[operation_id])
                del self.timeouts[operation_id]


# Usage
handler = TimeoutHandler()

def on_timeout():
    print("Operation timed out!")

# Set timeout
handler.set_timeout("op1", on_timeout, 5.0)

# Clear timeout before it fires
time.sleep(2)
handler.clear_timeout("op1")
print("Timeout cleared")

time.sleep(5)  # Timeout won't fire
```

---

## Key Takeaways

- Deferred callbacks enable scheduled execution
- Heap-based scheduling for efficient execution
- Support one-time and recurring callbacks
- Thread-safe registration and execution
- Useful for timeouts, retries, and periodic tasks

