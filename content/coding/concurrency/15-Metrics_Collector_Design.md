+++
title = "Metrics Collector Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 15
description = "Metrics Collector implementation: Thread-safe metrics collection, counters, gauges, histograms, and performance monitoring."
+++

# 📊 Metrics Collector Design

## Problem Statement

Design a metrics collector that:
- Collects metrics from multiple threads
- Supports counters, gauges, histograms
- Thread-safe metric updates
- Efficient aggregation and reporting
- Low overhead metric collection

**Use Cases**:
- Application monitoring
- Performance metrics
- Business metrics
- System health tracking
- Real-time analytics

---

## Requirements

1. **Thread Safety**: Safe concurrent metric updates
2. **Metric Types**: Counters, gauges, histograms, timers
3. **Aggregation**: Efficient metric aggregation
4. **Reporting**: Export metrics for monitoring
5. **Performance**: Low overhead collection

---

## Basic Metrics Collector

```python
import threading
import time
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Counter:
    """Counter metric"""
    name: str
    value: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def increment(self, amount: int = 1):
        with self.lock:
            self.value += amount
    
    def get(self) -> int:
        with self.lock:
            return self.value

@dataclass
class Gauge:
    """Gauge metric (current value)"""
    name: str
    value: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def set(self, value: float):
        with self.lock:
            self.value = value
    
    def get(self) -> float:
        with self.lock:
            return self.value

@dataclass
class Histogram:
    """Histogram metric"""
    name: str
    values: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record(self, value: float):
        with self.lock:
            self.values.append(value)
            # Keep only recent values (last 1000)
            if len(self.values) > 1000:
                self.values.pop(0)
    
    def get_stats(self) -> dict:
        with self.lock:
            if not self.values:
                return {}
            sorted_vals = sorted(self.values)
            return {
                'count': len(sorted_vals),
                'min': sorted_vals[0],
                'max': sorted_vals[-1],
                'avg': sum(sorted_vals) / len(sorted_vals),
                'p50': sorted_vals[len(sorted_vals) // 2],
                'p95': sorted_vals[int(len(sorted_vals) * 0.95)],
                'p99': sorted_vals[int(len(sorted_vals) * 0.99)]
            }

class MetricsCollector:
    """
    Thread-safe metrics collector
    """
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.lock = threading.RLock()
    
    def get_or_create_counter(self, name: str) -> Counter:
        """Get or create counter"""
        with self.lock:
            if name not in self.counters:
                self.counters[name] = Counter(name=name)
            return self.counters[name]
    
    def get_or_create_gauge(self, name: str) -> Gauge:
        """Get or create gauge"""
        with self.lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name=name)
            return self.gauges[name]
    
    def get_or_create_histogram(self, name: str) -> Histogram:
        """Get or create histogram"""
        with self.lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name=name)
            return self.histograms[name]
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment counter"""
        counter = self.get_or_create_counter(name)
        counter.increment(amount)
    
    def set_gauge(self, name: str, value: float):
        """Set gauge value"""
        gauge = self.get_or_create_gauge(name)
        gauge.set(value)
    
    def record_histogram(self, name: str, value: float):
        """Record histogram value"""
        histogram = self.get_or_create_histogram(name)
        histogram.record(value)
    
    def get_all_metrics(self) -> dict:
        """Get all metrics"""
        with self.lock:
            return {
                'counters': {name: counter.get() for name, counter in self.counters.items()},
                'gauges': {name: gauge.get() for name, gauge in self.gauges.items()},
                'histograms': {
                    name: histogram.get_stats()
                    for name, histogram in self.histograms.items()
                }
            }


# Usage
collector = MetricsCollector()

# Collect metrics from multiple threads
def worker(worker_id: int):
    for i in range(100):
        collector.increment_counter("requests.total")
        collector.set_gauge("active.workers", worker_id)
        
        # Record response time
        start = time.time()
        time.sleep(0.01)  # Simulate work
        duration = time.time() - start
        collector.record_histogram("response.time", duration)

# Multiple workers
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Get metrics
metrics = collector.get_all_metrics()
print(metrics)
```

---

## Advanced: Timer Metric

```python
import threading
import time
from contextlib import contextmanager

class Timer:
    """Timer metric for measuring duration"""
    
    def __init__(self, name: str, histogram: Histogram):
        self.name = name
        self.histogram = histogram
    
    @contextmanager
    def time(self):
        """Context manager for timing"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram.record(duration)

class AdvancedMetricsCollector(MetricsCollector):
    """Metrics collector with timer support"""
    
    def timer(self, name: str) -> Timer:
        """Get timer for metric"""
        histogram = self.get_or_create_histogram(name)
        return Timer(name, histogram)


# Usage
collector = AdvancedMetricsCollector()

def process_request():
    with collector.timer("request.processing"):
        time.sleep(0.1)  # Simulate processing

process_request()
```

---

## Real-World: Performance Metrics

```python
import threading
import time

class PerformanceMetrics:
    """
    Performance metrics collector for application monitoring
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.start_time = time.time()
    
    def record_request(self, endpoint: str, status_code: int, duration: float):
        """Record HTTP request"""
        self.collector.increment_counter(f"requests.{endpoint}.total")
        self.collector.increment_counter(f"requests.status.{status_code}")
        self.collector.record_histogram(f"requests.{endpoint}.duration", duration)
    
    def record_cache_hit(self, cache_name: str):
        """Record cache hit"""
        self.collector.increment_counter(f"cache.{cache_name}.hits")
    
    def record_cache_miss(self, cache_name: str):
        """Record cache miss"""
        self.collector.increment_counter(f"cache.{cache_name}.misses")
    
    def set_queue_size(self, queue_name: str, size: int):
        """Set queue size gauge"""
        self.collector.set_gauge(f"queue.{queue_name}.size", size)
    
    def get_summary(self) -> dict:
        """Get performance summary"""
        metrics = self.collector.get_all_metrics()
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'metrics': metrics
        }


# Usage: Web application
metrics = PerformanceMetrics()

# Simulate requests
def handle_request(endpoint: str):
    start = time.time()
    # Process request
    time.sleep(0.05)
    duration = time.time() - start
    metrics.record_request(endpoint, 200, duration)

handle_request("/api/users")
handle_request("/api/orders")
handle_request("/api/products")

print(metrics.get_summary())
```

---

## Key Takeaways

- Metrics collectors track application performance
- Counters track cumulative values
- Gauges track current values
- Histograms track value distributions
- Thread-safe operations enable concurrent collection
- Low overhead is critical for production use

