+++
title = "Pub/Sub Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 13
description = "Publish-Subscribe pattern implementation: Topic subscriptions, message broadcasting, subscriber management, and event distribution."
+++

# 📢 Pub/Sub Design

## Problem Statement

Design a publish-subscribe system that:
- Publishers send messages to topics
- Subscribers receive messages from subscribed topics
- One message can reach multiple subscribers
- Supports topic filtering and wildcards
- Thread-safe publish and subscribe operations

**Use Cases**:
- Event notifications
- Real-time updates
- Decoupled communication
- Observer pattern at scale
- News feeds

---

## Pub/Sub vs Message Broker

**Message Broker**: Point-to-point, message consumed by one consumer
**Pub/Sub**: Broadcast, message received by all subscribers

---

## Basic Pub/Sub Implementation

```python
import threading
import queue
import time
from typing import Dict, List, Callable, Set
from dataclasses import dataclass

@dataclass
class PubSubMessage:
    """Pub/Sub message"""
    topic: str
    payload: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class PubSub:
    """
    Basic publish-subscribe system
    """
    
    def __init__(self):
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.lock = threading.RLock()
    
    def subscribe(self, topic: str, callback: Callable[[PubSubMessage], None]) -> str:
        """
        Subscribe to topic
        
        Args:
            topic: Topic name
            callback: Function to call on message
        
        Returns:
            Subscription ID
        """
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = set()
            self.subscribers[topic].add(callback)
            return f"{topic}:{id(callback)}"
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from topic"""
        with self.lock:
            if topic in self.subscribers:
                self.subscribers[topic].discard(callback)
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
    
    def publish(self, topic: str, payload: Any):
        """
        Publish message to topic
        All subscribers receive the message
        """
        message = PubSubMessage(topic=topic, payload=payload)
        
        with self.lock:
            # Get subscribers for topic
            callbacks = self.subscribers.get(topic, set()).copy()
        
        # Notify all subscribers (outside lock to avoid deadlock)
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"Subscriber error: {e}")


# Usage
pubsub = PubSub()

# Subscribers
def subscriber1(message: PubSubMessage):
    print(f"Subscriber1 received: {message.payload}")

def subscriber2(message: PubSubMessage):
    print(f"Subscriber2 received: {message.payload}")

# Subscribe
pubsub.subscribe("news", subscriber1)
pubsub.subscribe("news", subscriber2)

# Publish (both subscribers receive)
pubsub.publish("news", "Breaking news!")
```

---

## Advanced: Pub/Sub with Wildcard Topics

```python
import re
import threading
from typing import Dict, List, Tuple

class WildcardPubSub:
    """
    Pub/Sub with wildcard topic matching
    Supports * (single level) and ** (multi-level)
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Tuple[Callable, str]]] = {}
        self.lock = threading.RLock()
    
    def _topic_to_pattern(self, topic: str) -> str:
        """Convert wildcard topic to regex pattern"""
        # Replace ** with .* and * with [^.]*
        pattern = topic.replace("**", "___MULTI___")
        pattern = pattern.replace("*", "[^.]*")
        pattern = pattern.replace("___MULTI___", ".*")
        return f"^{pattern}$"
    
    def subscribe(self, topic_pattern: str, callback: Callable):
        """
        Subscribe with wildcard support
        
        Examples:
            "news.*" matches "news.sports", "news.tech"
            "news.**" matches "news.sports.football", "news.tech.ai"
        """
        with self.lock:
            pattern = self._topic_to_pattern(topic_pattern)
            if pattern not in self.subscribers:
                self.subscribers[pattern] = []
            self.subscribers[pattern].append((callback, topic_pattern))
    
    def publish(self, topic: str, payload: Any):
        """Publish to topic, notify matching subscribers"""
        message = PubSubMessage(topic=topic, payload=payload)
        
        with self.lock:
            matching_subscribers = []
            for pattern, subscribers in self.subscribers.items():
                if re.match(pattern, topic):
                    matching_subscribers.extend(subscribers)
        
        # Notify matching subscribers
        for callback, _ in matching_subscribers:
            try:
                callback(message)
            except Exception as e:
                print(f"Subscriber error: {e}")


# Usage
pubsub = WildcardPubSub()

def sports_handler(message):
    print(f"Sports: {message.payload}")

def tech_handler(message):
    print(f"Tech: {message.payload}")

# Subscribe with wildcards
pubsub.subscribe("news.sports.*", sports_handler)
pubsub.subscribe("news.tech.*", tech_handler)

# Publish
pubsub.publish("news.sports.football", "Match result")
pubsub.publish("news.tech.ai", "AI breakthrough")
```

---

## Real-World: Event Bus

```python
import threading
from typing import Dict, List, Callable
from enum import Enum

class EventType(Enum):
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    ORDER_PLACED = "order.placed"
    PAYMENT_COMPLETED = "payment.completed"

class EventBus:
    """
    Event bus for application-wide events
    """
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.all_subscribers: List[Callable] = []  # Subscribe to all events
        self.lock = threading.RLock()
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to specific event type"""
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    def subscribe_all(self, handler: Callable):
        """Subscribe to all events"""
        with self.lock:
            self.all_subscribers.append(handler)
    
    def publish(self, event_type: EventType, event_data: dict):
        """Publish event"""
        event = {
            'type': event_type,
            'data': event_data,
            'timestamp': time.time()
        }
        
        with self.lock:
            # Get specific subscribers
            handlers = self.subscribers.get(event_type, []).copy()
            # Add all-event subscribers
            handlers.extend(self.all_subscribers)
        
        # Notify all handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error: {e}")


# Usage: E-commerce system
event_bus = EventBus()

# Specific handlers
def handle_user_created(event):
    print(f"New user: {event['data']}")

def handle_order_placed(event):
    print(f"Order placed: {event['data']}")

# Global handler (logging)
def log_all_events(event):
    print(f"LOG: {event['type']} - {event['data']}")

# Subscribe
event_bus.subscribe(EventType.USER_CREATED, handle_user_created)
event_bus.subscribe(EventType.ORDER_PLACED, handle_order_placed)
event_bus.subscribe_all(log_all_events)

# Publish events
event_bus.publish(EventType.USER_CREATED, {"user_id": "123", "name": "Alice"})
event_bus.publish(EventType.ORDER_PLACED, {"order_id": "456", "amount": 99.99})
```

---

## Key Takeaways

- Pub/Sub enables decoupled communication
- One message reaches all subscribers
- Wildcard support enables flexible subscriptions
- Event bus pattern for application-wide events
- Thread-safe operations support concurrent access

