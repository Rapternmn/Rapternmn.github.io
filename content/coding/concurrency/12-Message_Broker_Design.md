+++
title = "Message Broker Design"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 12
description = "Message Broker implementation: Topic-based messaging, producer-consumer pattern, message routing, and thread-safe message delivery."
+++

# 📨 Message Broker Design

## Problem Statement

Design a message broker that:
- Routes messages from producers to consumers
- Supports multiple topics/channels
- Ensures message delivery
- Handles multiple producers and consumers
- Thread-safe message operations

**Use Cases**:
- Microservices communication
- Event-driven architecture
- Task distribution
- Log aggregation
- Notification systems

---

## Requirements

1. **Topics/Channels**: Support multiple message topics
2. **Producer/Consumer**: Multiple producers and consumers per topic
3. **Message Delivery**: Reliable message delivery
4. **Thread Safety**: Safe concurrent operations
5. **Message Ordering**: Optional ordering guarantees

---

## Basic Message Broker

```python
import threading
import queue
import time
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"

@dataclass
class Message:
    """Message structure"""
    topic: str
    payload: Any
    message_type: MessageType = MessageType.TEXT
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBroker:
    """
    Basic message broker with topic-based routing
    """
    
    def __init__(self, queue_size: int = 100):
        """
        Args:
            queue_size: Maximum messages per topic queue
        """
        self.queues: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.RLock()
        self.running = True
        
        # Start dispatcher thread
        self.dispatcher = threading.Thread(target=self._dispatch_messages, daemon=True)
        self.dispatcher.start()
    
    def create_topic(self, topic: str):
        """Create a new topic"""
        with self.lock:
            if topic not in self.queues:
                self.queues[topic] = queue.Queue(maxsize=100)
                self.subscribers[topic] = []
    
    def publish(self, topic: str, message: Any, message_type: MessageType = MessageType.TEXT):
        """
        Publish message to topic
        
        Args:
            topic: Topic name
            message: Message payload
            message_type: Type of message
        """
        with self.lock:
            if topic not in self.queues:
                self.create_topic(topic)
            
            msg = Message(topic=topic, payload=message, message_type=message_type)
            try:
                self.queues[topic].put_nowait(msg)
            except queue.Full:
                print(f"Topic {topic} queue full, message dropped")
    
    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """
        Subscribe to topic with callback
        
        Args:
            topic: Topic name
            callback: Function to call when message arrives
        """
        with self.lock:
            if topic not in self.subscribers:
                self.create_topic(topic)
            self.subscribers[topic].append(callback)
    
    def _dispatch_messages(self):
        """Dispatcher thread that routes messages to subscribers"""
        while self.running:
            for topic, topic_queue in list(self.queues.items()):
                try:
                    message = topic_queue.get_nowait()
                    
                    # Deliver to all subscribers
                    with self.lock:
                        callbacks = self.subscribers.get(topic, [])
                    
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"Callback error for topic {topic}: {e}")
                    
                    topic_queue.task_done()
                except queue.Empty:
                    continue
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting
    
    def shutdown(self):
        """Shutdown broker"""
        self.running = False
        self.dispatcher.join()


# Usage
broker = MessageBroker()

# Subscriber
def handle_user_events(message: Message):
    print(f"User event: {message.payload}")

def handle_order_events(message: Message):
    print(f"Order event: {message.payload}")

# Subscribe
broker.subscribe("user.events", handle_user_events)
broker.subscribe("order.events", handle_order_events)

# Publish
broker.publish("user.events", "User logged in")
broker.publish("order.events", "Order created")
broker.publish("user.events", "User logged out")

time.sleep(0.1)
broker.shutdown()
```

---

## Advanced: Message Broker with Persistence

```python
import threading
import queue
import json
from typing import Dict, List, Optional

class PersistentMessageBroker:
    """
    Message broker with message persistence
    """
    
    def __init__(self, queue_size: int = 100):
        self.queues: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: Dict[str, List[Message]] = {}  # Topic -> messages
        self.lock = threading.RLock()
        self.running = True
        self.max_history = 1000  # Keep last 1000 messages per topic
        
        self.dispatcher = threading.Thread(target=self._dispatch_messages, daemon=True)
        self.dispatcher.start()
    
    def create_topic(self, topic: str):
        """Create topic with history"""
        with self.lock:
            if topic not in self.queues:
                self.queues[topic] = queue.Queue(maxsize=100)
                self.subscribers[topic] = []
                self.message_history[topic] = []
    
    def publish(self, topic: str, message: Any):
        """Publish and persist message"""
        with self.lock:
            if topic not in self.queues:
                self.create_topic(topic)
            
            msg = Message(topic=topic, payload=message)
            
            # Persist message
            history = self.message_history[topic]
            history.append(msg)
            if len(history) > self.max_history:
                history.pop(0)  # Remove oldest
            
            # Add to queue
            try:
                self.queues[topic].put_nowait(msg)
            except queue.Full:
                print(f"Topic {topic} queue full")
    
    def get_message_history(self, topic: str, limit: int = 10) -> List[Message]:
        """Get recent message history for topic"""
        with self.lock:
            history = self.message_history.get(topic, [])
            return history[-limit:]
    
    def _dispatch_messages(self):
        """Dispatch messages to subscribers"""
        while self.running:
            for topic, topic_queue in list(self.queues.items()):
                try:
                    message = topic_queue.get_nowait()
                    
                    with self.lock:
                        callbacks = self.subscribers.get(topic, [])
                    
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"Callback error: {e}")
                    
                    topic_queue.task_done()
                except queue.Empty:
                    continue
            
            time.sleep(0.01)
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        with self.lock:
            if topic not in self.subscribers:
                self.create_topic(topic)
            self.subscribers[topic].append(callback)
    
    def shutdown(self):
        """Shutdown broker"""
        self.running = False
        self.dispatcher.join()
```

---

## Real-World: Event-Driven Message Broker

```python
import threading
from typing import Dict, Set

class EventDrivenBroker:
    """
    Message broker for event-driven architecture
    """
    
    def __init__(self):
        self.topics: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.lock = threading.RLock()
        self.running = True
        
        self.dispatcher = threading.Thread(target=self._dispatch, daemon=True)
        self.dispatcher.start()
    
    def publish_event(self, event_type: str, event_data: dict):
        """Publish event to topic"""
        message = {
            'type': event_type,
            'data': event_data,
            'timestamp': time.time()
        }
        
        with self.lock:
            if event_type not in self.topics:
                self.topics[event_type] = queue.Queue()
                self.subscribers[event_type] = set()
            
            self.topics[event_type].put(message)
    
    def subscribe_event(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        with self.lock:
            if event_type not in self.subscribers:
                self.topics[event_type] = queue.Queue()
                self.subscribers[event_type] = set()
            self.subscribers[event_type].add(handler)
    
    def _dispatch(self):
        """Dispatch events to handlers"""
        while self.running:
            for event_type, topic_queue in list(self.topics.items()):
                try:
                    event = topic_queue.get_nowait()
                    
                    with self.lock:
                        handlers = self.subscribers.get(event_type, set())
                    
                    for handler in handlers:
                        try:
                            handler(event)
                        except Exception as e:
                            print(f"Handler error: {e}")
                    
                    topic_queue.task_done()
                except queue.Empty:
                    continue
            
            time.sleep(0.01)
    
    def shutdown(self):
        self.running = False


# Usage: E-commerce event system
broker = EventDrivenBroker()

# Event handlers
def handle_order_created(event):
    print(f"Order created: {event['data']}")

def handle_payment_processed(event):
    print(f"Payment processed: {event['data']}")

# Subscribe
broker.subscribe_event("order.created", handle_order_created)
broker.subscribe_event("payment.processed", handle_payment_processed)

# Publish events
broker.publish_event("order.created", {"order_id": "123", "amount": 99.99})
broker.publish_event("payment.processed", {"order_id": "123", "status": "success"})
```

---

## Key Takeaways

- Message brokers route messages from producers to consumers
- Topic-based routing enables organized messaging
- Thread-safe operations support concurrent access
- Useful for event-driven and microservices architectures
- Can be extended with persistence and ordering guarantees

