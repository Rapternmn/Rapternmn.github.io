+++
title = "Observer Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Comprehensive guide to Observer Pattern: Notifying multiple objects about state changes, with Python implementations, push vs pull models, use cases, and best practices."
+++

---

## Introduction

The **Observer Pattern** is a behavioral design pattern that defines a one-to-many dependency between objects. When one object changes state, all its dependents are notified and updated automatically.

### Intent

- Define one-to-many dependency between objects
- Notify multiple objects about state changes
- Decouple subject from observers
- Support broadcast communication

---

## Problem

Sometimes you need to notify multiple objects about changes:
- Event-driven systems
- Model-View architectures
- Real-time updates
- Loose coupling between components

### Example Problem

```python
class Stock:
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price
        # Problem: How to notify all interested parties?
    
    def set_price(self, price: float):
        self.price = price
        # Need to notify: investors, traders, displays, etc.
        # But Stock shouldn't know about all of them!
```

---

## Solution

The Observer Pattern solves this by:
1. Creating a subject (observable) that maintains a list of observers
2. Observers register/unregister with the subject
3. Subject notifies all observers when state changes
4. Observers update themselves based on notification

---

## Structure

```
┌──────────────┐
│   Subject    │
│ (Observable) │
├──────────────┤
│ + attach()   │
│ + detach()   │
│ + notify()   │
└──────┬───────┘
       │
       │ notifies
       ▼
┌──────────────┐
│   Observer   │
│  (interface) │
├──────────────┤
│ + update()   │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ Concrete     │
│ Observer     │
└──────────────┘
```

**Participants**:
- **Subject**: Maintains observers and notifies them
- **Observer**: Interface for objects that should be notified
- **ConcreteSubject**: Subject with state that triggers notifications
- **ConcreteObserver**: Observer that updates based on notification

---

## Implementation

### Basic Observer Pattern

```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Subject(ABC):
    def __init__(self):
        self._observers = []
    
    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class Stock(Subject):
    def __init__(self, symbol: str, price: float):
        super().__init__()
        self._symbol = symbol
        self._price = price
    
    @property
    def symbol(self):
        return self._symbol
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value: float):
        if self._price != value:
            self._price = value
            self.notify()  # Notify all observers

class Investor(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, subject: Stock):
        print(f"{self.name} notified: {subject.symbol} is now ${subject.price}")

class Trader(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, subject: Stock):
        if subject.price > 100:
            print(f"{self.name}: {subject.symbol} price is high! Consider selling.")
        elif subject.price < 50:
            print(f"{self.name}: {subject.symbol} price is low! Consider buying.")

# Usage
stock = Stock("AAPL", 150.0)

investor1 = Investor("Alice")
investor2 = Investor("Bob")
trader = Trader("Charlie")

stock.attach(investor1)
stock.attach(investor2)
stock.attach(trader)

stock.price = 160.0  # All observers notified
stock.price = 45.0   # All observers notified again
```

### Push Model (Subject sends data)

```python
class Observer(ABC):
    @abstractmethod
    def update(self, data: dict):
        pass

class Stock(Subject):
    def __init__(self, symbol: str, price: float):
        super().__init__()
        self._symbol = symbol
        self._price = price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value: float):
        if self._price != value:
            old_price = self._price
            self._price = value
            # Push data to observers
            self.notify({
                'symbol': self._symbol,
                'old_price': old_price,
                'new_price': value,
                'change': value - old_price
            })
    
    def notify(self, data: dict):
        for observer in self._observers:
            observer.update(data)

class Display(Observer):
    def update(self, data: dict):
        change = data['change']
        direction = "↑" if change > 0 else "↓"
        print(f"{data['symbol']}: ${data['old_price']} {direction} ${data['new_price']} ({change:+.2f})")

# Usage
stock = Stock("AAPL", 150.0)
display = Display()
stock.attach(display)
stock.price = 160.0  # AAPL: $150.0 ↑ $160.0 (+10.00)
```

### Pull Model (Observer requests data)

```python
class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Stock(Subject):
    def __init__(self, symbol: str, price: float, volume: int):
        super().__init__()
        self._symbol = symbol
        self._price = price
        self._volume = volume
    
    @property
    def symbol(self):
        return self._symbol
    
    @property
    def price(self):
        return self._price
    
    @property
    def volume(self):
        return self._volume
    
    @price.setter
    def price(self, value: float):
        if self._price != value:
            self._price = value
            self.notify()  # Observers pull data they need

class Analytics(Observer):
    def update(self, subject: Stock):
        # Pull only the data we need
        price = subject.price
        volume = subject.volume
        market_cap = price * volume
        print(f"Analytics: {subject.symbol} market cap = ${market_cap:,.2f}")
```

---

## Real-World Examples

### Example 1: Event System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class EventObserver(ABC):
    @abstractmethod
    def handle_event(self, event_type: str, data: Dict[str, Any]):
        pass

class EventSubject:
    def __init__(self):
        self._observers = {}
    
    def subscribe(self, event_type: str, observer: EventObserver):
        if event_type not in self._observers:
            self._observers[event_type] = []
        if observer not in self._observers[event_type]:
            self._observers[event_type].append(observer)
    
    def unsubscribe(self, event_type: str, observer: EventObserver):
        if event_type in self._observers:
            self._observers[event_type].remove(observer)
    
    def emit(self, event_type: str, data: Dict[str, Any]):
        if event_type in self._observers:
            for observer in self._observers[event_type]:
                observer.handle_event(event_type, data)

class UserService(EventSubject):
    def create_user(self, username: str, email: str):
        user = {"username": username, "email": email}
        # Emit event
        self.emit("user_created", user)
        return user
    
    def delete_user(self, user_id: int):
        self.emit("user_deleted", {"user_id": user_id})

class EmailService(EventObserver):
    def handle_event(self, event_type: str, data: Dict[str, Any]):
        if event_type == "user_created":
            print(f"Sending welcome email to {data['email']}")
        elif event_type == "user_deleted":
            print(f"Sending goodbye email for user {data['user_id']}")

class LoggingService(EventObserver):
    def handle_event(self, event_type: str, data: Dict[str, Any]):
        print(f"[LOG] Event: {event_type}, Data: {data}")

# Usage
user_service = UserService()
email_service = EmailService()
logging_service = LoggingService()

user_service.subscribe("user_created", email_service)
user_service.subscribe("user_created", logging_service)
user_service.subscribe("user_deleted", email_service)

user_service.create_user("john", "john@example.com")
```

### Example 2: Weather Station

```python
from abc import ABC, abstractmethod

class WeatherData(Subject):
    def __init__(self):
        super().__init__()
        self._temperature = 0.0
        self._humidity = 0.0
        self._pressure = 0.0
    
    def set_measurements(self, temperature: float, humidity: float, pressure: float):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.notify()
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def humidity(self):
        return self._humidity
    
    @property
    def pressure(self):
        return self._pressure

class Display(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, subject: WeatherData):
        print(f"{self.name}: Temp={subject.temperature}°F, "
              f"Humidity={subject.humidity}%, Pressure={subject.pressure}")

class StatisticsDisplay(Observer):
    def __init__(self):
        self.temps = []
        self.humidities = []
    
    def update(self, subject: WeatherData):
        self.temps.append(subject.temperature)
        self.humidities.append(subject.humidity)
        
        avg_temp = sum(self.temps) / len(self.temps)
        avg_humidity = sum(self.humidities) / len(self.humidities)
        
        print(f"Statistics: Avg Temp={avg_temp:.1f}°F, Avg Humidity={avg_humidity:.1f}%")

class ForecastDisplay(Observer):
    def update(self, subject: WeatherData):
        if subject.pressure > 30:
            forecast = "Improving weather"
        elif subject.pressure < 29:
            forecast = "Watch for storms"
        else:
            forecast = "More of the same"
        print(f"Forecast: {forecast}")

# Usage
weather_data = WeatherData()

current_display = Display("Current Conditions")
stats_display = StatisticsDisplay()
forecast_display = ForecastDisplay()

weather_data.attach(current_display)
weather_data.attach(stats_display)
weather_data.attach(forecast_display)

weather_data.set_measurements(80, 65, 30.4)
weather_data.set_measurements(82, 70, 29.2)
```

### Example 3: Model-View (MVC)

```python
from abc import ABC, abstractmethod

class Model(Subject):
    def __init__(self):
        super().__init__()
        self._data = {}
    
    def set_data(self, key: str, value):
        self._data[key] = value
        self.notify()
    
    def get_data(self, key: str):
        return self._data.get(key)

class View(Observer):
    def __init__(self, name: str):
        self.name = name
    
    def update(self, subject: Model):
        print(f"{self.name} updated with data: {subject._data}")

class Controller:
    def __init__(self, model: Model):
        self.model = model
    
    def update_model(self, key: str, value):
        self.model.set_data(key, value)

# Usage
model = Model()
view1 = View("View 1")
view2 = View("View 2")

model.attach(view1)
model.attach(view2)

controller = Controller(model)
controller.update_model("name", "John")
controller.update_model("age", 30)
```

---

## Use Cases

### When to Use Observer Pattern

✅ **Event-Driven Systems**: When you need event-driven architecture
✅ **Model-View**: When implementing MVC/MVP patterns
✅ **Real-Time Updates**: When multiple objects need real-time updates
✅ **Loose Coupling**: When you want to decouple subject from observers
✅ **Broadcast Communication**: When one object needs to notify many
✅ **State Changes**: When state changes need to trigger updates

### When NOT to Use

❌ **Tight Coupling**: When observers need to know too much about subject
❌ **Performance Critical**: When notifications are too frequent
❌ **Simple Updates**: Overkill for simple one-to-one updates
❌ **Circular Dependencies**: Can create circular dependencies

---

## Pros and Cons

### Advantages

✅ **Loose Coupling**: Subject and observers are loosely coupled
✅ **Dynamic Relationships**: Can add/remove observers at runtime
✅ **Broadcast**: One subject can notify many observers
✅ **Open/Closed**: Easy to add new observers without modifying subject
✅ **Separation of Concerns**: Separates subject from observers

### Disadvantages

❌ **Unexpected Updates**: Observers may receive unexpected updates
❌ **Performance**: Many observers can impact performance
❌ **Debugging**: Can be hard to debug notification chains
❌ **Memory Leaks**: Forgetting to detach can cause memory leaks

---

## Observer vs Other Patterns

### Observer vs Mediator

- **Observer**: One-to-many communication
- **Mediator**: Many-to-many communication through mediator

### Observer vs Publish-Subscribe

- **Observer**: Direct coupling between subject and observers
- **Publish-Subscribe**: Decoupled through message broker

### Observer vs Chain of Responsibility

- **Observer**: All observers notified
- **Chain**: Only one handler processes request

---

## Best Practices

### 1. Use Weak References (Avoid Memory Leaks)

```python
import weakref

class Subject:
    def __init__(self):
        self._observers = weakref.WeakSet()  # Weak references
    
    def attach(self, observer):
        self._observers.add(observer)
```

### 2. Specify Update Interface

```python
class Observer(ABC):
    @abstractmethod
    def update(self, subject, **kwargs):
        pass
```

### 3. Handle Observer Errors

```python
def notify(self):
    for observer in self._observers:
        try:
            observer.update(self)
        except Exception as e:
            # Log error, don't break other observers
            print(f"Observer error: {e}")
```

### 4. Use Event Types

```python
def notify(self, event_type: str, data: dict):
    for observer in self._observers:
        observer.update(event_type, data)
```

### 5. Consider Async Notifications

```python
import asyncio

async def notify_async(self):
    tasks = [observer.update_async(self) for observer in self._observers]
    await asyncio.gather(*tasks)
```

---

## Python-Specific Considerations

### 1. Using `collections.abc`

```python
from collections.abc import Set

class ObserverSet(Set):
    def __init__(self):
        self._observers = set()
    
    def __contains__(self, observer):
        return observer in self._observers
    
    def __iter__(self):
        return iter(self._observers)
    
    def __len__(self):
        return len(self._observers)
```

### 2. Using `weakref`

```python
import weakref

class Subject:
    def __init__(self):
        self._observers = weakref.WeakSet()
```

### 3. Using `typing.Protocol`

```python
from typing import Protocol

class Observer(Protocol):
    def update(self, subject) -> None:
        ...
```

### 4. Event-Driven with `asyncio`

```python
import asyncio

class AsyncSubject:
    def __init__(self):
        self._observers = []
    
    async def notify(self):
        tasks = [obs.update_async(self) for obs in self._observers]
        await asyncio.gather(*tasks)
```

---

## Common Pitfalls

### 1. Memory Leaks

```python
# Bad: Strong references
class Subject:
    def __init__(self):
        self._observers = []  # Strong reference

# Good: Weak references
import weakref
class Subject:
    def __init__(self):
        self._observers = weakref.WeakSet()  # Weak reference
```

### 2. Not Handling Errors

```python
# Bad: One error breaks all
def notify(self):
    for observer in self._observers:
        observer.update(self)  # May raise exception

# Good: Handle errors
def notify(self):
    for observer in self._observers:
        try:
            observer.update(self)
        except Exception as e:
            logger.error(f"Observer error: {e}")
```

### 3. Circular Dependencies

```python
# Bad: Circular dependency
subject.attach(observer)
observer.set_subject(subject)  # Can create cycles

# Good: One-way dependency
subject.attach(observer)  # Observer doesn't reference subject
```

---

## Key Takeaways

- **Purpose**: Notify multiple objects about state changes
- **Use when**: Event-driven systems, MVC, real-time updates
- **Models**: Push (subject sends data) or Pull (observer requests data)
- **Benefits**: Loose coupling, dynamic relationships, broadcast
- **Trade-off**: Can cause memory leaks, performance issues
- **Python**: Use weakref, asyncio for async, Protocol for typing
- **Best practice**: Use weak references, handle errors, specify interface

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python weakref](https://docs.python.org/3/library/weakref.html)
- [Observer Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/observer)

