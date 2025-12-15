+++
title = "Adapter Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Comprehensive guide to Adapter Pattern: Making incompatible interfaces work together, with Python implementations, object and class adapters, use cases, and best practices."
+++

---

## Introduction

The **Adapter Pattern** is a structural design pattern that allows objects with incompatible interfaces to collaborate. It acts as a bridge between two incompatible interfaces by wrapping an object to make it compatible with another interface.

### Intent

- Make incompatible interfaces work together
- Convert interface of a class into another interface clients expect
- Allow classes to work together that couldn't otherwise
- Wrap existing class with new interface

---

## Problem

Sometimes you need to use a class that has an interface different from what you expect. For example:
- Third-party libraries with different interfaces
- Legacy code that needs to work with new code
- APIs that don't match your requirements
- Services with incompatible interfaces

### Example Problem

```python
# Third-party payment service with different interface
class PayPalPayment:
    def make_payment(self, amount: float, currency: str):
        print(f"PayPal: Paying {amount} {currency}")

# Your application expects this interface
class PaymentProcessor:
    def process_payment(self, amount: float):
        pass

# Problem: Interfaces don't match!
paypal = PayPalPayment()
# paypal.process_payment(100)  # Doesn't exist!
```

---

## Solution

The Adapter Pattern solves this by creating an adapter class that:
1. Implements the target interface
2. Wraps the adaptee (incompatible class)
3. Translates calls from target interface to adaptee interface

---

## Structure

```
┌──────────────┐
│   Client     │
└──────┬───────┘
       │
       │ uses
       ▼
┌──────────────┐      ┌──────────────┐
│   Target     │      │   Adapter    │
│  (interface) │◄─────│              │
└──────────────┘      └──────┬───────┘
                             │
                             │ wraps
                             ▼
                      ┌──────────────┐
                      │   Adaptee    │
                      │ (incompatible)│
                      └──────────────┘
```

**Participants**:
- **Target**: Interface that client expects
- **Adapter**: Adapts adaptee to target interface
- **Adaptee**: Existing class with incompatible interface
- **Client**: Uses target interface

---

## Implementation

### Object Adapter (Composition)

```python
from abc import ABC, abstractmethod

# Target interface
class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

# Adaptee - incompatible interface
class PayPalPayment:
    def make_payment(self, amount: float, currency: str = "USD"):
        print(f"PayPal: Processing payment of {amount} {currency}")
        return True

# Adapter
class PayPalAdapter(PaymentProcessor):
    def __init__(self, paypal: PayPalPayment):
        self.paypal = paypal  # Composition
    
    def process_payment(self, amount: float) -> bool:
        # Adapt the interface
        return self.paypal.make_payment(amount, "USD")

# Usage
paypal = PayPalPayment()
adapter = PayPalAdapter(paypal)
adapter.process_payment(100.0)  # Works with expected interface!
```

### Class Adapter (Multiple Inheritance)

```python
from abc import ABC, abstractmethod

# Target interface
class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

# Adaptee
class PayPalPayment:
    def make_payment(self, amount: float, currency: str = "USD"):
        print(f"PayPal: Processing payment of {amount} {currency}")
        return True

# Class Adapter (multiple inheritance)
class PayPalAdapter(PaymentProcessor, PayPalPayment):
    def process_payment(self, amount: float) -> bool:
        # Adapt the interface
        return self.make_payment(amount, "USD")

# Usage
adapter = PayPalAdapter()
adapter.process_payment(100.0)
```

**Note**: Python supports multiple inheritance, but composition (object adapter) is generally preferred.

---

## Real-World Examples

### Example 1: Media Player Adapter

```python
from abc import ABC, abstractmethod

# Target interface
class MediaPlayer(ABC):
    @abstractmethod
    def play(self, audio_type: str, filename: str):
        pass

# Adaptee 1
class AdvancedMediaPlayer(ABC):
    @abstractmethod
    def play_vlc(self, filename: str):
        pass
    
    @abstractmethod
    def play_mp4(self, filename: str):
        pass

class VlcPlayer(AdvancedMediaPlayer):
    def play_vlc(self, filename: str):
        print(f"Playing VLC file: {filename}")
    
    def play_mp4(self, filename: str):
        pass

class Mp4Player(AdvancedMediaPlayer):
    def play_vlc(self, filename: str):
        pass
    
    def play_mp4(self, filename: str):
        print(f"Playing MP4 file: {filename}")

# Adapter
class MediaAdapter(MediaPlayer):
    def __init__(self, audio_type: str):
        if audio_type == "vlc":
            self.player = VlcPlayer()
        elif audio_type == "mp4":
            self.player = Mp4Player()
        else:
            raise ValueError(f"Unsupported audio type: {audio_type}")
    
    def play(self, audio_type: str, filename: str):
        if audio_type == "vlc":
            self.player.play_vlc(filename)
        elif audio_type == "mp4":
            self.player.play_mp4(filename)

# Client
class AudioPlayer(MediaPlayer):
    def play(self, audio_type: str, filename: str):
        if audio_type == "mp3":
            print(f"Playing MP3 file: {filename}")
        elif audio_type in ["vlc", "mp4"]:
            adapter = MediaAdapter(audio_type)
            adapter.play(audio_type, filename)
        else:
            print(f"Invalid media type: {audio_type}")

# Usage
player = AudioPlayer()
player.play("mp3", "song.mp3")      # Playing MP3 file: song.mp3
player.play("mp4", "video.mp4")     # Playing MP4 file: video.mp4
player.play("vlc", "movie.vlc")     # Playing VLC file: movie.vlc
```

### Example 2: Database Adapter

```python
from abc import ABC, abstractmethod

# Target interface
class Database(ABC):
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def execute_query(self, query: str):
        pass

# Adaptee - MongoDB (incompatible interface)
class MongoDB:
    def connect_to_mongodb(self):
        print("Connected to MongoDB")
    
    def find_documents(self, collection: str, filter: dict):
        print(f"Finding documents in {collection} with filter {filter}")

# Adaptee - MySQL (incompatible interface)
class MySQL:
    def connect_to_mysql(self):
        print("Connected to MySQL")
    
    def run_sql(self, sql: str):
        print(f"Executing SQL: {sql}")

# Adapters
class MongoDBAdapter(Database):
    def __init__(self, mongodb: MongoDB):
        self.mongodb = mongodb
    
    def connect(self):
        self.mongodb.connect_to_mongodb()
    
    def execute_query(self, query: str):
        # Convert SQL-like query to MongoDB query
        # Simplified conversion
        collection = query.split("FROM")[1].strip() if "FROM" in query else "default"
        self.mongodb.find_documents(collection, {})

class MySQLAdapter(Database):
    def __init__(self, mysql: MySQL):
        self.mysql = mysql
    
    def connect(self):
        self.mysql.connect_to_mysql()
    
    def execute_query(self, query: str):
        self.mysql.run_sql(query)

# Usage
mongodb = MongoDB()
mongo_adapter = MongoDBAdapter(mongodb)
mongo_adapter.connect()
mongo_adapter.execute_query("SELECT * FROM users")

mysql = MySQL()
mysql_adapter = MySQLAdapter(mysql)
mysql_adapter.connect()
mysql_adapter.execute_query("SELECT * FROM users")
```

### Example 3: Temperature Converter Adapter

```python
# Adaptee - Celsius temperature system
class CelsiusThermometer:
    def get_temperature_celsius(self) -> float:
        return 25.0  # 25°C

# Target interface - Fahrenheit expected
class FahrenheitThermometer(ABC):
    @abstractmethod
    def get_temperature_fahrenheit(self) -> float:
        pass

# Adapter
class TemperatureAdapter(FahrenheitThermometer):
    def __init__(self, celsius_thermometer: CelsiusThermometer):
        self.celsius_thermometer = celsius_thermometer
    
    def get_temperature_fahrenheit(self) -> float:
        celsius = self.celsius_thermometer.get_temperature_celsius()
        fahrenheit = (celsius * 9/5) + 32
        return fahrenheit

# Usage
celsius_therm = CelsiusThermometer()
adapter = TemperatureAdapter(celsius_therm)
print(f"Temperature: {adapter.get_temperature_fahrenheit()}°F")  # 77.0°F
```

---

## Use Cases

### When to Use Adapter Pattern

✅ **Third-Party Integration**: Integrating third-party libraries with different interfaces
✅ **Legacy Code**: Making legacy code work with new code
✅ **API Compatibility**: Adapting incompatible APIs
✅ **Interface Mismatch**: When interfaces don't match requirements
✅ **Reusability**: Reusing existing classes with incompatible interfaces
✅ **Multiple Formats**: Supporting multiple data formats

### When NOT to Use

❌ **Simple Interface Changes**: If you can modify the interface directly
❌ **Too Many Adapters**: If you need too many adapters, consider refactoring
❌ **Performance Critical**: Adds a layer of indirection
❌ **Can Modify Source**: If you can modify the incompatible class

---

## Pros and Cons

### Advantages

✅ **Reusability**: Reuse existing classes with incompatible interfaces
✅ **Separation of Concerns**: Separates interface conversion from business logic
✅ **Open/Closed Principle**: Can add new adapters without modifying existing code
✅ **Flexibility**: Can adapt multiple incompatible classes
✅ **Legacy Integration**: Easy to integrate legacy code

### Disadvantages

❌ **Complexity**: Adds additional classes and complexity
❌ **Performance**: Adds a layer of indirection
❌ **Overhead**: Can be overkill for simple interface mismatches
❌ **Maintenance**: More code to maintain

---

## Adapter vs Other Patterns

### Adapter vs Decorator

- **Adapter**: Changes interface to make incompatible classes work together
- **Decorator**: Adds behavior without changing interface

### Adapter vs Facade

- **Adapter**: Makes one interface compatible with another
- **Facade**: Provides simplified interface to complex subsystem

### Adapter vs Bridge

- **Adapter**: Makes incompatible interfaces work together (retrofit)
- **Bridge**: Separates abstraction from implementation (design-time)

---

## Best Practices

### 1. Use Composition Over Inheritance

```python
# Prefer: Object Adapter (Composition)
class Adapter(Target):
    def __init__(self, adaptee):
        self.adaptee = adaptee  # Composition

# Avoid: Class Adapter (Multiple Inheritance) unless necessary
class Adapter(Target, Adaptee):
    pass
```

### 2. Keep Adapters Simple

```python
class SimpleAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee
    
    def target_method(self):
        # Simple translation
        return self.adaptee.adaptee_method()
```

### 3. Use Type Hints

```python
from typing import Protocol

class Target(Protocol):
    def process(self, data: str) -> bool:
        ...

class Adapter:
    def __init__(self, adaptee: Adaptee):
        self.adaptee = adaptee
    
    def process(self, data: str) -> bool:
        return self.adaptee.adaptee_process(data)
```

### 4. Handle Errors Gracefully

```python
class Adapter:
    def __init__(self, adaptee):
        if adaptee is None:
            raise ValueError("Adaptee cannot be None")
        self.adaptee = adaptee
    
    def target_method(self):
        try:
            return self.adaptee.adaptee_method()
        except Exception as e:
            # Handle or translate errors
            raise AdapterError(f"Adaptation failed: {e}")
```

### 5. Document Adaptation Logic

```python
class Adapter:
    """
    Adapter for converting Adaptee interface to Target interface.
    
    Maps:
    - target_method() -> adaptee.adaptee_method()
    - target_param -> adaptee.adaptee_param
    """
    def __init__(self, adaptee):
        self.adaptee = adaptee
```

---

## Python-Specific Considerations

### 1. Using `Protocol` for Structural Typing

```python
from typing import Protocol

class PaymentProcessor(Protocol):
    def process_payment(self, amount: float) -> bool:
        ...

# Any class with process_payment method is compatible
class PayPalAdapter:
    def process_payment(self, amount: float) -> bool:
        return True
```

### 2. Duck Typing

Python's duck typing can sometimes eliminate the need for adapters:

```python
# If interfaces are similar enough, duck typing works
def process_payment(processor, amount: float):
    # Works with any object that has process_payment method
    return processor.process_payment(amount)
```

### 3. Using `__getattr__` for Dynamic Adaptation

```python
class DynamicAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee
    
    def __getattr__(self, name):
        # Dynamically forward calls to adaptee
        return getattr(self.adaptee, name)
```

### 4. Adapter with `functools`

```python
from functools import wraps

def adapt_method(adaptee_method):
    @wraps(adaptee_method)
    def wrapper(self, *args, **kwargs):
        # Adaptation logic
        return adaptee_method(*args, **kwargs)
    return wrapper
```

---

## Common Pitfalls

### 1. Over-Adapting

```python
# Bad: Too many adaptation layers
class Adapter1:
    pass

class Adapter2(Adapter1):
    pass

class Adapter3(Adapter2):
    pass

# Better: Direct adaptation
class SimpleAdapter:
    pass
```

### 2. Not Handling Edge Cases

```python
# Bad: No error handling
class Adapter:
    def target_method(self):
        return self.adaptee.adaptee_method()  # May fail

# Better: Handle errors
class Adapter:
    def target_method(self):
        try:
            return self.adaptee.adaptee_method()
        except Exception as e:
            # Handle appropriately
            pass
```

### 3. Breaking Encapsulation

```python
# Bad: Exposing adaptee directly
class Adapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee  # Public access

# Better: Keep adaptee private
class Adapter:
    def __init__(self, adaptee):
        self._adaptee = adaptee  # Protected
```

---

## Key Takeaways

- **Purpose**: Make incompatible interfaces work together
- **Use when**: Integrating incompatible classes or third-party libraries
- **Object vs Class**: Prefer object adapter (composition) over class adapter
- **Benefits**: Reusability, separation of concerns, legacy integration
- **Trade-off**: Adds complexity and indirection
- **Python**: Leverage duck typing and protocols when possible
- **Best practice**: Keep adapters simple and focused

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Adapter Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/adapter)

