+++
title = "Singleton Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Comprehensive guide to Singleton Pattern: Ensuring only one instance of a class exists, with Python implementations, thread-safety considerations, use cases, and best practices."
+++

---

## Introduction

The **Singleton Pattern** is a creational design pattern that ensures a class has only one instance and provides a global point of access to that instance. It's one of the most commonly used design patterns, though it's also one of the most controversial.

### Intent

- Ensure only one instance of a class exists
- Provide global access to that instance
- Control access to shared resources

---

## Problem

Sometimes you need exactly one instance of a class in your application. For example:
- Database connection pool
- Logger instance
- Configuration manager
- Cache manager
- Thread pool

Creating multiple instances could:
- Waste resources
- Cause inconsistent behavior
- Create conflicts
- Lead to unexpected side effects

### Example Problem

```python
class DatabaseConnection:
    def __init__(self):
        print("Creating new database connection...")
        # Expensive connection setup
        self.connection = "Database connection established"
    
    def query(self, sql: str):
        return f"Executing: {sql}"

# Problem: Multiple instances created
db1 = DatabaseConnection()  # Creates connection
db2 = DatabaseConnection()  # Creates another connection (wasteful!)
db3 = DatabaseConnection()  # Creates yet another connection
```

---

## Solution

The Singleton pattern solves this by:
1. Making the constructor private (or protected)
2. Creating a static method that returns the same instance
3. Storing the instance in a class variable

---

## Structure

```
┌─────────────────────────┐
│      Singleton          │
├─────────────────────────┤
│ - instance: Singleton    │
├─────────────────────────┤
│ + get_instance()        │
│ - __init__()            │
└─────────────────────────┘
```

**Participants**:
- **Singleton**: Class that ensures only one instance exists

---

## Implementation Approaches

### 1. Basic Singleton (Non-Thread-Safe)

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # This will be called multiple times!
        # Be careful with initialization logic
        pass

# Usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
```

**Issues**:
- Not thread-safe
- `__init__` called multiple times
- Can be bypassed by direct instantiation

### 2. Singleton with `__new__` and Initialization Control

```python
class Singleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Singleton._initialized:
            # Initialization code here
            self.value = 0
            Singleton._initialized = True
    
    def increment(self):
        self.value += 1
        return self.value

# Usage
s1 = Singleton()
s2 = Singleton()
s1.increment()
print(s2.value)  # 1 - same instance
```

### 3. Singleton with Decorator

```python
def singleton(cls):
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Creating database connection...")
        self.connection = "Connected"
    
    def query(self, sql: str):
        return f"Query: {sql}"

# Usage
db1 = DatabaseConnection()  # Prints: Creating database connection...
db2 = DatabaseConnection()   # No print - same instance
print(db1 is db2)  # True
```

### 4. Singleton with Metaclass (Pythonic)

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        print("Creating database connection...")
        self.connection = "Connected"
    
    def query(self, sql: str):
        return f"Query: {sql}"

# Usage
db1 = DatabaseConnection()  # Prints: Creating database connection...
db2 = DatabaseConnection()  # No print - same instance
print(db1 is db2)  # True
```

**Benefits of Metaclass Approach**:
- Clean and Pythonic
- Works with inheritance
- Initialization happens only once
- Easy to understand

### 5. Thread-Safe Singleton

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialization happens only once
        if not hasattr(self, '_initialized'):
            self.value = 0
            self._initialized = True

# Usage
def create_singleton():
    return ThreadSafeSingleton()

# Thread-safe test
s1 = create_singleton()
s2 = create_singleton()
print(s1 is s2)  # True
```

### 6. Singleton with Module (Pythonic Alternative)

In Python, modules are singletons by nature:

```python
# config.py (singleton module)
class Config:
    def __init__(self):
        self.database_url = "localhost:5432"
        self.api_key = "secret_key"

# Create instance at module level
config = Config()

# Usage in other files
from config import config

# config is a singleton - imported once
print(config.database_url)
```

**This is often the most Pythonic approach!**

---

## Real-World Examples

### Example 1: Logger Singleton

```python
import logging
from datetime import datetime

class Logger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Logger._initialized:
            self.logs = []
            Logger._initialized = True
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def get_logs(self):
        return self.logs

# Usage
logger1 = Logger()
logger2 = Logger()

logger1.log("Application started")
logger2.log("User logged in")

print(logger1 is logger2)  # True
print(len(logger1.get_logs()))  # 2 - both logs in same instance
```

### Example 2: Configuration Manager

```python
class ConfigManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.settings = {}
            self._load_config()
            self._initialized = True
    
    def _load_config(self):
        # Load from file or environment
        self.settings = {
            'database_url': 'localhost:5432',
            'api_key': 'secret',
            'debug': True
        }
    
    def get(self, key: str, default=None):
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        self.settings[key] = value

# Usage
config1 = ConfigManager()
config2 = ConfigManager()

print(config1.get('database_url'))  # localhost:5432
config2.set('database_url', 'new_host:5432')
print(config1.get('database_url'))  # new_host:5432 - same instance
```

### Example 3: Database Connection Pool

```python
class DatabasePool:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.connections = []
            self.max_connections = 10
            self._initialized = True
    
    def get_connection(self):
        if len(self.connections) < self.max_connections:
            conn = f"Connection-{len(self.connections) + 1}"
            self.connections.append(conn)
            return conn
        raise Exception("Connection pool exhausted")
    
    def release_connection(self, conn):
        if conn in self.connections:
            self.connections.remove(conn)

# Usage
pool1 = DatabasePool()
pool2 = DatabasePool()

conn1 = pool1.get_connection()
conn2 = pool2.get_connection()

print(pool1 is pool2)  # True
print(len(pool1.connections))  # 2 - shared pool
```

---

## Use Cases

### When to Use Singleton

✅ **Resource Management**: Database connections, file handles, network connections
✅ **Configuration**: Application settings, environment variables
✅ **Logging**: Centralized logging system
✅ **Caching**: Global cache manager
✅ **Thread Pools**: Shared thread pool
✅ **Hardware Interface**: Printer spooler, graphics card access
✅ **State Management**: Global application state

### When NOT to Use Singleton

❌ **When you need multiple instances**: Don't force singleton if multiple instances make sense
❌ **For testing**: Makes unit testing harder (hard to mock)
❌ **When state needs isolation**: If instances need separate state
❌ **For simple objects**: Overkill for simple data containers
❌ **When dependency injection is better**: Consider dependency injection instead

---

## Pros and Cons

### Advantages

✅ **Controlled Access**: Single point of access to instance
✅ **Resource Efficiency**: Prevents unnecessary object creation
✅ **Global State**: Provides global state management
✅ **Lazy Initialization**: Can delay instance creation
✅ **Memory Savings**: Only one instance in memory

### Disadvantages

❌ **Testing Difficulties**: Hard to test (can't easily mock)
❌ **Hidden Dependencies**: Global state can hide dependencies
❌ **Thread Safety**: Requires careful implementation for multi-threading
❌ **Violates SRP**: Can accumulate too many responsibilities
❌ **Global State Issues**: Can lead to hidden coupling
❌ **Inheritance Problems**: Difficult to subclass
❌ **Concurrency Issues**: Can cause problems in multi-threaded environments

---

## Thread Safety Considerations

### The Problem

```python
class UnsafeSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:  # Race condition here!
            cls._instance = super().__new__(cls)
        return cls._instance
```

In multi-threaded environments, multiple threads might check `_instance is None` simultaneously, creating multiple instances.

### Solutions

1. **Locking**: Use locks to synchronize access
2. **Double-Check Locking**: Check twice with lock
3. **Module-Level Singleton**: Use Python modules (thread-safe by default)
4. **Eager Initialization**: Create instance at import time

---

## Best Practices

### 1. Use Module-Level Singletons When Possible

```python
# config.py
class Config:
    pass

config = Config()  # Module-level singleton
```

### 2. Make Thread-Safe When Needed

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

### 3. Consider Dependency Injection

**Dependency Injection (DI)** is a design pattern where objects receive their dependencies from external sources rather than creating them internally. It's often a better alternative to Singleton because it:

- Makes code more testable (easy to inject mocks)
- Reduces coupling between classes
- Improves flexibility and maintainability
- Makes dependencies explicit

#### Singleton Approach (Tight Coupling)

```python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

class UserService:
    def __init__(self):
        # Tight coupling - directly depends on singleton
        self.db = DatabaseConnection()
    
    def get_user(self, user_id):
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

# Problem: Hard to test - can't easily mock DatabaseConnection
service = UserService()
```

#### Dependency Injection Approach (Loose Coupling)

```python
class DatabaseConnection:
    def query(self, sql: str):
        return f"Result: {sql}"

class UserService:
    def __init__(self, db_connection):
        # Dependency injected - loose coupling
        self.db = db_connection
    
    def get_user(self, user_id):
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

# Usage - inject the dependency
db = DatabaseConnection()
service = UserService(db)  # Dependency injected

# Testing - easy to inject mock
class MockDatabase:
    def query(self, sql: str):
        return "Mock result"

mock_db = MockDatabase()
test_service = UserService(mock_db)  # Easy to test!
```

#### Dependency Injection Container Example

```python
class Container:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services = {}
    
    def register(self, name, service):
        self._services[name] = service
    
    def get(self, name):
        return self._services.get(name)

# Setup
container = Container()
container.register('database', DatabaseConnection())
container.register('logger', Logger())

# Usage
class OrderService:
    def __init__(self, container):
        self.db = container.get('database')
        self.logger = container.get('logger')
    
    def create_order(self, order_data):
        self.logger.log("Creating order...")
        return self.db.query(f"INSERT INTO orders VALUES {order_data}")

order_service = OrderService(container)
```

**Benefits of Dependency Injection over Singleton**:
- ✅ Easier testing (can inject mocks)
- ✅ More flexible (can swap implementations)
- ✅ Explicit dependencies (clear what a class needs)
- ✅ Better for multi-threading (each thread can have its own instance)
- ✅ Follows Dependency Inversion Principle (SOLID)

### 4. Document Singleton Behavior

```python
class Singleton:
    """
    Singleton class - only one instance exists.
    
    Usage:
        instance1 = Singleton()
        instance2 = Singleton()
        assert instance1 is instance2  # True
    """
    _instance = None
    # ... implementation
```

### 5. Avoid Global Mutable State

```python
# Bad: Global mutable state
class Config:
    _instance = None
    def __new__(cls):
        # ... singleton logic
        return cls._instance
    
    def __init__(self):
        self.settings = {}  # Mutable - can cause issues

# Better: Immutable or controlled access
class Config:
    _instance = None
    _settings = {}  # Class variable
    
    def __new__(cls):
        # ... singleton logic
        return cls._instance
    
    def get_setting(self, key):
        return self._settings.get(key)
```

---

## Common Pitfalls

### 1. Multiple Instances via Inheritance

```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

class Child(Singleton):
    pass

# Problem: Child has its own instance!
s1 = Singleton()
c1 = Child()
print(s1 is c1)  # False - different instances!
```

**Solution**: Use metaclass approach for inheritance support.

### 2. Serialization Issues

```python
import pickle

class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s1.value = 42

# Serialize and deserialize
data = pickle.dumps(s1)
s2 = pickle.loads(data)

print(s1 is s2)  # False - new instance created!
```

### 3. Subclassing Problems

Subclassing singletons can be tricky. Consider composition instead of inheritance.

---

## Alternatives to Singleton

### 1. Dependency Injection

```python
class DatabaseService:
    def __init__(self, connection):
        self.connection = connection

# Inject instead of singleton
service = DatabaseService(connection)
```

### 2. Module-Level Variables

```python
# config.py
database_url = "localhost:5432"
api_key = "secret"
```

### 3. Factory Pattern

```python
class ConnectionFactory:
    _connection = None
    
    @classmethod
    def get_connection(cls):
        if cls._connection is None:
            cls._connection = create_connection()
        return cls._connection
```

### 4. Monostate Pattern

```python
class Monostate:
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state
```

---

## Key Takeaways

- **Purpose**: Ensure only one instance exists
- **Use when**: You need exactly one instance (resources, config, logging)
- **Avoid when**: Multiple instances make sense, or for simple objects
- **Thread-safety**: Important in multi-threaded environments
- **Pythonic approach**: Consider module-level variables
- **Testing**: Can make testing harder - consider alternatives
- **Best practice**: Use sparingly and document well
- **Alternatives**: Dependency injection, factory pattern, module variables

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python's `__new__` method](https://docs.python.org/3/reference/datamodel.html#object.__new__)
- [Python Metaclasses](https://docs.python.org/3/reference/datamodel.html#metaclasses)

