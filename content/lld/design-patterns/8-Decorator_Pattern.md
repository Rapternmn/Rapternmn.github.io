+++
title = "Decorator Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Comprehensive guide to Decorator Pattern: Adding behavior to objects dynamically at runtime, with Python implementations using decorators, composition, use cases, and best practices."
+++

---

## Introduction

The **Decorator Pattern** is a structural design pattern that lets you attach new behaviors to objects by placing them inside wrapper objects that contain these behaviors. It allows you to add functionality to objects dynamically without altering their structure.

### Intent

- Add behavior to objects dynamically
- Extend functionality without modifying existing code
- Compose behaviors at runtime
- Follow Open/Closed Principle

---

## Problem

Sometimes you need to add functionality to objects, but:
- Inheritance is not flexible (static)
- Subclassing creates too many classes
- You want to add/remove features at runtime
- You need to combine multiple features

### Example Problem

```python
class Coffee:
    def cost(self):
        return 5.0

class CoffeeWithMilk(Coffee):
    def cost(self):
        return super().cost() + 2.0

class CoffeeWithSugar(Coffee):
    def cost(self):
        return super().cost() + 1.0

class CoffeeWithMilkAndSugar(Coffee):
    def cost(self):
        return super().cost() + 2.0 + 1.0

# Problem: Combinatorial explosion!
# Coffee, CoffeeWithMilk, CoffeeWithSugar, CoffeeWithMilkAndSugar,
# CoffeeWithMilkAndSugarAndWhip, etc. - too many classes!
```

---

## Solution

The Decorator Pattern solves this by:
1. Creating a decorator class that wraps the component
2. Implementing the same interface as the component
3. Delegating to the wrapped component
4. Adding behavior before/after delegation

---

## Structure

```
┌──────────────┐
│  Component   │
│  (interface) │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│  Decorator   │
│  (abstract)  │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ Concrete     │
│ Decorator    │
└──────────────┘
```

**Participants**:
- **Component**: Interface for objects that can have responsibilities added
- **ConcreteComponent**: Defines object to which responsibilities can be added
- **Decorator**: Maintains reference to Component and defines interface
- **ConcreteDecorator**: Adds responsibilities to component

---

## Implementation

### Basic Decorator Pattern

```python
from abc import ABC, abstractmethod

# Component interface
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass

# Concrete Component
class SimpleCoffee(Coffee):
    def cost(self) -> float:
        return 5.0
    
    def description(self) -> str:
        return "Simple coffee"

# Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee
    
    def cost(self) -> float:
        return self._coffee.cost()
    
    def description(self) -> str:
        return self._coffee.description()

# Concrete Decorators
class MilkDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 2.0
    
    def description(self) -> str:
        return self._coffee.description() + ", milk"

class SugarDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 1.0
    
    def description(self) -> str:
        return self._coffee.description() + ", sugar"

class WhipDecorator(CoffeeDecorator):
    def cost(self) -> float:
        return self._coffee.cost() + 1.5
    
    def description(self) -> str:
        return self._coffee.description() + ", whip"

# Usage
coffee = SimpleCoffee()
print(f"{coffee.description()}: ${coffee.cost()}")  # Simple coffee: $5.0

coffee_with_milk = MilkDecorator(coffee)
print(f"{coffee_with_milk.description()}: ${coffee_with_milk.cost()}")  # Simple coffee, milk: $7.0

coffee_full = WhipDecorator(SugarDecorator(MilkDecorator(SimpleCoffee())))
print(f"{coffee_full.description()}: ${coffee_full.cost()}")  # Simple coffee, milk, sugar, whip: $9.5
```

### Python Decorator Syntax

Python's decorator syntax makes this pattern very natural:

```python
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def logging_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@timing_decorator
@logging_decorator
def expensive_operation(n):
    return sum(range(n))

result = expensive_operation(1000000)
```

---

## Real-World Examples

### Example 1: Text Formatting Decorators

```python
from abc import ABC, abstractmethod

class TextComponent(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class PlainText(TextComponent):
    def __init__(self, text: str):
        self.text = text
    
    def render(self) -> str:
        return self.text

class TextDecorator(TextComponent):
    def __init__(self, component: TextComponent):
        self._component = component
    
    def render(self) -> str:
        return self._component.render()

class BoldDecorator(TextDecorator):
    def render(self) -> str:
        return f"<b>{self._component.render()}</b>"

class ItalicDecorator(TextDecorator):
    def render(self) -> str:
        return f"<i>{self._component.render()}</i>"

class UnderlineDecorator(TextDecorator):
    def render(self) -> str:
        return f"<u>{self._component.render()}</u>"

# Usage
text = PlainText("Hello, World!")
formatted = UnderlineDecorator(ItalicDecorator(BoldDecorator(text)))
print(formatted.render())  # <u><i><b>Hello, World!</b></i></u>
```

### Example 2: HTTP Request Decorators

```python
from abc import ABC, abstractmethod

class HTTPRequest(ABC):
    @abstractmethod
    def execute(self) -> dict:
        pass

class BasicRequest(HTTPRequest):
    def __init__(self, url: str):
        self.url = url
    
    def execute(self) -> dict:
        return {"status": 200, "data": "Response from " + self.url}

class RequestDecorator(HTTPRequest):
    def __init__(self, request: HTTPRequest):
        self._request = request
    
    def execute(self) -> dict:
        return self._request.execute()

class RetryDecorator(RequestDecorator):
    def __init__(self, request: HTTPRequest, max_retries: int = 3):
        super().__init__(request)
        self.max_retries = max_retries
    
    def execute(self) -> dict:
        for attempt in range(self.max_retries):
            try:
                return self._request.execute()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries}")

class TimeoutDecorator(RequestDecorator):
    def __init__(self, request: HTTPRequest, timeout: int = 5):
        super().__init__(request)
        self.timeout = timeout
    
    def execute(self) -> dict:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Request timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            result = self._request.execute()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            signal.alarm(0)
            raise

class LoggingDecorator(RequestDecorator):
    def execute(self) -> dict:
        print(f"Executing request to {self._request.url if hasattr(self._request, 'url') else 'unknown'}")
        result = self._request.execute()
        print(f"Request completed with status {result.get('status')}")
        return result

# Usage
request = BasicRequest("https://api.example.com")
decorated_request = LoggingDecorator(
    RetryDecorator(
        TimeoutDecorator(request, timeout=10),
        max_retries=3
    )
)
response = decorated_request.execute()
```

### Example 3: File I/O Decorators

```python
from abc import ABC, abstractmethod

class FileOperation(ABC):
    @abstractmethod
    def read(self, filename: str) -> str:
        pass
    
    @abstractmethod
    def write(self, filename: str, content: str):
        pass

class BasicFileOperation(FileOperation):
    def read(self, filename: str) -> str:
        with open(filename, 'r') as f:
            return f.read()
    
    def write(self, filename: str, content: str):
        with open(filename, 'w') as f:
            f.write(content)

class FileOperationDecorator(FileOperation):
    def __init__(self, operation: FileOperation):
        self._operation = operation
    
    def read(self, filename: str) -> str:
        return self._operation.read(filename)
    
    def write(self, filename: str, content: str):
        self._operation.write(filename, content)

class EncryptionDecorator(FileOperationDecorator):
    def read(self, filename: str) -> str:
        encrypted = self._operation.read(filename)
        # Simple decryption (for demo)
        return encrypted[::-1]  # Reverse string
    
    def write(self, filename: str, content: str):
        # Simple encryption (for demo)
        encrypted = content[::-1]  # Reverse string
        self._operation.write(filename, encrypted)

class CompressionDecorator(FileOperationDecorator):
    def read(self, filename: str) -> str:
        compressed = self._operation.read(filename)
        # Simple decompression (for demo)
        return compressed.replace('_COMPRESSED_', '')
    
    def write(self, filename: str, content: str):
        # Simple compression (for demo)
        compressed = content.replace(' ', '_COMPRESSED_')
        self._operation.write(filename, compressed)

# Usage
file_op = BasicFileOperation()
encrypted_op = EncryptionDecorator(file_op)
compressed_encrypted_op = CompressionDecorator(EncryptionDecorator(file_op))

compressed_encrypted_op.write("test.txt", "Hello World")
content = compressed_encrypted_op.read("test.txt")
print(content)  # Hello World
```

---

## Use Cases

### When to Use Decorator Pattern

✅ **Dynamic Behavior**: When you need to add/remove behavior at runtime
✅ **Multiple Combinations**: When you need many combinations of features
✅ **Avoiding Subclass Explosion**: When subclassing would create too many classes
✅ **Extending Functionality**: When you want to extend functionality without modifying code
✅ **Composition Over Inheritance**: When composition is preferred over inheritance
✅ **Cross-Cutting Concerns**: For logging, caching, validation, etc.

### When NOT to Use

❌ **Simple Extensions**: Overkill for simple, static extensions
❌ **Performance Critical**: Adds layers that can impact performance
❌ **Too Many Decorators**: If you need too many decorators, consider refactoring
❌ **Can Modify Source**: If you can modify the component directly

---

## Pros and Cons

### Advantages

✅ **Flexibility**: Add/remove behavior dynamically
✅ **Open/Closed Principle**: Extend without modifying existing code
✅ **Composition**: Favor composition over inheritance
✅ **Single Responsibility**: Each decorator has one responsibility
✅ **Runtime Configuration**: Configure behavior at runtime

### Disadvantages

❌ **Complexity**: Can create many small classes
❌ **Ordering**: Order of decorators can matter
❌ **Debugging**: Can be harder to debug with many layers
❌ **Performance**: Multiple layers add overhead

---

## Decorator vs Other Patterns

### Decorator vs Inheritance

- **Decorator**: Adds behavior at runtime, composition-based
- **Inheritance**: Adds behavior at compile-time, inheritance-based

### Decorator vs Adapter

- **Decorator**: Adds behavior without changing interface
- **Adapter**: Changes interface to make incompatible classes work

### Decorator vs Strategy

- **Decorator**: Adds behavior by wrapping
- **Strategy**: Changes behavior by swapping algorithms

---

## Best Practices

### 1. Keep Decorators Focused

```python
class SinglePurposeDecorator:
    def __init__(self, component):
        self._component = component
    
    def operation(self):
        # Do one thing
        result = self._component.operation()
        # Add one behavior
        return result
```

### 2. Use Python Decorator Syntax When Appropriate

```python
def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Decorator logic
        return func(*args, **kwargs)
    return wrapper

@decorator
def my_function():
    pass
```

### 3. Document Decorator Behavior

```python
class LoggingDecorator:
    """
    Decorator that adds logging to component operations.
    
    Logs:
    - Method calls
    - Arguments
    - Return values
    - Execution time
    """
    pass
```

### 4. Handle Decorator Ordering

```python
# Order matters!
decorated = DecoratorB(DecoratorA(component))  # A then B
decorated = DecoratorA(DecoratorB(component))  # B then A
```

### 5. Use `functools.wraps`

```python
from functools import wraps

def decorator(func):
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

---

## Python-Specific Considerations

### 1. Built-in Decorator Syntax

Python's `@decorator` syntax is perfect for this pattern:

```python
@timing
@logging
@cache
def expensive_function():
    pass
```

### 2. Class Decorators

```python
def add_method(cls):
    def new_method(self):
        return "New method"
    cls.new_method = new_method
    return cls

@add_method
class MyClass:
    pass
```

### 3. Property Decorators

```python
class Person:
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
```

### 4. Decorator with Arguments

```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("Hello")
```

### 5. Decorator Classes

```python
class Decorator:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        # Decorator logic
        return self.func(*args, **kwargs)

@Decorator
def my_function():
    pass
```

---

## Common Pitfalls

### 1. Forgetting to Call Wrapped Component

```python
# Bad: Doesn't call wrapped component
class BadDecorator:
    def operation(self):
        return "Only decorator behavior"  # Missing component call

# Good: Calls wrapped component
class GoodDecorator:
    def operation(self):
        result = self._component.operation()
        # Add behavior
        return result
```

### 2. Not Preserving Function Metadata

```python
# Bad: Loses function metadata
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Good: Preserves metadata
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### 3. Decorator Ordering Issues

```python
# Order matters - be explicit
@decorator_a
@decorator_b
def function():
    pass
# Executes: decorator_b wraps function, decorator_a wraps that
```

---

## Key Takeaways

- **Purpose**: Add behavior to objects dynamically
- **Use when**: Need runtime behavior modification, avoiding subclass explosion
- **Python**: Built-in `@decorator` syntax makes this natural
- **Benefits**: Flexibility, Open/Closed Principle, composition
- **Trade-off**: Can add complexity with many layers
- **Best practice**: Keep decorators focused, use `@wraps`, document behavior
- **Common use**: Logging, caching, validation, timing, retry logic

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python Decorators](https://docs.python.org/3/glossary.html#term-decorator)
- [functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
- [Decorator Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/decorator)

