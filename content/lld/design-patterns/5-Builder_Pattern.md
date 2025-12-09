+++
title = "Builder Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Comprehensive guide to Builder Pattern: Constructing complex objects step by step, with Python implementations, fluent interface, director pattern, use cases, and best practices."
+++

---

## Introduction

The **Builder Pattern** is a creational design pattern that lets you construct complex objects step by step. It allows you to produce different types and representations of an object using the same construction code.

### Intent

- Separate construction of complex objects from their representation
- Allow step-by-step construction
- Create different representations using the same construction process
- Provide a fluent interface for object construction

---

## Problem

When constructing complex objects, you might face:
- Constructors with many parameters (telescoping constructor anti-pattern)
- Difficulty in creating objects with optional parameters
- Need to create objects in multiple steps
- Want to reuse construction code for different representations

### Example Problem

```python
class Pizza:
    def __init__(self, size, cheese=False, pepperoni=False, 
                 mushrooms=False, onions=False, bacon=False):
        self.size = size
        self.cheese = cheese
        self.pepperoni = pepperoni
        self.mushrooms = mushrooms
        self.onions = onions
        self.bacon = bacon
    
    def __str__(self):
        toppings = []
        if self.cheese: toppings.append("cheese")
        if self.pepperoni: toppings.append("pepperoni")
        if self.mushrooms: toppings.append("mushrooms")
        if self.onions: toppings.append("onions")
        if self.bacon: toppings.append("bacon")
        return f"{self.size} pizza with {', '.join(toppings) if toppings else 'no toppings'}"

# Problem: Hard to read, many parameters, order matters
pizza = Pizza("large", True, True, False, True, False)
# What does this mean? Hard to understand!
```

---

## Solution

The Builder Pattern solves this by:
1. Creating a builder class that handles construction
2. Providing methods for each construction step
3. Allowing method chaining (fluent interface)
4. Separating construction from representation

---

## Structure

```
┌──────────────┐
│   Product    │
└──────────────┘
       ▲
       │
┌──────────────┐      ┌──────────────┐
│   Builder    │─────>│   Concrete   │
│  (interface) │      │   Builder    │
└──────────────┘      └──────────────┘
       ▲
       │
┌──────────────┐
│   Director   │
│  (optional)  │
└──────────────┘
```

**Participants**:
- **Product**: Complex object being built
- **Builder**: Abstract interface for building steps
- **ConcreteBuilder**: Implements building steps
- **Director**: (Optional) Constructs using builder

---

## Implementation

### Basic Builder Pattern

```python
class Pizza:
    def __init__(self):
        self.size = None
        self.cheese = False
        self.pepperoni = False
        self.mushrooms = False
        self.onions = False
        self.bacon = False
    
    def __str__(self):
        toppings = []
        if self.cheese: toppings.append("cheese")
        if self.pepperoni: toppings.append("pepperoni")
        if self.mushrooms: toppings.append("mushrooms")
        if self.onions: toppings.append("onions")
        if self.bacon: toppings.append("bacon")
        return f"{self.size} pizza with {', '.join(toppings) if toppings else 'no toppings'}"

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size: str):
        self.pizza.size = size
        return self  # Return self for method chaining
    
    def add_cheese(self):
        self.pizza.cheese = True
        return self
    
    def add_pepperoni(self):
        self.pizza.pepperoni = True
        return self
    
    def add_mushrooms(self):
        self.pizza.mushrooms = True
        return self
    
    def add_onions(self):
        self.pizza.onions = True
        return self
    
    def add_bacon(self):
        self.pizza.bacon = True
        return self
    
    def build(self):
        return self.pizza

# Usage - Fluent interface
pizza = (PizzaBuilder()
         .set_size("large")
         .add_cheese()
         .add_pepperoni()
         .add_mushrooms()
         .build())

print(pizza)  # large pizza with cheese, pepperoni, mushrooms
```

### Builder with Abstract Base Class

```python
from abc import ABC, abstractmethod

class Pizza(ABC):
    def __init__(self):
        self.size = None
        self.toppings = []
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class MargheritaPizza(Pizza):
    def get_name(self) -> str:
        return "Margherita"

class PepperoniPizza(Pizza):
    def get_name(self) -> str:
        return "Pepperoni"

class PizzaBuilder(ABC):
    def __init__(self):
        self.pizza = None
    
    @abstractmethod
    def create_pizza(self):
        pass
    
    @abstractmethod
    def add_toppings(self):
        pass
    
    def set_size(self, size: str):
        self.pizza.size = size
        return self
    
    def build(self):
        return self.pizza

class MargheritaBuilder(PizzaBuilder):
    def create_pizza(self):
        self.pizza = MargheritaPizza()
        return self
    
    def add_toppings(self):
        self.pizza.toppings = ["cheese", "tomato", "basil"]
        return self

class PepperoniBuilder(PizzaBuilder):
    def create_pizza(self):
        self.pizza = PepperoniPizza()
        return self
    
    def add_toppings(self):
        self.pizza.toppings = ["cheese", "pepperoni"]
        return self

# Usage
margherita = (MargheritaBuilder()
              .create_pizza()
              .set_size("medium")
              .add_toppings()
              .build())

print(margherita.get_name())  # Margherita
```

### Builder with Director (Optional)

```python
class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size: str):
        self.pizza.size = size
        return self
    
    def add_cheese(self):
        self.pizza.cheese = True
        return self
    
    def add_pepperoni(self):
        self.pizza.pepperoni = True
        return self
    
    def build(self):
        return self.pizza

class PizzaDirector:
    """Director - knows how to build specific pizza types"""
    
    def __init__(self, builder: PizzaBuilder):
        self.builder = builder
    
    def build_margherita(self):
        return (self.builder
                .set_size("medium")
                .add_cheese()
                .build())
    
    def build_pepperoni(self):
        return (self.builder
                .set_size("large")
                .add_cheese()
                .add_pepperoni()
                .build())

# Usage
builder = PizzaBuilder()
director = PizzaDirector(builder)

margherita = director.build_margherita()
pepperoni = director.build_pepperoni()
```

---

## Real-World Examples

### Example 1: SQL Query Builder

```python
class QueryBuilder:
    def __init__(self):
        self.query = {
            'select': [],
            'from': None,
            'where': [],
            'order_by': None,
            'limit': None
        }
    
    def select(self, *columns):
        self.query['select'].extend(columns)
        return self
    
    def from_table(self, table: str):
        self.query['from'] = table
        return self
    
    def where(self, condition: str):
        self.query['where'].append(condition)
        return self
    
    def order_by(self, column: str, direction: str = "ASC"):
        self.query['order_by'] = f"{column} {direction}"
        return self
    
    def limit(self, count: int):
        self.query['limit'] = count
        return self
    
    def build(self) -> str:
        if not self.query['select']:
            raise ValueError("SELECT clause is required")
        if not self.query['from']:
            raise ValueError("FROM clause is required")
        
        sql = f"SELECT {', '.join(self.query['select'])}"
        sql += f" FROM {self.query['from']}"
        
        if self.query['where']:
            sql += f" WHERE {' AND '.join(self.query['where'])}"
        
        if self.query['order_by']:
            sql += f" ORDER BY {self.query['order_by']}"
        
        if self.query['limit']:
            sql += f" LIMIT {self.query['limit']}"
        
        return sql

# Usage
query = (QueryBuilder()
         .select("id", "name", "email")
         .from_table("users")
         .where("age > 18")
         .where("active = 1")
         .order_by("name", "ASC")
         .limit(10)
         .build())

print(query)
# SELECT id, name, email FROM users WHERE age > 18 AND active = 1 ORDER BY name ASC LIMIT 10
```

### Example 2: HTTP Request Builder

```python
class HTTPRequest:
    def __init__(self):
        self.method = None
        self.url = None
        self.headers = {}
        self.body = None
        self.params = {}
    
    def __str__(self):
        return f"{self.method} {self.url}"

class HTTPRequestBuilder:
    def __init__(self):
        self.request = HTTPRequest()
    
    def method(self, method: str):
        self.request.method = method.upper()
        return self
    
    def url(self, url: str):
        self.request.url = url
        return self
    
    def header(self, key: str, value: str):
        self.request.headers[key] = value
        return self
    
    def body(self, data: str):
        self.request.body = data
        return self
    
    def param(self, key: str, value: str):
        self.request.params[key] = value
        return self
    
    def build(self):
        return self.request

# Usage
request = (HTTPRequestBuilder()
           .method("POST")
           .url("https://api.example.com/users")
           .header("Content-Type", "application/json")
           .header("Authorization", "Bearer token123")
           .body('{"name": "John", "age": 30}')
           .build())

print(request)  # POST https://api.example.com/users
```

### Example 3: Computer Configuration Builder

```python
class Computer:
    def __init__(self):
        self.cpu = None
        self.memory = None
        self.storage = None
        self.gpu = None
        self.motherboard = None
    
    def __str__(self):
        parts = []
        if self.cpu: parts.append(f"CPU: {self.cpu}")
        if self.memory: parts.append(f"Memory: {self.memory}")
        if self.storage: parts.append(f"Storage: {self.storage}")
        if self.gpu: parts.append(f"GPU: {self.gpu}")
        if self.motherboard: parts.append(f"Motherboard: {self.motherboard}")
        return " | ".join(parts)

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()
    
    def set_cpu(self, cpu: str):
        self.computer.cpu = cpu
        return self
    
    def set_memory(self, memory: str):
        self.computer.memory = memory
        return self
    
    def set_storage(self, storage: str):
        self.computer.storage = storage
        return self
    
    def set_gpu(self, gpu: str):
        self.computer.gpu = gpu
        return self
    
    def set_motherboard(self, motherboard: str):
        self.computer.motherboard = motherboard
        return self
    
    def build(self):
        # Validation
        if not self.computer.cpu:
            raise ValueError("CPU is required")
        if not self.computer.memory:
            raise ValueError("Memory is required")
        return self.computer

# Usage
gaming_pc = (ComputerBuilder()
             .set_cpu("Intel i9-12900K")
             .set_memory("32GB DDR4")
             .set_storage("1TB NVMe SSD")
             .set_gpu("NVIDIA RTX 4090")
             .set_motherboard("ASUS ROG Strix Z690")
             .build())

print(gaming_pc)
```

---

## Use Cases

### When to Use Builder Pattern

✅ **Complex Objects**: When constructing complex objects with many parameters
✅ **Optional Parameters**: When object has many optional parameters
✅ **Step-by-Step Construction**: When construction needs multiple steps
✅ **Different Representations**: When you need different representations of same object
✅ **Fluent Interface**: When you want a readable, chainable API
✅ **Immutable Objects**: When building immutable objects

### When NOT to Use

❌ **Simple Objects**: Overkill for simple objects with few parameters
❌ **Fixed Structure**: When object structure is always the same
❌ **Performance Critical**: Adds overhead for simple cases
❌ **Few Parameters**: If object has only 2-3 parameters

---

## Pros and Cons

### Advantages

✅ **Readability**: Fluent interface makes code more readable
✅ **Flexibility**: Easy to add/remove construction steps
✅ **Reusability**: Same construction code for different representations
✅ **Validation**: Can validate object before construction
✅ **Step-by-Step**: Construct objects in multiple steps
✅ **Optional Parameters**: Handle optional parameters elegantly

### Disadvantages

❌ **Complexity**: Adds more classes to codebase
❌ **Overhead**: Additional abstraction layer
❌ **Verbose**: Can be verbose for simple objects
❌ **Maintenance**: More code to maintain

---

## Builder vs Other Patterns

### Builder vs Factory

- **Builder**: Focuses on step-by-step construction
- **Factory**: Focuses on creating objects (what to create)

### Builder vs Prototype

- **Builder**: Constructs objects from scratch
- **Prototype**: Clones existing objects

### Builder vs Constructor

- **Builder**: For complex objects with many parameters
- **Constructor**: For simple objects

---

## Best Practices

### 1. Return Self for Method Chaining

```python
class Builder:
    def set_value(self, value):
        self.value = value
        return self  # Enable chaining
```

### 2. Validate in build() Method

```python
def build(self):
    if not self.required_field:
        raise ValueError("Required field missing")
    return self.object
```

### 3. Use Type Hints

```python
from typing import Self

class Builder:
    def set_value(self, value: str) -> Self:
        self.value = value
        return self
```

### 4. Separate Builder from Product

```python
# Product
class Product:
    pass

# Builder
class ProductBuilder:
    def build(self) -> Product:
        return Product()
```

### 5. Consider Using Dataclasses

```python
from dataclasses import dataclass

@dataclass
class Product:
    field1: str
    field2: int

class ProductBuilder:
    def build(self) -> Product:
        return Product(field1=self.field1, field2=self.field2)
```

---

## Python-Specific Considerations

### 1. Using `@dataclass` with Builder

```python
from dataclasses import dataclass

@dataclass
class Product:
    field1: str
    field2: int = 0
    field3: bool = False

class ProductBuilder:
    def __init__(self):
        self.field1 = None
        self.field2 = 0
        self.field3 = False
    
    def set_field1(self, value: str):
        self.field1 = value
        return self
    
    def build(self) -> Product:
        return Product(
            field1=self.field1,
            field2=self.field2,
            field3=self.field3
        )
```

### 2. Builder with `__init_subclass__`

```python
class Builder:
    def __init_subclass__(cls, product_class=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._product_class = product_class
    
    def build(self):
        return self._product_class(**self._data)
```

### 3. Using `typing.Self` (Python 3.11+)

```python
from typing import Self

class Builder:
    def set_value(self, value: str) -> Self:
        self.value = value
        return self
```

---

## Key Takeaways

- **Purpose**: Construct complex objects step by step
- **Use when**: Object has many parameters or optional components
- **Benefits**: Readability, flexibility, validation
- **Fluent Interface**: Method chaining for readable code
- **Director**: Optional class that knows how to build specific types
- **Python**: Leverage type hints and dataclasses
- **Best practice**: Return self for method chaining

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Builder Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/builder)

