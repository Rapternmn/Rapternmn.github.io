+++
title = "Prototype Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Comprehensive guide to Prototype Pattern: Creating objects by cloning existing instances, with Python implementations using copy module, deep vs shallow copy, registry pattern, use cases, and best practices."
+++

---

## Introduction

The **Prototype Pattern** is a creational design pattern that lets you copy existing objects without making your code dependent on their classes. Instead of creating objects from scratch, you clone existing instances (prototypes).

### Intent

- Create objects by cloning existing instances
- Avoid expensive object creation
- Reduce subclassing
- Configure objects at runtime

---

## Problem

Sometimes object creation is expensive:
- Complex initialization logic
- Database queries or network calls
- Heavy computations
- Multiple dependencies

Creating objects from scratch repeatedly can be inefficient.

### Example Problem

```python
class DatabaseConnection:
    def __init__(self):
        # Expensive operations
        print("Connecting to database...")
        print("Loading configuration...")
        print("Establishing connection...")
        self.connection = "Connected"
        self.config = self._load_config()  # Expensive
    
    def _load_config(self):
        # Simulate expensive operation
        import time
        time.sleep(1)  # Simulate slow operation
        return {"host": "localhost", "port": 5432}
    
    def query(self, sql: str):
        return f"Executing: {sql}"

# Problem: Creating multiple instances is expensive
conn1 = DatabaseConnection()  # Expensive!
conn2 = DatabaseConnection()  # Expensive again!
conn3 = DatabaseConnection()  # Expensive again!
```

---

## Solution

The Prototype Pattern solves this by:
1. Creating a prototype interface with a clone method
2. Implementing clone in concrete classes
3. Cloning existing instances instead of creating new ones
4. Using a prototype registry to manage prototypes

---

## Structure

```
┌──────────────────┐
│   Prototype      │
│  (interface)     │
├──────────────────┤
│ + clone()        │
└──────────────────┘
       ▲
       │
┌──────────────────┐
│ ConcretePrototype│
├──────────────────┤
│ + clone()        │
└──────────────────┘
```

**Participants**:
- **Prototype**: Interface for cloning
- **ConcretePrototype**: Implements cloning
- **Client**: Uses prototypes to create objects

---

## Implementation

### Basic Prototype Pattern

```python
import copy

class Prototype:
    """Base prototype class"""
    
    def clone(self):
        """Clone the object"""
        return copy.deepcopy(self)

class DatabaseConnection(Prototype):
    def __init__(self, host="localhost", port=5432):
        print("Creating database connection...")
        self.host = host
        self.port = port
        self.config = self._load_config()
    
    def _load_config(self):
        # Expensive operation
        return {"host": self.host, "port": self.port}
    
    def query(self, sql: str):
        return f"Query on {self.host}:{self.port} - {sql}"

# Usage
original = DatabaseConnection("localhost", 5432)
clone1 = original.clone()  # Fast - no expensive initialization!
clone2 = original.clone()  # Fast - no expensive initialization!

print(clone1.query("SELECT * FROM users"))
print(clone2.query("SELECT * FROM products"))
```

### Prototype with Custom Clone

```python
import copy

class Shape(Prototype):
    def __init__(self, color: str, x: int, y: int):
        self.color = color
        self.x = x
        self.y = y
    
    def clone(self):
        """Custom clone implementation"""
        return copy.deepcopy(self)
    
    def __str__(self):
        return f"Shape(color={self.color}, x={self.x}, y={self.y})"

class Circle(Shape):
    def __init__(self, color: str, x: int, y: int, radius: int):
        super().__init__(color, x, y)
        self.radius = radius
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        return f"Circle(color={self.color}, x={self.x}, y={self.y}, radius={self.radius})"

# Usage
original_circle = Circle("red", 10, 20, 5)
cloned_circle = original_circle.clone()

print(original_circle)  # Circle(color=red, x=10, y=20, radius=5)
print(cloned_circle)     # Circle(color=red, x=10, y=20, radius=5)

# Modify clone independently
cloned_circle.x = 30
cloned_circle.color = "blue"
print(cloned_circle)     # Circle(color=blue, x=30, y=20, radius=5)
print(original_circle)  # Circle(color=red, x=10, y=20, radius=5) - unchanged
```

### Prototype Registry

```python
import copy

class PrototypeRegistry:
    """Manages a registry of prototypes"""
    
    def __init__(self):
        self._prototypes = {}
    
    def register(self, name: str, prototype: Prototype):
        """Register a prototype"""
        self._prototypes[name] = prototype
    
    def unregister(self, name: str):
        """Unregister a prototype"""
        if name in self._prototypes:
            del self._prototypes[name]
    
    def clone(self, name: str):
        """Clone a prototype by name"""
        if name not in self._prototypes:
            raise ValueError(f"Prototype '{name}' not found")
        return self._prototypes[name].clone()
    
    def list_prototypes(self):
        """List all registered prototypes"""
        return list(self._prototypes.keys())

class Document(Prototype):
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.created_at = "2024-01-01"
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        return f"Document: {self.title} by {self.author}"

# Usage
registry = PrototypeRegistry()

# Register prototypes
template1 = Document("Template 1", "Content 1", "Admin")
template2 = Document("Template 2", "Content 2", "Admin")

registry.register("report_template", template1)
registry.register("letter_template", template2)

# Clone from registry
report1 = registry.clone("report_template")
report2 = registry.clone("report_template")

# Modify clones
report1.title = "Monthly Report - January"
report2.title = "Monthly Report - February"

print(report1)  # Document: Monthly Report - January by Admin
print(report2)  # Document: Monthly Report - February by Admin
```

---

## Deep Copy vs Shallow Copy

### Shallow Copy

```python
import copy

class Person:
    def __init__(self, name: str, address: dict):
        self.name = name
        self.address = address  # Reference to dict

# Shallow copy
person1 = Person("John", {"city": "New York", "zip": "10001"})
person2 = copy.copy(person1)  # Shallow copy

person2.name = "Jane"  # Only affects person2
person2.address["city"] = "Boston"  # Affects both! (shared reference)

print(person1.name)           # John
print(person1.address["city"])  # Boston (changed!)
print(person2.name)           # Jane
print(person2.address["city"])  # Boston
```

### Deep Copy

```python
import copy

class Person:
    def __init__(self, name: str, address: dict):
        self.name = name
        self.address = address

# Deep copy
person1 = Person("John", {"city": "New York", "zip": "10001"})
person2 = copy.deepcopy(person1)  # Deep copy

person2.name = "Jane"
person2.address["city"] = "Boston"  # Only affects person2

print(person1.name)           # John
print(person1.address["city"])  # New York (unchanged!)
print(person2.name)           # Jane
print(person2.address["city"])  # Boston
```

**Use deep copy when**:
- Objects contain nested mutable objects
- You want completely independent copies
- Objects have complex structures

**Use shallow copy when**:
- Objects are simple (no nested mutable objects)
- You want to share some references
- Performance is critical

---

## Real-World Examples

### Example 1: Game Object Prototypes

```python
import copy

class GameObject(Prototype):
    def __init__(self, name: str, position: tuple, health: int):
        self.name = name
        self.position = position
        self.health = health
        self.inventory = []
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        return f"{self.name} at {self.position} (HP: {self.health})"

class Enemy(GameObject):
    def __init__(self, name: str, position: tuple, health: int, damage: int):
        super().__init__(name, position, health)
        self.damage = damage
    
    def clone(self):
        return copy.deepcopy(self)

# Create prototype
goblin_prototype = Enemy("Goblin", (0, 0), 50, 10)

# Clone to create multiple enemies
enemies = []
for i in range(5):
    enemy = goblin_prototype.clone()
    enemy.position = (i * 10, 0)  # Different positions
    enemies.append(enemy)

for enemy in enemies:
    print(enemy)
```

### Example 2: Configuration Prototypes

```python
import copy

class Configuration(Prototype):
    def __init__(self):
        self.settings = {}
        self.load_defaults()
    
    def load_defaults(self):
        """Expensive operation - load from file/database"""
        print("Loading default configuration...")
        self.settings = {
            "database_url": "localhost:5432",
            "api_key": "default_key",
            "timeout": 30,
            "retry_count": 3
        }
    
    def clone(self):
        return copy.deepcopy(self)
    
    def update_setting(self, key: str, value):
        self.settings[key] = value

# Create prototype (expensive)
default_config = Configuration()

# Clone for different environments
dev_config = default_config.clone()
dev_config.update_setting("database_url", "dev-db:5432")
dev_config.update_setting("api_key", "dev_key")

prod_config = default_config.clone()
prod_config.update_setting("database_url", "prod-db:5432")
prod_config.update_setting("api_key", "prod_key")

print(dev_config.settings)
print(prod_config.settings)
```

### Example 3: UI Component Prototypes

```python
import copy

class UIComponent(Prototype):
    def __init__(self, component_type: str, style: dict):
        self.component_type = component_type
        self.style = style
        self.content = ""
    
    def clone(self):
        return copy.deepcopy(self)
    
    def set_content(self, content: str):
        self.content = content
    
    def __str__(self):
        return f"{self.component_type}: {self.content}"

# Create button prototype
button_prototype = UIComponent("Button", {
    "color": "blue",
    "size": "medium",
    "border": "rounded"
})

# Clone to create multiple buttons
buttons = []
for label in ["Submit", "Cancel", "Reset"]:
    button = button_prototype.clone()
    button.set_content(label)
    buttons.append(button)

for button in buttons:
    print(button)
```

---

## Use Cases

### When to Use Prototype Pattern

✅ **Expensive Object Creation**: When creating objects is expensive
✅ **Similar Objects**: When you need many similar objects
✅ **Runtime Configuration**: When object configuration is determined at runtime
✅ **Avoid Subclassing**: When you want to avoid creating many subclasses
✅ **Object Pooling**: When implementing object pools
✅ **Undo/Redo**: When implementing undo/redo functionality

### When NOT to Use

❌ **Simple Objects**: Overkill for simple objects
❌ **Few Instances**: When you only need a few instances
❌ **Different Structures**: When objects have very different structures
❌ **Performance**: When copying is more expensive than creation

---

## Pros and Cons

### Advantages

✅ **Performance**: Avoids expensive object creation
✅ **Flexibility**: Configure objects at runtime
✅ **Reduces Subclasses**: No need for many subclasses
✅ **Dynamic Behavior**: Add/remove prototypes at runtime
✅ **Hides Complexity**: Client doesn't know concrete classes

### Disadvantages

❌ **Copy Complexity**: Deep copying complex objects can be expensive
❌ **Circular References**: Can cause issues with circular references
❌ **Clone Implementation**: Must implement clone correctly
❌ **Memory**: Cloning can use more memory

---

## Prototype vs Other Patterns

### Prototype vs Factory

- **Prototype**: Clones existing instances
- **Factory**: Creates new instances from scratch

### Prototype vs Builder

- **Prototype**: Copies existing objects
- **Builder**: Constructs objects step by step

### Prototype vs Singleton

- **Prototype**: Creates multiple copies
- **Singleton**: Ensures only one instance

---

## Best Practices

### 1. Use `copy.deepcopy()` for Complex Objects

```python
import copy

class ComplexObject(Prototype):
    def clone(self):
        return copy.deepcopy(self)
```

### 2. Implement Custom Clone When Needed

```python
class CustomObject(Prototype):
    def clone(self):
        # Custom clone logic
        cloned = CustomObject()
        cloned.field1 = self.field1
        cloned.field2 = self.field2.copy()  # Manual deep copy
        return cloned
```

### 3. Use Registry for Multiple Prototypes

```python
class PrototypeRegistry:
    def __init__(self):
        self._prototypes = {}
    
    def register(self, name, prototype):
        self._prototypes[name] = prototype
    
    def clone(self, name):
        return self._prototypes[name].clone()
```

### 4. Handle Circular References

```python
import copy

class Node(Prototype):
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def clone(self):
        # Use memo to handle circular references
        memo = {}
        return copy.deepcopy(self, memo)
```

### 5. Consider `__copy__` and `__deepcopy__`

```python
import copy

class CustomObject:
    def __init__(self, data):
        self.data = data
    
    def __copy__(self):
        # Custom shallow copy
        return CustomObject(self.data)
    
    def __deepcopy__(self, memo):
        # Custom deep copy
        return CustomObject(copy.deepcopy(self.data, memo))
```

---

## Python-Specific Considerations

### 1. Using `copy` Module

```python
import copy

# Shallow copy
shallow = copy.copy(original)

# Deep copy
deep = copy.deepcopy(original)
```

### 2. Using `__copy__` and `__deepcopy__`

```python
import copy

class MyClass:
    def __init__(self, value):
        self.value = value
        self.nested = {"key": "value"}
    
    def __copy__(self):
        # Custom shallow copy
        new = MyClass(self.value)
        new.nested = self.nested  # Shared reference
        return new
    
    def __deepcopy__(self, memo):
        # Custom deep copy
        new = MyClass(copy.deepcopy(self.value, memo))
        new.nested = copy.deepcopy(self.nested, memo)
        return new
```

### 3. Using `dataclasses` with `field()`

```python
from dataclasses import dataclass, field
import copy

@dataclass
class Prototype:
    name: str
    items: list = field(default_factory=list)
    
    def clone(self):
        return copy.deepcopy(self)
```

### 4. Prototype with `__slots__`

```python
import copy

class SlottedPrototype:
    __slots__ = ['value', 'data']
    
    def __init__(self, value, data):
        self.value = value
        self.data = data
    
    def clone(self):
        return copy.deepcopy(self)
```

---

## Common Pitfalls

### 1. Shallow Copy with Nested Objects

```python
# Problem: Shallow copy shares nested objects
original = {"nested": {"value": 1}}
shallow = copy.copy(original)
shallow["nested"]["value"] = 2
print(original["nested"]["value"])  # 2 - changed!

# Solution: Use deep copy
deep = copy.deepcopy(original)
deep["nested"]["value"] = 2
print(original["nested"]["value"])  # 1 - unchanged
```

### 2. Circular References

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

# Create circular reference
node1 = Node(1)
node2 = Node(2)
node1.children.append(node2)
node2.parent = node1

# Deep copy handles circular references
cloned = copy.deepcopy(node1)
```

### 3. Not Implementing Clone Correctly

```python
# Wrong: Returns same reference
def clone(self):
    return self  # Not a clone!

# Correct: Returns new object
def clone(self):
    return copy.deepcopy(self)
```

---

## Key Takeaways

- **Purpose**: Create objects by cloning existing instances
- **Use when**: Object creation is expensive or you need similar objects
- **Deep vs Shallow**: Use deep copy for nested mutable objects
- **Registry**: Use prototype registry to manage multiple prototypes
- **Python**: Leverage `copy` module and `__deepcopy__` method
- **Best practice**: Use `copy.deepcopy()` for complex objects
- **Performance**: Cloning can be faster than creating from scratch

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python copy Module](https://docs.python.org/3/library/copy.html)
- [Prototype Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/prototype)

