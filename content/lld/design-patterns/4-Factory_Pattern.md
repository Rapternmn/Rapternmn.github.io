+++
title = "Factory Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Comprehensive guide to Factory Pattern: Creating objects without specifying exact classes, with Python implementations, variations (Simple Factory, Factory Method, Abstract Factory), use cases, and best practices."
+++

---

## Introduction

The **Factory Pattern** is a creational design pattern that provides an interface for creating objects without specifying their exact classes. It encapsulates object creation logic and makes the code more flexible and maintainable.

### Intent

- Encapsulate object creation logic
- Decouple object creation from usage
- Provide a way to create objects without specifying exact classes
- Support multiple types of objects with a common interface

---

## Problem

When you need to create objects, directly instantiating classes can lead to:
- Tight coupling between creator and concrete classes
- Difficulty in extending with new types
- Violation of Open/Closed Principle
- Complex conditional logic for object creation

### Example Problem

```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Duck:
    def speak(self):
        return "Quack!"

# Problem: Tight coupling and complex conditionals
def create_animal(animal_type):
    if animal_type == "dog":
        return Dog()
    elif animal_type == "cat":
        return Cat()
    elif animal_type == "duck":
        return Duck()
    else:
        raise ValueError(f"Unknown animal type: {animal_type}")

# Adding a new animal requires modifying this function
```

---

## Solution

The Factory Pattern solves this by:
1. Creating a factory class/method that handles object creation
2. Using a common interface for all products
3. Encapsulating creation logic in one place
4. Making it easy to add new types without modifying existing code

---

## Variations of Factory Pattern

### 1. Simple Factory (Static Factory)

A simple factory method that creates objects based on a parameter.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

class Duck(Animal):
    def speak(self) -> str:
        return "Quack!"

class AnimalFactory:
    """Simple Factory - static method to create animals"""
    
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        animals = {
            "dog": Dog,
            "cat": Cat,
            "duck": Duck
        }
        
        animal_class = animals.get(animal_type.lower())
        if animal_class:
            return animal_class()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")

print(dog.speak())  # Woof!
print(cat.speak())  # Meow!
```

**Benefits**:
- Simple and straightforward
- Centralizes creation logic
- Easy to understand

**Limitations**:
- Adding new types requires modifying the factory
- Not as flexible as Factory Method

---

### 2. Factory Method Pattern

Each factory subclass creates a specific type of object. The creator class defines an abstract factory method.

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def drive(self) -> str:
        pass

class Car(Vehicle):
    def drive(self) -> str:
        return "Driving a car"

class Motorcycle(Vehicle):
    def drive(self) -> str:
        return "Riding a motorcycle"

class Truck(Vehicle):
    def drive(self) -> str:
        return "Driving a truck"

class VehicleFactory(ABC):
    """Abstract factory - defines factory method"""
    
    @abstractmethod
    def create_vehicle(self) -> Vehicle:
        pass
    
    def deliver(self) -> str:
        """Template method using factory method"""
        vehicle = self.create_vehicle()
        return f"Delivering: {vehicle.drive()}"

class CarFactory(VehicleFactory):
    def create_vehicle(self) -> Vehicle:
        return Car()

class MotorcycleFactory(VehicleFactory):
    def create_vehicle(self) -> Vehicle:
        return Motorcycle()

class TruckFactory(VehicleFactory):
    def create_vehicle(self) -> Vehicle:
        return Truck()

# Usage
car_factory = CarFactory()
print(car_factory.deliver())  # Delivering: Driving a car

motorcycle_factory = MotorcycleFactory()
print(motorcycle_factory.deliver())  # Delivering: Riding a motorcycle
```

**Benefits**:
- Follows Open/Closed Principle
- Easy to add new product types
- Creator and product are decoupled

---

### 3. Abstract Factory Pattern

Provides an interface for creating families of related objects without specifying their concrete classes.

```python
from abc import ABC, abstractmethod

# Abstract Products
class Button(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

class Checkbox(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

# Concrete Products - Windows
class WindowsButton(Button):
    def render(self) -> str:
        return "Windows Button"

class WindowsCheckbox(Checkbox):
    def render(self) -> str:
        return "Windows Checkbox"

# Concrete Products - Mac
class MacButton(Button):
    def render(self) -> str:
        return "Mac Button"

class MacCheckbox(Checkbox):
    def render(self) -> str:
        return "Mac Checkbox"

# Abstract Factory
class GUIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button:
        pass
    
    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass

# Concrete Factories
class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()
    
    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()

class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()
    
    def create_checkbox(self) -> Checkbox:
        return MacCheckbox()

# Client code
def create_ui(factory: GUIFactory):
    button = factory.create_button()
    checkbox = factory.create_checkbox()
    
    print(button.render())
    print(checkbox.render())

# Usage
windows_factory = WindowsFactory()
create_ui(windows_factory)
# Output:
# Windows Button
# Windows Checkbox

mac_factory = MacFactory()
create_ui(mac_factory)
# Output:
# Mac Button
# Mac Checkbox
```

**Benefits**:
- Ensures products from same family are used together
- Isolates concrete classes from client
- Easy to exchange product families
- Promotes consistency among products

---

## Real-World Examples

### Example 1: Database Connection Factory

```python
from abc import ABC, abstractmethod

class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self) -> str:
        pass
    
    @abstractmethod
    def query(self, sql: str) -> str:
        pass

class MySQLConnection(DatabaseConnection):
    def connect(self) -> str:
        return "Connected to MySQL"
    
    def query(self, sql: str) -> str:
        return f"MySQL: {sql}"

class PostgreSQLConnection(DatabaseConnection):
    def connect(self) -> str:
        return "Connected to PostgreSQL"
    
    def query(self, sql: str) -> str:
        return f"PostgreSQL: {sql}"

class SQLiteConnection(DatabaseConnection):
    def connect(self) -> str:
        return "Connected to SQLite"
    
    def query(self, sql: str) -> str:
        return f"SQLite: {sql}"

class DatabaseFactory:
    @staticmethod
    def create_connection(db_type: str) -> DatabaseConnection:
        connections = {
            "mysql": MySQLConnection,
            "postgresql": PostgreSQLConnection,
            "sqlite": SQLiteConnection
        }
        
        connection_class = connections.get(db_type.lower())
        if connection_class:
            return connection_class()
        raise ValueError(f"Unknown database type: {db_type}")

# Usage
factory = DatabaseFactory()
db = factory.create_connection("mysql")
print(db.connect())  # Connected to MySQL
print(db.query("SELECT * FROM users"))  # MySQL: SELECT * FROM users
```

### Example 2: Payment Processor Factory

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> str:
        pass

class CreditCardProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> str:
        return f"Processing ${amount} via Credit Card"

class PayPalProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> str:
        return f"Processing ${amount} via PayPal"

class BankTransferProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> str:
        return f"Processing ${amount} via Bank Transfer"

class PaymentFactory:
    _processors = {
        "credit_card": CreditCardProcessor,
        "paypal": PayPalProcessor,
        "bank_transfer": BankTransferProcessor
    }
    
    @classmethod
    def create_processor(cls, payment_type: str) -> PaymentProcessor:
        processor_class = cls._processors.get(payment_type.lower())
        if processor_class:
            return processor_class()
        raise ValueError(f"Unknown payment type: {payment_type}")
    
    @classmethod
    def register_processor(cls, payment_type: str, processor_class):
        """Allow dynamic registration of new processors"""
        cls._processors[payment_type.lower()] = processor_class

# Usage
factory = PaymentFactory()
processor = factory.create_processor("credit_card")
print(processor.process_payment(100.0))  # Processing $100.0 via Credit Card

# Register new processor dynamically
class CryptoProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> str:
        return f"Processing ${amount} via Cryptocurrency"

PaymentFactory.register_processor("crypto", CryptoProcessor)
crypto_processor = PaymentFactory.create_processor("crypto")
print(crypto_processor.process_payment(50.0))  # Processing $50.0 via Cryptocurrency
```

### Example 3: Document Parser Factory (Abstract Factory)

```python
from abc import ABC, abstractmethod

# Abstract Products
class Parser(ABC):
    @abstractmethod
    def parse(self, content: str) -> dict:
        pass

class Formatter(ABC):
    @abstractmethod
    def format(self, data: dict) -> str:
        pass

# Concrete Products - JSON
class JSONParser(Parser):
    def parse(self, content: str) -> dict:
        import json
        return json.loads(content)

class JSONFormatter(Formatter):
    def format(self, data: dict) -> str:
        import json
        return json.dumps(data, indent=2)

# Concrete Products - XML
class XMLParser(Parser):
    def parse(self, content: str) -> dict:
        # Simplified XML parsing
        return {"parsed": "XML data"}

class XMLFormatter(Formatter):
    def format(self, data: dict) -> str:
        return f"<root>{data}</root>"

# Abstract Factory
class DocumentFactory(ABC):
    @abstractmethod
    def create_parser(self) -> Parser:
        pass
    
    @abstractmethod
    def create_formatter(self) -> Formatter:
        pass

# Concrete Factories
class JSONFactory(DocumentFactory):
    def create_parser(self) -> Parser:
        return JSONParser()
    
    def create_formatter(self) -> Formatter:
        return JSONFormatter()

class XMLFactory(DocumentFactory):
    def create_parser(self) -> Parser:
        return XMLParser()
    
    def create_formatter(self) -> Formatter:
        return XMLFormatter()

# Client
def process_document(factory: DocumentFactory, content: str):
    parser = factory.create_parser()
    formatter = factory.create_formatter()
    
    data = parser.parse(content)
    return formatter.format(data)

# Usage
json_factory = JSONFactory()
result = process_document(json_factory, '{"key": "value"}')
print(result)
```

---

## Use Cases

### When to Use Factory Pattern

✅ **Object Creation Complexity**: When object creation is complex or involves multiple steps
✅ **Multiple Product Types**: When you have multiple related product types
✅ **Decoupling**: When you want to decouple object creation from usage
✅ **Extensibility**: When you need to easily add new product types
✅ **Configuration-Based**: When object type is determined at runtime
✅ **Family of Products**: When products need to be used together (Abstract Factory)

### When NOT to Use

❌ **Simple Object Creation**: Overkill for simple object creation
❌ **Few Product Types**: If you only have one or two product types
❌ **Static Types**: When product type is known at compile time
❌ **Performance Critical**: Adds a layer of indirection

---

## Pros and Cons

### Advantages

✅ **Loose Coupling**: Decouples object creation from usage
✅ **Single Responsibility**: Creation logic in one place
✅ **Open/Closed Principle**: Easy to add new types without modifying existing code
✅ **Code Reusability**: Factory logic can be reused
✅ **Centralized Control**: All object creation in one place
✅ **Flexibility**: Easy to swap implementations

### Disadvantages

❌ **Complexity**: Adds another layer of abstraction
❌ **More Classes**: Can lead to more classes in the codebase
❌ **Over-engineering**: May be overkill for simple scenarios
❌ **Indirection**: Adds indirection which can make code harder to follow

---

## Factory Pattern vs Other Patterns

### Factory vs Builder

- **Factory**: Creates objects (focus on what)
- **Builder**: Constructs complex objects step by step (focus on how)

### Factory vs Singleton

- **Factory**: Creates multiple instances
- **Singleton**: Ensures only one instance

### Simple Factory vs Factory Method

- **Simple Factory**: One factory creates all types
- **Factory Method**: Each factory creates one specific type

### Factory Method vs Abstract Factory

- **Factory Method**: Creates one product
- **Abstract Factory**: Creates families of related products

---

## Best Practices

### 1. Use Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Product(ABC):
    @abstractmethod
    def operation(self):
        pass
```

### 2. Register Products Dynamically

```python
class Factory:
    _products = {}
    
    @classmethod
    def register(cls, name, product_class):
        cls._products[name] = product_class
    
    @classmethod
    def create(cls, name):
        return cls._products[name]()
```

### 3. Use Enums for Type Safety

```python
from enum import Enum

class AnimalType(Enum):
    DOG = "dog"
    CAT = "cat"
    DUCK = "duck"

class AnimalFactory:
    @staticmethod
    def create(animal_type: AnimalType):
        # Type-safe creation
        pass
```

### 4. Error Handling

```python
class Factory:
    @staticmethod
    def create(product_type: str):
        if product_type not in available_products:
            raise ValueError(f"Unknown product: {product_type}")
        return available_products[product_type]()
```

### 5. Consider Using `@classmethod` or `@staticmethod`

```python
class Factory:
    @classmethod
    def create(cls, type: str):
        # Can access class variables
        return cls._products[type]()
    
    @staticmethod
    def create_static(type: str):
        # Pure function, no class access needed
        return products[type]()
```

---

## Python-Specific Considerations

### 1. Using `__init_subclass__` for Registration

```python
class Animal(ABC):
    _registry = {}
    
    def __init_subclass__(cls, animal_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if animal_type:
            cls._registry[animal_type] = cls
    
    @classmethod
    def create(cls, animal_type: str):
        return cls._registry[animal_type]()

class Dog(Animal, animal_type="dog"):
    pass

class Cat(Animal, animal_type="cat"):
    pass

# Usage
dog = Animal.create("dog")
```

### 2. Using Dataclasses with Factory

```python
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

class ProductFactory:
    @staticmethod
    def create_product(name: str, price: float) -> Product:
        return Product(name=name, price=price)
```

### 3. Factory with Type Hints

```python
from typing import Dict, Type, TypeVar

T = TypeVar('T', bound=Product)

class Factory:
    _creators: Dict[str, Type[T]] = {}
    
    @classmethod
    def create(cls, product_type: str) -> T:
        return cls._creators[product_type]()
```

---

## Key Takeaways

- **Purpose**: Encapsulate object creation logic
- **Variations**: Simple Factory, Factory Method, Abstract Factory
- **Use when**: Object creation is complex or needs to be decoupled
- **Benefits**: Loose coupling, extensibility, centralized control
- **Trade-off**: Adds abstraction layer
- **Python**: Leverage ABC, type hints, and dynamic registration
- **Best practice**: Use appropriate variation for your needs

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python ABC Module](https://docs.python.org/3/library/abc.html)
- [Factory Method Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/factory-method)

