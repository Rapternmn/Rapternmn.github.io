+++
title = "SOLID Principles"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion with Python examples and best practices."
+++

---

## Introduction

**SOLID** is an acronym for five object-oriented design principles that make software designs more understandable, flexible, and maintainable. These principles were introduced by Robert C. Martin (Uncle Bob) and are fundamental to writing clean, maintainable code.

The five principles are:
1. **S** - Single Responsibility Principle
2. **O** - Open/Closed Principle
3. **L** - Liskov Substitution Principle
4. **I** - Interface Segregation Principle
5. **D** - Dependency Inversion Principle

Understanding and applying these principles helps create code that is easier to test, extend, and maintain.

---

## 1. Single Responsibility Principle (SRP)

**"A class should have only one reason to change."**

A class should have only one job or responsibility. If a class has multiple responsibilities, it becomes harder to maintain and modify.

### Violation Example

```python
class User:
    """Violates SRP - handles multiple responsibilities"""
    
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
    
    def get_user_info(self) -> str:
        """User management responsibility"""
        return f"{self.name} - {self.email}"
    
    def save_to_database(self):
        """Database responsibility"""
        # Database save logic
        print(f"Saving {self.name} to database...")
    
    def send_email(self, message: str):
        """Email responsibility"""
        # Email sending logic
        print(f"Sending email to {self.email}: {message}")
```

**Problem**: The `User` class has three responsibilities:
- User data management
- Database operations
- Email operations

### Correct Implementation

```python
class User:
    """Handles only user data"""
    
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
    
    def get_user_info(self) -> str:
        return f"{self.name} - {self.email}"

class UserRepository:
    """Handles only database operations"""
    
    def save(self, user: User):
        print(f"Saving {user.name} to database...")
    
    def find_by_email(self, email: str) -> User:
        # Database query logic
        pass

class EmailService:
    """Handles only email operations"""
    
    def send_email(self, user: User, message: str):
        print(f"Sending email to {user.email}: {message}")
```

**Benefits**:
- Each class has a single, well-defined responsibility
- Changes to database logic don't affect email logic
- Easier to test each component independently

### Key Points

- **Purpose**: Ensure each class has one reason to change
- **When to use**: Always - this is a fundamental principle
- **Benefits**: Better maintainability, easier testing, clearer code
- **Trade-off**: May create more classes, but improves organization

---

## 2. Open/Closed Principle (OCP)

**"Software entities should be open for extension but closed for modification."**

You should be able to extend a class's behavior without modifying its source code.

### Violation Example

```python
class AreaCalculator:
    """Violates OCP - must modify for each new shape"""
    
    def calculate_area(self, shape):
        if isinstance(shape, Rectangle):
            return shape.width * shape.height
        elif isinstance(shape, Circle):
            return 3.14 * shape.radius ** 2
        elif isinstance(shape, Triangle):
            return 0.5 * shape.base * shape.height
        # Must modify this method for every new shape!
```

**Problem**: Adding a new shape requires modifying the `AreaCalculator` class.

### Correct Implementation

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for all shapes"""
    
    @abstractmethod
    def calculate_area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def calculate_area(self) -> float:
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def calculate_area(self) -> float:
        return 3.14 * self.radius ** 2

class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height
    
    def calculate_area(self) -> float:
        return 0.5 * self.base * self.height

class AreaCalculator:
    """Open for extension, closed for modification"""
    
    def calculate_total_area(self, shapes: list[Shape]) -> float:
        total = 0
        for shape in shapes:
            total += shape.calculate_area()
        return total

# Adding a new shape doesn't require modifying AreaCalculator
class Pentagon(Shape):
    def __init__(self, side: float):
        self.side = side
    
    def calculate_area(self) -> float:
        # Pentagon area calculation
        return 1.72 * self.side ** 2
```

**Benefits**:
- New shapes can be added without modifying existing code
- Reduces risk of introducing bugs in existing functionality
- Follows the "open for extension" principle

### Key Points

- **Purpose**: Allow extension without modification
- **When to use**: When you anticipate future changes
- **Benefits**: Reduced risk, better maintainability
- **Trade-off**: Requires upfront design with abstractions

---

## 3. Liskov Substitution Principle (LSP)

**"Objects of a superclass should be replaceable with objects of its subclasses without breaking the application."**

Subtypes must be substitutable for their base types. Derived classes should not weaken the base class contract.

### Violation Example

```python
class Bird:
    def fly(self):
        print("Flying...")

class Sparrow(Bird):
    def fly(self):
        print("Sparrow is flying...")

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins cannot fly!")

def make_bird_fly(bird: Bird):
    bird.fly()  # This will fail for Penguin!

# Usage
sparrow = Sparrow()
penguin = Penguin()

make_bird_fly(sparrow)  # Works
make_bird_fly(penguin)   # Raises exception - violates LSP!
```

**Problem**: `Penguin` cannot be substituted for `Bird` because it cannot fly.

### Correct Implementation

```python
from abc import ABC, abstractmethod

class Bird(ABC):
    """Base class for all birds"""
    
    def eat(self):
        print("Eating...")
    
    def sleep(self):
        print("Sleeping...")

class FlyingBird(Bird):
    """Birds that can fly"""
    
    @abstractmethod
    def fly(self):
        pass

class NonFlyingBird(Bird):
    """Birds that cannot fly"""
    
    def walk(self):
        print("Walking...")

class Sparrow(FlyingBird):
    def fly(self):
        print("Sparrow is flying...")

class Penguin(NonFlyingBird):
    def swim(self):
        print("Penguin is swimming...")

def make_bird_fly(bird: FlyingBird):
    bird.fly()

# Usage
sparrow = Sparrow()
penguin = Penguin()

make_bird_fly(sparrow)  # Works - Sparrow is a FlyingBird
# make_bird_fly(penguin)  # Type error - Penguin is not a FlyingBird
```

**Benefits**:
- Subtypes can be safely substituted
- Clearer inheritance hierarchy
- Prevents unexpected behavior

### Key Points

- **Purpose**: Ensure substitutability of subtypes
- **When to use**: Always when using inheritance
- **Benefits**: Prevents bugs, clearer contracts
- **Trade-off**: May require more abstract base classes

---

## 4. Interface Segregation Principle (ISP)

**"Clients should not be forced to depend on interfaces they do not use."**

Many client-specific interfaces are better than one general-purpose interface.

### Violation Example

```python
from abc import ABC, abstractmethod

class Worker(ABC):
    """Violates ISP - forces all workers to implement all methods"""
    
    @abstractmethod
    def work(self):
        pass
    
    @abstractmethod
    def eat(self):
        pass
    
    @abstractmethod
    def sleep(self):
        pass

class Human(Worker):
    def work(self):
        print("Human working...")
    
    def eat(self):
        print("Human eating...")
    
    def sleep(self):
        print("Human sleeping...")

class Robot(Worker):
    def work(self):
        print("Robot working...")
    
    def eat(self):
        # Robots don't eat!
        raise NotImplementedError("Robots don't eat!")
    
    def sleep(self):
        # Robots don't sleep!
        raise NotImplementedError("Robots don't sleep!")
```

**Problem**: `Robot` is forced to implement methods it doesn't need.

### Correct Implementation

```python
from abc import ABC, abstractmethod

class Workable(ABC):
    """Interface for entities that can work"""
    
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    """Interface for entities that can eat"""
    
    @abstractmethod
    def eat(self):
        pass

class Sleepable(ABC):
    """Interface for entities that can sleep"""
    
    @abstractmethod
    def sleep(self):
        pass

class Human(Workable, Eatable, Sleepable):
    """Human implements all interfaces"""
    
    def work(self):
        print("Human working...")
    
    def eat(self):
        print("Human eating...")
    
    def sleep(self):
        print("Human sleeping...")

class Robot(Workable):
    """Robot only implements Workable interface"""
    
    def work(self):
        print("Robot working...")

# Usage
def manage_worker(worker: Workable):
    worker.work()

def manage_human(human: Eatable):
    human.eat()

human = Human()
robot = Robot()

manage_worker(human)  # Works
manage_worker(robot)  # Works
manage_human(human)   # Works
# manage_human(robot)  # Type error - Robot doesn't implement Eatable
```

**Benefits**:
- Clients only depend on methods they use
- No forced implementation of unnecessary methods
- More flexible and maintainable

### Key Points

- **Purpose**: Prevent clients from depending on unused methods
- **When to use**: When interfaces become too large
- **Benefits**: Better separation, no unnecessary dependencies
- **Trade-off**: More interfaces to manage

---

## 5. Dependency Inversion Principle (DIP)

**"High-level modules should not depend on low-level modules. Both should depend on abstractions."**

Depend on abstractions (interfaces), not concrete implementations.

### Violation Example

```python
class MySQLDatabase:
    """Low-level module"""
    
    def connect(self):
        print("Connecting to MySQL...")
    
    def query(self, sql: str):
        print(f"Executing MySQL query: {sql}")

class UserService:
    """High-level module - depends on concrete MySQLDatabase"""
    
    def __init__(self):
        self.db = MySQLDatabase()  # Direct dependency on concrete class
    
    def get_user(self, user_id: int):
        self.db.connect()
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

# Problem: If we want to switch to PostgreSQL, we must modify UserService
```

**Problem**: `UserService` directly depends on `MySQLDatabase`, making it hard to switch databases.

### Correct Implementation

```python
from abc import ABC, abstractmethod

class Database(ABC):
    """Abstraction for database operations"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def query(self, sql: str):
        pass

class MySQLDatabase(Database):
    """Concrete implementation for MySQL"""
    
    def connect(self):
        print("Connecting to MySQL...")
    
    def query(self, sql: str):
        print(f"Executing MySQL query: {sql}")

class PostgreSQLDatabase(Database):
    """Concrete implementation for PostgreSQL"""
    
    def connect(self):
        print("Connecting to PostgreSQL...")
    
    def query(self, sql: str):
        print(f"Executing PostgreSQL query: {sql}")

class UserService:
    """High-level module - depends on Database abstraction"""
    
    def __init__(self, database: Database):  # Depends on abstraction
        self.db = database
    
    def get_user(self, user_id: int):
        self.db.connect()
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

# Usage - can easily switch between databases
mysql_db = MySQLDatabase()
postgres_db = PostgreSQLDatabase()

user_service_mysql = UserService(mysql_db)
user_service_postgres = UserService(postgres_db)
```

**Benefits**:
- Easy to swap implementations
- High-level modules don't depend on low-level details
- Better testability (can use mock databases)

### Key Points

- **Purpose**: Decouple high-level from low-level modules
- **When to use**: When you need flexibility in implementations
- **Benefits**: Better testability, easier to change implementations
- **Trade-off**: Requires more abstractions

---

## Complete Example: Applying All SOLID Principles

Let's see how all SOLID principles work together in a practical example:

```python
from abc import ABC, abstractmethod
from typing import List

# ISP: Separate interfaces
class Readable(ABC):
    @abstractmethod
    def read(self) -> str:
        pass

class Writable(ABC):
    @abstractmethod
    def write(self, data: str):
        pass

# DIP: Depend on abstractions
class FileStorage(ABC):
    @abstractmethod
    def save(self, filename: str, data: str):
        pass
    
    @abstractmethod
    def load(self, filename: str) -> str:
        pass

# Concrete implementations
class LocalFileStorage(FileStorage):
    def save(self, filename: str, data: str):
        print(f"Saving {data} to {filename}")
    
    def load(self, filename: str) -> str:
        print(f"Loading from {filename}")
        return "file content"

class CloudFileStorage(FileStorage):
    def save(self, filename: str, data: str):
        print(f"Uploading {data} to cloud: {filename}")
    
    def load(self, filename: str) -> str:
        print(f"Downloading from cloud: {filename}")
        return "cloud content"

# SRP: Single responsibility
class FileReader(Readable):
    """Only responsible for reading"""
    
    def __init__(self, storage: FileStorage):  # DIP
        self.storage = storage
    
    def read(self) -> str:
        return self.storage.load("file.txt")

class FileWriter(Writable):
    """Only responsible for writing"""
    
    def __init__(self, storage: FileStorage):  # DIP
        self.storage = storage
    
    def write(self, data: str):
        self.storage.save("file.txt", data)

# OCP: Open for extension
class FileProcessor:
    """Can be extended without modification"""
    
    def __init__(self, reader: Readable, writer: Writable):
        self.reader = reader
        self.writer = writer
    
    def process(self):
        data = self.reader.read()
        processed = data.upper()
        self.writer.write(processed)

# Usage
local_storage = LocalFileStorage()
cloud_storage = CloudFileStorage()

reader = FileReader(local_storage)  # Can switch to cloud_storage
writer = FileWriter(local_storage)  # Can switch to cloud_storage

processor = FileProcessor(reader, writer)
processor.process()
```

---

## Best Practices

1. **Start with SRP**: Ensure each class has one responsibility
2. **Use abstractions**: Apply DIP to depend on interfaces, not concrete classes
3. **Design for extension**: Apply OCP when you anticipate changes
4. **Respect inheritance**: Follow LSP when using inheritance
5. **Keep interfaces focused**: Apply ISP to avoid bloated interfaces
6. **Combine principles**: SOLID principles work best together
7. **Don't over-engineer**: Apply principles where they add value

---

## Common Violations and How to Fix Them

### Violation: God Class (SRP)
**Problem**: One class doing too much
**Solution**: Split into multiple classes with single responsibilities

### Violation: Switch Statements (OCP)
**Problem**: Using if/elif chains for type checking
**Solution**: Use polymorphism and abstractions

### Violation: Square-Rectangle Problem (LSP)
**Problem**: Square extending Rectangle but changing behavior
**Solution**: Use composition or separate inheritance hierarchy

### Violation: Fat Interface (ISP)
**Problem**: Interface with too many methods
**Solution**: Split into smaller, focused interfaces

### Violation: Direct Dependencies (DIP)
**Problem**: High-level modules depending on concrete classes
**Solution**: Introduce abstractions and use dependency injection

---

## Key Takeaways

- **SRP**: One class, one responsibility
- **OCP**: Extend without modifying
- **LSP**: Subtypes must be substitutable
- **ISP**: Clients shouldn't depend on unused methods
- **DIP**: Depend on abstractions, not concretions
- **Together**: SOLID principles create maintainable, flexible code
- **Balance**: Don't over-apply; use where it adds value
- **Practice**: Apply these principles in code reviews and refactoring

---

## References

- [SOLID Principles - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2020/10/18/SolidPrinciples.html)
- [Python ABC Module](https://docs.python.org/3/library/abc.html)
- [Clean Code by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

