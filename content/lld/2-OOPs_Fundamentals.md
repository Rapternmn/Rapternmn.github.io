+++
title = "OOPs Fundamentals"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Comprehensive guide to Object-Oriented Programming fundamentals: Classes, Objects, Encapsulation, Inheritance, Polymorphism, and Abstraction with Python examples and best practices."
+++

---

## Introduction

Object-Oriented Programming (OOPs) is a programming paradigm that organizes code into objects, which contain both data (attributes) and behavior (methods). OOPs is fundamental to Low Level Design as it provides the building blocks for creating well-structured, maintainable, and scalable software systems.

This guide covers the four pillars of OOPs:
1. **Encapsulation** - Data hiding and access control
2. **Inheritance** - Code reuse and hierarchical relationships
3. **Polymorphism** - Multiple forms of the same interface
4. **Abstraction** - Hiding implementation details

---

## Classes and Objects

### What is a Class?

A **class** is a blueprint or template for creating objects. It defines:
- **Attributes** (data members): Variables that hold data
- **Methods** (member functions): Functions that define behavior

### What is an Object?

An **object** is an instance of a class. It represents a specific entity with its own state (attribute values) and behavior (methods).

### Example: Basic Class and Object

```python
class Car:
    """A simple Car class to demonstrate classes and objects"""
    
    def __init__(self, brand, model, year):
        """Constructor - initializes object attributes"""
        self.brand = brand
        self.model = model
        self.year = year
        self.speed = 0
    
    def start(self):
        """Method to start the car"""
        return f"{self.brand} {self.model} started!"
    
    def accelerate(self, increment):
        """Method to increase speed"""
        self.speed += increment
        return f"Speed increased to {self.speed} km/h"
    
    def get_info(self):
        """Method to get car information"""
        return f"{self.year} {self.brand} {self.model}"

# Creating objects (instances)
car1 = Car("Toyota", "Camry", 2023)
car2 = Car("Honda", "Civic", 2022)

print(car1.start())  # Output: Toyota Camry started!
print(car2.get_info())  # Output: 2022 Honda Civic
print(car1.accelerate(20))  # Output: Speed increased to 20 km/h
```

### Key Points:
- `__init__` is the constructor method that initializes object attributes
- `self` refers to the instance of the class
- Each object has its own copy of instance attributes
- Methods are called on objects using dot notation

---

## Encapsulation

**Encapsulation** is the bundling of data and methods that operate on that data within a single unit (class), while restricting access to some components. It's about data hiding and access control.

### Access Modifiers in Python

Python doesn't have strict access modifiers like Java or C++, but uses naming conventions:
- **Public**: No prefix (default) - `attribute_name`
- **Protected**: Single underscore prefix - `_attribute_name` (convention, not enforced)
- **Private**: Double underscore prefix - `__attribute_name` (name mangling)

### Example: Encapsulation with Access Control

```python
class BankAccount:
    """Demonstrates encapsulation with access control"""
    
    def __init__(self, account_number, balance=0):
        # Public attribute
        self.account_number = account_number
        
        # Protected attribute (convention)
        self._owner_name = None
        
        # Private attribute (name mangling)
        self.__balance = balance
        self.__transaction_history = []
    
    def deposit(self, amount):
        """Public method to deposit money"""
        if amount > 0:
            self.__balance += amount
            self.__add_transaction("Deposit", amount)
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        """Public method to withdraw money"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__add_transaction("Withdrawal", amount)
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        """Public method to get balance (read-only access)"""
        return self.__balance
    
    def __add_transaction(self, transaction_type, amount):
        """Private method - internal use only"""
        self.__transaction_history.append({
            'type': transaction_type,
            'amount': amount,
            'balance': self.__balance
        })
    
    def get_transaction_history(self):
        """Public method to view transaction history"""
        return self.__transaction_history.copy()  # Return copy to prevent modification

# Usage
account = BankAccount("ACC123", 1000)
print(account.deposit(500))  # Output: Deposited $500. New balance: $1500
print(account.withdraw(200))  # Output: Withdrew $200. New balance: $1300
print(account.get_balance())  # Output: 1300

# Private attribute access (not recommended, but possible with name mangling)
# print(account._BankAccount__balance)  # Works but violates encapsulation

# Protected attribute access
account._owner_name = "John Doe"  # Convention suggests this is for internal use
```

### Key Points:
- Encapsulation protects data integrity by controlling access
- Private attributes prevent direct modification from outside the class
- Public methods provide controlled access to private data
- Python's name mangling (`__attribute`) makes attributes harder to access but not impossible

---

## Inheritance

**Inheritance** allows a class (child/derived class) to inherit attributes and methods from another class (parent/base class). It promotes code reuse and establishes an "is-a" relationship.

### Types of Inheritance

1. **Single Inheritance**: One child class inherits from one parent class
2. **Multiple Inheritance**: One child class inherits from multiple parent classes
3. **Multilevel Inheritance**: Child class becomes parent for another class
4. **Hierarchical Inheritance**: Multiple child classes inherit from one parent

### Example: Single Inheritance

```python
class Vehicle:
    """Base class (parent)"""
    
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return f"{self.brand} {self.model} started"
    
    def stop(self):
        self.is_running = False
        return f"{self.brand} {self.model} stopped"
    
    def get_info(self):
        return f"{self.year} {self.brand} {self.model}"

class Car(Vehicle):
    """Derived class (child) - inherits from Vehicle"""
    
    def __init__(self, brand, model, year, num_doors):
        # Call parent constructor
        super().__init__(brand, model, year)
        self.num_doors = num_doors
    
    def honk(self):
        """Child-specific method"""
        return "Beep beep!"
    
    def get_info(self):
        """Method overriding - child's version is used"""
        base_info = super().get_info()
        return f"{base_info} with {self.num_doors} doors"

class Motorcycle(Vehicle):
    """Another derived class"""
    
    def __init__(self, brand, model, year, engine_cc):
        super().__init__(brand, model, year)
        self.engine_cc = engine_cc
    
    def wheelie(self):
        """Child-specific method"""
        return "Doing a wheelie!"

# Usage
car = Car("Toyota", "Camry", 2023, 4)
motorcycle = Motorcycle("Yamaha", "R1", 2023, 1000)

print(car.start())  # Inherited method
print(car.honk())  # Child-specific method
print(car.get_info())  # Overridden method

print(motorcycle.start())  # Inherited method
print(motorcycle.wheelie())  # Child-specific method
```

### Example: Multiple Inheritance

```python
class Flyable:
    """Mixin class for flying capability"""
    
    def fly(self):
        return "Flying high!"

class Swimmable:
    """Mixin class for swimming capability"""
    
    def swim(self):
        return "Swimming deep!"

class Duck(Flyable, Swimmable):
    """Multiple inheritance - Duck can both fly and swim"""
    
    def __init__(self, name):
        self.name = name
    
    def quack(self):
        return f"{self.name} says Quack!"

# Usage
duck = Duck("Donald")
print(duck.fly())  # From Flyable
print(duck.swim())  # From Swimmable
print(duck.quack())  # From Duck itself
```

### Key Points:
- `super()` is used to call parent class methods
- Child classes inherit all public and protected attributes/methods
- Method overriding allows child classes to provide their own implementation
- Multiple inheritance can lead to complexity (Method Resolution Order - MRO)

---

## Polymorphism

**Polymorphism** means "many forms". It allows objects of different classes to be treated as objects of a common base class. There are two types:

1. **Compile-time Polymorphism (Method Overloading)**: Multiple methods with the same name but different parameters
2. **Runtime Polymorphism (Method Overriding)**: Child class provides its own implementation of a parent method

### Method Overriding

```python
class Animal:
    """Base class"""
    
    def make_sound(self):
        return "Some generic animal sound"

class Dog(Animal):
    """Child class overriding make_sound"""
    
    def make_sound(self):
        return "Woof! Woof!"

class Cat(Animal):
    """Child class overriding make_sound"""
    
    def make_sound(self):
        return "Meow! Meow!"

class Duck(Animal):
    """Child class overriding make_sound"""
    
    def make_sound(self):
        return "Quack! Quack!"

def animal_sound(animal):
    """Polymorphic function - works with any Animal"""
    return animal.make_sound()

# Usage
animals = [Dog(), Cat(), Duck()]

for animal in animals:
    print(animal_sound(animal))
# Output:
# Woof! Woof!
# Meow! Meow!
# Quack! Quack!
```

### Method Overloading (Python Style)

Python doesn't support traditional method overloading, but we can achieve similar behavior using default parameters or `*args`/`**kwargs`:

```python
class Calculator:
    """Demonstrates method overloading-like behavior"""
    
    def add(self, a, b=None, c=None):
        """Can add 2 or 3 numbers"""
        if c is not None:
            return a + b + c
        elif b is not None:
            return a + b
        else:
            return a
    
    def multiply(self, *args):
        """Can multiply any number of arguments"""
        result = 1
        for num in args:
            result *= num
        return result

# Usage
calc = Calculator()
print(calc.add(5, 10))  # Output: 15
print(calc.add(5, 10, 15))  # Output: 30
print(calc.multiply(2, 3, 4))  # Output: 24
```

### Duck Typing (Python's Polymorphism)

Python uses "duck typing" - if it walks like a duck and quacks like a duck, it's a duck:

```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Robot:
    def speak(self):
        return "Beep boop!"

def make_it_speak(obj):
    """Works with any object that has a speak() method"""
    return obj.speak()

# Usage - all work because they have speak() method
print(make_it_speak(Dog()))  # Output: Woof!
print(make_it_speak(Cat()))  # Output: Meow!
print(make_it_speak(Robot()))  # Output: Beep boop!
```

### Key Points:
- Polymorphism allows writing code that works with multiple types
- Method overriding is runtime polymorphism
- Python's duck typing provides flexibility
- Abstract base classes can enforce polymorphism contracts

---

## Abstraction

**Abstraction** is the process of hiding complex implementation details and showing only essential features. It focuses on "what" rather than "how".

### Abstract Classes in Python

Python provides the `abc` module for creating abstract classes:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class"""
    
    @abstractmethod
    def area(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Abstract method - must be implemented by child classes"""
        pass
    
    def describe(self):
        """Concrete method - can be used by all child classes"""
        return f"Shape with area: {self.area()} and perimeter: {self.perimeter()}"

class Rectangle(Shape):
    """Concrete implementation of Shape"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """Implementation of abstract method"""
        return self.width * self.height
    
    def perimeter(self):
        """Implementation of abstract method"""
        return 2 * (self.width + self.height)

class Circle(Shape):
    """Concrete implementation of Shape"""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        """Implementation of abstract method"""
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        """Implementation of abstract method"""
        return 2 * 3.14159 * self.radius

# Usage
rectangle = Rectangle(5, 3)
circle = Circle(4)

print(rectangle.describe())  # Output: Shape with area: 15 and perimeter: 16
print(circle.describe())  # Output: Shape with area: 50.26544 and perimeter: 25.13272

# Cannot instantiate abstract class
# shape = Shape()  # This would raise TypeError
```

### Interface-like Behavior

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    """Abstract interface for payment processing"""
    
    @abstractmethod
    def process_payment(self, amount):
        """Process a payment - implementation varies"""
        pass
    
    @abstractmethod
    def refund(self, transaction_id):
        """Refund a payment - implementation varies"""
        pass

class CreditCardProcessor(PaymentProcessor):
    """Concrete implementation"""
    
    def process_payment(self, amount):
        return f"Processing ${amount} via Credit Card"
    
    def refund(self, transaction_id):
        return f"Refunding transaction {transaction_id} via Credit Card"

class PayPalProcessor(PaymentProcessor):
    """Concrete implementation"""
    
    def process_payment(self, amount):
        return f"Processing ${amount} via PayPal"
    
    def refund(self, transaction_id):
        return f"Refunding transaction {transaction_id} via PayPal"

# Usage
processors = [CreditCardProcessor(), PayPalProcessor()]
for processor in processors:
    print(processor.process_payment(100))
```

### Key Points:
- Abstraction hides implementation details
- Abstract classes cannot be instantiated
- Abstract methods must be implemented by child classes
- Provides a contract that all implementations must follow

---

## Python-Specific OOPs Features

### 1. Class Methods and Static Methods

```python
class Employee:
    """Demonstrates class methods and static methods"""
    
    company = "Tech Corp"  # Class variable
    employee_count = 0
    
    def __init__(self, name, salary):
        self.name = name  # Instance variable
        self.salary = salary
        Employee.employee_count += 1
    
    @classmethod
    def from_string(cls, emp_string):
        """Class method - alternative constructor"""
        name, salary = emp_string.split('-')
        return cls(name, int(salary))
    
    @classmethod
    def get_employee_count(cls):
        """Class method - works with class, not instance"""
        return cls.employee_count
    
    @staticmethod
    def is_workday(day):
        """Static method - doesn't need class or instance"""
        return day.weekday() < 5  # Monday to Friday
    
    def get_info(self):
        """Instance method - works with instance"""
        return f"{self.name} works at {self.company} and earns ${self.salary}"

# Usage
emp1 = Employee("Alice", 50000)
emp2 = Employee.from_string("Bob-60000")  # Using class method

print(emp1.get_info())
print(Employee.get_employee_count())  # Output: 2

from datetime import date
print(Employee.is_workday(date.today()))  # Static method
```

### 2. Property Decorators

```python
class Temperature:
    """Demonstrates property decorators for getters and setters"""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter for celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Computed property"""
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Setter that converts fahrenheit to celsius"""
        self._celsius = (value - 32) * 5/9

# Usage
temp = Temperature(25)
print(temp.celsius)  # Output: 25
print(temp.fahrenheit)  # Output: 77.0

temp.fahrenheit = 100
print(temp.celsius)  # Output: 37.77777777777778
```

### 3. Magic Methods (Dunder Methods)

```python
class Book:
    """Demonstrates magic methods"""
    
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        """String representation for users"""
        return f"{self.title} by {self.author}"
    
    def __repr__(self):
        """String representation for developers"""
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __len__(self):
        """Length of book (pages)"""
        return self.pages
    
    def __eq__(self, other):
        """Equality comparison"""
        if isinstance(other, Book):
            return self.title == other.title and self.author == other.author
        return False
    
    def __lt__(self, other):
        """Less than comparison (by pages)"""
        if isinstance(other, Book):
            return self.pages < other.pages
        return NotImplemented

# Usage
book1 = Book("Python Guide", "John Doe", 300)
book2 = Book("Python Guide", "John Doe", 250)

print(str(book1))  # Output: Python Guide by John Doe
print(len(book1))  # Output: 300
print(book1 == book2)  # Output: True
print(book1 < book2)  # Output: False
```

### 4. Multiple Inheritance and MRO

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

# Method Resolution Order (MRO)
print(D.__mro__)  # Shows the order: D -> B -> C -> A -> object
d = D()
print(d.method())  # Output: B (first in MRO)
```

---

## Best Practices

1. **Use meaningful class and method names**: Follow PEP 8 naming conventions
2. **Keep classes focused**: Single Responsibility Principle
3. **Use composition over inheritance when appropriate**: "Has-a" vs "Is-a"
4. **Document your classes**: Use docstrings
5. **Use properties for computed attributes**: Instead of getter/setter methods
6. **Prefer composition**: Favor composition over inheritance for flexibility
7. **Use abstract classes**: When you want to enforce a contract

---

## Key Takeaways

- **Classes** are blueprints; **Objects** are instances
- **Encapsulation** protects data through access control
- **Inheritance** promotes code reuse and establishes relationships
- **Polymorphism** allows different classes to be used interchangeably
- **Abstraction** hides complexity and shows only essential features
- Python's OOPs features (properties, magic methods, MRO) provide flexibility
- Understanding OOPs is crucial for applying design patterns and SOLID principles

---

## References

- [Python Classes Documentation](https://docs.python.org/3/tutorial/classes.html)
- [PEP 8 - Style Guide](https://peps.python.org/pep-0008/)
- [abc Module Documentation](https://docs.python.org/3/library/abc.html)

