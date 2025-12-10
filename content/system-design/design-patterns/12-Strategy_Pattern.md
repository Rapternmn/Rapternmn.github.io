+++
title = "Strategy Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 12
description = "Comprehensive guide to Strategy Pattern: Defining a family of algorithms and making them interchangeable, with Python implementations using functions and classes, use cases, and best practices."
+++

---

## Introduction

The **Strategy Pattern** is a behavioral design pattern that defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it.

### Intent

- Define a family of algorithms
- Encapsulate each algorithm
- Make algorithms interchangeable
- Let algorithm vary independently from clients

---

## Problem

Sometimes you need different ways to perform the same operation:
- Different sorting algorithms
- Different payment methods
- Different compression algorithms
- Different validation rules

### Example Problem

```python
class Order:
    def calculate_total(self, items: list, discount_type: str):
        total = sum(item['price'] for item in items)
        
        if discount_type == "percentage":
            return total * 0.9  # 10% off
        elif discount_type == "fixed":
            return total - 10
        elif discount_type == "buy_one_get_one":
            return total / 2
        # Problem: Adding new discount types requires modifying this class!
```

---

## Solution

The Strategy Pattern solves this by:
1. Creating a strategy interface for algorithms
2. Implementing concrete strategies
3. Using composition to inject strategy
4. Allowing runtime strategy selection

---

## Structure

```
┌──────────────┐
│   Context    │
├──────────────┤
│ - strategy   │
├──────────────┤
│ + set_strategy()│
│ + execute()  │
└──────┬───────┘
       │
       │ uses
       ▼
┌──────────────┐
│   Strategy   │
│  (interface) │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ Concrete     │
│ Strategies   │
└──────────────┘
```

**Participants**:
- **Strategy**: Interface for algorithms
- **ConcreteStrategy**: Implements specific algorithm
- **Context**: Uses strategy to perform operation

---

## Implementation

### Basic Strategy Pattern

```python
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, total: float) -> float:
        pass

class PercentageDiscount(DiscountStrategy):
    def __init__(self, percentage: float):
        self.percentage = percentage
    
    def calculate(self, total: float) -> float:
        return total * (1 - self.percentage / 100)

class FixedDiscount(DiscountStrategy):
    def __init__(self, amount: float):
        self.amount = amount
    
    def calculate(self, total: float) -> float:
        return max(0, total - self.amount)

class BuyOneGetOneDiscount(DiscountStrategy):
    def calculate(self, total: float) -> float:
        return total / 2

class Order:
    def __init__(self, discount_strategy: DiscountStrategy = None):
        self.discount_strategy = discount_strategy
        self.items = []
    
    def add_item(self, item: dict):
        self.items.append(item)
    
    def set_discount_strategy(self, strategy: DiscountStrategy):
        self.discount_strategy = strategy
    
    def calculate_total(self) -> float:
        total = sum(item['price'] for item in self.items)
        if self.discount_strategy:
            total = self.discount_strategy.calculate(total)
        return total

# Usage
order = Order()
order.add_item({"name": "Laptop", "price": 1000})
order.add_item({"name": "Mouse", "price": 20})

# Use different strategies
order.set_discount_strategy(PercentageDiscount(10))
print(f"Total with 10% discount: ${order.calculate_total()}")  # $918.0

order.set_discount_strategy(FixedDiscount(50))
print(f"Total with $50 discount: ${order.calculate_total()}")  # $970.0
```

### Strategy with Functions (Pythonic)

```python
# Strategies as functions
def percentage_discount(percentage: float):
    def calculate(total: float) -> float:
        return total * (1 - percentage / 100)
    return calculate

def fixed_discount(amount: float):
    def calculate(total: float) -> float:
        return max(0, total - amount)
    return calculate

class Order:
    def __init__(self, discount_func=None):
        self.discount_func = discount_func
        self.items = []
    
    def calculate_total(self) -> float:
        total = sum(item['price'] for item in self.items)
        if self.discount_func:
            total = self.discount_func(total)
        return total

# Usage
order = Order()
order.add_item({"name": "Laptop", "price": 1000})

order.discount_func = percentage_discount(10)
print(order.calculate_total())  # $900.0
```

---

## Real-World Examples

### Example 1: Payment Processing

```python
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        pass

class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str, cvv: str):
        self.card_number = card_number
        self.cvv = cvv
    
    def pay(self, amount: float) -> bool:
        print(f"Processing ${amount} payment via Credit Card ending in {self.card_number[-4:]}")
        return True

class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email
    
    def pay(self, amount: float) -> bool:
        print(f"Processing ${amount} payment via PayPal ({self.email})")
        return True

class CryptocurrencyPayment(PaymentStrategy):
    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address
    
    def pay(self, amount: float) -> bool:
        print(f"Processing ${amount} payment via Cryptocurrency ({self.wallet_address[:10]}...)")
        return True

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    def process_payment(self, amount: float) -> bool:
        return self.strategy.pay(amount)

# Usage
processor = PaymentProcessor(CreditCardPayment("1234567890123456", "123"))
processor.process_payment(100.0)

processor.set_strategy(PayPalPayment("user@example.com"))
processor.process_payment(50.0)
```

### Example 2: Sorting Strategies

```python
from abc import ABC, abstractmethod
from typing import List

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List) -> List:
        pass

class BubbleSort(SortStrategy):
    def sort(self, data: List) -> List:
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

class QuickSort(SortStrategy):
    def sort(self, data: List) -> List:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortStrategy):
    def sort(self, data: List) -> List:
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)
    
    def _merge(self, left: List, right: List) -> List:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def sort(self, data: List) -> List:
        return self.strategy.sort(data)

# Usage
data = [64, 34, 25, 12, 22, 11, 90]

sorter = Sorter(BubbleSort())
print("Bubble Sort:", sorter.sort(data))

sorter.set_strategy(QuickSort())
print("Quick Sort:", sorter.sort(data))

sorter.set_strategy(MergeSort())
print("Merge Sort:", sorter.sort(data))
```

### Example 3: Validation Strategies

```python
from abc import ABC, abstractmethod

class ValidationStrategy(ABC):
    @abstractmethod
    def validate(self, value: str) -> tuple[bool, str]:
        pass

class EmailValidation(ValidationStrategy):
    def validate(self, value: str) -> tuple[bool, str]:
        if "@" in value and "." in value.split("@")[1]:
            return True, "Valid email"
        return False, "Invalid email format"

class PhoneValidation(ValidationStrategy):
    def validate(self, value: str) -> tuple[bool, str]:
        if value.isdigit() and len(value) == 10:
            return True, "Valid phone"
        return False, "Phone must be 10 digits"

class PasswordValidation(ValidationStrategy):
    def validate(self, value: str) -> tuple[bool, str]:
        if len(value) < 8:
            return False, "Password must be at least 8 characters"
        if not any(c.isupper() for c in value):
            return False, "Password must contain uppercase letter"
        if not any(c.islower() for c in value):
            return False, "Password must contain lowercase letter"
        if not any(c.isdigit() for c in value):
            return False, "Password must contain digit"
        return True, "Valid password"

class FormField:
    def __init__(self, name: str, validation_strategy: ValidationStrategy):
        self.name = name
        self.validation_strategy = validation_strategy
        self.value = ""
    
    def set_value(self, value: str):
        self.value = value
    
    def validate(self) -> tuple[bool, str]:
        return self.validation_strategy.validate(self.value)

# Usage
email_field = FormField("email", EmailValidation())
email_field.set_value("user@example.com")
is_valid, message = email_field.validate()
print(f"Email validation: {is_valid}, {message}")

password_field = FormField("password", PasswordValidation())
password_field.set_value("Weak")
is_valid, message = password_field.validate()
print(f"Password validation: {is_valid}, {message}")
```

---

## Use Cases

### When to Use Strategy Pattern

✅ **Multiple Algorithms**: When you have multiple ways to do something
✅ **Runtime Selection**: When algorithm needs to be selected at runtime
✅ **Avoid Conditionals**: When you want to avoid if/else chains
✅ **Extensibility**: When you need to add new algorithms easily
✅ **Algorithm Encapsulation**: When you want to encapsulate algorithms

### When NOT to Use

❌ **Simple Cases**: Overkill for simple, rarely changing algorithms
❌ **Few Algorithms**: If you only have 1-2 algorithms
❌ **Tight Coupling**: When strategies need too much context data
❌ **Performance Critical**: Adds indirection overhead

---

## Pros and Cons

### Advantages

✅ **Flexibility**: Easy to switch algorithms at runtime
✅ **Extensibility**: Easy to add new strategies
✅ **Eliminates Conditionals**: Replaces if/else chains
✅ **Open/Closed Principle**: Open for extension, closed for modification
✅ **Testability**: Each strategy can be tested independently

### Disadvantages

❌ **Complexity**: Adds more classes/interfaces
❌ **Client Awareness**: Clients must know about strategies
❌ **Overhead**: Adds indirection layer
❌ **Communication**: Strategies may need context data

---

## Strategy vs Other Patterns

### Strategy vs State

- **Strategy**: Algorithm selection (what to do)
- **State**: Behavior based on state (how to behave)

### Strategy vs Template Method

- **Strategy**: Different algorithms (composition)
- **Template Method**: Same algorithm, different steps (inheritance)

### Strategy vs Command

- **Strategy**: Algorithm selection
- **Command**: Request encapsulation

---

## Best Practices

### 1. Use Functions for Simple Strategies

```python
# Simple strategies as functions
def strategy_a(data):
    return data * 2

def strategy_b(data):
    return data + 10

# Use directly
result = strategy_a(5)
```

### 2. Use Classes for Complex Strategies

```python
# Complex strategies as classes
class ComplexStrategy:
    def __init__(self, config):
        self.config = config
    
    def execute(self, data):
        # Complex logic
        pass
```

### 3. Strategy Factory

```python
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type: str):
        strategies = {
            "percentage": PercentageDiscount,
            "fixed": FixedDiscount,
            "bogo": BuyOneGetOneDiscount
        }
        return strategies.get(strategy_type)()
```

### 4. Default Strategy

```python
class Context:
    def __init__(self, strategy=None):
        self.strategy = strategy or DefaultStrategy()
```

### 5. Strategy with Context Data

```python
class Strategy(ABC):
    @abstractmethod
    def execute(self, data: dict) -> dict:
        pass

class Context:
    def execute_strategy(self, data: dict) -> dict:
        return self.strategy.execute(data)
```

---

## Python-Specific Considerations

### 1. Functions as Strategies

Python's first-class functions make strategies simple:

```python
strategies = {
    'add': lambda x, y: x + y,
    'multiply': lambda x, y: x * y,
    'subtract': lambda x, y: x - y
}

def calculate(operation, a, b):
    return strategies[operation](a, b)
```

### 2. Using `functools.partial`

```python
from functools import partial

def discount(percentage, total):
    return total * (1 - percentage / 100)

# Create strategies
ten_percent = partial(discount, 10)
twenty_percent = partial(discount, 20)
```

### 3. Strategy with `typing.Protocol`

```python
from typing import Protocol

class Strategy(Protocol):
    def execute(self, data: str) -> str:
        ...

def use_strategy(strategy: Strategy, data: str) -> str:
    return strategy.execute(data)
```

### 4. Strategy Registry

```python
class StrategyRegistry:
    _strategies = {}
    
    @classmethod
    def register(cls, name: str, strategy):
        cls._strategies[name] = strategy
    
    @classmethod
    def get(cls, name: str):
        return cls._strategies.get(name)

# Register strategies
StrategyRegistry.register("strategy1", Strategy1())
StrategyRegistry.register("strategy2", Strategy2())
```

---

## Common Pitfalls

### 1. Over-Engineering

```python
# Bad: Strategy for simple case
class AddStrategy:
    def execute(self, a, b):
        return a + b

# Good: Just use function
def add(a, b):
    return a + b
```

### 2. Strategy Needs Too Much Context

```python
# Bad: Strategy needs too much context
class Strategy:
    def execute(self, context):
        # Needs many context attributes
        return context.a + context.b + context.c

# Better: Pass only needed data
class Strategy:
    def execute(self, a, b, c):
        return a + b + c
```

### 3. Not Using Default Strategy

```python
# Bad: Strategy can be None
if self.strategy:
    result = self.strategy.execute()
else:
    result = default_result

# Good: Always have strategy
self.strategy = strategy or DefaultStrategy()
```

---

## Key Takeaways

- **Purpose**: Define interchangeable algorithms
- **Use when**: Multiple ways to perform same operation
- **Python**: Functions work great for simple strategies
- **Benefits**: Flexibility, extensibility, eliminates conditionals
- **Trade-off**: Adds complexity, client must know strategies
- **Best practice**: Use functions for simple, classes for complex
- **Common use**: Sorting, validation, payment, compression

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python functools.partial](https://docs.python.org/3/library/functools.html#functools.partial)
- [Strategy Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/strategy)

