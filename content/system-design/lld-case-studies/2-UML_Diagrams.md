+++
title = "UML Diagrams"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Comprehensive guide to UML diagrams: Class, Sequence, Activity, State, Component, and Use Case diagrams with examples and best practices for Low Level Design."
+++

---

## Introduction

**Unified Modeling Language (UML)** is a standardized visual language for modeling software systems. UML diagrams are essential tools in Low Level Design for:

- **Communication**: Visual representation of system design
- **Documentation**: Clear documentation of system architecture
- **Analysis**: Understanding relationships and interactions
- **Design**: Planning system structure before implementation

UML provides multiple diagram types, each serving different purposes in the software development lifecycle. This guide covers the most important diagrams for Low Level Design.

---

## 1. Class Diagrams

**Class diagrams** are the most fundamental UML diagrams in LLD. They show the static structure of a system by depicting classes, their attributes, methods, and relationships.

### Components of a Class Diagram

1. **Class**: Represented as a rectangle with three compartments:
   - **Top**: Class name
   - **Middle**: Attributes (properties)
   - **Bottom**: Methods (operations)

2. **Relationships**:
   - **Association**: Simple relationship between classes
   - **Inheritance**: "Is-a" relationship (extends)
   - **Composition**: Strong "has-a" relationship (part cannot exist without whole)
   - **Aggregation**: Weak "has-a" relationship (part can exist independently)
   - **Dependency**: One class uses another

### Visibility Modifiers

- `+` : Public
- `-` : Private
- `#` : Protected
- `~` : Package/Internal

### Example: Library Management System

```
┌─────────────────────────────────┐
│         Book                    │
├─────────────────────────────────┤
│ - isbn: str                     │
│ - title: str                    │
│ - author: str                   │
│ - is_available: bool            │
├─────────────────────────────────┤
│ + borrow()                      │
│ + return_book()                 │
│ + get_info(): str               │
└─────────────────────────────────┘
      ▲                    │
      │                    │
      │ (inheritance)      │ (association)
      │                    │
┌─────────────────────────────────┐ │
│      ReferenceBook              │ │
├─────────────────────────────────┤ │
│ - edition: int                  │ │
│ - is_reference_only: bool       │ │
├─────────────────────────────────┤ │
│ + can_borrow(): bool            │ │
└─────────────────────────────────┘ │
                                    │
┌─────────────────────────────────┐ │
│         Member                  │ │
├─────────────────────────────────┤ │
│ - member_id: str                │ │
│ - name: str                     │ │
│ - email: str                    │ │
├─────────────────────────────────┤ │
│ + borrow_book(book: Book)       │ │
│ + return_book(book: Book)       │ │
└─────────────────────────────────┘ │
      │                             │
      │ (association)               │
      │                             │
      └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │      Library                    │
        ├─────────────────────────────────┤
        │ - books: List[Book]             │
        │ - members: List[Member]         │
        ├─────────────────────────────────┤
        │ + add_book(book: Book)          │
        │ + register_member(member: Member)│
        │ + search_book(title: str): Book │
        └─────────────────────────────────┘
```

### Python Implementation

```python
from typing import List
from abc import ABC, abstractmethod

class Book(ABC):
    """Base class for all books"""
    
    def __init__(self, isbn: str, title: str, author: str):
        self._isbn = isbn
        self._title = title
        self._author = author
        self._is_available = True
    
    @property
    def isbn(self) -> str:
        return self._isbn
    
    @property
    def title(self) -> str:
        return self._title
    
    def borrow(self) -> bool:
        if self._is_available:
            self._is_available = False
            return True
        return False
    
    def return_book(self):
        self._is_available = True
    
    def get_info(self) -> str:
        return f"{self._title} by {self._author}"

class ReferenceBook(Book):
    """Reference books that cannot be borrowed"""
    
    def __init__(self, isbn: str, title: str, author: str, edition: int):
        super().__init__(isbn, title, author)
        self._edition = edition
        self._is_reference_only = True
    
    def can_borrow(self) -> bool:
        return False

class Member:
    """Library member"""
    
    def __init__(self, member_id: str, name: str, email: str):
        self._member_id = member_id
        self._name = name
        self._email = email
        self._borrowed_books: List[Book] = []
    
    def borrow_book(self, book: Book) -> bool:
        if book.borrow():
            self._borrowed_books.append(book)
            return True
        return False
    
    def return_book(self, book: Book):
        if book in self._borrowed_books:
            book.return_book()
            self._borrowed_books.remove(book)

class Library:
    """Library system managing books and members"""
    
    def __init__(self):
        self._books: List[Book] = []
        self._members: List[Member] = []
    
    def add_book(self, book: Book):
        self._books.append(book)
    
    def register_member(self, member: Member):
        self._members.append(member)
    
    def search_book(self, title: str) -> Book:
        for book in self._books:
            if book.title.lower() == title.lower():
                return book
        return None
```

### Key Points

- **Purpose**: Show static structure and relationships
- **When to use**: Initial design phase, documenting architecture
- **Relationships**: Understand inheritance, composition, aggregation
- **Best practice**: Keep diagrams focused; avoid showing all details

---

## 2. Sequence Diagrams

**Sequence diagrams** show how objects interact with each other over time, focusing on the order of messages exchanged between objects.

### Components

1. **Lifelines**: Vertical lines representing objects/actors
2. **Activation boxes**: Rectangles on lifelines showing when object is active
3. **Messages**: Arrows showing method calls
4. **Return messages**: Dashed arrows showing return values
5. **Self-calls**: Messages an object sends to itself

### Example: Book Borrowing Process

```
Member          Library         Book
  │               │              │
  │──borrow()────>│              │
  │               │              │
  │               │──search()──>│
  │               │<──return─────│
  │               │              │
  │               │──borrow()───>│
  │               │<──success────│
  │<──success─────│              │
  │               │              │
```

### Python Implementation

```python
class Library:
    """Library with sequence diagram example"""
    
    def __init__(self):
        self._books: List[Book] = []
    
    def borrow_book(self, member: Member, book_title: str) -> bool:
        """Sequence: Member -> Library.borrow_book() -> Library.search() -> Book.borrow()"""
        # Step 1: Library searches for book
        book = self._search_book(book_title)
        
        if book is None:
            return False
        
        # Step 2: Library calls book's borrow method
        if book.borrow():
            # Step 3: Library updates member's borrowed books
            member._borrowed_books.append(book)
            return True
        
        return False
    
    def _search_book(self, title: str) -> Book:
        """Internal search method"""
        for book in self._books:
            if book.title.lower() == title.lower():
                return book
        return None
```

### Key Points

- **Purpose**: Show dynamic behavior and message flow
- **When to use**: Understanding interactions, debugging flow
- **Time flows downward**: Top to bottom represents time progression
- **Best practice**: Focus on one use case per diagram

---

## 3. Activity Diagrams

**Activity diagrams** represent workflows and business processes, showing the flow of control from one activity to another.

### Components

1. **Start node**: Filled circle (initial state)
2. **Activity nodes**: Rounded rectangles (actions)
3. **Decision nodes**: Diamonds (if/else conditions)
4. **Merge nodes**: Diamonds (merge control flows)
5. **End node**: Filled circle with border (final state)
6. **Swimlanes**: Vertical partitions (different actors/objects)

### Example: Book Return Process

```
    [Start]
      │
      ▼
┌─────────────┐
│ Check Book  │
│ Condition   │
└─────────────┘
      │
      ▼
    ┌───┐
    │OK?│
    └───┘
   ╱     ╲
 Yes     No
 ╱         ╲
▼           ▼
┌──────┐  ┌──────────┐
│Return│  │Charge Fee│
│Book  │  └──────────┘
└──────┘      │
      │       │
      └───┬───┘
          ▼
    ┌─────────┐
    │ Update  │
    │ Records │
    └─────────┘
          │
          ▼
      [End]
```

### Python Implementation

```python
class Library:
    """Library with activity diagram example"""
    
    def return_book(self, member: Member, book: Book, condition: str) -> bool:
        """
        Activity diagram flow:
        1. Check book condition
        2. Decision: Is condition OK?
        3. If Yes: Return book
        4. If No: Charge fee
        5. Update records
        """
        # Activity: Check book condition
        is_condition_ok = (condition == "good" or condition == "excellent")
        
        # Decision node
        if is_condition_ok:
            # Activity: Return book
            book.return_book()
            if book in member._borrowed_books:
                member._borrowed_books.remove(book)
        else:
            # Activity: Charge fee
            fee = self._calculate_damage_fee(book)
            member._charge_fee(fee)
        
        # Activity: Update records
        self._update_return_records(member, book)
        return True
    
    def _calculate_damage_fee(self, book: Book) -> float:
        return 10.0  # Simplified
    
    def _update_return_records(self, member: Member, book: Book):
        # Update database/logs
        pass
```

### Key Points

- **Purpose**: Model workflows and business processes
- **When to use**: Complex processes, parallel activities
- **Swimlanes**: Show responsibilities of different actors
- **Best practice**: Keep decision logic clear and simple

---

## 4. State Diagrams

**State diagrams** (State Machine Diagrams) show the different states an object can be in and the transitions between states.

### Components

1. **States**: Rounded rectangles
2. **Initial state**: Filled circle
3. **Final state**: Filled circle with border
4. **Transitions**: Arrows with triggers/conditions
5. **Guards**: Conditions in square brackets

### Example: Book State Machine

```
    [Start]
      │
      ▼
┌─────────────┐
│  Available  │
└─────────────┘
      │
      │ borrow()
      ▼
┌─────────────┐
│   Borrowed  │
└─────────────┘
      │
      │ return()
      ▼
┌─────────────┐
│  Available  │
└─────────────┘
      │
      │ [damaged]
      ▼
┌─────────────┐
│   Damaged   │
└─────────────┘
      │
      │ repair()
      ▼
┌─────────────┐
│  Available  │
└─────────────┘
```

### Python Implementation

```python
from enum import Enum

class BookState(Enum):
    """States a book can be in"""
    AVAILABLE = "available"
    BORROWED = "borrowed"
    DAMAGED = "damaged"
    RESERVED = "reserved"

class Book:
    """Book with state diagram implementation"""
    
    def __init__(self, isbn: str, title: str, author: str):
        self._isbn = isbn
        self._title = title
        self._author = author
        self._state = BookState.AVAILABLE  # Initial state
    
    def borrow(self) -> bool:
        """Transition: AVAILABLE -> BORROWED"""
        if self._state == BookState.AVAILABLE:
            self._state = BookState.BORROWED
            return True
        return False
    
    def return_book(self, condition: str = "good"):
        """Transition: BORROWED -> AVAILABLE or DAMAGED"""
        if self._state == BookState.BORROWED:
            if condition == "damaged":
                self._state = BookState.DAMAGED
            else:
                self._state = BookState.AVAILABLE
    
    def repair(self):
        """Transition: DAMAGED -> AVAILABLE"""
        if self._state == BookState.DAMAGED:
            self._state = BookState.AVAILABLE
    
    def reserve(self):
        """Transition: AVAILABLE -> RESERVED"""
        if self._state == BookState.AVAILABLE:
            self._state = BookState.RESERVED
    
    @property
    def state(self) -> BookState:
        return self._state
```

### Key Points

- **Purpose**: Model object lifecycle and state changes
- **When to use**: Objects with clear states and transitions
- **State vs Activity**: States are conditions, activities are actions
- **Best practice**: Keep state transitions clear and well-defined

---

## 5. Component Diagrams

**Component diagrams** show the physical structure of a system, including components, interfaces, and dependencies.

### Components

1. **Component**: Rectangle with component icon
2. **Interface**: Circle (provided) or semicircle (required)
3. **Dependencies**: Dashed arrows
4. **Ports**: Connection points on components

### Example: E-commerce System Components

```
┌─────────────────┐
│  User Interface │
└─────────────────┘
        │
        │ uses
        ▼
┌─────────────────┐      ┌─────────────────┐
│  Order Service  │──────>│ Payment Service │
└─────────────────┘      └─────────────────┘
        │
        │ uses
        ▼
┌─────────────────┐      ┌─────────────────┐
│ Inventory Service│──────>│  Database      │
└─────────────────┘      └─────────────────┘
```

### Python Implementation

```python
from abc import ABC, abstractmethod

# Component: Payment Service Interface
class PaymentService(ABC):
    """Interface for payment processing"""
    
    @abstractmethod
    def process_payment(self, amount: float, payment_method: str) -> bool:
        pass

# Component: Order Service
class OrderService:
    """Order management component"""
    
    def __init__(self, payment_service: PaymentService, inventory_service: 'InventoryService'):
        self._payment_service = payment_service  # Dependency
        self._inventory_service = inventory_service  # Dependency
    
    def create_order(self, items: List[dict], payment_method: str) -> bool:
        """Uses payment and inventory services"""
        total = sum(item['price'] for item in items)
        
        # Use payment service
        if not self._payment_service.process_payment(total, payment_method):
            return False
        
        # Use inventory service
        if not self._inventory_service.reserve_items(items):
            return False
        
        return True

# Component: Inventory Service
class InventoryService:
    """Inventory management component"""
    
    def reserve_items(self, items: List[dict]) -> bool:
        # Check and reserve inventory
        return True
```

### Key Points

- **Purpose**: Show system architecture and component relationships
- **When to use**: High-level system design, microservices architecture
- **Interfaces**: Define contracts between components
- **Best practice**: Focus on major components and their interactions

---

## 6. Use Case Diagrams

**Use case diagrams** show the interactions between actors (users, systems) and the system, representing functional requirements.

### Components

1. **Actor**: Stick figure (user, external system)
2. **Use case**: Oval (functionality)
3. **System boundary**: Rectangle (system scope)
4. **Relationships**: Lines connecting actors to use cases
5. **Include/Extend**: Dependencies between use cases

### Example: Library Management System

```
        ┌─────────────────────────────────────┐
        │   Library Management System        │
        │                                     │
        │  ┌─────────────┐                   │
        │  │ Search Book │                   │
        │  └─────────────┘                   │
        │                                     │
        │  ┌─────────────┐                   │
        │  │ Borrow Book │◄──────include─────┤
        │  └─────────────┘                   │
        │                                     │
        │  ┌─────────────┐                   │
        │  │ Return Book │                   │
        │  └─────────────┘                   │
        │                                     │
        │  ┌─────────────┐                   │
        │  │ Reserve Book│                   │
        │  └─────────────┘                   │
        └─────────────────────────────────────┘
              ▲           ▲           ▲
              │           │           │
        ┌─────┘           │           └─────┐
        │                 │                 │
   [Member]          [Librarian]      [Admin]
```

### Python Implementation

```python
from abc import ABC, abstractmethod

# Actor: Member
class Member:
    """Actor in use case diagram"""
    
    def search_book(self, library: 'Library', title: str):
        """Use case: Search Book"""
        return library.search_book(title)
    
    def borrow_book(self, library: 'Library', title: str):
        """Use case: Borrow Book (includes Search Book)"""
        book = library.search_book(title)  # Include relationship
        if book:
            return library.borrow_book(self, book)
        return False
    
    def return_book(self, library: 'Library', book: 'Book'):
        """Use case: Return Book"""
        return library.return_book(self, book)
    
    def reserve_book(self, library: 'Library', title: str):
        """Use case: Reserve Book"""
        book = library.search_book(title)
        if book:
            return library.reserve_book(self, book)
        return False

# Actor: Librarian
class Librarian:
    """Actor with additional privileges"""
    
    def add_book(self, library: 'Library', book: 'Book'):
        """Use case: Add Book"""
        library.add_book(book)
    
    def remove_book(self, library: 'Library', book: 'Book'):
        """Use case: Remove Book"""
        library.remove_book(book)
```

### Key Points

- **Purpose**: Capture functional requirements from user perspective
- **When to use**: Requirements gathering, system scope definition
- **Actors**: Represent different user roles
- **Best practice**: Keep use cases at appropriate level of detail

---

## Tools and Best Practices

### Popular UML Tools

1. **Draw.io (diagrams.net)**: Free, web-based, supports UML
2. **Lucidchart**: Professional, collaborative
3. **PlantUML**: Text-based UML, version control friendly
4. **Visual Paradigm**: Comprehensive UML tool
5. **StarUML**: Open-source UML modeling tool

### Best Practices

1. **Keep it simple**: Don't overcomplicate diagrams
2. **Be consistent**: Use standard UML notation
3. **Focus on clarity**: Diagrams should communicate clearly
4. **Update regularly**: Keep diagrams in sync with code
5. **Use appropriate diagrams**: Choose the right diagram for the purpose
6. **Document assumptions**: Add notes for clarity
7. **Version control**: Track diagram changes

### When to Use Each Diagram

- **Class Diagram**: System structure, relationships
- **Sequence Diagram**: Object interactions, method calls
- **Activity Diagram**: Business processes, workflows
- **State Diagram**: Object lifecycle, state changes
- **Component Diagram**: System architecture, modules
- **Use Case Diagram**: Requirements, user interactions

---

## Key Takeaways

- **UML is a communication tool**: Helps teams understand and discuss design
- **Different diagrams for different purposes**: Choose based on what you need to show
- **Class diagrams are fundamental**: Essential for LLD interviews
- **Sequence diagrams show behavior**: Critical for understanding interactions
- **Keep diagrams updated**: Outdated diagrams are worse than no diagrams
- **Practice drawing**: Improves design thinking and communication
- **Tools matter**: Use tools that fit your workflow

---

## References

- [UML 2.5 Specification](https://www.omg.org/spec/UML/2.5.1/)
- [PlantUML Documentation](https://plantuml.com/)
- [Draw.io UML Guide](https://www.diagrams.net/blog/uml-diagrams)
- [Martin Fowler on UML](https://martinfowler.com/bliki/UmlAsSketch.html)

