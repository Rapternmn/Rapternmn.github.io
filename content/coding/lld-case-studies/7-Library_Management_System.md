+++
title = "Library Management System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Complete implementation of Library Management System: Book management, member management, borrowing and returning books, fine calculation, using Observer pattern and CRUD operations."
+++

---

## Problem Statement

Design a library management system that can:
- Manage books (add, remove, search)
- Manage members (register, update)
- Handle book borrowing and returning
- Calculate fines for overdue books
- Track book availability
- Send notifications

---

## Requirements

### Functional Requirements
1. Add/remove books
2. Register/update members
3. Borrow books
4. Return books
5. Search books
6. Calculate fines
7. Reserve books
8. Send notifications

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Optional
import threading

class BookStatus(Enum):
    AVAILABLE = "available"
    BORROWED = "borrowed"
    RESERVED = "reserved"
    DAMAGED = "damaged"

class Book:
    def __init__(self, isbn: str, title: str, author: str, copies: int = 1):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.total_copies = copies
        self.available_copies = copies
        self.status = BookStatus.AVAILABLE
        self._lock = threading.Lock()
    
    def borrow(self) -> bool:
        with self._lock:
            if self.available_copies > 0:
                self.available_copies -= 1
                if self.available_copies == 0:
                    self.status = BookStatus.BORROWED
                return True
            return False
    
    def return_book(self):
        with self._lock:
            self.available_copies += 1
            if self.status == BookStatus.BORROWED:
                self.status = BookStatus.AVAILABLE
    
    def is_available(self) -> bool:
        return self.available_copies > 0

class Member:
    def __init__(self, member_id: str, name: str, email: str):
        self.member_id = member_id
        self.name = name
        self.email = email
        self.borrowed_books: List[BookItem] = []
        self.max_books = 5
        self._lock = threading.Lock()
    
    def can_borrow(self) -> bool:
        return len(self.borrowed_books) < self.max_books
    
    def add_borrowed_book(self, book_item):
        with self._lock:
            self.borrowed_books.append(book_item)
    
    def remove_borrowed_book(self, book_item):
        with self._lock:
            if book_item in self.borrowed_books:
                self.borrowed_books.remove(book_item)

class BookItem:
    def __init__(self, book: Book, member: Member):
        self.book = book
        self.member = member
        self.borrow_date = datetime.now()
        self.due_date = self.borrow_date + timedelta(days=14)
        self.return_date: Optional[datetime] = None
    
    def is_overdue(self) -> bool:
        return datetime.now() > self.due_date and self.return_date is None
    
    def calculate_fine(self, daily_fine: float = 1.0) -> float:
        if not self.is_overdue():
            return 0.0
        days_overdue = (datetime.now() - self.due_date).days
        return days_overdue * daily_fine

class NotificationService(ABC):
    @abstractmethod
    def notify(self, member: Member, message: str):
        pass

class EmailNotification(NotificationService):
    def notify(self, member: Member, message: str):
        print(f"Email to {member.email}: {message}")

class SMSNotification(NotificationService):
    def notify(self, member: Member, message: str):
        print(f"SMS to {member.member_id}: {message}")

class Library:
    def __init__(self):
        self.books: dict = {}  # isbn -> Book
        self.members: dict = {}  # member_id -> Member
        self.book_items: List[BookItem] = []
        self.notification_service: Optional[NotificationService] = None
        self._lock = threading.Lock()
    
    def set_notification_service(self, service: NotificationService):
        self.notification_service = service
    
    def add_book(self, book: Book):
        with self._lock:
            self.books[book.isbn] = book
    
    def remove_book(self, isbn: str):
        with self._lock:
            if isbn in self.books:
                del self.books[isbn]
    
    def register_member(self, member: Member):
        with self._lock:
            self.members[member.member_id] = member
    
    def search_books(self, query: str) -> List[Book]:
        query_lower = query.lower()
        results = []
        for book in self.books.values():
            if (query_lower in book.title.lower() or 
                query_lower in book.author.lower() or
                query_lower in book.isbn.lower()):
                results.append(book)
        return results
    
    def borrow_book(self, isbn: str, member_id: str) -> bool:
        with self._lock:
            if isbn not in self.books:
                print("Book not found")
                return False
            
            if member_id not in self.members:
                print("Member not found")
                return False
            
            book = self.books[isbn]
            member = self.members[member_id]
            
            if not member.can_borrow():
                print("Member has reached maximum borrowing limit")
                return False
            
            if not book.is_available():
                print("Book not available")
                return False
            
            if book.borrow():
                book_item = BookItem(book, member)
                member.add_borrowed_book(book_item)
                self.book_items.append(book_item)
                
                if self.notification_service:
                    self.notification_service.notify(
                        member, 
                        f"You have borrowed '{book.title}'. Due date: {book_item.due_date.strftime('%Y-%m-%d')}"
                    )
                return True
            return False
    
    def return_book(self, isbn: str, member_id: str) -> bool:
        with self._lock:
            member = self.members.get(member_id)
            if not member:
                return False
            
            book_item = None
            for item in member.borrowed_books:
                if item.book.isbn == isbn:
                    book_item = item
                    break
            
            if not book_item:
                print("Book not borrowed by this member")
                return False
            
            book_item.return_date = datetime.now()
            fine = book_item.calculate_fine()
            
            book_item.book.return_book()
            member.remove_borrowed_book(book_item)
            self.book_items.remove(book_item)
            
            if fine > 0:
                print(f"Book returned. Fine: ${fine:.2f}")
                if self.notification_service:
                    self.notification_service.notify(
                        member,
                        f"Book '{book_item.book.title}' returned. Fine: ${fine:.2f}"
                    )
            else:
                print("Book returned successfully")
            
            return True
    
    def reserve_book(self, isbn: str, member_id: str) -> bool:
        # Implementation for book reservation
        pass
    
    def get_overdue_books(self) -> List[BookItem]:
        return [item for item in self.book_items if item.is_overdue()]

# Usage
library = Library()
library.set_notification_service(EmailNotification())

book1 = Book("ISBN001", "Python Programming", "John Doe", 3)
book2 = Book("ISBN002", "Design Patterns", "Jane Smith", 2)

library.add_book(book1)
library.add_book(book2)

member = Member("M001", "Alice", "alice@example.com")
library.register_member(member)

library.borrow_book("ISBN001", "M001")
library.return_book("ISBN001", "M001")
```

---

## Design Patterns Used

1. **[Observer Pattern]({{< ref "../../design-patterns/11-Observer_Pattern.md" >}})**: `NotificationService` interface with implementations (`EmailNotification`, `SMSNotification`) allows notifying members about book events (borrow, return, overdue)
2. **[Factory Pattern]({{< ref "../../design-patterns/4-Factory_Pattern.md" >}})**: `Book` objects can be created using Factory pattern, especially useful when creating different book types
3. **[Strategy Pattern]({{< ref "../../design-patterns/12-Strategy_Pattern.md" >}})**: Fine calculation can use Strategy pattern to support different fine calculation strategies (daily, flat, tiered)

---

## Key Points

- **Time Complexity**: 
  - Search: O(B) where B = number of books
  - Borrow/Return: O(1) with proper indexing
- **Space Complexity**: O(B + M) for books and members
- **Features**: Fine calculation, notifications, reservations

---

## Practice Problems

- Add support for different book types
- Implement book reservation queue
- Add support for e-books
- Implement book rating system

