+++
title = "Online Book Reader"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Complete implementation of Online Book Reader: Book storage, reading progress tracking, user management, bookmarking, and reading history."
+++

---

## Problem Statement

Design an online book reader system that can:
- Store and retrieve books
- Track reading progress
- Manage user accounts
- Support bookmarks
- Maintain reading history
- Support multiple formats

---

## Implementation

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
import threading

class Book:
    def __init__(self, book_id: str, title: str, author: str, content: str):
        self.book_id = book_id
        self.title = title
        self.author = author
        self.content = content
        self.total_pages = len(content) // 1000  # Simplified
    
    def get_page(self, page_number: int) -> str:
        start = page_number * 1000
        end = start + 1000
        return self.content[start:end]

class User:
    def __init__(self, user_id: str, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.reading_sessions: List[ReadingSession] = []

class ReadingSession:
    def __init__(self, user: User, book: Book):
        self.user = user
        self.book = book
        self.current_page = 0
        self.start_time = datetime.now()
        self.last_read_time = datetime.now()
        self.bookmarks: List[Bookmark] = []
    
    def read_page(self, page_number: int) -> str:
        self.current_page = page_number
        self.last_read_time = datetime.now()
        return self.book.get_page(page_number)
    
    def add_bookmark(self, page_number: int, note: str = ""):
        bookmark = Bookmark(page_number, note, datetime.now())
        self.bookmarks.append(bookmark)
    
    def get_progress(self) -> float:
        return (self.current_page / self.book.total_pages) * 100

class Bookmark:
    def __init__(self, page_number: int, note: str, created_at: datetime):
        self.page_number = page_number
        self.note = note
        self.created_at = created_at

class BookReader:
    def __init__(self):
        self.books: dict = {}  # book_id -> Book
        self.users: dict = {}  # user_id -> User
        self.active_sessions: dict = {}  # user_id -> ReadingSession
        self._lock = threading.Lock()
    
    def add_book(self, book: Book):
        with self._lock:
            self.books[book.book_id] = book
    
    def register_user(self, user: User):
        with self._lock:
            self.users[user.user_id] = user
    
    def start_reading(self, user_id: str, book_id: str) -> bool:
        with self._lock:
            if user_id not in self.users or book_id not in self.books:
                return False
            
            user = self.users[user_id]
            book = self.books[book_id]
            
            # Check if session exists
            if user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                if session.book.book_id == book_id:
                    return True  # Continue existing session
            
            # Create new session
            session = ReadingSession(user, book)
            self.active_sessions[user_id] = session
            user.reading_sessions.append(session)
            return True
    
    def read_page(self, user_id: str, page_number: int) -> Optional[str]:
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            return session.read_page(page_number)
        return None
    
    def add_bookmark(self, user_id: str, page_number: int, note: str = ""):
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            session.add_bookmark(page_number, note)
    
    def get_reading_progress(self, user_id: str) -> Optional[float]:
        if user_id in self.active_sessions:
            return self.active_sessions[user_id].get_progress()
        return None

# Usage
reader = BookReader()

book = Book("B001", "Python Guide", "Author", "Content..." * 100)
user = User("U001", "John", "john@example.com")

reader.add_book(book)
reader.register_user(user)

reader.start_reading("U001", "B001")
page = reader.read_page("U001", 0)
reader.add_bookmark("U001", 5, "Important concept")
```

---

## Key Points

- **Time Complexity**: O(1) for most operations
- **Space Complexity**: O(B + U) for books and users
- **Features**: Progress tracking, bookmarks, reading history

---

## Practice Problems

- Add support for annotations
- Implement reading statistics
- Add social features (sharing, reviews)
- Support offline reading

