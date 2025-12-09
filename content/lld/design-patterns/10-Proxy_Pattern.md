+++
title = "Proxy Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Comprehensive guide to Proxy Pattern: Controlling access to objects, with Python implementations, virtual proxy, protection proxy, remote proxy, use cases, and best practices."
+++

---

## Introduction

The **Proxy Pattern** is a structural design pattern that provides a surrogate or placeholder for another object to control access to it. A proxy acts as an intermediary between the client and the real object.

### Intent

- Control access to an object
- Add functionality before/after object access
- Lazy initialization of expensive objects
- Provide a local representative for remote objects

---

## Problem

Sometimes you need to control access to objects:
- Expensive object creation (lazy loading)
- Access control (permissions)
- Remote objects (network communication)
- Caching (avoid repeated operations)
- Logging/monitoring

### Example Problem

```python
class ExpensiveObject:
    def __init__(self):
        print("Creating expensive object...")
        # Expensive initialization
        import time
        time.sleep(2)  # Simulate expensive operation
        self.data = "Expensive data loaded"
    
    def process(self):
        return f"Processing {self.data}"

# Problem: Object created even if never used
obj = ExpensiveObject()  # Expensive, even if we don't use it!
# What if we don't always need this object?
```

---

## Solution

The Proxy Pattern solves this by:
1. Creating a proxy class with the same interface as the real object
2. Proxy controls access to the real object
3. Proxy can create, cache, or control the real object
4. Client interacts with proxy as if it were the real object

---

## Structure

```
┌──────────────┐
│   Client     │
└──────┬───────┘
       │
       │ uses
       ▼
┌──────────────┐      ┌──────────────┐
│   Subject    │      │    Proxy     │
│  (interface) │◄─────│              │
└──────┬───────┘      └──────┬───────┘
       ▲                     │
       │                     │ controls
       │                     ▼
┌──────────────┐      ┌──────────────┐
│ RealSubject  │      │ RealSubject  │
└──────────────┘      └──────────────┘
```

**Participants**:
- **Subject**: Interface for RealSubject and Proxy
- **RealSubject**: The real object that proxy represents
- **Proxy**: Maintains reference to RealSubject, controls access

---

## Types of Proxies

### 1. Virtual Proxy (Lazy Loading)

```python
from abc import ABC, abstractmethod

class Image(ABC):
    @abstractmethod
    def display(self):
        pass

class RealImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        print(f"Loading {self.filename} from disk...")
        # Expensive operation
        import time
        time.sleep(1)
        print(f"Loaded {self.filename}")
    
    def display(self):
        print(f"Displaying {self.filename}")

class ImageProxy(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self._real_image = None  # Lazy loading
    
    def display(self):
        if self._real_image is None:
            self._real_image = RealImage(self.filename)  # Create only when needed
        self._real_image.display()

# Usage
proxy = ImageProxy("photo.jpg")
# RealImage not created yet

proxy.display()  # Now RealImage is created and displayed
```

### 2. Protection Proxy (Access Control)

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def query(self, sql: str) -> list:
        pass

class RealDatabase(Database):
    def query(self, sql: str) -> list:
        print(f"Executing query: {sql}")
        return [{"id": 1, "name": "Result"}]

class DatabaseProxy(Database):
    def __init__(self, database: Database, user_role: str):
        self._database = database
        self._user_role = user_role
    
    def query(self, sql: str) -> list:
        # Access control
        if self._user_role == "admin":
            return self._database.query(sql)
        elif self._user_role == "user":
            # Only allow SELECT queries
            if sql.strip().upper().startswith("SELECT"):
                return self._database.query(sql)
            else:
                raise PermissionError("Only SELECT queries allowed for users")
        else:
            raise PermissionError("Access denied")

# Usage
real_db = RealDatabase()
admin_proxy = DatabaseProxy(real_db, "admin")
user_proxy = DatabaseProxy(real_db, "user")

admin_proxy.query("DELETE FROM users")  # Allowed
user_proxy.query("SELECT * FROM users")  # Allowed
# user_proxy.query("DELETE FROM users")  # Raises PermissionError
```

### 3. Remote Proxy

```python
from abc import ABC, abstractmethod

class RemoteService(ABC):
    @abstractmethod
    def get_data(self, key: str) -> str:
        pass

class RealRemoteService(RemoteService):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # Would establish network connection here
    
    def get_data(self, key: str) -> str:
        # Simulate network call
        print(f"Network call to {self.host}:{self.port} for key: {key}")
        return f"Data for {key}"

class RemoteProxy(RemoteService):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._service = None
        self._cache = {}
    
    def _get_service(self):
        if self._service is None:
            self._service = RealRemoteService(self.host, self.port)
        return self._service
    
    def get_data(self, key: str) -> str:
        # Check cache first
        if key in self._cache:
            print(f"Cache hit for {key}")
            return self._cache[key]
        
        # Make remote call
        service = self._get_service()
        data = service.get_data(key)
        
        # Cache result
        self._cache[key] = data
        return data

# Usage
proxy = RemoteProxy("api.example.com", 8080)
data1 = proxy.get_data("user1")  # Network call
data2 = proxy.get_data("user1")  # Cache hit
```

### 4. Caching Proxy

```python
from abc import ABC, abstractmethod
import time

class ExpensiveOperation(ABC):
    @abstractmethod
    def compute(self, value: int) -> int:
        pass

class RealOperation(ExpensiveOperation):
    def compute(self, value: int) -> int:
        # Expensive computation
        print(f"Computing for {value}...")
        time.sleep(1)  # Simulate expensive operation
        return value * value

class CachingProxy(ExpensiveOperation):
    def __init__(self, operation: ExpensiveOperation):
        self._operation = operation
        self._cache = {}
    
    def compute(self, value: int) -> int:
        if value in self._cache:
            print(f"Cache hit for {value}")
            return self._cache[value]
        
        result = self._operation.compute(value)
        self._cache[value] = result
        return result
    
    def clear_cache(self):
        self._cache.clear()

# Usage
real_op = RealOperation()
proxy = CachingProxy(real_op)

result1 = proxy.compute(5)  # Expensive computation
result2 = proxy.compute(5)  # Cache hit - fast!
```

---

## Real-World Examples

### Example 1: Lazy Loading Proxy

```python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def get_page(self, page_number: int) -> str:
        pass

class RealDocument(Document):
    def __init__(self, filename: str):
        self.filename = filename
        self._pages = self._load_all_pages()
    
    def _load_all_pages(self) -> list:
        print(f"Loading all pages from {self.filename}...")
        # Expensive: load all pages at once
        import time
        time.sleep(2)
        return [f"Page {i} content" for i in range(100)]
    
    def get_page(self, page_number: int) -> str:
        return self._pages[page_number]

class DocumentProxy(Document):
    def __init__(self, filename: str):
        self.filename = filename
        self._document = None
        self._loaded_pages = {}
    
    def _get_document(self):
        if self._document is None:
            self._document = RealDocument(self.filename)
        return self._document
    
    def get_page(self, page_number: int) -> str:
        # Lazy load: only load page when requested
        if page_number not in self._loaded_pages:
            doc = self._get_document()
            self._loaded_pages[page_number] = doc.get_page(page_number)
        return self._loaded_pages[page_number]

# Usage
proxy = DocumentProxy("large_document.pdf")
# Document not loaded yet

page1 = proxy.get_page(0)  # Now loads document
page2 = proxy.get_page(1)  # Uses already loaded document
```

### Example 2: Access Control Proxy

```python
from abc import ABC, abstractmethod
from enum import Enum

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

class File(ABC):
    @abstractmethod
    def read(self) -> str:
        pass
    
    @abstractmethod
    def write(self, content: str):
        pass
    
    @abstractmethod
    def delete(self):
        pass

class RealFile(File):
    def __init__(self, filename: str):
        self.filename = filename
        self._content = ""
    
    def read(self) -> str:
        return self._content
    
    def write(self, content: str):
        self._content = content
        print(f"Written to {self.filename}")
    
    def delete(self):
        print(f"Deleted {self.filename}")

class FileProxy(File):
    def __init__(self, file: File, permissions: set):
        self._file = file
        self._permissions = permissions
    
    def read(self) -> str:
        if Permission.READ in self._permissions:
            return self._file.read()
        raise PermissionError("Read permission denied")
    
    def write(self, content: str):
        if Permission.WRITE in self._permissions:
            self._file.write(content)
        else:
            raise PermissionError("Write permission denied")
    
    def delete(self):
        if Permission.DELETE in self._permissions:
            self._file.delete()
        else:
            raise PermissionError("Delete permission denied")

# Usage
real_file = RealFile("document.txt")
read_only_proxy = FileProxy(real_file, {Permission.READ})
full_access_proxy = FileProxy(real_file, {Permission.READ, Permission.WRITE, Permission.DELETE})

content = read_only_proxy.read()  # Allowed
# read_only_proxy.write("New content")  # Raises PermissionError

full_access_proxy.write("New content")  # Allowed
```

### Example 3: Smart Reference Proxy

```python
class ExpensiveResource:
    def __init__(self):
        self._reference_count = 0
    
    def acquire(self):
        self._reference_count += 1
        print(f"Resource acquired. References: {self._reference_count}")
    
    def release(self):
        self._reference_count -= 1
        print(f"Resource released. References: {self._reference_count}")
        if self._reference_count == 0:
            print("Resource can be garbage collected")

class ResourceProxy:
    def __init__(self, resource: ExpensiveResource):
        self._resource = resource
        self._resource.acquire()
    
    def __del__(self):
        # Automatic cleanup
        if hasattr(self, '_resource'):
            self._resource.release()
    
    def use(self):
        print("Using resource...")

# Usage
resource = ExpensiveResource()
proxy1 = ResourceProxy(resource)
proxy2 = ResourceProxy(resource)

proxy1.use()
del proxy1  # Reference count decreases
del proxy2  # Reference count becomes 0
```

---

## Use Cases

### When to Use Proxy Pattern

✅ **Lazy Loading**: When object creation is expensive
✅ **Access Control**: When you need to control access to objects
✅ **Remote Objects**: When dealing with remote services
✅ **Caching**: When you want to cache expensive operations
✅ **Logging**: When you need to log object access
✅ **Monitoring**: When you need to monitor object usage

### When NOT to Use

❌ **Simple Objects**: Overkill for simple objects
❌ **Performance Critical**: Adds indirection overhead
❌ **Direct Access Needed**: When clients need direct access
❌ **Too Many Proxies**: If you need too many proxies, consider refactoring

---

## Pros and Cons

### Advantages

✅ **Lazy Loading**: Create objects only when needed
✅ **Access Control**: Control who can access objects
✅ **Caching**: Cache expensive operations
✅ **Remote Access**: Handle remote objects transparently
✅ **Separation of Concerns**: Separate access control from business logic

### Disadvantages

❌ **Complexity**: Adds another layer of abstraction
❌ **Performance**: Adds indirection overhead
❌ **Debugging**: Can make debugging harder
❌ **Maintenance**: More code to maintain

---

## Proxy vs Other Patterns

### Proxy vs Decorator

- **Proxy**: Controls access to object (same interface)
- **Decorator**: Adds behavior to object (enhances interface)

### Proxy vs Adapter

- **Proxy**: Same interface, controls access
- **Adapter**: Different interface, makes compatible

### Proxy vs Facade

- **Proxy**: Controls access to one object
- **Facade**: Simplifies interface to subsystem

---

## Best Practices

### 1. Implement Same Interface

```python
class Proxy(Subject):
    def operation(self):
        # Control access
        return self._real_subject.operation()
```

### 2. Lazy Initialization

```python
class Proxy:
    def __init__(self):
        self._real_subject = None  # Not created yet
    
    def operation(self):
        if self._real_subject is None:
            self._real_subject = RealSubject()  # Create when needed
        return self._real_subject.operation()
```

### 3. Handle Errors Gracefully

```python
class Proxy:
    def operation(self):
        try:
            return self._real_subject.operation()
        except Exception as e:
            # Handle or log error
            raise ProxyError(f"Proxy operation failed: {e}")
```

### 4. Use `__getattr__` for Transparent Proxies

```python
class TransparentProxy:
    def __init__(self, real_object):
        self._real_object = real_object
    
    def __getattr__(self, name):
        # Forward all attribute access
        return getattr(self._real_object, name)
```

### 5. Document Proxy Type

```python
class VirtualProxy:
    """
    Virtual Proxy - lazy loads expensive object.
    
    The real object is created only when first accessed.
    """
    pass
```

---

## Python-Specific Considerations

### 1. Using `__getattr__` for Transparent Proxy

```python
class Proxy:
    def __init__(self, real_object):
        self._real_object = real_object
    
    def __getattr__(self, name):
        # Transparently forward attribute access
        return getattr(self._real_object, name)
```

### 2. Property-Based Proxy

```python
class Proxy:
    def __init__(self):
        self._real_object = None
    
    @property
    def real_object(self):
        if self._real_object is None:
            self._real_object = RealObject()
        return self._real_object
```

### 3. Context Manager Proxy

```python
class ResourceProxy:
    def __enter__(self):
        self._resource = ExpensiveResource()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._resource.cleanup()

# Usage
with ResourceProxy() as proxy:
    proxy.use()
```

### 4. Descriptor Proxy

```python
class ProxyDescriptor:
    def __init__(self, real_object):
        self._real_object = real_object
    
    def __get__(self, instance, owner):
        return self._real_object
```

---

## Common Pitfalls

### 1. Not Implementing Full Interface

```python
# Bad: Missing methods
class Proxy:
    def method1(self):
        pass
    # Missing method2

# Good: Complete interface
class Proxy(Subject):
    def method1(self):
        pass
    
    def method2(self):
        pass
```

### 2. Forgetting Lazy Initialization

```python
# Bad: Creates object immediately
class Proxy:
    def __init__(self):
        self._real_object = RealObject()  # Created even if not used

# Good: Lazy initialization
class Proxy:
    def __init__(self):
        self._real_object = None
    
    def operation(self):
        if self._real_object is None:
            self._real_object = RealObject()
        return self._real_object.operation()
```

### 3. Not Handling Edge Cases

```python
# Bad: No error handling
class Proxy:
    def operation(self):
        return self._real_object.operation()  # May fail if None

# Good: Handle errors
class Proxy:
    def operation(self):
        if self._real_object is None:
            raise ValueError("Real object not initialized")
        return self._real_object.operation()
```

---

## Key Takeaways

- **Purpose**: Control access to objects
- **Types**: Virtual, Protection, Remote, Caching proxies
- **Use when**: Need lazy loading, access control, caching, remote access
- **Benefits**: Lazy loading, access control, caching, separation of concerns
- **Trade-off**: Adds complexity and indirection
- **Python**: Use `__getattr__`, properties, context managers
- **Best practice**: Implement same interface, lazy initialization, handle errors

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Proxy Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/proxy)

