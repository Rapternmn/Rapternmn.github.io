+++
title = "Template Method Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 15
description = "Comprehensive guide to Template Method Pattern: Defining skeleton of algorithm in base class, letting subclasses fill in details, with Python implementations, hooks, use cases, and best practices."
+++

---

## Introduction

The **Template Method Pattern** is a behavioral design pattern that defines the skeleton of an algorithm in a base class, letting subclasses override specific steps of the algorithm without changing its structure.

### Intent

- Define algorithm skeleton in base class
- Let subclasses implement specific steps
- Control algorithm structure
- Promote code reuse

---

## Problem

Sometimes you have algorithms with similar structure but different steps:
- Same overall process, different implementations
- Code duplication across similar algorithms
- Want to control algorithm flow
- Need to ensure certain steps are always executed

### Example Problem

```python
class Coffee:
    def prepare(self):
        self.boil_water()
        self.brew_coffee()
        self.pour_in_cup()
        self.add_sugar()
        self.add_milk()

class Tea:
    def prepare(self):
        self.boil_water()
        self.steep_tea()  # Different from coffee
        self.pour_in_cup()
        self.add_lemon()  # Different from coffee
        # Problem: Code duplication, similar structure
```

---

## Solution

The Template Method Pattern solves this by:
1. Creating abstract base class with template method
2. Template method defines algorithm skeleton
3. Abstract methods for steps that vary
4. Subclasses implement specific steps

---

## Structure

```
┌──────────────┐
│ AbstractClass│
├──────────────┤
│+template_method()│
│#step1()      │
│#step2()      │
│#step3()      │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ ConcreteClass│
├──────────────┤
│#step1()      │
│#step2()      │
└──────────────┘
```

**Participants**:
- **AbstractClass**: Defines template method and abstract steps
- **ConcreteClass**: Implements specific steps

---

## Implementation

### Basic Template Method Pattern

```python
from abc import ABC, abstractmethod

class Beverage(ABC):
    """Abstract class with template method"""
    
    def prepare(self):
        """Template method - defines algorithm skeleton"""
        self.boil_water()
        self.brew()
        self.pour_in_cup()
        if self.customer_wants_condiments():
            self.add_condiments()
    
    def boil_water(self):
        """Common step - implemented in base class"""
        print("Boiling water...")
    
    @abstractmethod
    def brew(self):
        """Abstract step - must be implemented by subclasses"""
        pass
    
    def pour_in_cup(self):
        """Common step"""
        print("Pouring into cup...")
    
    @abstractmethod
    def add_condiments(self):
        """Abstract step"""
        pass
    
    def customer_wants_condiments(self) -> bool:
        """Hook method - can be overridden"""
        return True

class Coffee(Beverage):
    def brew(self):
        print("Dripping coffee through filter...")
    
    def add_condiments(self):
        print("Adding sugar and milk...")

class Tea(Beverage):
    def brew(self):
        print("Steeping the tea...")
    
    def add_condiments(self):
        print("Adding lemon...")
    
    def customer_wants_condiments(self) -> bool:
        """Override hook method"""
        response = input("Would you like lemon? (y/n): ")
        return response.lower() == 'y'

# Usage
coffee = Coffee()
coffee.prepare()
# Boiling water...
# Dripping coffee through filter...
# Pouring into cup...
# Adding sugar and milk...

tea = Tea()
tea.prepare()
```

### Template Method with Hooks

```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Template method with hooks"""
    
    def process(self, data: list):
        """Template method"""
        data = self.validate(data)
        data = self.before_process(data)  # Hook
        data = self.transform(data)
        data = self.after_process(data)  # Hook
        return self.save(data)
    
    def validate(self, data: list) -> list:
        """Common step"""
        if not data:
            raise ValueError("Data cannot be empty")
        return data
    
    @abstractmethod
    def transform(self, data: list) -> list:
        """Abstract step"""
        pass
    
    @abstractmethod
    def save(self, data: list):
        """Abstract step"""
        pass
    
    def before_process(self, data: list) -> list:
        """Hook - can be overridden"""
        return data
    
    def after_process(self, data: list) -> list:
        """Hook - can be overridden"""
        return data

class NumberProcessor(DataProcessor):
    def transform(self, data: list) -> list:
        return [x * 2 for x in data if isinstance(x, (int, float))]
    
    def save(self, data: list):
        print(f"Saving numbers: {data}")
        return data
    
    def before_process(self, data: list) -> list:
        print("Processing numbers...")
        return data

class StringProcessor(DataProcessor):
    def transform(self, data: list) -> list:
        return [str(x).upper() for x in data]
    
    def save(self, data: list):
        print(f"Saving strings: {data}")
        return data
    
    def after_process(self, data: list) -> list:
        print("String processing complete")
        return data

# Usage
number_processor = NumberProcessor()
result = number_processor.process([1, 2, 3, 4, 5])

string_processor = StringProcessor()
result = string_processor.process(["hello", "world"])
```

---

## Real-World Examples

### Example 1: Build Process Template

```python
from abc import ABC, abstractmethod

class BuildProcess(ABC):
    """Template for build processes"""
    
    def build(self):
        """Template method"""
        self.fetch_dependencies()
        self.compile()
        self.run_tests()
        if self.should_deploy():
            self.deploy()
        self.cleanup()
    
    def fetch_dependencies(self):
        """Common step"""
        print("Fetching dependencies...")
    
    @abstractmethod
    def compile(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def run_tests(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def deploy(self):
        """Abstract step"""
        pass
    
    def should_deploy(self) -> bool:
        """Hook method"""
        return True
    
    def cleanup(self):
        """Common step"""
        print("Cleaning up...")

class JavaBuild(BuildProcess):
    def compile(self):
        print("Compiling Java code with javac...")
    
    def run_tests(self):
        print("Running JUnit tests...")
    
    def deploy(self):
        print("Deploying JAR to server...")

class PythonBuild(BuildProcess):
    def compile(self):
        print("Checking Python syntax...")
    
    def run_tests(self):
        print("Running pytest...")
    
    def deploy(self):
        print("Deploying to PyPI...")
    
    def should_deploy(self) -> bool:
        # Python builds might not always deploy
        return False

# Usage
java_build = JavaBuild()
java_build.build()

python_build = PythonBuild()
python_build.build()
```

### Example 2: Game Loop Template

```python
from abc import ABC, abstractmethod
import time

class Game(ABC):
    """Template for game loop"""
    
    def run(self):
        """Template method - game loop"""
        self.initialize()
        while not self.is_game_over():
            self.handle_input()
            self.update()
            self.render()
            time.sleep(0.016)  # ~60 FPS
        self.cleanup()
    
    @abstractmethod
    def initialize(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def handle_input(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def update(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def render(self):
        """Abstract step"""
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """Abstract step"""
        pass
    
    def cleanup(self):
        """Hook method"""
        print("Cleaning up game resources...")

class Tetris(Game):
    def __init__(self):
        self.score = 0
        self.game_over = False
    
    def initialize(self):
        print("Initializing Tetris...")
        print("Loading sprites...")
        print("Setting up board...")
    
    def handle_input(self):
        # Simplified
        pass
    
    def update(self):
        self.score += 1
        if self.score > 100:
            self.game_over = True
    
    def render(self):
        print(f"Rendering Tetris - Score: {self.score}")
    
    def is_game_over(self) -> bool:
        return self.game_over

# Usage
tetris = Tetris()
# tetris.run()  # Would run game loop
```

### Example 3: Data Import Template

```python
from abc import ABC, abstractmethod

class DataImporter(ABC):
    """Template for data import processes"""
    
    def import_data(self, source: str):
        """Template method"""
        data = self.read(source)
        data = self.validate(data)
        data = self.transform(data)
        data = self.enrich(data)  # Hook
        self.save(data)
        self.notify_completion()
    
    @abstractmethod
    def read(self, source: str):
        """Abstract step"""
        pass
    
    @abstractmethod
    def validate(self, data):
        """Abstract step"""
        pass
    
    @abstractmethod
    def transform(self, data):
        """Abstract step"""
        pass
    
    @abstractmethod
    def save(self, data):
        """Abstract step"""
        pass
    
    def enrich(self, data):
        """Hook method - can be overridden"""
        return data
    
    def notify_completion(self):
        """Common step"""
        print("Import completed successfully")

class CSVImporter(DataImporter):
    def read(self, source: str):
        print(f"Reading CSV from {source}...")
        return [{"name": "John", "age": 30}]
    
    def validate(self, data):
        print("Validating CSV data...")
        return data
    
    def transform(self, data):
        print("Transforming CSV data...")
        return data
    
    def save(self, data):
        print("Saving to database...")

class JSONImporter(DataImporter):
    def read(self, source: str):
        print(f"Reading JSON from {source}...")
        return [{"name": "Jane", "age": 25}]
    
    def validate(self, data):
        print("Validating JSON data...")
        return data
    
    def transform(self, data):
        print("Transforming JSON data...")
        return data
    
    def save(self, data):
        print("Saving to database...")
    
    def enrich(self, data):
        """Override hook to add enrichment"""
        print("Enriching JSON data with additional fields...")
        for item in data:
            item['imported_at'] = '2024-01-01'
        return data

# Usage
csv_importer = CSVImporter()
csv_importer.import_data("data.csv")

json_importer = JSONImporter()
json_importer.import_data("data.json")
```

---

## Use Cases

### When to Use Template Method Pattern

✅ **Similar Algorithms**: When you have algorithms with similar structure
✅ **Code Reuse**: When you want to reuse common algorithm steps
✅ **Control Flow**: When you want to control algorithm structure
✅ **Framework Development**: When building frameworks
✅ **Eliminate Duplication**: When you have code duplication

### When NOT to Use

❌ **Different Structures**: When algorithms have very different structures
❌ **Too Many Variations**: When there are too many variations
❌ **Simple Algorithms**: Overkill for simple algorithms
❌ **Performance Critical**: Adds inheritance overhead

---

## Pros and Cons

### Advantages

✅ **Code Reuse**: Common code in base class
✅ **Control Structure**: Base class controls algorithm flow
✅ **Extensibility**: Easy to add new variations
✅ **Eliminates Duplication**: Reduces code duplication
✅ **Open/Closed**: Open for extension, closed for modification

### Disadvantages

❌ **Inheritance**: Requires inheritance (composition might be better)
❌ **Rigid Structure**: Algorithm structure is fixed
❌ **Limited Flexibility**: Less flexible than composition
❌ **Tight Coupling**: Subclasses tightly coupled to base class

---

## Template Method vs Other Patterns

### Template Method vs Strategy

- **Template Method**: Inheritance-based, controls structure
- **Strategy**: Composition-based, algorithm selection

### Template Method vs Factory Method

- **Template Method**: Defines algorithm skeleton
- **Factory Method**: Creates objects

### Template Method vs Hook Method

- **Template Method**: Uses hook methods
- **Hook Method**: Extension point in template method

---

## Best Practices

### 1. Make Template Method Final

```python
class BaseClass:
    def template_method(self):
        # Final - subclasses shouldn't override
        self.step1()
        self.step2()
    
    @abstractmethod
    def step1(self):
        pass
```

### 2. Use Hook Methods

```python
class BaseClass:
    def template_method(self):
        self.step1()
        if self.should_do_step2():  # Hook
            self.step2()
    
    def should_do_step2(self) -> bool:
        return True  # Default implementation
```

### 3. Document Template Method

```python
class BaseClass:
    def template_method(self):
        """
        Template method defining algorithm structure.
        
        Steps:
        1. step1() - Must be implemented
        2. step2() - Optional (hook)
        3. step3() - Must be implemented
        """
        self.step1()
        self.step2()
        self.step3()
```

### 4. Minimize Abstract Methods

```python
# Bad: Too many abstract methods
class BaseClass:
    @abstractmethod
    def step1(self): pass
    @abstractmethod
    def step2(self): pass
    @abstractmethod
    def step3(self): pass
    # ... many more

# Good: Only essential steps are abstract
class BaseClass:
    @abstractmethod
    def essential_step(self): pass
    # Common steps implemented in base
```

### 5. Use Composition When Appropriate

```python
# Consider composition instead of inheritance
class Processor:
    def __init__(self, transformer, validator):
        self.transformer = transformer
        self.validator = validator
    
    def process(self, data):
        data = self.validator.validate(data)
        return self.transformer.transform(data)
```

---

## Python-Specific Considerations

### 1. Using `@abstractmethod`

```python
from abc import ABC, abstractmethod

class BaseClass(ABC):
    @abstractmethod
    def step(self):
        pass
```

### 2. Template Method with `super()`

```python
class BaseClass:
    def template_method(self):
        self.step1()
        self.step2()

class DerivedClass(BaseClass):
    def step1(self):
        super().step1()  # Call base implementation
        # Add additional behavior
```

### 3. Using Mixins

```python
class TemplateMixin:
    def template_method(self):
        self.step1()
        self.step2()

class ConcreteClass(TemplateMixin):
    def step1(self):
        pass
    def step2(self):
        pass
```

### 4. Template Method with Functions

```python
def template_function(step1_func, step2_func):
    """Template function"""
    step1_func()
    step2_func()

# Usage
template_function(
    lambda: print("Step 1"),
    lambda: print("Step 2")
)
```

---

## Common Pitfalls

### 1. Overriding Template Method

```python
# Bad: Overriding template method
class Derived(Base):
    def template_method(self):
        # Changes algorithm structure
        self.step2()
        self.step1()

# Good: Override only steps
class Derived(Base):
    def step1(self):
        # Custom implementation
        pass
```

### 2. Too Many Abstract Methods

```python
# Bad: Too many abstract methods
class Base:
    @abstractmethod
    def step1(self): pass
    @abstractmethod
    def step2(self): pass
    # ... 10 more

# Good: Minimize abstract methods
class Base:
    def step1(self):
        # Default implementation
        pass
    @abstractmethod
    def step2(self): pass  # Only essential
```

### 3. Not Using Hooks

```python
# Bad: No flexibility
class Base:
    def template_method(self):
        self.step1()
        self.step2()  # Always executed

# Good: Use hooks
class Base:
    def template_method(self):
        self.step1()
        if self.should_do_step2():
            self.step2()
```

---

## Key Takeaways

- **Purpose**: Define algorithm skeleton, let subclasses fill details
- **Use when**: Similar algorithms, code reuse, control structure
- **Benefits**: Code reuse, control flow, eliminates duplication
- **Trade-off**: Requires inheritance, rigid structure
- **Python**: Use `@abstractmethod`, hooks, consider composition
- **Best practice**: Make template method final, use hooks, minimize abstract methods
- **Common use**: Frameworks, build processes, data processing pipelines

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python ABC Module](https://docs.python.org/3/library/abc.html)
- [Template Method Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/template-method)

