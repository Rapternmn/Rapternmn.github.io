+++
title = "Command Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 13
description = "Comprehensive guide to Command Pattern: Encapsulating requests as objects, with Python implementations, undo/redo, macro commands, use cases, and best practices."
+++

---

## Introduction

The **Command Pattern** is a behavioral design pattern that turns a request into a stand-alone object that contains all information about the request. This transformation lets you parameterize methods with different requests, delay or queue a request's execution, and support undoable operations.

### Intent

- Encapsulate requests as objects
- Parameterize objects with operations
- Queue operations, log requests, support undo
- Decouple sender from receiver

---

## Problem

Sometimes you need to:
- Queue operations for later execution
- Support undo/redo functionality
- Log operations
- Decouple request sender from receiver

### Example Problem

```python
class Light:
    def on(self):
        print("Light is ON")
    
    def off(self):
        print("Light is OFF")

class Button:
    def __init__(self, light: Light):
        self.light = light
    
    def click(self):
        # Problem: Button is tightly coupled to Light
        # What if we want to control TV, Fan, etc.?
        self.light.on()
```

---

## Solution

The Command Pattern solves this by:
1. Creating command objects that encapsulate requests
2. Commands have execute() method
3. Invoker calls execute() on command
4. Commands can support undo()

---

## Structure

```
┌──────────────┐
│   Invoker    │
├──────────────┤
│ + set_command()│
│ + execute()  │
└──────┬───────┘
       │
       │ calls
       ▼
┌──────────────┐
│   Command    │
│  (interface) │
├──────────────┤
│ + execute()  │
│ + undo()     │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ Concrete     │
│ Commands     │
└──────┬───────┘
       │
       │ uses
       ▼
┌──────────────┐
│   Receiver   │
└──────────────┘
```

**Participants**:
- **Command**: Interface for executing operations
- **ConcreteCommand**: Implements execute() and undo()
- **Receiver**: Object that performs actual work
- **Invoker**: Calls command's execute()

---

## Implementation

### Basic Command Pattern

```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class Light:
    def on(self):
        print("Light is ON")
    
    def off(self):
        print("Light is OFF")

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.on()
    
    def undo(self):
        self.light.off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.off()
    
    def undo(self):
        self.light.on()

class RemoteControl:
    def __init__(self):
        self.command = None
        self.history = []
    
    def set_command(self, command: Command):
        self.command = command
    
    def press_button(self):
        if self.command:
            self.command.execute()
            self.history.append(self.command)
    
    def press_undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()

# Usage
light = Light()
light_on = LightOnCommand(light)
light_off = LightOffCommand(light)

remote = RemoteControl()
remote.set_command(light_on)
remote.press_button()  # Light is ON

remote.set_command(light_off)
remote.press_button()  # Light is OFF

remote.press_undo()  # Light is ON (undo last command)
```

### Command with State (Undo/Redo)

```python
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class TV:
    def __init__(self):
        self.channel = 1
        self.volume = 10
    
    def set_channel(self, channel: int):
        old_channel = self.channel
        self.channel = channel
        print(f"TV channel changed from {old_channel} to {channel}")
        return old_channel
    
    def set_volume(self, volume: int):
        old_volume = self.volume
        self.volume = volume
        print(f"TV volume changed from {old_volume} to {volume}")
        return old_volume

class ChangeChannelCommand(Command):
    def __init__(self, tv: TV, new_channel: int):
        self.tv = tv
        self.new_channel = new_channel
        self.old_channel = None
    
    def execute(self):
        self.old_channel = self.tv.set_channel(self.new_channel)
    
    def undo(self):
        if self.old_channel is not None:
            self.tv.set_channel(self.old_channel)

class ChangeVolumeCommand(Command):
    def __init__(self, tv: TV, new_volume: int):
        self.tv = tv
        self.new_volume = new_volume
        self.old_volume = None
    
    def execute(self):
        self.old_volume = self.tv.set_volume(self.new_volume)
    
    def undo(self):
        if self.old_volume is not None:
            self.tv.set_volume(self.old_volume)

class CommandManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
    
    def execute_command(self, command: Command):
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()  # Clear redo stack
    
    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
    
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.undo_stack.append(command)

# Usage
tv = TV()
manager = CommandManager()

manager.execute_command(ChangeChannelCommand(tv, 5))
manager.execute_command(ChangeVolumeCommand(tv, 20))
manager.undo()  # Undo volume change
manager.undo()  # Undo channel change
manager.redo()  # Redo channel change
```

---

## Real-World Examples

### Example 1: Text Editor Commands

```python
from abc import ABC, abstractmethod

class TextEditor:
    def __init__(self):
        self.text = ""
        self.cursor = 0
    
    def insert(self, text: str, position: int):
        self.text = self.text[:position] + text + self.text[position:]
        self.cursor = position + len(text)
        return position
    
    def delete(self, position: int, length: int):
        deleted = self.text[position:position + length]
        self.text = self.text[:position] + self.text[position + length:]
        return deleted

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str, position: int):
        self.editor = editor
        self.text = text
        self.position = position
    
    def execute(self):
        self.editor.insert(self.text, self.position)
    
    def undo(self):
        self.editor.delete(self.position, len(self.text))

class DeleteCommand(Command):
    def __init__(self, editor: TextEditor, position: int, length: int):
        self.editor = editor
        self.position = position
        self.length = length
        self.deleted_text = None
    
    def execute(self):
        self.deleted_text = self.editor.delete(self.position, self.length)
    
    def undo(self):
        if self.deleted_text:
            self.editor.insert(self.deleted_text, self.position)

class CommandHistory:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
    
    def execute(self, command: Command):
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()
    
    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
    
    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.undo_stack.append(command)

# Usage
editor = TextEditor()
history = CommandHistory()

history.execute(InsertCommand(editor, "Hello", 0))
history.execute(InsertCommand(editor, " World", 5))
print(editor.text)  # "Hello World"

history.undo()
print(editor.text)  # "Hello"

history.redo()
print(editor.text)  # "Hello World"
```

### Example 2: Macro Commands

```python
class MacroCommand(Command):
    def __init__(self, commands: list):
        self.commands = commands
    
    def execute(self):
        for command in self.commands:
            command.execute()
    
    def undo(self):
        # Undo in reverse order
        for command in reversed(self.commands):
            command.undo()

# Usage
light = Light()
tv = TV()

commands = [
    LightOnCommand(light),
    ChangeChannelCommand(tv, 5),
    ChangeVolumeCommand(tv, 15)
]

macro = MacroCommand(commands)
macro.execute()  # Executes all commands
macro.undo()     # Undoes all commands
```

### Example 3: Queue and Log Commands

```python
class CommandQueue:
    def __init__(self):
        self.queue = []
    
    def add_command(self, command: Command):
        self.queue.append(command)
    
    def process_queue(self):
        while self.queue:
            command = self.queue.pop(0)
            command.execute()

class LoggingCommand(Command):
    def __init__(self, command: Command, logger):
        self.command = command
        self.logger = logger
    
    def execute(self):
        self.logger.log(f"Executing: {self.command.__class__.__name__}")
        self.command.execute()
    
    def undo(self):
        self.logger.log(f"Undoing: {self.command.__class__.__name__}")
        self.command.undo()

# Usage
queue = CommandQueue()
queue.add_command(LightOnCommand(light))
queue.add_command(ChangeChannelCommand(tv, 5))
queue.process_queue()  # Process all queued commands
```

---

## Use Cases

### When to Use Command Pattern

✅ **Undo/Redo**: When you need undo/redo functionality
✅ **Queue Operations**: When you need to queue operations
✅ **Logging**: When you need to log operations
✅ **Macro Commands**: When you need to combine commands
✅ **Decoupling**: When you want to decouple sender from receiver
✅ **Transactions**: When implementing transactional operations

### When NOT to Use

❌ **Simple Operations**: Overkill for simple operations
❌ **Performance Critical**: Adds overhead for simple cases
❌ **No Undo Needed**: If undo/redo not needed
❌ **Direct Calls**: When direct method calls are sufficient

---

## Pros and Cons

### Advantages

✅ **Decoupling**: Decouples sender from receiver
✅ **Undo/Redo**: Easy to implement undo/redo
✅ **Queueing**: Can queue and schedule commands
✅ **Logging**: Easy to log commands
✅ **Macro Commands**: Can combine commands
✅ **Extensibility**: Easy to add new commands

### Disadvantages

❌ **Complexity**: Adds many command classes
❌ **Overhead**: Adds indirection layer
❌ **Memory**: Commands store state for undo
❌ **Over-engineering**: Can be overkill for simple cases

---

## Command vs Other Patterns

### Command vs Strategy

- **Command**: Encapsulates request (what to do)
- **Strategy**: Encapsulates algorithm (how to do)

### Command vs Memento

- **Command**: Stores how to undo operation
- **Memento**: Stores state snapshot

### Command vs Chain of Responsibility

- **Command**: Single command executed
- **Chain**: Multiple handlers process request

---

## Best Practices

### 1. Store State for Undo

```python
class Command:
    def execute(self):
        # Store state before change
        self.old_state = self.receiver.get_state()
        # Make change
        self.receiver.do_something()
    
    def undo(self):
        # Restore old state
        self.receiver.set_state(self.old_state)
```

### 2. Use Command Manager

```python
class CommandManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []
    
    def execute(self, command):
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()
```

### 3. Implement NoOp Command

```python
class NoOpCommand(Command):
    def execute(self):
        pass
    
    def undo(self):
        pass
```

### 4. Command Validation

```python
class Command:
    def can_execute(self) -> bool:
        # Validate if command can be executed
        return True
    
    def execute(self):
        if not self.can_execute():
            raise ValueError("Cannot execute command")
        # Execute command
```

### 5. Command Logging

```python
class LoggingCommand(Command):
    def __init__(self, command: Command, logger):
        self.command = command
        self.logger = logger
    
    def execute(self):
        self.logger.log(f"Executing: {self.command}")
        self.command.execute()
```

---

## Python-Specific Considerations

### 1. Using `functools.partial`

```python
from functools import partial

def execute_command(func, *args, **kwargs):
    func(*args, **kwargs)

# Create commands
command1 = partial(execute_command, light.on)
command2 = partial(execute_command, light.off)
```

### 2. Using `callable` Protocol

```python
from typing import Protocol

class Command(Protocol):
    def __call__(self) -> None:
        ...

def execute(cmd: Command):
    cmd()
```

### 3. Command as Functions

```python
# Commands as functions
def light_on():
    light.on()

def light_off():
    light.off()

# Store in list
commands = [light_on, light_off]
for cmd in commands:
    cmd()
```

### 4. Using `__call__`

```python
class Command:
    def __call__(self):
        self.execute()

# Can be called directly
command = LightOnCommand(light)
command()  # Calls __call__, which calls execute()
```

---

## Common Pitfalls

### 1. Not Storing State for Undo

```python
# Bad: Can't undo
class Command:
    def execute(self):
        self.receiver.do_something()
    
    def undo(self):
        # How to undo? No state stored!
        pass

# Good: Store state
class Command:
    def execute(self):
        self.old_state = self.receiver.get_state()
        self.receiver.do_something()
    
    def undo(self):
        self.receiver.set_state(self.old_state)
```

### 2. Memory Leaks with Commands

```python
# Bad: Commands never cleared
class Manager:
    def __init__(self):
        self.undo_stack = []  # Grows indefinitely

# Good: Limit stack size
class Manager:
    def __init__(self, max_size=100):
        self.undo_stack = []
        self.max_size = max_size
    
    def add_command(self, command):
        self.undo_stack.append(command)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
```

### 3. Commands Too Complex

```python
# Bad: Command does too much
class Command:
    def execute(self):
        # Too many responsibilities
        pass

# Good: Single responsibility
class Command:
    def execute(self):
        # One clear action
        pass
```

---

## Key Takeaways

- **Purpose**: Encapsulate requests as objects
- **Use when**: Need undo/redo, queue operations, logging, decoupling
- **Benefits**: Decoupling, undo/redo, queueing, logging, macro commands
- **Trade-off**: Adds complexity, many command classes
- **Python**: Can use functions, `__call__`, `functools.partial`
- **Best practice**: Store state for undo, use command manager, validate commands
- **Common use**: Text editors, undo/redo, job queues, transaction systems

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Command Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/command)

