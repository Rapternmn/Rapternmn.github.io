+++
title = "State Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 14
description = "Comprehensive guide to State Pattern: Allowing objects to alter behavior when internal state changes, with Python implementations, state machines, use cases, and best practices."
+++

---

## Introduction

The **State Pattern** is a behavioral design pattern that allows an object to alter its behavior when its internal state changes. The object will appear to change its class.

### Intent

- Allow object to change behavior based on state
- Encapsulate state-specific behavior
- Make state transitions explicit
- Eliminate large conditional statements

---

## Problem

Sometimes objects have complex state-dependent behavior:
- Many conditional statements based on state
- State transitions are scattered
- Adding new states requires modifying existing code
- State logic is hard to maintain

### Example Problem

```python
class Document:
    def __init__(self):
        self.state = "draft"  # draft, moderation, published
    
    def publish(self):
        if self.state == "draft":
            self.state = "moderation"
            print("Document sent to moderation")
        elif self.state == "moderation":
            self.state = "published"
            print("Document published")
        elif self.state == "published":
            print("Document already published")
        # Problem: Adding new states requires modifying this method!
    
    def approve(self):
        if self.state == "moderation":
            self.state = "published"
            print("Document approved and published")
        else:
            print("Can only approve documents in moderation")
```

---

## Solution

The State Pattern solves this by:
1. Creating state classes that encapsulate state-specific behavior
2. Context delegates behavior to current state
3. States handle their own transitions
4. New states can be added without modifying existing code

---

## Structure

```
┌──────────────┐
│   Context    │
├──────────────┤
│ - state      │
├──────────────┤
│ + request()  │
└──────┬───────┘
       │
       │ delegates
       ▼
┌──────────────┐
│    State     │
│  (interface)│
├──────────────┤
│ + handle()   │
└──────┬───────┘
       ▲
       │
┌──────┴───────┐
│ Concrete     │
│ States       │
└──────────────┘
```

**Participants**:
- **Context**: Maintains reference to current state
- **State**: Interface for state-specific behavior
- **ConcreteState**: Implements behavior for specific state

---

## Implementation

### Basic State Pattern

```python
from abc import ABC, abstractmethod

class State(ABC):
    @abstractmethod
    def handle(self, context):
        pass

class DraftState(State):
    def handle(self, context):
        print("Document is in draft state")
        # Can transition to moderation
        context.set_state(ModerationState())
    
    def publish(self, context):
        print("Sending document to moderation...")
        context.set_state(ModerationState())

class ModerationState(State):
    def handle(self, context):
        print("Document is in moderation")
    
    def approve(self, context):
        print("Document approved, publishing...")
        context.set_state(PublishedState())
    
    def reject(self, context):
        print("Document rejected, returning to draft...")
        context.set_state(DraftState())

class PublishedState(State):
    def handle(self, context):
        print("Document is published")
    
    def archive(self, context):
        print("Archiving document...")
        context.set_state(ArchivedState())

class ArchivedState(State):
    def handle(self, context):
        print("Document is archived")

class Document:
    def __init__(self):
        self._state = DraftState()
    
    def set_state(self, state: State):
        self._state = state
    
    def publish(self):
        if hasattr(self._state, 'publish'):
            self._state.publish(self)
        else:
            print("Cannot publish from current state")
    
    def approve(self):
        if hasattr(self._state, 'approve'):
            self._state.approve(self)
        else:
            print("Cannot approve from current state")
    
    def reject(self):
        if hasattr(self._state, 'reject'):
            self._state.reject(self)
        else:
            print("Cannot reject from current state")
    
    def archive(self):
        if hasattr(self._state, 'archive'):
            self._state.archive(self)
        else:
            print("Cannot archive from current state")

# Usage
doc = Document()
doc.publish()      # Sending document to moderation...
doc.approve()      # Document approved, publishing...
doc.archive()      # Archiving document...
```

### State Pattern with State Machine

```python
from abc import ABC, abstractmethod
from enum import Enum

class State(Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    FAILED = "failed"

class StateHandler(ABC):
    @abstractmethod
    def handle(self, context, event: str):
        pass

class LockedState(StateHandler):
    def handle(self, context, event: str):
        if event == "coin":
            print("Unlocking...")
            context.set_state(State.UNLOCKED)
            context.state_handler = UnlockedState()
        elif event == "push":
            print("Turnstile is locked. Insert coin.")
        else:
            print("Invalid event in locked state")

class UnlockedState(StateHandler):
    def handle(self, context, event: str):
        if event == "push":
            print("Opening turnstile...")
            context.set_state(State.LOCKED)
            context.state_handler = LockedState()
        elif event == "coin":
            print("Already unlocked. Thank you!")
        else:
            print("Invalid event in unlocked state")

class Turnstile:
    def __init__(self):
        self._state = State.LOCKED
        self.state_handler = LockedState()
    
    def set_state(self, state: State):
        self._state = state
    
    def event(self, event: str):
        self.state_handler.handle(self, event)

# Usage
turnstile = Turnstile()
turnstile.event("push")  # Turnstile is locked. Insert coin.
turnstile.event("coin")  # Unlocking...
turnstile.event("push")  # Opening turnstile...
```

---

## Real-World Examples

### Example 1: Vending Machine States

```python
from abc import ABC, abstractmethod

class VendingMachineState(ABC):
    @abstractmethod
    def insert_coin(self, machine, amount: float):
        pass
    
    @abstractmethod
    def select_product(self, machine, product: str):
        pass
    
    @abstractmethod
    def dispense(self, machine):
        pass

class IdleState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        machine.add_balance(amount)
        print(f"Balance: ${machine.balance}")
        machine.set_state(HasMoneyState())
    
    def select_product(self, machine, product: str):
        print("Please insert money first")
    
    def dispense(self, machine):
        print("No product selected")

class HasMoneyState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        machine.add_balance(amount)
        print(f"Balance: ${machine.balance}")
    
    def select_product(self, machine, product: str):
        price = machine.get_price(product)
        if machine.balance >= price:
            machine.selected_product = product
            machine.deduct_balance(price)
            machine.set_state(DispensingState())
        else:
            print(f"Insufficient balance. Need ${price}, have ${machine.balance}")
    
    def dispense(self, machine):
        print("Please select a product first")

class DispensingState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        print("Please wait, dispensing product...")
    
    def select_product(self, machine, product: str):
        print("Please wait, dispensing product...")
    
    def dispense(self, machine):
        print(f"Dispensing {machine.selected_product}...")
        machine.selected_product = None
        if machine.balance > 0:
            print(f"Returning change: ${machine.balance}")
            machine.balance = 0
        machine.set_state(IdleState())

class VendingMachine:
    def __init__(self):
        self._state = IdleState()
        self.balance = 0.0
        self.selected_product = None
        self.products = {
            "cola": 1.50,
            "chips": 1.00,
            "candy": 0.75
        }
    
    def set_state(self, state: VendingMachineState):
        self._state = state
    
    def add_balance(self, amount: float):
        self.balance += amount
    
    def deduct_balance(self, amount: float):
        self.balance -= amount
    
    def get_price(self, product: str) -> float:
        return self.products.get(product, 0.0)
    
    def insert_coin(self, amount: float):
        self._state.insert_coin(self, amount)
    
    def select_product(self, product: str):
        self._state.select_product(self, product)
    
    def dispense(self):
        self._state.dispense(self)

# Usage
machine = VendingMachine()
machine.insert_coin(2.0)      # Balance: $2.0
machine.select_product("cola")  # Dispensing cola...
machine.dispense()            # Dispensing cola... Returning change: $0.5
```

### Example 2: Media Player States

```python
from abc import ABC, abstractmethod

class MediaPlayerState(ABC):
    @abstractmethod
    def play(self, player):
        pass
    
    @abstractmethod
    def pause(self, player):
        pass
    
    @abstractmethod
    def stop(self, player):
        pass

class StoppedState(MediaPlayerState):
    def play(self, player):
        print("Starting playback...")
        player.set_state(PlayingState())
    
    def pause(self, player):
        print("Cannot pause, player is stopped")
    
    def stop(self, player):
        print("Player is already stopped")

class PlayingState(MediaPlayerState):
    def play(self, player):
        print("Already playing")
    
    def pause(self, player):
        print("Pausing playback...")
        player.set_state(PausedState())
    
    def stop(self, player):
        print("Stopping playback...")
        player.set_state(StoppedState())

class PausedState(MediaPlayerState):
    def play(self, player):
        print("Resuming playback...")
        player.set_state(PlayingState())
    
    def pause(self, player):
        print("Already paused")
    
    def stop(self, player):
        print("Stopping playback...")
        player.set_state(StoppedState())

class MediaPlayer:
    def __init__(self):
        self._state = StoppedState()
    
    def set_state(self, state: MediaPlayerState):
        self._state = state
    
    def play(self):
        self._state.play(self)
    
    def pause(self):
        self._state.pause(self)
    
    def stop(self):
        self._state.stop(self)

# Usage
player = MediaPlayer()
player.play()   # Starting playback...
player.pause()  # Pausing playback...
player.play()   # Resuming playback...
player.stop()   # Stopping playback...
```

---

## Use Cases

### When to Use State Pattern

✅ **State-Dependent Behavior**: When object behavior depends on state
✅ **Many Conditionals**: When you have many if/else based on state
✅ **State Transitions**: When state transitions are complex
✅ **State Machines**: When implementing state machines
✅ **Avoid Conditionals**: When you want to eliminate conditionals

### When NOT to Use

❌ **Simple States**: Overkill for simple state-dependent behavior
❌ **Few States**: If you only have 2-3 states
❌ **No State Transitions**: When states don't change
❌ **Performance Critical**: Adds indirection overhead

---

## Pros and Cons

### Advantages

✅ **Eliminates Conditionals**: Replaces if/else with polymorphism
✅ **Encapsulation**: Each state encapsulates its behavior
✅ **Open/Closed**: Easy to add new states
✅ **Clear Transitions**: State transitions are explicit
✅ **Single Responsibility**: Each state has one responsibility

### Disadvantages

❌ **Complexity**: Adds many state classes
❌ **Overhead**: Adds indirection layer
❌ **State Explosion**: Can lead to many state classes
❌ **Over-engineering**: Can be overkill for simple cases

---

## State vs Other Patterns

### State vs Strategy

- **State**: Behavior changes based on internal state
- **Strategy**: Algorithm selection (external)

### State vs Command

- **State**: Object's behavior changes with state
- **Command**: Encapsulates request

### State vs Flyweight

- **State**: Each context has its own state
- **Flyweight**: States can be shared

---

## Best Practices

### 1. Use Enums for State Types

```python
from enum import Enum

class StateType(Enum):
    DRAFT = "draft"
    MODERATION = "moderation"
    PUBLISHED = "published"
```

### 2. State Factory

```python
class StateFactory:
    @staticmethod
    def create_state(state_type: StateType):
        states = {
            StateType.DRAFT: DraftState(),
            StateType.MODERATION: ModerationState(),
            StateType.PUBLISHED: PublishedState()
        }
        return states[state_type]
```

### 3. Validate State Transitions

```python
class State:
    def __init__(self, allowed_transitions: list):
        self.allowed_transitions = allowed_transitions
    
    def can_transition_to(self, new_state):
        return new_state in self.allowed_transitions
```

### 4. State Entry/Exit Actions

```python
class State:
    def enter(self, context):
        # Actions when entering state
        pass
    
    def exit(self, context):
        # Actions when exiting state
        pass
```

### 5. State History

```python
class Context:
    def __init__(self):
        self._state = None
        self._state_history = []
    
    def set_state(self, state):
        if self._state:
            self._state.exit(self)
            self._state_history.append(self._state)
        self._state = state
        self._state.enter(self)
```

---

## Python-Specific Considerations

### 1. Using `enum.Enum`

```python
from enum import Enum, auto

class State(Enum):
    DRAFT = auto()
    MODERATION = auto()
    PUBLISHED = auto()
```

### 2. State as Dataclass

```python
from dataclasses import dataclass

@dataclass
class State:
    name: str
    allowed_actions: list
```

### 3. State with `__call__`

```python
class State:
    def __call__(self, context, action):
        return self.handle(context, action)
```

### 4. State Machine Library

```python
# Consider using state machine libraries
# e.g., transitions, python-statemachine
```

---

## Common Pitfalls

### 1. State Classes Too Large

```python
# Bad: State does too much
class State:
    def handle(self, context):
        # Too many responsibilities
        pass

# Good: Focused state
class State:
    def handle(self, context):
        # Single responsibility
        pass
```

### 2. Not Validating Transitions

```python
# Bad: Any state can transition to any state
def set_state(self, state):
    self._state = state

# Good: Validate transitions
def set_state(self, state):
    if self._state.can_transition_to(state):
        self._state = state
    else:
        raise ValueError("Invalid state transition")
```

### 3. Circular Dependencies

```python
# Bad: States reference each other
class StateA:
    def __init__(self):
        self.next_state = StateB()  # Circular

# Good: Use factory or context
class StateA:
    def transition(self, context):
        context.set_state(StateB())
```

---

## Key Takeaways

- **Purpose**: Allow object to change behavior based on state
- **Use when**: State-dependent behavior, many conditionals, state machines
- **Benefits**: Eliminates conditionals, encapsulation, easy to extend
- **Trade-off**: Adds complexity, many state classes
- **Python**: Use enums, dataclasses, state machine libraries
- **Best practice**: Validate transitions, use state factory, entry/exit actions
- **Common use**: State machines, workflow systems, game states

---

## References

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612) - Gang of Four
- [Python enum](https://docs.python.org/3/library/enum.html)
- [State Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/state)

