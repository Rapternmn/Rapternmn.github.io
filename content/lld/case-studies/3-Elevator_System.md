+++
title = "Elevator System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Complete implementation of Elevator System: Multiple elevators, floor requests, scheduling algorithms, state management using State pattern, and elevator control logic."
+++

---

## Problem Statement

Design an elevator control system that can:
- Handle multiple elevators
- Process floor requests from inside and outside elevators
- Implement efficient scheduling algorithms
- Manage elevator states (Idle, Moving, Stopped)
- Handle direction (Up, Down)
- Display current floor and direction

---

## Requirements

### Functional Requirements
1. Request elevator from a floor
2. Select destination floor from inside elevator
3. Efficient scheduling of multiple elevators
4. Handle elevator states and direction
5. Display elevator status

### Non-Functional Requirements
1. Thread-safe operations
2. Efficient request handling
3. Minimize waiting time

---

## Class Diagram

```
┌──────────────┐
│   Elevator   │
├──────────────┤
│ - elevatorId │
│ - currentFloor│
│ - direction  │
│ - state      │
│ - requests   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ElevatorState│
└──────┬───────┘
       ▲
   ┌───┴───┐
   │       │
┌──┴──┐ ┌──┴──┐
│Idle │ │Moving│
└─────┘ └─────┘

┌──────────────┐
│   Request    │
├──────────────┤
│ - floor      │
│ - direction  │
└──────────────┘

┌──────────────┐
│ElevatorController│
├──────────────┤
│ - elevators  │
│ + requestElevator()│
│ + selectFloor()│
└──────────────┘
```

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
import threading
from queue import PriorityQueue

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    IDLE = "idle"

class ElevatorState(ABC):
    @abstractmethod
    def handle_request(self, elevator, floor: int):
        pass
    
    @abstractmethod
    def move(self, elevator):
        pass

class IdleState(ElevatorState):
    def handle_request(self, elevator, floor: int):
        if floor > elevator.current_floor:
            elevator.direction = Direction.UP
        elif floor < elevator.current_floor:
            elevator.direction = Direction.DOWN
        elevator.add_request(floor)
        elevator.set_state(MovingState())
    
    def move(self, elevator):
        # Do nothing when idle
        pass

class MovingState(ElevatorState):
    def handle_request(self, elevator, floor: int):
        elevator.add_request(floor)
    
    def move(self, elevator):
        if not elevator.has_requests():
            elevator.set_state(IdleState())
            elevator.direction = Direction.IDLE
            return
        
        next_floor = elevator.get_next_floor()
        if next_floor == elevator.current_floor:
            elevator.stop_at_floor()
        elif next_floor > elevator.current_floor:
            elevator.current_floor += 1
            elevator.direction = Direction.UP
        else:
            elevator.current_floor -= 1
            elevator.direction = Direction.DOWN

class StoppedState(ElevatorState):
    def handle_request(self, elevator, floor: int):
        elevator.add_request(floor)
    
    def move(self, elevator):
        elevator.set_state(MovingState())

class Request:
    def __init__(self, floor: int, direction: Optional[Direction] = None):
        self.floor = floor
        self.direction = direction
        self.priority = abs(floor)  # For priority queue

class Elevator:
    def __init__(self, elevator_id: int, total_floors: int):
        self.elevator_id = elevator_id
        self.total_floors = total_floors
        self.current_floor = 0
        self.direction = Direction.IDLE
        self.state: ElevatorState = IdleState()
        self.up_requests = []  # Floors to visit going up
        self.down_requests = []  # Floors to visit going down
        self._lock = threading.Lock()
    
    def set_state(self, state: ElevatorState):
        self.state = state
    
    def add_request(self, floor: int):
        with self._lock:
            if floor == self.current_floor:
                return
            
            if floor > self.current_floor:
                if floor not in self.up_requests:
                    self.up_requests.append(floor)
                    self.up_requests.sort()
            else:
                if floor not in self.down_requests:
                    self.down_requests.append(floor)
                    self.down_requests.sort(reverse=True)
    
    def get_next_floor(self) -> Optional[int]:
        with self._lock:
            if self.direction == Direction.UP and self.up_requests:
                return self.up_requests[0]
            elif self.direction == Direction.DOWN and self.down_requests:
                return self.down_requests[0]
            elif self.direction == Direction.IDLE:
                if self.up_requests:
                    return self.up_requests[0]
                elif self.down_requests:
                    return self.down_requests[0]
            return None
    
    def has_requests(self) -> bool:
        return len(self.up_requests) > 0 or len(self.down_requests) > 0
    
    def stop_at_floor(self):
        with self._lock:
            if self.current_floor in self.up_requests:
                self.up_requests.remove(self.current_floor)
            if self.current_floor in self.down_requests:
                self.down_requests.remove(self.current_floor)
            
            print(f"Elevator {self.elevator_id} stopped at floor {self.current_floor}")
            self.set_state(StoppedState())
    
    def request_floor(self, floor: int):
        self.state.handle_request(self, floor)
    
    def move(self):
        self.state.move(self)
    
    def get_status(self) -> dict:
        return {
            "elevator_id": self.elevator_id,
            "current_floor": self.current_floor,
            "direction": self.direction.value,
            "state": self.state.__class__.__name__
        }

class ElevatorController:
    def __init__(self, elevators: List[Elevator]):
        self.elevators = elevators
        self._lock = threading.Lock()
    
    def request_elevator(self, floor: int, direction: Direction) -> int:
        """Request elevator from a floor, returns elevator ID"""
        best_elevator = self._find_best_elevator(floor, direction)
        best_elevator.request_floor(floor)
        return best_elevator.elevator_id
    
    def select_floor(self, elevator_id: int, floor: int):
        """Select destination floor from inside elevator"""
        elevator = self._get_elevator(elevator_id)
        if elevator:
            elevator.request_floor(floor)
    
    def _find_best_elevator(self, floor: int, direction: Direction) -> Elevator:
        """Find best elevator using nearest elevator algorithm"""
        best_elevator = None
        min_distance = float('inf')
        
        for elevator in self.elevators:
            distance = abs(elevator.current_floor - floor)
            
            # Prefer idle elevators
            if elevator.direction == Direction.IDLE:
                if distance < min_distance:
                    min_distance = distance
                    best_elevator = elevator
            # Prefer elevators moving in same direction
            elif (elevator.direction == direction and
                  ((direction == Direction.UP and elevator.current_floor < floor) or
                   (direction == Direction.DOWN and elevator.current_floor > floor))):
                if distance < min_distance:
                    min_distance = distance
                    best_elevator = elevator
        
        # If no suitable elevator found, use nearest
        if not best_elevator:
            for elevator in self.elevators:
                distance = abs(elevator.current_floor - floor)
                if distance < min_distance:
                    min_distance = distance
                    best_elevator = elevator
        
        return best_elevator
    
    def _get_elevator(self, elevator_id: int) -> Optional[Elevator]:
        for elevator in self.elevators:
            if elevator.elevator_id == elevator_id:
                return elevator
        return None
    
    def get_all_status(self) -> List[dict]:
        return [elevator.get_status() for elevator in self.elevators]
    
    def simulate_step(self):
        """Simulate one step of elevator movement"""
        for elevator in self.elevators:
            elevator.move()

# Usage
elevator1 = Elevator(1, 10)
elevator2 = Elevator(2, 10)

controller = ElevatorController([elevator1, elevator2])

# Request elevator from floor 3 going up
controller.request_elevator(3, Direction.UP)

# Select floor 7 from inside elevator 1
controller.select_floor(1, 7)

# Simulate movement
for _ in range(10):
    controller.simulate_step()
    print(controller.get_all_status())
```

---

## Design Patterns Used

1. **State Pattern**: Elevator states (Idle, Moving, Stopped)
2. **Strategy Pattern**: Different scheduling algorithms
3. **Singleton**: ElevatorController (optional)

---

## Key Points

- **Time Complexity**: 
  - Request elevator: O(E) where E = number of elevators
  - Select floor: O(1)
- **Space Complexity**: O(R) where R = number of requests
- **Scheduling Algorithms**: Nearest elevator, SCAN, LOOK
- **Thread Safety**: Uses locks for concurrent requests

---

## Practice Problems

- Implement SCAN algorithm
- Add support for express elevators
- Implement elevator grouping by floors
- Add maintenance mode

