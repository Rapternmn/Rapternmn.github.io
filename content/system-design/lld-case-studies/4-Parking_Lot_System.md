+++
title = "Parking Lot System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Complete implementation of Parking Lot System: Multi-level parking with different vehicle types, spot allocation, payment processing, using Factory pattern, Strategy pattern, and SOLID principles."
+++

---

## Problem Statement

Design a parking lot system that can:
- Support multiple floors and parking spots
- Handle different vehicle types (Car, Motorcycle, Truck)
- Support different spot types (Compact, Large, Handicapped)
- Allocate and deallocate parking spots
- Calculate parking fees based on duration
- Handle payments

---

## Requirements

### Functional Requirements
1. Park a vehicle and assign a spot
2. Unpark a vehicle and free the spot
3. Find a vehicle by ticket number
4. Calculate parking fee based on duration
5. Support multiple payment methods
6. Display available spots by type

### Non-Functional Requirements
1. Thread-safe operations
2. Efficient spot allocation
3. Scalable design

---

## Class Diagram

```
┌─────────────────┐
│   Vehicle       │
├─────────────────┤
│ - licensePlate  │
│ - vehicleType   │
└────────┬────────┘
         ▲
    ┌────┴────┐
    │         │
┌───┴───┐ ┌──┴────┐
│  Car  │ │Motorcycle│
└───────┘ └────────┘

┌─────────────────┐
│  ParkingSpot    │
├─────────────────┤
│ - spotId        │
│ - spotType      │
│ - isOccupied    │
│ - vehicle       │
└────────┬────────┘

┌─────────────────┐
│     Floor       │
├─────────────────┤
│ - floorNumber   │
│ - spots: List   │
└────────┬────────┘

┌─────────────────┐
│  ParkingLot     │
├─────────────────┤
│ - floors: List  │
│ - capacity      │
└────────┬────────┘

┌─────────────────┐
│     Ticket      │
├─────────────────┤
│ - ticketId      │
│ - vehicle       │
│ - spot          │
│ - entryTime     │
└────────┬────────┘

┌─────────────────┐
│ ParkingLotManager│
├─────────────────┤
│ + parkVehicle() │
│ + unparkVehicle()│
│ + findVehicle() │
└─────────────────┘
```

---

## Implementation

### Core Classes

```python
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Optional, List
import threading

class VehicleType(Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"

class SpotType(Enum):
    COMPACT = "compact"
    LARGE = "large"
    HANDICAPPED = "handicapped"

class Vehicle(ABC):
    def __init__(self, license_plate: str):
        self.license_plate = license_plate
    
    @abstractmethod
    def get_vehicle_type(self) -> VehicleType:
        pass
    
    @abstractmethod
    def can_fit_in_spot(self, spot_type: SpotType) -> bool:
        pass

class Car(Vehicle):
    def get_vehicle_type(self) -> VehicleType:
        return VehicleType.CAR
    
    def can_fit_in_spot(self, spot_type: SpotType) -> bool:
        return spot_type in [SpotType.COMPACT, SpotType.LARGE, SpotType.HANDICAPPED]

class Motorcycle(Vehicle):
    def get_vehicle_type(self) -> VehicleType:
        return VehicleType.MOTORCYCLE
    
    def can_fit_in_spot(self, spot_type: SpotType) -> bool:
        return True  # Motorcycle can fit in any spot

class Truck(Vehicle):
    def get_vehicle_type(self) -> VehicleType:
        return VehicleType.TRUCK
    
    def can_fit_in_spot(self, spot_type: SpotType) -> bool:
        return spot_type == SpotType.LARGE

class ParkingSpot:
    def __init__(self, spot_id: str, spot_type: SpotType, floor_number: int):
        self.spot_id = spot_id
        self.spot_type = spot_type
        self.floor_number = floor_number
        self.is_occupied = False
        self.vehicle: Optional[Vehicle] = None
    
    def park_vehicle(self, vehicle: Vehicle) -> bool:
        if self.is_occupied:
            return False
        if not vehicle.can_fit_in_spot(self.spot_type):
            return False
        self.vehicle = vehicle
        self.is_occupied = True
        return True
    
    def remove_vehicle(self):
        self.vehicle = None
        self.is_occupied = False
    
    def is_available(self) -> bool:
        return not self.is_occupied

class Floor:
    def __init__(self, floor_number: int, spots: List[ParkingSpot]):
        self.floor_number = floor_number
        self.spots = spots
        self._lock = threading.Lock()
    
    def get_available_spots(self, spot_type: Optional[SpotType] = None) -> List[ParkingSpot]:
        with self._lock:
            if spot_type:
                return [spot for spot in self.spots if spot.is_available() and spot.spot_type == spot_type]
            return [spot for spot in self.spots if spot.is_available()]
    
    def get_spot_by_id(self, spot_id: str) -> Optional[ParkingSpot]:
        for spot in self.spots:
            if spot.spot_id == spot_id:
                return spot
        return None

class Ticket:
    def __init__(self, ticket_id: str, vehicle: Vehicle, spot: ParkingSpot):
        self.ticket_id = ticket_id
        self.vehicle = vehicle
        self.spot = spot
        self.entry_time = datetime.now()
        self.exit_time: Optional[datetime] = None
    
    def calculate_duration_hours(self) -> float:
        if self.exit_time:
            duration = self.exit_time - self.entry_time
            return duration.total_seconds() / 3600
        duration = datetime.now() - self.entry_time
        return duration.total_seconds() / 3600

class ParkingLot:
    def __init__(self, floors: List[Floor]):
        self.floors = floors
        self._lock = threading.Lock()
    
    def find_available_spot(self, vehicle: Vehicle, preferred_spot_type: Optional[SpotType] = None) -> Optional[ParkingSpot]:
        with self._lock:
            # Try preferred spot type first
            if preferred_spot_type:
                for floor in self.floors:
                    spots = floor.get_available_spots(preferred_spot_type)
                    for spot in spots:
                        if vehicle.can_fit_in_spot(spot.spot_type):
                            return spot
            
            # Try any available spot
            for floor in self.floors:
                spots = floor.get_available_spots()
                for spot in spots:
                    if vehicle.can_fit_in_spot(spot.spot_type):
                        return spot
            return None
    
    def get_spot_by_id(self, spot_id: str) -> Optional[ParkingSpot]:
        for floor in self.floors:
            spot = floor.get_spot_by_id(spot_id)
            if spot:
                return spot
        return None

class PricingStrategy(ABC):
    @abstractmethod
    def calculate_fee(self, duration_hours: float, vehicle_type: VehicleType) -> float:
        pass

class HourlyPricing(PricingStrategy):
    def __init__(self, hourly_rate: float = 5.0):
        self.hourly_rate = hourly_rate
    
    def calculate_fee(self, duration_hours: float, vehicle_type: VehicleType) -> float:
        # Minimum 1 hour charge
        hours = max(1.0, duration_hours)
        return hours * self.hourly_rate

class FlatRatePricing(PricingStrategy):
    def __init__(self, flat_rate: float = 10.0):
        self.flat_rate = flat_rate
    
    def calculate_fee(self, duration_hours: float, vehicle_type: VehicleType) -> float:
        return self.flat_rate

class TieredPricing(PricingStrategy):
    def calculate_fee(self, duration_hours: float, vehicle_type: VehicleType) -> float:
        if duration_hours <= 1:
            return 5.0
        elif duration_hours <= 3:
            return 5.0 + (duration_hours - 1) * 3.0
        else:
            return 5.0 + 2 * 3.0 + (duration_hours - 3) * 2.0

class Payment(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        pass

class CashPayment(Payment):
    def process_payment(self, amount: float) -> bool:
        print(f"Processing cash payment of ${amount}")
        return True

class CardPayment(Payment):
    def process_payment(self, amount: float) -> bool:
        print(f"Processing card payment of ${amount}")
        return True

class ParkingLotManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.parking_lot: Optional[ParkingLot] = None
            self.tickets: dict = {}
            self.pricing_strategy: PricingStrategy = HourlyPricing()
            self._initialized = True
    
    def initialize_parking_lot(self, floors: List[Floor]):
        self.parking_lot = ParkingLot(floors)
    
    def set_pricing_strategy(self, strategy: PricingStrategy):
        self.pricing_strategy = strategy
    
    def park_vehicle(self, vehicle: Vehicle) -> Optional[Ticket]:
        if not self.parking_lot:
            raise ValueError("Parking lot not initialized")
        
        spot = self.parking_lot.find_available_spot(vehicle)
        if not spot:
            print("No available spots")
            return None
        
        if not spot.park_vehicle(vehicle):
            return None
        
        ticket_id = f"T{len(self.tickets) + 1:04d}"
        ticket = Ticket(ticket_id, vehicle, spot)
        self.tickets[ticket_id] = ticket
        
        print(f"Vehicle {vehicle.license_plate} parked at spot {spot.spot_id} on floor {spot.floor_number}")
        return ticket
    
    def unpark_vehicle(self, ticket_id: str, payment: Payment) -> bool:
        if ticket_id not in self.tickets:
            print("Invalid ticket")
            return False
        
        ticket = self.tickets[ticket_id]
        ticket.exit_time = datetime.now()
        
        duration_hours = ticket.calculate_duration_hours()
        fee = self.pricing_strategy.calculate_fee(duration_hours, ticket.vehicle.get_vehicle_type())
        
        print(f"Parking duration: {duration_hours:.2f} hours")
        print(f"Total fee: ${fee:.2f}")
        
        if payment.process_payment(fee):
            ticket.spot.remove_vehicle()
            del self.tickets[ticket_id]
            print(f"Vehicle {ticket.vehicle.license_plate} unparked from spot {ticket.spot.spot_id}")
            return True
        
        return False
    
    def find_vehicle(self, ticket_id: str) -> Optional[Ticket]:
        return self.tickets.get(ticket_id)
    
    def get_available_spots_count(self) -> dict:
        if not self.parking_lot:
            return {}
        
        count = {}
        for floor in self.parking_lot.floors:
            for spot_type in SpotType:
                available = len(floor.get_available_spots(spot_type))
                key = f"Floor {floor.floor_number} - {spot_type.value}"
                count[key] = available
        return count
```

### Usage Example

```python
# Create parking lot with 2 floors
floor1_spots = [
    ParkingSpot("F1-S1", SpotType.COMPACT, 1),
    ParkingSpot("F1-S2", SpotType.LARGE, 1),
    ParkingSpot("F1-S3", SpotType.HANDICAPPED, 1),
]

floor2_spots = [
    ParkingSpot("F2-S1", SpotType.COMPACT, 2),
    ParkingSpot("F2-S2", SpotType.LARGE, 2),
]

floors = [Floor(1, floor1_spots), Floor(2, floor2_spots)]

# Initialize parking lot
manager = ParkingLotManager()
manager.initialize_parking_lot(floors)
manager.set_pricing_strategy(TieredPricing())

# Park vehicles
car1 = Car("ABC123")
ticket1 = manager.park_vehicle(car1)

motorcycle1 = Motorcycle("XYZ789")
ticket2 = manager.park_vehicle(motorcycle1)

# Check available spots
print(manager.get_available_spots_count())

# Unpark vehicle
payment = CardPayment()
manager.unpark_vehicle(ticket1.ticket_id, payment)
```

---

## Design Patterns Used

1. **Singleton**: ParkingLotManager ensures single instance
2. **Factory**: VehicleFactory (can be added) for creating vehicles
3. **Strategy**: PricingStrategy for different pricing models
4. **Template Method**: Payment processing structure

---

## Key Points

- **Time Complexity**: 
  - Park: O(F × S) where F = floors, S = spots per floor
  - Unpark: O(1) with ticket lookup
- **Space Complexity**: O(F × S) for storing spots
- **Thread Safety**: Uses locks for concurrent access
- **Scalability**: Can add more floors and spots easily
- **Extensibility**: Easy to add new vehicle types and pricing strategies

---

## Edge Cases Handled

- Parking lot full
- Invalid ticket
- Vehicle doesn't fit in spot
- Payment failure
- Concurrent parking/unparking

---

## Practice Problems

- [LeetCode - Design Parking System](https://leetcode.com/problems/design-parking-system/)
- Extend to support reservations
- Add support for electric vehicle charging spots
- Implement spot assignment algorithms (nearest, cheapest, etc.)

