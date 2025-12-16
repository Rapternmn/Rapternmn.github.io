+++
title = "Car Rental System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 9
description = "Complete implementation of Car Rental System: Vehicle inventory, booking management, payment processing, and rental tracking."
+++

---

## Problem Statement

Design a car rental system that can:
- Manage vehicle inventory
- Handle bookings
- Process payments
- Track rentals
- Calculate rental fees
- Manage vehicle maintenance

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Optional
import threading

class VehicleType(Enum):
    SEDAN = "sedan"
    SUV = "suv"
    LUXURY = "luxury"
    SPORTS = "sports"

class VehicleStatus(Enum):
    AVAILABLE = "available"
    RENTED = "rented"
    MAINTENANCE = "maintenance"

class Vehicle:
    def __init__(self, vehicle_id: str, make: str, model: str, vehicle_type: VehicleType, daily_rate: float):
        self.vehicle_id = vehicle_id
        self.make = make
        self.model = model
        self.vehicle_type = vehicle_type
        self.daily_rate = daily_rate
        self.status = VehicleStatus.AVAILABLE
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        return self.status == VehicleStatus.AVAILABLE
    
    def rent(self):
        with self._lock:
            if self.status == VehicleStatus.AVAILABLE:
                self.status = VehicleStatus.RENTED
                return True
            return False
    
    def return_vehicle(self):
        with self._lock:
            self.status = VehicleStatus.AVAILABLE

class Customer:
    def __init__(self, customer_id: str, name: str, email: str, license_number: str):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.license_number = license_number
        self.rentals: List[Rental] = []

class Rental:
    def __init__(self, rental_id: str, vehicle: Vehicle, customer: Customer, start_date: datetime, end_date: datetime):
        self.rental_id = rental_id
        self.vehicle = vehicle
        self.customer = customer
        self.start_date = start_date
        self.end_date = end_date
        self.actual_return_date: Optional[datetime] = None
        self.total_cost: float = 0.0
        self.is_paid = False
    
    def calculate_cost(self) -> float:
        days = (self.end_date - self.start_date).days
        return days * self.vehicle.daily_rate
    
    def calculate_late_fee(self, daily_late_fee: float = 50.0) -> float:
        if self.actual_return_date and self.actual_return_date > self.end_date:
            days_late = (self.actual_return_date - self.end_date).days
            return days_late * daily_late_fee
        return 0.0

class Payment(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool:
        pass

class CreditCardPayment(Payment):
    def process(self, amount: float) -> bool:
        print(f"Processing credit card payment: ${amount}")
        return True

class CarRentalSystem:
    def __init__(self):
        self.vehicles: dict = {}  # vehicle_id -> Vehicle
        self.customers: dict = {}  # customer_id -> Customer
        self.rentals: dict = {}  # rental_id -> Rental
        self._lock = threading.Lock()
    
    def add_vehicle(self, vehicle: Vehicle):
        with self._lock:
            self.vehicles[vehicle.vehicle_id] = vehicle
    
    def register_customer(self, customer: Customer):
        with self._lock:
            self.customers[customer.customer_id] = customer
    
    def search_vehicles(self, vehicle_type: Optional[VehicleType] = None) -> List[Vehicle]:
        results = []
        for vehicle in self.vehicles.values():
            if vehicle.is_available():
                if vehicle_type is None or vehicle.vehicle_type == vehicle_type:
                    results.append(vehicle)
        return results
    
    def create_rental(self, vehicle_id: str, customer_id: str, start_date: datetime, end_date: datetime) -> Optional[Rental]:
        with self._lock:
            if vehicle_id not in self.vehicles or customer_id not in self.customers:
                return None
            
            vehicle = self.vehicles[vehicle_id]
            customer = self.customers[customer_id]
            
            if not vehicle.rent():
                return None
            
            rental_id = f"R{len(self.rentals) + 1:04d}"
            rental = Rental(rental_id, vehicle, customer, start_date, end_date)
            rental.total_cost = rental.calculate_cost()
            
            self.rentals[rental_id] = rental
            customer.rentals.append(rental)
            return rental
    
    def return_vehicle(self, rental_id: str, actual_return_date: datetime, payment: Payment) -> bool:
        with self._lock:
            if rental_id not in self.rentals:
                return False
            
            rental = self.rentals[rental_id]
            rental.actual_return_date = actual_return_date
            
            late_fee = rental.calculate_late_fee()
            total_amount = rental.total_cost + late_fee
            
            if payment.process(total_amount):
                rental.is_paid = True
                rental.vehicle.return_vehicle()
                print(f"Vehicle returned. Total: ${total_amount:.2f}")
                return True
            return False

# Usage
system = CarRentalSystem()

vehicle = Vehicle("V001", "Toyota", "Camry", VehicleType.SEDAN, 50.0)
customer = Customer("C001", "John", "john@example.com", "LIC123")

system.add_vehicle(vehicle)
system.register_customer(customer)

start = datetime.now()
end = start + timedelta(days=3)

rental = system.create_rental("V001", "C001", start, end)
if rental:
    payment = CreditCardPayment()
    system.return_vehicle(rental.rental_id, datetime.now(), payment)
```

---

## Design Patterns Used

1. **[Factory Pattern]({{< ref "../design-patterns/4-Factory_Pattern.md" >}})**: `Vehicle` objects can be created using Factory pattern based on vehicle type
2. **[Strategy Pattern]({{< ref "../design-patterns/12-Strategy_Pattern.md" >}})**: `Payment` interface with implementations (`CreditCardPayment`, `CashPayment`, etc.) allows different payment strategies
3. **[Template Method Pattern]({{< ref "../design-patterns/15-Template_Method_Pattern.md" >}})**: Rental processing follows a template structure with steps: create rental, calculate cost, process payment, return vehicle

---

## Key Points

- **Time Complexity**: O(V) for search, O(1) for rental operations
- **Space Complexity**: O(V + C + R)
- **Features**: Late fee calculation, payment processing, maintenance tracking

---

## Practice Problems

- Add support for insurance
- Implement vehicle maintenance scheduling
- Add loyalty program
- Support vehicle upgrades

