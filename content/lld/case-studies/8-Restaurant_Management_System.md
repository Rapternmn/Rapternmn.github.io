+++
title = "Restaurant Management System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Complete implementation of Restaurant Management System: Order management, table management, kitchen workflow, and payment processing."
+++

---

## Problem Statement

Design a restaurant management system that can:
- Manage tables and reservations
- Handle orders
- Track kitchen workflow
- Process payments
- Manage menu items
- Handle staff assignments

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from typing import List, Optional
import threading

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    SERVED = "served"
    CANCELLED = "cancelled"

class TableStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"

class MenuItem:
    def __init__(self, item_id: str, name: str, price: float, category: str):
        self.item_id = item_id
        self.name = name
        self.price = price
        self.category = category

class Table:
    def __init__(self, table_id: str, capacity: int):
        self.table_id = table_id
        self.capacity = capacity
        self.status = TableStatus.AVAILABLE
        self.current_order: Optional[Order] = None
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        return self.status == TableStatus.AVAILABLE
    
    def assign_order(self, order):
        with self._lock:
            if self.is_available():
                self.current_order = order
                self.status = TableStatus.OCCUPIED
                return True
            return False
    
    def free_table(self):
        with self._lock:
            self.current_order = None
            self.status = TableStatus.AVAILABLE

class OrderItem:
    def __init__(self, menu_item: MenuItem, quantity: int, special_instructions: str = ""):
        self.menu_item = menu_item
        self.quantity = quantity
        self.special_instructions = special_instructions
    
    def get_total(self) -> float:
        return self.menu_item.price * self.quantity

class Order:
    def __init__(self, order_id: str, table: Table, customer_name: str):
        self.order_id = order_id
        self.table = table
        self.customer_name = customer_name
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self.created_at = datetime.now()
        self._lock = threading.Lock()
    
    def add_item(self, item: OrderItem):
        with self._lock:
            self.items.append(item)
    
    def calculate_total(self) -> float:
        return sum(item.get_total() for item in self.items)
    
    def update_status(self, status: OrderStatus):
        with self._lock:
            self.status = status

class Kitchen:
    def __init__(self):
        self.orders: List[Order] = []
        self._lock = threading.Lock()
    
    def receive_order(self, order: Order):
        with self._lock:
            order.update_status(OrderStatus.CONFIRMED)
            self.orders.append(order)
            print(f"Kitchen received order {order.order_id}")
    
    def start_preparing(self, order_id: str):
        with self._lock:
            for order in self.orders:
                if order.order_id == order_id:
                    order.update_status(OrderStatus.PREPARING)
                    print(f"Kitchen preparing order {order_id}")
                    break
    
    def mark_ready(self, order_id: str):
        with self._lock:
            for order in self.orders:
                if order.order_id == order_id:
                    order.update_status(OrderStatus.READY)
                    print(f"Order {order_id} is ready")
                    break

class Payment(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool:
        pass

class CashPayment(Payment):
    def process(self, amount: float) -> bool:
        print(f"Processing cash payment: ${amount}")
        return True

class Restaurant:
    def __init__(self):
        self.tables: dict = {}  # table_id -> Table
        self.menu_items: dict = {}  # item_id -> MenuItem
        self.orders: dict = {}  # order_id -> Order
        self.kitchen = Kitchen()
        self._lock = threading.Lock()
    
    def add_table(self, table: Table):
        with self._lock:
            self.tables[table.table_id] = table
    
    def add_menu_item(self, item: MenuItem):
        with self._lock:
            self.menu_items[item.item_id] = item
    
    def create_order(self, table_id: str, customer_name: str) -> Optional[Order]:
        with self._lock:
            if table_id not in self.tables:
                return None
            
            table = self.tables[table_id]
            if not table.is_available():
                return None
            
            order_id = f"ORD{len(self.orders) + 1:04d}"
            order = Order(order_id, table, customer_name)
            
            if table.assign_order(order):
                self.orders[order_id] = order
                return order
            return None
    
    def add_item_to_order(self, order_id: str, item_id: str, quantity: int):
        if order_id in self.orders and item_id in self.menu_items:
            order = self.orders[order_id]
            menu_item = self.menu_items[item_id]
            order_item = OrderItem(menu_item, quantity)
            order.add_item(order_item)
    
    def place_order(self, order_id: str):
        if order_id in self.orders:
            order = self.orders[order_id]
            self.kitchen.receive_order(order)
    
    def serve_order(self, order_id: str, payment: Payment) -> bool:
        with self._lock:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            if order.status != OrderStatus.READY:
                return False
            
            total = order.calculate_total()
            if payment.process(total):
                order.update_status(OrderStatus.SERVED)
                order.table.free_table()
                print(f"Order {order_id} served. Total: ${total:.2f}")
                return True
            return False

# Usage
restaurant = Restaurant()

table1 = Table("T001", 4)
menu_item1 = MenuItem("I001", "Pizza", 12.99, "Main Course")

restaurant.add_table(table1)
restaurant.add_menu_item(menu_item1)

order = restaurant.create_order("T001", "John")
restaurant.add_item_to_order(order.order_id, "I001", 2)
restaurant.place_order(order.order_id)
restaurant.kitchen.start_preparing(order.order_id)
restaurant.kitchen.mark_ready(order.order_id)

payment = CashPayment()
restaurant.serve_order(order.order_id, payment)
```

---

## Key Points

- **Time Complexity**: O(1) for most operations
- **Space Complexity**: O(T + M + O)
- **Features**: Order tracking, kitchen workflow, table management

---

## Practice Problems

- Add support for online orders
- Implement staff management
- Add inventory management
- Support split bills

