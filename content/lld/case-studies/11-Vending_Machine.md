+++
title = "Vending Machine"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Complete implementation of Vending Machine: Product inventory, payment processing, change dispensing, and state management using State pattern."
+++

---

## Problem Statement

Design a vending machine that can:
- Store products with inventory
- Accept coins and bills
- Dispense products
- Return change
- Handle different states (Idle, HasMoney, Dispensing)
- Support different payment methods

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional
import threading

class VendingMachineState(ABC):
    @abstractmethod
    def insert_coin(self, machine, amount: float):
        pass
    
    @abstractmethod
    def select_product(self, machine, product_id: str):
        pass
    
    @abstractmethod
    def dispense(self, machine):
        pass

class IdleState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        machine.add_balance(amount)
        machine.set_state(HasMoneyState())
        print(f"Balance: ${machine.balance:.2f}")
    
    def select_product(self, machine, product_id: str):
        print("Please insert money first")
    
    def dispense(self, machine):
        print("No product selected")

class HasMoneyState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        machine.add_balance(amount)
        print(f"Balance: ${machine.balance:.2f}")
    
    def select_product(self, machine, product_id: str):
        if product_id not in machine.products:
            print("Product not found")
            return
        
        product = machine.products[product_id]
        if product['quantity'] == 0:
            print("Product out of stock")
            return
        
        if machine.balance >= product['price']:
            machine.selected_product = product_id
            machine.set_state(DispensingState())
        else:
            print(f"Insufficient balance. Need ${product['price']:.2f}, have ${machine.balance:.2f}")
    
    def dispense(self, machine):
        print("Please select a product first")

class DispensingState(VendingMachineState):
    def insert_coin(self, machine, amount: float):
        print("Please wait, dispensing product...")
    
    def select_product(self, machine, product_id: str):
        print("Please wait, dispensing product...")
    
    def dispense(self, machine):
        if not machine.selected_product:
            return
        
        product = machine.products[machine.selected_product]
        price = product['price']
        
        if machine.balance >= price:
            # Deduct price
            machine.balance -= price
            change = machine.balance
            
            # Dispense product
            product['quantity'] -= 1
            print(f"Dispensing {product['name']}...")
            
            # Return change
            if change > 0:
                machine.return_change(change)
            
            machine.balance = 0
            machine.selected_product = None
            machine.set_state(IdleState())
            print("Thank you!")
        else:
            print("Insufficient balance")
            machine.set_state(HasMoneyState())

class VendingMachine:
    def __init__(self):
        self.state: VendingMachineState = IdleState()
        self.balance = 0.0
        self.selected_product: Optional[str] = None
        self.products: Dict = {}
        self.change_inventory = {
            0.25: 10,  # quarters
            0.10: 10,  # dimes
            0.05: 10,  # nickels
            1.0: 5,    # dollars
        }
        self._lock = threading.Lock()
    
    def set_state(self, state: VendingMachineState):
        self.state = state
    
    def add_product(self, product_id: str, name: str, price: float, quantity: int):
        with self._lock:
            self.products[product_id] = {
                'name': name,
                'price': price,
                'quantity': quantity
            }
    
    def add_balance(self, amount: float):
        with self._lock:
            self.balance += amount
    
    def insert_coin(self, amount: float):
        self.state.insert_coin(self, amount)
    
    def select_product(self, product_id: str):
        self.state.select_product(self, product_id)
    
    def dispense(self):
        self.state.dispense(self)
    
    def return_change(self, amount: float):
        # Simplified change return
        print(f"Returning change: ${amount:.2f}")
        # In real implementation, would calculate coins/bills to return

# Usage
machine = VendingMachine()
machine.add_product("P001", "Coke", 1.50, 5)
machine.add_product("P002", "Chips", 1.00, 3)

machine.insert_coin(2.0)
machine.select_product("P001")
machine.dispense()
```

---

## Key Points

- **Time Complexity**: O(1) for most operations
- **Space Complexity**: O(P) for products
- **Features**: State management, payment processing, change dispensing

---

## Practice Problems

- Add support for bills
- Implement exact change calculation
- Add product restocking
- Support card payments

