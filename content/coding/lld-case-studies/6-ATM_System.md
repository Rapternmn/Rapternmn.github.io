+++
title = "ATM System"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Complete implementation of ATM System: Card authentication, transaction processing, cash dispensing, account management using State pattern and transaction handling."
+++

---

## Problem Statement

Design an ATM system that can:
- Authenticate users with card and PIN
- Check account balance
- Withdraw cash
- Deposit cash
- Transfer funds
- Handle transaction history
- Manage ATM cash inventory

---

## Requirements

### Functional Requirements
1. Card insertion and authentication
2. PIN verification
3. Balance inquiry
4. Cash withdrawal
5. Cash deposit
6. Fund transfer
7. Transaction history
8. Receipt printing

### Non-Functional Requirements
1. Secure authentication
2. Transaction integrity
3. Error handling
4. Logging

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
from typing import Optional, List
import threading

class ATMState(ABC):
    @abstractmethod
    def insert_card(self, atm, card_number: str):
        pass
    
    @abstractmethod
    def enter_pin(self, atm, pin: str):
        pass
    
    @abstractmethod
    def select_operation(self, atm, operation: str):
        pass

class IdleState(ATMState):
    def insert_card(self, atm, card_number: str):
        atm.card_number = card_number
        atm.set_state(HasCardState())
        print("Card inserted. Please enter PIN.")
    
    def enter_pin(self, atm, pin: str):
        print("Please insert card first")
    
    def select_operation(self, atm, operation: str):
        print("Please insert card and enter PIN first")

class HasCardState(ATMState):
    def insert_card(self, atm, card_number: str):
        print("Card already inserted")
    
    def enter_pin(self, atm, pin: str):
        if atm.authenticate(pin):
            atm.set_state(HasPinState())
            print("PIN verified. Please select operation.")
        else:
            atm.attempts += 1
            if atm.attempts >= 3:
                print("Too many failed attempts. Card retained.")
                atm.set_state(IdleState())
                atm.card_number = None
            else:
                print(f"Invalid PIN. {3 - atm.attempts} attempts remaining.")
    
    def select_operation(self, atm, operation: str):
        print("Please enter PIN first")

class HasPinState(ATMState):
    def insert_card(self, atm, card_number: str):
        print("Card already inserted")
    
    def enter_pin(self, atm, pin: str):
        print("PIN already entered")
    
    def select_operation(self, atm, operation: str):
        if operation == "balance":
            atm.check_balance()
        elif operation == "withdraw":
            atm.set_state(WithdrawState())
        elif operation == "deposit":
            atm.set_state(DepositState())
        elif operation == "transfer":
            atm.set_state(TransferState())
        elif operation == "exit":
            atm.eject_card()

class WithdrawState(ATMState):
    def withdraw(self, atm, amount: float):
        if atm.account.withdraw(amount) and atm.dispenser.dispense(amount):
            transaction = Transaction("withdraw", amount, datetime.now())
            atm.account.add_transaction(transaction)
            print(f"Withdrawn ${amount}. Please take your cash.")
            atm.set_state(HasPinState())
        else:
            print("Transaction failed. Insufficient funds or ATM out of cash.")
            atm.set_state(HasPinState())

class DepositState(ATMState):
    def deposit(self, atm, amount: float):
        if atm.deposit_slot.accept_cash(amount):
            atm.account.deposit(amount)
            transaction = Transaction("deposit", amount, datetime.now())
            atm.account.add_transaction(transaction)
            print(f"Deposited ${amount} successfully.")
            atm.set_state(HasPinState())

class TransferState(ATMState):
    def transfer(self, atm, to_account: str, amount: float):
        if atm.account.transfer(to_account, amount):
            transaction = Transaction("transfer", amount, datetime.now(), to_account)
            atm.account.add_transaction(transaction)
            print(f"Transferred ${amount} to {to_account}")
            atm.set_state(HasPinState())
        else:
            print("Transfer failed. Insufficient funds.")
            atm.set_state(HasPinState())

class Account:
    def __init__(self, account_number: str, balance: float, pin: str):
        self.account_number = account_number
        self.balance = balance
        self.pin = pin
        self.transactions: List[Transaction] = []
        self._lock = threading.Lock()
    
    def verify_pin(self, pin: str) -> bool:
        return self.pin == pin
    
    def get_balance(self) -> float:
        return self.balance
    
    def withdraw(self, amount: float) -> bool:
        with self._lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False
    
    def deposit(self, amount: float):
        with self._lock:
            self.balance += amount
    
    def transfer(self, to_account: str, amount: float) -> bool:
        with self._lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False
    
    def add_transaction(self, transaction):
        self.transactions.append(transaction)
    
    def get_transaction_history(self, limit: int = 10) -> List:
        return self.transactions[-limit:]

class Transaction:
    def __init__(self, transaction_type: str, amount: float, timestamp: datetime, to_account: Optional[str] = None):
        self.transaction_type = transaction_type
        self.amount = amount
        self.timestamp = timestamp
        self.to_account = to_account

class CashDispenser:
    def __init__(self, initial_cash: float):
        self.available_cash = initial_cash
        self._lock = threading.Lock()
    
    def dispense(self, amount: float) -> bool:
        with self._lock:
            if self.available_cash >= amount:
                self.available_cash -= amount
                return True
            return False
    
    def refill(self, amount: float):
        with self._lock:
            self.available_cash += amount

class DepositSlot:
    def accept_cash(self, amount: float) -> bool:
        # Simulate cash acceptance
        print(f"Accepting ${amount} in cash...")
        return True

class ATM:
    def __init__(self, cash_dispenser: CashDispenser, deposit_slot: DepositSlot):
        self.state: ATMState = IdleState()
        self.card_number: Optional[str] = None
        self.account: Optional[Account] = None
        self.attempts = 0
        self.dispenser = cash_dispenser
        self.deposit_slot = deposit_slot
        self.accounts: dict = {}  # card_number -> Account mapping
    
    def set_state(self, state: ATMState):
        self.state = state
    
    def authenticate(self, pin: str) -> bool:
        if self.card_number in self.accounts:
            account = self.accounts[self.card_number]
            if account.verify_pin(pin):
                self.account = account
                self.attempts = 0
                return True
        return False
    
    def insert_card(self, card_number: str):
        self.state.insert_card(self, card_number)
    
    def enter_pin(self, pin: str):
        self.state.enter_pin(self, pin)
    
    def select_operation(self, operation: str):
        self.state.select_operation(self, operation)
    
    def check_balance(self):
        if self.account:
            print(f"Current balance: ${self.account.get_balance():.2f}")
    
    def withdraw(self, amount: float):
        if isinstance(self.state, WithdrawState):
            self.state.withdraw(self, amount)
    
    def deposit(self, amount: float):
        if isinstance(self.state, DepositState):
            self.state.deposit(self, amount)
    
    def transfer(self, to_account: str, amount: float):
        if isinstance(self.state, TransferState):
            self.state.transfer(self, to_account, amount)
    
    def eject_card(self):
        print("Card ejected. Thank you!")
        self.card_number = None
        self.account = None
        self.attempts = 0
        self.set_state(IdleState())
    
    def register_account(self, card_number: str, account: Account):
        self.accounts[card_number] = account

# Usage
account = Account("ACC001", 1000.0, "1234")
dispenser = CashDispenser(5000.0)
deposit_slot = DepositSlot()

atm = ATM(dispenser, deposit_slot)
atm.register_account("CARD123", account)

atm.insert_card("CARD123")
atm.enter_pin("1234")
atm.select_operation("balance")
atm.select_operation("withdraw")
atm.withdraw(100.0)
atm.eject_card()
```

---

## Design Patterns Used

1. **[State Pattern]({{< ref "../design-patterns/14-State_Pattern.md" >}})**: `ATMState` interface with concrete states (`IdleState`, `HasCardState`, `HasPinState`, `WithdrawState`, `DepositState`, `TransferState`) manages ATM behavior and transitions
2. **[Strategy Pattern]({{< ref "../design-patterns/12-Strategy_Pattern.md" >}})**: Different transaction types (withdraw, deposit, transfer) can be implemented as interchangeable strategies
3. **[Factory Pattern]({{< ref "../design-patterns/4-Factory_Pattern.md" >}})**: `Transaction` objects can be created using Factory pattern based on transaction type

---

## Key Points

- **Time Complexity**: O(1) for most operations
- **Space Complexity**: O(T) for transaction history
- **Security**: PIN verification, transaction limits
- **Thread Safety**: Uses locks for account operations

---

## Practice Problems

- Add support for multiple account types
- Implement transaction limits
- Add biometric authentication
- Implement card retention mechanism

