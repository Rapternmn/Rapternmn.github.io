+++
title = "Tic Tac Toe"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 12
description = "Complete implementation of Tic Tac Toe: Game board, move validation, win condition checking, and player management."
+++

---

## Problem Statement

Design a Tic Tac Toe game that can:
- Represent 3x3 game board
- Handle player moves
- Validate moves
- Check win conditions
- Detect draw
- Support two players

---

## Implementation

```python
from enum import Enum
from typing import Optional, Tuple, List

class Player(Enum):
    X = "X"
    O = "O"
    EMPTY = " "

class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    X_WON = "x_won"
    O_WON = "o_won"
    DRAW = "draw"

class Board:
    def __init__(self):
        self.grid = [[Player.EMPTY for _ in range(3)] for _ in range(3)]
        self.move_count = 0
    
    def make_move(self, row: int, col: int, player: Player) -> bool:
        if not self.is_valid_move(row, col):
            return False
        
        self.grid[row][col] = player
        self.move_count += 1
        return True
    
    def is_valid_move(self, row: int, col: int) -> bool:
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return False
        return self.grid[row][col] == Player.EMPTY
    
    def check_winner(self) -> Optional[Player]:
        # Check rows
        for row in self.grid:
            if row[0] != Player.EMPTY and row[0] == row[1] == row[2]:
                return row[0]
        
        # Check columns
        for col in range(3):
            if (self.grid[0][col] != Player.EMPTY and 
                self.grid[0][col] == self.grid[1][col] == self.grid[2][col]):
                return self.grid[0][col]
        
        # Check diagonals
        if (self.grid[0][0] != Player.EMPTY and 
            self.grid[0][0] == self.grid[1][1] == self.grid[2][2]):
            return self.grid[0][0]
        
        if (self.grid[0][2] != Player.EMPTY and 
            self.grid[0][2] == self.grid[1][1] == self.grid[2][0]):
            return self.grid[0][2]
        
        return None
    
    def is_full(self) -> bool:
        return self.move_count == 9
    
    def display(self):
        for i, row in enumerate(self.grid):
            print(" | ".join(cell.value for cell in row))
            if i < 2:
                print("-" * 9)

class TicTacToe:
    def __init__(self):
        self.board = Board()
        self.current_player = Player.X
        self.status = GameStatus.IN_PROGRESS
    
    def make_move(self, row: int, col: int) -> bool:
        if self.status != GameStatus.IN_PROGRESS:
            return False
        
        if not self.board.make_move(row, col, self.current_player):
            return False
        
        winner = self.board.check_winner()
        if winner:
            self.status = GameStatus.X_WON if winner == Player.X else GameStatus.O_WON
        elif self.board.is_full():
            self.status = GameStatus.DRAW
        else:
            self.current_player = Player.O if self.current_player == Player.X else Player.X
        
        return True
    
    def get_status(self) -> GameStatus:
        return self.status
    
    def get_current_player(self) -> Player:
        return self.current_player

# Usage
game = TicTacToe()
game.make_move(0, 0)  # X
game.make_move(1, 1)  # O
game.make_move(0, 1)  # X
game.make_move(1, 0)  # O
game.make_move(0, 2)  # X wins
game.board.display()
```

---

## Design Patterns Used

1. **[Strategy Pattern]({{< ref "../design-patterns/12-Strategy_Pattern.md" >}})**: Different player strategies (human, AI with minimax) can be implemented as interchangeable strategies
2. **[State Pattern]({{< ref "../design-patterns/14-State_Pattern.md" >}})**: `GameStatus` enum manages game state transitions (IN_PROGRESS → X_WON/O_WON/DRAW)

---

## Key Points

- **Time Complexity**: O(1) for move and win check
- **Space Complexity**: O(1) for 3x3 board
- **Features**: Simple game logic, win detection, draw detection

---

## Practice Problems

- Add AI player with minimax
- Support different board sizes
- Add game history
- Implement online multiplayer

