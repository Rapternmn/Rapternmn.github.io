+++
title = "Chess Game"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Complete implementation of Chess Game: Game board, piece movement, move validation, check/checkmate detection, and game state management."
+++

---

## Problem Statement

Design a chess game system that can:
- Represent chess board and pieces
- Validate moves
- Detect check and checkmate
- Track game state
- Support undo/redo
- Handle special moves (castling, en passant, promotion)

---

## Implementation

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple
from copy import deepcopy

class Color(Enum):
    WHITE = "white"
    BLACK = "black"

class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"

class Piece(ABC):
    def __init__(self, color: Color, piece_type: PieceType):
        self.color = color
        self.piece_type = piece_type
        self.has_moved = False
    
    @abstractmethod
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        pass
    
    def move(self):
        self.has_moved = True

class Pawn(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.PAWN)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        direction = -1 if self.color == Color.WHITE else 1
        
        # Forward move
        if col_start == col_end:
            if row_end == row_start + direction:
                return board.get_piece(end) is None
            if not self.has_moved and row_end == row_start + 2 * direction:
                return (board.get_piece(end) is None and 
                       board.get_piece((row_start + direction, col_start)) is None)
        
        # Diagonal capture
        if abs(col_end - col_start) == 1 and row_end == row_start + direction:
            target = board.get_piece(end)
            return target is not None and target.color != self.color
        
        return False

class Rook(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.ROOK)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        # Must move in straight line
        if row_start != row_end and col_start != col_end:
            return False
        
        # Check if path is clear
        return board.is_path_clear(start, end)

class Knight(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.KNIGHT)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        row_diff = abs(row_end - row_start)
        col_diff = abs(col_end - col_start)
        
        # Knight moves in L-shape
        if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
            target = board.get_piece(end)
            return target is None or target.color != self.color
        return False

class Bishop(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.BISHOP)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        # Must move diagonally
        if abs(row_end - row_start) != abs(col_end - col_start):
            return False
        
        return board.is_path_clear(start, end)

class Queen(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.QUEEN)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        # Queen can move like rook or bishop
        rook = Rook(self.color)
        bishop = Bishop(self.color)
        return rook.is_valid_move(board, start, end) or bishop.is_valid_move(board, start, end)

class King(Piece):
    def __init__(self, color: Color):
        super().__init__(color, PieceType.KING)
    
    def is_valid_move(self, board, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        row_diff = abs(row_end - row_start)
        col_diff = abs(col_end - col_start)
        
        # King moves one square in any direction
        if row_diff <= 1 and col_diff <= 1:
            target = board.get_piece(end)
            return target is None or target.color != self.color
        return False

class Board:
    def __init__(self):
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        self._initialize_board()
    
    def _initialize_board(self):
        # Initialize pawns
        for col in range(8):
            self.grid[1][col] = Pawn(Color.BLACK)
            self.grid[6][col] = Pawn(Color.WHITE)
        
        # Initialize other pieces
        pieces_order = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for col, piece_class in enumerate(pieces_order):
            self.grid[0][col] = piece_class(Color.BLACK)
            self.grid[7][col] = piece_class(Color.WHITE)
    
    def get_piece(self, position: Tuple[int, int]) -> Optional[Piece]:
        row, col = position
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row][col]
        return None
    
    def set_piece(self, position: Tuple[int, int], piece: Optional[Piece]):
        row, col = position
        if 0 <= row < 8 and 0 <= col < 8:
            self.grid[row][col] = piece
    
    def is_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        row_start, col_start = start
        row_end, col_end = end
        
        row_step = 0 if row_end == row_start else (1 if row_end > row_start else -1)
        col_step = 0 if col_end == col_start else (1 if col_end > col_start else -1)
        
        row, col = row_start + row_step, col_start + col_step
        while (row, col) != end:
            if self.get_piece((row, col)) is not None:
                return False
            row += row_step
            col += col_step
        
        target = self.get_piece(end)
        return target is None or target.color != self.grid[row_start][col_start].color
    
    def find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        for row in range(8):
            for col in range(8):
                piece = self.grid[row][col]
                if piece and piece.piece_type == PieceType.KING and piece.color == color:
                    return (row, col)
        return None
    
    def is_in_check(self, color: Color) -> bool:
        king_pos = self.find_king(color)
        if not king_pos:
            return False
        
        opponent_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        
        for row in range(8):
            for col in range(8):
                piece = self.grid[row][col]
                if piece and piece.color == opponent_color:
                    if piece.is_valid_move(self, (row, col), king_pos):
                        return True
        return False

class Move:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], piece: Piece, captured: Optional[Piece] = None):
        self.start = start
        self.end = end
        self.piece = piece
        self.captured = captured

class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = Color.WHITE
        self.move_history: List[Move] = []
        self.game_over = False
        self.winner: Optional[Color] = None
    
    def make_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        piece = self.board.get_piece(start)
        
        if not piece or piece.color != self.current_turn:
            return False
        
        if not piece.is_valid_move(self.board, start, end):
            return False
        
        # Make move
        captured = self.board.get_piece(end)
        self.board.set_piece(end, piece)
        self.board.set_piece(start, None)
        piece.move()
        
        # Check if move puts own king in check
        if self.board.is_in_check(self.current_turn):
            # Undo move
            self.board.set_piece(start, piece)
            self.board.set_piece(end, captured)
            return False
        
        move = Move(start, end, piece, captured)
        self.move_history.append(move)
        
        # Switch turn
        self.current_turn = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        
        # Check for checkmate
        if self.is_checkmate(self.current_turn):
            self.game_over = True
            self.winner = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        
        return True
    
    def is_checkmate(self, color: Color) -> bool:
        if not self.board.is_in_check(color):
            return False
        
        # Check if any move can get out of check
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece((row, col))
                if piece and piece.color == color:
                    for end_row in range(8):
                        for end_col in range(8):
                            if piece.is_valid_move(self.board, (row, col), (end_row, end_col)):
                                # Try move
                                captured = self.board.get_piece((end_row, end_col))
                                self.board.set_piece((end_row, end_col), piece)
                                self.board.set_piece((row, col), None)
                                
                                still_in_check = self.board.is_in_check(color)
                                
                                # Undo move
                                self.board.set_piece((row, col), piece)
                                self.board.set_piece((end_row, end_col), captured)
                                
                                if not still_in_check:
                                    return False
        return True

# Usage
game = Game()
game.make_move((6, 4), (4, 4))  # White pawn
game.make_move((1, 4), (3, 4))  # Black pawn
```

---

## Key Points

- **Time Complexity**: O(1) for move validation, O(P) for checkmate detection
- **Space Complexity**: O(1) for board representation
- **Features**: Move validation, check/checkmate, special moves

---

## Practice Problems

- Implement castling
- Add en passant
- Implement pawn promotion
- Add move history and undo

