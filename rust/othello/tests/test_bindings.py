"""
Unit tests for PyO3 bindings of the Othello game engine.

Tests type conversions, error propagation, and reset functionality.
"""

import pytest
import numpy as np
import othello_rust


class TestOthelloGameBindings:
    """Test suite for OthelloGame Python bindings."""

    def test_new_game_initialization(self):
        """Test that a new game initializes with correct state."""
        game = othello_rust.OthelloGame()
        
        # Get initial board
        board = game.get_board()
        
        # Check board shape and type
        assert board.shape == (8, 8)
        assert board.dtype == np.uint8
        
        # Check initial piece positions
        assert board[3, 3] == 2  # White
        assert board[3, 4] == 1  # Black
        assert board[4, 3] == 1  # Black
        assert board[4, 4] == 2  # White
        
        # Check that other cells are empty
        empty_count = np.sum(board == 0)
        assert empty_count == 60  # 64 - 4 initial pieces

    def test_get_current_player(self):
        """Test getting the current player."""
        game = othello_rust.OthelloGame()
        
        # Black starts
        assert game.get_current_player() == 0
        
        # After a valid move, should switch to White
        valid, _, _ = game.step(19)  # (2, 3)
        assert valid
        assert game.get_current_player() == 1

    def test_get_piece_counts(self):
        """Test getting piece counts."""
        game = othello_rust.OthelloGame()
        
        # Initial counts
        black_count, white_count = game.get_piece_counts()
        assert black_count == 2
        assert white_count == 2
        
        # After a move
        game.step(19)  # Black plays at (2, 3)
        black_count, white_count = game.get_piece_counts()
        assert black_count == 4  # Placed 1, flipped 1
        assert white_count == 1

    def test_get_valid_moves(self):
        """Test getting valid moves."""
        game = othello_rust.OthelloGame()
        
        valid_moves = game.get_valid_moves()
        
        # Check type and shape
        assert valid_moves.shape == (64,)
        assert valid_moves.dtype == bool
        
        # Check that there are exactly 4 valid moves initially
        assert np.sum(valid_moves) == 4
        
        # Check specific valid positions
        assert valid_moves[19]  # (2, 3)
        assert valid_moves[26]  # (3, 2)
        assert valid_moves[37]  # (4, 5)
        assert valid_moves[44]  # (5, 4)

    def test_step_valid_move(self):
        """Test applying a valid move."""
        game = othello_rust.OthelloGame()
        
        # Apply valid move
        valid, pieces_flipped, game_over = game.step(19)  # (2, 3)
        
        assert valid is True
        assert pieces_flipped == 1
        assert game_over is False
        
        # Check that the piece was placed
        board = game.get_board()
        assert board[2, 3] == 1  # Black piece

    def test_step_invalid_move(self):
        """Test applying an invalid move."""
        game = othello_rust.OthelloGame()
        
        # Try invalid move (corner with no adjacent pieces)
        valid, pieces_flipped, game_over = game.step(0)  # (0, 0)
        
        assert valid is False
        assert pieces_flipped == 0
        assert game_over is False
        
        # Board should be unchanged
        board = game.get_board()
        assert board[0, 0] == 0  # Still empty

    def test_step_out_of_range_action(self):
        """Test that out-of-range actions raise ValueError."""
        game = othello_rust.OthelloGame()
        
        # Action >= 64 should raise ValueError
        with pytest.raises(ValueError, match="out of range"):
            game.step(64)
        
        with pytest.raises(ValueError, match="out of range"):
            game.step(100)

    def test_reset_functionality(self):
        """Test that reset returns game to initial state."""
        game = othello_rust.OthelloGame()
        
        # Make some moves
        game.step(19)  # Black
        game.step(18)  # White
        
        # Verify game state changed
        black_count, white_count = game.get_piece_counts()
        assert black_count != 2 or white_count != 2
        
        # Reset
        game.reset()
        
        # Check that game is back to initial state
        board = game.get_board()
        assert board[3, 3] == 2  # White
        assert board[3, 4] == 1  # Black
        assert board[4, 3] == 1  # Black
        assert board[4, 4] == 2  # White
        
        black_count, white_count = game.get_piece_counts()
        assert black_count == 2
        assert white_count == 2
        
        assert game.get_current_player() == 0  # Black

    def test_get_winner_game_not_over(self):
        """Test getting winner when game is not over."""
        game = othello_rust.OthelloGame()
        
        winner = game.get_winner()
        assert winner == 3  # Game not finished

    def test_type_conversion_board_state(self):
        """Test that board state is correctly converted to numpy array."""
        game = othello_rust.OthelloGame()
        
        board = game.get_board()
        
        # Check that it's a numpy array
        assert isinstance(board, np.ndarray)
        
        # Check dtype
        assert board.dtype == np.uint8
        
        # Check values are in valid range
        assert np.all((board >= 0) & (board <= 2))

    def test_type_conversion_valid_moves(self):
        """Test that valid moves are correctly converted to numpy array."""
        game = othello_rust.OthelloGame()
        
        valid_moves = game.get_valid_moves()
        
        # Check that it's a numpy array
        assert isinstance(valid_moves, np.ndarray)
        
        # Check dtype
        assert valid_moves.dtype == bool

    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        game = othello_rust.OthelloGame()
        
        for _ in range(3):
            # Make a move
            game.step(19)
            
            # Reset
            game.reset()
            
            # Verify initial state
            black_count, white_count = game.get_piece_counts()
            assert black_count == 2
            assert white_count == 2
            assert game.get_current_player() == 0

    def test_game_state_consistency(self):
        """Test that game state remains consistent across operations."""
        game = othello_rust.OthelloGame()
        
        # Get initial state
        board1 = game.get_board()
        valid_moves1 = game.get_valid_moves()
        
        # Get state again (should be identical)
        board2 = game.get_board()
        valid_moves2 = game.get_valid_moves()
        
        assert np.array_equal(board1, board2)
        assert np.array_equal(valid_moves1, valid_moves2)

    def test_error_propagation_invalid_move(self):
        """Test that invalid moves are handled gracefully without exceptions."""
        game = othello_rust.OthelloGame()
        
        # Invalid move should return False, not raise exception
        valid, _, _ = game.step(0)
        assert valid is False
        
        # Game should still be playable
        valid, _, _ = game.step(19)
        assert valid is True
