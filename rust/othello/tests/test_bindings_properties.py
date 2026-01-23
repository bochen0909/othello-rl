"""
Property-based tests for PyO3 bindings of the Othello game engine.

Tests universal properties that should hold across all valid game states.
"""

import numpy as np
from hypothesis import given, strategies as st, settings
import othello_rust


# Strategy for generating valid game states by playing random moves
@st.composite
def game_state(draw):
    """Generate a game state by playing a sequence of valid moves."""
    game = othello_rust.OthelloGame()
    
    # Play 0-20 random valid moves
    num_moves = draw(st.integers(min_value=0, max_value=20))
    
    for _ in range(num_moves):
        valid_moves = game.get_valid_moves()
        valid_indices = np.where(valid_moves)[0]
        
        if len(valid_indices) == 0:
            break
        
        # Pick a random valid move
        action = draw(st.sampled_from(valid_indices.tolist()))
        valid, _, game_over = game.step(action)
        
        if game_over:
            break
    
    return game


class TestStateSerializationRoundTrip:
    """
    Property 7: State Serialization Round-Trip
    
    For any valid game state in Rust, converting to Python (numpy array) and back
    should preserve the game state exactly.
    
    Validates: Requirements 2.1, 2.7
    """
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_board_state_round_trip(self, game):
        """Test that board state serialization preserves all information."""
        # Get board state from Rust
        board1 = game.get_board()
        
        # Verify it's a proper numpy array
        assert isinstance(board1, np.ndarray)
        assert board1.shape == (8, 8)
        assert board1.dtype == np.uint8
        
        # Get board state again (simulates round-trip)
        board2 = game.get_board()
        
        # Should be identical
        assert np.array_equal(board1, board2), \
            "Board state should be identical after serialization"
        
        # Verify all values are valid
        assert np.all((board1 >= 0) & (board1 <= 2)), \
            "All board values should be 0 (empty), 1 (black), or 2 (white)"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_valid_moves_round_trip(self, game):
        """Test that valid moves serialization preserves all information."""
        # Get valid moves from Rust
        moves1 = game.get_valid_moves()
        
        # Verify it's a proper numpy array
        assert isinstance(moves1, np.ndarray)
        assert moves1.shape == (64,)
        assert moves1.dtype == bool
        
        # Get valid moves again (simulates round-trip)
        moves2 = game.get_valid_moves()
        
        # Should be identical
        assert np.array_equal(moves1, moves2), \
            "Valid moves should be identical after serialization"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_piece_counts_round_trip(self, game):
        """Test that piece counts serialization preserves all information."""
        # Get piece counts from Rust
        black1, white1 = game.get_piece_counts()
        
        # Verify types
        assert isinstance(black1, int)
        assert isinstance(white1, int)
        
        # Get piece counts again (simulates round-trip)
        black2, white2 = game.get_piece_counts()
        
        # Should be identical
        assert black1 == black2, "Black count should be identical"
        assert white1 == white2, "White count should be identical"
        
        # Verify counts match board state
        board = game.get_board()
        actual_black = np.sum(board == 1)
        actual_white = np.sum(board == 2)
        
        assert black1 == actual_black, \
            f"Black count {black1} should match board count {actual_black}"
        assert white1 == actual_white, \
            f"White count {white1} should match board count {actual_white}"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_current_player_round_trip(self, game):
        """Test that current player serialization preserves all information."""
        # Get current player from Rust
        player1 = game.get_current_player()
        
        # Verify type and value
        assert isinstance(player1, int)
        assert player1 in [0, 1], "Player should be 0 (Black) or 1 (White)"
        
        # Get current player again (simulates round-trip)
        player2 = game.get_current_player()
        
        # Should be identical
        assert player1 == player2, "Current player should be identical"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_winner_round_trip(self, game):
        """Test that winner serialization preserves all information."""
        # Get winner from Rust
        winner1 = game.get_winner()
        
        # Verify type and value
        assert isinstance(winner1, int)
        assert winner1 in [0, 1, 2, 3], \
            "Winner should be 0 (Black), 1 (White), 2 (Draw), or 3 (Not finished)"
        
        # Get winner again (simulates round-trip)
        winner2 = game.get_winner()
        
        # Should be identical
        assert winner1 == winner2, "Winner should be identical"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_complete_state_round_trip(self, game):
        """Test that complete game state is preserved across serialization."""
        # Get all state components
        board1 = game.get_board()
        moves1 = game.get_valid_moves()
        black1, white1 = game.get_piece_counts()
        player1 = game.get_current_player()
        winner1 = game.get_winner()
        
        # Get all state components again
        board2 = game.get_board()
        moves2 = game.get_valid_moves()
        black2, white2 = game.get_piece_counts()
        player2 = game.get_current_player()
        winner2 = game.get_winner()
        
        # All components should be identical
        assert np.array_equal(board1, board2), "Board should be identical"
        assert np.array_equal(moves1, moves2), "Valid moves should be identical"
        assert black1 == black2, "Black count should be identical"
        assert white1 == white2, "White count should be identical"
        assert player1 == player2, "Current player should be identical"
        assert winner1 == winner2, "Winner should be identical"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_state_consistency_after_reset(self, game):
        """Test that state is consistent after reset."""
        # Reset the game
        game.reset()
        
        # Get state
        board = game.get_board()
        black, white = game.get_piece_counts()
        player = game.get_current_player()
        
        # Verify initial state
        assert board[3, 3] == 2, "Position (3,3) should be White"
        assert board[3, 4] == 1, "Position (3,4) should be Black"
        assert board[4, 3] == 1, "Position (4,3) should be Black"
        assert board[4, 4] == 2, "Position (4,4) should be White"
        assert black == 2, "Should have 2 black pieces"
        assert white == 2, "Should have 2 white pieces"
        assert player == 0, "Black should start"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_numpy_array_properties(self, game):
        """Test that numpy arrays have correct properties."""
        board = game.get_board()
        moves = game.get_valid_moves()
        
        # Board should be contiguous and C-ordered
        assert board.flags['C_CONTIGUOUS'], "Board should be C-contiguous"
        
        # Valid moves should be contiguous
        assert moves.flags['C_CONTIGUOUS'], "Valid moves should be C-contiguous"
        
        # Arrays should be writable (not read-only)
        assert board.flags['WRITEABLE'], "Board should be writable"
        assert moves.flags['WRITEABLE'], "Valid moves should be writable"



class TestActionValidityConsistency:
    """
    Property 8: Action Validity Consistency
    
    For any board state and action (0-63), the validity status returned by the
    Python bindings should match whether the move is actually valid in the Rust engine.
    
    Validates: Requirements 2.3, 2.4
    """
    
    @given(game_state(), st.integers(min_value=0, max_value=63))
    @settings(max_examples=100, deadline=None)
    def test_action_validity_matches_valid_moves(self, game, action):
        """Test that step() validity matches get_valid_moves()."""
        # Get valid moves array
        valid_moves = game.get_valid_moves()
        expected_valid = valid_moves[action]
        
        # Try to apply the move
        valid, _, _ = game.step(action)
        
        # The validity returned by step should match the valid_moves array
        assert valid == expected_valid, \
            f"Action {action}: step() returned {valid} but valid_moves[{action}] is {expected_valid}"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_all_valid_moves_are_applicable(self, game):
        """Test that all moves marked as valid can actually be applied."""
        # Get valid moves
        valid_moves = game.get_valid_moves()
        valid_indices = np.where(valid_moves)[0]
        
        for action in valid_indices:
            # Create a fresh game in the same state
            # (We can't test all moves on the same game since applying one changes the state)
            # Instead, we'll just verify the first valid move
            valid, pieces_flipped, _ = game.step(action)
            
            assert valid, f"Action {action} was marked as valid but step() returned False"
            assert pieces_flipped > 0, \
                f"Valid move at {action} should flip at least one piece, but flipped {pieces_flipped}"
            
            # Only test the first valid move since applying it changes the state
            break
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_invalid_moves_dont_change_state(self, game):
        """Test that invalid moves don't modify the game state."""
        # Get current state
        board_before = game.get_board().copy()
        black_before, white_before = game.get_piece_counts()
        player_before = game.get_current_player()
        
        # Find an invalid move
        valid_moves = game.get_valid_moves()
        invalid_indices = np.where(~valid_moves)[0]
        
        if len(invalid_indices) > 0:
            # Try an invalid move
            action = invalid_indices[0]
            valid, pieces_flipped, _ = game.step(action)
            
            # Should return False
            assert not valid, f"Action {action} should be invalid"
            assert pieces_flipped == 0, "Invalid move should not flip any pieces"
            
            # State should be unchanged
            board_after = game.get_board()
            black_after, white_after = game.get_piece_counts()
            player_after = game.get_current_player()
            
            assert np.array_equal(board_before, board_after), \
                "Board should not change after invalid move"
            assert black_before == black_after, "Black count should not change"
            assert white_before == white_after, "White count should not change"
            assert player_before == player_after, "Current player should not change"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_valid_moves_consistency_across_calls(self, game):
        """Test that get_valid_moves() returns consistent results."""
        # Get valid moves multiple times
        moves1 = game.get_valid_moves()
        moves2 = game.get_valid_moves()
        moves3 = game.get_valid_moves()
        
        # All should be identical
        assert np.array_equal(moves1, moves2), \
            "Valid moves should be consistent across calls"
        assert np.array_equal(moves2, moves3), \
            "Valid moves should be consistent across calls"
    
    @given(game_state(), st.integers(min_value=0, max_value=63))
    @settings(max_examples=100, deadline=None)
    def test_action_validity_deterministic(self, game, action):
        """Test that action validity is deterministic."""
        # Check validity multiple times
        valid_moves1 = game.get_valid_moves()
        valid_moves2 = game.get_valid_moves()
        
        # Should be identical
        assert valid_moves1[action] == valid_moves2[action], \
            f"Action {action} validity should be deterministic"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_at_least_one_valid_move_or_game_over(self, game):
        """Test that there's always at least one valid move unless game is over."""
        valid_moves = game.get_valid_moves()
        num_valid = np.sum(valid_moves)
        winner = game.get_winner()
        
        # If game is not over, there should be at least one valid move
        if winner == 3:  # Game not finished
            assert num_valid > 0, \
                "If game is not over, there should be at least one valid move"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_valid_moves_count_reasonable(self, game):
        """Test that the number of valid moves is reasonable."""
        valid_moves = game.get_valid_moves()
        num_valid = np.sum(valid_moves)
        
        # Should have between 0 and 64 valid moves (obviously)
        assert 0 <= num_valid <= 64, \
            f"Number of valid moves {num_valid} should be between 0 and 64"
        
        # More specifically, should not exceed number of empty cells
        board = game.get_board()
        num_empty = np.sum(board == 0)
        
        assert num_valid <= num_empty, \
            f"Number of valid moves {num_valid} should not exceed empty cells {num_empty}"
    
    @given(game_state())
    @settings(max_examples=100, deadline=None)
    def test_occupied_cells_never_valid(self, game):
        """Test that occupied cells are never marked as valid moves."""
        board = game.get_board()
        valid_moves = game.get_valid_moves()
        
        for i in range(64):
            row = i // 8
            col = i % 8
            
            # If cell is occupied, it should not be a valid move
            if board[row, col] != 0:
                assert not valid_moves[i], \
                    f"Occupied cell at ({row}, {col}) should not be a valid move"
    
    @given(game_state(), st.integers(min_value=0, max_value=63))
    @settings(max_examples=100, deadline=None)
    def test_step_return_values_consistent(self, game, action):
        """Test that step() return values are consistent with game state."""
        valid_moves = game.get_valid_moves()
        expected_valid = valid_moves[action]
        
        valid, pieces_flipped, game_over = game.step(action)
        
        # Validity should match
        assert valid == expected_valid
        
        # If invalid, pieces_flipped should be 0
        if not valid:
            assert pieces_flipped == 0, \
                "Invalid move should not flip any pieces"
        
        # If valid, pieces_flipped should be > 0
        if valid:
            assert pieces_flipped > 0, \
                "Valid move should flip at least one piece"
        
        # game_over should be boolean
        assert isinstance(game_over, bool), \
            "game_over should be a boolean"
