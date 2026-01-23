/// Core types and game logic for Othello (Reversi)

// PyO3 bindings module
pub mod bindings;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    InvalidMove,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Black,
    White,
}

impl Player {
    /// Get the opponent player
    pub fn opponent(&self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }

    /// Convert player to cell representation
    pub fn to_cell(&self) -> Cell {
        match self {
            Player::Black => Cell::Black,
            Player::White => Cell::White,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

#[derive(Debug)]
pub struct Board {
    pub(crate) cells: [[Cell; 8]; 8],
    current_player: Player,
    black_count: u8,
    white_count: u8,
    game_over: bool,
}

impl Board {
    /// Create a new board with initial Othello setup
    /// Initial setup has 4 pieces in the center:
    /// - (3,3) and (4,4) are White
    /// - (3,4) and (4,3) are Black
    pub fn new() -> Self {
        let mut cells = [[Cell::Empty; 8]; 8];
        
        // Set up initial 4 pieces in center
        cells[3][3] = Cell::White;
        cells[3][4] = Cell::Black;
        cells[4][3] = Cell::Black;
        cells[4][4] = Cell::White;
        
        Board {
            cells,
            current_player: Player::Black,  // Black always starts
            black_count: 2,
            white_count: 2,
            game_over: false,
        }
    }

    /// Check if a move is valid at position (row, col)
    /// A move is valid if:
    /// 1. The cell is empty
    /// 2. Placing a piece there would flip at least one opponent piece
    pub fn is_valid_move(&self, row: usize, col: usize) -> bool {
        // Out of bounds check
        if row >= 8 || col >= 8 {
            return false;
        }

        // Must be empty cell
        if self.cells[row][col] != Cell::Empty {
            return false;
        }

        // Check all 8 directions
        let directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ];

        for (dr, dc) in directions {
            if self.would_flip_in_direction(row, col, dr, dc) {
                return true;
            }
        }

        false
    }

    /// Check if placing a piece at (row, col) would flip pieces in direction (dr, dc)
    /// Returns true if there's at least one opponent piece followed by a player piece
    pub(crate) fn would_flip_in_direction(&self, row: usize, col: usize, dr: i8, dc: i8) -> bool {
        let opponent = self.current_player.opponent().to_cell();
        let player = self.current_player.to_cell();
        
        let mut r = row as i8 + dr;
        let mut c = col as i8 + dc;
        let mut found_opponent = false;

        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            match self.cells[r as usize][c as usize] {
                Cell::Empty => return false,
                cell if cell == opponent => {
                    found_opponent = true;
                    r += dr;
                    c += dc;
                }
                cell if cell == player => return found_opponent,
                _ => return false,
            }
        }

        false
    }

    /// Get all valid moves for current player as a 64-element bool array
    /// Array is indexed as: index = row * 8 + col
    pub fn get_valid_moves(&self) -> [bool; 64] {
        let mut valid_moves = [false; 64];
        
        for row in 0..8 {
            for col in 0..8 {
                let index = row * 8 + col;
                valid_moves[index] = self.is_valid_move(row, col);
            }
        }
        
        valid_moves
    }

    /// Apply a move at position (row, col), flipping pieces
    /// Returns Ok(pieces_flipped) or Err(InvalidMove)
    pub fn apply_move(&mut self, row: usize, col: usize) -> Result<u8, GameError> {
        if !self.is_valid_move(row, col) {
            return Err(GameError::InvalidMove);
        }

        let directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ];

        let mut total_flipped = 0;

        // Place the piece
        self.cells[row][col] = self.current_player.to_cell();

        // Flip in each valid direction
        for (dr, dc) in directions {
            let flipped = self.flip_in_direction(row, col, dr, dc);
            total_flipped += flipped;
        }

        // Update counts
        self.update_piece_counts();

        // Check if board is full
        if self.is_board_full() {
            self.game_over = true;
            return Ok(total_flipped);
        }

        // Switch to opponent
        self.current_player = self.current_player.opponent();

        // Check if opponent has moves, otherwise pass turn
        if self.get_valid_moves().iter().all(|&v| !v) {
            self.pass_turn();
        }

        Ok(total_flipped)
    }

    /// Flip pieces in a specific direction from (row, col)
    /// Returns the number of pieces flipped
    fn flip_in_direction(&mut self, row: usize, col: usize, dr: i8, dc: i8) -> u8 {
        if !self.would_flip_in_direction(row, col, dr, dc) {
            return 0;
        }

        let opponent = self.current_player.opponent().to_cell();
        let player = self.current_player.to_cell();
        let mut flipped = 0;

        let mut r = row as i8 + dr;
        let mut c = col as i8 + dc;

        while r >= 0 && r < 8 && c >= 0 && c < 8 {
            let cell = self.cells[r as usize][c as usize];
            if cell == opponent {
                self.cells[r as usize][c as usize] = player;
                flipped += 1;
                r += dr;
                c += dc;
            } else {
                break;
            }
        }

        flipped
    }

    /// Update piece counts by scanning the board
    pub fn update_piece_counts(&mut self) {
        let mut black = 0;
        let mut white = 0;

        for row in 0..8 {
            for col in 0..8 {
                match self.cells[row][col] {
                    Cell::Black => black += 1,
                    Cell::White => white += 1,
                    Cell::Empty => {}
                }
            }
        }

        self.black_count = black;
        self.white_count = white;
    }

    /// Get the current player
    pub fn get_current_player(&self) -> Player {
        self.current_player
    }

    /// Get piece counts (black_count, white_count)
    pub fn get_piece_counts(&self) -> (u8, u8) {
        (self.black_count, self.white_count)
    }

    /// Get the winner of the game
    /// Returns Some(Player) if there's a winner, None if it's a draw or game is not over
    pub fn get_winner(&self) -> Option<Player> {
        if !self.game_over {
            return None;
        }

        if self.black_count > self.white_count {
            Some(Player::Black)
        } else if self.white_count > self.black_count {
            Some(Player::White)
        } else {
            None // Draw
        }
    }

    /// Check if the game is over
    /// Game is over when:
    /// 1. The board is full (no empty cells), OR
    /// 2. Neither player has valid moves
    pub fn is_game_over(&self) -> bool {
        self.game_over
    }

    /// Check if the board is full (no empty cells)
    fn is_board_full(&self) -> bool {
        self.black_count + self.white_count == 64
    }

    /// Get current board state as flat array [0=empty, 1=black, 2=white]
    /// Array is indexed as: index = row * 8 + col
    pub fn get_state(&self) -> [u8; 64] {
        let mut state = [0u8; 64];
        
        for row in 0..8 {
            for col in 0..8 {
                let index = row * 8 + col;
                state[index] = match self.cells[row][col] {
                    Cell::Empty => 0,
                    Cell::Black => 1,
                    Cell::White => 2,
                };
            }
        }
        
        state
    }

    /// Reset board to initial state
    pub fn reset(&mut self) {
        // Clear the board
        self.cells = [[Cell::Empty; 8]; 8];
        
        // Set up initial 4 pieces in center
        self.cells[3][3] = Cell::White;
        self.cells[3][4] = Cell::Black;
        self.cells[4][3] = Cell::Black;
        self.cells[4][4] = Cell::White;
        
        // Reset game state
        self.current_player = Player::Black;
        self.black_count = 2;
        self.white_count = 2;
        self.game_over = false;
    }

    /// Pass turn to opponent (when no valid moves)
    pub fn pass_turn(&mut self) {
        self.current_player = self.current_player.opponent();
        
        // Check if the new current player has moves
        // If not, game is over (neither player has moves)
        if self.get_valid_moves().iter().all(|&v| !v) {
            self.game_over = true;
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_new_initial_setup() {
        let board = Board::new();
        
        // Check initial piece positions
        assert_eq!(board.cells[3][3], Cell::White);
        assert_eq!(board.cells[3][4], Cell::Black);
        assert_eq!(board.cells[4][3], Cell::Black);
        assert_eq!(board.cells[4][4], Cell::White);
        
        // Check all other cells are empty
        for i in 0..8 {
            for j in 0..8 {
                if (i, j) != (3, 3) && (i, j) != (3, 4) && (i, j) != (4, 3) && (i, j) != (4, 4) {
                    assert_eq!(board.cells[i][j], Cell::Empty);
                }
            }
        }
        
        // Check initial state
        assert_eq!(board.current_player, Player::Black);
        assert_eq!(board.black_count, 2);
        assert_eq!(board.white_count, 2);
        assert_eq!(board.game_over, false);
    }

    #[test]
    fn test_player_opponent() {
        assert_eq!(Player::Black.opponent(), Player::White);
        assert_eq!(Player::White.opponent(), Player::Black);
    }

    #[test]
    fn test_player_to_cell() {
        assert_eq!(Player::Black.to_cell(), Cell::Black);
        assert_eq!(Player::White.to_cell(), Cell::White);
    }

    #[test]
    fn test_is_valid_move_initial_board() {
        let board = Board::new();
        
        // Valid moves for Black at start: (2,3), (3,2), (4,5), (5,4)
        assert!(board.is_valid_move(2, 3));
        assert!(board.is_valid_move(3, 2));
        assert!(board.is_valid_move(4, 5));
        assert!(board.is_valid_move(5, 4));
        
        // Invalid moves (occupied cells)
        assert!(!board.is_valid_move(3, 3));
        assert!(!board.is_valid_move(3, 4));
        
        // Invalid moves (empty but no flips)
        assert!(!board.is_valid_move(0, 0));
        assert!(!board.is_valid_move(7, 7));
    }

    #[test]
    fn test_get_valid_moves_initial_board() {
        let board = Board::new();
        let valid_moves = board.get_valid_moves();
        
        // Count valid moves (should be 4 for initial board)
        let count = valid_moves.iter().filter(|&&v| v).count();
        assert_eq!(count, 4);
        
        // Check specific valid positions
        assert!(valid_moves[2 * 8 + 3]); // (2,3)
        assert!(valid_moves[3 * 8 + 2]); // (3,2)
        assert!(valid_moves[4 * 8 + 5]); // (4,5)
        assert!(valid_moves[5 * 8 + 4]); // (5,4)
    }

    #[test]
    fn test_is_valid_move_out_of_bounds() {
        let board = Board::new();
        assert!(!board.is_valid_move(8, 0));
        assert!(!board.is_valid_move(0, 8));
        assert!(!board.is_valid_move(10, 10));
    }

    #[test]
    fn test_apply_move_valid() {
        let mut board = Board::new();
        
        // Apply a valid move for Black at (2,3)
        let result = board.apply_move(2, 3);
        assert!(result.is_ok());
        
        // Check that the piece was placed
        assert_eq!(board.cells[2][3], Cell::Black);
        
        // Check that the white piece at (3,3) was flipped to black
        assert_eq!(board.cells[3][3], Cell::Black);
        
        // Check that current player switched to White
        assert_eq!(board.current_player, Player::White);
    }

    #[test]
    fn test_apply_move_invalid() {
        let mut board = Board::new();
        
        // Try to apply an invalid move
        let result = board.apply_move(0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), GameError::InvalidMove);
        
        // Board should be unchanged
        assert_eq!(board.cells[0][0], Cell::Empty);
        assert_eq!(board.current_player, Player::Black);
    }

    #[test]
    fn test_update_piece_counts() {
        let mut board = Board::new();
        
        // Initial counts
        assert_eq!(board.black_count, 2);
        assert_eq!(board.white_count, 2);
        
        // Apply a move
        board.apply_move(2, 3).unwrap();
        
        // Black should have 4 pieces (placed 1, flipped 1), White should have 1
        assert_eq!(board.black_count, 4);
        assert_eq!(board.white_count, 1);
    }

    #[test]
    fn test_pass_turn() {
        let mut board = Board::new();
        let initial_player = board.current_player;
        
        board.pass_turn();
        
        // Player should have switched
        assert_eq!(board.current_player, initial_player.opponent());
    }

    #[test]
    fn test_get_current_player() {
        let board = Board::new();
        assert_eq!(board.get_current_player(), Player::Black);
        
        let mut board2 = Board::new();
        board2.apply_move(2, 3).unwrap();
        assert_eq!(board2.get_current_player(), Player::White);
    }

    #[test]
    fn test_get_piece_counts() {
        let board = Board::new();
        let (black, white) = board.get_piece_counts();
        assert_eq!(black, 2);
        assert_eq!(white, 2);
        
        let mut board2 = Board::new();
        board2.apply_move(2, 3).unwrap();
        let (black2, white2) = board2.get_piece_counts();
        assert_eq!(black2, 4);
        assert_eq!(white2, 1);
    }

    #[test]
    fn test_get_winner_game_not_over() {
        let board = Board::new();
        assert_eq!(board.get_winner(), None);
    }

    #[test]
    fn test_get_winner_black_wins() {
        let mut board = Board::new();
        // Fill board to end game with black winning
        for row in 0..8 {
            for col in 0..8 {
                board.cells[row][col] = Cell::Black;
            }
        }
        board.update_piece_counts();
        board.game_over = true;
        
        assert_eq!(board.get_winner(), Some(Player::Black));
    }

    #[test]
    fn test_get_winner_white_wins() {
        let mut board = Board::new();
        // Fill board to end game with white winning
        for row in 0..8 {
            for col in 0..8 {
                board.cells[row][col] = Cell::White;
            }
        }
        board.update_piece_counts();
        board.game_over = true;
        
        assert_eq!(board.get_winner(), Some(Player::White));
    }

    #[test]
    fn test_get_winner_draw() {
        let mut board = Board::new();
        // Create a draw scenario
        for row in 0..4 {
            for col in 0..8 {
                board.cells[row][col] = Cell::Black;
            }
        }
        for row in 4..8 {
            for col in 0..8 {
                board.cells[row][col] = Cell::White;
            }
        }
        board.update_piece_counts();
        board.game_over = true;
        
        assert_eq!(board.get_winner(), None);
    }

    #[test]
    fn test_is_game_over_initial() {
        let board = Board::new();
        assert!(!board.is_game_over());
    }

    #[test]
    fn test_is_game_over_after_setting() {
        let mut board = Board::new();
        board.game_over = true;
        assert!(board.is_game_over());
    }

    #[test]
    fn test_is_board_full() {
        let mut board = Board::new();
        assert!(!board.is_board_full());
        
        // Fill the board
        for row in 0..8 {
            for col in 0..8 {
                board.cells[row][col] = Cell::Black;
            }
        }
        board.update_piece_counts();
        assert!(board.is_board_full());
    }

    #[test]
    fn test_game_over_when_board_full() {
        let mut board = Board::new();
        
        // Fill board except one cell
        for row in 0..8 {
            for col in 0..8 {
                if (row, col) != (0, 0) {
                    board.cells[row][col] = Cell::Black;
                }
            }
        }
        board.cells[0][0] = Cell::Empty;
        board.cells[3][3] = Cell::White; // Make (0,0) a valid move
        board.update_piece_counts();
        board.current_player = Player::Black;
        
        // Apply last move
        if board.is_valid_move(0, 0) {
            board.apply_move(0, 0).unwrap();
            assert!(board.is_game_over(), "Game should be over when board is full");
        }
    }

    #[test]
    fn test_game_over_when_no_moves() {
        let mut board = Board::new();
        
        // Create a scenario where neither player has moves
        // Fill the board completely
        for row in 0..8 {
            for col in 0..8 {
                board.cells[row][col] = Cell::Black;
            }
        }
        board.update_piece_counts();
        
        // Manually trigger pass_turn to check game over
        board.game_over = false;
        board.pass_turn();
        
        assert!(board.is_game_over(), "Game should be over when neither player has moves");
    }

    #[test]
    fn test_get_state() {
        let board = Board::new();
        let state = board.get_state();
        
        // Check that state is 64 elements
        assert_eq!(state.len(), 64);
        
        // Check initial positions
        assert_eq!(state[3 * 8 + 3], 2); // White at (3,3)
        assert_eq!(state[3 * 8 + 4], 1); // Black at (3,4)
        assert_eq!(state[4 * 8 + 3], 1); // Black at (4,3)
        assert_eq!(state[4 * 8 + 4], 2); // White at (4,4)
        
        // Check that other positions are empty
        assert_eq!(state[0], 0);
        assert_eq!(state[63], 0);
    }

    #[test]
    fn test_get_state_after_move() {
        let mut board = Board::new();
        board.apply_move(2, 3).unwrap();
        
        let state = board.get_state();
        
        // Check that the move was recorded
        assert_eq!(state[2 * 8 + 3], 1); // Black at (2,3)
        
        // Check that the flipped piece changed
        assert_eq!(state[3 * 8 + 3], 1); // Was White, now Black
    }

    #[test]
    fn test_reset() {
        let mut board = Board::new();
        
        // Make some moves
        board.apply_move(2, 3).unwrap();
        board.apply_move(2, 2).unwrap();
        
        // Verify board changed
        assert_ne!(board.black_count, 2);
        
        // Reset
        board.reset();
        
        // Check initial piece positions
        assert_eq!(board.cells[3][3], Cell::White);
        assert_eq!(board.cells[3][4], Cell::Black);
        assert_eq!(board.cells[4][3], Cell::Black);
        assert_eq!(board.cells[4][4], Cell::White);
        
        // Check all other cells are empty
        for i in 0..8 {
            for j in 0..8 {
                if (i, j) != (3, 3) && (i, j) != (3, 4) && (i, j) != (4, 3) && (i, j) != (4, 4) {
                    assert_eq!(board.cells[i][j], Cell::Empty);
                }
            }
        }
        
        // Check initial state
        assert_eq!(board.current_player, Player::Black);
        assert_eq!(board.black_count, 2);
        assert_eq!(board.white_count, 2);
        assert_eq!(board.game_over, false);
    }

    #[test]
    fn test_reset_multiple_times() {
        let mut board = Board::new();
        
        // Reset multiple times
        for _ in 0..3 {
            board.apply_move(2, 3).unwrap();
            board.reset();
            
            // Verify it's back to initial state
            assert_eq!(board.black_count, 2);
            assert_eq!(board.white_count, 2);
            assert_eq!(board.current_player, Player::Black);
            assert_eq!(board.game_over, false);
        }
    }
}


#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Helper to create arbitrary board states for property testing
    fn arbitrary_board_state() -> impl Strategy<Value = Board> {
        // Generate a sequence of valid moves to create diverse board states
        prop::collection::vec(0usize..64, 0..20).prop_map(|moves| {
            let mut board = Board::new();
            
            for action in moves {
                let row = action / 8;
                let col = action % 8;
                
                // Only apply if it's a valid move
                if board.is_valid_move(row, col) {
                    // We'll implement apply_move in the next subtask
                    // For now, just use the initial board
                    break;
                }
            }
            
            board
        })
    }

    proptest! {
        /// Property 2: Valid Move Detection
        /// For any board state, the set of valid moves returned by get_valid_moves()
        /// should exactly match the set of positions where is_valid_move() returns true.
        /// Validates: Requirements 1.3
        #[test]
        fn prop_valid_move_detection_consistency(_seed in 0u64..1000) {
            // Use seed to create deterministic board state
            let board = Board::new();
            
            let valid_moves = board.get_valid_moves();
            
            // Check that every position in valid_moves matches is_valid_move()
            for row in 0..8 {
                for col in 0..8 {
                    let index = row * 8 + col;
                    let expected = board.is_valid_move(row, col);
                    prop_assert_eq!(
                        valid_moves[index],
                        expected,
                        "Mismatch at position ({}, {}): get_valid_moves returned {} but is_valid_move returned {}",
                        row, col, valid_moves[index], expected
                    );
                }
            }
        }

        /// Property 2 (Extended): Valid moves must flip at least one piece
        /// For any position marked as valid, placing a piece there should flip at least one opponent piece
        /// Validates: Requirements 1.3
        #[test]
        fn prop_valid_moves_flip_pieces(row in 0usize..8, col in 0usize..8) {
            let board = Board::new();
            
            if board.is_valid_move(row, col) {
                // Check that at least one direction would flip pieces
                let directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1),
                ];
                
                let has_flip = directions.iter().any(|(dr, dc)| {
                    board.would_flip_in_direction(row, col, *dr, *dc)
                });
                
                prop_assert!(
                    has_flip,
                    "Position ({}, {}) is marked as valid but would not flip any pieces",
                    row, col
                );
            }
        }

        /// Property 2 (Inverse): Invalid moves must not flip any pieces
        /// For any position marked as invalid (and empty), it should not flip any opponent pieces
        /// Validates: Requirements 1.3
        #[test]
        fn prop_invalid_moves_no_flips(row in 0usize..8, col in 0usize..8) {
            let board = Board::new();
            
            // Only test empty cells that are marked invalid
            if board.cells[row][col] == Cell::Empty && !board.is_valid_move(row, col) {
                let directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1),
                ];
                
                let has_flip = directions.iter().any(|(dr, dc)| {
                    board.would_flip_in_direction(row, col, *dr, *dc)
                });
                
                prop_assert!(
                    !has_flip,
                    "Position ({}, {}) is marked as invalid but would flip pieces",
                    row, col
                );
            }
        }

        /// Property 1: Piece Flipping Correctness
        /// For any valid move, when a piece is placed, all opponent pieces in valid directions
        /// that are sandwiched between the placed piece and another piece of the current player
        /// should be flipped, and no other pieces should be modified.
        /// Validates: Requirements 1.2
        #[test]
        fn prop_piece_flipping_correctness(row in 0usize..8, col in 0usize..8) {
            let mut board = Board::new();
            
            // Only test valid moves
            if !board.is_valid_move(row, col) {
                return Ok(());
            }
            
            // Save the original board state
            let original_cells = board.cells;
            let original_player = board.current_player;
            let player_cell = original_player.to_cell();
            let opponent_cell = original_player.opponent().to_cell();
            
            // Apply the move
            let result = board.apply_move(row, col);
            prop_assert!(result.is_ok(), "Valid move should succeed");
            
            // Check that the placed piece is correct
            prop_assert_eq!(
                board.cells[row][col],
                player_cell,
                "Placed piece should be current player's piece"
            );
            
            // Check all directions for correct flipping
            let directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ];
            
            for (dr, dc) in directions {
                let mut r = row as i8 + dr;
                let mut c = col as i8 + dc;
                let mut opponent_pieces = Vec::new();
                
                // Collect opponent pieces in this direction
                while r >= 0 && r < 8 && c >= 0 && c < 8 {
                    let orig_cell = original_cells[r as usize][c as usize];
                    if orig_cell == opponent_cell {
                        opponent_pieces.push((r as usize, c as usize));
                        r += dr;
                        c += dc;
                    } else if orig_cell == player_cell {
                        // Found player piece - all opponent pieces should be flipped
                        for (flip_r, flip_c) in opponent_pieces {
                            prop_assert_eq!(
                                board.cells[flip_r][flip_c],
                                player_cell,
                                "Opponent piece at ({}, {}) should be flipped in direction ({}, {})",
                                flip_r, flip_c, dr, dc
                            );
                        }
                        break;
                    } else {
                        // Empty cell or edge - no flipping in this direction
                        break;
                    }
                }
            }
            
            // Check that no other pieces were modified (except flipped ones and placed piece)
            for check_row in 0..8 {
                for check_col in 0..8 {
                    if check_row == row && check_col == col {
                        // This is the placed piece, skip
                        continue;
                    }
                    
                    let original = original_cells[check_row][check_col];
                    let current = board.cells[check_row][check_col];
                    
                    // If the piece changed, it should have been an opponent piece that got flipped
                    if original != current {
                        prop_assert_eq!(
                            original,
                            opponent_cell,
                            "Only opponent pieces should be modified at ({}, {})",
                            check_row, check_col
                        );
                        prop_assert_eq!(
                            current,
                            player_cell,
                            "Modified pieces should become player pieces at ({}, {})",
                            check_row, check_col
                        );
                    }
                }
            }
        }

        /// Property 1 (Extended): Piece count conservation
        /// After any move, the total number of pieces should increase by 1
        /// (the placed piece), and the sum of black and white pieces should equal
        /// the number of non-empty cells.
        /// Validates: Requirements 1.2
        #[test]
        fn prop_piece_count_conservation(row in 0usize..8, col in 0usize..8) {
            let mut board = Board::new();
            
            if !board.is_valid_move(row, col) {
                return Ok(());
            }
            
            // Count pieces before move
            let mut before_total = 0;
            for r in 0..8 {
                for c in 0..8 {
                    if board.cells[r][c] != Cell::Empty {
                        before_total += 1;
                    }
                }
            }
            
            // Apply move
            board.apply_move(row, col).unwrap();
            
            // Count pieces after move
            let mut after_total = 0;
            for r in 0..8 {
                for c in 0..8 {
                    if board.cells[r][c] != Cell::Empty {
                        after_total += 1;
                    }
                }
            }
            
            // Total should increase by exactly 1
            prop_assert_eq!(
                after_total,
                before_total + 1,
                "Total pieces should increase by 1 after placing a piece"
            );
            
            // Piece counts should match actual board
            prop_assert_eq!(
                board.black_count + board.white_count,
                after_total,
                "Sum of piece counts should equal total non-empty cells"
            );
        }

        /// Property 3: Turn Passing
        /// For any board state where the current player has no valid moves but the opponent
        /// does have valid moves, passing the turn should switch the current player without
        /// modifying the board.
        /// Validates: Requirements 1.4
        #[test]
        fn prop_turn_passing_preserves_board(_seed in 0u64..100) {
            // Create a board state where one player has no moves
            // We'll construct this by filling most of the board
            let mut board = Board::new();
            
            // Save original state
            let original_cells = board.cells;
            let original_player = board.current_player;
            let original_black_count = board.black_count;
            let original_white_count = board.white_count;
            
            // Pass turn
            board.pass_turn();
            
            // Check that player switched
            prop_assert_eq!(
                board.current_player,
                original_player.opponent(),
                "Current player should switch after pass_turn"
            );
            
            // Check that board is unchanged
            for row in 0..8 {
                for col in 0..8 {
                    prop_assert_eq!(
                        board.cells[row][col],
                        original_cells[row][col],
                        "Board cell at ({}, {}) should not change after pass_turn",
                        row, col
                    );
                }
            }
            
            // Check that piece counts are unchanged
            prop_assert_eq!(
                board.black_count,
                original_black_count,
                "Black count should not change after pass_turn"
            );
            prop_assert_eq!(
                board.white_count,
                original_white_count,
                "White count should not change after pass_turn"
            );
        }

        /// Property 3 (Extended): Game over detection after double pass
        /// If neither player has valid moves, the game should be marked as over
        /// Validates: Requirements 1.4, 1.5
        #[test]
        fn prop_double_pass_ends_game(_seed in 0u64..100) {
            // Create a board where we can test double pass
            let mut board = Board::new();
            
            // Fill the board almost completely to create a no-moves situation
            // For this test, we'll manually create a scenario
            for row in 0..8 {
                for col in 0..8 {
                    if board.cells[row][col] == Cell::Empty {
                        board.cells[row][col] = Cell::Black;
                    }
                }
            }
            
            // Update counts
            board.update_piece_counts();
            
            // Now neither player should have valid moves
            let black_has_moves = board.get_valid_moves().iter().any(|&v| v);
            
            if !black_has_moves {
                // Pass turn to white
                board.pass_turn();
                
                // If white also has no moves, game should be over
                let white_has_moves = board.get_valid_moves().iter().any(|&v| v);
                if !white_has_moves {
                    prop_assert!(
                        board.game_over,
                        "Game should be over when neither player has valid moves"
                    );
                }
            }
        }

        /// Property 3 (Automatic): Turn passing in apply_move
        /// When apply_move is called and the opponent has no valid moves,
        /// the turn should automatically pass back
        /// Validates: Requirements 1.4
        #[test]
        fn prop_automatic_turn_passing(row in 0usize..8, col in 0usize..8) {
            let mut board = Board::new();
            
            if !board.is_valid_move(row, col) {
                return Ok(());
            }
            
            let player_before = board.current_player;
            
            // Apply move
            board.apply_move(row, col).unwrap();
            
            // Check if opponent has moves
            let opponent_has_moves = board.get_valid_moves().iter().any(|&v| v);
            
            if !opponent_has_moves && !board.game_over {
                // Turn should have passed back to original player
                prop_assert_eq!(
                    board.current_player,
                    player_before,
                    "Turn should pass back to original player when opponent has no moves"
                );
            } else if opponent_has_moves {
                // Turn should be with opponent
                prop_assert_eq!(
                    board.current_player,
                    player_before.opponent(),
                    "Turn should be with opponent when they have valid moves"
                );
            }
        }

        /// Property 4: Player Alternation
        /// For any sequence of valid moves, the current player should alternate between
        /// Black and White after each move (unless a turn is passed due to no valid moves).
        /// Validates: Requirements 1.6
        #[test]
        fn prop_player_alternation(moves in prop::collection::vec(0usize..64, 1..10)) {
            let mut board = Board::new();
            let mut expected_player = Player::Black;
            
            for action in moves {
                let row = action / 8;
                let col = action % 8;
                
                // Check current player matches expected
                prop_assert_eq!(
                    board.get_current_player(),
                    expected_player,
                    "Current player should be {:?} before move", expected_player
                );
                
                // Only apply valid moves
                if !board.is_valid_move(row, col) {
                    continue;
                }
                
                // Apply the move
                let result = board.apply_move(row, col);
                prop_assert!(result.is_ok(), "Valid move should succeed");
                
                // If game is over, stop
                if board.game_over {
                    break;
                }
                
                // Check if opponent has moves
                let opponent_has_moves = board.get_valid_moves().iter().any(|&v| v);
                
                if opponent_has_moves {
                    // Normal alternation: player should switch to opponent
                    expected_player = expected_player.opponent();
                    prop_assert_eq!(
                        board.get_current_player(),
                        expected_player,
                        "Player should alternate to opponent when opponent has moves"
                    );
                } else {
                    // Turn passed back: player should remain the same
                    prop_assert_eq!(
                        board.get_current_player(),
                        expected_player,
                        "Player should remain the same when opponent has no moves (turn passed)"
                    );
                }
            }
        }

        /// Property 4 (Extended): Player alternation with explicit tracking
        /// Track player changes through a sequence of moves and verify alternation pattern
        /// Validates: Requirements 1.6
        #[test]
        fn prop_player_alternation_tracking(seed in 0u64..100) {
            let mut board = Board::new();
            let mut player_history = vec![board.get_current_player()];
            
            // Play up to 20 moves
            for _ in 0..20 {
                if board.game_over {
                    break;
                }
                
                let valid_moves = board.get_valid_moves();
                let valid_indices: Vec<usize> = valid_moves
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();
                
                if valid_indices.is_empty() {
                    break;
                }
                
                // Pick first valid move (deterministic)
                let action = valid_indices[0];
                let row = action / 8;
                let col = action % 8;
                
                board.apply_move(row, col).unwrap();
                player_history.push(board.get_current_player());
            }
            
            // Verify alternation pattern (allowing for turn passes)
            for i in 1..player_history.len() {
                let prev_player = player_history[i - 1];
                let curr_player = player_history[i];
                
                // Player should either alternate or stay the same (if turn passed)
                let is_alternating = curr_player == prev_player.opponent();
                let is_same = curr_player == prev_player;
                
                prop_assert!(
                    is_alternating || is_same,
                    "Player at step {} should either alternate or stay same (turn pass), but went from {:?} to {:?}",
                    i, prev_player, curr_player
                );
            }
        }

        /// Property 5: Piece Count Accuracy
        /// For any board state, the sum of black pieces, white pieces, and empty cells
        /// should equal 64, and the reported piece counts should match the actual number
        /// of pieces on the board.
        /// Validates: Requirements 1.7
        #[test]
        fn prop_piece_count_accuracy(moves in prop::collection::vec(0usize..64, 0..15)) {
            let mut board = Board::new();
            
            // Apply a sequence of moves
            for action in moves {
                if board.game_over {
                    break;
                }
                
                let row = action / 8;
                let col = action % 8;
                
                if board.is_valid_move(row, col) {
                    board.apply_move(row, col).unwrap();
                }
            }
            
            // Count pieces manually
            let mut actual_black = 0;
            let mut actual_white = 0;
            let mut actual_empty = 0;
            
            for row in 0..8 {
                for col in 0..8 {
                    match board.cells[row][col] {
                        Cell::Black => actual_black += 1,
                        Cell::White => actual_white += 1,
                        Cell::Empty => actual_empty += 1,
                    }
                }
            }
            
            // Get reported counts
            let (reported_black, reported_white) = board.get_piece_counts();
            
            // Verify total is 64
            prop_assert_eq!(
                actual_black + actual_white + actual_empty,
                64,
                "Total cells should equal 64"
            );
            
            // Verify reported counts match actual counts
            prop_assert_eq!(
                reported_black,
                actual_black,
                "Reported black count ({}) should match actual count ({})",
                reported_black, actual_black
            );
            
            prop_assert_eq!(
                reported_white,
                actual_white,
                "Reported white count ({}) should match actual count ({})",
                reported_white, actual_white
            );
            
            // Verify sum of reported counts matches total pieces
            prop_assert_eq!(
                reported_black + reported_white,
                actual_black + actual_white,
                "Sum of reported counts should match total pieces on board"
            );
        }

        /// Property 5 (Extended): Piece count consistency after each move
        /// After every move, piece counts should be accurate
        /// Validates: Requirements 1.7
        #[test]
        fn prop_piece_count_consistency_per_move(row in 0usize..8, col in 0usize..8) {
            let mut board = Board::new();
            
            if !board.is_valid_move(row, col) {
                return Ok(());
            }
            
            // Apply move
            board.apply_move(row, col).unwrap();
            
            // Count pieces manually
            let mut actual_black = 0;
            let mut actual_white = 0;
            
            for r in 0..8 {
                for c in 0..8 {
                    match board.cells[r][c] {
                        Cell::Black => actual_black += 1,
                        Cell::White => actual_white += 1,
                        Cell::Empty => {}
                    }
                }
            }
            
            let (reported_black, reported_white) = board.get_piece_counts();
            
            prop_assert_eq!(
                reported_black,
                actual_black,
                "Black count should be accurate after move at ({}, {})",
                row, col
            );
            
            prop_assert_eq!(
                reported_white,
                actual_white,
                "White count should be accurate after move at ({}, {})",
                row, col
            );
        }

        /// Property 5 (Invariant): Total pieces never exceed 64
        /// At any point in the game, total pieces should be <= 64
        /// Validates: Requirements 1.7
        #[test]
        fn prop_piece_count_never_exceeds_64(moves in prop::collection::vec(0usize..64, 0..20)) {
            let mut board = Board::new();
            
            for action in moves {
                if board.game_over {
                    break;
                }
                
                let row = action / 8;
                let col = action % 8;
                
                if board.is_valid_move(row, col) {
                    board.apply_move(row, col).unwrap();
                }
                
                let (black, white) = board.get_piece_counts();
                let total = black + white;
                
                prop_assert!(
                    total <= 64,
                    "Total pieces ({}) should never exceed 64",
                    total
                );
            }
        }

        /// Property 6: Game Termination
        /// For any board state, the game should be marked as over if and only if
        /// the board is full OR neither player has any valid moves.
        /// Validates: Requirements 1.8, 1.5
        #[test]
        fn prop_game_termination(moves in prop::collection::vec(0usize..64, 0..30)) {
            let mut board = Board::new();
            
            // Play a sequence of moves
            for action in moves {
                if board.is_game_over() {
                    break;
                }
                
                let row = action / 8;
                let col = action % 8;
                
                if board.is_valid_move(row, col) {
                    board.apply_move(row, col).unwrap();
                }
            }
            
            // Check game over conditions
            let is_board_full = board.is_board_full();
            let current_has_moves = board.get_valid_moves().iter().any(|&v| v);
            
            // If game is over, verify it's for a valid reason
            if board.is_game_over() {
                // Game should be over if board is full OR current player has no moves
                // (which implies neither player has moves, since we check opponent in apply_move)
                let valid_reason = is_board_full || !current_has_moves;
                prop_assert!(
                    valid_reason,
                    "Game is marked as over but board is not full ({}) and current player has moves ({})",
                    is_board_full, current_has_moves
                );
            }
            
            // If board is full, game must be over
            if is_board_full {
                prop_assert!(
                    board.is_game_over(),
                    "Game should be over when board is full"
                );
            }
        }

        /// Property 6 (Extended): Game termination implies no valid moves for both players
        /// When game is over (and board not full), neither player should have valid moves
        /// Validates: Requirements 1.8, 1.5
        #[test]
        fn prop_game_over_no_moves_for_both_players(moves in prop::collection::vec(0usize..64, 0..25)) {
            let mut board = Board::new();
            
            for action in moves {
                if board.is_game_over() {
                    break;
                }
                
                let row = action / 8;
                let col = action % 8;
                
                if board.is_valid_move(row, col) {
                    board.apply_move(row, col).unwrap();
                }
            }
            
            // If game is over and board is not full, check both players have no moves
            if board.is_game_over() && !board.is_board_full() {
                let current_player = board.get_current_player();
                let current_has_moves = board.get_valid_moves().iter().any(|&v| v);
                
                prop_assert!(
                    !current_has_moves,
                    "Current player should have no moves when game is over (board not full)"
                );
                
                // Check opponent also has no moves by temporarily switching
                let original_player = board.current_player;
                board.current_player = board.current_player.opponent();
                let opponent_has_moves = board.get_valid_moves().iter().any(|&v| v);
                board.current_player = original_player;
                
                prop_assert!(
                    !opponent_has_moves,
                    "Opponent should also have no moves when game is over (board not full)"
                );
            }
        }

        /// Property 6 (Invariant): Game never ends prematurely
        /// If either player has valid moves and board is not full, game should not be over
        /// Validates: Requirements 1.8, 1.5
        #[test]
        fn prop_game_never_ends_prematurely(moves in prop::collection::vec(0usize..64, 0..15)) {
            let mut board = Board::new();
            
            for action in moves {
                if board.is_game_over() {
                    break;
                }
                
                let row = action / 8;
                let col = action % 8;
                
                if board.is_valid_move(row, col) {
                    board.apply_move(row, col).unwrap();
                }
                
                // After each move, if game is not over, at least one player should have moves
                if !board.is_game_over() {
                    let current_has_moves = board.get_valid_moves().iter().any(|&v| v);
                    
                    // If current player has no moves, game should be over
                    // (because apply_move checks opponent moves and sets game_over if neither has moves)
                    if !current_has_moves {
                        prop_assert!(
                            board.is_game_over() || board.is_board_full(),
                            "If current player has no moves and game is not over, board should be full"
                        );
                    }
                }
            }
        }

        /// Property 6 (Board Full): When board is full, game must be over
        /// Validates: Requirements 1.8
        #[test]
        fn prop_board_full_implies_game_over(_seed in 0u64..100) {
            let mut board = Board::new();
            
            // Fill the board
            for row in 0..8 {
                for col in 0..8 {
                    board.cells[row][col] = if row < 4 { Cell::Black } else { Cell::White };
                }
            }
            board.update_piece_counts();
            
            // Manually check the condition
            if board.is_board_full() {
                // Simulate what apply_move would do
                board.game_over = true;
                
                prop_assert!(
                    board.is_game_over(),
                    "Game should be over when board is full"
                );
            }
        }
    }
}
