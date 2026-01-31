use crate::{Board, GameError, Player};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use othello_engines::{compute_move_aelskens, compute_move_drohh, compute_move_nealetham};
use pyo3::exceptions::PyValueError;
/// PyO3 bindings for the Othello game engine
/// Exposes the Rust Board implementation to Python
use pyo3::prelude::*;

/// Python wrapper for the Othello game board
///
/// This class provides a Python interface to the high-performance Rust game engine.
/// All game logic is implemented in Rust for optimal performance.
#[pyclass]
pub struct OthelloGame {
    board: Board,
}

#[pymethods]
impl OthelloGame {
    /// Create a new Othello game with the standard initial setup
    ///
    /// Returns:
    ///     OthelloGame: A new game instance with 4 pieces in the center
    #[new]
    pub fn new() -> Self {
        Self {
            board: Board::new(),
        }
    }

    /// Reset the game to initial state
    ///
    /// Clears the board and sets up the standard 4-piece starting position.
    pub fn reset(&mut self) {
        self.board.reset();
    }

    /// Apply a move to the board
    ///
    /// Args:
    ///     action (int): Position on the board (0-63), where action = row * 8 + col
    ///
    /// Returns:
    ///     tuple: (valid, pieces_flipped, game_over)
    ///         - valid (bool): Whether the move was valid and applied
    ///         - pieces_flipped (int): Number of opponent pieces flipped
    ///         - game_over (bool): Whether the game has ended
    ///
    /// Raises:
    ///     ValueError: If action is out of range [0, 63]
    pub fn step(&mut self, action: usize) -> PyResult<(bool, u8, bool)> {
        // Validate action range
        if action >= 64 {
            return Err(PyValueError::new_err(format!(
                "Action {} is out of range. Must be between 0 and 63 (inclusive).",
                action
            )));
        }

        let row = action / 8;
        let col = action % 8;

        match self.board.apply_move(row, col) {
            Ok(flipped) => Ok((true, flipped, self.board.is_game_over())),
            Err(GameError::InvalidMove) => {
                // Return false for invalid move, but don't raise exception
                // This allows the Python layer to handle invalid moves gracefully
                Ok((false, 0, self.board.is_game_over()))
            }
        }
    }

    /// Get the current board state as a 2D numpy array
    ///
    /// Returns:
    ///     np.ndarray: Shape (8, 8) with dtype uint8
    ///         - 0 = Empty cell
    ///         - 1 = Black piece
    ///         - 2 = White piece
    pub fn get_board<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<u8>> {
        let state = self.board.get_state();

        // Convert flat array to 2D ndarray
        let array = Array2::from_shape_fn((8, 8), |(row, col)| state[row * 8 + col]);

        Ok(PyArray2::from_owned_array(py, array))
    }

    /// Get valid moves for the current player
    ///
    /// Returns:
    ///     np.ndarray: Shape (64,) with dtype bool
    ///         - True at index i means position i is a valid move
    ///         - Index i corresponds to position (i // 8, i % 8)
    pub fn get_valid_moves<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<bool>> {
        let moves = self.board.get_valid_moves();
        Ok(PyArray1::from_slice(py, &moves))
    }

    /// Get the current player
    ///
    /// Returns:
    ///     int: 0 for Black, 1 for White
    pub fn get_current_player(&self) -> u8 {
        match self.board.get_current_player() {
            Player::Black => 0,
            Player::White => 1,
        }
    }

    /// Get piece counts for both players
    ///
    /// Returns:
    ///     tuple: (black_count, white_count)
    pub fn get_piece_counts(&self) -> (u8, u8) {
        self.board.get_piece_counts()
    }

    /// Get the winner of the game
    ///
    /// Returns:
    ///     int:
    ///         - 0 = Black wins
    ///         - 1 = White wins
    ///         - 2 = Draw
    ///         - 3 = Game not finished
    pub fn get_winner(&self) -> u8 {
        if !self.board.is_game_over() {
            return 3;
        }

        match self.board.get_winner() {
            Some(Player::Black) => 0,
            Some(Player::White) => 1,
            None => 2, // Draw
        }
    }
}

/// Compute move for aelskels engine
///
/// Args:
///     board (list): Flat board state as 64 elements (0=Empty, 1=Black, 2=White)
///     player (int): Current player (1=Black, 2=White)
///
/// Returns:
///     int: Move index (0-63) or 255 if no valid moves
#[pyfunction]
fn compute_move_aelskels_py(board: Vec<u8>, player: u8) -> PyResult<u8> {
    if board.len() != 64 {
        return Err(PyValueError::new_err(format!(
            "Board must have exactly 64 elements, got {}",
            board.len()
        )));
    }

    let mut board_array: [u8; 64] = [0; 64];
    board_array.copy_from_slice(&board);

    Ok(compute_move_aelskens(&board_array, player))
}

/// Compute move for drohh engine
///
/// Args:
///     board (list): Flat board state as 64 elements (0=Empty, 1=Black, 2=White)
///     player (int): Current player (1=Black, 2=White)
///
/// Returns:
///     int: Move index (0-63) or 255 if no valid moves
#[pyfunction]
fn compute_move_drohh_py(board: Vec<u8>, player: u8) -> PyResult<u8> {
    if board.len() != 64 {
        return Err(PyValueError::new_err(format!(
            "Board must have exactly 64 elements, got {}",
            board.len()
        )));
    }

    let mut board_array: [u8; 64] = [0; 64];
    board_array.copy_from_slice(&board);

    Ok(compute_move_drohh(&board_array, player))
}

/// Compute move for nealetham engine
///
/// Args:
///     board (list): Flat board state as 64 elements (0=Empty, 1=Black, 2=White)
///     player (int): Current player (1=Black, 2=White)
///
/// Returns:
///     int: Move index (0-63) or 255 if no valid moves
#[pyfunction]
fn compute_move_nealetham_py(board: Vec<u8>, player: u8) -> PyResult<u8> {
    if board.len() != 64 {
        return Err(PyValueError::new_err(format!(
            "Board must have exactly 64 elements, got {}",
            board.len()
        )));
    }

    let mut board_array: [u8; 64] = [0; 64];
    board_array.copy_from_slice(&board);

    Ok(compute_move_nealetham(&board_array, player))
}

/// Python module definition
///
/// This module can be imported in Python as `othello_rust`
#[pymodule]
fn othello_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OthelloGame>()?;
    m.add_function(wrap_pyfunction!(compute_move_aelskels_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_move_drohh_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_move_nealetham_py, m)?)?;
    Ok(())
}
