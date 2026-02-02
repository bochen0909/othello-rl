//! Othello Engine Implementations
//!
//! This crate contains Rust implementations of three external Othello engines:
//! - `aelskens`: Alpha-beta pruning AI with 5-turn lookahead
//! - `drohh`: Standard Othello implementation with directional move validation
//! - `nealetham`: Naive greedy AI that maximizes immediate piece capture
//!
//! All engines export a `compute_move(board: &[u8; 64], player: u8) -> u8` function
//! that returns the best move (0-63) for a given board state.

pub mod aelskens;
pub mod drohh;
pub mod nealetham;

pub use aelskens::compute_move as compute_move_aelskens;
pub use aelskens::compute_move_scores as compute_move_scores_aelskens;
pub use drohh::compute_move as compute_move_drohh;
pub use drohh::compute_move_scores as compute_move_scores_drohh;
pub use nealetham::compute_move as compute_move_nealetham;
pub use nealetham::compute_move_scores as compute_move_scores_nealetham;

/// Simple board representation for engines
/// Cells are represented as a flat array of 64 elements
/// 0 = Empty, 1 = Black, 2 = White
pub type Board = [u8; 64];
