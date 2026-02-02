//! Aelskels Engine - Alpha-Beta Pruning AI
//!
//! This module implements the aelskels Othello engine, which uses an alpha-beta pruning
//! minimax algorithm with a lookahead depth of 5 moves. It's based on a C++ implementation
//! that featured a `Player` class and `Map` board representation.
//!
//! Algorithm Details:
//! - Uses minimax search with alpha-beta pruning optimization
//! - Searches 5 moves deep (depth = 5)
//! - Combines multiple heuristics for position evaluation:
//!   1. Token parity (normalized difference in piece counts)
//!   2. Positional weights (static weight table favoring corners and edges)
//!   3. Mobility (normalized difference in available moves)
//!   4. Corner captures (weighted evaluation of corner control)
//!
//! Key characteristics:
//! - Deterministic move selection (same board state always produces same move)
//! - More sophisticated than greedy approaches (looks ahead multiple moves)
//! - Can be slower due to deep lookahead (may take multiple seconds for complex positions)
//! - Guarantees optimal play within its 5-move horizon

use crate::Board;

// Weighted positions for heuristic evaluation
const WEIGHTED_POSITIONS: [i32; 64] = [
    20, -3, 11, 8, 8, 11, -3, 20, -3, -7, -4, 1, 1, -4, -7, -3, 11, -4, 2, 2, 2, 2, -4, 11, 8, 1,
    2, -3, -3, 2, 1, 8, 8, 1, 2, -3, -3, 2, 1, 8, 11, -4, 2, 2, 2, 2, -4, 11, -3, -7, -4, 1, 1, -4,
    -7, -3, 20, -3, 11, 8, 8, 11, -3, 20,
];

// Constants for cell values
const EMPTY: u8 = 0;
const BLACK: u8 = 1;
const WHITE: u8 = 2;

/// Compute the best move for the aelskels AI engine
///
/// # Arguments
/// * `board` - Current board state as [u8; 64] where 0=Empty, 1=Black, 2=White
/// * `player` - Current player (1 = Black, 2 = White)
///
/// # Returns
/// Action index (0-63) representing the best move, or u8::MAX if no valid moves
pub fn compute_move(board: &Board, player: u8) -> u8 {
    // Get valid moves
    let valid_moves = get_valid_moves(board, player);

    // If no valid moves, return u8::MAX
    if !valid_moves.iter().any(|&x| x) {
        return u8::MAX;
    }

    // Use alpha-beta pruning with depth 5
    let depth = 5;
    let (_, best_move) = alpha_beta_pruning_minimax(board, player, depth, i32::MIN, i32::MAX);

    best_move.unwrap_or(u8::MAX)
}

/// Compute heuristic scores for all legal moves on the board
///
/// # Arguments
/// * `board` - Current board state as [u8; 64] where 0=Empty, 1=Black, 2=White
/// * `player` - Current player (1 = Black, 2 = White)
///
/// # Returns
/// Array of 64 scores (i32) where legal moves have non-zero scores and illegal moves have 0
pub fn compute_move_scores(board: &Board, player: u8) -> [i32; 64] {
    let mut scores = [0i32; 64];
    let valid_moves = get_valid_moves(board, player);

    // For each position on the board
    for i in 0..64 {
        if valid_moves[i] {
            // This is a legal move - evaluate the board after this move
            let row = i / 8;
            let col = i % 8;

            if let Some(new_board) = apply_move_to_board(board, player, row, col) {
                // Use the heuristic to score this resulting position
                scores[i] = heuristics_othello(&new_board, player) as i32;
            }
        }
        // If not a valid move, score remains 0
    }

    scores
}

/// Get all valid moves for a player as a 64-element bool array
fn get_valid_moves(board: &Board, player: u8) -> [bool; 64] {
    let mut valid_moves = [false; 64];

    for row in 0..8 {
        for col in 0..8 {
            let index = row * 8 + col;
            if board[index] == EMPTY && is_valid_move(board, player, row, col) {
                valid_moves[index] = true;
            }
        }
    }

    valid_moves
}

/// Check if a move is valid at position (row, col)
fn is_valid_move(board: &Board, player: u8, row: usize, col: usize) -> bool {
    // Must be empty cell
    let index = row * 8 + col;
    if board[index] != EMPTY {
        return false;
    }

    // Check all 8 directions
    let directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    for (dr, dc) in directions {
        if would_flip_in_direction(board, player, row, col, dr, dc) {
            return true;
        }
    }

    false
}

/// Check if placing a piece at (row, col) would flip pieces in direction (dr, dc)
fn would_flip_in_direction(
    board: &Board,
    player: u8,
    row: usize,
    col: usize,
    dr: i8,
    dc: i8,
) -> bool {
    let opponent = if player == BLACK { WHITE } else { BLACK };

    let mut r = row as i8 + dr;
    let mut c = col as i8 + dc;
    let mut found_opponent = false;

    while r >= 0 && r < 8 && c >= 0 && c < 8 {
        let index = (r as usize) * 8 + (c as usize);
        let cell = board[index];

        if cell == EMPTY {
            return false;
        } else if cell == opponent {
            found_opponent = true;
            r += dr;
            c += dc;
        } else if cell == player {
            return found_opponent;
        } else {
            return false;
        }
    }

    false
}

/// Apply a move to a board and return the new board
fn apply_move_to_board(board: &Board, player: u8, row: usize, col: usize) -> Option<Board> {
    // Check if move is valid
    if !is_valid_move(board, player, row, col) {
        return None;
    }

    let mut new_board = *board;
    let index = row * 8 + col;
    new_board[index] = player;

    // Flip in each valid direction
    let directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    for (dr, dc) in directions {
        flip_in_direction(&mut new_board, player, row, col, dr, dc);
    }

    Some(new_board)
}

/// Flip pieces in a specific direction from (row, col)
fn flip_in_direction(board: &mut Board, player: u8, row: usize, col: usize, dr: i8, dc: i8) {
    if !would_flip_in_direction(board, player, row, col, dr, dc) {
        return;
    }

    let opponent = if player == BLACK { WHITE } else { BLACK };

    let mut r = row as i8 + dr;
    let mut c = col as i8 + dc;

    while r >= 0 && r < 8 && c >= 0 && c < 8 {
        let index = (r as usize) * 8 + (c as usize);
        let cell = board[index];

        if cell == opponent {
            board[index] = player;
            r += dr;
            c += dc;
        } else {
            break;
        }
    }
}

/// Count pieces on the board
fn count_pieces(board: &Board, player: u8) -> u8 {
    board.iter().filter(|&&cell| cell == player).count() as u8
}

/// Heuristic evaluation function using weighted positions
fn heuristic_static_weights(board: &Board, player: u8) -> i32 {
    let mut total = 0;
    let opponent = if player == BLACK { WHITE } else { BLACK };

    for index in 0..64 {
        match board[index] {
            cell if cell == player => total += WEIGHTED_POSITIONS[index],
            cell if cell == opponent => total -= WEIGHTED_POSITIONS[index],
            _ => {}
        }
    }

    total
}

/// Heuristic based on token parity (normalized)
fn heuristic_token_parity_normed(board: &Board, player: u8) -> f32 {
    let opponent = if player == BLACK { WHITE } else { BLACK };
    let player_count = count_pieces(board, player) as f32;
    let opponent_count = count_pieces(board, opponent) as f32;

    if player_count + opponent_count == 0.0 {
        0.0
    } else {
        100.0 * ((player_count - opponent_count) / (player_count + opponent_count))
    }
}

/// Heuristic based on corner captures (normalized)
fn heuristic_corners_captured_normed(board: &Board, player: u8) -> f32 {
    let corners = [0, 7, 56, 63]; // Indices of corners: a1, h1, a8, h8
    let mut ai_corners = 0.0;
    let mut opponent_corners = 0.0;
    let _opponent = if player == BLACK { WHITE } else { BLACK };

    for &index in &corners {
        if board[index] == player {
            ai_corners += 3.0;
        } else if board[index] == EMPTY {
            // Check if corner is a valid move for player
            let row = index / 8;
            let col = index % 8;
            if is_valid_move(board, player, row, col) {
                ai_corners += 1.0;
            } else {
                ai_corners -= 1.0;
            }
        } else {
            let _opponent = if player == BLACK { WHITE } else { BLACK };
            opponent_corners += 3.0;
        }
    }

    if ai_corners + opponent_corners == 0.0 {
        0.0
    } else {
        100.0 * ((ai_corners - opponent_corners) / (ai_corners + opponent_corners))
    }
}

/// Heuristic based on mobility (normalized)
fn heuristic_actual_mobility_normed(board: &Board, player: u8) -> f32 {
    let player_moves = get_valid_moves(board, player);
    let opponent = if player == BLACK { WHITE } else { BLACK };
    let opponent_moves = get_valid_moves(board, opponent);

    let player_count = player_moves.iter().filter(|&&x| x).count() as f32;
    let opponent_count = opponent_moves.iter().filter(|&&x| x).count() as f32;

    if player_count + opponent_count == 0.0 {
        0.0
    } else {
        100.0 * ((player_count - opponent_count) / (player_count + opponent_count))
    }
}

/// Combined heuristic function that evaluates a board position
///
/// Combines four different heuristics with weighted importance:
/// - Token parity: Raw difference in piece counts (weight: 1.0)
/// - Positional weights: Strategic value of piece placement (weight: 10.0)
/// - Mobility: Difference in available moves (weight: 1.0)
/// - Corner captures: Control of valuable corner positions (weight: 1.0)
///
/// The positional weights heuristic has a higher weight because controlling
/// strategic positions (especially corners) is typically more valuable than
/// just having more pieces.
fn heuristics_othello(board: &Board, player: u8) -> f32 {
    heuristic_token_parity_normed(board, player)
        + 10.0 * (heuristic_static_weights(board, player) as f32)
        + heuristic_actual_mobility_normed(board, player)
        + heuristic_corners_captured_normed(board, player)
}

/// Alpha-beta pruning minimax implementation
///
/// This function implements the core of the aelskels AI:
/// - Uses minimax search to evaluate all possible move sequences
/// - Applies alpha-beta pruning to eliminate branches that won't affect the final decision
/// - Returns the best score and move for the current player
///
/// Alpha-beta pruning optimization:
/// - Alpha: Best score the maximizing player (current player) is guaranteed
/// - Beta: Best score the minimizing player (opponent) is guaranteed
/// - If alpha >= beta, we can prune the remaining branches
fn alpha_beta_pruning_minimax(
    board: &Board,
    player: u8,
    depth: i32,
    alpha: i32,
    beta: i32,
) -> (i32, Option<u8>) {
    // Base case: depth is 0, evaluate the board position
    if depth == 0 {
        return (heuristics_othello(board, player) as i32, None);
    }

    let valid_moves = get_valid_moves(board, player);
    let mut best_move: Option<u8> = None;

    // Optimization: if only one move available at root level (depth 5), return it immediately
    // This saves computation time for forced moves, as there's no decision to make
    if depth == 5 {
        let moves_count = valid_moves.iter().filter(|&&x| x).count();
        if moves_count == 1 {
            for i in 0..64 {
                if valid_moves[i] {
                    return (0, Some(i as u8));
                }
            }
        }
    }

    // If no valid moves, evaluate current position (pass turn)
    if !valid_moves.iter().any(|&x| x) {
        return (heuristics_othello(board, player) as i32, None);
    }

    let mut best_score = i32::MIN;

    // Try all valid moves to find the best one
    for i in 0..64 {
        if valid_moves[i] {
            let row = i / 8;
            let col = i % 8;

            // Apply move to get new board state and simulate the future position
            if let Some(new_board) = apply_move_to_board(board, player, row, col) {
                // Recursively call minimax for opponent (minimizing player)
                // The helper function will evaluate from opponent's perspective
                let (score, _) = alpha_beta_pruning_minimax_helper(
                    &new_board,
                    if player == BLACK { WHITE } else { BLACK },
                    depth - 1,
                    alpha,
                    beta,
                );

                // Negate score since it's from opponent's perspective
                // We invert the score because the helper returns opponent's best evaluation,
                // but we want the score from our perspective. Opponent's best = our worst,
                // so we negate to convert to our perspective.

                if score > best_score {
                    best_score = score;
                    best_move = Some(i as u8);
                }

                // Alpha-beta pruning: if this move is better than the best
                // the opponent can guarantee, the opponent won't let us reach this node
                // Beta cutoff: if we can achieve a score >= beta, no need to search further
                // as the opponent will avoid this outcome
                if best_score >= beta {
                    return (best_score, best_move);
                }

                // Update alpha (best guaranteed score for maximizing player)
                let new_alpha = alpha.max(best_score);
                if new_alpha >= beta {
                    // Beta cutoff: opponent won't allow this path
                    break;
                }
            }
        }
    }

    (best_score, best_move)
}

/// Helper function for minimax that evaluates from opponent's perspective
///
/// This function represents the opponent's turn in the minimax search:
/// - The opponent is the minimizing player (wants to minimize our score)
/// - Returns the minimum score the opponent can force upon us
fn alpha_beta_pruning_minimax_helper(
    board: &Board,
    player: u8,
    depth: i32,
    alpha: i32,
    beta: i32,
) -> (i32, Option<u8>) {
    // Base case: depth is 0, evaluate the board position from opponent's perspective
    if depth == 0 {
        let opponent = if player == BLACK { WHITE } else { BLACK };
        return (heuristics_othello(board, opponent) as i32, None);
    }

    let valid_moves = get_valid_moves(board, player);

    // If no valid moves, evaluate current position (opponent passes turn)
    if !valid_moves.iter().any(|&x| x) {
        let opponent = if player == BLACK { WHITE } else { BLACK };
        return (heuristics_othello(board, opponent) as i32, None);
    }

    let mut best_score = i32::MAX; // Minimizing player seeks minimum score

    // Try all valid moves for the opponent
    // As the minimizing player, we want to find the move that results in the lowest score
    for i in 0..64 {
        if valid_moves[i] {
            let row = i / 8;
            let col = i % 8;

            // Apply move to get new board state
            if let Some(new_board) = apply_move_to_board(board, player, row, col) {
                // Recursively call minimax for original player (maximizing player)
                // This returns the best score the original player can achieve
                let (score, _) = alpha_beta_pruning_minimax(
                    &new_board,
                    if player == BLACK { WHITE } else { BLACK },
                    depth - 1,
                    alpha,
                    beta,
                );

                // Keep track of the minimum score the opponent can force
                // (opponent wants to minimize the score, so we track the minimum)
                if score < best_score {
                    best_score = score;
                }

                // Alpha-beta pruning: if opponent can force a worse outcome
                // than what we can guarantee (alpha), we won't let them reach this node
                // Alpha cutoff: if opponent's best possible score <= alpha,
                // they can't force us into a worse outcome
                if best_score <= alpha {
                    return (best_score, None);
                }

                // Update beta (best guaranteed score for minimizing player)
                let new_beta = beta.min(best_score);
                if alpha >= new_beta {
                    // Alpha cutoff: we won't allow this path
                    break;
                }
            }
        }
    }

    (best_score, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_move_exists() {
        // Create initial board state
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        let move_result = compute_move(&board, BLACK);
        // Should return a valid move index or u8::MAX if no moves
        assert!(move_result <= 63 || move_result == u8::MAX);
    }

    #[test]
    fn test_heuristic_functions() {
        // Create initial board state
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        let score = heuristics_othello(&board, BLACK);
        // Should return a finite value
        assert!(score.is_finite());
    }

    #[test]
    fn test_initial_board_moves() {
        // Create initial board state
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        // For the initial board, there should be valid moves for Black
        let valid_moves = get_valid_moves(&board, BLACK);
        let valid_count = valid_moves.iter().filter(|&&x| x).count();
        assert_eq!(valid_count, 4); // Should have 4 valid moves initially

        // Test that compute_move returns a valid move
        let move_result = compute_move(&board, BLACK);
        assert!(move_result < 64);
        assert!(valid_moves[move_result as usize]);
    }

    #[test]
    fn test_weighted_positions() {
        // Test that corner positions have high positive weights
        assert_eq!(WEIGHTED_POSITIONS[0], 20); // a1
        assert_eq!(WEIGHTED_POSITIONS[7], 20); // h1
        assert_eq!(WEIGHTED_POSITIONS[56], 20); // a8
        assert_eq!(WEIGHTED_POSITIONS[63], 20); // h8

        // Test that positions adjacent to corners have negative weights
        assert_eq!(WEIGHTED_POSITIONS[1], -3); // b1
        assert_eq!(WEIGHTED_POSITIONS[8], -3); // a2
    }

    #[test]
    fn test_compute_move_scores_returns_64_elements() {
        let mut board = [EMPTY; 64];
        board[27] = WHITE;
        board[28] = BLACK;
        board[35] = BLACK;
        board[36] = WHITE;

        let scores = compute_move_scores(&board, BLACK);
        assert_eq!(scores.len(), 64);
    }

    #[test]
    fn test_compute_move_scores_legal_moves_non_zero() {
        let mut board = [EMPTY; 64];
        board[27] = WHITE;
        board[28] = BLACK;
        board[35] = BLACK;
        board[36] = WHITE;

        let scores = compute_move_scores(&board, BLACK);
        let valid_moves = get_valid_moves(&board, BLACK);

        // All legal moves should have non-zero scores
        for i in 0..64 {
            if valid_moves[i] {
                assert_ne!(
                    scores[i], 0,
                    "Legal move at index {} should have non-zero score",
                    i
                );
            }
        }
    }

    #[test]
    fn test_compute_move_scores_illegal_moves_zero() {
        let mut board = [EMPTY; 64];
        board[27] = WHITE;
        board[28] = BLACK;
        board[35] = BLACK;
        board[36] = WHITE;

        let scores = compute_move_scores(&board, BLACK);
        let valid_moves = get_valid_moves(&board, BLACK);

        // All illegal moves should have zero scores
        for i in 0..64 {
            if !valid_moves[i] {
                assert_eq!(
                    scores[i], 0,
                    "Illegal move at index {} should have zero score",
                    i
                );
            }
        }
    }

    #[test]
    fn test_compute_move_scores_initial_board() {
        let mut board = [EMPTY; 64];
        board[27] = WHITE;
        board[28] = BLACK;
        board[35] = BLACK;
        board[36] = WHITE;

        let scores = compute_move_scores(&board, BLACK);

        // Count non-zero scores (legal moves)
        let non_zero_count = scores.iter().filter(|&&s| s != 0).count();
        // Initial board for Black should have 4 legal moves
        assert_eq!(non_zero_count, 4);
    }
}
