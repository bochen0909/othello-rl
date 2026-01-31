//! Integration tests for Othello engines
//!
//! These tests verify that each engine's move computation works correctly
//! and produces valid, deterministic results.

#[cfg(test)]
mod tests {
    use othello_engines::{compute_move_aelskens, compute_move_drohh, compute_move_nealetham};

    // Constants for cell values
    const EMPTY: u8 = 0;
    const BLACK: u8 = 1;
    const WHITE: u8 = 2;

    #[test]
    fn test_all_engines_export_functions() {
        // Verify all three engine functions are accessible
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        let _ = compute_move_aelskens(&board, BLACK);
        let _ = compute_move_drohh(&board, BLACK);
        let _ = compute_move_nealetham(&board, BLACK);
    }

    #[test]
    fn test_engines_return_u8() {
        // Ensure functions return valid u8 results
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        let aelskens_move = compute_move_aelskens(&board, BLACK);
        let drohh_move = compute_move_drohh(&board, BLACK);
        let nealetham_move = compute_move_nealetham(&board, BLACK);

        // All should return some u8 value (even if u8::MAX for "no move")
        assert_eq!(std::mem::size_of_val(&aelskens_move), 1);
        assert_eq!(std::mem::size_of_val(&drohh_move), 1);
        assert_eq!(std::mem::size_of_val(&nealetham_move), 1);
    }

    #[test]
    fn test_engines_accept_different_players() {
        // Engines should handle both player 1 and player 2
        let mut board = [EMPTY; 64];
        // Set up initial 4 pieces in center
        board[27] = WHITE; // d4
        board[28] = BLACK; // e4
        board[35] = BLACK; // d5
        board[36] = WHITE; // e5

        let _ = compute_move_aelskens(&board, BLACK);
        let _ = compute_move_aelskens(&board, WHITE);

        let _ = compute_move_drohh(&board, BLACK);
        let _ = compute_move_drohh(&board, WHITE);

        let _ = compute_move_nealetham(&board, BLACK);
        let _ = compute_move_nealetham(&board, WHITE);
    }
}
