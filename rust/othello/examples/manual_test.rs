/// Manual test to verify game logic correctness
use othello_rust::{Board, Player, Cell};

fn main() {
    println!("=== Othello Game Manual Test ===\n");
    
    let mut board = Board::new();
    
    // Test 1: Initial board state
    println!("Test 1: Initial Board State");
    print_board(&board);
    let (black, white) = board.get_piece_counts();
    println!("Black: {}, White: {}", black, white);
    println!("Current player: {:?}", board.get_current_player());
    assert_eq!(black, 2);
    assert_eq!(white, 2);
    assert_eq!(board.get_current_player(), Player::Black);
    println!("✓ Initial state correct\n");
    
    // Test 2: Valid moves detection
    println!("Test 2: Valid Moves Detection");
    let valid_moves = board.get_valid_moves();
    let count = valid_moves.iter().filter(|&&v| v).count();
    println!("Valid moves count: {}", count);
    println!("Valid positions:");
    for (i, &valid) in valid_moves.iter().enumerate() {
        if valid {
            let row = i / 8;
            let col = i % 8;
            println!("  ({}, {})", row, col);
        }
    }
    assert_eq!(count, 4);
    println!("✓ Valid moves detection correct\n");
    
    // Test 3: Apply a move
    println!("Test 3: Apply Move at (2, 3)");
    let result = board.apply_move(2, 3);
    assert!(result.is_ok());
    let flipped = result.unwrap();
    println!("Pieces flipped: {}", flipped);
    print_board(&board);
    let (black, white) = board.get_piece_counts();
    println!("Black: {}, White: {}", black, white);
    println!("Current player: {:?}", board.get_current_player());
    assert_eq!(black, 4);
    assert_eq!(white, 1);
    assert_eq!(board.get_current_player(), Player::White);
    println!("✓ Move application correct\n");
    
    // Test 4: Continue playing
    println!("Test 4: White's Turn - Move at (2, 2)");
    let result = board.apply_move(2, 2);
    assert!(result.is_ok());
    print_board(&board);
    let (black, white) = board.get_piece_counts();
    println!("Black: {}, White: {}", black, white);
    println!("Current player: {:?}", board.get_current_player());
    println!("✓ Player alternation correct\n");
    
    // Test 5: Invalid move
    println!("Test 5: Invalid Move at (0, 0)");
    let result = board.apply_move(0, 0);
    assert!(result.is_err());
    println!("✓ Invalid move rejected correctly\n");
    
    // Test 6: Reset
    println!("Test 6: Reset Board");
    board.reset();
    print_board(&board);
    let (black, white) = board.get_piece_counts();
    println!("Black: {}, White: {}", black, white);
    assert_eq!(black, 2);
    assert_eq!(white, 2);
    assert_eq!(board.get_current_player(), Player::Black);
    println!("✓ Reset correct\n");
    
    // Test 7: Game state
    println!("Test 7: Game State");
    let state = board.get_state();
    println!("Board state array length: {}", state.len());
    assert_eq!(state.len(), 64);
    println!("✓ Game state retrieval correct\n");
    
    println!("=== All Manual Tests Passed! ===");
}

fn print_board(board: &Board) {
    let state = board.get_state();
    println!("  0 1 2 3 4 5 6 7");
    for row in 0..8 {
        print!("{} ", row);
        for col in 0..8 {
            let idx = row * 8 + col;
            let symbol = match state[idx] {
                0 => ".",
                1 => "●",
                2 => "○",
                _ => "?",
            };
            print!("{} ", symbol);
        }
        println!();
    }
    println!();
}
