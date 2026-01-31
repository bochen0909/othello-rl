## ADDED Requirements

### Requirement: Rust engine implementations exist
Rust implementations of three external Othello engines (aelskens, drohh, nealetham) SHALL exist in `rust/engines/src/` with public move computation functions.

#### Scenario: aelskels engine module exports compute_move function
- **WHEN** `rust/engines/src/aelskels.rs` is compiled
- **THEN** it exports a public `compute_move(board: &Board, player: u8) -> u8` function

#### Scenario: drohh engine module exports compute_move function
- **WHEN** `rust/engines/src/drohh.rs` is compiled
- **THEN** it exports a public `compute_move(board: &Board, player: u8) -> u8` function

#### Scenario: nealetham engine module exports compute_move function
- **WHEN** `rust/engines/src/nealetham.rs` is compiled
- **THEN** it exports a public `compute_move(board: &Board, player: u8) -> u8` function

### Requirement: PyO3 bindings expose engine move functions
The PyO3 bindings in `rust/othello/src/bindings.rs` SHALL expose Python functions to call each engine's move computation.

#### Scenario: Python can call aelskels move computation
- **WHEN** Python code calls `othello_rust.compute_move_aelskels(board_state, player)`
- **THEN** function returns an integer action [0, 63] representing the engine's chosen move

#### Scenario: Python can call drohh move computation
- **WHEN** Python code calls `othello_rust.compute_move_drohh(board_state, player)`
- **THEN** function returns an integer action [0, 63] representing the engine's chosen move

#### Scenario: Python can call nealetham move computation
- **WHEN** Python code calls `othello_rust.compute_move_nealetham(board_state, player)`
- **THEN** function returns an integer action [0, 63] representing the engine's chosen move

### Requirement: Engine move computation accepts correct input format
Each engine's move computation function SHALL accept a Board structure and player indicator as input.

#### Scenario: Board state input format
- **WHEN** move computation is called with `board` parameter
- **THEN** board parameter represents an 8x8 Othello board with piece positions encoded as bits

#### Scenario: Player indicator input format
- **WHEN** move computation is called with `player` parameter
- **THEN** player parameter is an integer where 0 represents Black and 1 represents White

### Requirement: Engine move computation latency is acceptable
Engine move computation SHALL complete within specified latency targets.

#### Scenario: Move computation completes within timeout
- **WHEN** engine move computation is called on valid board state
- **THEN** function returns within 100 milliseconds (worst case)

#### Scenario: Typical move computation is fast
- **WHEN** engine move computation is called on typical mid-game board state
- **THEN** function returns within 10 milliseconds (typical case)

### Requirement: Engine implementations handle edge cases
Each engine SHALL handle board states with limited or no valid moves.

#### Scenario: Engine handles endgame positions
- **WHEN** engine computes move on endgame board with few valid moves
- **THEN** engine returns a valid move or None if no moves available

#### Scenario: Engine handles opening positions
- **WHEN** engine computes move on initial or near-initial board state
- **THEN** engine returns a valid move from available options
