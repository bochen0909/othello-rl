## 1. Project Setup and Analysis

- [x] 1.1 Analyze aelskels C/C++ engine code and document algorithm/structure
- [x] 1.2 Analyze drohh C/C++ engine code and document algorithm/structure
- [x] 1.3 Analyze nealetham C/C++ engine code and document algorithm/structure
- [x] 1.4 Identify any external dependencies or special libraries used by engines
- [x] 1.5 Create `rust/engines/` directory structure with Cargo.toml

## 2. Rust Engine Conversion - aelskels

- [x] 2.1 Convert aelskels C/C++ code to Rust in `rust/engines/src/aelskels.rs`
- [x] 2.2 Implement `compute_move(board: &Board, player: u8) -> u8` function for aelskels
- [x] 2.3 Create unit tests for aelskels move computation with known positions
- [x] 2.4 Validate aelskels Rust implementation against original C/C++ behavior
- [x] 2.5 Add documentation/comments explaining aelskels algorithm in Rust code

## 3. Rust Engine Conversion - drohh

- [x] 3.1 Convert drohh C/C++ code to Rust in `rust/engines/src/drohh.rs`
- [x] 3.2 Implement `compute_move(board: &Board, player: u8) -> u8` function for drohh
- [x] 3.3 Create unit tests for drohh move computation with known positions
- [x] 3.4 Validate drohh Rust implementation against original C/C++ behavior
- [x] 3.5 Add documentation/comments explaining drohh algorithm in Rust code

## 4. Rust Engine Conversion - nealetham

- [x] 4.1 Convert nealetham C/C++ code to Rust in `rust/engines/src/nealetham.rs`
- [x] 4.2 Implement `compute_move(board: &Board, player: u8) -> u8` function for nealetham
- [x] 4.3 Create unit tests for nealetham move computation with known positions
- [x] 4.4 Validate nealetham Rust implementation against original C/C++ behavior
- [x] 4.5 Add documentation/comments explaining nealetham algorithm in Rust code

## 5. Rust Engine Library Integration

- [x] 5.1 Update `rust/engines/src/lib.rs` to export all three engine modules
- [x] 5.2 Create public API module in `rust/engines/` that all three engines export
- [x] 5.3 Add `rust/engines/tests/integration_tests.rs` with integration tests for all engines
- [x] 5.4 Test compilation of `rust/engines` as standalone crate
- [x] 5.5 Verify all engine tests pass: `cargo test --manifest-path rust/engines/Cargo.toml`

## 6. PyO3 Bindings for Engine Move Computation

- [x] 6.1 Update `rust/othello/src/bindings.rs` to include engine module imports
- [x] 6.2 Add `compute_move_aelskels()` PyO3 function binding
- [x] 6.3 Add `compute_move_drohh()` PyO3 function binding
- [x] 6.4 Add `compute_move_nealetham()` PyO3 function binding
- [x] 6.5 Each binding converts Python board representation to Rust Board type
- [x] 6.6 Test PyO3 bindings: `maturin develop --release --manifest-path rust/othello/Cargo.toml`

## 7. Engine Configuration and Factory

- [x] 7.1 Create `aip_rl/othello/engines.py` module with engine configuration
- [x] 7.2 Define `ENGINE_REGISTRY` dict mapping engine names to move functions
- [x] 7.3 Create `get_engine_opponent(engine_name)` factory function
- [x] 7.4 Implement engine opponent callable that accepts observation and returns action
- [x] 7.5 Add error handling for invalid engine names with helpful messages
- [x] 7.6 Add `get_available_engines()` function to list all engines

## 8. Environment Integration

- [x] 8.1 Modify `aip_rl/othello/env.py` to recognize engine names as opponent type
- [x] 8.2 Update opponent initialization logic to handle external engines
- [x] 8.3 Update opponent step logic to call engine move computation
- [x] 8.4 Handle edge cases (no valid moves, etc.) for engine opponents
- [x] 8.5 Add validation that opponent parameter accepts engine names
- [x] 8.6 Update environment docstring to document engine opponent options

## 9. Training Script Integration

- [x] 9.1 Update `scripts/train_othello.py` argument parser to accept engine names
- [x] 9.2 Update training script help text to list available opponent options
- [x] 9.3 Pass engine opponent name to environment configuration
- [x] 9.4 Test training with `python scripts/train_othello.py --opponent aelskels --num-iterations 5`
- [x] 9.5 Test training with `python scripts/train_othello.py --opponent drohh --num-iterations 5`
- [x] 9.6 Test training with `python scripts/train_othello.py --opponent nealetham --num-iterations 5`

## 10. Backward Compatibility Verification

- [x] 10.1 Test existing random opponent still works: `--opponent random`
- [x] 10.2 Test existing greedy opponent still works: `--opponent greedy`
- [x] 10.3 Test self-play still works: `--opponent self`
- [x] 10.4 Test custom callable opponent still works
- [x] 10.5 Run existing test suite: `poetry run pytest aip_rl/othello/tests/`
- [x] 10.6 Run Rust tests: `cd rust/othello && cargo test`

## 11. Performance and Validation Testing

- [x] 11.1 Benchmark engine move latency vs. greedy opponent baseline
- [x] 11.2 Verify engine moves are deterministic (same board → same move)
- [x] 11.3 Test engine behavior on edge cases (endgame, opening, no valid moves)
- [x] 11.4 Run a full training run with each engine and verify convergence
- [x] 11.5 Document performance characteristics in README

## 12. Documentation and Cleanup

- [x] 12.1 Update `README.md` with engine opponent usage examples
- [x] 12.2 Add engine opponent options to opponent configuration section
- [x] 12.3 Add troubleshooting section for engine-related issues
- [x] 12.4 Document the three engine algorithms and their strategies
- [x] 12.5 Create inline code comments explaining engine move logic
- [x] 12.6 Verify all code follows project style guidelines

## 13. Final Verification

- [x] 13.1 Run full test suite one final time: `poetry run pytest`
- [x] 13.2 Run Rust tests one final time: `cargo test --manifest-path rust/othello/Cargo.toml`
- [x] 13.3 Verify maturin builds successfully: `maturin develop --release`
- [x] 13.4 Test environment creation with all opponent types
- [x] 13.5 Run a complete training iteration with each engine opponent
- [x] 13.6 Verify README examples work as documented
