# Implementation Plan: Othello RL Environment

## Overview

This implementation plan breaks down the Othello RL environment into incremental steps, building from the core Rust game engine through Python bindings to the final Gymnasium environment. Each task builds on previous work, with testing integrated throughout to catch errors early.

The implementation follows a bottom-up approach: Rust core → Python bindings → Gymnasium wrapper → RLlib integration.

## Tasks

- [x] 1. Set up Rust project structure and core types
  - Create `rust/othello` directory with Cargo.toml
  - Define `Player`, `Cell`, and `Board` types
  - Implement `Board::new()` with initial Othello setup (4 pieces in center)
  - Add basic Cargo configuration for PyO3 support
  - _Requirements: 1.1_

- [x] 2. Implement core game logic in Rust
  - [x] 2.1 Implement move validation logic
    - Write `is_valid_move()` method checking all 8 directions
    - Write `would_flip_in_direction()` helper method
    - Write `get_valid_moves()` returning 64-element bool array
    - _Requirements: 1.3_
  
  - [x] 2.2 Write property test for valid move detection
    - **Property 2: Valid Move Detection**
    - **Validates: Requirements 1.3**
  
  - [x] 2.3 Implement piece flipping logic
    - Write `apply_move()` method that places piece and flips in all directions
    - Write `flip_in_direction()` helper method
    - Handle turn passing when no valid moves exist
    - _Requirements: 1.2, 1.4_
  
  - [x] 2.4 Write property test for piece flipping correctness
    - **Property 1: Piece Flipping Correctness**
    - **Validates: Requirements 1.2**
  
  - [x] 2.5 Write property test for turn passing
    - **Property 3: Turn Passing**
    - **Validates: Requirements 1.4**

- [x] 3. Implement game state management in Rust
  - [x] 3.1 Implement player tracking and piece counting
    - Write `update_piece_counts()` method
    - Implement player alternation logic
    - Write `get_winner()` method
    - _Requirements: 1.6, 1.7_
  
  - [x] 3.2 Write property test for player alternation
    - **Property 4: Player Alternation**
    - **Validates: Requirements 1.6**
  
  - [x] 3.3 Write property test for piece count accuracy
    - **Property 5: Piece Count Accuracy**
    - **Validates: Requirements 1.7**
  
  - [x] 3.4 Implement game termination detection
    - Write `is_game_over()` method checking board full and no valid moves
    - Handle case where neither player has moves
    - _Requirements: 1.8, 1.5_
  
  - [x] 3.5 Write property test for game termination
    - **Property 6: Game Termination**
    - **Validates: Requirements 1.8, 1.5**
  
  - [x] 3.6 Implement board state accessors
    - Write `get_state()` returning flat [u8; 64] array
    - Write `reset()` method
    - _Requirements: 1.1_

- [x] 4. Checkpoint - Rust core functionality complete
  - Run all Rust tests (unit and property tests)
  - Verify game logic correctness with manual testing
  - Ensure all tests pass, ask the user if questions arise

- [x] 5. Implement PyO3 bindings
  - [x] 5.1 Create PyO3 module structure
    - Set up `rust/othello/src/bindings.rs`
    - Define `#[pyclass] OthelloGame` wrapping `Board`
    - Implement `#[pymodule]` for Python import
    - Configure Cargo.toml for cdylib output
    - _Requirements: 2.1_
  
  - [x] 5.2 Implement Python-facing methods
    - Write `new()`, `reset()`, `step(action)` methods
    - Write `get_board()` returning PyArray2<u8>
    - Write `get_valid_moves()` returning PyArray1<bool>
    - Write `get_current_player()`, `get_piece_counts()`, `get_winner()` methods
    - _Requirements: 2.2, 2.3, 2.4, 2.5_
  
  - [x] 5.3 Implement error handling in bindings
    - Convert Rust errors to Python exceptions
    - Add input validation for action range [0, 63]
    - Handle type conversion errors gracefully
    - _Requirements: 2.6_
  
  - [x] 5.4 Write unit tests for Python bindings
    - Test type conversions for standard board states
    - Test error propagation for invalid moves
    - Test reset functionality
    - _Requirements: 2.1, 2.7_
  
  - [x] 5.5 Write property test for state serialization round-trip
    - **Property 7: State Serialization Round-Trip**
    - **Validates: Requirements 2.1, 2.7**
  
  - [x] 5.6 Write property test for action validity consistency
    - **Property 8: Action Validity Consistency**
    - **Validates: Requirements 2.3, 2.4**

- [x] 6. Build and install Python package
  - Create `maturin` or `setuptools-rust` build configuration
  - Build Rust extension module
  - Install in development mode for testing
  - Verify Python can import `othello_rust` module
  - _Requirements: 2.1_

- [x] 7. Implement Gymnasium environment class
  - [x] 7.1 Create environment class structure
    - Create `aip_rl/othello/env.py`
    - Define `OthelloEnv(gym.Env)` class
    - Implement `__init__` with configuration parameters
    - Define observation_space as Box(0, 1, (3, 8, 8), float32)
    - Define action_space as Discrete(64)
    - _Requirements: 3.1, 3.4, 3.5_
  
  - [x] 7.2 Implement reset method
    - Write `reset()` returning initial observation and info
    - Initialize game state
    - Set agent_player to 0 (Black)
    - _Requirements: 3.2_
  
  - [x] 7.3 Write unit test for reset method
    - Verify reset returns correct initial observation shape
    - Verify info dictionary contains required fields
    - _Requirements: 3.2_
  
  - [x] 7.4 Implement observation generation
    - Write `_get_observation()` creating (3, 8, 8) array
    - Implement channel 0: agent's pieces
    - Implement channel 1: opponent's pieces
    - Implement channel 2: valid moves
    - Ensure all values in [0, 1]
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_
  
  - [x] 7.5 Write property test for observation encoding correctness
    - **Property 10: Observation Encoding Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**
  
  - [x] 7.6 Implement info dictionary generation
    - Write `_get_info()` returning dict with action_mask, current_player, piece counts, agent_player
    - _Requirements: 3.6, 4.5, 7.7, 7.9_
  
  - [x] 7.7 Write property test for info dictionary completeness
    - **Property 11: Info Dictionary Completeness**
    - **Validates: Requirements 3.6, 4.5, 7.7, 7.9**

- [x] 8. Implement step method and reward calculation
  - [x] 8.1 Implement basic step method
    - Write `step(action)` returning (obs, reward, terminated, truncated, info)
    - Validate action against valid moves
    - Apply move to game
    - _Requirements: 3.3_
  
  - [x] 8.2 Write property test for step return signature
    - **Property 12: Step Return Signature**
    - **Validates: Requirements 3.3**
  
  - [x] 8.3 Implement invalid move handling
    - Add logic for "penalty" mode (negative reward, maintain state)
    - Add logic for "random" mode (select random valid move)
    - Add logic for "error" mode (raise ValueError)
    - _Requirements: 3.7, 5.3_
  
  - [x] 8.4 Write property test for invalid move handling
    - **Property 13: Invalid Move Handling**
    - **Validates: Requirements 3.7, 5.3**
  
  - [x] 8.5 Implement sparse reward calculation
    - Write `_calculate_reward()` for sparse mode
    - Return 0 during game, +1/-1/0 at end based on winner
    - _Requirements: 5.1_
  
  - [x] 8.6 Write property test for sparse reward correctness
    - **Property 14: Sparse Reward Correctness**
    - **Validates: Requirements 5.1**
  
  - [x] 8.7 Implement dense reward calculation
    - Add dense mode to `_calculate_reward()`
    - Calculate normalized piece differential
    - _Requirements: 5.2, 5.5_
  
  - [x] 8.8 Write property test for dense reward correctness
    - **Property 15: Dense Reward Correctness**
    - **Validates: Requirements 5.2, 5.5**
  
  - [x] 8.9 Implement custom reward function support
    - Add support for callable reward functions in config
    - Call custom function with game state
    - _Requirements: 5.4_
  
  - [x] 8.10 Write property test for custom reward function
    - **Property 16: Custom Reward Function**
    - **Validates: Requirements 5.4**

- [x] 9. Checkpoint - Basic environment functionality complete
  - Test environment with random actions
  - Verify observations, rewards, and termination work correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 10. Implement self-play and opponent logic
  - [x] 10.1 Implement opponent move execution
    - Write `_execute_opponent_move()` method
    - Implement random opponent policy
    - Implement greedy opponent policy (most pieces flipped)
    - Support callable opponent policies
    - _Requirements: 6.4, 6.5_
  
  - [x] 10.2 Write property test for opponent move execution
    - **Property 18: Opponent Move Execution**
    - **Validates: Requirements 6.4, 6.5**
  
  - [x] 10.3 Implement self-play logic in step method
    - Add opponent move execution after agent move
    - Track agent_player correctly
    - Handle game termination after opponent move
    - _Requirements: 6.1, 6.2_
  
  - [x] 10.4 Implement perspective swapping
    - Ensure observation is always from agent's perspective
    - Swap channels when agent_player changes
    - _Requirements: 6.3_
  
  - [x] 10.5 Write property test for perspective consistency
    - **Property 17: Perspective Consistency**
    - **Validates: Requirements 6.3**

- [x] 11. Implement rendering functionality
  - [x] 11.1 Implement ANSI rendering
    - Write `_render_ansi()` returning string representation
    - Include board with pieces (●, ○, .)
    - Show valid moves with * markers
    - Display piece counts and current player
    - _Requirements: 10.2, 10.4, 10.5_
  
  - [x] 11.2 Write property test for ANSI rendering completeness
    - **Property 19: ANSI Rendering Completeness**
    - **Validates: Requirements 10.2, 10.4, 10.5**
  
  - [x] 11.3 Implement RGB array rendering
    - Write `_render_rgb()` returning (H, W, 3) uint8 array
    - Draw board with green background
    - Draw black and white pieces as circles
    - Draw valid move markers
    - _Requirements: 10.3_
  
  - [x] 11.4 Write property test for RGB array format
    - **Property 20: RGB Array Format**
    - **Validates: Requirements 10.3**
  
  - [x] 11.5 Implement render method dispatcher
    - Write `render()` method handling all modes
    - Support "human", "ansi", "rgb_array" modes
    - _Requirements: 10.1_

- [x] 12. Implement configuration and validation
  - [x] 12.1 Add configuration validation
    - Validate reward_mode in ["sparse", "dense"]
    - Validate opponent type
    - Validate render_mode
    - Raise ValueError for invalid configs
    - _Requirements: 11.6_
  
  - [x] 12.2 Write property test for configuration validation
    - **Property 21: Configuration Validation**
    - **Validates: Requirements 11.6**
  
  - [x] 12.3 Implement state save/load functionality
    - Add methods to save game state to dict
    - Add methods to load game state from dict
    - _Requirements: 11.7_
  
  - [x] 12.4 Write property test for state persistence round-trip
    - **Property 22: State Persistence Round-Trip**
    - **Validates: Requirements 11.7**
  
  - [x] 12.5 Implement episode termination signals
    - Ensure terminated flag is set correctly when game ends
    - Ensure truncated is always False
    - _Requirements: 7.6_
  
  - [x] 12.6 Write property test for episode termination signals
    - **Property 23: Episode Termination Signals**
    - **Validates: Requirements 7.6**

- [x] 13. Register environment with Gymnasium
  - Create `aip_rl/othello/__init__.py`
  - Register "Othello-v0" with Gymnasium
  - Set max_episode_steps to 60
  - Verify environment can be created with `gym.make("Othello-v0")`
  - _Requirements: 7.1_

- [x] 14. Checkpoint - Complete environment implementation
  - Run all Python tests (unit and property tests)
  - Test environment manually with random agent
  - Test all rendering modes
  - Ensure all tests pass, ask the user if questions arise

- [ ] 15. Create RLlib integration and training script
  - [x] 15.1 Implement custom CNN model for Othello
    - Create `scripts/train_othello.py`
    - Define `OthelloCNN` model class with conv layers for (3, 8, 8) input
    - Implement forward pass and value function
    - Register model with ModelCatalog
    - _Requirements: 7.8_
  
  - [x] 15.2 Configure PPO algorithm
    - Set up PPOConfig with Othello-v0 environment
    - Configure training hyperparameters
    - Set up evaluation
    - Use custom CNN model
    - _Requirements: 7.2_
  
  - [x] 15.3 Implement training loop
    - Build algorithm from config
    - Run training iterations
    - Log episode rewards and lengths
    - Save checkpoints periodically
    - _Requirements: 7.2_

- [ ] 16. Integration testing with RLlib
  - [x] 16.1 Test PPO training
    - Run 10 training iterations with PPO
    - Verify no crashes or errors
    - _Requirements: 7.2_
  
  - [x] 16.2 Test vectorized environments
    - Configure 4 parallel environments
    - Verify parallel execution works
    - _Requirements: 7.5_
  
  - [x] 16.3 Test action masking
    - Verify action_mask is used by PPO
    - Verify invalid actions are not selected
    - _Requirements: 7.9_
  
  - [x] 16.4 Test checkpoint save/restore
    - Save checkpoint during training
    - Restore and continue training
    - Verify state is preserved
    - _Requirements: 7.2_

- [x] 17. Create human interaction utilities
  - [x] 17.1 Create human player script
    - Write script for human vs agent games
    - Accept moves via console input
    - Display board after each move
    - Validate human inputs
    - _Requirements: 10.6, 10.7_
  
  - [x] 17.2 Create spectator mode script
    - Write script to watch two agents play
    - Display board after each move
    - Show game statistics at end
    - _Requirements: 10.8_

- [x] 18. Documentation and examples
  - [x] 18.1 Create README for Othello environment
    - Document installation steps
    - Provide usage examples
    - Explain configuration options
    - Include training example
  
  - [x] 18.2 Add docstrings to all public methods
    - Document OthelloEnv class and methods
    - Document configuration parameters
    - Add type hints
  
  - [x] 18.3 Create example notebooks
    - Notebook for basic environment usage
    - Notebook for training with RLlib
    - Notebook for evaluating trained agents

- [x] 19. Final checkpoint - Complete implementation
  - Run full test suite (Rust + Python)
  - Verify all property tests pass with 100+ iterations
  - Test end-to-end training for 50+ iterations
  - Verify environment works with different RLlib algorithms
  - Ensure all tests pass, ask the user if questions arise

## Notes

- All tasks are required for comprehensive implementation
- Each property test references a specific design property for traceability
- Checkpoints ensure incremental validation and provide natural stopping points
- The implementation builds bottom-up: Rust → Bindings → Environment → Integration
- Property tests use hypothesis (Python) and proptest/quickcheck (Rust) with 100+ iterations
- Integration tests verify RLlib compatibility and are part of the complete implementation
