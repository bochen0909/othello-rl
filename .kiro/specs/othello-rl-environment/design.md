# Design Document: Othello RL Environment

## Overview

The Othello RL Environment is a high-performance, Gymnasium-compatible reinforcement learning environment for training agents to play Othello (Reversi). The system is architected in three layers:

1. **Rust Core Engine** (`rust/othello`): High-performance game logic implementation
2. **Python Bindings** (`aip_rl/othello/bindings.py`): PyO3-based interface exposing Rust to Python
3. **Gymnasium Wrapper** (`aip_rl/othello/env.py`): Standard RL environment interface

This layered architecture provides the performance benefits of Rust while maintaining the flexibility and ecosystem compatibility of Python-based RL frameworks.

### Design Goals

- **Performance**: Rust implementation for fast game logic (target: <1ms per step)
- **Compatibility**: Full Gymnasium API compliance for RLlib integration
- **Flexibility**: Configurable rewards, opponents, and observation formats
- **Usability**: Human-playable interface for testing and visualization
- **Correctness**: Comprehensive testing including property-based tests

## Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────┐
│  Ray RLlib Training Loop                                │
│  (PPO/DQN/APPO algorithms)                             │
└────────────────────┬────────────────────────────────────┘
                     │ Gymnasium API
┌────────────────────▼────────────────────────────────────┐
│  Gymnasium Environment Wrapper (Python)                 │
│  - OthelloEnv class                                     │
│  - Observation/Action space definitions                 │
│  - Reward calculation                                   │
│  - Self-play logic                                      │
│  - Rendering                                            │
└────────────────────┬────────────────────────────────────┘
                     │ PyO3 Bindings
┌────────────────────▼────────────────────────────────────┐
│  Python Bindings (PyO3)                                 │
│  - OthelloGame class                                    │
│  - Type conversions (Rust ↔ Python)                    │
│  - Error handling                                       │
└────────────────────┬────────────────────────────────────┘
                     │ FFI
┌────────────────────▼────────────────────────────────────┐
│  Rust Game Engine                                       │
│  - Board representation                                 │
│  - Move validation                                      │
│  - Piece flipping logic                                 │
│  - Game state management                                │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

**Training Step:**
1. RLlib calls `env.step(action)`
2. Gymnasium wrapper validates action and calls Python bindings
3. Python bindings invoke Rust engine to apply move
4. Rust engine updates board state and returns result
5. Python bindings convert Rust types to Python/NumPy
6. Gymnasium wrapper calculates reward and formats observation
7. Observation, reward, terminated, truncated, info returned to RLlib

**Self-Play:**
1. Environment tracks current player (agent vs opponent)
2. When opponent's turn, environment automatically executes opponent policy
3. Observation is flipped to maintain agent's perspective
4. Rewards are calculated from agent's perspective

## Components and Interfaces

### 1. Rust Game Engine (`rust/othello/src/lib.rs`)

#### Core Types

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Black,
    White,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

pub struct Board {
    cells: [[Cell; 8]; 8],
    current_player: Player,
    black_count: u8,
    white_count: u8,
    game_over: bool,
}
```

#### Key Methods

```rust
impl Board {
    /// Create a new board with initial Othello setup
    pub fn new() -> Self;
    
    /// Check if a move is valid at position (row, col)
    pub fn is_valid_move(&self, row: usize, col: usize) -> bool;
    
    /// Get all valid moves for current player as a 64-element bitmask
    pub fn get_valid_moves(&self) -> [bool; 64];
    
    /// Apply a move at position (row, col), flipping pieces
    /// Returns Ok(pieces_flipped) or Err(InvalidMove)
    pub fn apply_move(&mut self, row: usize, col: usize) -> Result<u8, GameError>;
    
    /// Pass turn to opponent (when no valid moves)
    pub fn pass_turn(&mut self);
    
    /// Get current board state as flat array [0=empty, 1=black, 2=white]
    pub fn get_state(&self) -> [u8; 64];
    
    /// Check if game is over
    pub fn is_game_over(&self) -> bool;
    
    /// Get winner (None if draw)
    pub fn get_winner(&self) -> Option<Player>;
    
    /// Reset board to initial state
    pub fn reset(&mut self);
}
```

#### Move Validation Algorithm

```rust
fn is_valid_move(&self, row: usize, col: usize) -> bool {
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

fn would_flip_in_direction(&self, row: usize, col: usize, dr: i8, dc: i8) -> bool {
    let opponent = self.current_player.opponent();
    let mut r = row as i8 + dr;
    let mut c = col as i8 + dc;
    let mut found_opponent = false;
    
    while r >= 0 && r < 8 && c >= 0 && c < 8 {
        match self.cells[r as usize][c as usize] {
            Cell::Empty => return false,
            cell if cell == opponent.to_cell() => {
                found_opponent = true;
                r += dr;
                c += dc;
            }
            _ => return found_opponent, // Found our piece after opponent pieces
        }
    }
    
    false
}
```

#### Piece Flipping Algorithm

```rust
fn apply_move(&mut self, row: usize, col: usize) -> Result<u8, GameError> {
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
    
    // Check if opponent has moves, otherwise pass or end game
    self.current_player = self.current_player.opponent();
    if self.get_valid_moves().iter().all(|&v| !v) {
        self.pass_turn();
    }
    
    Ok(total_flipped)
}
```

### 2. Python Bindings (`rust/othello/src/bindings.rs`)

#### PyO3 Class

```rust
use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3};

#[pyclass]
pub struct OthelloGame {
    board: Board,
}

#[pymethods]
impl OthelloGame {
    #[new]
    pub fn new() -> Self {
        Self { board: Board::new() }
    }
    
    /// Reset the game to initial state
    pub fn reset(&mut self) {
        self.board.reset();
    }
    
    /// Apply a move (action is 0-63)
    /// Returns (valid, pieces_flipped, game_over)
    pub fn step(&mut self, action: usize) -> PyResult<(bool, u8, bool)> {
        let row = action / 8;
        let col = action % 8;
        
        match self.board.apply_move(row, col) {
            Ok(flipped) => Ok((true, flipped, self.board.is_game_over())),
            Err(_) => Ok((false, 0, self.board.is_game_over())),
        }
    }
    
    /// Get board state as numpy array (8, 8)
    pub fn get_board<'py>(&self, py: Python<'py>) -> &'py PyArray2<u8> {
        let state = self.board.get_state();
        let array: [[u8; 8]; 8] = /* reshape state */;
        PyArray2::from_array(py, &array)
    }
    
    /// Get valid moves as numpy array (64,)
    pub fn get_valid_moves<'py>(&self, py: Python<'py>) -> &'py PyArray1<bool> {
        let moves = self.board.get_valid_moves();
        PyArray1::from_slice(py, &moves)
    }
    
    /// Get current player (0=Black, 1=White)
    pub fn get_current_player(&self) -> u8 {
        match self.board.current_player {
            Player::Black => 0,
            Player::White => 1,
        }
    }
    
    /// Get piece counts (black_count, white_count)
    pub fn get_piece_counts(&self) -> (u8, u8) {
        (self.board.black_count, self.board.white_count)
    }
    
    /// Get winner (0=Black, 1=White, 2=Draw, 3=NotFinished)
    pub fn get_winner(&self) -> u8 {
        if !self.board.is_game_over() {
            return 3;
        }
        match self.board.get_winner() {
            Some(Player::Black) => 0,
            Some(Player::White) => 1,
            None => 2,
        }
    }
}

#[pymodule]
fn othello_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OthelloGame>()?;
    Ok(())
}
```

### 3. Gymnasium Environment (`aip_rl/othello/env.py`)

#### Environment Class

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from aip_rl.othello.bindings import OthelloGame

class OthelloEnv(gym.Env):
    """
    Othello (Reversi) environment for reinforcement learning.
    
    Observation Space:
        Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        - Channel 0: Current player's pieces
        - Channel 1: Opponent's pieces  
        - Channel 2: Valid move mask
    
    Action Space:
        Discrete(64) - positions 0-63 representing board squares
    
    Rewards:
        - Configurable: sparse (win/loss) or dense (piece differential)
        - Invalid move penalty (configurable)
    """
    
    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}
    
    def __init__(
        self,
        opponent: str = "self",  # "self", "random", "greedy", or callable
        reward_mode: str = "sparse",  # "sparse" or "dense"
        invalid_move_penalty: float = -1.0,
        invalid_move_mode: str = "penalty",  # "penalty", "random", or "error"
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.game = OthelloGame()
        self.opponent = opponent
        self.reward_mode = reward_mode
        self.invalid_move_penalty = invalid_move_penalty
        self.invalid_move_mode = invalid_move_mode
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 8, 8), dtype=np.float32
        )
        self.action_space = spaces.Discrete(64)
        
        # Track agent's player color
        self.agent_player = 0  # 0=Black, 1=White
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.game.reset()
        self.agent_player = 0  # Agent always plays as Black initially
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Validate action
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves[action]:
            # Handle invalid move
            if self.invalid_move_mode == "error":
                raise ValueError(f"Invalid move: {action}")
            elif self.invalid_move_mode == "random":
                action = np.random.choice(np.where(valid_moves)[0])
            else:  # penalty
                reward = self.invalid_move_penalty
                obs = self._get_observation()
                info = self._get_info()
                return obs, reward, False, False, info
        
        # Apply move
        valid, pieces_flipped, game_over = self.game.step(action)
        
        # Calculate reward
        reward = self._calculate_reward(pieces_flipped, game_over)
        
        # If self-play and game not over, execute opponent move
        if self.opponent == "self" and not game_over:
            self._execute_opponent_move()
            game_over = self.game.get_winner() != 3
        
        obs = self._get_observation()
        info = self._get_info()
        
        terminated = game_over
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from agent's perspective.
        Returns (3, 8, 8) array with:
        - Channel 0: Agent's pieces
        - Channel 1: Opponent's pieces
        - Channel 2: Valid moves
        """
        board = self.game.get_board()  # (8, 8) with 0=empty, 1=black, 2=white
        valid_moves = self.game.get_valid_moves().reshape(8, 8)
        current_player = self.game.get_current_player()
        
        # Determine agent's piece value
        agent_piece = self.agent_player + 1  # 1=Black, 2=White
        opponent_piece = 3 - agent_piece
        
        # Create observation channels
        agent_channel = (board == agent_piece).astype(np.float32)
        opponent_channel = (board == opponent_piece).astype(np.float32)
        valid_channel = valid_moves.astype(np.float32)
        
        # Stack channels
        obs = np.stack([agent_channel, opponent_channel, valid_channel], axis=0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary with action mask and game state."""
        valid_moves = self.game.get_valid_moves()
        black_count, white_count = self.game.get_piece_counts()
        current_player = self.game.get_current_player()
        
        return {
            "action_mask": valid_moves,
            "current_player": current_player,
            "black_count": black_count,
            "white_count": white_count,
            "agent_player": self.agent_player,
        }
    
    def _calculate_reward(self, pieces_flipped: int, game_over: bool) -> float:
        """Calculate reward based on reward mode."""
        if self.reward_mode == "dense":
            # Reward based on piece differential
            black_count, white_count = self.game.get_piece_counts()
            if self.agent_player == 0:  # Agent is Black
                return (black_count - white_count) / 64.0
            else:  # Agent is White
                return (white_count - black_count) / 64.0
        
        else:  # sparse
            if not game_over:
                return 0.0
            
            winner = self.game.get_winner()
            if winner == 2:  # Draw
                return 0.0
            elif winner == self.agent_player:
                return 1.0
            else:
                return -1.0
    
    def _execute_opponent_move(self):
        """Execute opponent's move based on opponent policy."""
        valid_moves = self.game.get_valid_moves()
        
        if not np.any(valid_moves):
            return  # No valid moves, turn passes
        
        if self.opponent == "random":
            action = np.random.choice(np.where(valid_moves)[0])
        elif self.opponent == "greedy":
            action = self._get_greedy_move()
        elif callable(self.opponent):
            obs = self._get_observation()
            action = self.opponent(obs)
        else:
            raise ValueError(f"Unknown opponent type: {self.opponent}")
        
        self.game.step(action)
    
    def _get_greedy_move(self) -> int:
        """Get move that flips the most pieces."""
        valid_moves = self.game.get_valid_moves()
        best_action = -1
        best_flips = -1
        
        for action in np.where(valid_moves)[0]:
            # Simulate move
            game_copy = OthelloGame()
            game_copy.copy_from(self.game)
            _, flips, _ = game_copy.step(action)
            
            if flips > best_flips:
                best_flips = flips
                best_action = action
        
        return best_action
    
    def render(self):
        """Render the game state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
    
    def _render_ansi(self) -> str:
        """Render board as ASCII string."""
        board = self.game.get_board()
        valid_moves = self.game.get_valid_moves().reshape(8, 8)
        black_count, white_count = self.game.get_piece_counts()
        current_player = self.game.get_current_player()
        
        symbols = {0: ".", 1: "●", 2: "○"}
        lines = ["  0 1 2 3 4 5 6 7"]
        
        for i in range(8):
            row = f"{i} "
            for j in range(8):
                if board[i, j] == 0 and valid_moves[i, j]:
                    row += "* "  # Valid move
                else:
                    row += symbols[board[i, j]] + " "
            lines.append(row)
        
        lines.append(f"\n● Black: {black_count}  ○ White: {white_count}")
        lines.append(f"Current player: {'Black' if current_player == 0 else 'White'}")
        
        return "\n".join(lines)
    
    def _render_rgb(self) -> np.ndarray:
        """Render board as RGB array (for video recording)."""
        # Create 512x512 RGB image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cell_size = 64
        
        board = self.game.get_board()
        valid_moves = self.game.get_valid_moves().reshape(8, 8)
        
        # Draw board
        for i in range(8):
            for j in range(8):
                x, y = j * cell_size, i * cell_size
                
                # Board color (green)
                color = (0, 128, 0) if (i + j) % 2 == 0 else (0, 100, 0)
                img[y:y+cell_size, x:x+cell_size] = color
                
                # Draw piece
                if board[i, j] == 1:  # Black
                    self._draw_circle(img, x + cell_size//2, y + cell_size//2, 
                                     cell_size//3, (0, 0, 0))
                elif board[i, j] == 2:  # White
                    self._draw_circle(img, x + cell_size//2, y + cell_size//2,
                                     cell_size//3, (255, 255, 255))
                elif valid_moves[i, j]:  # Valid move marker
                    self._draw_circle(img, x + cell_size//2, y + cell_size//2,
                                     cell_size//6, (255, 255, 0))
        
        return img
    
    def _draw_circle(self, img: np.ndarray, cx: int, cy: int, 
                     radius: int, color: Tuple[int, int, int]):
        """Draw a filled circle on the image."""
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        img[mask] = color
```

### 4. Environment Registration (`aip_rl/othello/__init__.py`)

```python
from gymnasium.envs.registration import register
from aip_rl.othello.env import OthelloEnv

register(
    id="Othello-v0",
    entry_point="aip_rl.othello.env:OthelloEnv",
    max_episode_steps=60,  # Max moves per player
)

__all__ = ["OthelloEnv"]
```

### 5. RLlib Integration Example (`scripts/train_othello.py`)

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import gymnasium as gym

# Register environment
import aip_rl.othello

class OthelloCNN(TorchModelV2, nn.Module):
    """Custom CNN model for Othello board."""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                             model_config, name)
        nn.Module.__init__(self)
        
        # CNN layers for (3, 8, 8) input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_outputs)
        
        # Value function head
        self.value_fc = nn.Linear(512, 1)
        
        self._features = None
    
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        
        # CNN forward pass
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        self._features = x
        
        # Policy logits
        logits = self.fc2(x)
        
        return logits, state
    
    def value_function(self):
        return self.value_fc(self._features).squeeze(1)

def train_othello():
    ray.init(ignore_reinit_error=True)
    
    # Register custom model
    ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)
    
    # Configure PPO with action masking
    config = (
        PPOConfig()
        .environment(
            env="Othello-v0",
            env_config={
                "opponent": "self",
                "reward_mode": "sparse",
                "invalid_move_mode": "penalty",
            }
        )
        .framework("torch")
        .env_runners(num_env_runners=4)
        .resources(num_gpus=0)
        .training(
            train_batch_size=8000,
            minibatch_size=256,
            num_sgd_iter=20,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
        )
        .model({
            "custom_model": "othello_cnn",
        })
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=20,
            evaluation_num_env_runners=1,
        )
    )
    
    algo = config.build()
    
    # Training loop
    for i in range(200):
        result = algo.train()
        
        print(f"\nIteration {i + 1}")
        if "env_runners" in result:
            print(f"  Reward Mean: {result['env_runners']['episode_return_mean']:.2f}")
            print(f"  Episode Length: {result['env_runners']['episode_len_mean']:.2f}")
        
        if (i + 1) % 20 == 0:
            checkpoint = algo.save()
            print(f"  Checkpoint: {checkpoint}")
    
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    train_othello()
```

## Data Models

### Board State Representation

**Internal (Rust):**
- 2D array `[[Cell; 8]; 8]` for efficient access
- Enum-based cell values for type safety
- Separate counters for piece counts

**External (Python/NumPy):**
- Shape `(8, 8)` with dtype `uint8`
- Values: 0=Empty, 1=Black, 2=White

**Observation (RL Agent):**
- Shape `(3, 8, 8)` with dtype `float32`
- Channel 0: Agent's pieces (0 or 1)
- Channel 1: Opponent's pieces (0 or 1)
- Channel 2: Valid moves (0 or 1)

### Action Representation

- Integer in range [0, 63]
- Mapping: `action = row * 8 + col`
- Inverse: `row = action // 8, col = action % 8`

### Game State

```python
@dataclass
class GameState:
    board: np.ndarray  # (8, 8) uint8
    current_player: int  # 0=Black, 1=White
    black_count: int
    white_count: int
    valid_moves: np.ndarray  # (64,) bool
    game_over: bool
    winner: Optional[int]  # 0=Black, 1=White, None=Draw
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Core Game Logic Properties

**Property 1: Piece Flipping Correctness**

*For any* valid board state and any valid move, when a piece is placed, all opponent pieces in valid directions (horizontal, vertical, diagonal) that are sandwiched between the placed piece and another piece of the current player should be flipped, and no other pieces should be modified.

**Validates: Requirements 1.2**

**Property 2: Valid Move Detection**

*For any* board state, the set of valid moves returned by the engine should exactly match the set of positions where placing a piece would flip at least one opponent piece.

**Validates: Requirements 1.3**

**Property 3: Turn Passing**

*For any* board state where the current player has no valid moves but the opponent does have valid moves, passing the turn should switch the current player without modifying the board.

**Validates: Requirements 1.4**

**Property 4: Player Alternation**

*For any* sequence of valid moves, the current player should alternate between Black and White after each move (unless a turn is passed due to no valid moves).

**Validates: Requirements 1.6**

**Property 5: Piece Count Accuracy**

*For any* board state, the sum of black pieces, white pieces, and empty cells should equal 64, and the reported piece counts should match the actual number of pieces on the board.

**Validates: Requirements 1.7**

**Property 6: Game Termination**

*For any* board state, the game should be marked as over if and only if the board is full OR neither player has any valid moves.

**Validates: Requirements 1.8, 1.5 (edge case)**

### Python Bindings Properties

**Property 7: State Serialization Round-Trip**

*For any* valid game state in Rust, converting to Python (numpy array) and back should preserve the game state exactly.

**Validates: Requirements 2.1, 2.7**

**Property 8: Action Validity Consistency**

*For any* board state and action (0-63), the validity status returned by the Python bindings should match whether the move is actually valid in the Rust engine.

**Validates: Requirements 2.3, 2.4**

**Property 9: Error Handling**

*For any* invalid input (out of range actions, invalid types), the Python bindings should raise appropriate Python exceptions rather than crashing or returning incorrect results.

**Validates: Requirements 2.6**

### Gymnasium Environment Properties

**Property 10: Observation Encoding Correctness**

*For any* game state, the observation should be a (3, 8, 8) array where channel 0 contains exactly the agent's pieces, channel 1 contains exactly the opponent's pieces, and channel 2 contains exactly the valid moves, all normalized to [0, 1].

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**

**Property 11: Info Dictionary Completeness**

*For any* step or reset call, the returned info dictionary should contain all required fields: action_mask, current_player, black_count, white_count, and agent_player, with correct values.

**Validates: Requirements 3.6, 4.5, 7.7, 7.9**

**Property 12: Step Return Signature**

*For any* action, calling step(action) should return exactly 5 values: observation (ndarray), reward (float), terminated (bool), truncated (bool), and info (dict), with correct types.

**Validates: Requirements 3.3**

**Property 13: Invalid Move Handling**

*For any* invalid move, the environment should handle it according to the configured policy: apply penalty and maintain state (penalty mode), select a random valid move (random mode), or raise an error (error mode).

**Validates: Requirements 3.7, 5.3**

### Reward Structure Properties

**Property 14: Sparse Reward Correctness**

*For any* game in sparse reward mode, rewards should be 0 for all non-terminal steps, and at terminal steps should be +1 for agent win, -1 for agent loss, and 0 for draw.

**Validates: Requirements 5.1**

**Property 15: Dense Reward Correctness**

*For any* game in dense reward mode, the reward at each step should equal the normalized piece count differential (agent_pieces - opponent_pieces) / 64.

**Validates: Requirements 5.2, 5.5**

**Property 16: Custom Reward Function**

*For any* custom reward function provided in configuration, the environment should call that function with the correct game state and use its return value as the reward.

**Validates: Requirements 5.4**

### Self-Play Properties

**Property 17: Perspective Consistency**

*For any* game in self-play mode, the observation should always be from the agent's perspective regardless of which player (Black/White) the agent is currently controlling, meaning the agent's pieces are always in channel 0.

**Validates: Requirements 6.3**

**Property 18: Opponent Move Execution**

*For any* game with a specified opponent policy (random, greedy, or learned), after the agent's move, the opponent should automatically make a move according to its policy before the next agent observation.

**Validates: Requirements 6.4, 6.5**

### Rendering Properties

**Property 19: ANSI Rendering Completeness**

*For any* game state, the ANSI rendering should return a string containing the board representation, valid move indicators, piece counts for both players, and the current player.

**Validates: Requirements 10.2, 10.4, 10.5**

**Property 20: RGB Array Format**

*For any* game state, the RGB array rendering should return a numpy array with shape (H, W, 3) and dtype uint8, suitable for video recording.

**Validates: Requirements 10.3**

### Configuration Properties

**Property 21: Configuration Validation**

*For any* invalid configuration (e.g., unknown reward mode, invalid opponent type), the environment initialization should raise a ValueError with a descriptive message.

**Validates: Requirements 11.6**

**Property 22: State Persistence Round-Trip**

*For any* game state, saving the state and then loading it should result in an identical game state (same board, same current player, same piece counts).

**Validates: Requirements 11.7**

### Integration Properties

**Property 23: Episode Termination Signals**

*For any* completed game, the terminated flag should be True if and only if the game is over (board full or no valid moves for both players), and truncated should be False (games are never truncated).

**Validates: Requirements 7.6**

## Error Handling

### Rust Layer Errors

**Invalid Move Error:**
- Triggered when attempting to place a piece on an occupied cell or a position that doesn't flip any pieces
- Returns `Result::Err(GameError::InvalidMove)`
- Includes position information for debugging

**Out of Bounds Error:**
- Triggered when row or column is outside [0, 7]
- Returns `Result::Err(GameError::OutOfBounds)`

### Python Bindings Error Handling

**Type Conversion Errors:**
- Invalid action types (non-integer) → `TypeError`
- Invalid action range (< 0 or >= 64) → `ValueError`
- Numpy array conversion failures → `RuntimeError`

**Rust Error Propagation:**
- Rust `GameError::InvalidMove` → Python `ValueError`
- Rust `GameError::OutOfBounds` → Python `IndexError`
- Rust panics → Python `RuntimeError` (should never happen in production)

### Gymnasium Environment Error Handling

**Configuration Errors:**
- Invalid reward_mode → `ValueError` at init
- Invalid opponent type → `ValueError` at init
- Invalid render_mode → `ValueError` at init

**Runtime Errors:**
- Invalid action in "error" mode → `ValueError` with message
- Invalid action in "penalty" mode → negative reward, no error
- Invalid action in "random" mode → automatic valid move selection

**Graceful Degradation:**
- If opponent policy fails → fall back to random opponent
- If rendering fails → return empty string/array, log warning
- If custom reward function fails → fall back to sparse reward, log warning

## Testing Strategy

### Unit Testing

Unit tests focus on specific examples, edge cases, and error conditions. They complement property-based tests by providing concrete, reproducible test cases.

**Rust Engine Unit Tests:**
- Initial board setup (4 pieces in center, Black to move)
- Corner moves and edge moves
- Move that flips in multiple directions
- Move that flips maximum pieces (theoretical maximum)
- Board full termination
- No valid moves for both players termination
- Piece counting edge cases (all black, all white, empty board)

**Python Bindings Unit Tests:**
- Type conversion for standard board states
- Error propagation for invalid moves
- Reset to initial state
- Action mask format and values

**Gymnasium Environment Unit Tests:**
- Environment registration and creation via gym.make()
- Reset returns correct initial observation
- Step with valid move returns correct tuple
- Invalid move handling in each mode (penalty, random, error)
- Sparse reward calculation at game end
- Dense reward calculation during game
- Rendering in each mode (human, ansi, rgb_array)

### Property-Based Testing

Property-based tests verify universal properties across many randomly generated inputs. Each test should run a minimum of 100 iterations.

**Test Configuration:**
- Library: `hypothesis` for Python, `proptest` or `quickcheck` for Rust
- Minimum iterations: 100 per property
- Shrinking enabled to find minimal failing examples
- Seed recording for reproducibility

**Generators:**

```python
# Hypothesis generators for property tests

@st.composite
def board_state(draw):
    """Generate a random valid Othello board state."""
    # Start with empty board
    board = np.zeros((8, 8), dtype=np.uint8)
    
    # Place random pieces ensuring at least one of each color
    num_black = draw(st.integers(min_value=1, max_value=30))
    num_white = draw(st.integers(min_value=1, max_value=30))
    
    positions = draw(st.lists(
        st.tuples(st.integers(0, 7), st.integers(0, 7)),
        min_size=num_black + num_white,
        max_size=num_black + num_white,
        unique=True
    ))
    
    for i, (r, c) in enumerate(positions):
        if i < num_black:
            board[r, c] = 1  # Black
        else:
            board[r, c] = 2  # White
    
    return board

@st.composite
def valid_move_sequence(draw):
    """Generate a sequence of valid moves from initial state."""
    game = OthelloGame()
    moves = []
    
    for _ in range(draw(st.integers(0, 30))):
        valid_moves = game.get_valid_moves()
        if not np.any(valid_moves):
            break
        
        action = draw(st.sampled_from(np.where(valid_moves)[0]))
        moves.append(action)
        game.step(action)
    
    return moves
```

**Property Test Implementation Examples:**

```python
from hypothesis import given, strategies as st
import numpy as np

# Property 1: Piece Flipping Correctness
@given(board_state(), st.integers(0, 63))
def test_piece_flipping_correctness(board, action):
    """Feature: othello-rl-environment, Property 1: Piece Flipping Correctness"""
    game = OthelloGame()
    game.set_board(board)
    
    if not game.is_valid_move(action):
        return  # Skip invalid moves
    
    # Record pieces before move
    before_board = game.get_board().copy()
    current_player = game.get_current_player()
    
    # Apply move
    game.step(action)
    after_board = game.get_board()
    
    # Verify only flipped pieces and placed piece changed
    row, col = action // 8, action % 8
    
    # Check placed piece
    assert after_board[row, col] == current_player + 1
    
    # Check all other changes are valid flips
    for r in range(8):
        for c in range(8):
            if (r, c) == (row, col):
                continue
            if before_board[r, c] != after_board[r, c]:
                # This piece was flipped, verify it was opponent's piece
                assert before_board[r, c] == (3 - current_player - 1)
                assert after_board[r, c] == current_player + 1

# Property 5: Piece Count Accuracy
@given(board_state())
def test_piece_count_accuracy(board):
    """Feature: othello-rl-environment, Property 5: Piece Count Accuracy"""
    game = OthelloGame()
    game.set_board(board)
    
    black_count, white_count = game.get_piece_counts()
    
    # Count pieces manually
    actual_black = np.sum(board == 1)
    actual_white = np.sum(board == 2)
    actual_empty = np.sum(board == 0)
    
    assert black_count == actual_black
    assert white_count == actual_white
    assert black_count + white_count + actual_empty == 64

# Property 7: State Serialization Round-Trip
@given(valid_move_sequence())
def test_state_serialization_round_trip(moves):
    """Feature: othello-rl-environment, Property 7: State Serialization Round-Trip"""
    game1 = OthelloGame()
    
    # Apply moves
    for move in moves:
        game1.step(move)
    
    # Serialize to Python
    board = game1.get_board()
    current_player = game1.get_current_player()
    black_count, white_count = game1.get_piece_counts()
    
    # Create new game and deserialize
    game2 = OthelloGame()
    game2.set_board(board)
    game2.set_current_player(current_player)
    
    # Verify state matches
    assert np.array_equal(game1.get_board(), game2.get_board())
    assert game1.get_current_player() == game2.get_current_player()
    assert game1.get_piece_counts() == game2.get_piece_counts()

# Property 10: Observation Encoding Correctness
@given(valid_move_sequence(), st.integers(0, 1))
def test_observation_encoding_correctness(moves, agent_player):
    """Feature: othello-rl-environment, Property 10: Observation Encoding Correctness"""
    env = OthelloEnv()
    env.reset()
    env.agent_player = agent_player
    
    # Apply moves
    for move in moves:
        valid_moves = env.game.get_valid_moves()
        if not valid_moves[move]:
            break
        env.step(move)
    
    # Get observation
    obs = env._get_observation()
    
    # Verify shape
    assert obs.shape == (3, 8, 8)
    
    # Verify values in [0, 1]
    assert np.all(obs >= 0) and np.all(obs <= 1)
    
    # Verify channel 0 is agent's pieces
    board = env.game.get_board()
    agent_piece = agent_player + 1
    expected_agent = (board == agent_piece).astype(np.float32)
    assert np.array_equal(obs[0], expected_agent)
    
    # Verify channel 1 is opponent's pieces
    opponent_piece = 3 - agent_piece
    expected_opponent = (board == opponent_piece).astype(np.float32)
    assert np.array_equal(obs[1], expected_opponent)
    
    # Verify channel 2 is valid moves
    valid_moves = env.game.get_valid_moves().reshape(8, 8)
    expected_valid = valid_moves.astype(np.float32)
    assert np.array_equal(obs[2], expected_valid)

# Property 14: Sparse Reward Correctness
@given(valid_move_sequence())
def test_sparse_reward_correctness(moves):
    """Feature: othello-rl-environment, Property 14: Sparse Reward Correctness"""
    env = OthelloEnv(reward_mode="sparse")
    env.reset()
    
    # Play through moves
    for i, move in enumerate(moves):
        valid_moves = env.game.get_valid_moves()
        if not valid_moves[move]:
            break
        
        obs, reward, terminated, truncated, info = env.step(move)
        
        if not terminated:
            # Non-terminal reward should be 0
            assert reward == 0.0
        else:
            # Terminal reward should be +1, -1, or 0
            assert reward in [-1.0, 0.0, 1.0]
            
            # Verify reward matches winner
            winner = env.game.get_winner()
            if winner == 2:  # Draw
                assert reward == 0.0
            elif winner == env.agent_player:
                assert reward == 1.0
            else:
                assert reward == -1.0
```

**Property Test Tags:**
Each property test must include a comment tag in the format:
```python
"""Feature: othello-rl-environment, Property N: [Property Title]"""
```

This enables traceability between design properties and test implementation.

### Integration Testing

**RLlib Integration Tests:**
- Environment registration with Gymnasium
- PPO training loop (10 iterations, verify no crashes)
- DQN training loop (10 iterations, verify no crashes)
- Vectorized environment (4 parallel environments)
- Action masking with PPO
- Checkpoint save and restore

**End-to-End Tests:**
- Complete game from start to finish
- Self-play training for 100 episodes
- Human vs trained agent game
- Agent vs random opponent game
- Agent vs greedy opponent game

### Test Organization

```
rust/othello/
  src/
    lib.rs          # Game engine
    bindings.rs     # PyO3 bindings
  tests/
    unit_tests.rs   # Rust unit tests
    property_tests.rs  # Rust property tests (using proptest)

aip_rl/othello/
  __init__.py
  env.py           # Gymnasium environment
  bindings.py      # Python bindings wrapper
  tests/
    test_bindings.py      # Python bindings unit tests
    test_env.py           # Environment unit tests
    test_properties.py    # Property-based tests (using hypothesis)
    test_integration.py   # RLlib integration tests

scripts/
  test_othello_e2e.py    # End-to-end tests
```

### Continuous Integration

**Test Execution Order:**
1. Rust unit tests (fast, catch basic errors)
2. Rust property tests (medium, verify core logic)
3. Python bindings tests (fast, verify FFI)
4. Python environment unit tests (fast, verify Gymnasium API)
5. Python property tests (slow, comprehensive verification)
6. Integration tests (slowest, verify RLlib compatibility)

**Coverage Goals:**
- Rust code: >90% line coverage
- Python code: >85% line coverage
- All properties: 100% implementation (each property has a test)
- All requirements: >80% coverage by tests
