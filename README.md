# Othello RL Environment

A high-performance, Gymnasium-compatible reinforcement learning environment for training agents to play Othello (Reversi). Built with a Rust game engine for speed and wrapped in Python for easy integration with modern RL frameworks like Ray RLlib.

## Features

- **High Performance**: Rust-based game engine for fast execution (<1ms per step)
- **Gymnasium Compatible**: Standard RL interface for seamless integration
- **Flexible Configuration**: Multiple reward modes, opponent policies, and observation formats
- **Self-Play Support**: Train agents against themselves or various opponent policies
- **Action Masking**: Built-in support for invalid action masking
- **Visualization**: Multiple rendering modes (ANSI, RGB array, human-readable)
- **RLlib Integration**: Works with PPO, DQN, APPO, and other RLlib algorithms
- **Property-Based Testing**: Comprehensive test suite ensuring correctness

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (for building from source)
- Poetry (recommended) or pip

### Install from Source

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install Python dependencies:
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -e .
```

3. Build the Rust extension:
```bash
# Install maturin if not already installed
pip install maturin

# Build and install the Rust extension (run from project root)
maturin develop --release --manifest-path rust/othello/Cargo.toml
```

The Rust extension will be built and installed as `othello_rust` module.

### Verify Installation

```python
import gymnasium as gym
import aip_rl.othello

env = gym.make("Othello-v0")
print("Environment created successfully!")
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
import aip_rl.othello

# Create environment
env = gym.make("Othello-v0")

# Reset environment
observation, info = env.reset()

# Run episode
done = False
total_reward = 0

while not done:
    # Get valid moves from info
    action_mask = info["action_mask"]
    
    # Select random valid action
    import numpy as np
    valid_actions = np.where(action_mask)[0]
    action = np.random.choice(valid_actions)
    
    # Take step
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward}")
```

### Training an Othello Agent

#### Quick Start: Train for 10 Iterations
```bash
python scripts/train_othello.py --num-iterations 10 --checkpoint-freq 5
```

#### Fast Training: Self-Play with 4 Workers
```bash
python scripts/train_othello.py \
  --num-iterations 100 \
  --checkpoint-freq 10 \
  --opponent self \
  --num-workers 4 \
  --lr 0.0001
```

#### Training Against Greedy Opponent
```bash
python scripts/train_othello.py \
  --num-iterations 200 \
  --opponent greedy \
  --reward-mode heuristic \
  --num-workers 4
```

#### Training Against External Engine Opponents

Training against `aelskels` (strategic AI with lookahead):
```bash
python scripts/train_othello.py \
  --num-iterations 200 \
  --opponent aelskels \
  --num-workers 4
```

Training against `drohh` (minimax AI):
```bash
python scripts/train_othello.py \
  --num-iterations 200 \
  --opponent drohh \
  --num-workers 4
```

Training against `nealetham` (naive greedy AI):
```bash
python scripts/train_othello.py \
  --num-iterations 200 \
  --opponent nealetham \
  --num-workers 4
```

Training against multiple opponents (diverse opponents):
```bash
python scripts/train_othello.py \
  --num-iterations 500 \
  --opponent "aelskels,drohh,nealetham,random,greedy" \
  --num-workers 8
```

#### Full Training Run (Longer)
```bash
python scripts/train_othello.py \
  --num-iterations 500 \
  --checkpoint-freq 25 \
  --opponent random \
  --reward-mode sparse \
  --num-workers 8 \
  --num-gpus 1 \
  --checkpoint-dir ./checkpoints/full_run
```

#### Train Programmatically
```python
from aip_rl.othello.train import train_othello
import argparse

args = argparse.Namespace(
    num_iterations=100,
    checkpoint_freq=10,
    checkpoint_dir="checkpoints",
    opponent="random",
    reward_mode="sparse",
    start_player="random",
    lr=0.0001,
    gamma=0.99,
    lambda_=0.95,
    clip_param=0.2,
    train_batch_size=8000,
    minibatch_size=256,
    num_sgd_iter=20,
    num_workers=4,
    num_gpus=0,
    num_cpus=None,
    eval_interval=10,
    eval_duration=20,
)

train_othello(args)
```

#### All Available Options
```bash
python scripts/train_othello.py --help
```

**Common Options:**
- `--num-iterations`: Number of training iterations (default: 200)
- `--checkpoint-freq`: Save checkpoint every N iterations (default: 20)
- `--opponent`: Opponent type - `random`, `greedy` (default: `random`)
- `--reward-mode`: Reward structure - `sparse`, `heuristic` (default: `sparse`)
- `--num-workers`: Parallel environment workers (default: 4)
- `--num-gpus`: GPUs to use (default: 0)
- `--lr`: Learning rate (default: 0.0001)

**Output:**
Checkpoints are saved to `checkpoints/` directory:
```
checkpoints/
├── iter_000010/    # Checkpoint after 10 iterations
├── iter_000020/    # Checkpoint after 20 iterations
└── final/          # Final trained model
```

### With Rendering

```python
import gymnasium as gym
import aip_rl.othello

env = gym.make("Othello-v0", render_mode="human")
observation, info = env.reset()

for _ in range(10):
    action_mask = info["action_mask"]
    valid_actions = np.where(action_mask)[0]
    action = np.random.choice(valid_actions)
    
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # Prints board to console
    
    if terminated:
        break
```

## Configuration Options

The environment supports extensive configuration through initialization parameters:

```python
env = gym.make(
    "Othello-v0",
    opponent="random",            # Opponent policy
    reward_mode="sparse",         # Reward structure
    invalid_move_penalty=-1.0,    # Penalty for invalid moves
    invalid_move_mode="penalty",  # How to handle invalid moves
    render_mode="human"           # Rendering mode
)
```

### Opponent Policies

Built-in opponents:
- **`"random"`** (default): Random opponent that selects random valid moves
- **`"greedy"`**: Greedy opponent that maximizes pieces flipped

External engines (high-performance AI):
- **`"aelskels"`**: Alpha-beta pruning AI with 5-turn lookahead (~17ms/move)
- **`"drohh"`**: Minimax with strategic evaluation (~0.5ms/move)
- **`"nealetham"`**: Naive greedy AI maximizing immediate capture (~0.08ms/move)

Custom:
- **`callable`**: Custom policy function that takes observation and returns action

Example with custom opponent:
```python
def my_opponent_policy(observation):
    # Your custom logic here
    # observation shape: (3, 8, 8)
    # Return action in range [0, 63]
    return action

env = gym.make("Othello-v0", opponent=my_opponent_policy)
```

### External Engine Algorithms

The three external engines implement different strategic approaches to Othello:

#### aelskels - Alpha-Beta Pruning (Strategic)
- **Algorithm**: Minimax with alpha-beta pruning
- **Lookahead**: 5 moves deep
- **Heuristic**: Piece count + mobility + corner control
- **Performance**: ~17ms/move
- **Style**: Aggressive, long-term strategic play
- **Best for**: Training agents against sophisticated opponents

#### drohh - Minimax AI (Balanced)
- **Algorithm**: Minimax with alpha-beta pruning
- **Lookahead**: 5 moves deep
- **Heuristic**: Mobility + piece count + corner weighting
- **Performance**: ~0.5ms/move
- **Style**: Balanced strategic play
- **Best for**: General training with good opponent quality

#### nealetham - Greedy AI (Fast)
- **Algorithm**: Greedy heuristic evaluation
- **Lookahead**: 0 moves (immediate)
- **Heuristic**: Maximizes immediate piece captures
- **Performance**: ~0.08ms/move
- **Style**: Aggressive capture-oriented play
- **Best for**: Fast training sessions, baseline opponent

### Reward Modes

#### Sparse Rewards (default)
```python
env = gym.make("Othello-v0", reward_mode="sparse")
```
- Returns 0 during the game
- Returns +1 for win, -1 for loss, 0 for draw at game end

#### Dense Rewards
```python
env = gym.make("Othello-v0", reward_mode="dense")
```
- Returns normalized piece differential at each step: `(agent_pieces - opponent_pieces) / 64`
- Provides intermediate feedback for learning

#### Custom Rewards
```python
def custom_reward_fn(game_state):
    # game_state contains: board, black_count, white_count, 
    # current_player, agent_player, game_over, pieces_flipped
    return reward_value

env = gym.make("Othello-v0", reward_mode="custom", reward_fn=custom_reward_fn)
```

### Invalid Move Handling

- **`"penalty"`** (default): Apply penalty and maintain state
- **`"random"`**: Automatically select random valid move
- **`"error"`**: Raise ValueError exception

```python
env = gym.make(
    "Othello-v0",
    invalid_move_mode="penalty",
    invalid_move_penalty=-1.0
)
```

### Render Modes

- **`"human"`**: Print board to console
- **`"ansi"`**: Return string representation
- **`"rgb_array"`**: Return RGB numpy array (512x512x3) for video recording

```python
env = gym.make("Othello-v0", render_mode="rgb_array")
observation, info = env.reset()
rgb_frame = env.render()  # Returns numpy array
```

## Observation Space

The observation is a 3D numpy array with shape `(3, 8, 8)` and dtype `float32`:

- **Channel 0**: Agent's pieces (1 where agent has pieces, 0 otherwise)
- **Channel 1**: Opponent's pieces (1 where opponent has pieces, 0 otherwise)
- **Channel 2**: Valid moves (1 for valid positions, 0 otherwise)

All values are normalized to [0, 1].

```python
observation, info = env.reset()
print(observation.shape)  # (3, 8, 8)
print(observation.dtype)  # float32

agent_pieces = observation[0]      # Agent's pieces
opponent_pieces = observation[1]   # Opponent's pieces
valid_moves = observation[2]       # Valid move mask
```

## Action Space

The action space is `Discrete(64)`, representing the 64 board positions:

- Actions are integers in range [0, 63]
- Mapping: `action = row * 8 + col`
- Inverse: `row = action // 8`, `col = action % 8`

```python
# Action 0 = top-left corner (row=0, col=0)
# Action 7 = top-right corner (row=0, col=7)
# Action 63 = bottom-right corner (row=7, col=7)
```

### Action Masking

The environment provides action masks in the info dictionary to indicate valid moves:

```python
observation, info = env.reset()
action_mask = info["action_mask"]  # Boolean array of shape (64,)

# Get valid actions
valid_actions = np.where(action_mask)[0]

# Select from valid actions only
action = np.random.choice(valid_actions)
```

## Info Dictionary

The info dictionary returned by `step()` and `reset()` contains:

```python
{
    "action_mask": np.ndarray,    # Boolean array (64,) of valid moves
    "current_player": int,        # 0=Black, 1=White
    "black_count": int,           # Number of black pieces
    "white_count": int,           # Number of white pieces
    "agent_player": int,          # Agent's color (0=Black, 1=White)
}
```

## Training with Ray RLlib

### Using the Training Script (Recommended)

The easiest way to train a PPO agent is using the built-in training script:

```bash
# Basic training
python scripts/train_othello.py --num-iterations 100

# With custom options
python scripts/train_othello.py \
  --num-iterations 200 \
  --opponent random \
  --reward-mode sparse \
  --num-workers 8 \
  --checkpoint-freq 25
```

See the "Training an Othello Agent" section above for more examples.

### Using the Train Module Programmatically

```python
from aip_rl.othello.train import train_othello
import argparse

args = argparse.Namespace(
    num_iterations=100,
    checkpoint_freq=10,
    checkpoint_dir="checkpoints",
    opponent="random",
    reward_mode="sparse",
    start_player="random",
    lr=0.0001,
    gamma=0.99,
    lambda_=0.95,
    clip_param=0.2,
    train_batch_size=8000,
    minibatch_size=256,
    num_sgd_iter=20,
    num_workers=4,
    num_gpus=0,
    num_cpus=None,
    eval_interval=10,
    eval_duration=20,
)

train_othello(args)
```

### Basic PPO Training (Custom Implementation)

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import aip_rl.othello

ray.init()

config = (
    PPOConfig()
    .environment(
        env="Othello-v0",
        env_config={
            "opponent": "random",
            "reward_mode": "sparse",
        }
    )
    .framework("torch")
    .env_runners(num_env_runners=4)
    .training(
        train_batch_size=8000,
        lr=0.0003,
    )
)

algo = config.build()

# Train for 100 iterations
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: Reward = {result['env_runners']['episode_return_mean']:.2f}")

# Save checkpoint
checkpoint = algo.save()
print(f"Checkpoint saved: {checkpoint}")

algo.stop()
ray.shutdown()
```

### Using a Custom CNN Model

The training script automatically uses an enhanced CNN model with residual connections. To use it in a custom training loop:

```python
from ray.rllib.models import ModelCatalog
from aip_rl.othello.models import OthelloCNN

# Register the model
ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)

# Use in config
config = PPOConfig().model({"custom_model": "othello_cnn"})

algo = config.build()
for _ in range(100):
    algo.train()
```

The OthelloCNN model features:
- **3 input channels**: Agent pieces, opponent pieces, valid move mask
- **128 channels**: Initial convolution with batch norm
- **Residual blocks**: 2 residual blocks (128 channels) for feature extraction
- **256 channels**: Expansion convolution with batch norm
- **1 residual block**: Final residual block (256 channels)
- **1024 hidden units**: Fully connected layer
- **Action masking**: Built-in support for masking invalid actions
- **~11.5M parameters**: Sufficient capacity for strategic game learning

### Action Masking with RLlib

RLlib can use action masks to prevent invalid actions:

```python
config = (
    PPOConfig()
    .environment(env="Othello-v0")
    .training(
        # Action masking is automatically handled via info["action_mask"]
    )
)
```

The environment automatically provides action masks in the info dictionary, which RLlib uses to mask out invalid actions during training.

### Vectorized Environments

For faster training, use multiple parallel environments:

```python
config = (
    PPOConfig()
    .environment(env="Othello-v0")
    .env_runners(
        num_env_runners=8,           # 8 parallel workers
        num_envs_per_env_runner=4,   # 4 environments per worker
    )
)

# This creates 8 * 4 = 32 parallel environments
```

## Agent Evaluation: ELO Ratings

### Computing Pair-Game Performance with ELO Ratings

Evaluate agent performance using round-robin tournaments with Elo rating calculation:

#### Basic Evaluation (Deterministic)
```bash
python scripts/play_agent_vs_agent_elo.py --folder zoo --games-per-side 50 --rounds 1
```

#### Evaluation with Soft Engines (Probabilistic - Recommended)
```bash
python scripts/play_agent_vs_agent_elo.py \
  --folder zoo \
  --games-per-side 50 \
  --soft-engines \
  --temperature 1.0 \
  --rounds 1
```

#### Soft Engine Evaluation with Top-K Sampling
```bash
python scripts/play_agent_vs_agent_elo.py \
  --folder zoo \
  --games-per-side 50 \
  --soft-engines \
  --temperature 0.8 \
  --top-k 5 \
  --rounds 5
```

#### Extended Evaluation with Multiple Rounds
```bash
python scripts/play_agent_vs_agent_elo.py \
  --folder zoo \
  --games-per-side 100 \
  --soft-engines \
  --temperature 1.0 \
  --rounds 3 \
  --k-factor 32.0
```

### ELO Evaluation Options

**Core Arguments:**
- `--folder`: Directory containing agent checkpoint folders (default: `zoo`)
- `--games-per-side`: Number of games per side in each matchup (default: 50)
- `--rounds`: Number of evaluation rounds to run (default: 1)
- `--k-factor`: Elo K-factor controlling rating volatility (default: 32.0)
- `--initial-rating`: Starting rating for new agents (default: 1000.0)

**Soft Engine Options (for probabilistic evaluation):**
- `--soft-engines`: Enable probabilistic engine decisions instead of deterministic
- `--temperature`: Sampling temperature for soft engines (range: 0.1-10.0, default: 1.0)
  - Lower values (e.g., 0.1-0.5): More deterministic, strategic play
  - Mid values (e.g., 0.8-1.2): Balanced randomness
  - Higher values (e.g., 5.0-10.0): More exploratory, varied play
- `--top-k`: Restrict sampling to top-k legal moves (optional, default: all moves)

**Performance Options:**
- `--cpu-only`: Force CPU loading for checkpoints (default: true)

### Output Files

The script generates two files in the target folder:

1. **`elo.json`**: Current Elo ratings for all agents
2. **`matchups.json`**: Detailed statistics for each matchup:
   - Black wins, White wins, Draws
   - Win rates by side

Example output structure:
```json
{
  "agent_1": 1050.3,
  "agent_2": 980.5,
  "random": 850.0,
  "greedy": 920.0
}
```

### Why Use Soft Engines?

Soft (probabilistic) engines provide more realistic evaluation:
- **Avoid Deterministic Loops**: Prevents overfitting to specific engine strategies
- **Varied Play**: Different temperature settings produce different playstyles
- **Realistic Variance**: Games have different outcomes despite same opponents
- **Better Statistics**: More varied results with soft engines improve rating reliability

## Advanced Features

### State Persistence

Save and load game states for replay analysis:

```python
env = gym.make("Othello-v0")
observation, info = env.reset()

# Play some moves
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

# Save state
state = env.save_state()

# Later, restore state
env.load_state(state)
```

### Human vs Agent Games

Play against a trained agent in the console or with a GUI!

**Console Version:**
```python
# See scripts/play_human_vs_agent.py for full implementation
import gymnasium as gym
import aip_rl.othello

env = gym.make("Othello-v0", render_mode="human")
observation, info = env.reset()

while True:
    env.render()
    
    # Human move
    action_mask = info["action_mask"]
    print(f"Valid moves: {np.where(action_mask)[0]}")
    action = int(input("Enter your move (0-63): "))
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        env.render()
        print(f"Game over! Reward: {reward}")
        break
```

**GUI Version (Recommended):**
```bash
# Install pygame
poetry install --extras gui

# Play with beautiful graphics!
python scripts/play_human_vs_agent_gui.py --opponent greedy
```

Features:
- Click to make moves (no typing!)
- Visual indicators for valid moves
- Real-time score display
- Play against random, greedy, or trained agents

See `scripts/README_HUMAN_INTERACTION.md` for more details.

### Watch Agents Play

Observe two agents playing against each other:

```python
# See scripts/watch_agents_play.py for full implementation
import time

env = gym.make("Othello-v0", opponent="greedy", render_mode="human")
observation, info = env.reset()

while True:
    env.render()
    time.sleep(1)  # Pause for visualization
    
    # Agent move
    action_mask = info["action_mask"]
    valid_actions = np.where(action_mask)[0]
    action = np.random.choice(valid_actions)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        env.render()
        break
```

## Project Structure

```
.
├── aip_rl/
│   └── othello/
│       ├── __init__.py          # Environment registration & module exports
│       ├── env.py               # Gymnasium environment wrapper
│       ├── models.py            # OthelloCNN model for training
│       ├── train.py             # PPO training logic and CLI
│       └── tests/
│           ├── test_env.py      # Unit tests
│           └── test_properties.py  # Property-based tests
├── rust/
│   └── othello/
│       ├── src/
│       │   ├── lib.rs           # Rust game engine
│       │   └── bindings.rs      # PyO3 Python bindings
│       ├── tests/
│       │   ├── test_bindings.py
│       │   └── test_bindings_properties.py
│       └── Cargo.toml
├── scripts/
│   ├── train_othello.py         # Training script entry point
│   ├── play_human_vs_agent.py   # Human interaction (console)
│   ├── play_human_vs_agent_gui.py  # Human interaction (GUI)
│   └── watch_agents_play.py     # Spectator mode
└── README.md
```

**Key Modules:**

- **`aip_rl/othello/env.py`**: Core Gymnasium environment (1165 lines)
- **`aip_rl/othello/models.py`**: OthelloCNN neural network model for training
- **`aip_rl/othello/train.py`**: PPO training with Ray RLlib support
- **`scripts/train_othello.py`**: Command-line entry point for training

## Testing

Run the test suite:

```bash
# Python tests
poetry run pytest aip_rl/othello/tests/

# Rust tests
cd rust/othello
cargo test

# Python binding tests
poetry run pytest rust/othello/tests/

# Property-based tests (with more iterations)
poetry run pytest --hypothesis-iterations=1000
```

## Performance

The Rust-based game engine provides excellent performance:

- **Step time**: <1ms per step (typical)
- **Vectorized**: Supports 100+ parallel environments
- **Memory efficient**: Minimal allocations during gameplay

Benchmark on typical hardware (M1 Mac):
- Single environment: ~1000 steps/second
- 32 parallel environments: ~25,000 steps/second

## Troubleshooting

### Import Error: `othello_rust` module not found

The Rust extension needs to be built with maturin:
```bash
# Install maturin if not already installed
pip install maturin

# Build and install the Rust extension (run from project root)
maturin develop --release --manifest-path rust/othello/Cargo.toml
```

### Invalid Move Errors

Ensure you're using action masking:
```python
action_mask = info["action_mask"]
valid_actions = np.where(action_mask)[0]
action = np.random.choice(valid_actions)
```

Or configure the environment to handle invalid moves:
```python
env = gym.make("Othello-v0", invalid_move_mode="random")
```

### External Engine Issues

**Engine not recognized**:
```bash
# Make sure the engines are available
python -c "from aip_rl.othello.engines import get_available_engines; print(get_available_engines())"
```

Expected output:
```
['aelskels', 'drohh', 'nealetham']
```

**Engine moves seem slow**:
- `aelskels` uses deeper lookahead (5 moves) and is slower (~17ms/move)
- `drohh` provides a balance (~0.5ms/move)
- `nealetham` is fastest (~0.08ms/move) but less strategic
- For training with many workers, use `nealetham` for speed

**Engine not responding**:
Ensure the PyO3 bindings are properly built:
```bash
unset CONDA_PREFIX  # If using virtual env
maturin develop --release --manifest-path rust/othello/Cargo.toml
```

### RLlib Training Issues

Make sure the environment is properly registered:
```python
import aip_rl.othello  # This registers the environment
```

For action masking, ensure your RLlib version supports it (Ray 2.0+).

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code follows style guidelines: `black .` and `flake8`
3. Property-based tests are included for new features
4. Documentation is updated

## License

[Your License Here]

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{othello_rl_env,
  title = {Othello RL Environment},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/othello-rl}
}
```

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- Powered by [Rust](https://www.rust-lang.org/) and [PyO3](https://pyo3.rs/)
- Integrated with [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
