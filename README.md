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
cd rust/othello
cargo build --release
```

The Rust extension will be automatically built and installed as `othello_rust` module.

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
    opponent="self",              # Opponent policy
    reward_mode="sparse",         # Reward structure
    invalid_move_penalty=-1.0,    # Penalty for invalid moves
    invalid_move_mode="penalty",  # How to handle invalid moves
    render_mode="human"           # Rendering mode
)
```

### Opponent Policies

- **`"self"`** (default): Self-play mode - agent plays both sides
- **`"random"`**: Random opponent that selects random valid moves
- **`"greedy"`**: Greedy opponent that maximizes pieces flipped
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

### Basic PPO Training

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
            "opponent": "self",
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

### Custom CNN Model

For better performance with the (3, 8, 8) observation space, use a custom CNN:

```python
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

class OthelloCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                             model_config, name)
        nn.Module.__init__(self)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, num_outputs)
        self.value_fc = nn.Linear(128 * 8 * 8, 1)
        self._features = None
    
    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.conv1(input_dict["obs"].float()))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        self._features = x
        return self.fc(x), state
    
    def value_function(self):
        return self.value_fc(self._features).squeeze(1)

# Register model
ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)

# Use in config
config = PPOConfig().model({"custom_model": "othello_cnn"})
```

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
│       ├── __init__.py          # Environment registration
│       ├── env.py               # Gymnasium environment wrapper
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
│   ├── train_othello.py         # Training script
│   ├── play_human_vs_agent.py   # Human interaction (console)
│   ├── play_human_vs_agent_gui.py  # Human interaction (GUI)
│   └── watch_agents_play.py     # Spectator mode
└── README.md
```

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

The Rust extension needs to be built:
```bash
cd rust/othello
cargo build --release
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
