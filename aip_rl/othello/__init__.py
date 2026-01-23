"""
Othello RL Environment for Gymnasium.

This package provides a high-performance Othello (Reversi) environment for
reinforcement learning, built with a Rust game engine and wrapped in a
Gymnasium-compatible Python interface.

The environment supports:
- Self-play and opponent policies (random, greedy, custom)
- Multiple reward modes (sparse, dense, custom)
- Action masking for invalid move handling
- Multiple rendering modes (human, ANSI, RGB array)
- State persistence for replay analysis
- Integration with Ray RLlib and other RL frameworks

Example:
    >>> import gymnasium as gym
    >>> import aip_rl.othello
    >>> env = gym.make("Othello-v0")
    >>> observation, info = env.reset()
    >>> action = env.action_space.sample()
    >>> observation, reward, terminated, truncated, info = env.step(action)

Registered Environments:
    - Othello-v0: Standard Othello environment with 60 max steps per episode

Classes:
    OthelloEnv: Main Gymnasium environment class

For detailed documentation, see the README.md file or the OthelloEnv class
docstring.
"""

from gymnasium.envs.registration import register
from aip_rl.othello.env import OthelloEnv

register(
    id="Othello-v0",
    entry_point="aip_rl.othello.env:OthelloEnv",
    max_episode_steps=60,
)

__all__ = ["OthelloEnv"]
