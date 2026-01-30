"""
Entry point for training an Othello RL agent.

This script provides a command-line interface to train PPO agents on the Othello
environment. The actual training logic is implemented in aip_rl.othello.train.

Example:
    python scripts/train_othello.py --num-gpus-per-worker 0.1
"""

from aip_rl.othello.train import main

if __name__ == "__main__":
    main()
