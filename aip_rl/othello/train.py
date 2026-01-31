"""
Training script for Othello RL agent using PPO and Ray RLlib.

This module provides functionality to train a PPO agent on the Othello environment
with support for various opponent types, reward modes, and hyperparameter tuning.
"""

import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
import gymnasium as gym

# Register Othello environment
import aip_rl.othello  # noqa: F401

# Import the model
from aip_rl.othello.models import OthelloCNN


def train_othello(args):
    """Train a PPO agent on Othello-v0 environment.

    Args:
        args: Parsed command-line arguments containing training configuration.
    """
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    # Register custom model
    ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)

    # Define environment creator to ensure registration in workers
    def env_creator(env_config):
        import aip_rl.othello  # noqa: F401 - needed to register environment

        return gym.make("Othello-v0", **env_config)

    # Register environment with Ray
    from ray.tune.registry import register_env

    register_env("Othello-v0", env_creator)

    # Prepare opponent list (allow comma-separated builtins/checkpoints)
    opponent_specs = [
        spec.strip()
        for spec in args.opponent.split(",")
        if spec.strip()
    ]
    if not opponent_specs:
        opponent_specs = ["random", "greedy"]

    # Configure PPO with action masking
    # Note: Using old API stack to support custom ModelV2 models
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="Othello-v0",
            env_config={
                "opponent": opponent_specs,
                "reward_mode": args.reward_mode,
                "invalid_move_mode": "penalty",
                "start_player": args.start_player,
            },
        )
        .framework("torch")
        .resources(
            num_cpus_per_worker=1,
            num_gpus_per_worker=args.num_gpus_per_worker,
            num_gpus=args.num_gpus,
        )
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            grad_clip=0.5,  # Clip gradients to prevent NaN/Inf
            grad_clip_by="global_norm",
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_duration=args.eval_duration,
            evaluation_num_env_runners=1,
        )
    )
    config["num_env_runners"] = args.num_workers

    # Set custom model (model is a dict attribute, not a method)
    # Include max_seq_len for RNN compatibility
    # (even though we're not using RNNs)
    config.model = {
        "custom_model": "othello_cnn",
        "max_seq_len": 1,  # Not using RNNs, but required by PPO
    }

    algo = config.build()

    if args.resume_checkpoint:
        print(f"Restoring checkpoint from {args.resume_checkpoint}...")
        algo.restore(args.resume_checkpoint)

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print(f"Starting training for {args.num_iterations} iterations...")

    for i in range(args.num_iterations):
        result = algo.train()

        print(f"\nIteration {i + 1}/{args.num_iterations}")
        if "env_runners" in result:
            if "episode_return_mean" in result["env_runners"]:
                reward_mean = result["env_runners"]["episode_return_mean"]
                print(f"  Reward Mean: {reward_mean:.2f}")
            if "episode_len_mean" in result["env_runners"]:
                episode_len = result["env_runners"]["episode_len_mean"]
                print(f"  Episode Length: {episode_len:.2f}")

        if (i + 1) % args.checkpoint_freq == 0:
            iter_dir = os.path.join(checkpoint_dir, f"iter_{i + 1:06d}")
            os.makedirs(iter_dir, exist_ok=True)
            checkpoint = algo.save(checkpoint_dir=iter_dir)
            print(f"  Checkpoint: {checkpoint}")

    # Final checkpoint
    final_dir = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_checkpoint = algo.save(checkpoint_dir=final_dir)
    print(f"\nTraining complete! Final checkpoint: {final_checkpoint}")

    algo.stop()
    ray.shutdown()


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train PPO agent on Othello-v0")

    # Training parameters
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="Save checkpoint every N iterations (default: 20)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to write checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to an existing checkpoint to restore before training (default: None)",
    )

    # Environment parameters
    parser.add_argument(
        "--opponent",
        type=str,
        default="random,greedy",
        help=(
            "Comma-separated opponent list (built-in 'random', 'greedy', and/or "
            "checkpoint paths, default: random + greedy)"
        ),
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="sparse",
        choices=["sparse", "heuristic"],
        help="Reward mode (default: sparse)",
    )
    parser.add_argument(
        "--start-player",
        type=str,
        default="black",
        choices=["black", "white", "random"],
        help="Agent starting side (default: black)",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,  # Reduced from 0.0003 for stability
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.95,
        dest="lambda_",
        help="GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="PPO clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8000,
        help="Training batch size (default: 8000)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for SGD (default: 256)",
    )
    parser.add_argument(
        "--num-sgd-iter",
        type=int,
        default=20,
        help="Number of SGD iterations per training batch (default: 20)",
    )

    # Resource parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (default: 0)",
    )
    parser.add_argument(
        "--num-gpus-per-worker",
        type=float,
        default=None,
        help="GPUs per worker (default: 1 / (num_workers + 2))",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for Ray (default: all available)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluate every N iterations (default: 10)",
    )
    parser.add_argument(
        "--eval-duration",
        type=int,
        default=20,
        help="Number of episodes per evaluation (default: 20)",
    )

    args = parser.parse_args()
    if args.num_gpus_per_worker is None:
        args.num_gpus_per_worker = args.num_gpus / (args.num_workers + 2)
    train_othello(args)


if __name__ == "__main__":
    main()
