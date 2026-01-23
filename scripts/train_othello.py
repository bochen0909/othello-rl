"""
Ray RLlib Training Script for Othello
Trains a PPO agent with custom CNN model on Othello-v0 environment
"""

import argparse
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

# Register Othello environment
import aip_rl.othello  # noqa: F401


class OthelloCNN(TorchModelV2, nn.Module):
    """
    Custom CNN model for Othello board.

    Architecture:
    - Input: (3, 8, 8) - 3 channels (agent pieces, opponent pieces,
      valid moves)
    - Conv1: 3 -> 64 channels, 3x3 kernel, padding=1
    - Conv2: 64 -> 128 channels, 3x3 kernel, padding=1
    - Conv3: 128 -> 128 channels, 3x3 kernel, padding=1
    - Flatten: 128 * 8 * 8 = 8192 features
    - FC1: 8192 -> 512
    - FC2 (policy): 512 -> 64 (action logits)
    - Value FC: 512 -> 1 (value function)
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
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
        """
        Forward pass through the network.

        Args:
            input_dict: Dictionary containing 'obs' key with observations
            state: RNN state (not used for this feedforward network)
            seq_lens: Sequence lengths (not used for this feedforward network)

        Returns:
            logits: Action logits of shape (batch_size, num_outputs)
            state: Unchanged state
        """
        x = input_dict["obs"].float()

        # CNN forward pass with ReLU activations
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
        """
        Compute value function from cached features.

        Returns:
            value: Value estimates of shape (batch_size,)
        """
        return self.value_fc(self._features).squeeze(1)


def train_othello(args):
    """Train a PPO agent on Othello-v0 environment."""
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    # Register custom model
    ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)

    # Define environment creator to ensure registration in workers
    def env_creator(env_config):
        import gymnasium as gym
        return gym.make("Othello-v0", **env_config)

    # Register environment with Ray
    from ray.tune.registry import register_env
    register_env("Othello-v0", env_creator)

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
                "opponent": args.opponent,
                "reward_mode": args.reward_mode,
                "invalid_move_mode": "penalty",
            },
        )
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_duration=args.eval_duration,
            evaluation_num_env_runners=1,
        )
    )

    # Set number of parallel workers
    config["num_env_runners"] = args.num_workers

    # Set custom model (model is a dict attribute, not a method)
    # Include max_seq_len for RNN compatibility
    # (even though we're not using RNNs)
    config.model = {
        "custom_model": "othello_cnn",
        "max_seq_len": 1,  # Not using RNNs, but required by PPO
    }

    algo = config.build()

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
            checkpoint = algo.save()
            print(f"  Checkpoint: {checkpoint}")

    # Final checkpoint
    final_checkpoint = algo.save()
    print(
        f"\nTraining complete! Final checkpoint: {final_checkpoint}"
    )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Othello-v0"
    )
    
    # Training parameters
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="Save checkpoint every N iterations (default: 20)"
    )
    
    # Environment parameters
    parser.add_argument(
        "--opponent",
        type=str,
        default="self",
        choices=["self", "random"],
        help="Opponent type (default: self)"
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="sparse",
        choices=["sparse", "heuristic"],
        help="Reward mode (default: sparse)"
    )
    
    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate (default: 0.0003)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.95,
        dest="lambda_",
        help="GAE lambda (default: 0.95)"
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="PPO clip parameter (default: 0.2)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8000,
        help="Training batch size (default: 8000)"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for SGD (default: 256)"
    )
    parser.add_argument(
        "--num-sgd-iter",
        type=int,
        default=20,
        help="Number of SGD iterations per training batch (default: 20)"
    )
    
    # Resource parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (default: 0)"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for Ray (default: all available)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluate every N iterations (default: 10)"
    )
    parser.add_argument(
        "--eval-duration",
        type=int,
        default=20,
        help="Number of episodes per evaluation (default: 20)"
    )
    
    args = parser.parse_args()
    train_othello(args)
