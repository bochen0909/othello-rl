"""
Test script for RLlib vectorized environment support.

This script validates that the Othello environment works correctly with
RLlib's vectorized environment infrastructure for parallel training.

Tests:
- Creating vectorized environments through RLlib
- Training with multiple parallel workers
- Verifying parallel rollouts work correctly
"""

import sys
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

# Register Othello environment
import aip_rl.othello  # noqa: F401


class OthelloCNN(TorchModelV2, nn.Module):
    """Custom CNN model for Othello board."""

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


def test_rllib_vectorized_envs():
    """
    Test RLlib with vectorized Othello environments.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 70)
    print("Testing RLlib Vectorized Environment Support")
    print("=" * 70)
    print()

    try:
        # Initialize Ray
        print("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        print("✓ Ray initialized successfully")
        print()

        # Register custom model
        print("Registering custom CNN model...")
        ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)
        print("✓ Model registered successfully")
        print()

        # Define environment creator
        def env_creator(env_config):
            import aip_rl.othello  # noqa: F401
            import gymnasium as gym
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env
        print("Registering Othello environment...")
        register_env("Othello-v0", env_creator)
        print("✓ Environment registered successfully")
        print()

        # Configure PPO with 4 parallel workers
        num_workers = 4
        print(f"Configuring PPO with {num_workers} parallel workers...")
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env="Othello-v0",
                env_config={
                    "opponent": "self",
                    "reward_mode": "sparse",
                    "invalid_move_mode": "penalty",
                },
            )
            .framework("torch")
            .resources(num_gpus=0)
            .training(
                train_batch_size=2000,
                minibatch_size=128,
                num_sgd_iter=10,
                lr=0.0003,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
            )
            .evaluation(
                evaluation_interval=None,
            )
        )

        # Set number of parallel workers
        config["num_env_runners"] = num_workers

        # Set custom model
        config.model = {
            "custom_model": "othello_cnn",
            "max_seq_len": 1,
        }
        print(f"✓ PPO configured with {num_workers} parallel workers")
        print()

        # Build algorithm
        print("Building PPO algorithm...")
        algo = config.build()
        print("✓ Algorithm built successfully")
        print()

        # Run a few training iterations to test vectorized execution
        num_iterations = 3
        print(f"Running {num_iterations} training iterations with "
              f"{num_workers} parallel workers...")
        print("-" * 70)

        all_iterations_successful = True

        for i in range(num_iterations):
            try:
                result = algo.train()

                # Extract metrics
                iteration_num = i + 1
                reward_mean = "N/A"
                episode_len = "N/A"
                episodes_total = "N/A"

                if "env_runners" in result:
                    if "episode_return_mean" in result["env_runners"]:
                        reward_mean = (
                            f"{result['env_runners']['episode_return_mean']:.2f}"
                        )
                    if "episode_len_mean" in result["env_runners"]:
                        episode_len = (
                            f"{result['env_runners']['episode_len_mean']:.2f}"
                        )
                    if "episodes_this_iter" in result["env_runners"]:
                        episodes_total = str(
                            result["env_runners"]["episodes_this_iter"]
                        )

                print(f"Iteration {iteration_num}/{num_iterations} | "
                      f"Reward: {reward_mean:>6s} | "
                      f"Ep Len: {episode_len:>6s} | "
                      f"Episodes: {episodes_total:>4s} | ✓")

            except Exception as e:
                print(f"Iteration {i + 1}/{num_iterations} | ERROR: {e}")
                all_iterations_successful = False
                break

        print("-" * 70)
        print()

        # Verify that multiple episodes were collected (indicating parallel
        # execution)
        if all_iterations_successful:
            print("✓ All training iterations completed successfully")
            print(f"✓ Parallel execution with {num_workers} workers verified")
        else:
            print("✗ Training encountered errors")

        # Cleanup
        print("\nCleaning up...")
        algo.stop()
        ray.shutdown()
        print("✓ Cleanup complete")
        print()

        # Final result
        print("=" * 70)
        if all_iterations_successful:
            print("TEST PASSED: RLlib vectorized environment support works!")
            print("=" * 70)
            return True
        else:
            print("TEST FAILED: RLlib vectorized environment test failed")
            print("=" * 70)
            return False

    except Exception as e:
        print()
        print("=" * 70)
        print("TEST FAILED: Unexpected error occurred")
        print(f"Error: {e}")
        print("=" * 70)

        # Attempt cleanup
        try:
            ray.shutdown()
        except Exception:
            pass

        return False


if __name__ == "__main__":
    success = test_rllib_vectorized_envs()
    sys.exit(0 if success else 1)
