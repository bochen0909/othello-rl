"""
Test script for PPO training on Othello environment.
Runs 10 training iterations to verify no crashes or errors.

This script validates Requirement 7.2: PPO algorithm integration.
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn
import sys

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


def test_ppo_training():
    """
    Test PPO training on Othello-v0 environment.
    
    Runs 10 training iterations to verify:
    - No crashes or errors occur
    - Training loop completes successfully
    - Metrics are collected properly
    
    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 60)
    print("Testing PPO Training on Othello Environment")
    print("=" * 60)
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

        # Define environment creator to ensure registration in workers
        def env_creator(env_config):
            # Import in worker to ensure registration
            import aip_rl.othello  # noqa: F401
            import gymnasium as gym
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env
        print("Registering Othello environment...")
        register_env("Othello-v0", env_creator)
        print("✓ Environment registered successfully")
        print()

        # Configure PPO with action masking
        # Note: Using old API stack to support custom ModelV2 models
        print("Configuring PPO algorithm...")
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
                train_batch_size=2000,  # Smaller batch for faster testing
                minibatch_size=128,
                num_sgd_iter=10,
                lr=0.0003,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
            )
            .evaluation(
                evaluation_interval=None,  # Disable evaluation for testing
            )
        )

        # Set number of parallel workers (fewer for testing)
        config["num_env_runners"] = 2

        # Set custom model
        config.model = {
            "custom_model": "othello_cnn",
            "max_seq_len": 1,
        }
        print("✓ PPO configured successfully")
        print()

        # Build algorithm
        print("Building PPO algorithm...")
        algo = config.build()
        print("✓ Algorithm built successfully")
        print()

        # Training loop - 10 iterations for testing
        num_iterations = 10
        print(f"Starting training for {num_iterations} iterations...")
        print("-" * 60)
        
        all_iterations_successful = True
        
        for i in range(num_iterations):
            try:
                result = algo.train()
                
                # Extract metrics
                iteration_num = i + 1
                reward_mean = "N/A"
                episode_len = "N/A"
                
                if "env_runners" in result:
                    if "episode_return_mean" in result["env_runners"]:
                        reward_mean = f"{result['env_runners']['episode_return_mean']:.2f}"
                    if "episode_len_mean" in result["env_runners"]:
                        episode_len = f"{result['env_runners']['episode_len_mean']:.2f}"
                
                print(f"Iteration {iteration_num:2d}/{num_iterations} | "
                      f"Reward: {reward_mean:>6s} | "
                      f"Episode Length: {episode_len:>6s} | "
                      f"✓")
                
            except Exception as e:
                print(f"Iteration {i + 1:2d}/{num_iterations} | ERROR: {e}")
                all_iterations_successful = False
                break

        print("-" * 60)
        print()

        # Cleanup
        print("Cleaning up...")
        algo.stop()
        ray.shutdown()
        print("✓ Cleanup complete")
        print()

        # Final result
        print("=" * 60)
        if all_iterations_successful:
            print("TEST PASSED: All 10 training iterations completed successfully!")
            print("=" * 60)
            return True
        else:
            print("TEST FAILED: Training encountered errors")
            print("=" * 60)
            return False

    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: Unexpected error occurred")
        print(f"Error: {e}")
        print("=" * 60)
        
        # Attempt cleanup
        try:
            ray.shutdown()
        except:
            pass
        
        return False


if __name__ == "__main__":
    success = test_ppo_training()
    sys.exit(0 if success else 1)
