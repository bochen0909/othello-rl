"""
Integration tests for Othello RL training pipeline.

These tests verify that the entire training pipeline works end-to-end,
including model initialization, environment integration, PPO configuration,
and training loop execution.
"""

import os
import pytest
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Ray imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# Environment and training imports
import aip_rl.othello  # noqa: F401
import gymnasium as gym


class OthelloCNN(TorchModelV2, nn.Module):
    """
    Enhanced CNN model for Othello board with residual connections.
    This is a copy of the model from train_othello.py for testing.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(128)

        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(128)

        # Expansion convolution
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # Final residual block
        self.res3_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res3_bn1 = nn.BatchNorm2d(256)
        self.res3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res3_bn2 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_outputs)

        # Value function head
        self.value_fc = nn.Linear(1024, 1)

        self._features = None

    def _residual_block(self, x, conv1, bn1, conv2, bn2):
        """Apply a residual block with skip connection."""
        identity = x
        out = torch.relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        out += identity  # Skip connection
        out = torch.relu(out)
        return out

    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network with action masking support."""
        x = input_dict["obs"].float()

        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # Initial convolution
        x = torch.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self._residual_block(
            x, self.res1_conv1, self.res1_bn1, self.res1_conv2, self.res1_bn2
        )
        x = self._residual_block(
            x, self.res2_conv1, self.res2_bn1, self.res2_conv2, self.res2_bn2
        )

        # Expansion
        x = torch.relu(self.bn2(self.conv2(x)))

        # Final residual block
        x = self._residual_block(
            x, self.res3_conv1, self.res3_bn1, self.res3_conv2, self.res3_bn2
        )

        # Flatten
        x_flat = x.reshape(x.size(0), -1)

        # FC layers
        x_fc = torch.relu(self.fc1(x_flat))
        self._features = x_fc

        # Policy logits
        logits = self.fc2(x_fc)

        # Check for NaN/Inf in logits before masking
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Apply action masking
        obs = input_dict["obs"]
        action_mask = obs[:, 2, :, :].reshape(-1, 64)

        # Mask invalid actions
        inf_mask = torch.where(
            action_mask > 0.5, torch.zeros_like(logits), torch.full_like(logits, -1e10)
        )
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        """Compute value function from cached features."""
        return self.value_fc(self._features).squeeze(1)


class TestOthelloCNNModel:
    """Test suite for OthelloCNN model initialization and forward pass."""

    def test_model_initialization(self):
        """Test that OthelloCNN model initializes correctly."""
        obs_space = gym.spaces.Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        action_space = gym.spaces.Discrete(64)

        model = OthelloCNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=64,
            model_config={},
            name="test_model",
        )

        assert model is not None
        assert isinstance(model, TorchModelV2)
        assert isinstance(model, nn.Module)

    def test_model_forward_pass(self):
        """Test that OthelloCNN forward pass works correctly."""
        obs_space = gym.spaces.Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        action_space = gym.spaces.Discrete(64)

        model = OthelloCNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=64,
            model_config={},
            name="test_model",
        )

        # Create dummy observations (batch_size=2)
        obs = torch.randn(2, 3, 8, 8)
        obs = torch.clamp(obs, 0, 1)  # Ensure in valid range

        input_dict = {"obs": obs}
        state = []
        seq_lens = None

        logits, state_out = model.forward(input_dict, state, seq_lens)

        # Check output shape and type
        assert logits.shape == (2, 64)
        assert isinstance(logits, torch.Tensor)
        assert state_out == state

    def test_model_value_function(self):
        """Test that OthelloCNN value function works correctly."""
        obs_space = gym.spaces.Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        action_space = gym.spaces.Discrete(64)

        model = OthelloCNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=64,
            model_config={},
            name="test_model",
        )

        # Create dummy observations
        obs = torch.randn(2, 3, 8, 8)
        obs = torch.clamp(obs, 0, 1)

        input_dict = {"obs": obs}
        model.forward(input_dict, [], None)

        # Get value function
        values = model.value_function()

        # Check output shape and type
        assert values.shape == (2,)
        assert isinstance(values, torch.Tensor)

    def test_model_action_masking(self):
        """Test that OthelloCNN applies action masking correctly."""
        obs_space = gym.spaces.Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        action_space = gym.spaces.Discrete(64)

        model = OthelloCNN(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=64,
            model_config={},
            name="test_model",
        )

        # Create observation with specific action mask
        obs = torch.ones(1, 3, 8, 8)
        obs[:, 2, :, :] = 0  # All invalid initially
        obs[0, 2, 0:2, 0:2] = 1  # Only 4 positions valid

        input_dict = {"obs": obs}
        logits, _ = model.forward(input_dict, [], None)

        # Invalid actions should have very negative logits
        # Valid actions (positions 0,1,8,9) should have higher logits
        for i in range(64):
            if i not in [0, 1, 8, 9]:
                assert logits[0, i] < -1e9, f"Invalid action {i} not masked"


class TestPPOConfiguration:
    """Test suite for PPO configuration and algorithm building."""

    def setup_method(self):
        """Set up test fixtures."""
        # Shutdown any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()

    def teardown_method(self):
        """Clean up after tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_ppo_config_creation(self):
        """Test that PPO configuration can be created."""
        config = PPOConfig()

        assert config is not None
        assert isinstance(config, PPOConfig)

    def test_ppo_config_with_environment(self):
        """Test that PPO configuration accepts environment settings."""
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
                    "start_player": "random",
                },
            )
            .framework("torch")
        )

        assert config is not None

    def test_ppo_algorithm_building(self):
        """Test that PPO algorithm can be built with Othello environment."""
        ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register custom model
        ModelCatalog.register_custom_model("othello_cnn_build", OthelloCNN)

        # Define environment creator
        def env_creator(env_config):
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env

        register_env("Othello-v0-build", env_creator)

        try:
            # Configure PPO
            config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    env="Othello-v0-build",
                    env_config={
                        "opponent": "self",
                        "reward_mode": "sparse",
                        "invalid_move_mode": "penalty",
                        "start_player": "random",
                    },
                )
                .framework("torch")
                .resources(num_gpus=0)
                .training(
                    train_batch_size=256,
                    minibatch_size=64,
                    num_sgd_iter=2,
                    lr=0.0001,
                    gamma=0.99,
                    lambda_=0.95,
                    clip_param=0.2,
                    grad_clip=0.5,
                )
                .evaluation(
                    evaluation_interval=None,
                )
                .env_runners(num_env_runners=0)
            )

            # Set custom model
            config.model = {
                "custom_model": "othello_cnn_build",
                "max_seq_len": 1,
            }

            # Build algorithm
            algo = config.build()

            assert algo is not None

            # Clean up
            algo.stop()

        finally:
            ray.shutdown()

    def test_ppo_single_training_iteration(self):
        """Test that PPO can perform a single training iteration."""
        ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register custom model
        ModelCatalog.register_custom_model("othello_cnn_train", OthelloCNN)

        # Define environment creator
        def env_creator(env_config):
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env

        register_env("Othello-v0-train", env_creator)

        try:
            # Configure PPO with minimal settings for quick testing
            config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    env="Othello-v0-train",
                    env_config={
                        "opponent": "self",
                        "reward_mode": "sparse",
                        "invalid_move_mode": "penalty",
                        "start_player": "random",
                    },
                )
                .framework("torch")
                .resources(num_gpus=0)
                .training(
                    train_batch_size=256,
                    minibatch_size=64,
                    num_sgd_iter=1,
                    lr=0.0001,
                    gamma=0.99,
                    lambda_=0.95,
                    clip_param=0.2,
                    grad_clip=0.5,
                )
                .evaluation(
                    evaluation_interval=None,
                )
                .env_runners(num_env_runners=0)
            )

            # Set custom model
            config.model = {
                "custom_model": "othello_cnn_train",
                "max_seq_len": 1,
            }

            # Build algorithm
            algo = config.build()

            # Run one training iteration
            result = algo.train()

            # Verify result contains expected metrics
            assert result is not None

            # Clean up
            algo.stop()

        finally:
            ray.shutdown()


class TestCheckpointOperations:
    """Test suite for checkpoint saving and loading."""

    def setup_method(self):
        """Set up test fixtures."""
        # Shutdown any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after tests."""
        if ray.is_initialized():
            ray.shutdown()
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created correctly."""
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        assert os.path.exists(checkpoint_dir)
        assert os.path.isdir(checkpoint_dir)

    def test_checkpoint_saving(self):
        """Test that checkpoints can be saved."""
        ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register custom model
        ModelCatalog.register_custom_model("othello_cnn_ckpt", OthelloCNN)

        # Define environment creator
        def env_creator(env_config):
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env

        register_env("Othello-v0-checkpoint", env_creator)

        try:
            # Configure PPO
            config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    env="Othello-v0-checkpoint",
                    env_config={
                        "opponent": "self",
                        "reward_mode": "sparse",
                        "invalid_move_mode": "penalty",
                        "start_player": "random",
                    },
                )
                .framework("torch")
                .resources(num_gpus=0)
                .training(
                    train_batch_size=256,
                    minibatch_size=64,
                    num_sgd_iter=1,
                )
                .evaluation(
                    evaluation_interval=None,
                )
                .env_runners(num_env_runners=0)
            )

            config.model = {
                "custom_model": "othello_cnn_ckpt",
                "max_seq_len": 1,
            }

            # Build and train algorithm
            algo = config.build()
            algo.train()

            # Save checkpoint
            checkpoint_dir = os.path.join(self.temp_dir, "iter_000001")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = algo.save(checkpoint_dir=checkpoint_dir)

            # Verify checkpoint was created
            assert checkpoint is not None
            assert os.path.exists(checkpoint_dir)

            # Clean up
            algo.stop()

        finally:
            ray.shutdown()


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        if ray.is_initialized():
            ray.shutdown()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after tests."""
        if ray.is_initialized():
            ray.shutdown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_environment_integration(self):
        """Test that environment integrates with training pipeline."""
        env = gym.make("Othello-v0")
        obs, info = env.reset()

        # Test that environment works
        assert obs.shape == (3, 8, 8)
        assert "action_mask" in info

        # Take a valid action
        valid_actions = np.where(info["action_mask"])[0]
        action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (3, 8, 8)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_training_with_multiple_iterations(self):
        """Test training loop with multiple iterations."""
        ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register custom model
        ModelCatalog.register_custom_model("othello_cnn_multi", OthelloCNN)

        # Define environment creator
        def env_creator(env_config):
            return gym.make("Othello-v0", **env_config)

        # Register environment with Ray
        from ray.tune.registry import register_env

        register_env("Othello-v0-multi", env_creator)

        try:
            # Configure PPO
            config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    env="Othello-v0-multi",
                    env_config={
                        "opponent": "self",
                        "reward_mode": "sparse",
                        "invalid_move_mode": "penalty",
                        "start_player": "random",
                    },
                )
                .framework("torch")
                .resources(num_gpus=0)
                .training(
                    train_batch_size=256,
                    minibatch_size=64,
                    num_sgd_iter=1,
                    lr=0.0001,
                )
                .evaluation(
                    evaluation_interval=None,
                )
                .env_runners(num_env_runners=0)
            )

            config.model = {
                "custom_model": "othello_cnn_multi",
                "max_seq_len": 1,
            }

            # Build algorithm
            algo = config.build()

            # Run multiple training iterations
            num_iterations = 2
            results = []

            for i in range(num_iterations):
                result = algo.train()
                results.append(result)

            # Verify we got results for each iteration
            assert len(results) == num_iterations

            # Each result should have some metrics
            for result in results:
                assert result is not None

            # Clean up
            algo.stop()

        finally:
            ray.shutdown()

    def test_training_with_reward_modes(self):
        """Test training with different reward modes."""
        ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register custom model
        ModelCatalog.register_custom_model("othello_cnn_reward", OthelloCNN)

        reward_modes = ["sparse"]

        for reward_mode in reward_modes:
            # Define environment creator
            def env_creator(env_config):
                import aip_rl.othello  # noqa: F401

                return gym.make("Othello-v0", **env_config)

            # Register environment with Ray
            from ray.tune.registry import register_env

            env_id = f"Othello-v0-{reward_mode}"
            register_env(env_id, env_creator)

            try:
                # Configure PPO
                config = (
                    PPOConfig()
                    .api_stack(
                        enable_rl_module_and_learner=False,
                        enable_env_runner_and_connector_v2=False,
                    )
                    .environment(
                        env=env_id,
                        env_config={
                            "opponent": "self",
                            "reward_mode": reward_mode,
                            "invalid_move_mode": "penalty",
                            "start_player": "random",
                        },
                    )
                    .framework("torch")
                    .resources(num_gpus=0)
                    .training(
                        train_batch_size=256,
                        minibatch_size=64,
                        num_sgd_iter=1,
                    )
                    .evaluation(
                        evaluation_interval=None,
                    )
                    .env_runners(num_env_runners=0)
                )

                config.model = {
                    "custom_model": "othello_cnn_reward",
                    "max_seq_len": 1,
                }

                # Build and train algorithm
                algo = config.build()
                result = algo.train()

                # Verify training worked
                assert result is not None

                # Clean up
                algo.stop()

            finally:
                pass

        ray.shutdown()
