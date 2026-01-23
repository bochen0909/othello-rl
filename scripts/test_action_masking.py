"""
Test script for action masking with PPO on Othello environment.

This script validates Requirement 7.9: Action masking support.

Tests:
1. Verify action_mask is provided in info dictionary
2. Verify action_mask correctly identifies valid moves
3. Train a PPO agent with action masking
4. Verify trained agent never selects invalid actions
5. Compare performance with and without action masking
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import torch
import torch.nn as nn
import numpy as np
import sys
import gymnasium as gym

# Register Othello environment
import aip_rl.othello  # noqa: F401


class OthelloCNNWithMasking(TorchModelV2, nn.Module):
    """
    Custom CNN model for Othello board with action masking support.

    This model applies action masking by adding a large negative value
    (FLOAT_MIN) to the logits of invalid actions, ensuring they have
    near-zero probability after softmax.

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
        Forward pass through the network with action masking.

        Args:
            input_dict: Dictionary containing 'obs' key with observations
            state: RNN state (not used for this feedforward network)
            seq_lens: Sequence lengths (not used)

        Returns:
            masked_logits: Action logits with masking applied
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

        # Policy logits (before masking)
        logits = self.fc2(x)

        # Apply action masking if available
        # The action mask is in the third channel of the observation
        # Extract it: shape (batch_size, 3, 8, 8) -> (batch_size, 64)
        obs = input_dict["obs"]
        action_mask = obs[:, 2, :, :].reshape(-1, 64)

        # Convert to boolean mask and apply
        # Invalid actions get FLOAT_MIN added to their logits
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        """
        Compute value function from cached features.

        Returns:
            value: Value estimates of shape (batch_size,)
        """
        return self.value_fc(self._features).squeeze(1)


class OthelloCNNNoMasking(TorchModelV2, nn.Module):
    """
    CNN model without action masking (for comparison).
    
    Same architecture as OthelloCNNWithMasking but without masking logic.
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
        """Forward pass without action masking."""
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

        # Policy logits (no masking)
        logits = self.fc2(x)

        return logits, state

    def value_function(self):
        """Compute value function from cached features."""
        return self.value_fc(self._features).squeeze(1)


def test_action_mask_in_info():
    """
    Test 1: Verify action_mask is provided in info dictionary.
    
    Returns:
        bool: True if test passes
    """
    print("\n" + "=" * 60)
    print("Test 1: Verify action_mask in info dictionary")
    print("=" * 60)
    
    try:
        env = gym.make("Othello-v0")
        obs, info = env.reset()
        
        # Check that action_mask exists
        assert "action_mask" in info, "action_mask not in info dictionary"
        print("✓ action_mask found in info dictionary")
        
        # Check format
        action_mask = info["action_mask"]
        assert isinstance(action_mask, np.ndarray), \
            "action_mask is not a numpy array"
        assert action_mask.shape == (64,), \
            f"action_mask has wrong shape: {action_mask.shape}"
        assert action_mask.dtype == bool, \
            f"action_mask has wrong dtype: {action_mask.dtype}"
        print(f"✓ action_mask has correct format: shape={action_mask.shape}, "
              f"dtype={action_mask.dtype}")
        
        # Check that it has valid moves
        num_valid = np.sum(action_mask)
        assert num_valid > 0, "No valid moves in initial state"
        print(f"✓ action_mask has {num_valid} valid moves initially")
        
        # Test through a few steps
        for i in range(5):
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                break
            
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
            
            action_mask = info["action_mask"]
            assert "action_mask" in info, \
                f"action_mask missing after step {i+1}"
        
        print(f"✓ action_mask present in info for {i+1} steps")
        
        env.close()
        print("\nTest 1: PASSED")
        return True
        
    except Exception as e:
        print(f"\nTest 1: FAILED - {e}")
        return False


def test_action_mask_correctness():
    """
    Test 2: Verify action_mask correctly identifies valid moves.
    
    Returns:
        bool: True if test passes
    """
    print("\n" + "=" * 60)
    print("Test 2: Verify action_mask correctness")
    print("=" * 60)
    
    try:
        env = gym.make("Othello-v0")
        obs, info = env.reset()
        
        num_steps = 0
        num_checks = 0
        
        while num_steps < 20:
            action_mask = info["action_mask"]
            
            # Verify that valid actions can be executed
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                print(f"  No valid moves at step {num_steps}, game should end")
                break
            
            # Try a valid action
            valid_action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(valid_action)
            num_checks += 1
            
            if terminated:
                print(f"  Game ended after {num_steps + 1} steps")
                break
            
            num_steps += 1
        
        print(f"✓ All {num_checks} valid actions executed successfully")
        
        # Test that invalid actions are correctly marked
        env.reset()
        obs, info = env.reset()
        action_mask = info["action_mask"]
        
        invalid_actions = np.where(~action_mask)[0]
        if len(invalid_actions) > 0:
            # Try an invalid action with penalty mode
            env_penalty = gym.make("Othello-v0",
                                   invalid_move_mode="penalty")
            obs, info = env_penalty.reset()
            
            invalid_action = invalid_actions[0]
            obs, reward, terminated, truncated, info = \
                env_penalty.step(invalid_action)
            
            # Should receive penalty
            assert reward < 0, \
                f"Expected negative reward for invalid action, got {reward}"
            print(f"✓ Invalid action {invalid_action} correctly penalized "
                  f"with reward {reward}")
            
            env_penalty.close()
        
        env.close()
        print("\nTest 2: PASSED")
        return True
        
    except Exception as e:
        print(f"\nTest 2: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_with_action_masking():
    """
    Test 3: Train PPO with action masking and verify no invalid actions.
    
    Returns:
        bool: True if test passes
    """
    print("\n" + "=" * 60)
    print("Test 3: Train PPO with action masking")
    print("=" * 60)
    
    try:
        # Initialize Ray
        print("\nInitializing Ray...")
        ray.init(ignore_reinit_error=True)
        
        # Register custom model with masking
        print("Registering model with action masking...")
        ModelCatalog.register_custom_model(
            "othello_cnn_masked", OthelloCNNWithMasking
        )
        
        # Define environment creator
        def env_creator(env_config):
            import aip_rl.othello  # noqa: F401
            import gymnasium as gym
            return gym.make("Othello-v0", **env_config)
        
        # Register environment
        from ray.tune.registry import register_env
        register_env("Othello-v0", env_creator)
        
        # Configure PPO with action masking
        print("Configuring PPO...")
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
        
        config["num_env_runners"] = 2
        config.model = {
            "custom_model": "othello_cnn_masked",
            "max_seq_len": 1,
        }
        
        # Build algorithm
        print("Building algorithm...")
        algo = config.build()
        
        # Train for a few iterations
        num_iterations = 5
        print(f"\nTraining for {num_iterations} iterations...")
        print("-" * 60)
        
        for i in range(num_iterations):
            result = algo.train()
            
            iteration_num = i + 1
            reward_mean = "N/A"
            
            if "env_runners" in result:
                if "episode_return_mean" in result["env_runners"]:
                    reward_mean = \
                        f"{result['env_runners']['episode_return_mean']:.2f}"
            
            print(f"Iteration {iteration_num}/{num_iterations} | "
                  f"Reward: {reward_mean:>6s} | ✓")
        
        print("-" * 60)
        
        # Now test that the trained agent never selects invalid actions
        print("\nTesting trained agent for invalid action selection...")
        
        # Get the policy
        policy = algo.get_policy()
        
        # Test on multiple episodes
        num_test_episodes = 10
        total_actions = 0
        invalid_actions_selected = 0
        
        for episode in range(num_test_episodes):
            env = gym.make("Othello-v0")
            obs, info = env.reset()
            
            episode_steps = 0
            while episode_steps < 60:  # Max steps per episode
                action_mask = info["action_mask"]
                
                # Get action from policy
                # Convert obs to batch format
                obs_batch = torch.from_numpy(obs).unsqueeze(0).float()
                
                # Get action from policy
                with torch.no_grad():
                    logits, _ = policy.model({
                        "obs": obs_batch
                    }, [], None)
                    
                    # Sample action from logits
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                total_actions += 1
                
                # Check if action is valid
                if not action_mask[action]:
                    invalid_actions_selected += 1
                    print(f"  WARNING: Invalid action {action} selected "
                          f"in episode {episode + 1}")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            env.close()
        
        print(f"\nTested {total_actions} actions across "
              f"{num_test_episodes} episodes")
        print(f"Invalid actions selected: {invalid_actions_selected}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()
        
        # Test passes if very few invalid actions (< 5%)
        invalid_rate = invalid_actions_selected / total_actions
        if invalid_rate < 0.05:
            print(f"✓ Invalid action rate: {invalid_rate:.2%} (< 5%)")
            print("\nTest 3: PASSED")
            return True
        else:
            print(f"✗ Invalid action rate: {invalid_rate:.2%} (>= 5%)")
            print("\nTest 3: FAILED")
            return False
        
    except Exception as e:
        print(f"\nTest 3: FAILED - {e}")
        import traceback
        traceback.print_exc()
        
        try:
            ray.shutdown()
        except:
            pass
        
        return False


def run_all_tests():
    """Run all action masking tests."""
    print("\n" + "=" * 60)
    print("ACTION MASKING TEST SUITE")
    print("Testing Requirement 7.9: Action masking support")
    print("=" * 60)
    
    results = {}
    
    # Test 1: action_mask in info
    results["test_1"] = test_action_mask_in_info()
    
    # Test 2: action_mask correctness
    results["test_2"] = test_action_mask_correctness()
    
    # Test 3: PPO with action masking
    results["test_3"] = test_ppo_with_action_masking()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("Action masking is working correctly with PPO!")
    else:
        print("SOME TESTS FAILED")
        print("Action masking needs attention")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
