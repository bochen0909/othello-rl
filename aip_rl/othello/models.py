"""
Neural network models for Othello RL training.

This module provides custom PyTorch models compatible with Ray RLlib
for training agents to play Othello.
"""

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class OthelloCNN(TorchModelV2, nn.Module):
    """
    Lightweight CNN model for Othello board with residual connections.

    Architecture:
    - Input: (3, 8, 8) - 3 channels (agent pieces, opponent pieces,
      valid moves)
    - Conv1: 3 -> 64 channels, 3x3 kernel, padding=1 + BatchNorm + ReLU
    - ResBlock1: 64 -> 64 channels (2 conv layers with skip connection)
    - Flatten: 64 * 8 * 8 = 4096 features
    - FC1: 4096 -> 256
    - FC2 (policy): 256 -> 64 (action logits)
    - Value FC: 256 -> 1 (value function)

    Total parameters: ~1.5M (4x lighter, faster inference)
    Inference time: ~2-3ms per step (vs 9ms for large model)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Single residual block (lightweight)
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_outputs)

        # Value function head
        self.value_fc = nn.Linear(256, 1)

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
        """
        Forward pass through the network with action masking support.

        Args:
            input_dict: Dictionary containing 'obs' key with observations
                       Channel 2 of obs contains the action mask
            state: RNN state (not used for this feedforward network)
            seq_lens: Sequence lengths (not used for this feedforward network)

        Returns:
            logits: Action logits of shape (batch_size, num_outputs)
                   with large negative values for invalid actions
            state: Unchanged state
        """
        x = input_dict["obs"].float()

        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN/Inf in input observation!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # Initial convolution
        x = torch.relu(self.bn1(self.conv1(x)))

        # Single residual block (lightweight)
        x = self._residual_block(
            x, self.res1_conv1, self.res1_bn1, self.res1_conv2, self.res1_bn2
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
            print(f"WARNING: NaN/Inf in logits before masking!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Apply action masking
        # Extract action mask from channel 2 of observation (shape: batch, 3, 8, 8)
        obs = input_dict["obs"]
        action_mask = obs[:, 2, :, :].reshape(-1, 64)

        # Mask invalid actions by adding large negative value to their logits
        # Use torch.where to avoid log(0) = -inf issues
        inf_mask = torch.where(
            action_mask > 0.5,
            torch.zeros_like(logits),
            torch.full_like(logits, -1e10),
        )
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        """
        Compute value function from cached features.

        Returns:
            value: Value estimates of shape (batch_size,)
        """
        return self.value_fc(self._features).squeeze(1)
