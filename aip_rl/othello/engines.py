"""External Othello Engine Opponents

This module provides integration for external Othello engines that can be used
as training opponents. Each engine is a high-performance AI player with different
strategies and evaluation functions.

Available engines:
- aelskels: Alpha-beta pruning AI with 5-turn lookahead
- drohh: Standard Othello with minimax and strategic evaluation
- nealetham: Naive greedy AI that maximizes immediate piece capture
"""

from typing import Callable, Dict, Optional
import numpy as np

try:
    import othello_rust
except ImportError:
    raise ImportError(
        "Failed to import othello_rust module. "
        "Please ensure the Rust bindings are built and installed: "
        "maturin develop --release --manifest-path rust/othello/Cargo.toml"
    )


# Define the engine registry mapping engine names to their move functions
ENGINE_REGISTRY: Dict[str, Callable] = {
    "aelskels": othello_rust.compute_move_aelskels_py,
    "drohh": othello_rust.compute_move_drohh_py,
    "nealetham": othello_rust.compute_move_nealetham_py,
}


def get_available_engines() -> list:
    """Get list of all available engine names.

    Returns:
        list: Names of all available engines
    """
    return list(ENGINE_REGISTRY.keys())


def get_engine_opponent(engine_name: str) -> Callable:
    """Factory function to create an engine opponent callable.

    Args:
        engine_name (str): Name of the engine (must be in ENGINE_REGISTRY)

    Returns:
        Callable: A function that accepts observation and returns action

    Raises:
        ValueError: If engine_name is not recognized
    """
    if engine_name not in ENGINE_REGISTRY:
        available = ", ".join(get_available_engines())
        raise ValueError(
            f"Unknown engine '{engine_name}'. Available engines: {available}"
        )

    engine_func = ENGINE_REGISTRY[engine_name]

    def engine_opponent(observation: np.ndarray) -> int:
        """Engine opponent callable.

        Args:
            observation (np.ndarray): Board observation from environment
                - Shape (3, 8, 8): 3-channel observation with agent pieces, opponent pieces, valid moves
                - Or shape (64,) or (8, 8): Raw board state (0=Empty, 1=Black, 2=White)

        Returns:
            int: Action index (0-63) for the computed move

        Raises:
            ValueError: If observation shape is invalid or no valid moves available
        """
        # Reconstruct board from multi-channel observation or use raw board
        if observation.ndim == 3 and observation.shape == (3, 8, 8):
            # Multi-channel observation from environment
            # Channel 0: Agent's pieces, Channel 1: Opponent's pieces, Channel 2: Valid moves
            agent_channel = observation[0]
            opponent_channel = observation[1]

            # Reconstruct board: 0=Empty, 1=Agent pieces, 2=Opponent pieces
            board = np.zeros((8, 8), dtype=np.uint8)
            board[agent_channel.astype(bool)] = 1  # Agent pieces are player 1
            board[opponent_channel.astype(bool)] = 2  # Opponent pieces are player 2

            board = board.flatten().astype(np.uint8)
        else:
            # Assume it's a raw board state
            if observation.ndim == 2 and observation.shape == (8, 8):
                board = observation.flatten().astype(np.uint8)
            elif observation.ndim == 1 and len(observation) == 64:
                board = observation.astype(np.uint8)
            else:
                raise ValueError(
                    f"Invalid observation shape. Expected (3, 8, 8), (8, 8), or (64,), "
                    f"got {observation.shape}"
                )

        # Ensure it's a list of 64 elements
        if len(board) != 64:
            raise ValueError(f"Board must have 64 cells, got {len(board)}")

        # Call the engine to compute move
        # Player is 1 since we placed agent pieces as 1 and opponent pieces as 2
        move = engine_func(list(board), 1)

        # Check if no valid moves (u8::MAX = 255)
        if move == 255:
            # Fall back to random move if engine returns no valid moves
            # This can happen if board representation is incorrect
            from aip_rl.othello.env import OthelloEnv

            # Get valid moves from the observation's valid moves channel
            if observation.ndim == 3 and observation.shape == (3, 8, 8):
                valid_moves = observation[2].flatten().astype(bool)
                valid_actions = np.where(valid_moves)[0]
                if len(valid_actions) > 0:
                    # Return first valid action as fallback
                    return int(valid_actions[0])

            raise ValueError(
                "Engine returned no valid moves (u8::MAX). "
                "This should not happen during normal gameplay."
            )

        return int(move)

    # Add metadata to the callable
    engine_opponent.__name__ = f"engine_{engine_name}"
    engine_opponent.__doc__ = f"Engine opponent using {engine_name} algorithm"

    return engine_opponent
