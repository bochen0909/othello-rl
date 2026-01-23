"""
Othello (Reversi) environment for reinforcement learning.

This module provides a Gymnasium-compatible environment for training RL agents
to play Othello using a high-performance Rust game engine. The environment
supports various training configurations including self-play, different opponent
policies, multiple reward structures, and action masking.

Key Features:
    - High-performance Rust game engine (<1ms per step)
    - Gymnasium API compliance for RL framework integration
    - Self-play and configurable opponent policies
    - Multiple reward modes (sparse, dense, custom)
    - Action masking for invalid move handling
    - State persistence for replay analysis
    - Multiple rendering modes (human, ANSI, RGB array)
    - Ray RLlib integration with vectorized environments

Classes:
    OthelloEnv: Main Gymnasium environment class implementing the Othello game

Example:
    Basic usage with random agent:
    
    >>> import gymnasium as gym
    >>> import numpy as np
    >>> import aip_rl.othello
    >>> 
    >>> env = gym.make("Othello-v0")
    >>> observation, info = env.reset()
    >>> 
    >>> done = False
    >>> while not done:
    ...     action_mask = info["action_mask"]
    ...     valid_actions = np.where(action_mask)[0]
    ...     action = np.random.choice(valid_actions)
    ...     observation, reward, terminated, truncated, info = env.step(action)
    ...     done = terminated or truncated

    Training with Ray RLlib:
    
    >>> import ray
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> 
    >>> config = PPOConfig().environment(
    ...     env="Othello-v0",
    ...     env_config={"opponent": "self", "reward_mode": "sparse"}
    ... )
    >>> algo = config.build()
    >>> result = algo.train()

For detailed documentation, see:
    - README.md: Installation, usage examples, and training guides
    - Design document: Architecture and implementation details
    - Requirements document: Formal specifications and acceptance criteria

Author: [Your Name]
Version: 1.0.0
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Callable, Union
import othello_rust


class OthelloEnv(gym.Env):
    """
    Othello (Reversi) environment for reinforcement learning.
    
    A Gymnasium-compatible environment for training RL agents to play Othello,
    featuring a high-performance Rust game engine, flexible configuration
    options, and seamless integration with Ray RLlib.
    
    Observation Space:
        Box(0, 1, shape=(3, 8, 8), dtype=np.float32)
        - Channel 0: Agent's pieces (1 where agent has pieces, 0 otherwise)
        - Channel 1: Opponent's pieces (1 where opponent has pieces, 0 otherwise)
        - Channel 2: Valid move mask (1 for valid positions, 0 otherwise)
        
        The observation is always from the agent's perspective, regardless of
        which player (Black/White) the agent is controlling.
    
    Action Space:
        Discrete(64) - Integer actions representing board positions [0-63]
        - Mapping: action = row * 8 + col
        - Inverse: row = action // 8, col = action % 8
        - Action 0 = top-left corner (0,0)
        - Action 63 = bottom-right corner (7,7)
    
    Rewards:
        Configurable reward structure:
        - **sparse** (default): 0 during game, +1 for win, -1 for loss, 0 for draw
        - **dense**: Normalized piece differential at each step
        - **custom**: User-provided reward function
        - Invalid move penalty (configurable, default: -1.0)
    
    Args:
        opponent (Union[str, Callable], optional): Opponent policy. Options:
            - "self" (default): Self-play mode, agent plays both sides
            - "random": Random opponent selecting random valid moves
            - "greedy": Greedy opponent maximizing pieces flipped
            - callable: Custom policy function(observation) -> action
        
        reward_mode (str, optional): Reward structure. Options:
            - "sparse" (default): Terminal rewards only
            - "dense": Step-wise piece differential rewards
            - "custom": User-provided reward function
        
        reward_fn (Callable, optional): Custom reward function for "custom" mode.
            Function signature: reward_fn(game_state: Dict) -> float
            game_state contains: board, black_count, white_count, current_player,
            agent_player, game_over, pieces_flipped
        
        invalid_move_penalty (float, optional): Penalty for invalid moves.
            Default: -1.0. Only used when invalid_move_mode="penalty".
        
        invalid_move_mode (str, optional): How to handle invalid moves. Options:
            - "penalty" (default): Apply penalty and maintain state
            - "random": Automatically select random valid move
            - "error": Raise ValueError exception
        
        render_mode (str, optional): Rendering mode. Options:
            - None (default): No rendering
            - "human": Print board to console
            - "ansi": Return string representation
            - "rgb_array": Return RGB numpy array (512x512x3)
    
    Attributes:
        observation_space (spaces.Box): Observation space definition
        action_space (spaces.Discrete): Action space definition
        game (othello_rust.OthelloGame): Rust game engine instance
        agent_player (int): Agent's player color (0=Black, 1=White)
    
    Example:
        >>> import gymnasium as gym
        >>> import aip_rl.othello
        >>> 
        >>> # Create environment with default settings
        >>> env = gym.make("Othello-v0")
        >>> 
        >>> # Or with custom configuration
        >>> env = gym.make(
        ...     "Othello-v0",
        ...     opponent="greedy",
        ...     reward_mode="dense",
        ...     render_mode="human"
        ... )
        >>> 
        >>> # Run episode
        >>> observation, info = env.reset()
        >>> action_mask = info["action_mask"]
        >>> valid_actions = np.where(action_mask)[0]
        >>> action = np.random.choice(valid_actions)
        >>> observation, reward, terminated, truncated, info = env.step(action)
    
    Notes:
        - The environment automatically handles turn passing when no valid moves exist
        - Action masking is provided in the info dictionary for all steps
        - The observation is always from the agent's perspective (agent pieces in channel 0)
        - Game terminates when board is full or neither player has valid moves
        - Maximum episode length is 60 steps (configurable via Gymnasium registration)
    
    See Also:
        - README.md for detailed usage examples and training guides
        - Design document for architecture and implementation details
    """
    
    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}
    
    def __init__(
        self,
        opponent: Union[str, Callable] = "self",
        reward_mode: str = "sparse",
        reward_fn: Optional[Callable] = None,
        invalid_move_penalty: float = -1.0,
        invalid_move_mode: str = "penalty",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Othello environment.
        
        Args:
            opponent: Opponent policy configuration. Options:
                - "self": Self-play mode where agent plays both sides
                - "random": Random opponent selecting random valid moves
                - "greedy": Greedy opponent maximizing pieces flipped per move
                - callable: Custom policy function with signature:
                    policy(observation: np.ndarray) -> int
                    Must return action in range [0, 63]
            
            reward_mode: Reward structure configuration. Options:
                - "sparse": Terminal rewards only (0 during game, +1/-1/0 at end)
                - "dense": Step-wise normalized piece differential rewards
                - "custom": User-provided reward function (requires reward_fn)
            
            reward_fn: Custom reward function for "custom" reward_mode.
                Function signature: reward_fn(game_state: Dict) -> float
                game_state dictionary contains:
                    - board: np.ndarray (8, 8) with 0=empty, 1=black, 2=white
                    - black_count: int, number of black pieces
                    - white_count: int, number of white pieces
                    - current_player: int, 0=Black, 1=White
                    - agent_player: int, agent's color (0=Black, 1=White)
                    - game_over: bool, whether game has ended
                    - pieces_flipped: int, pieces flipped by last move
                Required when reward_mode="custom", ignored otherwise.
            
            invalid_move_penalty: Penalty value for invalid moves.
                Only used when invalid_move_mode="penalty".
                Typical values: -1.0 (default) to -10.0 for stronger penalties.
            
            invalid_move_mode: How to handle invalid move attempts. Options:
                - "penalty": Apply invalid_move_penalty and maintain state
                - "random": Automatically select random valid move instead
                - "error": Raise ValueError exception
            
            render_mode: Rendering mode for visualization. Options:
                - None: No rendering (default, fastest)
                - "human": Print board to console
                - "ansi": Return string representation
                - "rgb_array": Return RGB numpy array (512x512x3) for video
        
        Raises:
            ValueError: If invalid configuration parameters are provided:
                - reward_mode not in ["sparse", "dense", "custom"]
                - reward_mode="custom" but reward_fn is None
                - invalid_move_mode not in ["penalty", "random", "error"]
                - render_mode not in [None, "human", "ansi", "rgb_array"]
                - opponent not in ["self", "random", "greedy"] and not callable
        
        Example:
            >>> # Default configuration (self-play, sparse rewards)
            >>> env = OthelloEnv()
            >>> 
            >>> # Dense rewards with greedy opponent
            >>> env = OthelloEnv(opponent="greedy", reward_mode="dense")
            >>> 
            >>> # Custom reward function
            >>> def my_reward(game_state):
            ...     return game_state["black_count"] - game_state["white_count"]
            >>> env = OthelloEnv(reward_mode="custom", reward_fn=my_reward)
            >>> 
            >>> # Custom opponent policy
            >>> def my_policy(observation):
            ...     # Your policy logic here
            ...     return action
            >>> env = OthelloEnv(opponent=my_policy)
        """
        super().__init__()
        
        # Validate configuration
        if reward_mode not in ["sparse", "dense", "custom"]:
            raise ValueError(
                f"Invalid reward_mode: {reward_mode}. "
                "Must be 'sparse', 'dense', or 'custom'."
            )
        
        if reward_mode == "custom" and reward_fn is None:
            raise ValueError(
                "reward_fn must be provided when reward_mode is 'custom'"
            )
        
        if invalid_move_mode not in ["penalty", "random", "error"]:
            raise ValueError(
                f"Invalid invalid_move_mode: {invalid_move_mode}. "
                "Must be 'penalty', 'random', or 'error'."
            )
        
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Must be one of {self.metadata['render_modes']}."
            )
        
        # Validate opponent parameter
        if isinstance(opponent, str):
            if opponent not in ["self", "random", "greedy"]:
                raise ValueError(
                    f"Invalid opponent: {opponent}. "
                    "Must be 'self', 'random', 'greedy', or a callable."
                )
        elif not callable(opponent):
            raise ValueError(
                "opponent must be a string ('self', 'random', 'greedy') "
                "or a callable function."
            )
        
        # Initialize game engine
        self.game = othello_rust.OthelloGame()
        
        # Configuration
        self.opponent = opponent
        self.reward_mode = reward_mode
        self.reward_fn = reward_fn
        self.invalid_move_penalty = invalid_move_penalty
        self.invalid_move_mode = invalid_move_mode
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 8, 8), dtype=np.float32
        )
        self.action_space = spaces.Discrete(64)
        
        # Track agent's player color (0=Black, 1=White)
        self.agent_player = 0

        # Track move history for state persistence
        self._move_history = []
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Resets the game board to the standard Othello starting position with
        4 pieces in the center (2 black, 2 white). The agent always starts as
        Black (player 0).
        
        Args:
            seed: Random seed for reproducibility. Sets the random number
                generator seed for the environment. Useful for deterministic
                testing and reproducible experiments.
            
            options: Additional options dictionary (currently unused).
                Reserved for future extensions.
        
        Returns:
            observation: Initial observation as (3, 8, 8) float32 array
                - Channel 0: Agent's pieces (Black, 2 pieces in center)
                - Channel 1: Opponent's pieces (White, 2 pieces in center)
                - Channel 2: Valid moves (4 valid positions initially)
            
            info: Dictionary containing initial game state:
                - action_mask: Boolean array (64,) with 4 valid moves
                - current_player: 0 (Black, agent's turn)
                - black_count: 2 (initial black pieces)
                - white_count: 2 (initial white pieces)
                - agent_player: 0 (agent plays as Black)
        
        Example:
            >>> env = gym.make("Othello-v0")
            >>> 
            >>> # Reset with random seed for reproducibility
            >>> observation, info = env.reset(seed=42)
            >>> 
            >>> # Check initial state
            >>> print(f"Valid moves: {np.sum(info['action_mask'])}")  # 4
            >>> print(f"Black pieces: {info['black_count']}")  # 2
            >>> print(f"White pieces: {info['white_count']}")  # 2
            >>> 
            >>> # Observation shape
            >>> print(observation.shape)  # (3, 8, 8)
        
        Notes:
            - Agent always starts as Black (player 0)
            - Initial board has 4 pieces in center: Black at (3,3) and (4,4),
              White at (3,4) and (4,3)
            - Initial valid moves are at positions: (2,3), (3,2), (4,5), (5,4)
            - Move history is cleared on reset
        """
        super().reset(seed=seed)
        
        # Reset game to initial state
        self.game.reset()
        
        # Agent always plays as Black initially
        self.agent_player = 0

        # Clear move history
        self._move_history = []
        
        # Get initial observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from agent's perspective.
        
        Returns (3, 8, 8) array with:
        - Channel 0: Agent's pieces
        - Channel 1: Opponent's pieces
        - Channel 2: Valid moves
        """
        # Get board state (8, 8) with 0=empty, 1=black, 2=white
        board = self.game.get_board()
        
        # Get valid moves and reshape to (8, 8)
        valid_moves = self.game.get_valid_moves().reshape(8, 8)
        
        # Determine agent's piece value (1=Black, 2=White)
        agent_piece = self.agent_player + 1
        opponent_piece = 3 - agent_piece
        
        # Create observation channels
        agent_channel = (board == agent_piece).astype(np.float32)
        opponent_channel = (board == opponent_piece).astype(np.float32)
        valid_channel = valid_moves.astype(np.float32)
        
        # Stack channels to create (3, 8, 8) observation
        obs = np.stack([agent_channel, opponent_channel, valid_channel], axis=0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary with action mask and game state.
        
        Returns:
            Dictionary containing:
            - action_mask: Boolean array of valid moves (64,)
            - current_player: Current player (0=Black, 1=White)
            - black_count: Number of black pieces
            - white_count: Number of white pieces
            - agent_player: Agent's player color (0=Black, 1=White)
        """
        valid_moves = self.game.get_valid_moves()
        black_count, white_count = self.game.get_piece_counts()
        current_player = self.game.get_current_player()
        
        return {
            "action_mask": valid_moves,
            "current_player": current_player,
            "black_count": black_count,
            "white_count": white_count,
            "agent_player": self.agent_player,
        }

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Takes an action (board position) and applies it to the game. If the
        action is invalid, handles it according to invalid_move_mode. If
        opponent is not "self", automatically executes opponent's move after
        the agent's move.
        
        Args:
            action: Integer action in range [0, 63] representing board position.
                Mapping: action = row * 8 + col
                Example: action=0 is top-left (0,0), action=63 is bottom-right (7,7)
        
        Returns:
            observation: Current observation as (3, 8, 8) float32 array
                - Channel 0: Agent's pieces
                - Channel 1: Opponent's pieces
                - Channel 2: Valid moves
            
            reward: Reward for this step (float)
                - Sparse mode: 0 during game, +1/-1/0 at end
                - Dense mode: Normalized piece differential
                - Custom mode: Value from reward_fn
                - Invalid move: invalid_move_penalty (if mode="penalty")
            
            terminated: Whether episode has ended (bool)
                True when game is over (board full or no valid moves for both players)
            
            truncated: Whether episode was truncated (bool)
                Always False for this environment (handled by Gymnasium wrapper)
            
            info: Dictionary containing:
                - action_mask: Boolean array (64,) of valid moves
                - current_player: Current player (0=Black, 1=White)
                - black_count: Number of black pieces
                - white_count: Number of white pieces
                - agent_player: Agent's color (0=Black, 1=White)
        
        Raises:
            ValueError: If action is invalid and invalid_move_mode="error"
        
        Example:
            >>> env = gym.make("Othello-v0")
            >>> observation, info = env.reset()
            >>> 
            >>> # Get valid actions from action mask
            >>> action_mask = info["action_mask"]
            >>> valid_actions = np.where(action_mask)[0]
            >>> 
            >>> # Select and execute action
            >>> action = np.random.choice(valid_actions)
            >>> obs, reward, terminated, truncated, info = env.step(action)
            >>> 
            >>> print(f"Reward: {reward}, Game over: {terminated}")
        
        Notes:
            - If opponent != "self", opponent's move is executed automatically
            - Observation is always from agent's perspective
            - Turn passing is handled automatically when no valid moves exist
            - Action mask in info dictionary indicates valid moves for next step
        """
        # Get valid moves
        valid_moves = self.game.get_valid_moves()
        
        # Check if action is valid
        if not valid_moves[action]:
            # Handle invalid move according to policy
            if self.invalid_move_mode == "error":
                raise ValueError(f"Invalid move: {action}")
            elif self.invalid_move_mode == "random":
                # Select random valid move
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    # No valid moves available - game should be over
                    obs = self._get_observation()
                    info = self._get_info()
                    reward = self._calculate_reward(0, True)
                    return obs, reward, True, False, info
            else:  # penalty mode
                # Apply penalty and return current state
                reward = self.invalid_move_penalty
                obs = self._get_observation()
                info = self._get_info()
                return obs, reward, False, False, info
        
        # Apply the move
        valid, pieces_flipped, game_over = self.game.step(action)

        # Record move in history
        self._move_history.append(action)
        
        # Calculate reward for agent's move
        reward = self._calculate_reward(pieces_flipped, game_over)
        
        # If self-play mode and game not over, execute opponent move
        if self.opponent != "self" and not game_over:
            # Check if opponent has valid moves
            valid_moves = self.game.get_valid_moves()
            if np.any(valid_moves):
                # Execute opponent's move
                self._execute_opponent_move()
                
                # Check if game is over after opponent's move
                game_over = self.game.get_winner() != 3
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        # Set termination flags
        terminated = game_over
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(
        self, pieces_flipped: int, game_over: bool
    ) -> float:
        """
        Calculate reward based on reward mode.
        
        Args:
            pieces_flipped: Number of pieces flipped by the move
            game_over: Whether the game is over
        
        Returns:
            Reward value as float
        """
        if self.reward_mode == "sparse":
            # Sparse rewards: 0 during game, +1/-1/0 at end
            if not game_over:
                return 0.0
            
            winner = self.game.get_winner()
            if winner == 2:  # Draw
                return 0.0
            elif winner == self.agent_player:
                return 1.0
            else:
                return -1.0
        
        elif self.reward_mode == "dense":
            # Dense rewards: normalized piece differential
            black_count, white_count = self.game.get_piece_counts()
            if self.agent_player == 0:  # Agent is Black
                return (black_count - white_count) / 64.0
            else:  # Agent is White
                return (white_count - black_count) / 64.0
        
        elif self.reward_mode == "custom":
            # Custom reward function
            if self.reward_fn is None:
                raise ValueError("reward_fn is None but reward_mode is 'custom'")
            
            # Create game state dict for custom function
            black_count, white_count = self.game.get_piece_counts()
            current_player = self.game.get_current_player()
            board = self.game.get_board()
            
            game_state = {
                "board": board,
                "black_count": black_count,
                "white_count": white_count,
                "current_player": current_player,
                "agent_player": self.agent_player,
                "game_over": game_over,
                "pieces_flipped": pieces_flipped,
            }
            
            return float(self.reward_fn(game_state))
        
        else:
            # Should not reach here due to validation in __init__
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _execute_opponent_move(self) -> None:
        """
        Execute opponent's move based on opponent policy.
        
        Supports:
        - "random": Select random valid move
        - "greedy": Select move that flips most pieces
        - callable: Custom policy function that takes observation and returns action
        
        If no valid moves exist, the turn is automatically passed.
        """
        # Get valid moves for current player (opponent)
        valid_moves = self.game.get_valid_moves()
        
        # If no valid moves, turn will be passed automatically by game engine
        if not np.any(valid_moves):
            return
        
        # Select action based on opponent policy
        if self.opponent == "random":
            # Random policy: select random valid move
            valid_indices = np.where(valid_moves)[0]
            action = np.random.choice(valid_indices)
        
        elif self.opponent == "greedy":
            # Greedy policy: select move that flips most pieces
            action = self._get_greedy_move()
        
        elif callable(self.opponent):
            # Custom policy: call function with observation
            obs = self._get_observation()
            action = self.opponent(obs)
            
            # Validate that callable returned a valid move
            if not valid_moves[action]:
                # Fall back to random if callable returns invalid move
                valid_indices = np.where(valid_moves)[0]
                action = np.random.choice(valid_indices)
        
        else:
            # Should not reach here if opponent is "self"
            # or if validation in __init__ worked correctly
            raise ValueError(f"Unknown opponent type: {self.opponent}")
        
        # Execute the opponent's move
        self.game.step(action)
    
    def _get_greedy_move(self) -> int:
        """
        Get move that flips the most pieces (greedy policy).
        
        Returns:
            Action (0-63) that flips the maximum number of pieces
        """
        valid_moves = self.game.get_valid_moves()
        best_action = -1
        best_flips = -1
        
        # Try each valid move and count flips
        for action in np.where(valid_moves)[0]:
            # Create a copy of the game to simulate the move
            game_copy = othello_rust.OthelloGame()
            
            # Copy current game state
            current_board = self.game.get_board()
            current_player = self.game.get_current_player()
            
            # Set up the copy
            game_copy.reset()
            # Manually set board state by replaying to similar position
            # Since we don't have a direct set_board method, we'll use a workaround
            # by counting flips directly from the current game
            
            # Actually, let's count flips by checking the move directly
            # We'll simulate by checking what would flip
            row, col = action // 8, action % 8
            flips = self._count_flips_for_move(row, col)
            
            if flips > best_flips:
                best_flips = flips
                best_action = action
        
        return best_action
    
    def _count_flips_for_move(self, row: int, col: int) -> int:
        """
        Count how many pieces would be flipped by a move at (row, col).
        
        Args:
            row: Row index (0-7)
            col: Column index (0-7)
        
        Returns:
            Number of pieces that would be flipped
        """
        board = self.game.get_board()
        current_player = self.game.get_current_player()
        player_piece = current_player + 1
        opponent_piece = 3 - player_piece
        
        total_flips = 0
        
        # Check all 8 directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        
        for dr, dc in directions:
            flips_in_direction = 0
            r, c = row + dr, col + dc
            
            # Count opponent pieces in this direction
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r, c] == opponent_piece:
                    flips_in_direction += 1
                    r += dr
                    c += dc
                elif board[r, c] == player_piece:
                    # Found our piece, these flips are valid
                    total_flips += flips_in_direction
                    break
                else:
                    # Empty cell, no flips in this direction
                    break
        
        return total_flips

    def render(self):
        """
        Render the game state according to the configured render mode.
        
        The rendering behavior depends on the render_mode set during
        initialization:
        
        - "human": Prints board to console (returns None)
        - "ansi": Returns string representation of board
        - "rgb_array": Returns RGB numpy array for video recording
        - None: No rendering (returns None)
        
        Returns:
            None: If render_mode is "human" or None
            str: If render_mode is "ansi", returns board as string with:
                - Board grid with pieces (● for black, ○ for white, . for empty)
                - Valid moves marked with *
                - Piece counts for both players
                - Current player indicator
            np.ndarray: If render_mode is "rgb_array", returns (512, 512, 3)
                uint8 array with:
                - Green board background
                - Black and white circles for pieces
                - Yellow circles for valid move markers
        
        Example:
            >>> # Human rendering (prints to console)
            >>> env = gym.make("Othello-v0", render_mode="human")
            >>> observation, info = env.reset()
            >>> env.render()
            #   0 1 2 3 4 5 6 7
            # 0 . . . . . . . .
            # 1 . . . . . . . .
            # 2 . . . * . . . .
            # 3 . . * ○ ● . . .
            # 4 . . . ● ○ * . .
            # 5 . . . . * . . .
            # 6 . . . . . . . .
            # 7 . . . . . . . .
            # 
            # ● Black: 2  ○ White: 2
            # Current player: Black
            
            >>> # ANSI rendering (returns string)
            >>> env = gym.make("Othello-v0", render_mode="ansi")
            >>> board_str = env.render()
            >>> print(board_str)
            
            >>> # RGB array rendering (for video)
            >>> env = gym.make("Othello-v0", render_mode="rgb_array")
            >>> rgb_frame = env.render()
            >>> print(rgb_frame.shape)  # (512, 512, 3)
            >>> print(rgb_frame.dtype)  # uint8
        
        Notes:
            - render_mode must be set during environment initialization
            - RGB array is 512x512 pixels with 64x64 pixels per cell
            - Valid moves are shown as * in ANSI mode, yellow circles in RGB mode
            - Rendering has minimal performance impact (<1ms typically)
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
        else:
            return None

    def _render_ansi(self) -> str:
        """
        Render board as ANSI string representation.

        Returns:
            String representation of the board with:
            - Board grid with pieces (● for black, ○ for white, . for empty)
            - Valid moves marked with *
            - Piece counts for both players
            - Current player indicator
        """
        board = self.game.get_board()
        valid_moves = self.game.get_valid_moves().reshape(8, 8)
        black_count, white_count = self.game.get_piece_counts()
        current_player = self.game.get_current_player()

        # Symbol mapping: 0=empty, 1=black, 2=white
        symbols = {0: ".", 1: "●", 2: "○"}

        lines = []
        lines.append("  0 1 2 3 4 5 6 7")

        for i in range(8):
            row = f"{i} "
            for j in range(8):
                if board[i, j] == 0 and valid_moves[i, j]:
                    row += "* "  # Valid move marker
                else:
                    row += symbols[board[i, j]] + " "
            lines.append(row)

        lines.append("")
        lines.append(f"● Black: {black_count}  ○ White: {white_count}")
        player_name = "Black" if current_player == 0 else "White"
        lines.append(f"Current player: {player_name}")

        return "\n".join(lines)

    def _render_rgb(self) -> np.ndarray:
        """
        Render board as RGB array suitable for video recording.

        Returns:
            RGB array with shape (512, 512, 3) and dtype uint8
            - Green background for board
            - Black and white circles for pieces
            - Yellow circles for valid move markers
        """
        # Create 512x512 RGB image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cell_size = 64

        board = self.game.get_board()
        valid_moves = self.game.get_valid_moves().reshape(8, 8)

        # Draw board
        for i in range(8):
            for j in range(8):
                x, y = j * cell_size, i * cell_size

                # Board color (alternating green shades)
                if (i + j) % 2 == 0:
                    color = (0, 128, 0)
                else:
                    color = (0, 100, 0)
                img[y:y+cell_size, x:x+cell_size] = color

                # Draw piece or valid move marker
                center_x = x + cell_size // 2
                center_y = y + cell_size // 2

                if board[i, j] == 1:  # Black piece
                    self._draw_circle(img, center_x, center_y,
                                    cell_size // 3, (0, 0, 0))
                elif board[i, j] == 2:  # White piece
                    self._draw_circle(img, center_x, center_y,
                                    cell_size // 3, (255, 255, 255))
                elif valid_moves[i, j]:  # Valid move marker
                    self._draw_circle(img, center_x, center_y,
                                    cell_size // 6, (255, 255, 0))

        return img

    def _draw_circle(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Draw a filled circle on the image.

        Args:
            img: Image array to draw on
            cx: Circle center x coordinate
            cy: Circle center y coordinate
            radius: Circle radius
            color: RGB color tuple
        """
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        img[mask] = color

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current game state to a dictionary.
        
        Captures the complete game state including board configuration, piece
        counts, current player, and move history. The saved state can be
        restored later using load_state() for replay analysis or debugging.
        
        Returns:
            Dictionary containing complete game state:
                - board: Board state as numpy array (8, 8) with dtype uint8
                    Values: 0=empty, 1=black, 2=white
                - current_player: Current player (0=Black, 1=White)
                - black_count: Number of black pieces on board
                - white_count: Number of white pieces on board
                - agent_player: Agent's player color (0=Black, 1=White)
                - game_over: Whether the game has ended
                - winner: Winner status (0=Black, 1=White, 2=Draw, 3=NotFinished)
                - move_history: List of actions played to reach this state
        
        Example:
            >>> env = gym.make("Othello-v0")
            >>> observation, info = env.reset()
            >>> 
            >>> # Play some moves
            >>> for _ in range(10):
            ...     action_mask = info["action_mask"]
            ...     action = np.random.choice(np.where(action_mask)[0])
            ...     observation, reward, terminated, truncated, info = env.step(action)
            ...     if terminated:
            ...         break
            >>> 
            >>> # Save current state
            >>> state = env.save_state()
            >>> print(f"Moves played: {len(state['move_history'])}")
            >>> print(f"Black: {state['black_count']}, White: {state['white_count']}")
            >>> 
            >>> # Continue playing or analyze state
            >>> board = state['board']
            >>> print(f"Board shape: {board.shape}")  # (8, 8)
        
        Notes:
            - The returned dictionary contains copies of arrays (safe to modify)
            - Move history allows state reconstruction via load_state()
            - State can be serialized to JSON or pickle for persistence
            - Useful for debugging, replay analysis, and checkpointing
        
        See Also:
            load_state: Restore a saved game state
        """
        board = self.game.get_board()
        black_count, white_count = self.game.get_piece_counts()
        current_player = self.game.get_current_player()
        winner = self.game.get_winner()

        return {
            "board": board.copy(),
            "current_player": int(current_player),
            "black_count": int(black_count),
            "white_count": int(white_count),
            "agent_player": int(self.agent_player),
            "game_over": winner != 3,
            "winner": int(winner),
            "move_history": self._move_history.copy(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a game state from a dictionary.

        Reconstructs the game state by replaying the move history from the
        initial state. This ensures the game engine's internal state is
        consistent with the saved state.
        
        Args:
            state: Dictionary containing state information (from save_state).
                Required fields:
                - board: numpy array (8, 8) with 0=empty, 1=black, 2=white
                - current_player: int (0=Black, 1=White)
                - black_count: int, number of black pieces
                - white_count: int, number of white pieces
                - agent_player: int (0=Black, 1=White)
                - game_over: bool, whether game has ended
                - winner: int (0=Black, 1=White, 2=Draw, 3=NotFinished)
                - move_history: list of int actions

        Raises:
            ValueError: If state dictionary is invalid or incomplete:
                - Missing required fields
                - board is not numpy array or wrong shape
                - move_history is not a list
                - Invalid move in move history (cannot reconstruct state)
                - Reconstructed board doesn't match saved board
        
        Example:
            >>> env = gym.make("Othello-v0")
            >>> observation, info = env.reset()
            >>> 
            >>> # Play and save state
            >>> for _ in range(5):
            ...     action_mask = info["action_mask"]
            ...     action = np.random.choice(np.where(action_mask)[0])
            ...     observation, reward, terminated, truncated, info = env.step(action)
            >>> 
            >>> saved_state = env.save_state()
            >>> 
            >>> # Reset and play different moves
            >>> observation, info = env.reset()
            >>> for _ in range(10):
            ...     action_mask = info["action_mask"]
            ...     action = np.random.choice(np.where(action_mask)[0])
            ...     observation, reward, terminated, truncated, info = env.step(action)
            >>> 
            >>> # Restore previous state
            >>> env.load_state(saved_state)
            >>> 
            >>> # Verify state was restored
            >>> current_state = env.save_state()
            >>> assert np.array_equal(current_state['board'], saved_state['board'])
            >>> assert len(current_state['move_history']) == len(saved_state['move_history'])
        
        Notes:
            - State is reconstructed by replaying move history from initial state
            - This ensures internal game engine consistency
            - Validation checks ensure reconstructed state matches saved state
            - Move history must contain only valid moves
            - Useful for debugging, replay analysis, and state restoration
        
        See Also:
            save_state: Save the current game state
        """
        # Validate state dictionary
        required_fields = [
            "board", "current_player", "black_count", "white_count",
            "agent_player", "game_over", "winner", "move_history"
        ]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field in state: {field}")

        # Validate board shape
        board = state["board"]
        if not isinstance(board, np.ndarray):
            raise ValueError("board must be a numpy array")
        if board.shape != (8, 8):
            raise ValueError(f"board must have shape (8, 8), got {board.shape}")

        # Validate move_history
        move_history = state["move_history"]
        if not isinstance(move_history, list):
            raise ValueError("move_history must be a list")

        # Reset to initial state
        self.game.reset()
        self.agent_player = state["agent_player"]
        self._move_history = []

        # Replay move history to reconstruct state
        for action in move_history:
            valid, _, game_over = self.game.step(action)
            self._move_history.append(action)

            if not valid:
                raise ValueError(
                    f"Invalid move {action} in move history. "
                    "Cannot reconstruct state."
                )

            if game_over:
                break

        # Verify the reconstructed state matches the saved state
        reconstructed_board = self.game.get_board()
        if not np.array_equal(reconstructed_board, board):
            raise ValueError(
                "Reconstructed board state does not match saved state. "
                "Move history may be corrupted."
            )
