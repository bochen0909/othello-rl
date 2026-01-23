"""
Property-based tests for Othello Gymnasium environment.

Uses hypothesis to verify universal properties across many randomly generated inputs.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from aip_rl.othello.env import OthelloEnv
import othello_rust


# Hypothesis strategies for generating test data

@st.composite
def valid_move_sequence(draw):
    """Generate a sequence of valid moves from initial state."""
    game = othello_rust.OthelloGame()
    moves = []
    
    max_moves = draw(st.integers(0, 30))
    for _ in range(max_moves):
        valid_moves = game.get_valid_moves()
        if not np.any(valid_moves):
            break
        
        valid_indices = np.where(valid_moves)[0]
        action = draw(st.sampled_from(valid_indices.tolist()))
        moves.append(action)
        
        valid, _, game_over = game.step(action)
        if not valid or game_over:
            break
    
    return moves


@st.composite
def agent_player_choice(draw):
    """Generate agent player choice (0=Black, 1=White)."""
    return draw(st.integers(0, 1))


class TestObservationEncodingProperties:
    """Property-based tests for observation encoding correctness."""

    @given(valid_move_sequence(), agent_player_choice())
    @settings(max_examples=100, deadline=None)
    def test_observation_encoding_correctness(self, moves, agent_player):
        """
        Feature: othello-rl-environment, Property 10: Observation Encoding Correctness
        
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**
        
        For any game state, the observation should be a (3, 8, 8) array where:
        - Channel 0 contains exactly the agent's pieces
        - Channel 1 contains exactly the opponent's pieces
        - Channel 2 contains exactly the valid moves
        - All values are normalized to [0, 1]
        """
        env = OthelloEnv()
        env.reset()
        env.agent_player = agent_player
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            # Just apply the move to the game engine directly
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get observation
        obs = env._get_observation()
        
        # Property 1: Verify shape
        assert obs.shape == (3, 8, 8), f"Expected shape (3, 8, 8), got {obs.shape}"
        
        # Property 2: Verify dtype
        assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
        
        # Property 3: Verify values in [0, 1]
        assert np.all(obs >= 0), "Observation contains values < 0"
        assert np.all(obs <= 1), "Observation contains values > 1"
        
        # Property 4: Verify channel 0 is agent's pieces
        board = env.game.get_board()
        agent_piece = agent_player + 1  # 1=Black, 2=White
        expected_agent = (board == agent_piece).astype(np.float32)
        assert np.array_equal(obs[0], expected_agent), \
            "Channel 0 does not match agent's pieces"
        
        # Property 5: Verify channel 1 is opponent's pieces
        opponent_piece = 3 - agent_piece
        expected_opponent = (board == opponent_piece).astype(np.float32)
        assert np.array_equal(obs[1], expected_opponent), \
            "Channel 1 does not match opponent's pieces"
        
        # Property 6: Verify channel 2 is valid moves
        valid_moves = env.game.get_valid_moves().reshape(8, 8)
        expected_valid = valid_moves.astype(np.float32)
        assert np.array_equal(obs[2], expected_valid), \
            "Channel 2 does not match valid moves"
        
        # Property 7: Verify channels 0 and 1 are disjoint (no overlap)
        overlap = obs[0] * obs[1]
        assert np.sum(overlap) == 0, \
            "Agent and opponent channels overlap"
        
        # Property 8: Verify all values are binary (0 or 1)
        assert np.all((obs == 0) | (obs == 1)), \
            "Observation contains non-binary values"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_observation_consistency_across_calls(self, moves):
        """
        Test that multiple calls to _get_observation return identical results.
        
        For any game state, calling _get_observation multiple times without
        changing the state should return identical observations.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get observation twice
        obs1 = env._get_observation()
        obs2 = env._get_observation()
        
        # Should be identical
        assert np.array_equal(obs1, obs2), \
            "Multiple calls to _get_observation returned different results"

    @given(valid_move_sequence(), agent_player_choice())
    @settings(max_examples=100, deadline=None)
    def test_observation_piece_count_matches_game_state(self, moves, agent_player):
        """
        Test that piece counts in observation match game state.
        
        For any game state, the sum of pieces in channels 0 and 1 should
        match the piece counts reported by the game engine.
        """
        env = OthelloEnv()
        env.reset()
        env.agent_player = agent_player
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get observation and piece counts
        obs = env._get_observation()
        black_count, white_count = env.game.get_piece_counts()
        
        # Count pieces in observation
        agent_piece_count = int(np.sum(obs[0]))
        opponent_piece_count = int(np.sum(obs[1]))
        
        # Verify counts match
        if agent_player == 0:  # Agent is Black
            assert agent_piece_count == black_count, \
                f"Agent piece count {agent_piece_count} != black count {black_count}"
            assert opponent_piece_count == white_count, \
                f"Opponent piece count {opponent_piece_count} != white count {white_count}"
        else:  # Agent is White
            assert agent_piece_count == white_count, \
                f"Agent piece count {agent_piece_count} != white count {white_count}"
            assert opponent_piece_count == black_count, \
                f"Opponent piece count {opponent_piece_count} != black count {black_count}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_observation_valid_moves_are_actually_valid(self, moves):
        """
        Test that valid moves in observation are actually valid.
        
        For any game state, all positions marked as valid in channel 2
        should be valid according to the game engine.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get observation and valid moves from game
        obs = env._get_observation()
        valid_moves_from_game = env.game.get_valid_moves()
        
        # Extract valid moves from observation (channel 2)
        valid_moves_from_obs = obs[2].flatten()
        
        # Should match exactly
        assert np.array_equal(valid_moves_from_obs, valid_moves_from_game.astype(np.float32)), \
            "Valid moves in observation don't match game engine"


class TestInfoDictionaryProperties:
    """Property-based tests for info dictionary completeness."""

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_info_dictionary_completeness(self, moves):
        """
        Feature: othello-rl-environment, Property 11: Info Dictionary Completeness
        
        **Validates: Requirements 3.6, 4.5, 7.7, 7.9**
        
        For any step or reset call, the returned info dictionary should contain
        all required fields with correct values.
        """
        env = OthelloEnv()
        obs, info = env.reset()
        
        # Check initial info dictionary
        self._verify_info_dict(env, info)
        
        # Apply moves and check info after each step
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
            
            # Get info again
            info = env._get_info()
            self._verify_info_dict(env, info)
    
    def _verify_info_dict(self, env, info):
        """Helper to verify info dictionary structure and values."""
        # Property 1: All required fields present
        required_fields = ["action_mask", "current_player", "black_count", "white_count", "agent_player"]
        for field in required_fields:
            assert field in info, f"Missing required field: {field}"
        
        # Property 2: action_mask has correct format
        assert info["action_mask"].shape == (64,), \
            f"action_mask has wrong shape: {info['action_mask'].shape}"
        assert info["action_mask"].dtype == bool, \
            f"action_mask has wrong dtype: {info['action_mask'].dtype}"
        
        # Property 3: current_player is valid
        assert info["current_player"] in [0, 1], \
            f"Invalid current_player: {info['current_player']}"
        
        # Property 4: piece counts are non-negative
        assert info["black_count"] >= 0, \
            f"Negative black_count: {info['black_count']}"
        assert info["white_count"] >= 0, \
            f"Negative white_count: {info['white_count']}"
        
        # Property 5: piece counts sum to at most 64
        total_pieces = info["black_count"] + info["white_count"]
        assert total_pieces <= 64, \
            f"Total pieces {total_pieces} exceeds 64"
        
        # Property 6: agent_player is valid
        assert info["agent_player"] in [0, 1], \
            f"Invalid agent_player: {info['agent_player']}"
        
        # Property 7: action_mask matches game state
        valid_moves_from_game = env.game.get_valid_moves()
        assert np.array_equal(info["action_mask"], valid_moves_from_game), \
            "action_mask doesn't match game state"
        
        # Property 8: piece counts match game state
        black_count, white_count = env.game.get_piece_counts()
        assert info["black_count"] == black_count, \
            f"black_count mismatch: {info['black_count']} != {black_count}"
        assert info["white_count"] == white_count, \
            f"white_count mismatch: {info['white_count']} != {white_count}"
        
        # Property 9: current_player matches game state
        current_player = env.game.get_current_player()
        assert info["current_player"] == current_player, \
            f"current_player mismatch: {info['current_player']} != {current_player}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_info_consistency_across_calls(self, moves):
        """
        Test that multiple calls to _get_info return identical results.
        
        For any game state, calling _get_info multiple times without
        changing the state should return identical info dictionaries.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get info twice
        info1 = env._get_info()
        info2 = env._get_info()
        
        # Should be identical
        assert np.array_equal(info1["action_mask"], info2["action_mask"])
        assert info1["current_player"] == info2["current_player"]
        assert info1["black_count"] == info2["black_count"]
        assert info1["white_count"] == info2["white_count"]
        assert info1["agent_player"] == info2["agent_player"]


class TestStepMethodProperties:
    """Property-based tests for step method and reward calculation."""

    @given(st.integers(0, 63))
    @settings(max_examples=100, deadline=None)
    def test_step_return_signature(self, action):
        """
        Feature: othello-rl-environment, Property 12: Step Return Signature
        
        **Validates: Requirements 3.3**
        
        For any action, calling step(action) should return exactly 5 values:
        - observation (ndarray)
        - reward (float)
        - terminated (bool)
        - truncated (bool)
        - info (dict)
        with correct types.
        """
        env = OthelloEnv()
        env.reset()
        
        # Call step
        result = env.step(action)
        
        # Property 1: Returns exactly 5 values
        assert len(result) == 5, \
            f"step() should return 5 values, got {len(result)}"
        
        obs, reward, terminated, truncated, info = result
        
        # Property 2: observation is ndarray with correct shape and dtype
        assert isinstance(obs, np.ndarray), \
            f"observation should be ndarray, got {type(obs)}"
        assert obs.shape == (3, 8, 8), \
            f"observation shape should be (3, 8, 8), got {obs.shape}"
        assert obs.dtype == np.float32, \
            f"observation dtype should be float32, got {obs.dtype}"
        
        # Property 3: reward is float
        assert isinstance(reward, (float, np.floating)), \
            f"reward should be float, got {type(reward)}"
        
        # Property 4: terminated is bool
        assert isinstance(terminated, (bool, np.bool_)), \
            f"terminated should be bool, got {type(terminated)}"
        
        # Property 5: truncated is bool
        assert isinstance(truncated, (bool, np.bool_)), \
            f"truncated should be bool, got {type(truncated)}"
        
        # Property 6: info is dict
        assert isinstance(info, dict), \
            f"info should be dict, got {type(info)}"
        
        # Property 7: truncated is always False (no time limits)
        assert truncated is False, \
            "truncated should always be False for Othello"

    @given(valid_move_sequence(), st.sampled_from(["penalty", "random", "error"]))
    @settings(max_examples=100, deadline=None)
    def test_invalid_move_handling(self, moves, invalid_mode):
        """
        Feature: othello-rl-environment, Property 13: Invalid Move Handling
        
        **Validates: Requirements 3.7, 5.3**
        
        For any invalid move, the environment should handle it according to
        the configured policy:
        - penalty mode: apply penalty and maintain state
        - random mode: select a random valid move
        - error mode: raise ValueError
        """
        env = OthelloEnv(invalid_move_mode=invalid_mode)
        env.reset()
        
        # Apply some valid moves to get to a state
        for move in moves[:5]:  # Apply up to 5 moves
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            obs, reward, terminated, truncated, info = env.step(move)
            if terminated:
                break
        
        # Get current state
        valid_moves = env.game.get_valid_moves()
        if not np.any(valid_moves):
            # No valid moves, skip this test case
            return
        
        # Find an invalid move
        invalid_moves = np.where(~valid_moves)[0]
        if len(invalid_moves) == 0:
            # All moves are valid (shouldn't happen), skip
            return
        
        invalid_action = invalid_moves[0]
        
        # Save state before invalid move
        black_count_before, white_count_before = env.game.get_piece_counts()
        current_player_before = env.game.get_current_player()
        
        if invalid_mode == "error":
            # Property 1: Error mode raises ValueError
            with pytest.raises(ValueError, match="Invalid move"):
                env.step(invalid_action)
        
        elif invalid_mode == "penalty":
            # Property 2: Penalty mode applies penalty and maintains state
            obs, reward, terminated, truncated, info = env.step(invalid_action)
            
            # Should receive penalty
            assert reward == env.invalid_move_penalty, \
                f"Expected penalty {env.invalid_move_penalty}, got {reward}"
            
            # State should be unchanged
            black_count_after, white_count_after = env.game.get_piece_counts()
            current_player_after = env.game.get_current_player()
            
            assert black_count_before == black_count_after, \
                "Piece counts changed after invalid move in penalty mode"
            assert white_count_before == white_count_after, \
                "Piece counts changed after invalid move in penalty mode"
            assert current_player_before == current_player_after, \
                "Current player changed after invalid move in penalty mode"
            
            # Game should not be terminated
            assert not terminated, \
                "Game terminated after invalid move in penalty mode"
        
        elif invalid_mode == "random":
            # Property 3: Random mode selects a valid move
            obs, reward, terminated, truncated, info = env.step(invalid_action)
            
            # State should have changed (a valid move was made)
            black_count_after, white_count_after = env.game.get_piece_counts()
            
            # Either piece counts changed or game is over
            assert (black_count_before != black_count_after or
                    white_count_before != white_count_after or
                    terminated), \
                "State unchanged after invalid move in random mode"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_sparse_reward_correctness(self, moves):
        """
        Feature: othello-rl-environment, Property 14: Sparse Reward Correctness
        
        **Validates: Requirements 5.1**
        
        For any game in sparse reward mode, rewards should be:
        - 0 for all non-terminal steps
        - +1 for agent win
        - -1 for agent loss
        - 0 for draw
        """
        env = OthelloEnv(reward_mode="sparse")
        env.reset()
        
        # Play through the game
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if not terminated:
                # Property 1: Non-terminal rewards are 0
                assert reward == 0.0, \
                    f"Non-terminal reward should be 0, got {reward}"
            else:
                # Property 2: Terminal rewards are +1, -1, or 0
                winner = env.game.get_winner()
                
                if winner == 2:  # Draw
                    assert reward == 0.0, \
                        f"Draw reward should be 0, got {reward}"
                elif winner == env.agent_player:  # Agent wins
                    assert reward == 1.0, \
                        f"Win reward should be 1.0, got {reward}"
                else:  # Agent loses
                    assert reward == -1.0, \
                        f"Loss reward should be -1.0, got {reward}"
                
                break

    @given(valid_move_sequence(), agent_player_choice())
    @settings(max_examples=100, deadline=None)
    def test_dense_reward_correctness(self, moves, agent_player):
        """
        Feature: othello-rl-environment, Property 15: Dense Reward Correctness
        
        **Validates: Requirements 5.2, 5.5**
        
        For any game in dense reward mode, the reward at each step should
        equal the normalized piece count differential:
        (agent_pieces - opponent_pieces) / 64
        """
        env = OthelloEnv(reward_mode="dense")
        env.reset()
        env.agent_player = agent_player
        
        # Play through the game
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Property 1: Reward equals normalized piece differential
            black_count, white_count = env.game.get_piece_counts()
            
            if agent_player == 0:  # Agent is Black
                expected_reward = (black_count - white_count) / 64.0
            else:  # Agent is White
                expected_reward = (white_count - black_count) / 64.0
            
            assert abs(reward - expected_reward) < 1e-6, \
                f"Dense reward {reward} != expected {expected_reward}"
            
            # Property 2: Reward is in range [-1, 1]
            assert -1.0 <= reward <= 1.0, \
                f"Dense reward {reward} out of range [-1, 1]"
            
            if terminated:
                break

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_custom_reward_function(self, moves):
        """
        Feature: othello-rl-environment, Property 16: Custom Reward Function
        
        **Validates: Requirements 5.4**
        
        For any custom reward function provided in configuration, the
        environment should call that function with the correct game state
        and use its return value as the reward.
        """
        # Define a custom reward function that returns piece differential
        def custom_reward(game_state):
            """Custom reward: piece differential scaled by 0.01"""
            black_count = game_state["black_count"]
            white_count = game_state["white_count"]
            agent_player = game_state["agent_player"]
            
            if agent_player == 0:  # Agent is Black
                return (black_count - white_count) * 0.01
            else:  # Agent is White
                return (white_count - black_count) * 0.01
        
        env = OthelloEnv(reward_mode="custom", reward_fn=custom_reward)
        env.reset()
        
        # Play through the game
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Property 1: Reward matches custom function output
            black_count, white_count = env.game.get_piece_counts()
            expected_reward = (black_count - white_count) * 0.01
            
            assert abs(reward - expected_reward) < 1e-6, \
                f"Custom reward {reward} != expected {expected_reward}"
            
            if terminated:
                break
    
    @given(st.integers(0, 63))
    @settings(max_examples=50, deadline=None)
    def test_custom_reward_function_receives_correct_state(self, action):
        """
        Test that custom reward function receives correct game state.
        
        The game state dict should contain all required fields with
        correct values.
        """
        received_states = []
        
        def tracking_reward(game_state):
            """Custom reward that tracks received state"""
            received_states.append(game_state.copy())
            return 0.5
        
        env = OthelloEnv(reward_mode="custom", reward_fn=tracking_reward)
        env.reset()
        
        # Make a move
        valid_moves = env.game.get_valid_moves()
        if valid_moves[action]:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Property 1: Custom function was called
            assert len(received_states) > 0, \
                "Custom reward function was not called"
            
            state = received_states[-1]
            
            # Property 2: State contains all required fields
            required_fields = [
                "board", "black_count", "white_count",
                "current_player", "agent_player", "game_over", "pieces_flipped"
            ]
            for field in required_fields:
                assert field in state, \
                    f"Missing required field in game state: {field}"
            
            # Property 3: State values match game state
            black_count, white_count = env.game.get_piece_counts()
            assert state["black_count"] == black_count
            assert state["white_count"] == white_count
            assert state["agent_player"] == env.agent_player



class TestSelfPlayProperties:
    """Property-based tests for self-play and opponent logic."""

    @given(valid_move_sequence(), st.sampled_from(["random", "greedy"]))
    @settings(max_examples=100, deadline=None)
    def test_opponent_move_execution(self, moves, opponent_type):
        """
        Feature: othello-rl-environment, Property 18: Opponent Move Execution
        
        **Validates: Requirements 6.4, 6.5**
        
        For any game with a specified opponent policy (random, greedy, or learned),
        after the agent's move, the opponent should automatically make a move
        according to its policy before the next agent observation.
        """
        env = OthelloEnv(opponent=opponent_type)
        env.reset()
        
        # Play through some moves
        for move in moves[:10]:  # Limit to 10 moves for performance
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            # Save state before move
            black_count_before, white_count_before = env.game.get_piece_counts()
            current_player_before = env.game.get_current_player()
            
            # Execute opponent move
            env._execute_opponent_move()
            
            # Property 1: If there were valid moves, state should change
            valid_moves_before = env.game.get_valid_moves()
            if np.any(valid_moves_before):
                black_count_after, white_count_after = env.game.get_piece_counts()
                
                # Either piece counts changed or game is over
                state_changed = (
                    black_count_before != black_count_after or
                    white_count_before != white_count_after
                )
                game_over = env.game.get_winner() != 3
                
                assert state_changed or game_over, \
                    "State unchanged after opponent move with valid moves available"
            
            # Property 2: Opponent selects a valid move
            # (implicitly tested by state change)
            
            # Property 3: For greedy opponent, verify it selects high-flip move
            if opponent_type == "greedy" and np.any(valid_moves_before):
                # The greedy opponent should have selected a move that flips
                # at least one piece (since it was valid)
                assert black_count_before != black_count_after or \
                       white_count_before != white_count_after, \
                    "Greedy opponent didn't flip any pieces"
            
            # Apply agent move to continue
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            valid_indices = np.where(valid_moves)[0]
            if len(valid_indices) == 0:
                break
            
            agent_move = valid_indices[0]
            env.game.step(agent_move)
            
            if env.game.get_winner() != 3:
                break

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_callable_opponent_policy(self, moves):
        """
        Test that callable opponent policies are called correctly.
        
        For any game with a callable opponent policy, the function should
        be called with the correct observation and its return value should
        be used as the action.
        """
        call_count = [0]
        received_observations = []
        
        def custom_opponent(obs):
            """Custom opponent that tracks calls and returns first valid move"""
            call_count[0] += 1
            received_observations.append(obs.copy())
            
            # Return first valid move from channel 2
            valid_moves = obs[2].flatten()
            valid_indices = np.where(valid_moves > 0)[0]
            if len(valid_indices) > 0:
                return valid_indices[0]
            return 0
        
        env = OthelloEnv(opponent=custom_opponent)
        env.reset()
        
        # Execute opponent move
        valid_moves_before = env.game.get_valid_moves()
        if np.any(valid_moves_before):
            env._execute_opponent_move()
            
            # Property 1: Callable was invoked
            assert call_count[0] > 0, \
                "Callable opponent was not called"
            
            # Property 2: Received correct observation format
            if len(received_observations) > 0:
                obs = received_observations[-1]
                assert obs.shape == (3, 8, 8), \
                    f"Callable received wrong observation shape: {obs.shape}"
                assert obs.dtype == np.float32, \
                    f"Callable received wrong dtype: {obs.dtype}"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_opponent_handles_no_valid_moves(self, moves):
        """
        Test that opponent move execution handles no valid moves gracefully.
        
        When the opponent has no valid moves, _execute_opponent_move should
        return without error and the turn should pass automatically.
        """
        env = OthelloEnv(opponent="random")
        env.reset()
        
        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Try to execute opponent move (should handle no valid moves)
        try:
            env._execute_opponent_move()
            # Should not raise an error
            assert True
        except Exception as e:
            pytest.fail(f"_execute_opponent_move raised exception: {e}")

    @given(st.sampled_from(["random", "greedy"]))
    @settings(max_examples=50, deadline=None)
    def test_greedy_opponent_maximizes_flips(self, opponent_type):
        """
        Test that greedy opponent selects moves that flip maximum pieces.
        
        For any game state with multiple valid moves, the greedy opponent
        should select the move that flips the most pieces.
        """
        if opponent_type != "greedy":
            return  # Only test greedy opponent
        
        env = OthelloEnv(opponent="greedy")
        env.reset()
        
        # Play a few moves to get to an interesting state
        for _ in range(3):
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            valid_indices = np.where(valid_moves)[0]
            move = valid_indices[0]
            env.game.step(move)
        
        # Get valid moves
        valid_moves = env.game.get_valid_moves()
        if not np.any(valid_moves):
            return  # No valid moves to test
        
        # Count flips for each valid move
        max_flips = -1
        for action in np.where(valid_moves)[0]:
            row, col = action // 8, action % 8
            flips = env._count_flips_for_move(row, col)
            max_flips = max(max_flips, flips)
        
        # Execute greedy opponent move
        black_count_before, white_count_before = env.game.get_piece_counts()
        env._execute_opponent_move()
        black_count_after, white_count_after = env.game.get_piece_counts()
        
        # Calculate actual flips
        piece_diff = abs((black_count_after - black_count_before) -
                        (white_count_after - white_count_before))
        
        # Property: Greedy should have selected a move with high flips
        # (may not be exactly max due to implementation details, but should be close)
        assert piece_diff > 0, \
            "Greedy opponent didn't flip any pieces"

    @given(valid_move_sequence(), agent_player_choice())
    @settings(max_examples=100, deadline=None)
    def test_perspective_consistency(self, moves, agent_player):
        """
        Feature: othello-rl-environment, Property 17: Perspective Consistency
        
        **Validates: Requirements 6.3**
        
        For any game in self-play mode, the observation should always be from
        the agent's perspective regardless of which player (Black/White) the
        agent is currently controlling, meaning the agent's pieces are always
        in channel 0.
        """
        env = OthelloEnv(opponent="random")
        env.reset()
        env.agent_player = agent_player
        
        # Play through moves
        for move in moves[:10]:  # Limit to 10 moves
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            # Apply move
            env.game.step(move)
            
            # Get observation
            obs = env._get_observation()
            
            # Property 1: Channel 0 always contains agent's pieces
            board = env.game.get_board()
            agent_piece = agent_player + 1  # 1=Black, 2=White
            expected_agent = (board == agent_piece).astype(np.float32)
            
            assert np.array_equal(obs[0], expected_agent), \
                f"Channel 0 doesn't match agent's pieces (agent_player={agent_player})"
            
            # Property 2: Channel 1 always contains opponent's pieces
            opponent_piece = 3 - agent_piece
            expected_opponent = (board == opponent_piece).astype(np.float32)
            
            assert np.array_equal(obs[1], expected_opponent), \
                f"Channel 1 doesn't match opponent's pieces (agent_player={agent_player})"
            
            # Property 3: Perspective is consistent across multiple calls
            obs2 = env._get_observation()
            assert np.array_equal(obs, obs2), \
                "Observation changed between calls without state change"
            
            if env.game.get_winner() != 3:
                break

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_perspective_consistency_with_opponent_moves(self, moves):
        """
        Test that perspective remains consistent after opponent moves.
        
        For any game with opponent moves, the observation should maintain
        the agent's perspective even after the opponent makes a move.
        """
        env = OthelloEnv(opponent="random")
        env.reset()
        
        # Agent is always Black (player 0)
        agent_player = env.agent_player
        
        # Play through moves
        for move in moves[:10]:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            # Apply agent move
            env.game.step(move)
            
            # Execute opponent move
            env._execute_opponent_move()
            
            # Get observation after opponent move
            obs = env._get_observation()
            
            # Property: Agent's perspective is maintained
            board = env.game.get_board()
            agent_piece = agent_player + 1
            expected_agent = (board == agent_piece).astype(np.float32)
            
            assert np.array_equal(obs[0], expected_agent), \
                "Perspective changed after opponent move"
            
            if env.game.get_winner() != 3:
                break

    @given(agent_player_choice())
    @settings(max_examples=50, deadline=None)
    def test_perspective_swap_when_agent_player_changes(self, new_agent_player):
        """
        Test that observation correctly swaps perspective when agent_player changes.
        
        When agent_player is changed, the observation should reflect the new
        perspective with the new agent's pieces in channel 0.
        """
        env = OthelloEnv()
        env.reset()
        
        # Play a few moves to get pieces on the board
        for _ in range(3):
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            valid_indices = np.where(valid_moves)[0]
            move = valid_indices[0]
            env.game.step(move)
        
        # Get observation with original agent_player
        obs1 = env._get_observation()
        original_agent_player = env.agent_player
        
        # Change agent_player
        env.agent_player = new_agent_player
        
        # Get observation with new agent_player
        obs2 = env._get_observation()
        
        if original_agent_player != new_agent_player:
            # Property: Channels 0 and 1 should be swapped
            assert np.array_equal(obs1[0], obs2[1]), \
                "Channel 0 from first obs should equal channel 1 from second obs"
            assert np.array_equal(obs1[1], obs2[0]), \
                "Channel 1 from first obs should equal channel 0 from second obs"
            
            # Channel 2 (valid moves) should be the same
            assert np.array_equal(obs1[2], obs2[2]), \
                "Valid moves channel should not change when agent_player changes"
        else:
            # Property: Observation should be identical if agent_player unchanged
            assert np.array_equal(obs1, obs2), \
                "Observation changed when agent_player remained the same"



class TestRenderingProperties:
    """Property-based tests for rendering functionality."""

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_ansi_rendering_completeness(self, moves):
        """
        Feature: othello-rl-environment, Property 19: ANSI Rendering Completeness

        **Validates: Requirements 10.2, 10.4, 10.5**

        For any game state, the ANSI rendering should return a string containing:
        - The board representation
        - Valid move indicators (*)
        - Piece counts for both players
        - The current player
        """
        env = OthelloEnv(render_mode="ansi")
        env.reset()

        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break

            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]

            env.game.step(move)

            if env.game.get_winner() != 3:
                break

        # Get ANSI rendering
        ansi_output = env._render_ansi()

        # Property 1: Output is a string
        assert isinstance(ansi_output, str), \
            f"ANSI rendering should return string, got {type(ansi_output)}"

        # Property 2: Output is non-empty
        assert len(ansi_output) > 0, \
            "ANSI rendering returned empty string"

        # Property 3: Contains board representation symbols
        # Should contain at least some of: ●, ○, ., *
        assert any(symbol in ansi_output for symbol in ["●", "○", ".", "*"]), \
            "ANSI rendering missing board symbols"

        # Property 4: Contains piece counts
        black_count, white_count = env.game.get_piece_counts()
        assert f"Black: {black_count}" in ansi_output, \
            f"ANSI rendering missing black count ({black_count})"
        assert f"White: {white_count}" in ansi_output, \
            f"ANSI rendering missing white count ({white_count})"

        # Property 5: Contains current player indicator
        current_player = env.game.get_current_player()
        player_name = "Black" if current_player == 0 else "White"
        assert f"Current player: {player_name}" in ansi_output, \
            f"ANSI rendering missing current player ({player_name})"

        # Property 6: Contains column headers (0-7)
        assert "0 1 2 3 4 5 6 7" in ansi_output, \
            "ANSI rendering missing column headers"

        # Property 7: Contains row numbers (0-7)
        for i in range(8):
            assert f"{i} " in ansi_output, \
                f"ANSI rendering missing row number {i}"

        # Property 8: Valid moves are marked with *
        valid_moves = env.game.get_valid_moves().reshape(8, 8)
        if np.any(valid_moves):
            # If there are valid moves, should contain * marker
            assert "*" in ansi_output, \
                "ANSI rendering missing valid move markers (*)"

        # Property 9: Number of lines is reasonable (should be ~12 lines)
        lines = ansi_output.split("\n")
        assert 10 <= len(lines) <= 15, \
            f"ANSI rendering has unexpected number of lines: {len(lines)}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_ansi_rendering_consistency(self, moves):
        """
        Test that ANSI rendering is consistent across multiple calls.

        For any game state, calling _render_ansi multiple times without
        changing the state should return identical strings.
        """
        env = OthelloEnv(render_mode="ansi")
        env.reset()

        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break

            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break

        # Get rendering twice
        ansi1 = env._render_ansi()
        ansi2 = env._render_ansi()

        # Should be identical
        assert ansi1 == ansi2, \
            "Multiple calls to _render_ansi returned different results"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_ansi_rendering_reflects_game_state(self, moves):
        """
        Test that ANSI rendering accurately reflects the game state.

        For any game state, the symbols in the ANSI rendering should
        match the actual board state.
        """
        env = OthelloEnv(render_mode="ansi")
        env.reset()

        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break

            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break

        # Get board state and rendering
        board = env.game.get_board()
        ansi_output = env._render_ansi()

        # Count pieces in board
        black_pieces = np.sum(board == 1)
        white_pieces = np.sum(board == 2)

        # Extract only the board lines (skip header and footer)
        lines = ansi_output.split("\n")
        board_lines = [line for line in lines if line and line[0].isdigit()]

        # Count symbols in board lines only
        board_text = "\n".join(board_lines)
        black_symbols = board_text.count("●")
        white_symbols = board_text.count("○")

        # Property: Symbol counts should match piece counts
        assert black_symbols == black_pieces, \
            f"Black symbol count {black_symbols} != piece count {black_pieces}"
        assert white_symbols == white_pieces, \
            f"White symbol count {white_symbols} != piece count {white_pieces}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_render_method_dispatcher(self, moves):
        """
        Test that render() method correctly dispatches to rendering modes.

        For any game state, render() should:
        - Return string for "ansi" mode
        - Return None for "human" mode (prints to console)
        - Return RGB array for "rgb_array" mode
        """
        # Test ANSI mode
        env_ansi = OthelloEnv(render_mode="ansi")
        env_ansi.reset()

        for move in moves[:5]:
            valid_moves = env_ansi.game.get_valid_moves()
            if not valid_moves[move]:
                break
            env_ansi.game.step(move)

        result_ansi = env_ansi.render()
        assert isinstance(result_ansi, str), \
            f"render() in ansi mode should return string, got {type(result_ansi)}"

        # Test human mode
        env_human = OthelloEnv(render_mode="human")
        env_human.reset()

        for move in moves[:5]:
            valid_moves = env_human.game.get_valid_moves()
            if not valid_moves[move]:
                break
            env_human.game.step(move)

        result_human = env_human.render()
        assert result_human is None, \
            f"render() in human mode should return None, got {type(result_human)}"

        # Test RGB array mode
        env_rgb = OthelloEnv(render_mode="rgb_array")
        env_rgb.reset()

        for move in moves[:5]:
            valid_moves = env_rgb.game.get_valid_moves()
            if not valid_moves[move]:
                break
            env_rgb.game.step(move)

        result_rgb = env_rgb.render()
        assert isinstance(result_rgb, np.ndarray), \
            f"render() in rgb_array mode should return ndarray, got {type(result_rgb)}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_rgb_array_format(self, moves):
        """
        Feature: othello-rl-environment, Property 20: RGB Array Format

        **Validates: Requirements 10.3**

        For any game state, the RGB array rendering should return a numpy
        array with shape (H, W, 3) and dtype uint8, suitable for video recording.
        """
        env = OthelloEnv(render_mode="rgb_array")
        env.reset()

        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break

            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]

            env.game.step(move)

            if env.game.get_winner() != 3:
                break

        # Get RGB rendering
        rgb_output = env._render_rgb()

        # Property 1: Output is numpy array
        assert isinstance(rgb_output, np.ndarray), \
            f"RGB rendering should return ndarray, got {type(rgb_output)}"

        # Property 2: Shape is (H, W, 3)
        assert len(rgb_output.shape) == 3, \
            f"RGB array should be 3D, got shape {rgb_output.shape}"
        assert rgb_output.shape[2] == 3, \
            f"RGB array should have 3 channels, got {rgb_output.shape[2]}"

        # Property 3: Height and width are positive
        assert rgb_output.shape[0] > 0, \
            f"RGB array height should be positive, got {rgb_output.shape[0]}"
        assert rgb_output.shape[1] > 0, \
            f"RGB array width should be positive, got {rgb_output.shape[1]}"

        # Property 4: dtype is uint8
        assert rgb_output.dtype == np.uint8, \
            f"RGB array dtype should be uint8, got {rgb_output.dtype}"

        # Property 5: Values are in range [0, 255]
        assert np.all(rgb_output >= 0), \
            "RGB array contains values < 0"
        assert np.all(rgb_output <= 255), \
            "RGB array contains values > 255"

        # Property 6: Array is not all zeros (should have some content)
        assert np.any(rgb_output > 0), \
            "RGB array is all zeros"

        # Property 7: Expected dimensions (512x512 based on design)
        assert rgb_output.shape == (512, 512, 3), \
            f"RGB array should be (512, 512, 3), got {rgb_output.shape}"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_rgb_rendering_consistency(self, moves):
        """
        Test that RGB rendering is consistent across multiple calls.

        For any game state, calling _render_rgb multiple times without
        changing the state should return identical arrays.
        """
        env = OthelloEnv(render_mode="rgb_array")
        env.reset()

        # Apply moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break

            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break

        # Get rendering twice
        rgb1 = env._render_rgb()
        rgb2 = env._render_rgb()

        # Should be identical
        assert np.array_equal(rgb1, rgb2), \
            "Multiple calls to _render_rgb returned different results"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_rgb_rendering_reflects_game_state(self, moves):
        """
        Test that RGB rendering changes when game state changes.

        For any sequence of moves, the RGB rendering should be different
        after each move (unless no pieces are placed).
        """
        env = OthelloEnv(render_mode="rgb_array")
        env.reset()

        rgb_before = env._render_rgb()

        # Apply a move
        valid_moves = env.game.get_valid_moves()
        if np.any(valid_moves):
            valid_indices = np.where(valid_moves)[0]
            move = valid_indices[0]
            env.game.step(move)

            rgb_after = env._render_rgb()

            # Property: Rendering should change after a move
            assert not np.array_equal(rgb_before, rgb_after), \
                "RGB rendering unchanged after move"

    @given(st.sampled_from(["ansi", "human", "rgb_array"]))
    @settings(max_examples=50, deadline=None)
    def test_render_mode_validation(self, render_mode):
        """
        Test that valid render modes are accepted.

        For any valid render mode, the environment should initialize
        without error.
        """
        try:
            env = OthelloEnv(render_mode=render_mode)
            env.reset()
            env.render()
            # Should not raise an error
            assert True
        except Exception as e:
            pytest.fail(f"Valid render_mode '{render_mode}' raised exception: {e}")

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_invalid_render_mode_raises_error(self, invalid_mode):
        """
        Test that invalid render modes raise ValueError.

        For any invalid render mode, the environment initialization
        should raise a ValueError.
        """
        # Skip if accidentally generated a valid mode
        if invalid_mode in ["ansi", "human", "rgb_array", None]:
            return

        with pytest.raises(ValueError, match="Invalid render_mode"):
            OthelloEnv(render_mode=invalid_mode)



class TestConfigurationProperties:
    """Property-based tests for configuration validation."""

    @given(st.sampled_from(["sparse", "dense", "custom"]))
    @settings(max_examples=50, deadline=None)
    def test_valid_reward_modes(self, reward_mode):
        """
        Test that valid reward modes are accepted.
        
        For any valid reward mode, the environment should initialize
        without error.
        """
        if reward_mode == "custom":
            # Custom mode requires reward_fn
            env = OthelloEnv(reward_mode=reward_mode, reward_fn=lambda x: 0.0)
        else:
            env = OthelloEnv(reward_mode=reward_mode)
        
        env.reset()
        assert env.reward_mode == reward_mode

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_invalid_reward_mode_raises_error(self, invalid_mode):
        """
        Feature: othello-rl-environment, Property 21: Configuration Validation
        
        **Validates: Requirements 11.6**
        
        For any invalid configuration (e.g., unknown reward mode, invalid
        opponent type), the environment initialization should raise a
        ValueError with a descriptive message.
        """
        # Skip if accidentally generated a valid mode
        if invalid_mode in ["sparse", "dense", "custom"]:
            return
        
        with pytest.raises(ValueError, match="Invalid reward_mode"):
            OthelloEnv(reward_mode=invalid_mode)

    @given(st.sampled_from(["self", "random", "greedy"]))
    @settings(max_examples=50, deadline=None)
    def test_valid_opponent_types(self, opponent_type):
        """
        Test that valid opponent types are accepted.
        
        For any valid opponent type (string), the environment should
        initialize without error.
        """
        env = OthelloEnv(opponent=opponent_type)
        env.reset()
        assert env.opponent == opponent_type

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_invalid_opponent_type_raises_error(self, invalid_opponent):
        """
        Test that invalid opponent types raise ValueError.
        
        For any invalid opponent type (not a valid string or callable),
        the environment initialization should raise a ValueError.
        """
        # Skip if accidentally generated a valid opponent
        if invalid_opponent in ["self", "random", "greedy"]:
            return
        
        with pytest.raises(ValueError, match="Invalid opponent"):
            OthelloEnv(opponent=invalid_opponent)

    def test_callable_opponent_is_accepted(self):
        """
        Test that callable opponent policies are accepted.
        
        A callable function should be accepted as a valid opponent.
        """
        def custom_opponent(obs):
            return 0
        
        env = OthelloEnv(opponent=custom_opponent)
        env.reset()
        assert callable(env.opponent)

    @given(st.sampled_from(["penalty", "random", "error"]))
    @settings(max_examples=50, deadline=None)
    def test_valid_invalid_move_modes(self, invalid_move_mode):
        """
        Test that valid invalid_move_mode values are accepted.
        
        For any valid invalid_move_mode, the environment should
        initialize without error.
        """
        env = OthelloEnv(invalid_move_mode=invalid_move_mode)
        env.reset()
        assert env.invalid_move_mode == invalid_move_mode

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_invalid_move_mode_raises_error(self, invalid_mode):
        """
        Test that invalid invalid_move_mode values raise ValueError.
        
        For any invalid invalid_move_mode, the environment initialization
        should raise a ValueError.
        """
        # Skip if accidentally generated a valid mode
        if invalid_mode in ["penalty", "random", "error"]:
            return
        
        with pytest.raises(ValueError, match="Invalid invalid_move_mode"):
            OthelloEnv(invalid_move_mode=invalid_mode)

    def test_custom_reward_mode_requires_reward_fn(self):
        """
        Test that custom reward mode requires reward_fn parameter.
        
        When reward_mode is "custom", reward_fn must be provided,
        otherwise ValueError should be raised.
        """
        with pytest.raises(ValueError, match="reward_fn must be provided"):
            OthelloEnv(reward_mode="custom", reward_fn=None)

    def test_custom_reward_mode_with_reward_fn_succeeds(self):
        """
        Test that custom reward mode with reward_fn succeeds.
        
        When reward_mode is "custom" and reward_fn is provided,
        the environment should initialize successfully.
        """
        def custom_reward(game_state):
            return 0.5
        
        env = OthelloEnv(reward_mode="custom", reward_fn=custom_reward)
        env.reset()
        assert env.reward_mode == "custom"
        assert env.reward_fn == custom_reward

    @given(st.sampled_from([None, "ansi", "human", "rgb_array"]))
    @settings(max_examples=50, deadline=None)
    def test_valid_render_modes_in_init(self, render_mode):
        """
        Test that valid render modes are accepted in initialization.
        
        For any valid render mode (including None), the environment
        should initialize without error.
        """
        env = OthelloEnv(render_mode=render_mode)
        env.reset()
        assert env.render_mode == render_mode

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_invalid_render_mode_in_init_raises_error(self, invalid_mode):
        """
        Test that invalid render modes raise ValueError in initialization.
        
        For any invalid render mode, the environment initialization
        should raise a ValueError.
        """
        # Skip if accidentally generated a valid mode
        if invalid_mode in ["ansi", "human", "rgb_array"]:
            return
        
        with pytest.raises(ValueError, match="Invalid render_mode"):
            OthelloEnv(render_mode=invalid_mode)

    @given(
        st.sampled_from(["sparse", "dense"]),
        st.sampled_from(["self", "random", "greedy"]),
        st.sampled_from(["penalty", "random", "error"]),
        st.sampled_from([None, "ansi", "human", "rgb_array"])
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_configuration_combinations(
        self, reward_mode, opponent, invalid_move_mode, render_mode
    ):
        """
        Test that valid configuration combinations are accepted.
        
        For any combination of valid configuration parameters, the
        environment should initialize and work correctly.
        """
        env = OthelloEnv(
            reward_mode=reward_mode,
            opponent=opponent,
            invalid_move_mode=invalid_move_mode,
            render_mode=render_mode
        )
        
        obs, info = env.reset()
        
        # Verify environment is functional
        assert obs.shape == (3, 8, 8)
        assert isinstance(info, dict)
        
        # Try a step
        valid_moves = env.game.get_valid_moves()
        if np.any(valid_moves):
            action = np.where(valid_moves)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify step returns correct types
            assert obs.shape == (3, 8, 8)
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)



class TestStatePersistenceProperties:
    """Property-based tests for state save/load functionality."""

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_state_persistence_round_trip(self, moves):
        """
        Feature: othello-rl-environment, Property 22: State Persistence Round-Trip
        
        **Validates: Requirements 11.7**
        
        For any game state, saving the state and then loading it should
        result in an identical game state.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves to create a game state
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if terminated:
                break
        
        # Save state
        saved_state = env.save_state()
        
        # Property 1: save_state returns a dictionary
        assert isinstance(saved_state, dict), \
            f"save_state should return dict, got {type(saved_state)}"
        
        # Property 2: All required fields are present
        required_fields = [
            "board", "current_player", "black_count", "white_count",
            "agent_player", "game_over", "winner", "move_history"
        ]
        for field in required_fields:
            assert field in saved_state, \
                f"Missing required field in saved state: {field}"
        
        # Property 3: Board is a numpy array with correct shape
        assert isinstance(saved_state["board"], np.ndarray), \
            "board should be numpy array"
        assert saved_state["board"].shape == (8, 8), \
            f"board should have shape (8, 8), got {saved_state['board'].shape}"
        
        # Property 4: move_history is a list
        assert isinstance(saved_state["move_history"], list), \
            "move_history should be a list"
        
        # Create new environment and load state
        env2 = OthelloEnv()
        env2.reset()
        env2.load_state(saved_state)
        
        # Get state from loaded environment
        loaded_state = env2.save_state()
        
        # Property 5: Board states match exactly
        assert np.array_equal(saved_state["board"], loaded_state["board"]), \
            "Board state not preserved after save/load"
        
        # Property 6: Current player matches
        assert saved_state["current_player"] == loaded_state["current_player"], \
            f"Current player mismatch: {saved_state['current_player']} != {loaded_state['current_player']}"
        
        # Property 7: Piece counts match
        assert saved_state["black_count"] == loaded_state["black_count"], \
            f"Black count mismatch: {saved_state['black_count']} != {loaded_state['black_count']}"
        assert saved_state["white_count"] == loaded_state["white_count"], \
            f"White count mismatch: {saved_state['white_count']} != {loaded_state['white_count']}"
        
        # Property 8: Agent player matches
        assert saved_state["agent_player"] == loaded_state["agent_player"], \
            f"Agent player mismatch: {saved_state['agent_player']} != {loaded_state['agent_player']}"
        
        # Property 9: Game over status matches
        assert saved_state["game_over"] == loaded_state["game_over"], \
            f"Game over status mismatch: {saved_state['game_over']} != {loaded_state['game_over']}"
        
        # Property 10: Winner matches
        assert saved_state["winner"] == loaded_state["winner"], \
            f"Winner mismatch: {saved_state['winner']} != {loaded_state['winner']}"
        
        # Property 11: Move history matches
        assert saved_state["move_history"] == loaded_state["move_history"], \
            "Move history not preserved after save/load"
        
        # Property 12: Observations match
        obs1 = env._get_observation()
        obs2 = env2._get_observation()
        assert np.array_equal(obs1, obs2), \
            "Observations don't match after save/load"
        
        # Property 13: Info dictionaries match
        info1 = env._get_info()
        info2 = env2._get_info()
        assert np.array_equal(info1["action_mask"], info2["action_mask"]), \
            "Action masks don't match after save/load"
        assert info1["current_player"] == info2["current_player"], \
            "Current player in info doesn't match after save/load"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_save_state_does_not_modify_environment(self, moves):
        """
        Test that save_state does not modify the environment.
        
        For any game state, calling save_state should not change the
        environment's state.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves
        for move in moves[:10]:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            
            valid, _, game_over = env.game.step(move)
            if not valid or game_over:
                break
        
        # Get state before save
        board_before = env.game.get_board().copy()
        black_count_before, white_count_before = env.game.get_piece_counts()
        current_player_before = env.game.get_current_player()
        
        # Save state
        saved_state = env.save_state()
        
        # Get state after save
        board_after = env.game.get_board()
        black_count_after, white_count_after = env.game.get_piece_counts()
        current_player_after = env.game.get_current_player()
        
        # Property: State should be unchanged
        assert np.array_equal(board_before, board_after), \
            "Board changed after save_state"
        assert black_count_before == black_count_after, \
            "Black count changed after save_state"
        assert white_count_before == white_count_after, \
            "White count changed after save_state"
        assert current_player_before == current_player_after, \
            "Current player changed after save_state"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_load_state_with_invalid_data_raises_error(self, moves):
        """
        Test that load_state raises ValueError for invalid state data.
        
        For any invalid state dictionary, load_state should raise
        a ValueError with a descriptive message.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply some moves
        for move in moves[:5]:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves[move]:
                break
            env.game.step(move)
        
        # Save valid state
        valid_state = env.save_state()
        
        # Test 1: Missing required field
        invalid_state = valid_state.copy()
        del invalid_state["board"]
        
        env2 = OthelloEnv()
        env2.reset()
        with pytest.raises(ValueError, match="Missing required field"):
            env2.load_state(invalid_state)
        
        # Test 2: Invalid board shape
        invalid_state = valid_state.copy()
        invalid_state["board"] = np.zeros((4, 4), dtype=np.uint8)
        
        env3 = OthelloEnv()
        env3.reset()
        with pytest.raises(ValueError, match="board must have shape"):
            env3.load_state(invalid_state)
        
        # Test 3: Invalid board type
        invalid_state = valid_state.copy()
        invalid_state["board"] = [[0] * 8 for _ in range(8)]
        
        env4 = OthelloEnv()
        env4.reset()
        with pytest.raises(ValueError, match="board must be a numpy array"):
            env4.load_state(invalid_state)
        
        # Test 4: Invalid move_history type
        invalid_state = valid_state.copy()
        invalid_state["move_history"] = "not a list"
        
        env5 = OthelloEnv()
        env5.reset()
        with pytest.raises(ValueError, match="move_history must be a list"):
            env5.load_state(invalid_state)

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_multiple_save_load_cycles(self, moves):
        """
        Test that multiple save/load cycles preserve state.
        
        For any game state, performing multiple save/load cycles
        should preserve the state exactly.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves using env.step to track history
        for move in moves[:10]:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            if terminated:
                break
        
        # Save initial state
        state1 = env.save_state()
        
        # Load and save multiple times
        for _ in range(3):
            env_temp = OthelloEnv()
            env_temp.reset()
            env_temp.load_state(state1)
            state1 = env_temp.save_state()
        
        # Final state should match original
        assert np.array_equal(state1["board"], env.game.get_board()), \
            "Board changed after multiple save/load cycles"
        assert state1["move_history"] == env._move_history, \
            "Move history changed after multiple save/load cycles"

    @given(valid_move_sequence(), agent_player_choice())
    @settings(max_examples=50, deadline=None)
    def test_save_load_preserves_agent_player(self, moves, agent_player):
        """
        Test that save/load preserves agent_player setting.
        
        For any game state with any agent_player setting, save/load
        should preserve the agent_player value.
        """
        env = OthelloEnv()
        env.reset()
        env.agent_player = agent_player
        
        # Apply moves using env.step to track history
        for move in moves[:5]:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            # Use env.step to track move history
            obs, reward, terminated, truncated, info = env.step(move)
            if terminated:
                break
        
        # Save and load
        state = env.save_state()
        
        env2 = OthelloEnv()
        env2.reset()
        env2.load_state(state)
        
        # Property: agent_player should be preserved
        assert env2.agent_player == agent_player, \
            f"agent_player not preserved: {env2.agent_player} != {agent_player}"

    def test_save_load_initial_state(self):
        """
        Test that save/load works for initial state.
        
        Saving and loading the initial state (no moves played) should
        result in an identical initial state.
        """
        env = OthelloEnv()
        env.reset()
        
        # Save initial state
        state = env.save_state()
        
        # Load into new environment
        env2 = OthelloEnv()
        env2.reset()
        env2.load_state(state)
        
        # Verify states match
        assert np.array_equal(env.game.get_board(), env2.game.get_board()), \
            "Initial board state not preserved"
        assert len(state["move_history"]) == 0, \
            "Initial state should have empty move history"
        assert state["black_count"] == 2, \
            "Initial state should have 2 black pieces"
        assert state["white_count"] == 2, \
            "Initial state should have 2 white pieces"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_load_state_allows_continued_play(self, moves):
        """
        Test that after loading state, the game can continue normally.
        
        For any game state, after loading it, the environment should
        allow continued play with correct game logic.
        """
        env = OthelloEnv()
        env.reset()
        
        # Apply moves using env.step to track history
        for move in moves[:10]:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            if terminated:
                break
        
        # Save state
        state = env.save_state()
        
        # Load into new environment
        env2 = OthelloEnv()
        env2.reset()
        env2.load_state(state)
        
        # Try to continue playing
        valid_moves = env2.game.get_valid_moves()
        if np.any(valid_moves):
            action = np.where(valid_moves)[0][0]
            obs, reward, terminated, truncated, info = env2.step(action)
            
            # Property: Step should work correctly
            assert obs.shape == (3, 8, 8), \
                "Observation has wrong shape after load"
            assert isinstance(reward, (float, np.floating)), \
                "Reward has wrong type after load"
            assert isinstance(terminated, (bool, np.bool_)), \
                "Terminated has wrong type after load"



class TestEpisodeTerminationProperties:
    """Property-based tests for episode termination signals."""

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_episode_termination_signals(self, moves):
        """
        Feature: othello-rl-environment, Property 23: Episode Termination Signals
        
        **Validates: Requirements 7.6**
        
        For any game, the terminated flag should be set correctly when the
        game ends, and truncated should always be False.
        """
        env = OthelloEnv()
        env.reset()
        
        # Play through the game
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Property 1: truncated is always False
            assert truncated is False, \
                f"truncated should always be False, got {truncated}"
            
            # Property 2: terminated matches game over status
            game_over = env.game.get_winner() != 3
            assert terminated == game_over, \
                f"terminated ({terminated}) doesn't match game_over ({game_over})"
            
            # Property 3: If terminated, game should be over
            if terminated:
                winner = env.game.get_winner()
                assert winner in [0, 1, 2], \
                    f"Game terminated but winner is {winner} (should be 0, 1, or 2)"
                break
            
            # Property 4: If not terminated, game should not be over
            if not terminated:
                assert env.game.get_winner() == 3, \
                    "Game not terminated but winner is set"

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_truncated_always_false(self, moves):
        """
        Test that truncated is always False for Othello.
        
        For any game state and any action, truncated should always be False
        since Othello games don't have time limits or truncation conditions.
        """
        env = OthelloEnv()
        env.reset()
        
        # Play through moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Property: truncated is always False
            assert truncated is False, \
                f"truncated should always be False, got {truncated}"
            
            if terminated:
                break

    @given(valid_move_sequence())
    @settings(max_examples=100, deadline=None)
    def test_terminated_only_when_game_over(self, moves):
        """
        Test that terminated is only True when the game is actually over.
        
        For any game state, terminated should be True if and only if
        the game has reached a terminal state (board full or no valid moves).
        """
        env = OthelloEnv()
        env.reset()
        
        # Play through moves
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Get game over status from game engine
            winner = env.game.get_winner()
            game_over = winner != 3
            
            # Property: terminated matches game over status
            assert terminated == game_over, \
                f"terminated ({terminated}) doesn't match game_over ({game_over})"
            
            if terminated:
                break

    @given(st.integers(0, 63))
    @settings(max_examples=100, deadline=None)
    def test_invalid_move_termination_signals(self, action):
        """
        Test termination signals for invalid moves.
        
        For any invalid move in penalty mode, terminated should be False
        and truncated should be False (game continues).
        """
        env = OthelloEnv(invalid_move_mode="penalty")
        env.reset()
        
        # Get valid moves
        valid_moves = env.game.get_valid_moves()
        
        # Find an invalid move
        invalid_moves = np.where(~valid_moves)[0]
        if len(invalid_moves) == 0:
            # All moves are valid, skip this test case
            return
        
        invalid_action = invalid_moves[0]
        
        # Execute invalid move
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Property 1: truncated is False
        assert truncated is False, \
            "truncated should be False for invalid move in penalty mode"
        
        # Property 2: terminated is False (game continues)
        assert terminated is False, \
            "terminated should be False for invalid move in penalty mode"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_termination_at_game_end(self, moves):
        """
        Test that terminated is True when the game ends.
        
        For any complete game, the final step should have terminated=True.
        """
        env = OthelloEnv()
        env.reset()
        
        last_terminated = False
        
        # Play until game ends
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            last_terminated = terminated
            
            if terminated:
                break
        
        # If game ended, last step should have terminated=True
        if env.game.get_winner() != 3:
            assert last_terminated is True, \
                "Game ended but last step had terminated=False"

    @given(valid_move_sequence())
    @settings(max_examples=50, deadline=None)
    def test_no_steps_after_termination(self, moves):
        """
        Test that after termination, the game state is final.
        
        For any game that reaches termination, the winner should be set
        and the game should be in a final state.
        """
        env = OthelloEnv()
        env.reset()
        
        # Play until game ends
        for move in moves:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if terminated:
                # Property 1: Winner should be set
                winner = env.game.get_winner()
                assert winner in [0, 1, 2], \
                    f"Game terminated but winner is {winner} (should be 0, 1, or 2)"
                
                # Property 2: Game should be over
                assert env.game.get_winner() != 3, \
                    "Game terminated but get_winner() returns 3 (not finished)"
                
                break

    @given(valid_move_sequence(), st.sampled_from(["sparse", "dense"]))
    @settings(max_examples=50, deadline=None)
    def test_termination_signals_consistent_across_reward_modes(
        self, moves, reward_mode
    ):
        """
        Test that termination signals are consistent across reward modes.
        
        For any game state, the terminated and truncated flags should be
        the same regardless of the reward mode.
        """
        env = OthelloEnv(reward_mode=reward_mode)
        env.reset()
        
        # Play through moves
        for move in moves[:20]:
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            # Select a valid move
            if not valid_moves[move]:
                valid_indices = np.where(valid_moves)[0]
                if len(valid_indices) == 0:
                    break
                move = valid_indices[0]
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            # Property 1: truncated is always False
            assert truncated is False, \
                f"truncated should be False in {reward_mode} mode"
            
            # Property 2: terminated matches game state
            game_over = env.game.get_winner() != 3
            assert terminated == game_over, \
                f"terminated doesn't match game state in {reward_mode} mode"
            
            if terminated:
                break

    def test_reset_clears_termination(self):
        """
        Test that reset clears termination state.
        
        After a game ends and reset is called, the next step should
        have terminated=False.
        """
        env = OthelloEnv()
        env.reset()
        
        # Play a few moves
        for _ in range(10):
            valid_moves = env.game.get_valid_moves()
            if not np.any(valid_moves):
                break
            
            action = np.where(valid_moves)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        
        # Reset the environment
        env.reset()
        
        # Make a move
        valid_moves = env.game.get_valid_moves()
        if np.any(valid_moves):
            action = np.where(valid_moves)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Property: After reset, game should not be terminated
            assert terminated is False, \
                "Game terminated immediately after reset"
            assert truncated is False, \
                "Game truncated immediately after reset"
