#!/usr/bin/env python3
"""
Human vs Agent Othello Game Script

This script allows a human player to play Othello against a trained RL agent
or against built-in opponent policies (random, greedy).

Usage:
    python scripts/play_human_vs_agent.py [--checkpoint PATH] [--opponent random|greedy]
    
Examples:
    # Play against random opponent
    python scripts/play_human_vs_agent.py --opponent random
    
    # Play against greedy opponent
    python scripts/play_human_vs_agent.py --opponent greedy
    
    # Play against trained agent (requires checkpoint)
    python scripts/play_human_vs_agent.py --checkpoint /path/to/checkpoint
"""

import argparse
import sys
import gymnasium as gym
import numpy as np

# Import the Othello environment
import aip_rl.othello


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play Othello against an agent or built-in opponent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained agent checkpoint (for playing against trained agent)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Built-in opponent policy (default: random)"
    )
    parser.add_argument(
        "--human-color",
        type=str,
        choices=["black", "white"],
        default="black",
        help="Color for human player (default: black)"
    )
    
    return parser.parse_args()


def get_human_move(valid_moves: np.ndarray) -> int:
    """
    Get move input from human player via console.
    
    Args:
        valid_moves: Boolean array of valid moves (64,)
    
    Returns:
        Action (0-63) selected by human
    """
    valid_indices = np.where(valid_moves)[0]
    
    if len(valid_indices) == 0:
        print("No valid moves available. Turn will be passed.")
        return -1
    
    print("\nValid moves:")
    for idx in valid_indices:
        row, col = idx // 8, idx % 8
        print(f"  {idx}: row {row}, col {col}")
    
    while True:
        try:
            user_input = input("\nEnter your move (0-63) or 'q' to quit: ").strip()
            
            if user_input.lower() == 'q':
                print("Quitting game...")
                sys.exit(0)
            
            action = int(user_input)
            
            if action < 0 or action > 63:
                print(f"Invalid input: {action}. Must be between 0 and 63.")
                continue
            
            if not valid_moves[action]:
                print(f"Invalid move: {action}. That position is not a valid move.")
                print("Valid moves are:", valid_indices.tolist())
                continue
            
            return action
            
        except ValueError:
            print(f"Invalid input. Please enter a number between 0 and 63, or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nQuitting game...")
            sys.exit(0)


def load_trained_agent(checkpoint_path: str):
    """
    Load a trained agent from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
    
    Returns:
        Agent policy function that takes observation and returns action
    """
    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        
        # Load the algorithm from checkpoint
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        def agent_policy(obs):
            """Policy function that uses the trained agent."""
            # Compute action using the trained policy
            action = algo.compute_single_action(obs, explore=False)
            return action
        
        print(f"Loaded trained agent from: {checkpoint_path}")
        return agent_policy
        
    except ImportError:
        print("Error: Ray RLlib is required to load trained agents.")
        print("Install with: pip install ray[rllib]")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


def play_game(opponent_policy, human_color: str = "black"):
    """
    Play a game of Othello with human vs agent.
    
    Args:
        opponent_policy: Opponent policy ("random", "greedy", or callable)
        human_color: Color for human player ("black" or "white")
    """
    # Create environment with opponent policy
    env = gym.make(
        "Othello-v0",
        opponent=opponent_policy,
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode="human"
    )
    
    # Determine if human plays first
    human_is_black = (human_color == "black")
    
    print("\n" + "="*60)
    print("OTHELLO - Human vs Agent")
    print("="*60)
    print(f"Human plays as: {human_color.upper()}")
    print(f"Opponent: {opponent_policy if isinstance(opponent_policy, str) else 'trained agent'}")
    print("\nBoard positions are numbered 0-63:")
    print("  Row 0: 0-7")
    print("  Row 1: 8-15")
    print("  ...")
    print("  Row 7: 56-63")
    print("\nValid moves are marked with '*' on the board.")
    print("="*60 + "\n")
    
    # Reset environment
    obs, info = env.reset()
    
    # Display initial board
    print("Initial board:")
    env.render()
    
    terminated = False
    move_count = 0
    
    while not terminated:
        move_count += 1
        current_player = info["current_player"]
        is_human_turn = (current_player == 0 and human_is_black) or \
                       (current_player == 1 and not human_is_black)
        
        print(f"\n{'='*60}")
        print(f"Move {move_count}")
        print(f"{'='*60}")
        
        if is_human_turn:
            # Human's turn
            print("Your turn!")
            valid_moves = info["action_mask"]
            
            if not np.any(valid_moves):
                print("No valid moves. Turn passed to opponent.")
                # In self-play mode, the environment handles turn passing
                # We need to make a dummy step or handle this case
                # For now, we'll just continue and let the opponent play
                action = -1
            else:
                action = get_human_move(valid_moves)
            
            if action >= 0:
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    print(f"\nYou played: {action} (row {action//8}, col {action%8})")
                    print("\nBoard after your move:")
                    env.render()
                    
                    if terminated:
                        break
                    
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
        else:
            # Agent's turn (handled automatically by environment in self-play mode)
            # Since we're using opponent policy, the environment executes it automatically
            # We just need to wait for the next observation
            print("Opponent is thinking...")
            
            # The opponent move is executed automatically by the environment
            # after the human's move in the step() function
            # So we don't need to do anything here
            pass
    
    # Game over - display results
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    
    # Access the unwrapped environment to get game state
    unwrapped_env = env.unwrapped
    winner = info.get("winner", unwrapped_env.game.get_winner())
    black_count, white_count = unwrapped_env.game.get_piece_counts()
    
    print(f"\nFinal score:")
    print(f"  ● Black: {black_count}")
    print(f"  ○ White: {white_count}")
    
    if winner == 2:
        print("\nResult: DRAW")
    elif (winner == 0 and human_is_black) or (winner == 1 and not human_is_black):
        print("\n🎉 YOU WIN! 🎉")
    else:
        print("\nYou lost. Better luck next time!")
    
    print("\nFinal board:")
    env.render()
    
    env.close()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine opponent policy
    if args.checkpoint:
        opponent_policy = load_trained_agent(args.checkpoint)
    else:
        opponent_policy = args.opponent
    
    # Play the game
    play_game(opponent_policy, args.human_color)


if __name__ == "__main__":
    main()
