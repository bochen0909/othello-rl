#!/usr/bin/env python3
"""
Spectator Mode Script for Othello

This script allows you to watch two agents play Othello against each other.
Useful for evaluating agent performance and understanding agent behavior.

Usage:
    python scripts/watch_agents_play.py [OPTIONS]
    
Examples:
    # Watch two random agents play
    python scripts/watch_agents_play.py --agent1 random --agent2 random
    
    # Watch random vs greedy
    python scripts/watch_agents_play.py --agent1 random --agent2 greedy
    
    # Watch trained agent vs greedy (requires checkpoint)
    python scripts/watch_agents_play.py --agent1-checkpoint /path/to/checkpoint --agent2 greedy
    
    # Watch multiple games with statistics
    python scripts/watch_agents_play.py --agent1 random --agent2 greedy --num-games 10 --delay 0.5
"""

import argparse
import time
import sys
import gymnasium as gym
import numpy as np
from typing import Optional, Callable, Dict, Any

# Import the Othello environment
import aip_rl.othello


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch two agents play Othello against each other"
    )
    
    # Agent 1 configuration
    parser.add_argument(
        "--agent1",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Built-in policy for agent 1 (default: random)"
    )
    parser.add_argument(
        "--agent1-checkpoint",
        type=str,
        default=None,
        help="Path to trained agent checkpoint for agent 1"
    )
    
    # Agent 2 configuration
    parser.add_argument(
        "--agent2",
        type=str,
        choices=["random", "greedy"],
        default="greedy",
        help="Built-in policy for agent 2 (default: greedy)"
    )
    parser.add_argument(
        "--agent2-checkpoint",
        type=str,
        default=None,
        help="Path to trained agent checkpoint for agent 2"
    )
    
    # Game configuration
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between moves (default: 1.0)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable board rendering (only show statistics)"
    )
    parser.add_argument(
        "--swap-colors",
        action="store_true",
        help="Swap agent colors each game (for fair evaluation)"
    )
    
    return parser.parse_args()


def load_trained_agent(checkpoint_path: str, agent_name: str) -> Callable:
    """
    Load a trained agent from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        agent_name: Name of the agent (for display purposes)
    
    Returns:
        Agent policy function that takes observation and returns action
    """
    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Load the algorithm from checkpoint
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        def agent_policy(obs):
            """Policy function that uses the trained agent."""
            # Compute action using the trained policy
            action = algo.compute_single_action(obs, explore=False)
            return action
        
        print(f"Loaded {agent_name} from: {checkpoint_path}")
        return agent_policy
        
    except ImportError:
        print("Error: Ray RLlib is required to load trained agents.")
        print("Install with: pip install ray[rllib]")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint for {agent_name}: {e}")
        sys.exit(1)


def get_agent_policy(
    agent_type: str,
    checkpoint_path: Optional[str],
    agent_name: str
) -> Callable:
    """
    Get agent policy function.
    
    Args:
        agent_type: Type of agent ("random" or "greedy")
        checkpoint_path: Path to checkpoint (if using trained agent)
        agent_name: Name of the agent (for display)
    
    Returns:
        Policy function or string identifier
    """
    if checkpoint_path:
        return load_trained_agent(checkpoint_path, agent_name)
    else:
        return agent_type


def play_single_game(
    agent1_policy,
    agent2_policy,
    render: bool = True,
    delay: float = 1.0,
    agent1_is_black: bool = True
) -> Dict[str, Any]:
    """
    Play a single game between two agents.
    
    Args:
        agent1_policy: Policy for agent 1 (string or callable)
        agent2_policy: Policy for agent 2 (string or callable)
        render: Whether to render the board
        delay: Delay between moves in seconds
        agent1_is_black: Whether agent 1 plays as black
    
    Returns:
        Dictionary with game statistics
    """
    # Create environment
    # Agent 1 will be the "agent" in the environment
    # Agent 2 will be the "opponent"
    env = gym.make(
        "Othello-v0",
        opponent=agent2_policy,
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode="human" if render else None
    )
    
    # Reset environment
    obs, info = env.reset()
    
    if render:
        print("\nInitial board:")
        env.render()
        time.sleep(delay)
    
    terminated = False
    move_count = 0
    total_reward = 0
    
    while not terminated:
        move_count += 1
        
        # Get valid moves
        valid_moves = info["action_mask"]
        
        if not np.any(valid_moves):
            # No valid moves, turn will be passed
            if render:
                print(f"\nMove {move_count}: No valid moves, turn passed")
            break
        
        # Agent 1's turn (select action)
        if callable(agent1_policy):
            action = agent1_policy(obs)
        elif agent1_policy == "random":
            valid_indices = np.where(valid_moves)[0]
            action = np.random.choice(valid_indices)
        elif agent1_policy == "greedy":
            # Use environment's greedy move calculation
            action = env._get_greedy_move()
        else:
            raise ValueError(f"Unknown agent policy: {agent1_policy}")
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if render:
            current_player = "Black" if info["current_player"] == 0 else "White"
            print(f"\nMove {move_count}: Agent 1 played {action} (row {action//8}, col {action%8})")
            env.render()
            
            if not terminated:
                time.sleep(delay)
    
    # Get final statistics
    winner = env.game.get_winner()
    black_count, white_count = env.game.get_piece_counts()
    
    # Determine which agent won
    if winner == 2:
        result = "draw"
    elif (winner == 0 and agent1_is_black) or (winner == 1 and not agent1_is_black):
        result = "agent1"
    else:
        result = "agent2"
    
    env.close()
    
    return {
        "winner": result,
        "black_count": black_count,
        "white_count": white_count,
        "move_count": move_count,
        "total_reward": total_reward,
        "agent1_color": "black" if agent1_is_black else "white",
    }


def display_game_statistics(results: list, agent1_name: str, agent2_name: str):
    """
    Display statistics from multiple games.
    
    Args:
        results: List of game result dictionaries
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
    """
    print("\n" + "="*60)
    print("GAME STATISTICS")
    print("="*60)
    
    num_games = len(results)
    agent1_wins = sum(1 for r in results if r["winner"] == "agent1")
    agent2_wins = sum(1 for r in results if r["winner"] == "agent2")
    draws = sum(1 for r in results if r["winner"] == "draw")
    
    avg_black = np.mean([r["black_count"] for r in results])
    avg_white = np.mean([r["white_count"] for r in results])
    avg_moves = np.mean([r["move_count"] for r in results])
    
    print(f"\nTotal games played: {num_games}")
    print(f"\n{agent1_name} wins: {agent1_wins} ({agent1_wins/num_games*100:.1f}%)")
    print(f"{agent2_name} wins: {agent2_wins} ({agent2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    print(f"\nAverage final score:")
    print(f"  ● Black: {avg_black:.1f}")
    print(f"  ○ White: {avg_white:.1f}")
    
    print(f"\nAverage moves per game: {avg_moves:.1f}")
    
    # Show individual game results if not too many
    if num_games <= 10:
        print("\nIndividual game results:")
        for i, result in enumerate(results, 1):
            winner_str = {
                "agent1": agent1_name,
                "agent2": agent2_name,
                "draw": "Draw"
            }[result["winner"]]
            
            print(f"  Game {i}: {winner_str} wins "
                  f"(● {result['black_count']} - ○ {result['white_count']}, "
                  f"{result['move_count']} moves)")
    
    print("="*60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get agent policies
    agent1_policy = get_agent_policy(
        args.agent1,
        args.agent1_checkpoint,
        "Agent 1"
    )
    agent2_policy = get_agent_policy(
        args.agent2,
        args.agent2_checkpoint,
        "Agent 2"
    )
    
    # Determine agent names for display
    agent1_name = "Agent 1"
    if args.agent1_checkpoint:
        agent1_name += " (trained)"
    else:
        agent1_name += f" ({args.agent1})"
    
    agent2_name = "Agent 2"
    if args.agent2_checkpoint:
        agent2_name += " (trained)"
    else:
        agent2_name += f" ({args.agent2})"
    
    print("\n" + "="*60)
    print("OTHELLO SPECTATOR MODE")
    print("="*60)
    print(f"{agent1_name} vs {agent2_name}")
    print(f"Number of games: {args.num_games}")
    if args.swap_colors:
        print("Color swapping: ENABLED")
    print("="*60)
    
    # Play games
    results = []
    
    for game_num in range(args.num_games):
        # Determine colors for this game
        if args.swap_colors and game_num % 2 == 1:
            agent1_is_black = False
        else:
            agent1_is_black = True
        
        if not args.no_render:
            print(f"\n{'='*60}")
            print(f"GAME {game_num + 1} of {args.num_games}")
            print(f"{'='*60}")
            print(f"{agent1_name}: {'Black' if agent1_is_black else 'White'}")
            print(f"{agent2_name}: {'White' if agent1_is_black else 'Black'}")
        
        # Play the game
        result = play_single_game(
            agent1_policy,
            agent2_policy,
            render=not args.no_render,
            delay=args.delay,
            agent1_is_black=agent1_is_black
        )
        
        results.append(result)
        
        if not args.no_render:
            winner_str = {
                "agent1": agent1_name,
                "agent2": agent2_name,
                "draw": "Draw"
            }[result["winner"]]
            
            print(f"\nGame {game_num + 1} result: {winner_str}")
            print(f"Final score: ● {result['black_count']} - ○ {result['white_count']}")
            
            if game_num < args.num_games - 1:
                print("\nStarting next game in 2 seconds...")
                time.sleep(2)
    
    # Display overall statistics
    if args.num_games > 1:
        display_game_statistics(results, agent1_name, agent2_name)


if __name__ == "__main__":
    main()
