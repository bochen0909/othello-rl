"""
Human vs Agent Othello game (CLI).
"""

from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from aip_rl.othello.play_common import build_arg_parser, select_opponent


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

            if user_input.lower() == "q":
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
            print(
                "Invalid input. Please enter a number between 0 and 63, or 'q' to quit."
            )
        except KeyboardInterrupt:
            print("\nQuitting game...")
            sys.exit(0)


def play_game(opponent_policy, human_color: str = "black") -> None:
    """
    Play a game of Othello with human vs agent.

    Args:
        opponent_policy: Opponent policy ("random", "greedy", or callable)
        human_color: Color for human player ("black" or "white")
    """
    env = gym.make(
        "Othello-v0",
        opponent=opponent_policy,
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode="human",
        start_player=human_color,
    )

    human_is_black = human_color == "black"

    print("\n" + "=" * 60)
    print("OTHELLO - Human vs Agent")
    print("=" * 60)
    print(f"Human plays as: {human_color.upper()}")
    print(
        f"Opponent: {opponent_policy if isinstance(opponent_policy, str) else 'trained agent'}"
    )
    print("\nBoard positions are numbered 0-63:")
    print("  Row 0: 0-7")
    print("  Row 1: 8-15")
    print("  ...")
    print("  Row 7: 56-63")
    print("\nValid moves are marked with '*' on the board.")
    print("=" * 60 + "\n")

    obs, info = env.reset()

    print("Initial board:")
    env.render()

    terminated = False
    move_count = 0

    while not terminated:
        move_count += 1
        current_player = info["current_player"]
        is_human_turn = (current_player == 0 and human_is_black) or (
            current_player == 1 and not human_is_black
        )

        print(f"\n{'=' * 60}")
        print(f"Move {move_count}")
        print(f"{'=' * 60}")

        if is_human_turn:
            print("Your turn!")
            valid_moves = info["action_mask"]

            if not np.any(valid_moves):
                print("No valid moves. Turn passed to opponent.")
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
            print("Opponent is thinking...")
            pass

    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    unwrapped_env = env.unwrapped
    winner = info.get("winner", unwrapped_env.game.get_winner())
    black_count, white_count = unwrapped_env.game.get_piece_counts()

    print("\nFinal score:")
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


def parse_args():
    """Parse command line arguments."""
    parser = build_arg_parser(
        description="Play Othello against an agent or built-in opponent"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    opponent_policy = select_opponent(args)
    play_game(opponent_policy, args.human_color)


if __name__ == "__main__":
    main()
