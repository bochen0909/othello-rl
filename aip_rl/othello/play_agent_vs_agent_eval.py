"""
Agent vs Agent Othello evaluation (CLI).
"""

from __future__ import annotations

import gymnasium as gym

from aip_rl.othello.play_agent_vs_agent import (
    _policy_label,
    _select_action,
    _select_policy,
)


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate agent vs agent win rates"
    )
    parser.add_argument(
        "--black-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for Black agent",
    )
    parser.add_argument(
        "--white-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for White agent",
    )
    parser.add_argument(
        "--black-opponent",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Policy for Black if no checkpoint (default: random)",
    )
    parser.add_argument(
        "--white-opponent",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Policy for White if no checkpoint (default: random)",
    )
    parser.add_argument(
        "--cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU loading for checkpoints (default: true)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    return parser.parse_args()


def play_games(
    black_policy, white_policy, black_label: str, white_label: str, num_games: int
) -> None:
    env = gym.make(
        "Othello-v0",
        opponent="self",
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode=None,
        start_player="black",
    )

    black_wins = 0
    white_wins = 0
    draws = 0

    for _ in range(num_games):
        obs, info = env.reset()
        terminated = False

        while not terminated:
            current_player = info["current_player"]
            is_black_turn = current_player == 0
            policy = black_policy if is_black_turn else white_policy

            action = _select_action(policy, obs, info, env)
            if action < 0:
                if env.unwrapped.game.get_winner() != 3:
                    terminated = True
                    break
                break

            obs, reward, terminated, truncated, info = env.step(action)

        winner = info.get("winner", env.unwrapped.game.get_winner())
        if winner == 2:
            draws += 1
        elif winner == 0:
            black_wins += 1
        else:
            white_wins += 1

    total = max(1, num_games)
    print("\n" + "=" * 60)
    print("OTHELLO - Agent vs Agent Results")
    print("=" * 60)
    print(f"Black: {black_label}")
    print(f"White: {white_label}")
    print(f"Games: {num_games}")
    print(f"Black wins: {black_wins} ({black_wins / total:.3f})")
    print(f"White wins: {white_wins} ({white_wins / total:.3f})")
    print(f"Draws: {draws} ({draws / total:.3f})")

    env.close()


def main() -> None:
    args = parse_args()

    black_policy = _select_policy(
        args.black_checkpoint, args.black_opponent, args.cpu_only
    )
    white_policy = _select_policy(
        args.white_checkpoint, args.white_opponent, args.cpu_only
    )

    black_label = _policy_label(black_policy, args.black_checkpoint)
    white_label = _policy_label(white_policy, args.white_checkpoint)

    play_games(black_policy, white_policy, black_label, white_label, args.num_games)


if __name__ == "__main__":
    main()
