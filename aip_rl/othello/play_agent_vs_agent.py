"""
Agent vs Agent Othello game (CLI).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from aip_rl.othello.play_common import load_trained_agent
from aip_rl.othello.engines import get_available_engines, get_engine_opponent


def parse_args():
    """Parse command line arguments."""
    import argparse

    available_engines = get_available_engines()
    engines_str = ", ".join(available_engines)

    parser = argparse.ArgumentParser(description="Play Othello with agent vs agent")
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
        choices=["random", "greedy"] + available_engines,
        default="random",
        help=f"Policy for Black if no checkpoint (default: random). Available engines: {engines_str}",
    )
    parser.add_argument(
        "--white-opponent",
        type=str,
        choices=["random", "greedy"] + available_engines,
        default="random",
        help=f"Policy for White if no checkpoint (default: random). Available engines: {engines_str}",
    )
    parser.add_argument(
        "--cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU loading for checkpoints (default: true)",
    )
    return parser.parse_args()


def _policy_label(policy, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return f"trained agent ({checkpoint_path})"
    if isinstance(policy, str):
        return policy
    # Check if it's an engine opponent (has __name__ attribute starting with "engine_")
    if hasattr(policy, "__name__") and policy.__name__.startswith("engine_"):
        engine_name = policy.__name__.replace("engine_", "")
        return f"{engine_name} engine"
    return "trained agent"


def _select_policy(checkpoint_path: str | None, fallback: str, cpu_only: bool):
    if checkpoint_path:
        return load_trained_agent(checkpoint_path, cpu_only)

    # Check if it's an engine opponent
    if fallback in get_available_engines():
        return get_engine_opponent(fallback)

    # Otherwise treat it as a built-in policy (random, greedy)
    return fallback


def _select_action(policy, obs, info, env) -> int:
    valid_moves = info["action_mask"]
    valid_indices = np.where(valid_moves)[0]
    if len(valid_indices) == 0:
        return -1

    if isinstance(policy, str):
        if policy == "random":
            return int(np.random.choice(valid_indices))
        if policy == "greedy":
            return int(env.unwrapped._get_greedy_move())

    action = int(policy(obs))
    if not valid_moves[action]:
        action = int(np.random.choice(valid_indices))
    return action


def play_game(black_policy, white_policy, black_label: str, white_label: str) -> None:
    env = gym.make(
        "Othello-v0",
        opponent=white_policy,
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode="human",
        start_player="black",
    )

    print("\n" + "=" * 60)
    print("OTHELLO - Agent vs Agent")
    print("=" * 60)
    print(f"Black: {black_label}")
    print(f"White: {white_label}")
    print("=" * 60 + "\n")

    obs, info = env.reset()
    env.render()

    terminated = False

    while not terminated:
        action = _select_action(black_policy, obs, info, env)
        if action < 0:
            if env.unwrapped.game.get_winner() != 3:
                terminated = True
                break
            print("No valid moves for Black; ending to avoid stall.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Black played: {action} (row {action // 8}, col {action % 8})")
        env.render()

    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    winner = info.get("winner", env.unwrapped.game.get_winner())
    black_count, white_count = env.unwrapped.game.get_piece_counts()

    print("\nFinal score:")
    print(f"  ● Black: {black_count}")
    print(f"  ○ White: {white_count}")

    if winner == 2:
        print("\nResult: DRAW")
    elif winner == 0:
        print("\nResult: BLACK WINS")
    else:
        print("\nResult: WHITE WINS")

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

    play_game(black_policy, white_policy, black_label, white_label)


if __name__ == "__main__":
    main()
