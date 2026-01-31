"""
Compute Elo ratings for Othello agents in a folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, Tuple

import gymnasium as gym

from aip_rl.othello.play_agent_vs_agent import _select_action, _select_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute/update Elo ratings for Othello agents"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="zoo",
        help="Folder containing agent checkpoints (default: zoo)",
    )
    parser.add_argument(
        "--cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU loading for checkpoints (default: true)",
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=50,
        help="Games per side in each matchup (default: 50)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of matchup rounds to run (default: 1)",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor (default: 32)",
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=1000.0,
        help="Initial rating for new agents (default: 1000)",
    )
    return parser.parse_args()


def list_checkpoint_agents(folder: str) -> Dict[str, str]:
    agents: Dict[str, str] = {}
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():
                agents[entry.name] = entry.path
    return dict(sorted(agents.items(), key=lambda x: x[0]))


def load_ratings(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "ratings" in data:
        data = data["ratings"]
    if not isinstance(data, dict):
        return {}
    ratings: Dict[str, float] = {}
    for name, rating in data.items():
        if isinstance(name, str) and isinstance(rating, (int, float)):
            ratings[name] = float(rating)
    return ratings


def backup_file(path: str) -> None:
    if not os.path.exists(path):
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{path}.bak-{timestamp}"
    shutil.copy2(path, backup_path)


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_elo(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a
    r_a_new = r_a + k * (score_a - e_a)
    r_b_new = r_b + k * ((1.0 - score_a) - e_b)
    return r_a_new, r_b_new


def play_match(
    black_policy,
    white_policy,
    num_games: int,
) -> Iterable[int]:
    env = gym.make(
        "Othello-v0",
        opponent=white_policy,
        reward_mode="sparse",
        invalid_move_mode="error",
        render_mode=None,
        start_player="black",
    )

    for _ in range(num_games):
        obs, info = env.reset()
        terminated = False

        while not terminated:
            action = _select_action(black_policy, obs, info, env)
            if action < 0:
                if env.unwrapped.game.get_winner() != 3:
                    terminated = True
                    break
                break
            obs, reward, terminated, truncated, info = env.step(action)

        winner = info.get("winner", env.unwrapped.game.get_winner())
        yield winner

    env.close()


def score_from_winner(winner: int) -> float:
    if winner == 2:
        return 0.5
    if winner == 0:
        return 1.0
    return 0.0


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"Folder not found: {args.folder}")

    checkpoint_agents = list_checkpoint_agents(args.folder)
    agent_names = list(checkpoint_agents.keys()) + ["random", "greedy"]

    elo_path = os.path.join(args.folder, "elo.json")
    ratings = load_ratings(elo_path)

    ratings = {name: rating for name, rating in ratings.items() if name in agent_names}

    new_agents = [name for name in agent_names if name not in ratings]
    for name in new_agents:
        ratings[name] = float(args.initial_rating)

    policies = {}
    for name in agent_names:
        if name in ("random", "greedy"):
            policies[name] = name
            continue
        checkpoint_path = checkpoint_agents[name]
        policies[name] = _select_policy(checkpoint_path, "random", args.cpu_only)

    pairs = [
        (a, b)
        for a, b in combinations(agent_names, 2)
        if a in new_agents or b in new_agents
    ]

    print(f"Evaluating {len(agent_names)} agents ({len(checkpoint_agents)} from checkpoints) "
          f"over {len(pairs)} pairing(s) with {args.games_per_side} games per side.")
    if not pairs:
        print("No new opponents to compare; skipping matchup simulation.")
        backup_file(elo_path)
        with open(elo_path, "w", encoding="utf-8") as f:
            json.dump(ratings, f, indent=2, sort_keys=True)
            f.write("\n")
        return

    total_pairs = len(pairs)
    for round_index in range(args.rounds):
        print(f"Round {round_index + 1}/{args.rounds}")
        for pair_index, (black_name, white_name) in enumerate(pairs, start=1):
            print(f"  Match {pair_index}/{total_pairs}: {black_name} (black) vs {white_name} (white)")
            black_policy = policies[black_name]
            white_policy = policies[white_name]
            for winner in play_match(
                black_policy, white_policy, args.games_per_side
            ):
                score_black = score_from_winner(winner)
                r_b = ratings[black_name]
                r_w = ratings[white_name]
                r_b, r_w = update_elo(r_b, r_w, score_black, args.k_factor)
                ratings[black_name] = r_b
                ratings[white_name] = r_w

            for winner in play_match(
                policies[white_name], policies[black_name], args.games_per_side
            ):
                score_black = score_from_winner(winner)
                r_b = ratings[white_name]
                r_w = ratings[black_name]
                r_b, r_w = update_elo(r_b, r_w, score_black, args.k_factor)
                ratings[white_name] = r_b
                ratings[black_name] = r_w

            print(
                f"    Updated ratings: {black_name} {ratings[black_name]:.1f}, "
                f"{white_name} {ratings[white_name]:.1f}"
            )

    backup_file(elo_path)
    with open(elo_path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
