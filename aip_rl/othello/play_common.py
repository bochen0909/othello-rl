"""
Common utilities for playing Othello against an agent.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Union

import gymnasium as gym

# Ensure Othello env is registered with Gymnasium
import aip_rl.othello  # noqa: F401

OpponentPolicy = Union[str, Callable]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build a common argument parser for play scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained agent checkpoint",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "greedy"],
        default="random",
        help="Built-in opponent policy (default: random)",
    )
    parser.add_argument(
        "--human-color",
        type=str,
        choices=["black", "white"],
        default="black",
        help="Color for human player (default: black)",
    )
    parser.add_argument(
        "--cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU loading for checkpoints (default: true)",
    )
    return parser


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve a checkpoint directory or a parent directory containing checkpoints."""
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    def is_checkpoint_dir(dir_path: str) -> bool:
        return (
            os.path.isfile(os.path.join(dir_path, "rllib_checkpoint.json"))
            or os.path.isfile(os.path.join(dir_path, "algorithm_state.pkl"))
        )

    if os.path.isdir(path):
        base = os.path.basename(path)
        if base.startswith("checkpoint_") or is_checkpoint_dir(path):
            return path

        candidates = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if not os.path.isdir(full_path):
                continue
            if name.startswith("checkpoint_") or is_checkpoint_dir(full_path):
                candidates.append(full_path)

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint directories found under: {path}"
            )

        for candidate in candidates:
            if os.path.basename(candidate) == "final":
                return candidate

        def checkpoint_key(p: str) -> int:
            name = os.path.basename(p)
            if name.startswith("checkpoint_") or name.startswith("iter_"):
                try:
                    return int(name.split("_", 1)[1])
                except (IndexError, ValueError):
                    return -1
            return -1

        candidates.sort(key=checkpoint_key)
        return candidates[-1]

    return path


def load_trained_agent(checkpoint_path: str, cpu_only: bool) -> Callable:
    """Load a trained agent from checkpoint."""
    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.algorithms.algorithm import get_checkpoint_info
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        from aip_rl.othello.models import OthelloCNN

        def register_custom_model() -> None:
            try:
                ModelCatalog.register_custom_model("othello_cnn", OthelloCNN)
            except ValueError:
                pass

        checkpoint_path = resolve_checkpoint_path(checkpoint_path)

        def env_creator(env_config):
            import aip_rl.othello  # noqa: F401 - ensure env registered in workers
            register_custom_model()
            return gym.make("Othello-v0", **env_config)

        register_env("Othello-v0", env_creator)
        register_custom_model()

        ray.init(ignore_reinit_error=True)

        def force_cpu_config(config):
            if isinstance(config, dict):
                config["num_gpus"] = 0
                config["num_gpus_per_learner"] = 0
                config["num_gpus_per_env_runner"] = 0
                config["num_workers"] = 0
                if "num_rollout_workers" in config:
                    if "num_env_runners" not in config:
                        config["num_env_runners"] = config["num_rollout_workers"]
                    config.pop("num_rollout_workers", None)
                config["num_env_runners"] = 0
                return config
            if isinstance(config, AlgorithmConfig):
                config.num_gpus = 0
                config.num_gpus_per_learner = 0
                config.num_gpus_per_env_runner = 0
                config.num_workers = 0
                if hasattr(config, "num_env_runners"):
                    config.num_env_runners = 0
            return config

        checkpoint_info = get_checkpoint_info(checkpoint_path)
        state = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info=checkpoint_info,
            policy_mapping_fn=AlgorithmConfig.DEFAULT_POLICY_MAPPING_FN,
        )
        if cpu_only:
            state["config"] = force_cpu_config(state.get("config"))
        print("Loaded config for play:")
        print(state.get("config"))
        algo = Algorithm.from_state(state)

        def agent_policy(obs):
            return algo.compute_single_action(obs, explore=False)

        print(f"Loaded trained agent from: {checkpoint_path}")
        return agent_policy

    except ImportError:
        print("Error: Ray RLlib is required to load trained agents.")
        print("Install with: pip install ray[rllib]")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


def select_opponent(args) -> OpponentPolicy:
    """Pick opponent policy based on parsed args."""
    if args.checkpoint:
        return load_trained_agent(args.checkpoint, args.cpu_only)
    return args.opponent
