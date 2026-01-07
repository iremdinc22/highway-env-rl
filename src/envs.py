from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401  (register envs)
from gymnasium.wrappers import RecordEpisodeStatistics


SUPPORTED_ENVS: Tuple[str, ...] = (
    "highway-v0",
    "merge-v0",
    "roundabout-v0",
    "intersection-v0",
    "parking-v0",
    "racetrack-v0",
)


def default_env_config() -> Dict[str, Any]:
    return {
        "duration": 40,
        "policy_frequency": 15,
        "simulation_frequency": 15,
        "lanes_count": 4,
        "vehicles_count": 50,
        "controlled_vehicles": 1,
    }


def make_env(
    env_id: str,
    seed: int,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    if env_id not in SUPPORTED_ENVS:
        raise ValueError(
            f"Unsupported env_id='{env_id}'. Supported: {list(SUPPORTED_ENVS)}"
        )

    cfg = default_env_config()
    if env_config:
        cfg.update(env_config)

    # render_mode: None | "human" | "rgb_array"
    env = gym.make(env_id, config=cfg, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)

    # Proper seeding
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
