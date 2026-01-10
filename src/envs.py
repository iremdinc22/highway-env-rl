# src/envs.py
import gymnasium as gym
import highway_env  # noqa: F401

from typing import Any, Dict, Optional, Tuple

# Parking reward wrapper
from src.wrappers.parking_reward import ParkingRewardShaping

SUPPORTED_ENVS: Tuple[str, ...] = (
    "highway-v0", "merge-v0", "roundabout-v0",
    "intersection-v0", "parking-v0", "racetrack-v0",
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

    # 🔹 Base environment
    env = gym.make(env_id, config=cfg, render_mode=render_mode)

    # Reward shaping (only parking-v0)
    if env_id == "parking-v0":
        env = ParkingRewardShaping(
            env,
            w_dist=0.01,            # istersen 0.001 yapıp küçültebilirsin
            w_alignment=0.1,        # Düz durmayı ödüllendir
            collision_penalty=0.1,  # Cezayı azalt
            success_bonus=200.0,    # Başarı bonusu
            speed_threshold=0.3,    # Park edince durmasını iste
        )

    # 🔹 Seeding
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env



