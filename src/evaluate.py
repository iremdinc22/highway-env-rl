from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.envs import make_env


def run_episode(model: PPO, env_id: str, seed: int) -> float:
    env = make_env(env_id, seed)
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

    env.close()
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--model-path", type=str, default="artifacts/models/ppo_final.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model = PPO.load(model_path, device="auto")

    rewards = [run_episode(model, args.env_id, args.seed + i) for i in range(args.episodes)]

    print(f"Env: {args.env_id}")
    print(f"Mean reward over {args.episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
