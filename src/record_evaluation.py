from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import highway_env  # noqa: F401  (register envs)
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.envs import default_env_config


def make_video_env(env_id: str, seed: int, video_folder: Path) -> gym.Env:
    cfg = default_env_config()

    env = gym.make(env_id, config=cfg, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(video_folder), episode_trigger=lambda ep: True)

    # highway-env specific hint for correct video recording
    if hasattr(env.unwrapped, "set_record_video_wrapper"):
        env.unwrapped.set_record_video_wrapper(env)

    env.reset(seed=seed)
    return env


def _latest_video_file(out_dir: Path) -> Path:
    candidates = []
    candidates.extend(out_dir.glob("*.mp4"))
    candidates.extend(out_dir.glob("*.webm"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No video produced. Check render_mode and RecordVideo setup.")
    return candidates[0]


def record_episode(
    env_id: str,
    seed: int,
    out_dir: Path,
    stage_name: str,
    model_path: Optional[Path] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_video_env(env_id, seed, out_dir)
    obs, _ = env.reset(seed=seed)

    model = PPO.load(model_path, device="auto") if model_path is not None else None

    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

    latest = _latest_video_file(out_dir)
    final_path = out_dir / f"{stage_name}{latest.suffix}"
    latest.rename(final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half-model", type=str, default="artifacts/models/ppo_half.zip")
    parser.add_argument("--final-model", type=str, default="artifacts/models/ppo_final.zip")
    args = parser.parse_args()

    videos_dir = Path("artifacts/videos") / args.env_id

    p1 = record_episode(args.env_id, args.seed, videos_dir, "stage1_untrained", None)
    p2 = record_episode(args.env_id, args.seed, videos_dir, "stage2_half", Path(args.half_model))
    p3 = record_episode(args.env_id, args.seed, videos_dir, "stage3_full", Path(args.final_model))

    print("Saved:")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
