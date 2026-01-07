from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import TrainConfig
from src.envs import make_env
from src.plotting import plot_rewards

# Env-specific presets:
# - default works for highway/merge/roundabout/intersection usually
# - parking/racetrack often benefit from different sim/policy frequencies & horizon
ENV_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "parking-v0": {
        "duration": 60,
        "policy_frequency": 5,
        "simulation_frequency": 15,
        "vehicles_count": 20,
        "lanes_count": 2,
        "controlled_vehicles": 1,
    },
    "racetrack-v0": {
        "duration": 80,
        "policy_frequency": 10,
        "simulation_frequency": 30,
        "vehicles_count": 30,
        "lanes_count": 4,
        "controlled_vehicles": 1,
    },
}


class EpisodeRewardLogger(BaseCallback):
    """
    Collect episodic rewards from VecEnv infos.
    Works because we wrap env with Monitor (and envs.py with RecordEpisodeStatistics).
    """
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep and "r" in ep:
                self.episode_rewards.append(float(ep["r"]))
        return True


def safe_name(s: str) -> str:
    """Make filenames safe-ish."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=TrainConfig.env_id)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    # Optional overrides if you want quick smoke runs without editing config.py
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--half-at", type=int, default=None)

    args = parser.parse_args()

    # Pick env preset (fallback to default)
    preset = ENV_PRESETS.get(args.env_id, ENV_PRESETS["default"])

    # Build config (supports env_config override)
    cfg = TrainConfig(env_id=args.env_id, seed=args.seed, env_config=preset)

    # CLI overrides (handy for smoke tests)
    total_timesteps = int(args.timesteps) if args.timesteps is not None else int(cfg.total_timesteps)
    save_half_at = int(args.half_at) if args.half_at is not None else int(cfg.save_half_at)

    # Basic sanity
    if save_half_at <= 0 or total_timesteps <= 0:
        raise ValueError("timesteps and half-at must be positive.")
    if save_half_at > total_timesteps:
        raise ValueError("half-at cannot be greater than total timesteps.")

    # Artifacts layout: separate per-env to avoid overwriting
    env_tag = safe_name(cfg.env_id)
    artifacts = Path("artifacts") / env_tag
    models_dir = artifacts / "models"
    plots_dir = artifacts / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create env (pass env_config!)
    env = make_env(cfg.env_id, cfg.seed, env_config=cfg.env_config)
    env = Monitor(env)

    # Build model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        clip_range=cfg.clip_range,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        verbose=1,
        seed=cfg.seed,
    )

    reward_logger = EpisodeRewardLogger()

    # Save names include env id + stage
    half_name = f"{cfg.save_half_name}_{env_tag}"
    final_name = f"{cfg.save_final_name}_{env_tag}"

    # Train until half, save
    model.learn(total_timesteps=save_half_at, callback=reward_logger)
    model.save(models_dir / half_name)

    # Continue to final
    remaining = total_timesteps - save_half_at
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=reward_logger)

    model.save(models_dir / final_name)

    # Plot reward curve
    plot_rewards(
        reward_logger.episode_rewards,
        plots_dir / f"reward_curve_{env_tag}.png",
    )

    env.close()

    print("\nSaved artifacts:")
    print(f"- Models: {models_dir}")
    print(f"  - {half_name}.zip")
    print(f"  - {final_name}.zip")
    print(f"- Plot:   {plots_dir / f'reward_curve_{env_tag}.png'}")


if __name__ == "__main__":
    main()
