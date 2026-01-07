from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import TrainConfig
from src.envs import make_env
from src.plotting import plot_rewards


def safe_name(s: str) -> str:
    """Make filenames safe-ish."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def load_env_presets(yaml_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load env presets from YAML.
    Expected structure:
      default: { ... }
      highway-v0: { ... }
      merge-v0: { ... }
      ...
    Returns dict[str, dict].
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Preset YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict.")

    presets: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if v is None:
            presets[k] = {}
            continue
        if not isinstance(v, dict):
            raise ValueError(f"Preset for '{k}' must be a mapping/dict.")
        presets[k] = v

    return presets


def resolve_env_config(
    env_id: str,
    presets: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge logic:
      final = presets.get("default", {}) + presets.get(env_id, {})
    env-specific overrides win.
    """
    base = dict(presets.get("default", {}))
    override = presets.get(env_id, {})
    if override:
        base.update(override)
    return base


InfoType = Union[Dict[str, Any], Sequence[Dict[str, Any]]]


class EpisodeRewardLogger(BaseCallback):
    """
    Robust logger:
    - Works for single env and VecEnv.
    - Collects episode return from info["episode"]["r"] when available.
    """

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: List[float] = []

    def _extract_episode_rewards(self, infos: InfoType) -> None:
        infos_list = [infos] if isinstance(infos, dict) else list(infos)
        for info in infos_list:
            ep = info.get("episode")
            if isinstance(ep, dict) and "r" in ep:
                self.episode_rewards.append(float(ep["r"]))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            self._extract_episode_rewards(infos)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=TrainConfig.env_id)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    # Optional overrides (great for smoke tests)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--half-at", type=int, default=None)

    # ✅ YAML presets
    parser.add_argument(
        "--preset-yaml",
        type=str,
        default=None,
        help="Path to env presets YAML (e.g., env_presets.yaml). If omitted, uses envs.py defaults.",
    )

    args = parser.parse_args()

    # Build base config
    cfg = TrainConfig(env_id=args.env_id, seed=args.seed)

    # Apply CLI overrides
    total_timesteps = int(args.timesteps) if args.timesteps is not None else int(cfg.total_timesteps)
    save_half_at = int(args.half_at) if args.half_at is not None else int(cfg.save_half_at)

    if total_timesteps <= 0:
        raise ValueError("--timesteps (or config.total_timesteps) must be > 0")
    if save_half_at <= 0:
        raise ValueError("--half-at (or config.save_half_at) must be > 0")
    if save_half_at > total_timesteps:
        raise ValueError("--half-at cannot be greater than --timesteps")

    # ✅ Resolve env_config from YAML (if provided)
    env_config: Dict[str, Any] = {}
    if args.preset_yaml:
        presets = load_env_presets(Path(args.preset_yaml))
        env_config = resolve_env_config(cfg.env_id, presets)

    # Artifacts layout: separate per-env to avoid overwriting
    env_tag = safe_name(cfg.env_id)
    artifacts = Path("artifacts") / env_tag
    models_dir = artifacts / "models"
    plots_dir = artifacts / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create env (pass env_config)
    env = make_env(cfg.env_id, cfg.seed, env_config=env_config)
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
        device="auto",
    )

    reward_logger = EpisodeRewardLogger()

    half_name = f"{cfg.save_half_name}_{env_tag}"
    final_name = f"{cfg.save_final_name}_{env_tag}"

    # Ensure we save something even if interrupted
    try:
        # Train until half, save
        model.learn(total_timesteps=save_half_at, callback=reward_logger)
        model.save(models_dir / half_name)

        # Continue to final
        remaining = total_timesteps - save_half_at
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=reward_logger)

        model.save(models_dir / final_name)

    finally:
        # Backup save (best-effort) — in case of Ctrl+C / crash
        try:
            model.save(models_dir / f"{final_name}_backup")
        except Exception:
            pass
        env.close()

    # Plot reward curve (if we captured any)
    if reward_logger.episode_rewards:
        plot_rewards(
            reward_logger.episode_rewards,
            plots_dir / f"reward_curve_{env_tag}.png",
        )
    else:
        print("Warning: No episode rewards captured. (Still saved models.)")

    print("\nSaved artifacts:")
    print(f"- Models: {models_dir}")
    print(f"  - {half_name}.zip")
    print(f"  - {final_name}.zip")
    print(f"  - {final_name}_backup.zip (if created)")
    print(f"- Plot:   {plots_dir / f'reward_curve_{env_tag}.png'}")
    if args.preset_yaml:
        print(f"- Preset YAML: {args.preset_yaml}")
        print(f"- Effective env_config for {cfg.env_id}: {env_config}")


if __name__ == "__main__":
    main()
