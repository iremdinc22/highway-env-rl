from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Dict as DictSpace

from src.config import TrainConfig
from src.envs import make_env
from src.plotting import plot_rewards


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def load_env_presets(yaml_path: Path) -> Dict[str, Dict[str, Any]]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Preset YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dict.")

    return {k: v or {} for k, v in data.items() if isinstance(k, str)}


def resolve_env_config(env_id: str, presets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(presets.get("default", {}))
    base.update(presets.get(env_id, {}))
    return base


InfoType = Union[Dict[str, Any], Sequence[Dict[str, Any]]]


class EpisodeRewardLogger(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        infos_list = [infos] if isinstance(infos, dict) else infos or []
        for info in infos_list:
            ep = info.get("episode")
            if ep and "r" in ep:
                self.episode_rewards.append(float(ep["r"]))
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=TrainConfig.env_id)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--half-at", type=int, default=None)
    parser.add_argument("--preset-yaml", type=str, default=None)

    args = parser.parse_args()

    # 🔹 KRİTİK DEĞİŞİKLİK: Akıllı yapılandırıcıyı çağırıyoruz.
    # Bu sayede parking-v0 için 600k timesteps ve özel parametreler devreye girer.
    cfg = TrainConfig.get_config(env_id=args.env_id, seed=args.seed)

    # Komut satırından manuel değer girilmediyse, config'deki (get_config ile belirlenen) değeri kullan
    total_timesteps = args.timesteps if args.timesteps is not None else cfg.total_timesteps
    save_half_at = args.half_at if args.half_at is not None else cfg.save_half_at

    env_config: Dict[str, Any] = {}
    if args.preset_yaml:
        presets = load_env_presets(Path(args.preset_yaml))
        env_config = resolve_env_config(cfg.env_id, presets)

    env_tag = safe_name(cfg.env_id)
    artifacts = Path("artifacts") / env_tag
    models_dir = artifacts / "models"
    plots_dir = artifacts / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ENV
    env = make_env(cfg.env_id, cfg.seed, env_config=env_config)
    env = Monitor(env)

    # AUTO POLICY
    obs_space = env.observation_space
    policy = "MultiInputPolicy" if isinstance(obs_space, DictSpace) else "MlpPolicy"
    print(f"[INFO] env_id={cfg.env_id}")
    print(f"[INFO] total_timesteps={total_timesteps}")
    print(f"[INFO] batch_size={cfg.batch_size}, ent_coef={cfg.ent_coef}")
    print(f"[INFO] selected policy={policy}")

    # MODEL
    model = PPO(
        policy,
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

    try:
        # 1. Aşama: Yarısına kadar eğit ve kaydet
        if save_half_at > 0:
            print(f"[INFO] Training phase 1: 0 to {save_half_at}")
            model.learn(total_timesteps=save_half_at, callback=reward_logger)
            model.save(models_dir / half_name)

        # 2. Aşama: Kalan kısmı tamamla
        remaining = total_timesteps - save_half_at
        if remaining > 0:
            print(f"[INFO] Training phase 2: {save_half_at} to {total_timesteps}")
            model.learn(total_timesteps=remaining, callback=reward_logger)

        model.save(models_dir / final_name)

    finally:
        try:
            model.save(models_dir / f"{final_name}_backup")
        except Exception:
            pass
        env.close()

    # Ödül grafiğini çiz
    if reward_logger.episode_rewards:
        plot_rewards(
            reward_logger.episode_rewards,
            plots_dir / f"reward_curve_{env_tag}.png",
        )

    print(f"\n[SUCCESS] Training finished for {cfg.env_id}.")
    print(f"- Final Model: {models_dir / final_name}.zip")
        
  
if __name__ == "__main__":
    main()