from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401
import yaml
from stable_baselines3 import PPO
from stable_baselines3 import SAC


def load_env_presets(yaml_path: Path) -> Dict[str, Dict[str, Any]]:
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


def resolve_env_config(env_id: str, presets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(presets.get("default", {}))
    override = presets.get(env_id, {})
    if override:
        base.update(override)
    return base


def normalize_model_path(model_path: str) -> str:
    """
    Make SB3 load robust:
    - If user passes *.zip, strip it to avoid *.zip.zip issues.
    """
    path = model_path.strip()
    if path.endswith(".zip"):
        path = path[:-4]
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="highway-v0")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--preset-yaml",
        default=None,
        help="Path to env presets YAML (same file used in train).",
    )
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--keep-open", type=float, default=5.0)
    args = parser.parse_args()

    # ✅ Load env config from YAML (same as training)
    env_config: Dict[str, Any] = {}
    if args.preset_yaml:
        presets = load_env_presets(Path(args.preset_yaml))
        env_config = resolve_env_config(args.env_id, presets)

    # ✅ Create environment
    env = gym.make(
        args.env_id,
        render_mode="rgb_array",
        config=env_config,
    )

    # 🎥 VIDEO RECORDING (env-id bazlı klasör)
    video_dir = Path("artifacts") / "videos" / args.env_id
    video_dir.mkdir(parents=True, exist_ok=True)

    env = RecordVideo(
        env,
        video_folder=video_dir.as_posix(),
        episode_trigger=lambda ep: ep == 0,  # sadece 1 episode kaydet
        name_prefix=args.env_id,
    )

    # Algoritmayı dosya adına göre otomatik seç ve yükle
    model_path = normalize_model_path(args.model_path)
    
    if "sac" in model_path.lower():
        print(f"[INFO] SAC modeli tespit edildi, SAC ile yükleniyor...")
        model = SAC.load(model_path)
    elif "ppo" in model_path.lower():
        print(f"[INFO] PPO modeli tespit edildi, PPO ile yükleniyor...")
        model = PPO.load(model_path)
    else:
        # Eğer dosya adında ikisi de yoksa varsayılan olarak PPO dene
        print(f"[WARNING] Algoritma tespit edilemedi, varsayılan PPO deneniyor...")
        model = PPO.load(model_path)
        
        
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(args.sleep)

    print("Episode finished, keeping window open...")
    time.sleep(args.keep_open)
    env.close()


if __name__ == "__main__":
    main()

