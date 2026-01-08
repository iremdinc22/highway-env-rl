from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrainConfig:
    # environment
    env_id: str = "highway-v0"
    seed: int = 42

    # training (project-ready)
    total_timesteps: int = 300_000
    save_half_at: int = 150_000

    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.0

    # evaluation
    eval_freq: int = 25_000
    n_eval_episodes: int = 5

    # saving
    save_final_name: str = "ppo_final"
    save_half_name: str = "ppo_half"
    
    # --- YENİ: Parking İçin Akıllı Yapılandırıcı ---
    @classmethod
    def get_config(cls, env_id: str, **kwargs) -> TrainConfig:
        config = cls(env_id=env_id, **kwargs)
        
        if env_id == "parking-v0":
            return replace(
                config, 
                total_timesteps=1_000_000, # En az 1 milyon yapmalısın
                save_half_at=500_000,
                batch_size=256,            # 128'den 256'ya çıkardık (daha kararlı öğrenme)
                ent_coef=0.005,            # 0.02'den 0.005'e düşürdük (titremeyi azaltır)
                learning_rate=1e-4         # 5e-4'ten 1e-4'e düşürdük (hassas park için)
            )
        return config

