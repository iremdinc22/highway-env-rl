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
        
        # Parking Ayarları
        if env_id == "parking-v0":
            return replace(
                config, 
                learning_rate=5e-6,
                batch_size=128,           
                ent_coef=0.0,
                total_timesteps=250_000
            )
        
        # 🔹 Intersection Ayarları
        if env_id == "intersection-v0":
            return replace(
                config, 
                total_timesteps=1_000_000, 
                learning_rate=1e-4,
                batch_size=128,
                ent_coef=0.01  # 🔹 0.0 yerine 0.01 yaparak ajanın "daha güvenli" manevralar aramasını sağlıyoruz
            )
            
            
            # 🔹 Roundabout Ayarları
        if env_id == "roundabout-v0":
            return replace(
                config, 
                total_timesteps=1_000_000,  # 👈 Mevcut 800k bitti, üzerine 200k ekliyoruz
                learning_rate=5e-5,         # 👈 Hızı düşürdük (2e-4 -> 5e-5), bildiği manevraları unutmasın
                batch_size=128,
                ent_coef=0.01               # Keşif payı kalsın ki gaza basmayı denesin
            )
            
        
        return config
    
    