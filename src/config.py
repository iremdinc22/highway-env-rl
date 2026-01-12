from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Union

@dataclass(frozen=True)
class TrainConfig:
    # Genel Ayarlar
    env_id: str = "highway-v0"
    seed: int = 42
    algorithm: str = "PPO"  # Varsayılan PPO

    # Eğitim Parametreleri
    total_timesteps: int = 300_000
    save_half_at: int = 150_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    ent_coef: Union[float, str] = 0.0

    # PPO'ya Özel (SAC'da yoksayılır)
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    gae_lambda: float = 0.95

    # SAC'a Özel (PPO'da yoksayılır) 👈 BURASI KRİTİK
    buffer_size: int = 1_000_000
    learning_starts: int = 5000     # 👈 100 yerine 5000 yaptık (Titreme için daha güvenli)
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1

    # Kayıt ve Değerlendirme
    eval_freq: int = 25_000
    n_eval_episodes: int = 5
    save_final_name: str = "ppo_final"
    save_half_name: str = "ppo_half"
    
    @classmethod
    def get_config(cls, env_id: str, **kwargs) -> TrainConfig:
        config = cls(env_id=env_id, **kwargs)
        
        # 🔹 Racetrack İçin Pürüzsüz SAC Konfigürasyonu
        if env_id == "racetrack-v0":
            return replace(
                config, 
                algorithm="SAC",            # Algoritmayı SAC yaptık
                total_timesteps=1_000_000,  
                learning_rate=3e-4,         
                batch_size=256,             # SAC için daha stabil
                buffer_size=1_000_000,       # 👈 100k yerine 1M yaparsan trafik senaryolarını daha iyi hatırlar
                learning_starts=5000,       # Önce pürüzsüz veri toplasın
                tau=0.005,                  
                ent_coef="auto",            # Pürüzsüzlük için otomatik entropi
                save_final_name="sac_final"
            )
            
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
                total_timesteps=1_000_000,  #
                learning_rate=5e-5,         # 👈 Hızı daha da düşürdük (5e-5 -> 3e-5), usta manevraları asla bozulmasın
                batch_size=128,
                ent_coef=0.01  # 👈 Keşif payını azalttık, artık öğrendiği yola (asfalta) sadık kalsın
            )
        return config
    

    