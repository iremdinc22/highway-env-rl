from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass(frozen=True)
class TrainConfig:
    # ======================
    # Environment
    # ======================
    env_id: str = "highway-v0"
    seed: int = 42

    # ======================
    # Training (REALISTIC & PROJECT-READY)
    # ======================
    # Enough for learning + plots + video, not overkill
    total_timesteps: int = 300_000
    save_half_at: int = 150_000  # half-trained checkpoint

    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.0

    # ======================
    # Evaluation / Saving
    # ======================
    eval_freq: int = 25_000
    n_eval_episodes: int = 5

    save_final_name: str = "ppo_final"
    save_half_name: str = "ppo_half"

    # ======================
    # Env-specific override
    # (parking / racetrack etc.)
    # ======================
    env_config: Dict[str, Any] = field(default_factory=dict)
