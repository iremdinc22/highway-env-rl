from __future__ import annotations
from dataclasses import dataclass


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

