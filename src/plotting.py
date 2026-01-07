from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="valid")


def plot_rewards(
    reward_per_episode: List[float],
    out_path: Path,
    ma_window: Optional[int] = 25,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.array(reward_per_episode, dtype=float)
    x = np.arange(1, len(y) + 1)

    plt.figure()
    plt.plot(x, y, label="Episode reward")

    if ma_window is not None and len(y) >= 2:
        y_ma = moving_average(y, ma_window)
        x_ma = np.arange(len(y_ma)) + (len(y) - len(y_ma)) + 1
        plt.plot(x_ma, y_ma, label=f"Moving avg (w={min(ma_window, len(y))})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
