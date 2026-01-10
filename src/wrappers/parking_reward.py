from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple
from dataclasses import dataclass, replace

class ParkingRewardShaping(gym.Wrapper):
    """
    Parking-v0 için geliştirilmiş ödül wrapper'ı.
    - Hedefe olan mesafe (L2 norm)
    - Aracın hedefle olan açısal hizalanması
    - Çarpışma cezası
    - Başarı bonusu (belirli hızın altında ve doğru konumda)
    """

    def __init__(
        self,
        env: gym.Env,
        w_dist: float = 0.0005,      # 0.2 -> 0.0005 yapıldı
        w_alignment: float = 0.0005, # 0.2 -> 0.0005 yapıldı
        collision_penalty: float = 0.5,
        success_bonus: float = 50.0, # Sadece park ederse ödül alsın
        speed_threshold: float = 0.5
    ):
        super().__init__(env)
        self.w_dist = w_dist
        self.w_alignment = w_alignment
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.speed_threshold = speed_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Parking-v0 gözlemi genellikle sözlük yapısındadır:
        # achived_goal: Aracın şu anki hali [x, y, vx, vy, cos_h, sin_h]
        # desired_goal: Hedeflenen hal [x, y, vx, vy, cos_h, sin_h]
        
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        # 1. Mesafe Ödülü (Negatif L2 norm - ne kadar yakınsa o kadar iyi)
        dist = np.linalg.norm(achieved[:2] - desired[:2])
        reward_dist = -self.w_dist * dist

        # 2. Hizalanma Ödülü (Heading alignment)
        # Cosine ve Sine değerleri üzerinden yön farkına bakıyoruz
        # achieved[4:6] -> [cos, sin], desired[4:6] -> [cos, sin]
        alignment = np.dot(achieved[4:6], desired[4:6])
        reward_align = self.w_alignment * alignment

        # 3. Hız Kontrolü (Park ederken yavaşlama teşviki)
        velocity = np.linalg.norm(achieved[2:4])
        
        # Toplam Ödülü Birleştirme
        shaped_reward = reward_dist + reward_align

        # 4. Çarpışma Cezası
        if info.get("crashed", False):
            shaped_reward -= self.collision_penalty

        # 5. Başarı Bonusu
        # Mesafe çok azsa ve araç durma noktasındaysa (speed < threshold)
        is_success = dist < 0.1 and velocity < self.speed_threshold
        if is_success:
            shaped_reward += self.success_bonus
            # İsteğe bağlı: Başarı durumunda bölümü bitirebiliriz
            # terminated = True 

        return obs, shaped_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)