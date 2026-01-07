import argparse
import time
import gymnasium as gym
import highway_env  # noqa
from stable_baselines3 import PPO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="highway-v0")
    p.add_argument("--model-path", required=True)
    args = p.parse_args()

    env = gym.make(
        args.env_id,
        render_mode="human",
        config={
            "duration": 40,
            "policy_frequency": 15,
            "simulation_frequency": 15,
            "lanes_count": 4,
            "vehicles_count": 50,
        },
    )
    
    path = args.model_path
    if path.endswith(".zip"):
        path = path[:-4]

    model = PPO.load(path)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.05)

    print("Episode finished, keeping window open...")
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    main()
