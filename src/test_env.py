import gymnasium as gym
import highway_env  # noqa: F401

def main() -> None:
    env = gym.make("highway-v0", render_mode="human")
    env.reset()
    for _ in range(300):
        env.step(env.action_space.sample())
    env.close()

if __name__ == "__main__":
    main()
