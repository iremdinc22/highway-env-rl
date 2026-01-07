from __future__ import annotations

import argparse

from src.envs import SUPPORTED_ENVS, make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    results = []
    for env_id in SUPPORTED_ENVS:
        try:
            env = make_env(env_id, args.seed)
            obs, _ = env.reset(seed=args.seed)
            ok_steps = 0
            for _ in range(args.steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ok_steps += 1
                if terminated or truncated:
                    break
            env.close()
            results.append((env_id, "OK", ok_steps, ""))
        except Exception as e:
            results.append((env_id, f"FAIL: {type(e).__name__}", 0, str(e)))

    print("Smoke Test Results")
    for env_id, status, steps, msg in results:
        line = f"- {env_id:16s} | {status:20s} | steps: {steps}"
        if msg:
            line += f" | {msg}"
        print(line)


if __name__ == "__main__":
    main()
