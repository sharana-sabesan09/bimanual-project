"""
Random agent: samples uniform random actions in [-1, 1] every step.
Usage: python scripts/random_agent.py --task <gym_env_id>
"""

import argparse
import importlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str, required=True,
                        help="Gym env ID, e.g. BallBalance-DualArm-v0")
    parser.add_argument("-n", "--num_envs", type=int, default=1)
    args = parser.parse_args()

    import gymnasium as gym
    import source  # noqa: F401

    env_spec = gym.spec(args.task)
    EnvClass = getattr(
        importlib.import_module(env_spec.entry_point.rsplit(":", 1)[0]),
        env_spec.entry_point.rsplit(":", 1)[1],
    )

    gs.init(backend=gs.cpu)
    env = EnvClass(show_viewer=True, n_envs=args.num_envs)
    obs, _ = env.reset()

    while True:
        action = np.random.uniform(-1.0, 1.0, (env.n_envs, env.action_space.shape[0])).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)


if __name__ == "__main__":
    main()
