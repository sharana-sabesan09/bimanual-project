"""
Random agent: samples uniform random actions in [-1, 1] every step.
Usage: python scripts/random_agent.py --env {single,dual}
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["single", "dual"], required=True)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    if args.env == "single":
        from source.tasks.single.env import SingleArmBallBalanceEnv
        env = SingleArmBallBalanceEnv(show_viewer=True, n_envs=1)
    else:
        from source.tasks.double.env import DualArmBallBalanceEnv
        env = DualArmBallBalanceEnv(show_viewer=True, n_envs=1)

    obs = env.reset()

    while True:
        action = np.random.uniform(-1.0, 1.0, (1, env.n_arm_dofs)).astype(np.float32)
        obs, reward, done, _ = env.step(action)
        if done.any():
            obs = env.reset()


if __name__ == "__main__":
    main()
