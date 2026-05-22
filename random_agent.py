"""
Random agent: samples uniform random joint position targets each step.
Actions are drawn from [-1, 1] rad, roughly spanning the wrist/elbow range.
"""

import numpy as np
from ball_balance_env import BallBalanceEnv

ACTION_LOW  = -1.0
ACTION_HIGH =  1.0

if __name__ == "__main__":
    env = BallBalanceEnv(show_viewer=True, n_envs=1)
    obs = env.reset()

    while True:
        action = np.random.uniform(ACTION_LOW, ACTION_HIGH, (1, env.n_arm_dofs)).astype(np.float32)
        obs, reward, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
