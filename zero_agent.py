"""
Zero agent: passes all-zero joint position targets to the right arm every step.
The arm will collapse from the hold pose toward zero (shoulder down, elbow straight).
"""

import numpy as np
import genesis as gs
from ball_balance_env import BallBalanceEnv

if __name__ == "__main__":
    gs.init(backend=gs.cpu)
    env = BallBalanceEnv(show_viewer=True, n_envs=1)
    obs = env.reset()
    action = np.zeros((1, env.n_arm_dofs), dtype=np.float32)

    while True:
        obs, reward, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
