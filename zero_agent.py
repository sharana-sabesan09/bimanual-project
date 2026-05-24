"""
Zero agent: holds both arms at their working poses every step.
"""

import numpy as np
import genesis as gs
from ball_balance_env import BallBalanceEnv, RIGHT_ARM_HOLD_POS, LEFT_ARM_HOLD_POS

if __name__ == "__main__":
    gs.init(backend=gs.cpu)
    env = BallBalanceEnv(show_viewer=True, n_envs=1)
    obs = env.reset()
    action = np.concatenate([RIGHT_ARM_HOLD_POS, LEFT_ARM_HOLD_POS])[None, :]  # (1, 14)

    while True:
        obs, reward, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
