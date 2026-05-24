import sys
import os
import numpy as np
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ball_balance_env import BallBalanceEnv, RIGHT_ARM_HOLD_POS, LEFT_ARM_HOLD_POS

gs.init(backend=gs.cpu)
env = BallBalanceEnv(show_viewer=True, n_envs=1)

# Hold both arms at their working poses so the tray stays up
action = np.concatenate([RIGHT_ARM_HOLD_POS, LEFT_ARM_HOLD_POS])[None, :]  # (1, 14)
while True:
    obs, reward, done, _ = env.step(action)
    if done.any():
        env.reset()
