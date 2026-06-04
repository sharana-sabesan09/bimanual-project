"""
Manual step (no-grasp variant): works with both single-arm and dual-arm envs.
Action is always zero — just lets physics run and the ball fall naturally.

Usage:
    python scripts/manual_step_no_grasp.py --task BallBalance-SingleArm-v0
    python scripts/manual_step_no_grasp.py --task BallBalance-DualArm-v0

Controls:
    SPACE    — advance one sim step
    G        — reset the environment
    Ctrl-C / close window — quit
"""

import argparse
import importlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import genesis as gs
from genesis.vis.keybindings import Keybind, Key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str, required=True,
                        help="Gym env ID, e.g. BallBalance-SingleArm-v0")
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

    action_dim = env.action_space.shape[0]
    action = np.zeros((env.n_envs, action_dim), dtype=np.float32)
    step = [0]

    def do_step():
        obs, reward, terminated, truncated, _ = env.step(action)
        step[0] += 1
        print(f"step {step[0]:5d}  reward={reward[0]:.4f}  "
              f"terminated={terminated[0].item()}  truncated={truncated[0].item()}")

    def do_reset():
        env.reset()
        step[0] = 0
        print("--- reset ---")

    env.scene.viewer.register_keybinds(
        Keybind("manual_step",  Key.SPACE, callback=do_step),
        Keybind("manual_reset", Key.G,     callback=do_reset),
    )

    print(f"task={args.task}  action_dim={action_dim}  n_envs={env.n_envs}")
    print("SPACE = step  |  G = reset  |  Ctrl-C / close window = quit")

    while env.scene.viewer.is_alive():
        env.scene.viewer.update()


if __name__ == "__main__":
    main()
