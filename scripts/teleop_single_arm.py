"""
Keyboard teleop for the single-arm environment.

The env accepts a 6D Cartesian delta action [dx, dy, dz, droll, dpitch, dyaw].
This script maps key presses to that tensor and calls env.step() each frame.

Controls:
  w / s  — +x / -x        i / k  — pitch + / -
  a / d  — +y / -y        j / l  — roll  + / -
  q / e  — +z / -z        u / o  — yaw   + / -
  Esc    — quit
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import genesis as gs
import source  # noqa: F401 — triggers gym.register() calls

from source.tasks.single.env import SingleArmBallBalanceEnv

POS_STEP = 0.005   # metres per frame
ROT_STEP = 0.02    # radians per frame

KEY_TO_ACTION = {
    # pos: dx, dy, dz
    "w": np.array([ POS_STEP, 0, 0, 0, 0, 0], dtype=np.float32),
    "s": np.array([-POS_STEP, 0, 0, 0, 0, 0], dtype=np.float32),
    "a": np.array([0,  POS_STEP, 0, 0, 0, 0], dtype=np.float32),
    "d": np.array([0, -POS_STEP, 0, 0, 0, 0], dtype=np.float32),
    "q": np.array([0, 0,  POS_STEP, 0, 0, 0], dtype=np.float32),
    "e": np.array([0, 0, -POS_STEP, 0, 0, 0], dtype=np.float32),
    # rot: droll, dpitch, dyaw
    "j": np.array([0, 0, 0,  ROT_STEP, 0, 0], dtype=np.float32),
    "l": np.array([0, 0, 0, -ROT_STEP, 0, 0], dtype=np.float32),
    "i": np.array([0, 0, 0, 0,  ROT_STEP, 0], dtype=np.float32),
    "k": np.array([0, 0, 0, 0, -ROT_STEP, 0], dtype=np.float32),
    "u": np.array([0, 0, 0, 0, 0,  ROT_STEP], dtype=np.float32),
    "o": np.array([0, 0, 0, 0, 0, -ROT_STEP], dtype=np.float32),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    env = SingleArmBallBalanceEnv(show_viewer=not args.headless, n_envs=1)

    # Clear all Genesis viewer keybinds so they cannot fire during teleop
    if not args.headless and env.scene.viewer is not None:
        try:
            env.scene.viewer._pyrender_viewer._keybindings._keybinds.clear()
        except AttributeError:
            pass

    from pynput import keyboard as kb

    keys_held: set = set()
    quit_flag = [False]

    def on_press(key):
        try:
            keys_held.add(key.char)
        except AttributeError:
            if key == kb.Key.esc:
                quit_flag[0] = True

    def on_release(key):
        try:
            keys_held.discard(key.char)
        except AttributeError:
            pass

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nTeleop active — WASD/QE = position, IJKL/UO = orientation, Esc = quit\n")

    while not quit_flag[0]:
        delta = np.zeros(6, dtype=np.float32)
        for ch in list(keys_held):
            if ch in KEY_TO_ACTION:
                delta += KEY_TO_ACTION[ch]

        # broadcast same delta to all envs: (n_envs, 6)
        action = np.tile(delta, (env.n_envs, 1))
        env.step(action)

    listener.stop()
    print("Teleop finished.")


if __name__ == "__main__":
    main()
