"""
Manual step: press SPACE to advance one sim step, R to reset the env.
Usage: python scripts/manual_step.py --task <gym_env_id>
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
                        help="Gym env ID, e.g. BallBalance-TrayGrasp-v0")
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
    env = EnvClass(show_viewer=True, n_envs=args.num_envs, debug_contacts=True, debug=True)
    obs, _ = env.reset()
    action = np.tile(np.array([
         7.3888898e-02, -9.8003685e-02,  5.1653385e-04, -3.3074594e-01,
         7.9569197e-01,  9.7251117e-01,  3.2383990e-01, -1.0000000e+00,
        -7.1428764e-01, -1.0000000e+00, -7.1428812e-01, -1.0000000e+00,
        -7.1428722e-01, -1.0000000e+00, -7.1428788e-01,  6.3885069e-01,
         8.8501084e-01,  1.2721789e-01, -1.5328598e-01,
    ], dtype=np.float32), (env.n_envs, 1))
    step = [0]

    hand_names = getattr(env, "_hand_link_names", [f"link_{i}" for i in range(12)])

    def do_step():
        obs, reward, terminated, truncated, _ = env.step(action)
        step[0] += 1
        print(f"step {step[0]:5d}  reward={reward[0]:.4f}  "
              f"terminated={terminated[0].item()}  truncated={truncated[0].item()}")

        # Read contact forces directly from obs to verify correctness.
        # Slices match the obs layout defined in single_arm_grasp.py:
        #   [91:127] per-hand-link tray contact force vectors (12 x 3, flattened)
        #   [127]    total summed tray contact force (N)
        o = obs[0]  # env 0
        per_link_flat = o[91:127].cpu().reshape(12, 3)
        total         = o[127].item()

        print(f"  obs total tray contact force : {total:8.3f} N")
        print(f"  obs per-link tray forces (N) :")
        for name, fvec in zip(hand_names, per_link_flat.tolist()):
            mag = (fvec[0]**2 + fvec[1]**2 + fvec[2]**2) ** 0.5
            bar = "█" * int(mag / 0.5)
            print(f"    {name:<45s} [{fvec[0]:6.2f}, {fvec[1]:6.2f}, {fvec[2]:6.2f}]  |f|={mag:.3f}  {bar}")

    def do_reset():
        env.reset()
        step[0] = 0
        print("--- reset ---")

    env.scene.viewer.register_keybinds(
        Keybind("manual_step",  Key.SPACE, callback=do_step),
        Keybind("manual_reset", Key.G,     callback=do_reset),
    )

    print("SPACE = step  |  G = reset  |  Ctrl-C / close window = quit")

    while env.scene.viewer.is_alive():
        env.scene.viewer.update()


if __name__ == "__main__":
    main()
