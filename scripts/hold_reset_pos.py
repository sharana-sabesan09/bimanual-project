"""
Hold-pose agent: steps the env with the action that corresponds to the
reset/hold joint positions, scaled back to [-1, 1].

Useful for verifying the hold pose is stable and for reading off the
constant action vector that keeps the robot at its initial pose.

Usage:
    python scripts/hold_reset_pos.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import genesis as gs

from source.tasks.tray_grasp.single_arm_grasp import SingleArmTrayGraspEnv


def main():
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

    env = SingleArmTrayGraspEnv(show_viewer=True, n_envs=1)

    # Read arm + hand positions immediately after reset (hold pose)
    joint_pos = env.robot.get_dofs_position(dofs_idx_local=env._action_dofs)  # (1, n_action_dofs)

    # Invert: pos = lower + (action+1)*0.5*(upper-lower)  =>  action = 2*(pos-lower)/(upper-lower) - 1
    lower = env._action_lower
    upper = env._action_upper
    hold_action = (2.0 * (joint_pos - lower) / (upper - lower) - 1.0).clamp(-1.0, 1.0)

    print(f"\nAction space size: {env.n_action_dofs}  "
          f"(arm={env.n_arm_dofs}, hand={env.n_hand_dofs})")
    print("Hold-pose action vector ([-1, 1]):")
    print(hold_action.squeeze(0).cpu().numpy())

    while True:
        env.step(hold_action)


if __name__ == "__main__":
    main()
