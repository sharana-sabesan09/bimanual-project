"""
Debug script: load the G1 USD in Genesis, step with all-zero actions on the
right arm + hand, and print joint state every N steps to diagnose oscillations.

Prints on startup:
  - joint limits (lower / upper) for controlled dofs
  - what the zero action decodes to (midpoint of each joint range)
  - kp / kd gains (if queryable)

Prints every PRINT_EVERY steps:
  - joint positions vs target
  - joint velocities
  - applied torques (if queryable)

Usage:
    python scripts/debug_g1_usd.py [--headless] [--steps 200] [--print-every 10]
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import genesis as gs
from pathlib import Path

_ROOT = Path(__file__).parents[1]
G1_USD = str(_ROOT / "assets" / "g1-flattened-fixed.usd")

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
RIGHT_HAND_JOINTS = [
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",  "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",   "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
]
RIGHT_ARM_HOLD_POS = np.array([
    0.0,
    np.deg2rad(-30),
    0.0,
    0.0,
    np.deg2rad(90),
    np.deg2rad(90),
    np.deg2rad(30),
], dtype=np.float32)


def sep(title=""):
    w = 80
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
    else:
        print("─" * w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--steps",       type=int, default=300)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--no-gains",    action="store_true",
                        help="Skip setting kp/kd (reproduces the oscillation bug)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=not args.headless,
        sim_options=gs.options.SimOptions(dt=0.02),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.5, 2.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)),
        vis_mode="collision",
    )
    scene.build(n_envs=1)

    # ── build joint map ──────────────────────────────────────────────────
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}

    def dof_idx(name):
        return joint_map[name].dofs_idx_local[0]

    arm_dofs  = [dof_idx(n) for n in RIGHT_ARM_JOINTS]
    hand_dofs = [dof_idx(n) for n in RIGHT_HAND_JOINTS if n in joint_map]
    ctrl_dofs = arm_dofs + hand_dofs
    ctrl_names = RIGHT_ARM_JOINTS + [n for n in RIGHT_HAND_JOINTS if n in joint_map]
    n_ctrl = len(ctrl_dofs)

    # ── joint limits ─────────────────────────────────────────────────────
    sep("Joint limits & zero-action decode")
    try:
        lower, upper = robot.get_dofs_limit(dofs_idx_local=ctrl_dofs)
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
        limits_ok = True
    except Exception as e:
        print(f"[WARN] get_dofs_limit failed: {e}")
        lower = np.full(n_ctrl, -3.14159)
        upper = np.full(n_ctrl,  3.14159)
        limits_ok = False

    zero_action_target = lower + 0.5 * (upper - lower)   # what action=0 decodes to
    hold_target = np.concatenate([RIGHT_ARM_HOLD_POS, np.zeros(len(hand_dofs))])

    fmt = "{:<45s}  {:>8.3f}  {:>8.3f}  {:>10.3f}  {:>10.3f}"
    print(f"{'joint':<45s}  {'lower':>8}  {'upper':>8}  {'zero→pos':>10}  {'hold_pos':>10}")
    sep()
    for i, name in enumerate(ctrl_names):
        print(fmt.format(name, lower[i], upper[i], zero_action_target[i], hold_target[i]))

    # ── PD gains ─────────────────────────────────────────────────────────
    sep("PD gains (kp / kd) — as imported from USD")
    try:
        kp_imported = robot.get_dofs_kp(dofs_idx_local=ctrl_dofs).cpu().numpy()
        kd_imported = robot.get_dofs_kv(dofs_idx_local=ctrl_dofs).cpu().numpy()
        print(f"{'joint':<45s}  {'kp':>8}  {'kd':>8}")
        sep()
        for i, name in enumerate(ctrl_names):
            print(f"{name:<45s}  {kp_imported[i]:>8.2f}  {kd_imported[i]:>8.2f}")
    except Exception as e:
        print(f"[WARN] could not read kp/kd: {e}")

    if not args.no_gains:
        sep("Setting explicit kp / kd gains")
        # Gain tables matching single_arm_grasp.py
        all_joint_names = (
            ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
             "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"]
            + ["right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
               "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
            + ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"]
            + ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
               "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
            + RIGHT_ARM_JOINTS
            + [n for n in RIGHT_HAND_JOINTS if n in joint_map]
        )
        # kp=20, kd=0.6: 2x stiffer spring + more damping → ~half the settling time vs kp=10,kd=0.3.
        # Stay well below numerical instability (kp=100 oscillated at dt=0.02).
        kp_vals = [200]*6 + [200]*6 + [200,200,200] + [100,100,100,100,40,40,40] + [100,100,100,100,40,40,40] + [20]*len(hand_dofs)
        kd_vals = [10]*6  + [10]*6  + [10,10,10]   + [10,10,10,10,4,4,4]        + [10,10,10,10,4,4,4]        + [1.2]*len(hand_dofs)

        all_dofs = []
        all_kp   = []
        all_kd   = []
        for name, kp, kd in zip(all_joint_names, kp_vals, kd_vals):
            if name in joint_map:
                all_dofs.append(joint_map[name].dofs_idx_local[0])
                all_kp.append(kp)
                all_kd.append(kd)

        robot.set_dofs_kp(
            torch.tensor(all_kp, dtype=torch.float32, device=gs.device),
            dofs_idx_local=all_dofs,
        )
        robot.set_dofs_kv(
            torch.tensor(all_kd, dtype=torch.float32, device=gs.device),
            dofs_idx_local=all_dofs,
        )

        # Also command all non-arm joints to their zero / natural pose
        ctrl_set = set(ctrl_dofs)
        all_frozen_dofs = [d for d in all_dofs if d not in ctrl_set]
        frozen_target = torch.zeros(len(all_frozen_dofs), device=gs.device)
        robot.control_dofs_position(frozen_target, dofs_idx_local=all_frozen_dofs)

        sep("Final kp / kd (queried back from sim after setting)")
        final_kp = robot.get_dofs_kp(dofs_idx_local=all_dofs).cpu().numpy()
        final_kd = robot.get_dofs_kv(dofs_idx_local=all_dofs).cpu().numpy()
        name_lookup = {joint_map[n].dofs_idx_local[0]: n for n in all_joint_names if n in joint_map}
        print(f"{'joint':<45s}  {'kp':>8}  {'kd':>8}")
        sep()
        for d, kp, kd in zip(all_dofs, final_kp, final_kd):
            print(f"{name_lookup[d]:<45s}  {kp:>8.2f}  {kd:>8.3f}")
    else:
        all_frozen_dofs = []
        frozen_target   = None

    # ── step loop ────────────────────────────────────────────────────────
    sep("Stepping with zero action")
    print(f"Running {args.steps} steps, printing every {args.print_every}.\n")

    # Target = midpoint (what zero action produces)
    target_t = torch.tensor(zero_action_target, dtype=torch.float32, device=gs.device).unsqueeze(0)

    for step in range(args.steps):
        robot.control_dofs_position(target_t[:, :len(arm_dofs)],  dofs_idx_local=arm_dofs)
        if hand_dofs:
            robot.control_dofs_position(target_t[:, len(arm_dofs):], dofs_idx_local=hand_dofs)
        if args.no_gains:
            # Don't command frozen joints — shows the oscillation from floating limbs
            pass
        else:
            # Keep everything else at zero to isolate arm behaviour
            if all_frozen_dofs:
                robot.control_dofs_position(frozen_target, dofs_idx_local=all_frozen_dofs)
        scene.step()

        if step % args.print_every == 0:
            pos = robot.get_dofs_position(dofs_idx_local=ctrl_dofs).squeeze(0).cpu().numpy()
            vel = robot.get_dofs_velocity(dofs_idx_local=ctrl_dofs).squeeze(0).cpu().numpy()

            try:
                force = robot.get_dofs_force(dofs_idx_local=ctrl_dofs).squeeze(0).cpu().numpy()
                has_force = True
            except Exception:
                has_force = False

            print(f"\n── step {step:4d} ──────────────────────────────────────────────────────")
            hdr = f"{'joint':<45s}  {'target':>8}  {'pos':>8}  {'err':>8}  {'vel':>8}"
            if has_force:
                hdr += f"  {'torque':>8}"
            print(hdr)
            for i, name in enumerate(ctrl_names):
                err = zero_action_target[i] - pos[i]
                row = f"{name:<45s}  {zero_action_target[i]:>8.3f}  {pos[i]:>8.3f}  {err:>8.3f}  {vel[i]:>8.3f}"
                if has_force:
                    row += f"  {force[i]:>8.2f}"
                print(row)

            max_vel = np.abs(vel).max()
            max_err = np.abs(zero_action_target - pos).max()
            print(f"  → max |vel|={max_vel:.4f}  max |err|={max_err:.4f}")

    sep("Done")


if __name__ == "__main__":
    main()
