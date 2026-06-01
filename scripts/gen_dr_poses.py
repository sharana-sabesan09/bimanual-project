"""
DR pose generator — runs N parallel envs simultaneously.

Each round all N envs get independent DR samples, settle together,
then validity is checked per-env.  Valid envs get their ball placed
and are shown in the viewer; invalid envs keep the ball underground
so you can visually tell which configs were accepted.

Usage:
    python scripts/gen_dr_poses.py --n_envs 8 --n_samples 200
    python scripts/gen_dr_poses.py --n_envs 4 --n_samples 500 --headless
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import genesis as gs

from pathlib import Path
_ROOT = Path(__file__).parents[1]
G1_XML = str(_ROOT / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_dual_arm.xml")

# ------------------------------------------------------------------ #
# Joint definitions                                                    #
# ------------------------------------------------------------------ #
LEFT_LEG_JOINTS  = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
                    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint"]
RIGHT_LEG_JOINTS = ["right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
                    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
WAIST_JOINTS     = ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"]
LEFT_ARM_JOINTS  = ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
RIGHT_ARM_JOINTS = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]

STAND_LEG_POS   = np.zeros(6,  dtype=np.float32)
STAND_WAIST_POS = np.zeros(3,  dtype=np.float32)
RIGHT_ARM_HOLD  = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD   = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)

#              pitch  roll   yaw  elbow  w_roll  w_pitch  w_yaw
# Hardware limits from MJCF (for reference):
#   shoulder_pitch: [-3.09, 2.67]   shoulder_roll R: [-2.25, 1.59]  L: [-1.59, 2.25]
#   shoulder_yaw:   [-2.62, 2.62]   elbow:           [-1.05, 2.09]
#   wrist_roll:     [-1.97, 1.97]   wrist_pitch:     [-1.61, 1.61]  wrist_yaw: [-1.61, 1.61]
MIRROR_SIGN        = np.array([1,   -1,    1,    1,      1,      1,      1], dtype=np.float32)
RIGHT_SAFE_LOWER   = np.array([-2.5, -2.0, -1.5, 0.4, -1.97, -1.61, -1.5], dtype=np.float32)
RIGHT_SAFE_UPPER   = np.array([ 0.4,  0.8,  1.5, 2.09,  1.97,  0.5,  1.5], dtype=np.float32)
LEFT_SAFE_LOWER    = np.array([-2.5, -0.8, -1.5, 0.4, -1.97, -1.61, -1.5], dtype=np.float32)
LEFT_SAFE_UPPER    = np.array([ 0.4,  2.0,  1.5, 2.09,  1.97,  0.5,  1.5], dtype=np.float32)

TRAY_X_MIN        = 0.22
TRAY_Z_MIN        = 1.0
TRAY_Z_MAX        = 2.1
# Tilt threshold set at runtime via --max_tray_tilt_deg; stored here after init.
TRAY_NORMAL_Z_MIN = 0.3   # default fallback only — overridden by DRPoseGenerator

BALL_RADIUS = 0.03
TRAY_SIZE   = (0.36, 0.26, 0.01)
BALL_AWAY   = [0.0, 0.0, -5.0]


# ------------------------------------------------------------------ #
# Generator                                                            #
# ------------------------------------------------------------------ #

class DRPoseGenerator:
    def __init__(
        self,
        n_envs: int = 1,
        show_viewer: bool = True,
        dt: float = 0.002,
        sym_dr_scale  = np.array([0.50, 0.40, 0.60, 0.40, 0.80, 0.50, 0.60], dtype=np.float32),
        asym_dr_scale = np.array([0.25, 0.20, 0.30, 0.20, 0.40, 0.25, 0.30], dtype=np.float32),
        nominal_settle_steps: int = 150,
        dr_settle_steps: int = 80,
        low_x_thresh: float = 0.0,  # highlight accepted envs with tray_x below this; 0 = disabled
        max_tray_tilt_deg: float = 40.0,  # reject if tray is tilted more than this from horizontal
        vel_thresh: float = 0.5,  # max joint velocity (rad/s) to be considered settled
    ):
        self.n_envs           = n_envs
        self.show_viewer      = show_viewer
        self.sym_scale        = sym_dr_scale
        self.asym_scale       = asym_dr_scale
        self.nom_settle       = nominal_settle_steps
        self.dr_settle        = dr_settle_steps
        self.low_x_thresh     = low_x_thresh
        self.vel_thresh       = vel_thresh
        # normal_z of tray z-axis must be < -cos(tilt_deg) to be accepted
        self.tray_normal_z_thresh = float(np.cos(np.radians(max_tray_tilt_deg)))

        self._build_scene(dt)
        self._init_indices()
        self._freeze_pos = np.concatenate([STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS])
        self._reset_to_nominal(self.nom_settle, show=False)

    # -------------------------------------------------------------- #
    # Scene                                                            #
    # -------------------------------------------------------------- #

    def _build_scene(self, dt: float):
        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -3.5, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=50,
                max_FPS=60,
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=G1_XML, pos=(0.0, 0.0, 0.79))
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=BALL_AWAY),
            material=gs.materials.Rigid(rho=0.1 / (4/3 * np.pi * BALL_RADIUS**3)),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),
        )
        # Green non-colliding marker for all accepted envs
        self.accept_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS * 1.5, pos=BALL_AWAY, collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.9, 0.1, 0.9)),
        )
        # Yellow non-colliding marker placed above tray for low-x highlight envs
        self.low_x_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS * 1.5, pos=BALL_AWAY, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.9, 0.0, 0.9)),
        )
        self.scene.build(n_envs=self.n_envs, env_spacing=(2.0, 2.0))

    def _init_indices(self):
        def dof(name):
            return self.robot.get_joint(name).dofs_idx_local[0]
        self._left_leg_dofs  = [dof(n) for n in LEFT_LEG_JOINTS]
        self._right_leg_dofs = [dof(n) for n in RIGHT_LEG_JOINTS]
        self._waist_dofs     = [dof(n) for n in WAIST_JOINTS]
        self._left_arm_dofs  = [dof(n) for n in LEFT_ARM_JOINTS]
        self._right_arm_dofs = [dof(n) for n in RIGHT_ARM_JOINTS]
        self._frozen_dofs    = self._left_leg_dofs + self._right_leg_dofs + self._waist_dofs
        self._tray_link      = self.robot.get_link("tray")

    # -------------------------------------------------------------- #
    # Nominal reset                                                    #
    # -------------------------------------------------------------- #

    def _reset_to_nominal(self, n_steps: int, show: bool = False):
        """Snap all envs to nominal arm positions, then settle so the
        weld constraints bring the tray back upright."""
        self.robot.set_dofs_position(self._freeze_pos,  dofs_idx_local=self._frozen_dofs,    zero_velocity=True)
        self.robot.set_dofs_position(RIGHT_ARM_HOLD,    dofs_idx_local=self._right_arm_dofs,  zero_velocity=True)
        self.robot.set_dofs_position(LEFT_ARM_HOLD,     dofs_idx_local=self._left_arm_dofs,   zero_velocity=True)
        self.robot.control_dofs_position(self._freeze_pos, dofs_idx_local=self._frozen_dofs)
        self.robot.control_dofs_position(RIGHT_ARM_HOLD,   dofs_idx_local=self._right_arm_dofs)
        self.robot.control_dofs_position(LEFT_ARM_HOLD,    dofs_idx_local=self._left_arm_dofs)
        self.ball.set_pos(
            torch.tensor([BALL_AWAY] * self.n_envs, dtype=torch.float32, device=gs.device),
            zero_velocity=True,
        )
        for _ in range(n_steps):
            self.scene.step(update_visualizer=show)

    # -------------------------------------------------------------- #
    # DR sampling — independent per env                               #
    # -------------------------------------------------------------- #

    def _sample_dr_batch(self):
        """Returns (r_hold, l_hold) each shape (n_envs, 7) numpy float32."""
        n = self.n_envs
        delta   = (np.random.rand(n, 7) - 0.5) * 2.0 * self.sym_scale   # (n, 7)
        epsilon = (np.random.rand(n, 7) - 0.5) * 2.0 * self.asym_scale  # (n, 7)

        r_delta = delta + epsilon
        l_delta = delta * MIRROR_SIGN - epsilon * MIRROR_SIGN

        r_hold = np.clip(RIGHT_ARM_HOLD + r_delta, RIGHT_SAFE_LOWER, RIGHT_SAFE_UPPER).astype(np.float32)
        l_hold = np.clip(LEFT_ARM_HOLD  + l_delta, LEFT_SAFE_LOWER,  LEFT_SAFE_UPPER).astype(np.float32)
        return r_hold, l_hold

    # -------------------------------------------------------------- #
    # Validity — per env                                               #
    # -------------------------------------------------------------- #

    def _check_validity_batch(self, verbose: bool = True, vel_thresh: float = 0.5):
        """Returns valid (n_envs,) bool numpy array and per-env diagnostic strings.

        vel_thresh: max allowed joint velocity (rad/s) — rejects configs that are
        still settling and would drift to a different pose during display.
        """
        tray_pos  = self._tray_link.get_pos().cpu().numpy()   # (n_envs, 3)
        tray_quat = self._tray_link.get_quat().cpu().numpy()  # (n_envs, 4) xyzw

        qx, qy    = tray_quat[:, 0], tray_quat[:, 1]
        normal_z  = 1.0 - 2.0 * (qx**2 + qy**2)
        tilt_deg  = np.degrees(np.arccos(np.clip(-normal_z, -1.0, 1.0)))  # 0° = flat

        # Joint velocity check — a pose that looks valid now but has high velocity
        # will drift somewhere completely different during display
        r_vel = self.robot.get_dofs_velocity(dofs_idx_local=self._right_arm_dofs).cpu().numpy()
        l_vel = self.robot.get_dofs_velocity(dofs_idx_local=self._left_arm_dofs).cpu().numpy()
        max_joint_vel = np.abs(np.concatenate([r_vel, l_vel], axis=1)).max(axis=1)  # (n_envs,)
        settled = max_joint_vel < vel_thresh

        in_front       = tray_pos[:, 0] > TRAY_X_MIN
        valid_z        = (tray_pos[:, 2] > TRAY_Z_MIN) & (tray_pos[:, 2] < TRAY_Z_MAX)
        not_too_tilted = normal_z < -self.tray_normal_z_thresh

        valid = in_front & valid_z & not_too_tilted & settled

        if verbose:
            max_tilt_deg = np.degrees(np.arccos(self.tray_normal_z_thresh))
            for i in range(self.n_envs):
                tp = tray_pos[i]
                if valid[i]:
                    print(f"    env {i}: [accept] "
                          f"tray=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f})  "
                          f"tilt={tilt_deg[i]:.1f}°  max_vel={max_joint_vel[i]:.3f}")
                else:
                    reasons = []
                    if not in_front[i]:
                        reasons.append(f"not_in_front(x={tp[0]:.2f}<={TRAY_X_MIN})")
                    if tray_pos[i, 2] < TRAY_Z_MIN:
                        reasons.append(f"too_low(z={tp[2]:.2f})")
                    if tray_pos[i, 2] > TRAY_Z_MAX:
                        reasons.append(f"too_high(z={tp[2]:.2f})")
                    if not not_too_tilted[i]:
                        reasons.append(f"tilted({tilt_deg[i]:.1f}°>{max_tilt_deg:.0f}°)")
                    if not settled[i]:
                        reasons.append(f"unsettled(max_vel={max_joint_vel[i]:.3f}>{vel_thresh})")
                    print(f"    env {i}: [reject] {', '.join(reasons)}")

        return valid, tray_pos, tray_quat, normal_z, max_joint_vel

    # -------------------------------------------------------------- #
    # Ball placement                                                   #
    # -------------------------------------------------------------- #

    def _place_balls(self, valid: np.ndarray, tray_pos: np.ndarray, tray_quat: np.ndarray):
        """Place ball on the tray surface (accounting for tray orientation) for valid envs.
        Markers float above. Invalid envs keep everything underground."""
        # Tray local-z points DOWN at nominal; surface normal = -local_z_in_world.
        # local_z_in_world = [2(qx*qz+qy*qw), 2(qy*qz-qx*qw), 1-2(qx²+qy²)]
        qx = tray_quat[:, 0]; qy = tray_quat[:, 1]
        qz = tray_quat[:, 2]; qw = tray_quat[:, 3]
        surf_normal = np.stack([
            -2*(qx*qz + qy*qw),
            -2*(qy*qz - qx*qw),
            -(1 - 2*(qx**2 + qy**2)),
        ], axis=1)  # (n_envs, 3) — points away from tray top surface

        ball_surf_offset = TRAY_SIZE[2] / 2 + BALL_RADIUS + 0.005

        # Red ball — placed on tray surface along surface normal
        positions = np.tile(BALL_AWAY, (self.n_envs, 1)).astype(np.float32)
        positions[valid] = tray_pos[valid] + surf_normal[valid] * ball_surf_offset
        self.ball.set_pos(torch.tensor(positions, device=gs.device), zero_velocity=True)

        # Green marker — floats 0.25m above ball in world +z (orientation-independent)
        accept_pos = np.tile(BALL_AWAY, (self.n_envs, 1)).astype(np.float32)
        accept_pos[valid] = positions[valid].copy()
        accept_pos[valid, 2] += 0.25
        self.accept_marker.set_pos(torch.tensor(accept_pos, device=gs.device), zero_velocity=True)

        # Yellow marker — 0.25m above green, only for low-x envs
        low_x_pos = np.tile(BALL_AWAY, (self.n_envs, 1)).astype(np.float32)
        if self.low_x_thresh > 0:
            low_x = valid & (tray_pos[:, 0] < self.low_x_thresh)
            low_x_pos[low_x] = accept_pos[low_x].copy()
            low_x_pos[low_x, 2] += 0.25
        self.low_x_marker.set_pos(torch.tensor(low_x_pos, device=gs.device), zero_velocity=True)

    # -------------------------------------------------------------- #
    # Main generation loop                                             #
    # -------------------------------------------------------------- #

    def generate(
        self,
        n_samples: int,
        output_csv: str,
        max_rounds_without_progress: int = 20,
        display_steps: int = 60,
    ):
        import time

        r_cols    = [f"r_joint_{i}" for i in range(7)]
        l_cols    = [f"l_joint_{i}" for i in range(7)]
        tray_cols = ["tray_x","tray_y","tray_z","tray_qx","tray_qy","tray_qz","tray_qw"]
        fieldnames = r_cols + l_cols + tray_cols

        file_exists = os.path.exists(output_csv)
        f = open(output_csv, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        n_collected       = 0
        total_envs_tried  = 0
        total_envs_valid  = 0
        stall_rounds      = 0
        round_num         = 0
        t_start           = time.time()

        # Aggregate rejection reason counters
        rej_not_front  = 0
        rej_too_low    = 0
        rej_too_high   = 0
        rej_tilted     = 0
        rej_unsettled  = 0

        # Running stats for valid tray positions
        valid_tray_xyz = []

        max_tilt_deg = np.degrees(np.arccos(self.tray_normal_z_thresh))
        print(f"\nCollecting {n_samples} valid poses → {output_csv}")
        print(f"  n_envs={self.n_envs}  display_steps={display_steps}")
        print(f"  sym_scale:  {self.sym_scale}")
        print(f"  asym_scale: {self.asym_scale}")
        print(f"  max_tray_tilt: {max_tilt_deg:.0f}°  (normal_z_thresh={self.tray_normal_z_thresh:.3f})\n")

        while n_collected < n_samples:
            round_num += 1

            # --- settle all envs to nominal ---
            self._reset_to_nominal(self.nom_settle, show=False)

            # --- sample independent DR per env ---
            r_hold, l_hold = self._sample_dr_batch()
            self.robot.set_dofs_position(
                torch.tensor(r_hold, device=gs.device),
                dofs_idx_local=self._right_arm_dofs, zero_velocity=True,
            )
            self.robot.set_dofs_position(
                torch.tensor(l_hold, device=gs.device),
                dofs_idx_local=self._left_arm_dofs, zero_velocity=True,
            )
            self.robot.control_dofs_position(
                torch.tensor(r_hold, device=gs.device), dofs_idx_local=self._right_arm_dofs,
            )
            self.robot.control_dofs_position(
                torch.tensor(l_hold, device=gs.device), dofs_idx_local=self._left_arm_dofs,
            )

            # --- settle DR without rendering ---
            for _ in range(self.dr_settle):
                self.scene.step(update_visualizer=False)

            # --- read actual settled positions ---
            r_settled = self.robot.get_dofs_position(dofs_idx_local=self._right_arm_dofs).cpu().numpy()
            l_settled = self.robot.get_dofs_position(dofs_idx_local=self._left_arm_dofs).cpu().numpy()

            # --- check validity per env ---
            valid, tray_pos, tray_quat, normal_z, max_joint_vel = \
                self._check_validity_batch(verbose=False, vel_thresh=self.vel_thresh)

            # Tally rejection reasons for invalid envs
            for i in range(self.n_envs):
                if not valid[i]:
                    if tray_pos[i, 0] <= TRAY_X_MIN:
                        rej_not_front += 1
                    if tray_pos[i, 2] < TRAY_Z_MIN:
                        rej_too_low   += 1
                    if tray_pos[i, 2] > TRAY_Z_MAX:
                        rej_too_high  += 1
                    if normal_z[i] >= -self.tray_normal_z_thresh:
                        rej_tilted    += 1
                    if max_joint_vel[i] >= self.vel_thresh:
                        rej_unsettled += 1

            n_valid = valid.sum()
            total_envs_tried += self.n_envs
            total_envs_valid += n_valid
            accept_rate = total_envs_valid / total_envs_tried * 100

            # --- write valid envs to CSV ---
            for i in range(self.n_envs):
                if valid[i] and n_collected < n_samples:
                    row = {}
                    for j, v in enumerate(r_settled[i]): row[f"r_joint_{j}"] = float(v)
                    for j, v in enumerate(l_settled[i]): row[f"l_joint_{j}"] = float(v)
                    row["tray_x"]  = float(tray_pos[i, 0])
                    row["tray_y"]  = float(tray_pos[i, 1])
                    row["tray_z"]  = float(tray_pos[i, 2])
                    row["tray_qx"] = float(tray_quat[i, 0])
                    row["tray_qy"] = float(tray_quat[i, 1])
                    row["tray_qz"] = float(tray_quat[i, 2])
                    row["tray_qw"] = float(tray_quat[i, 3])
                    writer.writerow(row)
                    valid_tray_xyz.append(tray_pos[i].copy())
                    n_collected += 1

            f.flush()

            # --- per-round summary line ---
            elapsed  = time.time() - t_start
            rate_sps = n_collected / elapsed if elapsed > 0 else 0.0
            eta_s    = (n_samples - n_collected) / rate_sps if rate_sps > 0 else float("inf")
            eta_str  = f"{eta_s/60:.1f}min" if eta_s < 3600 else f"{eta_s/3600:.1f}h"

            total_rejected = total_envs_tried - total_envs_valid

            print(
                f"Round {round_num:4d} | {n_valid:2d}/{self.n_envs} accepted | "
                f"cumulative accept={accept_rate:5.1f}% | "
                f"collected={n_collected:4d}/{n_samples} | "
                f"rate={rate_sps:.2f}/s | ETA={eta_str}"
            )
            if total_rejected > 0:
                # counts are cumulative; an env can appear in multiple buckets
                print(
                    f"         rej reasons (cumulative, multi-count ok) | "
                    f"total_rejected={total_rejected} | "
                    f"not_front={rej_not_front}  "
                    f"too_low={rej_too_low}  too_high={rej_too_high}  "
                    f"tilted={rej_tilted}  unsettled={rej_unsettled}"
                )
            # Velocity distribution — use this to calibrate --vel_thresh
            print(
                f"         joint vel (rad/s) → "
                f"min={max_joint_vel.min():.3f}  "
                f"median={np.median(max_joint_vel):.3f}  "
                f"max={max_joint_vel.max():.3f}  "
                f"(thresh={self.vel_thresh})"
            )

            # --- tray distribution summary every 10 rounds ---
            if round_num % 10 == 0 and valid_tray_xyz:
                arr = np.array(valid_tray_xyz)
                print(
                    f"  [tray distribution so far]  "
                    f"x=[{arr[:,0].min():.2f},{arr[:,0].max():.2f}]  "
                    f"y=[{arr[:,1].min():.2f},{arr[:,1].max():.2f}]  "
                    f"z=[{arr[:,2].min():.2f},{arr[:,2].max():.2f}]"
                )

            if n_valid == 0:
                stall_rounds += 1
                if stall_rounds >= max_rounds_without_progress:
                    print(f"  [warning] {stall_rounds} consecutive rounds with 0 accepts — "
                          f"consider reducing DR scales")
                    stall_rounds = 0
            else:
                stall_rounds = 0

            # --- flag low-x accepted poses in terminal ---
            if self.low_x_thresh > 0:
                low_x_envs = [i for i in range(self.n_envs)
                              if valid[i] and tray_pos[i, 0] < self.low_x_thresh]
                if low_x_envs:
                    details = "  ".join(
                        f"env{i}(x={tray_pos[i,0]:.3f},z={tray_pos[i,2]:.3f})"
                        for i in low_x_envs
                    )
                    print(f"  *** LOW-X (< {self.low_x_thresh:.2f}): {details}  [yellow marker in viewer]")

            # --- display: valid envs keep DR pose; invalid envs teleport underground ---
            if self.show_viewer:
                invalid_idx = np.where(~valid)[0].tolist()
                valid_idx   = np.where(valid)[0].tolist()
                base_pos = self.robot.get_pos().clone()

                if invalid_idx:
                    inv_t = torch.tensor(invalid_idx, device=gs.device)
                    underground = base_pos[invalid_idx].clone()
                    underground[:, 2] = -5.0
                    self.robot.set_pos(underground, envs_idx=inv_t)

                # Place ball on tray surface (orientation-aware) for valid envs
                self._place_balls(valid, tray_pos, tray_quat)

                if valid_idx:
                    val_t = torch.tensor(valid_idx, device=gs.device)
                    # Lock valid envs to settled physics equilibrium — eliminates oscillation
                    self.robot.control_dofs_position(
                        torch.tensor(r_settled[valid_idx], device=gs.device),
                        dofs_idx_local=self._right_arm_dofs, envs_idx=val_t,
                    )
                    self.robot.control_dofs_position(
                        torch.tensor(l_settled[valid_idx], device=gs.device),
                        dofs_idx_local=self._left_arm_dofs, envs_idx=val_t,
                    )

                for _ in range(display_steps):
                    self.scene.step(update_visualizer=True)

                # Restore invalid envs before next round's nominal reset
                if invalid_idx:
                    inv_t = torch.tensor(invalid_idx, device=gs.device)
                    self.robot.set_pos(base_pos[invalid_idx], envs_idx=inv_t)

        f.close()
        elapsed = time.time() - t_start
        print(f"\nDone. {n_collected} poses saved to {output_csv}")
        print(f"Rounds: {round_num}  Total env-slots tried: {total_envs_tried}  "
              f"Accept rate: {total_envs_valid/total_envs_tried*100:.1f}%  "
              f"Elapsed: {elapsed:.1f}s")
        if valid_tray_xyz:
            arr = np.array(valid_tray_xyz)
            print(f"Valid tray x: mean={arr[:,0].mean():.3f} std={arr[:,0].std():.3f} "
                  f"range=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}]")
            print(f"Valid tray y: mean={arr[:,1].mean():.3f} std={arr[:,1].std():.3f} "
                  f"range=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]")
            print(f"Valid tray z: mean={arr[:,2].mean():.3f} std={arr[:,2].std():.3f} "
                  f"range=[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs",     type=int, default=4)
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--output",     type=str, default="dr_poses.csv")
    parser.add_argument("--sym_scale",  type=str,
                        default="0.50,0.40,0.60,0.40,0.80,0.50,0.60",
                        help="Per-joint sym scale:  pitch,roll,yaw,elbow,w_roll,w_pitch,w_yaw")
    parser.add_argument("--asym_scale", type=str,
                        default="0.25,0.20,0.30,0.20,0.40,0.25,0.30",
                        help="Per-joint asym scale: pitch,roll,yaw,elbow,w_roll,w_pitch,w_yaw")
    parser.add_argument("--settle_nominal", type=int, default=150)
    parser.add_argument("--settle_dr",      type=int, default=80)
    parser.add_argument("--display_steps",  type=int, default=60)
    parser.add_argument("--headless",     action="store_true")
    parser.add_argument("--cpu",          action="store_true",
                        help="Force CPU backend (default is CUDA). Use for visual checking.")
    parser.add_argument("--dt",           type=float, default=0.002)
    parser.add_argument("--low_x_thresh", type=float, default=0.0,
                        help="Flag accepted poses with tray_x below this value "
                             "(yellow marker in viewer + terminal callout). 0 = disabled.")
    parser.add_argument("--vel_thresh", type=float, default=0.5,
                        help="Max joint velocity (rad/s) for a config to be considered settled. "
                             "Run once and check the 'joint vel' line to calibrate.")
    parser.add_argument("--max_tray_tilt_deg", type=float, default=40.0,
                        help="Reject poses where tray is tilted more than this many degrees "
                             "from horizontal. 0°=flat only, 90°=any orientation. Default: 40°")
    args = parser.parse_args()

    sym_scale  = np.array([float(x) for x in args.sym_scale.split(",")],  dtype=np.float32)
    asym_scale = np.array([float(x) for x in args.asym_scale.split(",")], dtype=np.float32)
    assert len(sym_scale) == 7 and len(asym_scale) == 7, \
        "sym_scale and asym_scale must each have 7 comma-separated values"

    backend = gs.cpu if args.cpu else gs.cuda
    try:
        gs.init(backend=backend, logging_level="warning")
    except Exception:
        print("[warning] CUDA init failed, falling back to CPU")
        gs.init(backend=gs.cpu, logging_level="warning")

    gen = DRPoseGenerator(
        n_envs               = args.n_envs,
        show_viewer          = not args.headless,
        dt                   = args.dt,
        sym_dr_scale         = sym_scale,
        asym_dr_scale        = asym_scale,
        nominal_settle_steps = args.settle_nominal,
        dr_settle_steps      = args.settle_dr,
        low_x_thresh         = args.low_x_thresh,
        max_tray_tilt_deg    = args.max_tray_tilt_deg,
        vel_thresh           = args.vel_thresh,
    )
    gen.generate(
        n_samples    = args.n_samples,
        output_csv   = args.output,
        display_steps = args.display_steps,
    )


if __name__ == "__main__":
    main()
