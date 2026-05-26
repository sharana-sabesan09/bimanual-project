"""
Baseline ball-balancing environment.

One-arm baseline: Unitree G1 with a tray rigidly fixed to the right wrist
(right_wrist_yaw_link). A ball is spawned on the tray. The policy controls
only the 7 right-arm joints via position targets.

Legs (12 DOF), waist (3 DOF), and left arm (7 DOF) are held at the G1's
factory standing pose via position PD control every step. They are never
part of the RL action.

Right arm DOF (7):
  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
  right_elbow,
  right_wrist_roll, right_wrist_pitch, right_wrist_yaw

Standing pose from the MuJoCo Menagerie "stand" keyframe:
  legs  : all zeros
  waist : all zeros
  arms  : shoulder_pitch=0.2, shoulder_roll=±0.2, shoulder_yaw=0,
          elbow=1.28, wrist_roll/pitch/yaw=0
"""

import os
import numpy as np
import torch
import genesis as gs

# ─── paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# Fixed-base variant: freejoint removed so the torso is welded to the world.
# This lets us train arm control without fighting a falling floating base.
G1_XML = os.path.join(_HERE, "assets", "mujoco_menagerie", "unitree_g1", "g1_fixed_base.xml")

# ─── tray / ball geometry ────────────────────────────────────────────────────
TRAY_SIZE   = (0.36, 0.26, 0.01)   # 36 cm × 26 cm × 1 cm (defined in MJCF; kept here for ball spawn offset)
BALL_RADIUS = 0.03                  # 3 cm
BALL_MASS   = 0.1                   # kg

# ─── joint name groups ────────────────────────────────────────────────────────
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ─── standing / hold poses (from keyframe) ────────────────────────────────────
# All leg and waist joints at zero = upright stance
STAND_LEG_POS   = np.zeros(6, dtype=np.float32)
STAND_WAIST_POS = np.zeros(3, dtype=np.float32)
# Left arm parked at keyframe default (arm out of the way)
STAND_LEFT_ARM_POS = np.array([0.2,  0.2, 0.0, 1.28, 0.0, 0.0, 0.0], dtype=np.float32)

# Right arm working pose: arm raised forward, elbow bent, wrist adjusted so the
# tray is roughly horizontal (~42 cm in front, chest/shoulder height).
# Derived empirically: pitch=-0.7 lifts arm forward, wrist_pitch=-0.6 levels tray.
RIGHT_ARM_HOLD_POS = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD_POS  = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)


class BallBalanceEnv:
    """
    Batched environment wrapper. Supports n_envs parallel simulations.
    Legs/waist/left-arm are PD-frozen at the standing pose each step.
    Only the 7 right-arm joints are exposed as the RL action.
    """

    def __init__(
        self,
        show_viewer: bool = True,
        n_envs: int = 1,
        action_delta: float = 0.3,
        ball_vel_range: float = 0.0,
        action_conflict_penalty_scale: float = 0.05,
    ):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, -2.5, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=45,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=0.02),
        )

        self.action_delta = action_delta
        self.ball_vel_range = ball_vel_range
        self.action_conflict_penalty_scale = action_conflict_penalty_scale
        self.scene.add_entity(gs.morphs.Plane())

        # Spawn at z=0.79 to match the "stand" keyframe height
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=G1_XML, pos=(0.0, 0.0, 0.79)),
        )

        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(0.0, 0.0, 1.5)),
            material=gs.materials.Rigid(
                rho=BALL_MASS / (4 / 3 * np.pi * BALL_RADIUS**3)
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),
        )

        self.scene.build(n_envs=n_envs)
        self._cache_dof_indices()
        self.n_arm_dofs = len(self._both_arm_dofs)  # 14 for both arms
        self.reset()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _cache_dof_indices(self):
        # TODO this is cooked its bad to manually unbatch
        def dof(name):
            return self.robot.get_joint(name).dofs_idx_local[0]

        self._left_leg_dofs   = [dof(n) for n in LEFT_LEG_JOINTS]
        self._right_leg_dofs  = [dof(n) for n in RIGHT_LEG_JOINTS]
        self._waist_dofs      = [dof(n) for n in WAIST_JOINTS]
        self._left_arm_dofs   = [dof(n) for n in LEFT_ARM_JOINTS]
        self._right_arm_dofs  = [dof(n) for n in RIGHT_ARM_JOINTS]

        # Frozen: legs + waist only. Both arms active.
        self._frozen_dofs = (
            self._left_leg_dofs + self._right_leg_dofs + self._waist_dofs
        )
        self._frozen_pos = np.concatenate([
            STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS,
        ])
        # Both arms combined for a single control call in step()
        self._both_arm_dofs = self._right_arm_dofs + self._left_arm_dofs
        # store hold positions and simple joint limits as tensors on the scene device
        self._right_arm_hold = torch.tensor(RIGHT_ARM_HOLD_POS, device=gs.device, dtype=torch.float32)
        self._left_arm_hold = torch.tensor(LEFT_ARM_HOLD_POS, device=gs.device, dtype=torch.float32)
        # conservative joint limits: allow full rotation by default
        self._right_arm_lower = torch.full((len(self._right_arm_dofs),), -3.14159, device=gs.device)
        self._right_arm_upper = torch.full((len(self._right_arm_dofs),),  3.14159, device=gs.device)
        self._left_arm_lower = torch.full((len(self._left_arm_dofs),), -3.14159, device=gs.device)
        self._left_arm_upper = torch.full((len(self._left_arm_dofs),),  3.14159, device=gs.device)

    # ── reset / step ──────────────────────────────────────────────────────────

    def reset(self, envs_idx=None):
        self.robot.set_dofs_position(
            self._frozen_pos,
            dofs_idx_local=self._frozen_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            RIGHT_ARM_HOLD_POS,
            dofs_idx_local=self._right_arm_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            LEFT_ARM_HOLD_POS,
            dofs_idx_local=self._left_arm_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self._reset_ball(envs_idx=envs_idx)
        # Flush state changes for both full resets and per-env resets so the
        # next observation reflects the reset configuration immediately.
        self.scene.step()
        return self.get_obs()

    def _reset_ball(self, envs_idx=None):
        tray_pos = self.robot.get_link("tray").get_pos()  # (b, 3)
        if envs_idx is not None:
            tray_pos = tray_pos[envs_idx]
        b = tray_pos.shape[0]
        # random offset within tray bounds (half-size minus ball radius)
        xy_noise = (torch.rand(b, 2, device=tray_pos.device) - 0.5) * 2 * torch.tensor(
            [TRAY_SIZE[0] / 2 - BALL_RADIUS, TRAY_SIZE[1] / 2 - BALL_RADIUS],
            device=tray_pos.device,
        )
        ball_z = tray_pos[:, 2:3] + TRAY_SIZE[2] / 2 + BALL_RADIUS + 0.005
        spawn = torch.cat([tray_pos[:, :2] + xy_noise, ball_z], dim=-1)
        self.ball.set_pos(spawn, zero_velocity=True, envs_idx=envs_idx)
        if self.ball_vel_range > 0:
            # random XY velocity, capped at ball_vel_range m/s
            vel = torch.zeros(b, 6, device=tray_pos.device)
            vel[:, :2] = (torch.rand(b, 2, device=tray_pos.device) - 0.5) * 2 * self.ball_vel_range
            # set linear velocity; keep angular zero
            try:
                self.ball.set_dofs_velocity(vel, envs_idx=envs_idx)
            except Exception:
                # fallback if set_dofs_velocity isn't available
                self.ball.set_vel(vel[:, :3], envs_idx=envs_idx)

    def step(self, action: np.ndarray):
        """
        action : (b, 14) position targets (rad) — right arm (7) then left arm (7).
        Returns (obs, reward, done, info).
        """
        self.robot.control_dofs_position(
            self._frozen_pos,
            dofs_idx_local=self._frozen_dofs,
        )
        # TODO: normalize action to [-1, 1] range
        # action[:, :7] = right arm, action[:, 7:] = left arm
        action_tensor = torch.tensor(action, device=gs.device, dtype=torch.float32)
        # split into right / left
        right_action = action_tensor[:, : len(self._right_arm_dofs)]
        left_action = action_tensor[:, len(self._right_arm_dofs):]

        right_targets = self._right_arm_hold + right_action * self.action_delta
        right_targets = torch.clamp(right_targets, self._right_arm_lower, self._right_arm_upper)
        self.robot.control_dofs_position(right_targets, dofs_idx_local=self._right_arm_dofs)

        if left_action.shape[-1] == len(self._left_arm_dofs):
            left_targets = self._left_arm_hold + left_action * self.action_delta
            left_targets = torch.clamp(left_targets, self._left_arm_lower, self._left_arm_upper)
            self.robot.control_dofs_position(left_targets, dofs_idx_local=self._left_arm_dofs)

        self.scene.step()

        obs = self.get_obs()
        ball_pos, ball_vel, goal_pos = self._unpack_obs(obs)
        reward, done = self._compute_reward(ball_pos, goal_pos, ball_vel, right_action, left_action)
        return obs, reward, done, {}

    # ── observations ──────────────────────────────────────────────────────────
    
    def get_obs(self):
        """
        Flat vector:
          right_arm_pos  (b, 7)
          right_arm_vel  (b, 7)
          left_arm_pos   (b, 7)
          left_arm_vel   (b, 7)
          ball_pos       (b, 3)
          ball_vel       (b, 3)
          goal_pos       (b, 3)   — tray centre in world frame
        Returns (b, 37).
        """
        r_pos    = self.robot.get_dofs_position(dofs_idx_local=self._right_arm_dofs)
        r_vel    = self.robot.get_dofs_velocity(dofs_idx_local=self._right_arm_dofs)
        l_pos    = self.robot.get_dofs_position(dofs_idx_local=self._left_arm_dofs)
        l_vel    = self.robot.get_dofs_velocity(dofs_idx_local=self._left_arm_dofs)
        ball_pos = self.ball.get_pos()
        ball_vel = self.ball.get_vel()
        goal_pos = self.robot.get_link("tray").get_pos()
        return torch.cat([r_pos, r_vel, l_pos, l_vel, ball_pos, ball_vel, goal_pos], dim=-1)

    def _unpack_obs(self, obs):
        ball_pos = obs[:, 28:31]
        ball_vel = obs[:, 31:34]
        goal_pos = obs[:, 34:37]
        return ball_pos, ball_vel, goal_pos

    # ── reward ────────────────────────────────────────────────────────────────
    def _compute_reward(self, ball_pos, goal_pos, ball_vel, right_action, left_action):
        """
        Improved bimanual reward:
        + keep ball near tray center
        + encourage smooth / stable motion
        + discourage excessive arm motion
        + softer terminal penalty
        """
        xy_dist = torch.norm(
            ball_pos[:, :2] - goal_pos[:, :2],
            dim=-1
        )
        proximity = torch.exp(-1.5 * xy_dist)

        ball_speed = torch.norm(ball_vel, dim=-1)

        vel_pen = -0.1 * ball_speed

        all_actions = torch.cat([right_action, left_action], dim=-1)

        action_pen = -0.005 * torch.norm(all_actions, dim=-1)

        mirrored_left = left_action.clone()

        mirrored_left[:, 1] *= -1   # shoulder roll
        mirrored_left[:, 4] *= -1   # wrist roll

        coordination_pen = -0.01 * torch.norm(
            right_action - mirrored_left,
            dim=-1
        )

        fallen = ball_pos[:, 2] < (goal_pos[:, 2] - 0.15)

        fall_pen = torch.where(
            fallen,
            torch.full_like(proximity, -10.0),
            torch.zeros_like(proximity)
        )

        reward = (
            proximity
            + vel_pen
            + action_pen
            + coordination_pen
            + fall_pen
        )

        return reward, fallen

if __name__ == "__main__":
    print("this cannot be run standalone")

# TODO tasks
# [done] change the MJCF to have the tray directly attached as the hand, removing the hand visual entirely
# 3. teleop agent, spawns the env and with keyboard input to control hand behavior wasd for xyz and ijkl for pitch,yaw,roll
# 4. train agent, connects the env with RSL rl for training
# LOWER max joint velocities substantially
# probably also lower max joint torque
# add regularization terms in reward
