"""
Dual-arm ball-balancing environment.

Unitree G1 with a tray welded to both wrists via MuJoCo equality constraints.
The policy controls all 14 arm joints (right 7 + left 7) via position delta targets.
Legs and waist are PD-frozen.

Action space : (b, 14) — right_arm_delta(7) + left_arm_delta(7), scaled by action_delta
Observation  : (b, 37) — right_arm_pos(7) + right_arm_vel(7) + left_arm_pos(7)
                          + left_arm_vel(7) + ball_pos(3) + ball_vel(3) + goal_pos(3)
"""

import os
import numpy as np
import torch
import genesis as gs
from pathlib import Path

_ROOT = Path(__file__).parents[3]
G1_XML = str(_ROOT / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_dual_arm.xml")

TRAY_SIZE   = (0.36, 0.26, 0.01)
BALL_RADIUS = 0.03
BALL_MASS   = 0.1

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

STAND_LEG_POS      = np.zeros(6, dtype=np.float32)
STAND_WAIST_POS    = np.zeros(3, dtype=np.float32)
RIGHT_ARM_HOLD_POS = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD_POS  = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)


class DualArmBallBalanceEnv:
    """Batched dual-arm ball-balance env. Both arms (14 DOF total) are RL-controlled."""

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

        self.scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))
        self._cache_dof_indices()
        self.n_arm_dofs = len(self._both_arm_dofs)  # 14
        self.reset()

    def _cache_dof_indices(self):
        def dof(name):
            return self.robot.get_joint(name).dofs_idx_local[0]

        self._left_leg_dofs  = [dof(n) for n in LEFT_LEG_JOINTS]
        self._right_leg_dofs = [dof(n) for n in RIGHT_LEG_JOINTS]
        self._waist_dofs     = [dof(n) for n in WAIST_JOINTS]
        self._left_arm_dofs  = [dof(n) for n in LEFT_ARM_JOINTS]
        self._right_arm_dofs = [dof(n) for n in RIGHT_ARM_JOINTS]

        self._frozen_dofs = self._left_leg_dofs + self._right_leg_dofs + self._waist_dofs
        self._frozen_pos  = np.concatenate([STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS])
        self._both_arm_dofs = self._right_arm_dofs + self._left_arm_dofs

        self._right_arm_hold  = torch.tensor(RIGHT_ARM_HOLD_POS, device=gs.device, dtype=torch.float32)
        self._left_arm_hold   = torch.tensor(LEFT_ARM_HOLD_POS,  device=gs.device, dtype=torch.float32)
        self._right_arm_lower = torch.full((len(self._right_arm_dofs),), -3.14159, device=gs.device)
        self._right_arm_upper = torch.full((len(self._right_arm_dofs),),  3.14159, device=gs.device)
        self._left_arm_lower  = torch.full((len(self._left_arm_dofs),),  -3.14159, device=gs.device)
        self._left_arm_upper  = torch.full((len(self._left_arm_dofs),),   3.14159, device=gs.device)

    def reset(self, envs_idx=None):
        self.robot.set_dofs_position(
            self._frozen_pos, dofs_idx_local=self._frozen_dofs,
            zero_velocity=True, envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            RIGHT_ARM_HOLD_POS, dofs_idx_local=self._right_arm_dofs,
            zero_velocity=True, envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            LEFT_ARM_HOLD_POS, dofs_idx_local=self._left_arm_dofs,
            zero_velocity=True, envs_idx=envs_idx,
        )
        self._reset_ball(envs_idx=envs_idx)

        n_envs = self.scene.n_envs
        if envs_idx is None:
            self.prev_actions = torch.zeros(n_envs, self.n_arm_dofs, device=gs.device)
        else:
            if not hasattr(self, "prev_actions"):
                self.prev_actions = torch.zeros(n_envs, self.n_arm_dofs, device=gs.device)
            self.prev_actions[envs_idx] = 0.0

        self.scene.step()
        return self.get_obs()

    def _reset_ball(self, envs_idx=None):
        tray_pos = self.robot.get_link("tray").get_pos()
        if envs_idx is not None:
            tray_pos = tray_pos[envs_idx]
        b = tray_pos.shape[0]
        xy_noise = (torch.rand(b, 2, device=tray_pos.device) - 0.5) * 2 * torch.tensor(
            [TRAY_SIZE[0] / 2 - BALL_RADIUS, TRAY_SIZE[1] / 2 - BALL_RADIUS],
            device=tray_pos.device,
        )
        ball_z = tray_pos[:, 2:3] + TRAY_SIZE[2] / 2 + BALL_RADIUS + 0.005
        spawn  = torch.cat([tray_pos[:, :2] + xy_noise, ball_z], dim=-1)
        self.ball.set_pos(spawn, zero_velocity=True, envs_idx=envs_idx)
        if self.ball_vel_range > 0:
            vel = torch.zeros(b, 6, device=tray_pos.device)
            vel[:, :2] = (torch.rand(b, 2, device=tray_pos.device) - 0.5) * 2 * self.ball_vel_range
            try:
                self.ball.set_dofs_velocity(vel, envs_idx=envs_idx)
            except Exception:
                self.ball.set_vel(vel[:, :3], envs_idx=envs_idx)

    def step(self, action: np.ndarray):
        """action: (b, 14) — right_arm_delta(7) + left_arm_delta(7)."""
        self.robot.control_dofs_position(self._frozen_pos, dofs_idx_local=self._frozen_dofs)

        action_tensor = torch.tensor(action, device=gs.device, dtype=torch.float32)
        right_action = action_tensor[:, :len(self._right_arm_dofs)]
        left_action  = action_tensor[:, len(self._right_arm_dofs):]

        right_targets = torch.clamp(
            self._right_arm_hold + right_action * self.action_delta,
            self._right_arm_lower, self._right_arm_upper,
        )
        self.robot.control_dofs_position(right_targets, dofs_idx_local=self._right_arm_dofs)

        if left_action.shape[-1] == len(self._left_arm_dofs):
            left_targets = torch.clamp(
                self._left_arm_hold + left_action * self.action_delta,
                self._left_arm_lower, self._left_arm_upper,
            )
            self.robot.control_dofs_position(left_targets, dofs_idx_local=self._left_arm_dofs)

        self.scene.step()

        obs = self.get_obs()
        ball_pos, ball_vel, goal_pos = self._unpack_obs(obs)
        reward, done = self._compute_reward(ball_pos, goal_pos, ball_vel, right_action, left_action)
        return obs, reward, done, {}

    def get_obs(self):
        """Returns (b, 37): r_pos(7)+r_vel(7)+l_pos(7)+l_vel(7)+ball_pos(3)+ball_vel(3)+goal_pos(3)."""
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

    def _compute_reward(self, ball_pos, goal_pos, ball_vel, right_action, left_action):
        xy_dist   = torch.norm(ball_pos[:, :2] - goal_pos[:, :2], dim=-1)
        proximity = 1.0 - 0.5 * xy_dist + 0.5 * torch.exp(-2.0 * xy_dist)
        vel_pen   = -0.1 * torch.norm(ball_vel, dim=-1)
        all_actions    = torch.cat([right_action, left_action], dim=-1)
        action_pen     = -0.003 * torch.norm(all_actions, dim=-1)
        smoothness_pen = -0.02  * torch.norm(all_actions - self.prev_actions, dim=-1)
        alive_bonus    = 0.2
        fallen   = ball_pos[:, 2] < (goal_pos[:, 2] - 0.15)
        fall_pen = torch.where(fallen, torch.full_like(proximity, -10.0), torch.zeros_like(proximity))
        reward = proximity + alive_bonus + vel_pen + action_pen + smoothness_pen + fall_pen
        self.prev_actions = all_actions.detach()
        return reward, fallen


if __name__ == "__main__":
    print("this cannot be run standalone")
