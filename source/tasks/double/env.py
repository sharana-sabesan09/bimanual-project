"""
Dual-arm ball-balancing environment.

Unitree G1 — tray welded to both wrists via MuJoCo equality constraints.
Policy controls all 14 arm joints via position-delta targets.
Legs / waist are PD-frozen.

Action  : (b, 14) — right_arm_delta(7) + left_arm_delta(7), scaled by action_delta
Obs     : (b, 37) — r_pos(7) + r_vel(7) + l_pos(7) + l_vel(7) + ball_pos(3) + ball_vel(3) + goal_pos(3)
"""

import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from pathlib import Path

from source.tasks.base_env import BaseVecEnv

_ROOT = Path(__file__).parents[3]
G1_XML = str(_ROOT / "assets" / "mujoco_menagerie" / "unitree_g1" / "g1_dual_arm.xml")

TRAY_SIZE   = (0.36, 0.26, 0.01)
BALL_RADIUS = 0.03
BALL_MASS   = 0.1

LEFT_LEG_JOINTS  = ["left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
                    "left_knee_joint",  "left_ankle_pitch_joint",  "left_ankle_roll_joint"]
RIGHT_LEG_JOINTS = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
WAIST_JOINTS     = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS  = ["left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint", "left_elbow_joint",
                    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
RIGHT_ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint",
                    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

STAND_LEG_POS      = np.zeros(6, dtype=np.float32)
STAND_WAIST_POS    = np.zeros(3, dtype=np.float32)
RIGHT_ARM_HOLD_POS = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD_POS  = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)


class DualArmBallBalanceEnv(BaseVecEnv):

    def __init__(self, show_viewer=True, n_envs=1, action_delta=0.3,
                 ball_vel_range=0.0, max_episode_steps=500,
                 action_conflict_penalty_scale=0.05):
        self.action_delta                  = action_delta
        self.ball_vel_range                = ball_vel_range
        self.action_conflict_penalty_scale = action_conflict_penalty_scale
        super().__init__(show_viewer=show_viewer, n_envs=n_envs,
                         max_episode_steps=max_episode_steps)

    # ------------------------------------------------------------------ #
    # BaseVecEnv implementation                                            #
    # ------------------------------------------------------------------ #

    def _build_scene(self, n_envs: int):
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=G1_XML, pos=(0.0, 0.0, 0.79)),
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(0.0, 0.0, 1.5)),
            material=gs.materials.Rigid(rho=BALL_MASS / (4/3 * np.pi * BALL_RADIUS**3)),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),
        )
        self.scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))

    def _post_build_init(self):
        def dof(name):
            return self.robot.get_joint(name).dofs_idx_local[0]

        self._left_leg_dofs  = [dof(n) for n in LEFT_LEG_JOINTS]
        self._right_leg_dofs = [dof(n) for n in RIGHT_LEG_JOINTS]
        self._waist_dofs     = [dof(n) for n in WAIST_JOINTS]
        self._left_arm_dofs  = [dof(n) for n in LEFT_ARM_JOINTS]
        self._right_arm_dofs = [dof(n) for n in RIGHT_ARM_JOINTS]

        self._frozen_dofs   = self._left_leg_dofs + self._right_leg_dofs + self._waist_dofs
        self._frozen_pos    = np.concatenate([STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS])
        self._both_arm_dofs = self._right_arm_dofs + self._left_arm_dofs

        self._right_arm_hold  = torch.tensor(RIGHT_ARM_HOLD_POS, device=gs.device, dtype=torch.float32)
        self._left_arm_hold   = torch.tensor(LEFT_ARM_HOLD_POS,  device=gs.device, dtype=torch.float32)
        self._right_arm_lower = torch.full((7,), -3.14159, device=gs.device)
        self._right_arm_upper = torch.full((7,),  3.14159, device=gs.device)
        self._left_arm_lower  = torch.full((7,), -3.14159, device=gs.device)
        self._left_arm_upper  = torch.full((7,),  3.14159, device=gs.device)

        self.n_arm_dofs   = len(self._both_arm_dofs)
        self.prev_actions = torch.zeros(self.n_envs, self.n_arm_dofs, device=gs.device)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1.0, 1.0, shape=(self.n_arm_dofs,), dtype=np.float32)

    def _reset_env(self, envs_idx=None):
        self.robot.set_dofs_position(self._frozen_pos, dofs_idx_local=self._frozen_dofs,
                                     zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_dofs_position(RIGHT_ARM_HOLD_POS, dofs_idx_local=self._right_arm_dofs,
                                     zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_dofs_position(LEFT_ARM_HOLD_POS, dofs_idx_local=self._left_arm_dofs,
                                     zero_velocity=True, envs_idx=envs_idx)
        self._reset_ball(envs_idx)

        if envs_idx is None:
            self.prev_actions.zero_()
        else:
            self.prev_actions[envs_idx] = 0.0

    def _apply_action(self, action_tensor: torch.Tensor):
        self.robot.control_dofs_position(self._frozen_pos, dofs_idx_local=self._frozen_dofs)
        right = action_tensor[:, :7]
        left  = action_tensor[:, 7:]
        self.robot.control_dofs_position(
            torch.clamp(self._right_arm_hold + right * self.action_delta,
                        self._right_arm_lower, self._right_arm_upper),
            dofs_idx_local=self._right_arm_dofs,
        )
        self.robot.control_dofs_position(
            torch.clamp(self._left_arm_hold + left * self.action_delta,
                        self._left_arm_lower, self._left_arm_upper),
            dofs_idx_local=self._left_arm_dofs,
        )

    def get_obs(self) -> torch.Tensor:
        r_pos    = self.robot.get_dofs_position(dofs_idx_local=self._right_arm_dofs)
        r_vel    = self.robot.get_dofs_velocity(dofs_idx_local=self._right_arm_dofs)
        l_pos    = self.robot.get_dofs_position(dofs_idx_local=self._left_arm_dofs)
        l_vel    = self.robot.get_dofs_velocity(dofs_idx_local=self._left_arm_dofs)
        ball_pos = self.ball.get_pos()
        ball_vel = self.ball.get_vel()
        goal_pos = self.robot.get_link("tray").get_pos()
        return torch.cat([r_pos, r_vel, l_pos, l_vel, ball_pos, ball_vel, goal_pos], dim=-1)

    def get_termination(self, obs: torch.Tensor):
        ball_pos = obs[:, 28:31]
        goal_pos = obs[:, 34:37]
        self.terminated = ball_pos[:, 2] < (goal_pos[:, 2] - 0.15)
        self.truncated  = self.episode_length_buf >= self.max_episode_steps
        return self.terminated, self.truncated

    def _compute_reward(self, obs: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        ball_pos = obs[:, 28:31]
        ball_vel = obs[:, 31:34]
        goal_pos = obs[:, 34:37]

        xy_dist        = torch.norm(ball_pos[:, :2] - goal_pos[:, :2], dim=-1)
        proximity      = torch.exp(-1.5 * xy_dist)

        ball_speed = torch.norm(ball_vel, dim=-1)
        vel_pen    = -0.1 * ball_speed

        action_pen = -0.005 * torch.norm(action_tensor, dim=-1)

        # Slicing actions for coordination penalty
        right_action = action_tensor[:, :7]
        left_action  = action_tensor[:, 7:]

        # Mirror the left arm to check coordination with the right
        mirrored_left = left_action.clone()
        mirrored_left[:, 1] *= -1   # shoulder roll
        mirrored_left[:, 4] *= -1   # wrist roll

        coordination_pen = -0.01 * torch.norm(right_action - mirrored_left, dim=-1)

        # self.terminated is updated in get_termination() before this call
        fall_pen = torch.where(self.terminated, torch.full_like(proximity, -10.0), torch.zeros_like(proximity))

        self.prev_actions.copy_(action_tensor.detach())

        return proximity + vel_pen + action_pen + coordination_pen + fall_pen

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

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
        self.ball.set_pos(torch.cat([tray_pos[:, :2] + xy_noise, ball_z], dim=-1),
                          zero_velocity=True, envs_idx=envs_idx)
        if self.ball_vel_range > 0:
            vel = torch.zeros(b, 6, device=tray_pos.device)
            vel[:, :2] = (torch.rand(b, 2, device=tray_pos.device) - 0.5) * 2 * self.ball_vel_range
            try:
                self.ball.set_dofs_velocity(vel, envs_idx=envs_idx)
            except Exception:
                self.ball.set_vel(vel[:, :3], envs_idx=envs_idx)
