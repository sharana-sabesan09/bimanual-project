"""
Single-arm tray-grasp ball-balancing environment.

Unitree G1 loaded from USD (fixed base). The tray is a separate Genesis entity
that is kinematically pinned to the right_wrist_yaw_link each step via a
configurable local-frame offset — no tray in the USD itself.

Action  : (b, 7) — right_arm position delta, scaled by action_delta
Obs     : (b, 23) — right_arm_pos(7) + right_arm_vel(7) + ball_pos(3) + ball_vel(3) + goal_pos(3)
"""

import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

from source.tasks.base_env import BaseVecEnv

_ROOT = Path(__file__).parents[3]
G1_USD = str(_ROOT / "assets" / "g1-flattened-fixed.usd")

# TRAY_HALF_SIZE = (0.05, 0.05, 0.005)   
TRAY_HALF_SIZE = (0.18, 0.13, 0.005)

BALL_RADIUS    = 0.03
BALL_MASS      = 0.1
GOAL_PADDING   = 0.03

LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [
    "L_index_proximal_joint", "L_index_intermediate_joint",
    "L_middle_proximal_joint", "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",  "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",   "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint", "L_thumb_distal_joint",
]
RIGHT_HAND_JOINTS = [
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",  "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",   "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
]

RIGHT_ARM_FORCE_LIMITS = [25, 25, 25, 25, 25, 5, 5]

STAND_LEG_POS      = np.zeros(6,  dtype=np.float32)
STAND_WAIST_POS    = np.zeros(3,  dtype=np.float32)
STAND_LEFT_ARM_POS = np.array([0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0], dtype=np.float32)
RIGHT_ARM_HOLD_POS = np.array([
    0,  # shoulder pitch  
    np.deg2rad(-30),  # shoulder roll
    0.0,   # shoulder yaw 
    0.0,   # elbow joint 
    np.deg2rad(90),   # wrist roll 
    np.deg2rad(90),  # wrist pitch
    np.deg2rad(30)    # wrist yaw
    ], dtype=np.float32)

# Joints ordered as LEFT_HAND_JOINTS / RIGHT_HAND_JOINTS
LEFT_HAND_HOLD_POS = np.array([
    0.0,  # L_index_proximal_joint
    0.0,  # L_index_intermediate_joint
    0.0,  # L_middle_proximal_joint
    0.0,  # L_middle_intermediate_joint
    0.0,  # L_pinky_proximal_joint
    0.0,  # L_pinky_intermediate_joint
    0.0,  # L_ring_proximal_joint
    0.0,  # L_ring_intermediate_joint
    0.0,  # L_thumb_proximal_yaw_joint
    0.0,  # L_thumb_proximal_pitch_joint
    0.0,  # L_thumb_intermediate_joint
    0.0,  # L_thumb_distal_joint
], dtype=np.float32)

RIGHT_HAND_HOLD_POS = np.array([
    0.0,  # R_index_proximal_joint
    0.0,  # R_index_intermediate_joint
    0.0,  # R_middle_proximal_joint
    0.0,  # R_middle_intermediate_joint
    0.0,  # R_pinky_proximal_joint
    0.0,  # R_pinky_intermediate_joint
    0.0,  # R_ring_proximal_joint
    0.0,  # R_ring_intermediate_joint
    np.deg2rad(60),  # R_thumb_proximal_yaw_joint
    np.deg2rad(27),  # R_thumb_proximal_pitch_joint
    np.deg2rad(27),  # R_thumb_intermediate_joint
    np.deg2rad(27),  # R_thumb_distal_joint
], dtype=np.float32)


def _rotate_vec_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (n,3) by unit quaternions q (n,4) in w,x,y,z convention."""
    qw = q[:, 0:1]; qxyz = q[:, 1:]           # (n,1), (n,3)
    t = 2.0 * torch.linalg.cross(qxyz, v)     # (n,3)
    return v + qw * t + torch.linalg.cross(qxyz, t)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (n,4) quaternions in w,x,y,z convention."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class SingleArmTrayGraspEnv(BaseVecEnv):
    """
    Ball-balancing with the tray kinematically attached to the right wrist in Genesis.

    Parameters
    ----------
    tray_offset : (3,) float
        Position of the tray origin in the right_wrist_yaw_link local frame (metres).
        Default (0.10, 0.0, 0.0) places the tray 10 cm along the wrist's x-axis.
    tray_euler : (3,) float
        Orientation of the tray in the right_wrist_yaw_link local frame (degrees, xyz).
    """

    def __init__(
        self,
        show_viewer: bool = True,
        n_envs: int = 1,
        action_delta: float = 0.3,
        ball_vel_range: float = 1.0,
        max_episode_steps: int = 500,
        goal_randomization: bool = True,
        tray_offset: tuple = (0.1, 0.1, 0.02),
        tray_euler: tuple = (0.0, 0.0, 0.0),
        dt: float = 0.02,
        substeps: int = 4,
        debug_contacts: bool = False,
    ):
        self.action_delta       = action_delta
        self.ball_vel_range     = ball_vel_range
        self.goal_randomization = goal_randomization
        self.debug_contacts     = debug_contacts

        # Convert tray offset to tensors — stored as (1, 3) / (1, 4) for batched broadcast
        self._tray_offset_local = torch.tensor(tray_offset, dtype=torch.float32)
        tray_quat_scipy = Rot.from_euler("xyz", np.deg2rad(tray_euler)).as_quat()  # xyzw
        # Convert to Genesis w,x,y,z
        x, y, z, w = tray_quat_scipy
        self._tray_offset_quat_local = torch.tensor([w, x, y, z], dtype=torch.float32)

        super().__init__(show_viewer=show_viewer, n_envs=n_envs,
                         max_episode_steps=max_episode_steps, dt=dt, substeps=substeps)

    # ------------------------------------------------------------------ #
    # BaseVecEnv implementation                                            #
    # ------------------------------------------------------------------ #

    def _build_scene(self, n_envs: int):
        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)),
            visualize_contact=True,
            vis_mode="collision"
        )

        self.tray = self.scene.add_entity(
            gs.morphs.Box(size=tuple(s * 2 for s in TRAY_HALF_SIZE)),
            surface=gs.surfaces.Default(color=(0.8, 0.6, 0.3, 1.0)),
        )

        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(0.0, 0.0, 1.5)),
            material=gs.materials.Rigid(
                rho=BALL_MASS / (4 / 3 * np.pi * BALL_RADIUS**3)
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),
        )

        self.goal_marker = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.025, pos=(0.0, 0.0, 1.5), collision=False),
            surface=gs.surfaces.Default(color=(0.1, 0.9, 0.1, 0.8)),
        )

        self.scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))

    def _post_build_init(self):
        # USD joints are stored with full prim paths; match by the trailing segment.
        _joint_map = {
            j.name.split("/")[-1]: j
            for j in self.robot.joints
        }

        def dof(name):
            return _joint_map[name].dofs_idx_local[0]

        self._right_arm_dofs = [dof(n) for n in RIGHT_ARM_JOINTS]
        self._left_leg_dofs  = [dof(n) for n in LEFT_LEG_JOINTS]
        self._right_leg_dofs = [dof(n) for n in RIGHT_LEG_JOINTS]
        self._waist_dofs     = [dof(n) for n in WAIST_JOINTS]
        self._left_arm_dofs  = [dof(n) for n in LEFT_ARM_JOINTS]

        # Hand joints — freeze at specified pose (default zeros) if present
        self._left_hand_dofs  = [dof(n) for n in LEFT_HAND_JOINTS  if self._has_joint(n, _joint_map)]
        self._right_hand_dofs = [dof(n) for n in RIGHT_HAND_JOINTS if self._has_joint(n, _joint_map)]

        # Local link indices for right-hand joints (used for per-finger contact force queries)
        self._hand_link_idxs_local = [
            _joint_map[n].link.idx_local for n in RIGHT_HAND_JOINTS if n in _joint_map
        ]
        self._hand_link_names = [n for n in RIGHT_HAND_JOINTS if n in _joint_map]

        self.n_arm_dofs  = len(self._right_arm_dofs)
        self.n_hand_dofs = len(self._right_hand_dofs)
        self.n_action_dofs = self.n_arm_dofs + self.n_hand_dofs

        # Right hand is controlled via the action space, not frozen.
        self._frozen_dofs = (
            self._left_leg_dofs + self._right_leg_dofs
            + self._waist_dofs
            + self._left_arm_dofs
            + self._left_hand_dofs
        )
        self._frozen_pos = np.concatenate([
            STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS, STAND_LEFT_ARM_POS,
            LEFT_HAND_HOLD_POS[:len(self._left_hand_dofs)],
        ])

        # Action dofs = right arm + right hand, in that order
        self._action_dofs = self._right_arm_dofs + self._right_hand_dofs

        # Joint limits for the action dofs — used to map [-1, 1] → joint position
        _lower, _upper = self.robot.get_dofs_limit(dofs_idx_local=self._action_dofs)
        self._action_lower = _lower.to(gs.device)   # (n_action_dofs,)
        self._action_upper = _upper.to(gs.device)   # (n_action_dofs,)

        self._right_arm_hold = torch.tensor(
            RIGHT_ARM_HOLD_POS, device=gs.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.n_envs, -1).clone()

        self.robot.set_dofs_force_range(
            torch.tensor(RIGHT_ARM_FORCE_LIMITS, device=gs.device) * -1,
            torch.tensor(RIGHT_ARM_FORCE_LIMITS, device=gs.device),
            dofs_idx_local=self._right_arm_dofs,
        )

        _arm_kp  = [100, 100, 100, 100, 40, 40, 40]
        _arm_kd  = [10,  10,  10,  10,  4,  4,  4]
        _gain_dofs = (
            self._left_leg_dofs  + self._right_leg_dofs + self._waist_dofs
            + self._left_arm_dofs + self._right_arm_dofs
            + self._left_hand_dofs + self._right_hand_dofs
        )
        _gain_kp = (
            [200]*6 + [200]*6 + [200, 200, 200]
            + _arm_kp + _arm_kp
            + [20]*len(self._left_hand_dofs) + [20]*len(self._right_hand_dofs)
        )
        _gain_kd = (
            [10]*6  + [10]*6  + [10,  10,  10]
            + _arm_kd + _arm_kd
            + [1.2]*len(self._left_hand_dofs) + [1.2]*len(self._right_hand_dofs)
        )
        self.robot.set_dofs_kp(
            torch.tensor(_gain_kp, dtype=torch.float32, device=gs.device),
            dofs_idx_local=_gain_dofs,
        )
        self.robot.set_dofs_kv(
            torch.tensor(_gain_kd, dtype=torch.float32, device=gs.device),
            dofs_idx_local=_gain_dofs,
        )

        # USD link names also include prim-path prefix; find by suffix.
        palm_link_name = next(
            l.name for l in self.robot.links if l.name.endswith("R_middle_proximal")
        )
        self._middle_proximal_link = self.robot.get_link(palm_link_name)

        # Move tray offset tensors to device
        self._tray_offset_local  = self._tray_offset_local.to(gs.device)
        self._tray_offset_quat_local = self._tray_offset_quat_local.to(gs.device)

        # Cached state
        self.arm_pos   = torch.zeros(self.n_envs, 7, device=gs.device)
        self.arm_vel   = torch.zeros(self.n_envs, 7, device=gs.device)
        self.tray_pos  = torch.zeros(self.n_envs, 3, device=gs.device)
        self.ball_pos  = torch.zeros(self.n_envs, 3, device=gs.device)
        self.ball_vel  = torch.zeros(self.n_envs, 3, device=gs.device)
        self.goal_pos  = torch.zeros(self.n_envs, 3, device=gs.device)
        self.goal_offset = torch.zeros(self.n_envs, 3, device=gs.device)

        self.prev_actions = torch.zeros(self.n_envs, self.n_action_dofs, device=gs.device)

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(23,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.n_action_dofs,), dtype=np.float32
        )

    def _has_joint(self, name: str, joint_map: dict) -> bool:
        return name in joint_map

    # ------------------------------------------------------------------ #
    # Step hooks                                                           #
    # ------------------------------------------------------------------ #

    def _apply_action(self, action_tensor: torch.Tensor):
        # Map [-1, 1] → [lower, upper] for all action dofs
        pos = self._action_lower + (action_tensor + 1.0) * 0.5 * (self._action_upper - self._action_lower)
        self.robot.control_dofs_position(self._frozen_pos, dofs_idx_local=self._frozen_dofs)
        self.robot.control_dofs_position(pos[:, :self.n_arm_dofs],  dofs_idx_local=self._right_arm_dofs)
        self.robot.control_dofs_position(pos[:, self.n_arm_dofs:],  dofs_idx_local=self._right_hand_dofs)

    def _post_physics_step(self):
        self.arm_pos  = self.robot.get_dofs_position(dofs_idx_local=self._right_arm_dofs)
        self.arm_vel  = self.robot.get_dofs_velocity(dofs_idx_local=self._right_arm_dofs)
        self.tray_pos = self.tray.get_pos()
        self.ball_pos = self.ball.get_pos()
        self.ball_vel = self.ball.get_vel()

        # Goal lives at a fixed 2D offset from tray centre
        self.goal_pos = self.tray_pos + self.goal_offset
        self.goal_marker.set_pos(self.goal_pos)

        if self.debug_contacts:
            self._print_contact_forces()

    def _print_contact_forces(self):
        # ── per-finger net contact force (all contacts on each hand link) ──
        all_link_forces = self.robot.get_links_net_contact_force()  # (n_envs, n_links, 3)
        hand_forces = all_link_forces[0, self._hand_link_idxs_local, :]  # (n_hand, 3)
        hand_mags   = torch.norm(hand_forces, dim=-1)                     # (n_hand,)

        # ── tray-specific total: sum |force| over valid robot↔tray contact pairs ──
        contacts = self.robot.get_contacts(with_entity=self.tray)
        if "valid_mask" in contacts:
            # parallelised scene (n_envs >= 1)
            valid = contacts["valid_mask"][0]                  # (n_contacts,)
            fa    = contacts["force_a"][0][valid]              # (k, 3) force on robot geoms
            fb    = contacts["force_b"][0][valid]              # (k, 3) force on tray geoms
            tray_total = fb.norm(dim=-1).sum().item()
        else:
            fb = contacts["force_b"]
            tray_total = fb.norm(dim=-1).sum().item()

        print(f"\n── contact forces ─────────────────────────────")
        print(f"  tray-contact total (robot→tray): {tray_total:8.3f} N")
        print(f"  per-finger net (all contacts):")
        for name, mag in zip(self._hand_link_names, hand_mags.tolist()):
            bar = "█" * int(mag / 0.5)
            print(f"    {name:<40s} {mag:6.3f} N  {bar}")

    def _spawn_tray_at_wrist(self, envs_idx=None):
        """Place the tray at palm world-frame position + world-frame offset. Call after scene.step()."""
        idx = (
            torch.arange(self.n_envs, device=gs.device)
            if envs_idx is None else envs_idx
        )
        palm_pos = self._middle_proximal_link.get_pos()[idx]  # (b, 3) — world frame

        # Offset is added directly in world frame, no rotation needed
        tray_pos = palm_pos + self._tray_offset_local.to(gs.device)

        tray_quat = self._tray_offset_quat_local.to(gs.device).unsqueeze(0).expand(idx.shape[0], -1)

        self.tray.set_pos(tray_pos, zero_velocity=True, envs_idx=idx)
        self.tray.set_quat(tray_quat, zero_velocity=True, envs_idx=idx)

    # ------------------------------------------------------------------ #
    # Gym interface                                                        #
    # ------------------------------------------------------------------ #

    def get_obs(self) -> torch.Tensor:
        return torch.cat(
            [self.arm_pos, self.arm_vel, self.ball_pos, self.ball_vel, self.goal_pos],
            dim=-1,
        )

    def get_termination(self):
        tray_z = self.tray_pos[:, 2]
        # TODO FIX ME THIS IS WRONG I CHANGED IT FIX ME FIX ME 
        self.terminated = self.ball_pos[:, 2] < (tray_z - 20000)
        self.truncated  = self.episode_length_buf >= self.max_episode_steps
        return self.terminated, self.truncated

    def _compute_reward(self, action_tensor: torch.Tensor) -> torch.Tensor:
        xy_dist   = torch.norm(self.ball_pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        proximity = torch.exp(-5.0 * xy_dist)
        vel_pen   = -0.05 * torch.norm(self.ball_vel, dim=-1)
        action_pen = -0.0001 * torch.norm(action_tensor, dim=-1)
        fall_pen  = torch.where(
            self.terminated,
            torch.full_like(proximity, -10.0),
            torch.zeros_like(proximity),
        )
        self.prev_actions.copy_(action_tensor.detach())
        return proximity + 0.2 + vel_pen + action_pen + fall_pen

    # ------------------------------------------------------------------ #
    # Reset helpers                                                        #
    # ------------------------------------------------------------------ #

    def _reset_env(self, envs_idx=None):
        self.robot.set_dofs_position(
            self._frozen_pos,
            dofs_idx_local=self._frozen_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        hold = self._right_arm_hold if envs_idx is None else self._right_arm_hold[envs_idx]
        self.robot.set_dofs_position(
            hold,
            dofs_idx_local=self._right_arm_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        hand_hold = torch.tensor(
            RIGHT_HAND_HOLD_POS[:self.n_hand_dofs], device=gs.device, dtype=torch.float32
        ).unsqueeze(0).expand(hold.shape[0], -1)
        self.robot.set_dofs_position(
            hand_hold,
            dofs_idx_local=self._right_hand_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        if envs_idx is None:
            self.prev_actions.zero_()
            self.goal_offset.zero_()
        else:
            self.prev_actions[envs_idx] = 0.0
            self.goal_offset[envs_idx]  = 0.0

        # Ball will be placed after first scene.step() via reset() in base class

    def reset(self, envs_idx=None, seed=None, options=None):
        obs, info = super().reset(envs_idx=envs_idx, seed=seed, options=options)
        # After super().reset() the scene has stepped once so wrist FK is accurate.
        # Spawn the free tray near the hand and let physics take over from here.
        self._reset_goal_offset(envs_idx)
        self._spawn_tray_at_wrist(envs_idx)
        self._reset_ball(envs_idx)
        return obs, info

    def _reset_ball(self, envs_idx=None):
        idx = (
            torch.arange(self.n_envs, device=gs.device)
            if envs_idx is None else envs_idx
        )
        tray_p = self.tray_pos[idx]
        b = tray_p.shape[0]
        half = torch.tensor(TRAY_HALF_SIZE[:2], device=gs.device)
        xy_noise = (torch.rand(b, 2, device=gs.device) - 0.5) * 2 * (half - BALL_RADIUS)
        ball_z   = tray_p[:, 2:3] + TRAY_HALF_SIZE[2] + BALL_RADIUS + 0.005
        # TODO FIX ME FIX ME FIX ME 
        ball_z += -500
        self.ball.set_pos(
            torch.cat([tray_p[:, :2] + xy_noise, ball_z], dim=-1),
            zero_velocity=True,
            envs_idx=idx,
        )
        if self.ball_vel_range > 0:
            vel = torch.zeros(b, 6, device=gs.device)
            vel[:, :2] = (torch.rand(b, 2, device=gs.device) - 0.5) * 2 * self.ball_vel_range
            try:
                self.ball.set_dofs_velocity(vel, envs_idx=idx)
            except Exception:
                self.ball.set_vel(vel[:, :3], envs_idx=idx)

    def _reset_goal_offset(self, envs_idx=None):
        if not self.goal_randomization:
            return
        idx = (
            torch.arange(self.n_envs, device=gs.device)
            if envs_idx is None else envs_idx
        )
        half = torch.tensor(TRAY_HALF_SIZE[:2], device=gs.device) - GOAL_PADDING
        xy = (torch.rand(idx.shape[0], 2, device=gs.device) - 0.5) * 2 * half
        self.goal_offset[idx] = torch.cat([xy, torch.zeros(idx.shape[0], 1, device=gs.device)], dim=-1)
