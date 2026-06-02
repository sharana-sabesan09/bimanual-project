"""
Single-arm tray-grasp ball-balancing environment.

Unitree G1 loaded from USD (fixed base). The tray is a separate Genesis entity
spawned near the right hand at reset; the policy must grasp it and balance a ball.

Action  : (b, 19) — right_arm(7) + right_hand(12), each in [-1,1] → [joint_lower, joint_upper]
Obs     : (b, 53) — arm_pos(7) + arm_vel(7) + hand_pos(12) + hand_vel(12)
                     + tray_pos(3) + tray_z_axis(3) + ball_pos(3) + ball_vel(3) + goal_pos(3)

Reward is staged (curriculum_stage 1/2/3):
  Stage 1 — contact count (fingers touching tray) + action smoothness
  Stage 2 — + tray upright orientation
  Stage 3 — + ball XY proximity + ball velocity penalty
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
        ball_vel_range: float = 0.0,
        max_episode_steps: int = 500,
        goal_randomization: bool = True,
        tray_offset: tuple = (0.1, 0.1, 0.02),
        tray_euler: tuple = (0.0, 0.0, 0.0),
        dt: float = 0.02,
        substeps: int = 4,
        debug_contacts: bool = False,
        debug: bool = False,
        curriculum_stage: int = 3,
    ):
        self.show_viewer        = show_viewer
        self.ball_vel_range     = ball_vel_range
        self.goal_randomization = goal_randomization
        self.debug_contacts     = debug_contacts
        self.curriculum_stage   = curriculum_stage

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
            **({ "visualize_contact": True, "vis_mode": "collision" } if self.debug_contacts else {}),
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
        self.arm_pos           = torch.zeros(self.n_envs, 7,  device=gs.device)
        self.arm_vel           = torch.zeros(self.n_envs, 7,  device=gs.device)
        self.hand_pos          = torch.zeros(self.n_envs, self.n_hand_dofs, device=gs.device)
        self.hand_vel          = torch.zeros(self.n_envs, self.n_hand_dofs, device=gs.device)
        self.tray_pos          = torch.zeros(self.n_envs, 3,  device=gs.device)
        self.tray_quat         = torch.zeros(self.n_envs, 4,  device=gs.device)
        self.tray_quat[:, 0]   = 1.0   # identity
        self.tray_z_world      = torch.zeros(self.n_envs, 3,  device=gs.device)
        self.tray_z_world[:, 2] = 1.0
        self.ball_pos          = torch.zeros(self.n_envs, 3,  device=gs.device)
        self.ball_vel          = torch.zeros(self.n_envs, 3,  device=gs.device)
        self.goal_pos          = torch.zeros(self.n_envs, 3,  device=gs.device)
        # goal_marker_offset is in tray local frame; rotated by tray_quat each step
        self.goal_marker_offset = torch.zeros(self.n_envs, 3,  device=gs.device)

        self.prev_actions = torch.zeros(self.n_envs, self.n_action_dofs, device=gs.device)
        self._tray_init_z = torch.zeros(self.n_envs, device=gs.device)

        # Pre-allocated constants reused every step
        self._world_z = torch.zeros(self.n_envs, 3, device=gs.device)
        self._world_z[:, 2] = 1.0
        self._hand_hold = torch.tensor(
            RIGHT_HAND_HOLD_POS[:self.n_hand_dofs], device=gs.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.n_envs, -1).clone()

        # arm(7)+vel(7) + hand(12)+vel(12) + tray_pos(3)+tray_z(3) + ball_pos(3)+vel(3) + goal(3) = 53
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(53,), dtype=np.float32
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
        self.hand_pos = self.robot.get_dofs_position(dofs_idx_local=self._right_hand_dofs)
        self.hand_vel = self.robot.get_dofs_velocity(dofs_idx_local=self._right_hand_dofs)
        self.tray_pos  = self.tray.get_pos()
        self.tray_quat = self.tray.get_quat()  # (n_envs, 4) w,x,y,z
        self.ball_pos  = self.ball.get_pos()
        self.ball_vel  = self.ball.get_vel()

        # Tray z-axis in world frame (for obs + upright reward)
        self.tray_z_world = _rotate_vec_by_quat(self._world_z, self.tray_quat)

        # Goal in world frame: rotate tray-frame offset by current tray orientation
        self.goal_pos = _rotate_vec_by_quat(self.goal_marker_offset, self.tray_quat) + self.tray_pos
        if self.show_viewer:
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
        return torch.cat([
            self.arm_pos,      # (b,  7)
            self.arm_vel,      # (b,  7)
            self.hand_pos,     # (b, 12)
            self.hand_vel,     # (b, 12)
            self.tray_pos,     # (b,  3)
            self.tray_z_world, # (b,  3)
            self.ball_pos,     # (b,  3)
            self.ball_vel,     # (b,  3)
            self.goal_pos,     # (b,  3)
        ], dim=-1)

    def get_termination(self):
        z_floor = self._tray_init_z - 0.4
        self.terminated = (self.tray_pos[:, 2] < z_floor) | (self.ball_pos[:, 2] < z_floor)
        self.truncated  = self.episode_length_buf >= self.max_episode_steps
        return self.terminated, self.truncated

    def _compute_reward(self, action_tensor: torch.Tensor) -> torch.Tensor:
        # ── Stage 1: grasp contact ─────────────────────────────────────────
        # Count right-hand fingers with significant net contact force.
        # 1 N threshold suppresses self-contact noise between finger links.
        all_link_forces = self.robot.get_links_net_contact_force()   # (n_envs, n_links, 3)
        hand_mags = torch.norm(
            all_link_forces[:, self._hand_link_idxs_local, :], dim=-1
        )                                                              # (n_envs, n_hand)
        n_contact  = (hand_mags > 1.0).float().sum(dim=-1)            # (n_envs,)
        r_contact  = torch.clamp(n_contact / 3.0, 0.0, 1.0)          # saturates at 3 fingers

        # Action-rate smoothness
        r_action_pen = torch.norm(action_tensor - self.prev_actions, dim=-1)

        # Fall / drop
        r_fall_pen = torch.where(
            self.terminated,
            torch.ones(self.terminated.shape, device=gs.device),
            torch.zeros(self.terminated.shape, device=gs.device),
        )

        # ── Stage 2: tray orientation ──────────────────────────────────────
        tilt_sq   = self.tray_z_world[:, 0]**2 + self.tray_z_world[:, 1]**2
        r_upright = torch.exp(-10.0 * tilt_sq)

        # ── Stage 3: ball balancing ────────────────────────────────────────
        ball_dist  = torch.norm(self.ball_pos - self.goal_pos, dim=-1)
        r_ball_pos = 1.0 / (1.0 + 5.0 * ball_dist)
        r_ball_vel = 1.0 / (1.0 + torch.norm(self.ball_vel, dim=-1))

        # ── Weights ────────────────────────────────────────────────────────
        w_contact    = 1.0  if self.curriculum_stage >= 1 else 0.0
        w_upright    = 0.5  if self.curriculum_stage >= 2 else 0.0
        w_ball_pos   = 2.0  if self.curriculum_stage >= 3 else 0.0
        w_ball_vel   = 0.3  if self.curriculum_stage >= 3 else 0.0
        w_action_pen = 0.0001
        w_fall_pen   = 10.0

        final_reward = (
              w_contact    * r_contact
            + w_upright    * r_upright
            + w_ball_pos   * r_ball_pos
            + w_ball_vel   * r_ball_vel
            - w_action_pen * r_action_pen
            - w_fall_pen   * r_fall_pen
        )

        self.prev_actions.copy_(action_tensor.detach())

        log = self.extras["log"]
        log["r_contact"]    = r_contact.mean()
        log["r_upright"]    = r_upright.mean()
        log["r_ball_pos"]   = r_ball_pos.mean()
        log["r_ball_vel"]   = r_ball_vel.mean()
        log["r_action_pen"] = r_action_pen.mean()
        log["r_fall_pen"]   = r_fall_pen.mean()
        log["n_fingers"]    = n_contact.mean()
        log["ball_dist"]    = ball_dist.mean()
        log["tray_tilt"]    = tilt_sq.sqrt().mean()

        return final_reward

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
        hold      = self._right_arm_hold if envs_idx is None else self._right_arm_hold[envs_idx]
        hand_hold = self._hand_hold      if envs_idx is None else self._hand_hold[envs_idx]
        self.robot.set_dofs_position(
            hold,
            dofs_idx_local=self._right_arm_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            hand_hold,
            dofs_idx_local=self._right_hand_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        if envs_idx is None:
            self.prev_actions.zero_()
        else:
            self.prev_actions[envs_idx] = 0.0

        # Ball will be placed after first scene.step() via reset() in base class

    def reset(self, envs_idx=None, seed=None, options=None):
        if envs_idx is None:
            # Full reset (init / manual reset): call super() which does scene.step() for FK.
            obs, info = super().reset(envs_idx=None, seed=seed, options=options)
        else:
            # Partial reset (done envs during training): skip the extra scene.step().
            # Wrist FK is already accurate from the last _post_physics_step().
            self._reset_env(envs_idx)
            self.episode_length_buf[envs_idx] = 0
            obs, info = self.get_obs(), {}

        self._reset_goal_offset(envs_idx)
        self._spawn_tray_at_wrist(envs_idx)
        idx = torch.arange(self.n_envs, device=gs.device) if envs_idx is None else envs_idx
        self.tray_pos[idx]  = self.tray.get_pos()[idx]
        self.tray_quat[idx] = self.tray.get_quat()[idx]
        self._tray_init_z[idx] = self.tray_pos[idx, 2]
        self.goal_pos[idx]  = (
            _rotate_vec_by_quat(self.goal_marker_offset[idx], self.tray_quat[idx])
            + self.tray_pos[idx]
        )
        if self.show_viewer:
            self.goal_marker.set_pos(self.goal_pos)
        self._reset_ball(envs_idx)
        return obs, info

    def _reset_ball(self, envs_idx=None):
        idx = (
            torch.arange(self.n_envs, device=gs.device)
            if envs_idx is None else envs_idx
        )
        tray_p = self.tray.get_pos()[idx]  # fresh read — cache may be pre-spawn
        b = tray_p.shape[0]
        half = torch.tensor(TRAY_HALF_SIZE[:2], device=gs.device)
        xy_noise = (torch.rand(b, 2, device=gs.device) - 0.5) * 2 * (half - BALL_RADIUS)
        ball_z   = tray_p[:, 2:3] + TRAY_HALF_SIZE[2] + BALL_RADIUS + 0.005
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
        idx = (
            torch.arange(self.n_envs, device=gs.device)
            if envs_idx is None else envs_idx
        )
        b = idx.shape[0]
        if self.goal_randomization:
            half = torch.tensor(TRAY_HALF_SIZE[:2], device=gs.device) - GOAL_PADDING
            xy = (torch.rand(b, 2, device=gs.device) - 0.5) * 2 * half
        else:
            xy = torch.zeros(b, 2, device=gs.device)
        # z offset in tray frame: top surface + ball radius = ball centre above tray
        z = torch.full((b, 1), TRAY_HALF_SIZE[2] + BALL_RADIUS, device=gs.device)
        self.goal_marker_offset[idx] = torch.cat([xy, z], dim=-1)
