"""
Single-arm tray-grasp ball-balancing environment.

Unitree G1 loaded from USD (fixed base). The tray is a separate Genesis entity
spawned near the right hand at reset; the policy must grasp it and balance a ball.

Action  : (b, 19) — right_arm(7) + right_hand(12), each in [-1,1] → [joint_lower, joint_upper]
Obs     : (b, 147) — arm_pos(7) + arm_vel(7) + hand_pos(12) + hand_vel(12)
                      + tray_pos(3) + tray_z_axis(3) + ball_pos(3) + ball_vel(3) + goal_pos(3)
                      + action_lower(19) + action_upper(19)
                      + hand_contact_forces(36) + total_contact_force(1)
                      + prev_actions(19)

Reward is staged (curriculum_stage 1/2/3):
  Stage 1 — contact count (fingers touching tray) + action smoothness
  Stage 2 — + tray upright orientation
  Stage 3 — + ball XY proximity + ball velocity penalty
"""

import numpy as np
import torch
import torch.profiler
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
BALL_MASS      = 0.1   # kg

# Tray mass in kg.  Genesis derives density from mass / volume automatically below.
# At rest on the hand the tray contact force ≈ TRAY_MASS * 9.81 N, so this is the
# number to tune if the resting contact force in the obs looks wrong.
# Default ~22 N resting → mass ≈ 2.2 kg (too heavy); 0.3 kg gives ~3 N resting.
TRAY_MASS      = 0.3   # kg  ← change this to adjust tray weight

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

RIGHT_ARM_FORCE_LIMITS = [10, 10, 10, 10, 10, 5, 5]

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
    with torch.profiler.record_function("rotate_vec_by_quat"):
        qw = q[:, 0:1]; qxyz = q[:, 1:]
        t = 2.0 * torch.linalg.cross(qxyz, v)
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
        action_delta: float = 0.3,  # accepted for play.py compat; unused (env maps actions to joint limits directly)
    ):
        self.show_viewer        = show_viewer
        self.ball_vel_range     = ball_vel_range
        self.goal_randomization = goal_randomization
        self.debug_contacts     = debug_contacts
        self.curriculum_stage   = curriculum_stage
        self.debug              = debug

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

        _tray_vol = (2*TRAY_HALF_SIZE[0]) * (2*TRAY_HALF_SIZE[1]) * (2*TRAY_HALF_SIZE[2])
        self.tray = self.scene.add_entity(
            gs.morphs.Box(size=tuple(s * 2 for s in TRAY_HALF_SIZE)),
            material=gs.materials.Rigid(rho=TRAY_MASS / _tray_vol),
            surface=gs.surfaces.Default(color=(0.8, 0.6, 0.3, 1.0)),
        )

        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(0.0, 0.0, 1.5)),
            material=gs.materials.Rigid(
                rho=BALL_MASS / (4 / 3 * np.pi * BALL_RADIUS**3),
                friction=2.0,
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

        # Local link indices for right-hand joints (used for per-finger contact force queries).
        _hand_link_idxs_list = [
            _joint_map[n].link.idx_local for n in RIGHT_HAND_JOINTS if n in _joint_map
        ]
        self._hand_link_idxs_local = torch.tensor(
            _hand_link_idxs_list, dtype=torch.long, device=gs.device
        )
        # Global link indices = local + robot.link_start.
        # get_contacts() returns global link_a indices, so we need global for filtering.
        self._hand_link_idxs_global = self._hand_link_idxs_local + self.robot.link_start
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
        _frozen_pos_np = np.concatenate([
            STAND_LEG_POS, STAND_LEG_POS, STAND_WAIST_POS, STAND_LEFT_ARM_POS,
            LEFT_HAND_HOLD_POS[:len(self._left_hand_dofs)],
        ])
        # Pre-convert to tensor so _apply_action avoids numpy→tensor conversion every step.
        self._frozen_pos = torch.tensor(_frozen_pos_np, dtype=torch.float32, device=gs.device)

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

        # Action that corresponds to the hold pose — used to initialise prev_actions
        # at reset so the first obs is consistent rather than a fictitious zero action.
        _hold_pos_np = np.concatenate([RIGHT_ARM_HOLD_POS, RIGHT_HAND_HOLD_POS[:self.n_hand_dofs]])
        _hold_pos_t  = torch.tensor(_hold_pos_np, dtype=torch.float32, device=gs.device)
        self._hold_action = (
            2.0 * (_hold_pos_t - self._action_lower) / (self._action_upper - self._action_lower) - 1.0
        ).clamp(-1.0, 1.0)                                                # (n_action_dofs,)

        self.prev_actions = self._hold_action.unsqueeze(0).expand(self.n_envs, -1).clone()
        self._tray_init_pos = torch.zeros(self.n_envs, 3, device=gs.device)

        # Pre-allocated constants reused every step
        self._world_z = torch.zeros(self.n_envs, 3, device=gs.device)
        self._world_z[:, 2] = 1.0
        self._hand_hold = torch.tensor(
            RIGHT_HAND_HOLD_POS[:self.n_hand_dofs], device=gs.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.n_envs, -1).clone()

        # Joint limits expanded to (n_envs, n_action_dofs) — static, written once into obs buf.
        self._action_lower_obs = self._action_lower.unsqueeze(0).expand(self.n_envs, -1)
        self._action_upper_obs = self._action_upper.unsqueeze(0).expand(self.n_envs, -1)

        # Per-hand-link contact force vectors (n_envs, n_hand_dofs, 3), populated by _post_physics_step.
        self._hand_link_forces = torch.zeros(self.n_envs, self.n_hand_dofs, 3, device=gs.device)

        # Total robot→tray contact force per env, populated by _post_physics_step.
        self._tray_contact_force = torch.zeros(self.n_envs, device=gs.device)
        # Boolean: True if any robot↔tray contact pair is active this step.
        self.in_contact = torch.zeros(self.n_envs, dtype=torch.bool, device=gs.device)

        # Obs layout (147 total):
        #   arm_pos(7) + arm_vel(7) + hand_pos(12) + hand_vel(12)
        #   + tray_pos(3) + tray_z(3) + ball_pos(3) + ball_vel(3) + goal_pos(3)   [=53]
        #   + action_lower(19) + action_upper(19)                                  [=91]
        #   + hand_contact_forces(36) + total_contact_force(1)                     [=128]
        #   + prev_actions(19)                                                     [=147]
        self._obs_buf = torch.zeros(self.n_envs, 147, device=gs.device)
        # Write static joint-limit slices once — they never change.
        self._obs_buf[:, 53:72] = self._action_lower_obs
        self._obs_buf[:, 72:91] = self._action_upper_obs

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(147,), dtype=np.float32
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
        with torch.profiler.record_function("env/apply_action"):
            pos = self._action_lower + (action_tensor + 1.0) * 0.5 * (self._action_upper - self._action_lower)
            self.robot.control_dofs_position(self._frozen_pos, dofs_idx_local=self._frozen_dofs)
            self.robot.control_dofs_position(pos[:, :self.n_arm_dofs],  dofs_idx_local=self._right_arm_dofs)
            self.robot.control_dofs_position(pos[:, self.n_arm_dofs:],  dofs_idx_local=self._right_hand_dofs)

    def _post_physics_step(self):
        with torch.profiler.record_function("env/post_physics_step"):
            with torch.profiler.record_function("env/post_physics/dof_reads"):
                # 2 batched reads instead of 4 — halves the number of CPU←GPU sync points.
                _all_pos = self.robot.get_dofs_position(dofs_idx_local=self._action_dofs)
                _all_vel = self.robot.get_dofs_velocity(dofs_idx_local=self._action_dofs)
                self.arm_pos  = _all_pos[:, :self.n_arm_dofs]
                self.hand_pos = _all_pos[:, self.n_arm_dofs:]
                self.arm_vel  = _all_vel[:, :self.n_arm_dofs]
                self.hand_vel = _all_vel[:, self.n_arm_dofs:]

            with torch.profiler.record_function("env/post_physics/entity_reads"):
                self.tray_pos  = self.tray.get_pos()
                self.tray_quat = self.tray.get_quat()
                self.ball_pos  = self.ball.get_pos()
                self.ball_vel  = self.ball.get_vel()

            with torch.profiler.record_function("env/post_physics/contact_forces"):
                # Single call — tray-specific contacts only.
                contacts = self.robot.get_contacts(with_entity=self.tray)
                if "valid_mask" in contacts:
                    valid   = contacts["valid_mask"]    # (n_envs, max_contacts) bool
                    force_a = contacts["force_a"]       # (n_envs, max_contacts, 3) force on robot
                    link_a  = contacts["link_a"]        # (n_envs, max_contacts)   global link idx
                    link_b  = contacts["link_b"]        # (n_envs, max_contacts)

                    # Build hand-link match mask first — used for both reward and obs.
                    # A hand link can appear as link_a OR link_b depending on collision
                    # pair ordering; check both sides.
                    hand_idx  = self._hand_link_idxs_global.view(1, 1, -1)   # (1, 1, n_hand)
                    valid_exp = valid.unsqueeze(-1)                           # (n_envs, C, 1)
                    match = (
                        (link_a.unsqueeze(-1) == hand_idx) |
                        (link_b.unsqueeze(-1) == hand_idx)
                    ) & valid_exp                                             # (n_envs, C, n_hand)

                    # hand_valid: True only for contacts where a hand link is involved.
                    hand_valid   = match.any(dim=-1)                         # (n_envs, C)
                    force_a_mags = torch.norm(force_a, dim=-1)               # (n_envs, C)
                    self._tray_contact_force = (force_a_mags * hand_valid.float()).sum(dim=-1)
                    self.in_contact          = hand_valid.any(dim=-1)        # (n_envs,) bool

                    # Per-hand-link tray contact force vectors → (n_envs, n_hand, 3)
                    self._hand_link_forces = (
                        force_a.unsqueeze(2) * match.unsqueeze(-1).float()
                    ).sum(dim=1)
                else:
                    self._tray_contact_force.zero_()
                    self.in_contact = torch.zeros(self.n_envs, dtype=torch.bool, device=gs.device)
                    self._hand_link_forces.zero_()

            with torch.profiler.record_function("env/post_physics/tray_kinematics"):
                self.tray_z_world = _rotate_vec_by_quat(self._world_z, self.tray_quat)
                self.goal_pos = _rotate_vec_by_quat(self.goal_marker_offset, self.tray_quat) + self.tray_pos

            if self.show_viewer:
                self.goal_marker.set_pos(self.goal_pos)

            if self.debug_contacts:
                self._print_contact_forces()

    def _print_contact_forces(self):
        # ── per-finger net contact force (all contacts on each hand link) ──
        # Called only when debug_contacts=True; fetches on demand to keep the hot path clean.
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
        with torch.profiler.record_function("env/spawn_tray_at_wrist"):
            idx = (
                torch.arange(self.n_envs, device=gs.device)
                if envs_idx is None else envs_idx
            )
            palm_pos = self._middle_proximal_link.get_pos()[idx]  # (b, 3) — world frame

            # _tray_offset_local and _tray_offset_quat_local are already on device
            # (moved in _post_build_init); no .to() needed here.
            tray_pos  = palm_pos + self._tray_offset_local
            tray_quat = self._tray_offset_quat_local.unsqueeze(0).expand(idx.shape[0], -1)

            self.tray.set_pos(tray_pos, zero_velocity=True, envs_idx=idx)
            self.tray.set_quat(tray_quat, zero_velocity=True, envs_idx=idx)

    # ------------------------------------------------------------------ #
    # Gym interface                                                        #
    # ------------------------------------------------------------------ #

    def get_obs(self) -> torch.Tensor:
        with torch.profiler.record_function("env/get_obs"):
            self._obs_buf[:, 0:7]    = self.arm_pos
            self._obs_buf[:, 7:14]   = self.arm_vel
            self._obs_buf[:, 14:26]  = self.hand_pos
            self._obs_buf[:, 26:38]  = self.hand_vel
            self._obs_buf[:, 38:41]  = self.tray_pos
            self._obs_buf[:, 41:44]  = self.tray_z_world
            self._obs_buf[:, 44:47]  = self.ball_pos
            self._obs_buf[:, 47:50]  = self.ball_vel
            self._obs_buf[:, 50:53]  = self.goal_pos
            # [53:72] action_lower and [72:91] action_upper are written once at init — static.
            self._obs_buf[:, 91:127] = self._hand_link_forces.reshape(self.n_envs, -1)
            self._obs_buf[:, 127:128] = self._tray_contact_force.unsqueeze(-1)
            self._obs_buf[:, 128:147] = self.prev_actions
            return self._obs_buf

    def get_termination(self):
        with torch.profiler.record_function("env/get_termination"):
            tray_dist = torch.norm(self.tray_pos - self._tray_init_pos, dim=-1)
            ball_dist = torch.norm(self.ball_pos - self._tray_init_pos, dim=-1)
            tray_flipped = self.tray_z_world[:, 2] <= 0.0
            self.terminated = (tray_dist > 0.45) | (ball_dist > 0.5) | tray_flipped
            self.truncated  = self.episode_length_buf >= self.max_episode_steps
            return self.terminated, self.truncated

    def _compute_reward(self, action_tensor: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("env/compute_reward"):
            # ── Stage 1: grasp contact ─────────────────────────────────────────
            with torch.profiler.record_function("env/reward/contact"):
                r_contact = torch.clamp(self._tray_contact_force / 30.0, 0.0, 1.0)
                # Raw boolean gate: any collision geometry between robot and tray this step.
                # Ball rewards are fully on or fully off — no force threshold, no scaling.
                contact_gate = self.in_contact.float()

            with torch.profiler.record_function("env/reward/action_pen"):
                r_action_pen = torch.norm(action_tensor - self.prev_actions, dim=-1)

            r_fall_pen = self.terminated.float()

            with torch.profiler.record_function("env/reward/upright"):
                r_upright = self.tray_z_world[:, 2]
                if self.debug:
                    z = self.tray_z_world[0]
                    print(
                        f"  [upright] tray_z_world=[{z[0]:.4f}, {z[1]:.4f}, {z[2]:.4f}]"
                        f"  r_upright={r_upright[0]:.4f}"
                    )

            with torch.profiler.record_function("env/reward/ball"):
                ball_dist  = torch.norm(self.ball_pos - self.goal_pos, dim=-1)
                r_ball_pos = 1.0 / (1.0 + 5.0 * ball_dist)
                r_ball_vel = 1.0 / (1.0 + torch.norm(self.ball_vel, dim=-1))

            w_contact    = 3.0  if self.curriculum_stage >= 1 else 0.0
            w_in_contact = 1.0   # flat bonus every step any tray contact is active
            w_upright    = 0.5  if self.curriculum_stage >= 2 else 0.0
            w_ball_pos   = 1.5  if self.curriculum_stage >= 3 else 0.0
            w_ball_vel   = 0.3  if self.curriculum_stage >= 3 else 0.0
            w_action_pen = 0.01
            w_fall_pen   = 10.0

            # TODO maybe try tray ball contact reward 
            final_reward = (
                  w_contact    * r_contact
                + w_in_contact * contact_gate
                + w_upright    * r_upright  * contact_gate
                + w_ball_pos   * r_ball_pos * contact_gate * r_upright.clamp(min=0.0)
                + w_ball_vel   * r_ball_vel * contact_gate * r_upright.clamp(min=0.0)
                - w_action_pen * r_action_pen
                - w_fall_pen   * r_fall_pen
            )

            self.prev_actions.copy_(action_tensor.detach())

            log = self.extras["log"]
            log["r_contact"]       = r_contact.mean()
            log["r_upright"]       = r_upright.mean()
            log["r_ball_pos"]      = r_ball_pos.mean()
            log["r_ball_vel"]      = r_ball_vel.mean()
            log["r_action_pen"]    = r_action_pen.mean()
            log["r_fall_pen"]      = r_fall_pen.mean()
            log["tray_contact_N"]  = self._tray_contact_force.mean()
            log["in_contact_frac"] = contact_gate.mean()
            log["ball_dist"]       = ball_dist.mean()
            log["tray_upright_z"]  = r_upright.mean()

            return final_reward

    # ------------------------------------------------------------------ #
    # Reset helpers                                                        #
    # ------------------------------------------------------------------ #

    def _reset_env(self, envs_idx=None):
        with torch.profiler.record_function("env/reset_env"):
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
                self.prev_actions[:] = self._hold_action
            else:
                self.prev_actions[envs_idx] = self._hold_action

            # Ball will be placed after first scene.step() via reset() in base class

    def reset(self, envs_idx=None, seed=None, options=None):
        if envs_idx is None:
            # Full reset: super() runs _reset_env → scene.step() → _post_physics_step(),
            # which brings all caches up to date for the hold pose.
            super().reset(envs_idx=None, seed=seed, options=options)
        else:
            # Partial reset (done envs during training): no extra scene.step().
            self._reset_env(envs_idx)
            self.episode_length_buf[envs_idx] = 0
            # Directly write the known hold-pose into caches so get_obs() at the end
            # of this method sees correct arm/hand state, not stale last-episode values.
            self.arm_pos[envs_idx]  = self._right_arm_hold[envs_idx]
            self.arm_vel[envs_idx]  = 0.0
            self.hand_pos[envs_idx] = self._hand_hold[envs_idx]
            self.hand_vel[envs_idx] = 0.0

        idx = torch.arange(self.n_envs, device=gs.device) if envs_idx is None else envs_idx

        self._reset_goal_offset(envs_idx)
        self._spawn_tray_at_wrist(envs_idx)

        # Update tray caches after spawn.
        self.tray_pos[idx]      = self.tray.get_pos()[idx]
        self.tray_quat[idx]     = self.tray.get_quat()[idx]
        self.tray_z_world[idx]  = _rotate_vec_by_quat(self._world_z[idx], self.tray_quat[idx])
        self._tray_init_pos[idx] = self.tray_pos[idx]
        self.goal_pos[idx]      = (
            _rotate_vec_by_quat(self.goal_marker_offset[idx], self.tray_quat[idx])
            + self.tray_pos[idx]
        )
        if self.show_viewer:
            self.goal_marker.set_pos(self.goal_pos)

        self._reset_ball(envs_idx)

        # Update ball caches after placement so get_obs() sees the new position.
        self.ball_pos[idx] = self.ball.get_pos()[idx]
        self.ball_vel[idx] = 0.0

        # Build obs from fully-refreshed caches — no stale state from the previous episode.
        return self.get_obs(), {}

    def _reset_ball(self, envs_idx=None):
        with torch.profiler.record_function("env/reset_ball"):
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
        with torch.profiler.record_function("env/reset_goal_offset"):
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
            z = torch.full((b, 1), TRAY_HALF_SIZE[2] + BALL_RADIUS, device=gs.device)
            self.goal_marker_offset[idx] = torch.cat([xy, z], dim=-1)
