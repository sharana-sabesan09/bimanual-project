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


class BallBalanceEnv:
    """
    Batched environment wrapper. Supports n_envs parallel simulations.
    Legs/waist/left-arm are PD-frozen at the standing pose each step.
    Only the 7 right-arm joints are exposed as the RL action.
    """

    def __init__(self, show_viewer: bool = True, n_envs: int = 1):
        # TODO select cuda is available otherwise fallback to cpu
        gs.init(backend=gs.cpu)

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
        self.n_arm_dofs = len(self._right_arm_dofs)
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

        # All frozen joints together (legs + waist + left arm)
        self._frozen_dofs = (
            self._left_leg_dofs + self._right_leg_dofs +
            self._waist_dofs + self._left_arm_dofs
        )
        self._frozen_pos = np.concatenate([
            STAND_LEG_POS, STAND_LEG_POS,
            STAND_WAIST_POS, STAND_LEFT_ARM_POS,
        ])

    # ── reset / step ──────────────────────────────────────────────────────────

    def reset(self):
        # Snap frozen joints to standing pose (zero velocity)
        self.robot.set_dofs_position(
            self._frozen_pos,
            dofs_idx_local=self._frozen_dofs,
            zero_velocity=True,
        )
        # Snap right arm to hold pose (zero velocity)
        self.robot.set_dofs_position(
            RIGHT_ARM_HOLD_POS,
            dofs_idx_local=self._right_arm_dofs,
            zero_velocity=True,
        )
        # Place ball just above the tray centre
        self._reset_ball()
        self.scene.step()
        return self.get_obs()

    def _reset_ball(self):
        tray_pos = self.robot.get_link("tray").get_pos()          # (b, 3)
        ball_z   = tray_pos[:, 2:3] + TRAY_SIZE[2] / 2 + BALL_RADIUS + 0.005  # (b, 1)
        spawn    = torch.cat([tray_pos[:, :2], ball_z], dim=-1)          # (b, 3)
        self.ball.set_pos(spawn, zero_velocity=True)

    def step(self, action: np.ndarray):
        """
        action : (b, 7) position targets (rad) for the right arm.
        Returns (obs, reward, done, info).
        """
        # Hold legs / waist / left arm at standing pose every step
        self.robot.control_dofs_position(
            self._frozen_pos,
            dofs_idx_local=self._frozen_dofs,
        )
        # Apply RL action to right arm
        self.robot.control_dofs_position(
            action.astype(np.float32),
            dofs_idx_local=self._right_arm_dofs,
        )
        self.scene.step()

        obs              = self.get_obs()
        ball_pos, ball_vel, goal_pos = self._unpack_obs(obs)
        reward, done     = self._compute_reward(ball_pos, goal_pos, ball_vel)
        return obs, reward, done, {}

    # ── observations ──────────────────────────────────────────────────────────
    
    def get_obs(self):
        """
        Flat vector:
          right_arm_pos  (b, 7)   — joint positions
          right_arm_vel  (b, 7)   — joint velocities
          ball_pos       (b, 3)
          ball_vel       (b, 3)
          goal_pos       (b, 3)   — tray centre in world frame
        Returns (b, 23).
        """
        arm_pos  = self.robot.get_dofs_position(dofs_idx_local=self._right_arm_dofs)
        arm_vel  = self.robot.get_dofs_velocity(dofs_idx_local=self._right_arm_dofs)
        ball_pos = self.ball.get_pos()
        ball_vel = self.ball.get_vel()
        goal_pos = self.robot.get_link("tray").get_pos()
        return torch.cat([arm_pos, arm_vel, ball_pos, ball_vel, goal_pos], dim=-1)

    def _unpack_obs(self, obs):
        ball_pos = obs[:, 14:17]
        ball_vel = obs[:, 17:20]
        goal_pos = obs[:, 20:23]
        return ball_pos, ball_vel, goal_pos

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, ball_pos, goal_pos, ball_vel):
        """
        Shaped reward:
          + exp(-3 * XY_distance)     proximity to tray centre
          - 0.1 * ball_speed          discourage erratic motion
          - 10  if ball falls off     large terminal penalty
        """
        xy_dist   = torch.norm(ball_pos[:, :2] - goal_pos[:, :2], dim=-1)  # (b,)
        proximity = torch.exp(-3.0 * xy_dist)                              # (b,)
        vel_pen   = -0.1 * torch.norm(ball_vel, dim=-1)                    # (b,)
        fallen    = ball_pos[:, 2] < (goal_pos[:, 2] - 0.15)              # (b,)
        fall_pen  = torch.where(fallen, torch.full_like(proximity, -10.0), torch.zeros_like(proximity))
        return proximity + vel_pen + fall_pen, fallen


# ─── quick visualisation / sanity check ──────────────────────────────────────
if __name__ == "__main__":
    env = BallBalanceEnv(show_viewer=True)

    # Freeze every joint — legs, waist, left arm, and right arm all held static
    all_frozen_dofs = env._frozen_dofs + env._right_arm_dofs
    all_frozen_pos  = np.concatenate([env._frozen_pos, RIGHT_ARM_HOLD_POS])

    while True:
        env.robot.control_dofs_position(all_frozen_pos, dofs_idx_local=all_frozen_dofs)
        env._reset_ball()   # keep ball on tray while viewing
        
        env.scene.step()


# TODO tasks
# change the MJCF to have the tray directly attached as the hand, removing the hand visual entirely
# put this env as its own file and have extra targets 
# 1. zero agent, spawns the env and gives all 0's as actions 
# 2. random agent, spawns the env and gives random actions at each step 
# 3. teleop agent, spawns the env and with keyboard input to control hand behavior wasd for xyz and ijkl for pitch,yaw,roll
# 4. train agent, connects the env with RSL rl for training