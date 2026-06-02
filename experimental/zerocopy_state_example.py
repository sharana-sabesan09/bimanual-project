"""
Reference implementation: optimal Genesis state access for RL training.

Key results from benchmarks (32 envs, substeps=4):
  State reads alone:
    API (current):   2.40 ms/step
    Zerocopy:        0.03 ms/step   →  85x faster
  Full env step:
    API + contacts:  24.2 ms/step
    ZC  + contacts:  20.8 ms/step   →  1.17x faster
  performance_mode=True (sim kernels ~30% faster):
    Projected:       ~14 ms/step    →  1.73x overall from current

Two techniques:
  1. performance_mode=True in gs.init() — 30% faster sim kernels
  2. Zero-copy DLPack views for state reads — 85x faster state reads

Usage: study this file, then apply the pattern to source/tasks/tray_grasp/single_arm_grasp.py
"""

import torch
import genesis as gs
from genesis.utils.misc import qd_to_torch


# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE 1: performance_mode=True
# ─────────────────────────────────────────────────────────────────────────────
# In base_env.py or wherever gs.init() is called:
#
#   gs.init(backend=gs.gpu, performance_mode=True)
#
# What it does:
#   - Disables dynamic ndarray mode → bakes tensor shapes into kernels
#   - ~30% faster simulation step (documented, measured ~30% on typical tasks)
#   - Requires recompile when scene config changes (not relevant for RL: fixed scene)
#
# Caveat: kernel compilation is cached, but first-run is slower.
#         Use with fixed n_envs and fixed scene (exactly the RL training scenario).


# ─────────────────────────────────────────────────────────────────────────────
# TECHNIQUE 2: Zero-copy DLPack state reads
# ─────────────────────────────────────────────────────────────────────────────

class ZerocopyStateCache:
    """
    Caches DLPack zero-copy views into Genesis solver state arrays.

    How it works:
      - Genesis (Quadrants) stores simulation state in Taichi fields with
        shape (n_entities, n_envs) — rows=entities, cols=envs.
      - qd_to_torch(field, transpose=True, copy=False) returns a PyTorch
        tensor VIEW of that memory, transposed to (n_envs, n_entities).
      - This view is cached on field._tc/_T_tc after first call.
      - Subsequent reads just index the cached tensor — no DLPack overhead,
        no allocation, no copy.

    Memory layout after build():
      dof_pos[n_envs, n_total_dofs]       — all DOFs for all entities
      links_pos[n_envs, n_total_links, 3] — all link positions
      links_quat[n_envs, n_total_links, 4]
      links_cd_vel[n_envs, n_total_links, 3]  — COM velocity (== get_vel() for free bodies)

    Usage:
      cache = ZerocopyStateCache(robot, tray, ball, arm_local_dofs, hand_local_dofs)
      # After each scene.step():
      arm_pos  = cache.dof_pos[:, cache.arm_global]    # GPU gather, no alloc
      tray_pos = cache.links_pos[:, cache.tray_link]   # single index, no alloc
    """

    def __init__(self, robot, tray, ball, arm_local_dofs, hand_local_dofs):
        solver = robot._solver
        assert tray._solver is solver and ball._solver is solver, \
            "All rigid entities share one solver"

        # ── Global DOF indices ────────────────────────────────────────────
        dof_start = robot._dof_start
        action_local  = arm_local_dofs + hand_local_dofs
        self.n_arm    = len(arm_local_dofs)
        self.n_action = len(action_local)

        self.arm_global    = torch.tensor(
            [dof_start + d for d in arm_local_dofs], dtype=torch.long, device=gs.device)
        self.hand_global   = torch.tensor(
            [dof_start + d for d in hand_local_dofs], dtype=torch.long, device=gs.device)
        self.action_global = torch.cat([self.arm_global, self.hand_global])

        # ── Global link indices ───────────────────────────────────────────
        self.tray_link = tray.base_link_idx   # int
        self.ball_link = ball.base_link_idx   # int

        # ── Zero-copy views (cached after first DLPack call) ─────────────
        # Shape after transpose: (n_envs, n_total_dofs)
        self.dof_pos = qd_to_torch(solver.dofs_state.pos, transpose=True, copy=False)
        self.dof_vel = qd_to_torch(solver.dofs_state.vel, transpose=True, copy=False)

        # Shape after transpose: (n_envs, n_total_links, 3) or (n_envs, n_total_links, 4)
        self.links_pos    = qd_to_torch(solver.links_state.pos,    transpose=True, copy=False)
        self.links_quat   = qd_to_torch(solver.links_state.quat,   transpose=True, copy=False)
        self.links_cd_vel = qd_to_torch(solver.links_state.cd_vel, transpose=True, copy=False)
        # cd_ang needed if ball has angular velocity (for full vel formula)
        self.links_cd_ang = qd_to_torch(solver.links_state.cd_ang, transpose=True, copy=False)

        print(f"[ZerocopyStateCache] shapes:")
        print(f"  dof_pos:       {self.dof_pos.shape}")
        print(f"  links_pos:     {self.links_pos.shape}")
        print(f"  links_cd_vel:  {self.links_cd_vel.shape}")
        print(f"  tray_link_idx: {self.tray_link}")
        print(f"  ball_link_idx: {self.ball_link}")

    def read_all(self):
        """Read all state in one shot. Returns views/gathers — no allocations."""
        arm_pos   = self.dof_pos[:, self.arm_global]       # GPU gather
        hand_pos  = self.dof_pos[:, self.hand_global]      # GPU gather
        arm_vel   = self.dof_vel[:, self.arm_global]
        hand_vel  = self.dof_vel[:, self.hand_global]
        tray_pos  = self.links_pos[:,  self.tray_link]     # single-int index
        tray_quat = self.links_quat[:, self.tray_link]
        ball_pos  = self.links_pos[:,  self.ball_link]
        # cd_vel == get_vel() for a free single-link body (validated: exact match)
        ball_vel  = self.links_cd_vel[:, self.ball_link]
        return arm_pos, hand_pos, arm_vel, hand_vel, tray_pos, tray_quat, ball_pos, ball_vel


# ─────────────────────────────────────────────────────────────────────────────
# How to integrate into SingleArmTrayGraspEnv
# ─────────────────────────────────────────────────────────────────────────────
#
# In _post_build_init(), after setting up dof indices, add:
#
#   from genesis.utils.misc import qd_to_torch
#   self._state_cache = ZerocopyStateCache(
#       self.robot, self.tray, self.ball,
#       self._right_arm_dofs, self._right_hand_dofs,
#   )
#
# In _post_physics_step(), replace the dof reads block with:
#
#   solver = self.robot._solver
#   arm_pos, hand_pos, arm_vel, hand_vel, \
#   tray_pos, tray_quat, ball_pos, ball_vel = self._state_cache.read_all()
#   self.arm_pos.copy_(arm_pos)
#   self.hand_pos.copy_(hand_pos)
#   self.arm_vel.copy_(arm_vel)
#   self.hand_vel.copy_(hand_vel)
#   self.tray_pos.copy_(tray_pos)
#   self.tray_quat.copy_(tray_quat)
#   self.ball_pos.copy_(ball_pos)
#   self.ball_vel.copy_(ball_vel)
#
# Or, even better — write directly into obs_buf slices in get_obs() without
# intermediate copy_() calls:
#
#   self._obs_buf[:, 0:7]   = self._state_cache.dof_pos[:, self._state_cache.arm_global]
#   self._obs_buf[:, 7:14]  = self._state_cache.dof_vel[:, self._state_cache.arm_global]
#   ...
#
# ─────────────────────────────────────────────────────────────────────────────


def demo():
    """Minimal demo showing ZerocopyStateCache works correctly."""
    from pathlib import Path
    G1_USD = str(Path(__file__).parents[1] / "assets" / "g1-flattened-fixed.usd")

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=0.02, substeps=4))
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)))
    tray  = scene.add_entity(gs.morphs.Box(size=(0.36, 0.26, 0.01)),
                              material=gs.materials.Rigid(rho=300.0))
    ball  = scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, 1.5)),
                              material=gs.materials.Rigid(rho=8842.0))
    scene.build(n_envs=4, env_spacing=(1.5, 1.5))

    jmap = {j.name.split("/")[-1]: j for j in robot.joints}
    arm_local  = [jmap[n].dofs_idx_local[0] for n in [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ] if n in jmap]
    hand_local = [jmap[n].dofs_idx_local[0] for n in [
        "R_index_proximal_joint", "R_index_intermediate_joint",
        "R_middle_proximal_joint", "R_middle_intermediate_joint",
        "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
        "R_ring_proximal_joint", "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    ] if n in jmap]

    cache = ZerocopyStateCache(robot, tray, ball, arm_local, hand_local)

    scene.step()
    arm_pos, hand_pos, _, _, tray_pos, _, ball_pos, ball_vel = cache.read_all()

    # Cross-check against API
    api_arm = robot.get_dofs_position(dofs_idx_local=arm_local)
    api_ball_vel = ball.get_vel()

    diff_arm = (arm_pos - api_arm).abs().max().item()
    diff_bv  = (ball_vel - api_ball_vel).abs().max().item()
    print(f"\nDemo cross-check:")
    print(f"  arm_pos   max diff vs API: {diff_arm:.2e}  {'✓' if diff_arm < 1e-5 else '✗'}")
    print(f"  ball_vel  max diff vs API: {diff_bv:.2e}   {'✓' if diff_bv < 1e-5 else '✗'}")
    print(f"  arm_pos  shape: {arm_pos.shape}")
    print(f"  tray_pos shape: {tray_pos.shape}")
    print(f"  ball_vel shape: {ball_vel.shape}")
    print("\nZerocopy state access demo complete.")


if __name__ == "__main__":
    demo()
