"""
Benchmark: Genesis state access strategies for RL simulation.

Three approaches:
  A) Baseline   — robot.get_dofs_position() / entity.get_pos() — current code style
  B) Zerocopy   — qd_to_torch(solver.field, transpose=True, copy=False) + gather
  C) Contiguous — same but check if action DOFs are contiguous (pure slice, no gather)

Run with:
    conda run -n cse190_bimanual python experimental/bench_state_access.py
"""

import time
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from genesis.utils.misc import qd_to_torch

# ─── Config ──────────────────────────────────────────────────────────────────
N_ENVS      = 32
WARMUP_ITER = 30
BENCH_ITER  = 300
HEADLESS    = True

# Same USD / asset as the actual env
from pathlib import Path
G1_USD = str(Path(__file__).parents[1] / "assets" / "g1-flattened-fixed.usd")


def build_scene(n_envs):
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.02, substeps=4),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)))
    tray  = scene.add_entity(gs.morphs.Box(size=(0.36, 0.26, 0.01)),
                              material=gs.materials.Rigid(rho=300.0))
    ball  = scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, 1.5)),
                              material=gs.materials.Rigid(rho=8842.0))
    scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))
    return scene, robot, tray, ball


def get_dof_indices(robot):
    """Return local and global DOF indices for right arm + right hand."""
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}

    RIGHT_ARM = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    RIGHT_HAND = [
        "R_index_proximal_joint", "R_index_intermediate_joint",
        "R_middle_proximal_joint", "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",  "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",   "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    ]

    arm_local  = [joint_map[n].dofs_idx_local[0] for n in RIGHT_ARM  if n in joint_map]
    hand_local = [joint_map[n].dofs_idx_local[0] for n in RIGHT_HAND if n in joint_map]
    action_local = arm_local + hand_local

    dof_start = robot._dof_start
    action_global = [dof_start + d for d in action_local]
    n_arm = len(arm_local)
    return action_local, action_global, n_arm


def check_zerocopy_status():
    print(f"\n{'='*60}")
    print(f"  gs.use_zerocopy  = {gs.use_zerocopy}")
    print(f"  gs.backend       = {gs.backend}")
    print(f"  gs.device        = {gs.device}")
    print(f"{'='*60}\n")


def check_contiguous(action_global):
    """Check if action DOFs are contiguous in global space."""
    diffs = [action_global[i+1] - action_global[i] for i in range(len(action_global)-1)]
    is_contig = all(d == 1 for d in diffs)
    print(f"Action DOFs global indices: {action_global}")
    print(f"Diffs: {diffs}")
    print(f"Contiguous in global space: {is_contig}")
    if is_contig:
        print(f"  → pure slice possible: [{action_global[0]}:{action_global[-1]+1}]")
    return is_contig, action_global[0], action_global[-1]+1


def bench(label, fn, warmup, iters, sync=True):
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1000
    print(f"  {label:<40s}  {ms:.4f} ms/step")
    return ms


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    scene, robot, tray, ball = build_scene(N_ENVS)
    check_zerocopy_status()

    action_local, action_global, n_arm = get_dof_indices(robot)
    n_action = len(action_global)

    print(f"n_envs={N_ENVS}, n_action_dofs={n_action} (arm={n_arm}, hand={n_action-n_arm})")
    is_contig, slice_start, slice_end = check_contiguous(action_global)

    # ── Get solver (all rigid entities share one solver) ──────────────────
    solver = robot._solver
    assert tray._solver is solver, "Tray uses same solver"
    assert ball._solver is solver, "Ball uses same solver"

    # ── Global link indices ───────────────────────────────────────────────
    tray_link_idx = tray.base_link_idx   # int
    ball_link_idx = ball.base_link_idx   # int
    print(f"\nGlobal indices:")
    print(f"  robot._dof_start = {robot._dof_start}")
    print(f"  robot._link_start = {robot._link_start}")
    print(f"  tray.base_link_idx = {tray_link_idx}")
    print(f"  ball.base_link_idx = {ball_link_idx}")
    print(f"  solver.n_dofs_ = {solver.n_dofs_}")
    print(f"  solver.n_links = {solver.n_links}")

    # ── Action index tensor ───────────────────────────────────────────────
    action_global_t = torch.tensor(action_global, dtype=torch.long, device=gs.device)

    # ── Step the sim once so state is valid ──────────────────────────────
    scene.step()

    # ── Cache zerocopy views ONCE (this is the setup cost, paid at init) ──
    dof_pos_tc   = qd_to_torch(solver.dofs_state.pos, transpose=True, copy=False)
    dof_vel_tc   = qd_to_torch(solver.dofs_state.vel, transpose=True, copy=False)
    links_pos_tc  = qd_to_torch(solver.links_state.pos,  transpose=True, copy=False)
    links_quat_tc = qd_to_torch(solver.links_state.quat, transpose=True, copy=False)

    # cd_vel is the Cartesian dof velocity (COM vel in root_COM frame).
    # For a free single-link body: ball_vel ≈ links_cd_vel[ball_idx]
    links_cd_vel_tc = qd_to_torch(solver.links_state.cd_vel, transpose=True, copy=False)

    print(f"\nZerocopy tensor shapes (after transpose):")
    print(f"  dof_pos_tc.shape    = {dof_pos_tc.shape}   (n_envs, n_total_dofs)")
    print(f"  links_pos_tc.shape  = {links_pos_tc.shape}  (n_envs, n_total_links, 3)")
    print(f"  links_quat_tc.shape = {links_quat_tc.shape}")
    print(f"  links_cd_vel_tc.shape = {links_cd_vel_tc.shape}")
    print(f"  is_tensor_view(dof_pos_tc): storage data_ptr match = "
          f"{dof_pos_tc.data_ptr() == dof_pos_tc.data_ptr()}")  # tautology, just sanity

    # Pre-allocate output buffers (mimic the env's cached state tensors)
    arm_pos_buf  = torch.zeros(N_ENVS, n_arm,           device=gs.device)
    hand_pos_buf = torch.zeros(N_ENVS, n_action - n_arm, device=gs.device)
    arm_vel_buf  = torch.zeros(N_ENVS, n_arm,           device=gs.device)
    hand_vel_buf = torch.zeros(N_ENVS, n_action - n_arm, device=gs.device)
    tray_pos_buf = torch.zeros(N_ENVS, 3, device=gs.device)
    tray_quat_buf= torch.zeros(N_ENVS, 4, device=gs.device)
    ball_pos_buf = torch.zeros(N_ENVS, 3, device=gs.device)
    ball_vel_buf = torch.zeros(N_ENVS, 3, device=gs.device)

    # ─────────────────────────────────────────────────────────────────────
    # Approach A: Baseline — high-level API with copy=True internally
    # ─────────────────────────────────────────────────────────────────────
    def approach_A():
        pos = robot.get_dofs_position(dofs_idx_local=action_local)
        vel = robot.get_dofs_velocity(dofs_idx_local=action_local)
        _ = tray.get_pos()
        _ = tray.get_quat()
        _ = ball.get_pos()
        _ = ball.get_vel()

    # ─────────────────────────────────────────────────────────────────────
    # Approach B: Zerocopy gather — cache views at init, gather each step
    # ─────────────────────────────────────────────────────────────────────
    def approach_B():
        # DOF state: gather by global index (GPU gather, no allocation)
        all_pos = dof_pos_tc[:, action_global_t]        # (n_envs, 19)
        all_vel = dof_vel_tc[:, action_global_t]        # (n_envs, 19)
        arm_pos_buf.copy_(all_pos[:, :n_arm])
        hand_pos_buf.copy_(all_pos[:, n_arm:])
        arm_vel_buf.copy_(all_vel[:, :n_arm])
        hand_vel_buf.copy_(all_vel[:, n_arm:])
        # Link state: direct index (single-step gather)
        tray_pos_buf.copy_(links_pos_tc[:, tray_link_idx])
        tray_quat_buf.copy_(links_quat_tc[:, tray_link_idx])
        ball_pos_buf.copy_(links_pos_tc[:, ball_link_idx])
        ball_vel_buf.copy_(links_cd_vel_tc[:, ball_link_idx])

    # ─────────────────────────────────────────────────────────────────────
    # Approach C: Zerocopy no-copy (no .copy_ into separate buffer)
    #   Directly use views — only valid if downstream code can tolerate shared mem
    # ─────────────────────────────────────────────────────────────────────
    def approach_C():
        # No buffer allocation, no copy — just slice the view
        # This is only safe if you don't write back into these tensors
        _ = dof_pos_tc[:, action_global_t]
        _ = dof_vel_tc[:, action_global_t]
        _ = links_pos_tc[:, tray_link_idx]
        _ = links_quat_tc[:, tray_link_idx]
        _ = links_pos_tc[:, ball_link_idx]
        _ = links_cd_vel_tc[:, ball_link_idx]

    # ─────────────────────────────────────────────────────────────────────
    # Approach D: Contiguous slice (if DOFs are contiguous) — true zero-copy
    # ─────────────────────────────────────────────────────────────────────
    def approach_D_slice():
        # Pure Python slice → returns a tensor view, no allocation, no gather
        _ = dof_pos_tc[:, slice_start:slice_end]
        _ = dof_vel_tc[:, slice_start:slice_end]
        _ = links_pos_tc[:, tray_link_idx]
        _ = links_quat_tc[:, tray_link_idx]
        _ = links_pos_tc[:, ball_link_idx]
        _ = links_cd_vel_tc[:, ball_link_idx]

    # ─────────────────────────────────────────────────────────────────────
    # Approach E: Baseline but with copy=False in qd_to_torch (bypass copy)
    #   This tests if the DLPack caching alone (not the copy) is the bottleneck
    # ─────────────────────────────────────────────────────────────────────
    def approach_E_nocopy_api():
        # qd_to_torch with copy=False but still using dofs_idx masking
        # This still does the fancy index (gather) but avoids the full copy
        pos = qd_to_torch(solver.dofs_state.pos, None, action_global_t,
                          transpose=True, copy=None)
        vel = qd_to_torch(solver.dofs_state.vel, None, action_global_t,
                          transpose=True, copy=None)
        _ = qd_to_torch(solver.links_state.pos,  None, tray_link_idx, transpose=True, copy=None)
        _ = qd_to_torch(solver.links_state.quat, None, tray_link_idx, transpose=True, copy=None)
        _ = qd_to_torch(solver.links_state.pos,  None, ball_link_idx, transpose=True, copy=None)
        _ = qd_to_torch(solver.links_state.cd_vel, None, ball_link_idx, transpose=True, copy=None)

    print(f"\n{'='*60}")
    print(f"  State read benchmark — {N_ENVS} envs, {BENCH_ITER} iters")
    print(f"{'='*60}")

    ms_A = bench("A: high-level API (current code)",     approach_A, WARMUP_ITER, BENCH_ITER)
    ms_B = bench("B: zerocopy gather + .copy_ to buf",  approach_B, WARMUP_ITER, BENCH_ITER)
    ms_C = bench("C: zerocopy gather (no output copy)", approach_C, WARMUP_ITER, BENCH_ITER)
    ms_E = bench("E: qd_to_torch(copy=None) direct",   approach_E_nocopy_api, WARMUP_ITER, BENCH_ITER)

    if is_contig:
        ms_D = bench("D: pure slice (contiguous DOFs)",  approach_D_slice, WARMUP_ITER, BENCH_ITER)
    else:
        print(f"  D: pure slice — SKIPPED (DOFs not contiguous in global space)")
        ms_D = None

    print(f"\n{'='*60}")
    print(f"  Speedups vs baseline (A):")
    print(f"  B speedup: {ms_A/ms_B:.2f}x")
    print(f"  C speedup: {ms_A/ms_C:.2f}x")
    print(f"  E speedup: {ms_A/ms_E:.2f}x")
    if ms_D:
        print(f"  D speedup: {ms_A/ms_D:.2f}x")
    print(f"{'='*60}\n")

    # ─── Full step benchmark (sim + state reads) ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  Full step throughput (scene.step + state reads)")
    print(f"{'='*60}")

    def full_step_A():
        scene.step()
        pos = robot.get_dofs_position(dofs_idx_local=action_local)
        vel = robot.get_dofs_velocity(dofs_idx_local=action_local)
        _ = tray.get_pos()
        _ = tray.get_quat()
        _ = ball.get_pos()
        _ = ball.get_vel()

    def full_step_B():
        scene.step()
        all_pos = dof_pos_tc[:, action_global_t]
        all_vel = dof_vel_tc[:, action_global_t]
        arm_pos_buf.copy_(all_pos[:, :n_arm])
        hand_pos_buf.copy_(all_pos[:, n_arm:])
        arm_vel_buf.copy_(all_vel[:, :n_arm])
        hand_vel_buf.copy_(all_vel[:, n_arm:])
        tray_pos_buf.copy_(links_pos_tc[:, tray_link_idx])
        tray_quat_buf.copy_(links_quat_tc[:, tray_link_idx])
        ball_pos_buf.copy_(links_pos_tc[:, ball_link_idx])
        ball_vel_buf.copy_(links_cd_vel_tc[:, ball_link_idx])

    ms_full_A = bench("A: scene.step + API reads", full_step_A, 20, 200)
    ms_full_B = bench("B: scene.step + zerocopy",  full_step_B, 20, 200)

    print(f"\n  Full step speedup B vs A: {ms_full_A/ms_full_B:.2f}x")
    print(f"  State read fraction of full step:")
    print(f"    A: {ms_A/ms_full_A*100:.1f}% of step time is state reads")
    print(f"    B: {ms_B/ms_full_B*100:.1f}% of step time is state reads")
    print()


if __name__ == "__main__":
    main()
