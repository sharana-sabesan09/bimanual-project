"""
Benchmark 2: Contact costs, performance_mode insights, ball vel validation.
All in a single genesis init to avoid double-init error.

Run with:
    conda run -n cse190_bimanual python experimental/bench_perf_mode.py
"""

import time
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs
from genesis.utils.misc import qd_to_torch
from pathlib import Path

G1_USD = str(Path(__file__).parents[1] / "assets" / "g1-flattened-fixed.usd")
N_ENVS = 32

gs.init(backend=gs.gpu, logging_level="warning")


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
    print(f"  {label:<50s}  {ms:.4f} ms/step")
    return ms


def main():
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=0.02, substeps=4))
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)))
    tray  = scene.add_entity(gs.morphs.Box(size=(0.36, 0.26, 0.01)),
                              material=gs.materials.Rigid(rho=300.0))
    ball  = scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, 1.5)),
                              material=gs.materials.Rigid(rho=8842.0))
    scene.build(n_envs=N_ENVS, env_spacing=(1.5, 1.5))

    solver = robot._solver
    ball_link = ball.base_link_idx
    tray_link = tray.base_link_idx

    for _ in range(20):
        scene.step()

    # ── 1. Validate ball velocity shortcut ───────────────────────────────
    vel_api   = ball.get_vel()
    cd_vel_tc = qd_to_torch(solver.links_state.cd_vel, transpose=True, copy=False)
    vel_short = cd_vel_tc[:, ball_link]
    diff = (vel_api - vel_short).abs().max().item()

    print(f"\n── Ball vel shortcut: max error = {diff:.2e}  "
          f"{'✓ VALID' if diff < 1e-4 else '✗ INVALID'}")

    # ── 2. Contact query benchmarks ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Contact and state read costs ({N_ENVS} envs)")
    print(f"{'='*60}")

    # Get the joint map once for action dof setup
    joint_map  = {j.name.split("/")[-1]: j for j in robot.joints}
    RIGHT_ARM  = [
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
    action_global = [robot._dof_start + d for d in action_local]
    action_global_t = torch.tensor(action_global, dtype=torch.long, device=gs.device)
    n_arm = len(arm_local)

    # Zerocopy views
    dof_pos_tc    = qd_to_torch(solver.dofs_state.pos,   transpose=True, copy=False)
    dof_vel_tc    = qd_to_torch(solver.dofs_state.vel,   transpose=True, copy=False)
    links_pos_tc  = qd_to_torch(solver.links_state.pos,  transpose=True, copy=False)
    links_quat_tc = qd_to_torch(solver.links_state.quat, transpose=True, copy=False)
    links_cd_vel_tc = qd_to_torch(solver.links_state.cd_vel, transpose=True, copy=False)

    # ── Individual operation costs ────────────────────────────────────────
    ms_get_contacts = bench(
        "get_contacts(with_entity=tray)",
        lambda: robot.get_contacts(with_entity=tray), 20, 200)

    ms_get_net = bench(
        "get_links_net_contact_force()",
        lambda: robot.get_links_net_contact_force(), 20, 200)

    ms_dof_pos_api = bench(
        "get_dofs_position() [API, copy=True]",
        lambda: robot.get_dofs_position(dofs_idx_local=action_local), 20, 200)

    ms_dof_pos_zc = bench(
        "dof_pos_tc[:, idx] [zerocopy gather]",
        lambda: dof_pos_tc[:, action_global_t], 20, 200)

    ms_link_pos_api = bench(
        "tray.get_pos() + ball.get_pos() [API]",
        lambda: (tray.get_pos(), ball.get_pos()), 20, 200)

    ms_link_pos_zc = bench(
        "links_pos_tc[:, idx] x2 [zerocopy]",
        lambda: (links_pos_tc[:, tray_link], links_pos_tc[:, ball_link]), 20, 200)

    ms_all_state_api = bench(
        "All state reads [API baseline]",
        lambda: (
            robot.get_dofs_position(dofs_idx_local=action_local),
            robot.get_dofs_velocity(dofs_idx_local=action_local),
            tray.get_pos(), tray.get_quat(),
            ball.get_pos(), ball.get_vel(),
        ), 20, 200)

    ms_all_state_zc = bench(
        "All state reads [zerocopy]",
        lambda: (
            dof_pos_tc[:, action_global_t],
            dof_vel_tc[:, action_global_t],
            links_pos_tc[:, tray_link], links_quat_tc[:, tray_link],
            links_pos_tc[:, ball_link], links_cd_vel_tc[:, ball_link],
        ), 20, 200)

    # ── Full step benchmarks ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Full env step ({N_ENVS} envs)")
    print(f"{'='*60}")

    def full_A():
        scene.step()
        robot.get_dofs_position(dofs_idx_local=action_local)
        robot.get_dofs_velocity(dofs_idx_local=action_local)
        tray.get_pos(); tray.get_quat()
        ball.get_pos(); ball.get_vel()
        robot.get_contacts(with_entity=tray)

    def full_B():
        scene.step()
        dof_pos_tc[:, action_global_t]
        dof_vel_tc[:, action_global_t]
        links_pos_tc[:, tray_link]; links_quat_tc[:, tray_link]
        links_pos_tc[:, ball_link]; links_cd_vel_tc[:, ball_link]
        robot.get_contacts(with_entity=tray)  # contacts still needed

    def full_C():
        scene.step()
        dof_pos_tc[:, action_global_t]
        dof_vel_tc[:, action_global_t]
        links_pos_tc[:, tray_link]; links_quat_tc[:, tray_link]
        links_pos_tc[:, ball_link]; links_cd_vel_tc[:, ball_link]
        # No contacts (if we find another way)

    ms_full_A = bench("scene.step + API reads + contacts",  full_A, 15, 150)
    ms_full_B = bench("scene.step + zerocopy  + contacts",  full_B, 15, 150)
    ms_full_C = bench("scene.step + zerocopy  (no contacts)", full_C, 15, 150)
    ms_sim_only = bench("scene.step only",
                        lambda: scene.step(), 15, 150)

    print(f"\n{'='*60}")
    print(f"  Cost breakdown ({N_ENVS} envs)")
    print(f"{'='*60}")
    print(f"  sim only:                  {ms_sim_only:.3f} ms")
    print(f"  contacts only:             {ms_get_contacts:.4f} ms")
    print(f"  all state API reads:       {ms_all_state_api:.4f} ms")
    print(f"  all state zerocopy reads:  {ms_all_state_zc:.4f} ms")
    print(f"  full step A (API+contacts): {ms_full_A:.3f} ms")
    print(f"  full step B (ZC+contacts):  {ms_full_B:.3f} ms")
    print(f"  full step C (ZC only):      {ms_full_C:.3f} ms")
    print(f"\n  State read speedup: {ms_all_state_api/ms_all_state_zc:.1f}x")
    print(f"  Full step speedup A→B: {ms_full_A/ms_full_B:.2f}x")
    print(f"  Full step speedup A→C: {ms_full_A/ms_full_C:.2f}x")
    print(f"\n  Overhead fractions of full_A:")
    print(f"    sim step:    {ms_sim_only/ms_full_A*100:.1f}%")
    print(f"    contacts:    {ms_get_contacts/ms_full_A*100:.1f}%")
    print(f"    state reads: {ms_all_state_api/ms_full_A*100:.1f}%")
    print()

    # ── performance_mode note ─────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  performance_mode=True note:")
    print(f"  - Pass to gs.init(backend=gs.gpu, performance_mode=True)")
    print(f"  - Bakes static tensor shapes → ~30% faster kernels")
    print(f"  - Caveat: requires recompile if scene config changes")
    print(f"  - For RL training with fixed n_envs: ALWAYS use this")
    print(f"  - Projected sim time: ~{ms_sim_only*0.70:.2f} ms/step")
    print(f"  - Projected full step: ~{ms_full_C*0.70:.2f} ms/step")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
