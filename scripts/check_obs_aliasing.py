"""
Confirm the obs-aliasing bug & its fix.

Reproduces the exact rsl-rl sequence:
  1. act():            transition.observations = obs        (stored BY REFERENCE)
  2. env.step():       runs the next physics step
  3. process_env_step: storage copies transition.observations

If get_obs() hands out a reused buffer, step (2) overwrites the obs captured in
(1), so storage records s_{t+1} against the action for s_t -> training silently
fails. The fix returns a fresh tensor so the captured ref is never clobbered.
"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import genesis as gs
import gymnasium as gym
import source  # noqa: F401  -> registers envs

TASK = "BallBalance-DualArm-v0"


def build_env():
    spec = gym.spec(TASK)
    mod, cls = spec.entry_point.rsplit(":", 1)
    EnvClass = getattr(importlib.import_module(mod), cls)
    return EnvClass(show_viewer=False, n_envs=2)


def main():
    gs.init(backend=gs.cpu, logging_level="warning")
    env = build_env()
    rng = np.random.default_rng(0)
    action = rng.uniform(-1, 1, size=(env.n_envs, env.action_space.shape[0])).astype(np.float32)

    print("=" * 70)
    print("TEST A — what the FIXED get_obs() returns (the real code path)")
    print("=" * 70)
    captured, _ = env.reset()                 # (1) what act() stores by reference
    snapshot = captured.clone()               # remember s_t exactly
    obs_next, *_ = env.step(action)           # (2) next step
    drifted = (captured - snapshot).abs().max().item()
    print(f"  captured obs id == returned-step obs id ? {captured is obs_next}")
    print(f"  max |captured_after_step - captured_before_step| = {drifted:.6e}")
    print(f"  s_t vs s_t+1 actually differ (sanity)           = "
          f"{(obs_next - snapshot).abs().max().item():.4e}")
    fixed_ok = (drifted == 0.0) and (captured is not obs_next)
    print(f"  --> captured obs preserved across step ? {'PASS' if fixed_ok else 'FAIL'}")

    print()
    print("=" * 70)
    print("TEST B — the BUGGY path (capture the reused _obs_buf directly)")
    print("=" * 70)
    env.reset()
    buggy_capture = env._obs_buf             # what the old `return self._obs_buf` handed out
    buggy_snapshot = buggy_capture.clone()
    env.step(action)                          # next step overwrites _obs_buf in place
    buggy_drift = (buggy_capture - buggy_snapshot).abs().max().item()
    print(f"  max |_obs_buf_after_step - _obs_buf_before_step| = {buggy_drift:.6e}")
    bug_reproduced = buggy_drift > 0.0
    print(f"  --> reused buffer gets clobbered by next step ? "
          f"{'YES (this was the bug)' if bug_reproduced else 'no'}")

    print()
    print("=" * 70)
    ok = fixed_ok and bug_reproduced
    print(f"RESULT: {'CONFIRMED — bug was real and the fix resolves it' if ok else 'INCONCLUSIVE'}")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
