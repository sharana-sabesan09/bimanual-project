"""
Collect attention weights from a trained checkpoint and save aggregated stats.

Run once per checkpoint:
    python scripts/collect_attention.py --checkpoint logs/.../01kg/model_500.pt --label 0.1kg --steps 500
    python scripts/collect_attention.py --checkpoint logs/.../03kg/model_500.pt --label 0.3kg --steps 500
    python scripts/collect_attention.py --checkpoint logs/.../05kg/model_500.pt --label 0.5kg --steps 500

Output: attn_data/<label>.npz  (load with np.load)
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Remove scripts/ from sys.path to prevent scripts/rsl_rl/ shadowing the installed rsl_rl package
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir in sys.path:
    sys.path.remove(_scripts_dir)

# Add project root so we can import source, RL_lib etc.
sys.path.insert(0, os.path.dirname(_scripts_dir))

import numpy as np
import torch

BALL_TOKEN_LABELS = [
    "ball_x", "ball_y", "ball_z",
    "vel_x",  "vel_y",  "vel_z",
    "goal_x", "goal_y", "goal_z",
]


def find_actor(runner):
    """Try common rsl-rl attribute paths to locate the actor model."""
    for path in ["alg.actor", "actor", "alg.policy"]:
        obj = runner
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "collect_attn"):
            return obj
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model_500.pt")
    parser.add_argument("--label", type=str, required=True,
                        help="Label for this run, e.g. '0.1kg'")
    parser.add_argument("--task", type=str,
                        default="BallBalance-DualArm-Attention-v0")
    parser.add_argument("--steps", type=int, default=500,
                        help="How many env steps to collect attention over")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="attn_data")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    cfg_path = checkpoint.parent / "train_cfg.pkl"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_cfg.pkl not found at: {cfg_path}")

    import genesis as gs
    import gymnasium as gym
    import source  # noqa: F401 — triggers all gym.register() calls
    from rsl_rl.runners import OnPolicyRunner
    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper

    gs.init(backend=gs.cpu, precision="32", logging_level="warning", device="cpu")

    with open(cfg_path, "rb") as f:
        train_cfg = pickle.load(f)

    env_spec = gym.spec(args.task)

    import importlib
    def _resolve(ep):
        mod, cls = ep.rsplit(":", 1)
        return getattr(importlib.import_module(mod), cls)

    EnvClass = _resolve(env_spec.entry_point)
    raw_env  = EnvClass(n_envs=args.num_envs, show_viewer=False)
    env      = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    runner.load(checkpoint, map_location=torch.device("cpu"))
    print(f"Loaded {checkpoint}")

    actor = find_actor(runner)
    if actor is None:
        raise RuntimeError(
            "Could not find AttentionActor on runner. "
            "Make sure this checkpoint was trained with BallBalance-DualArm-Attention-v0."
        )

    actor.collect_attn = False  # we'll collect manually via last_attn

    policy = runner.get_inference_policy(device=gs.device)
    obs    = env.reset()

    left_accum  = []
    right_accum = []

    print(f"Collecting {args.steps} steps...")
    with torch.no_grad():
        for step in range(args.steps):
            actions = policy(obs)

            # grab attention weights from this forward pass
            if actor.last_attn is not None:
                lw = actor.last_attn["left"]   # (B, heads, 1, 9)
                rw = actor.last_attn["right"]  # (B, heads, 1, 9)
                # average over batch and heads, squeeze query dim → (9,)
                left_accum.append(lw.squeeze(2).mean(dim=(0, 1)).cpu().numpy())
                right_accum.append(rw.squeeze(2).mean(dim=(0, 1)).cpu().numpy())

            obs, _, dones, _ = env.step(actions)

            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}")

    if not left_accum:
        raise RuntimeError(
            "No attention weights were captured. "
            "The actor may not have called get_latent with return_weights — "
            "check that collect_attn is wired into the forward pass."
        )

    left_arr  = np.stack(left_accum)   # (steps, 9)
    right_arr = np.stack(right_accum)  # (steps, 9)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.npz"
    np.savez(out_path, left=left_arr, right=right_arr,
             token_labels=BALL_TOKEN_LABELS, label=args.label)

    print(f"\nSaved to {out_path}")
    print(f"Left  arm mean attention:  {left_arr.mean(axis=0).round(3)}")
    print(f"Right arm mean attention:  {right_arr.mean(axis=0).round(3)}")


if __name__ == "__main__":
    main()
