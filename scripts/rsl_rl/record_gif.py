"""
Record a single-episode GIF for a trained policy using Genesis offscreen camera.
Called as a subprocess from kaggle_eval.ipynb so Genesis reinitialises cleanly per run.

Usage:
    python scripts/rsl_rl/record_gif.py \\
        --task BallBalance-DualArm-v0 \\
        --checkpoint logs/.../model_1000.pt \\
        --out eval_DualArm_MLP.gif \\
        --steps 500 --downsample 3
"""

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner


def _resolve(entry_point: str):
    module_path, attr = entry_point.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         type=str,  required=True)
    parser.add_argument("--checkpoint",   type=Path, required=True)
    parser.add_argument("--out",          type=Path, required=True)
    parser.add_argument("--steps",        type=int,  default=500)
    parser.add_argument("--downsample",   type=int,  default=3,
                        help="Keep every Nth frame to reduce GIF size")
    parser.add_argument("--action_delta", type=float, default=0.3)
    parser.add_argument("--fps",          type=int,  default=20)
    args = parser.parse_args()

    import gymnasium as gym
    import source  # noqa: F401

    env_spec = gym.spec(args.task)
    EnvClass = _resolve(env_spec.entry_point)

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper

    checkpoint = args.checkpoint.resolve()
    cfg_path = checkpoint.parent / "train_cfg.pkl"
    with open(cfg_path, "rb") as f:
        train_cfg = pickle.load(f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    raw_env = EnvClass(n_envs=1, show_viewer=False,
                       action_delta=args.action_delta,
                       record=True)
    env = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    runner.load(checkpoint, map_location=torch.device("cpu"))
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    frames = []

    with torch.no_grad():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if step % args.downsample == 0:
                frame = raw_env.render_frame()  # (H, W, 3) uint8
                if frame is not None:
                    frames.append(frame)
            # Reset on done but keep rolling for full GIF length
            if dones.any():
                obs, _ = env._env.reset(), {}
                obs = env.get_observations()

    if not frames:
        print("ERROR: no frames captured — Genesis camera may not be supported in this build.")
        sys.exit(1)

    try:
        import imageio
        imageio.mimsave(str(args.out), frames, fps=args.fps, loop=0)
        size_mb = args.out.stat().st_size / 1e6
        print(f"Saved {len(frames)}-frame GIF → {args.out}  ({size_mb:.1f} MB)")
    except ImportError:
        # Fallback: save individual frames as PNG and use ffmpeg
        frame_dir = args.out.parent / f"_frames_{args.out.stem}"
        frame_dir.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            from PIL import Image
            Image.fromarray(f).save(frame_dir / f"frame_{i:04d}.png")
        os.system(
            f"ffmpeg -y -r {args.fps} -i {frame_dir}/frame_%04d.png "
            f"-vf 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' {args.out}"
        )
        print(f"Saved GIF via ffmpeg → {args.out}")


if __name__ == "__main__":
    main()
