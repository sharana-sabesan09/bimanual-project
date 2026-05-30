"""
Unified inference script.
Usage: python scripts/rsl_rl/play.py --task <gym_env_id> --checkpoint <path> [options]
"""

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from importlib import metadata
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs


def _resolve(entry_point: str):
    """Import and return the object at 'module.path:ClassName'."""
    module_path, attr = entry_point.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,   required=True,
                        help="Gym env ID, e.g. BallBalance-DualArm-v0")
    parser.add_argument("--checkpoint",     type=Path,  required=True,
                        help="Path to model_*.pt checkpoint file")
    parser.add_argument("-n", "--num_envs", type=int,   default=4)
    parser.add_argument("--headless",       action="store_true", default=False)
    parser.add_argument("--action_delta",   type=float, default=0.3)
    args = parser.parse_args()

    # importing source triggers all gym.register() calls
    import gymnasium as gym
    import source  # noqa: F401

    env_spec  = gym.spec(args.task)
    EnvClass  = _resolve(env_spec.entry_point)

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Checkpoint is {checkpoint}")
    # TODO: Remove this before pushing to main
    # checkpoint.map_location = torch.device('cpu')

    cfg_path = checkpoint.parent / "train_cfg.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_cfg.pkl not found next to checkpoint: {cfg_path}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    with open(cfg_path, "rb") as f:
        train_cfg = pickle.load(f)

    raw_env = EnvClass(n_envs=args.num_envs, show_viewer=not args.headless,
                       action_delta=args.action_delta)
    env = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    
    runner.load(checkpoint, map_location=torch.device("cpu"))
    #runner.load(checkpoint)
    print(f"Loaded {checkpoint}")
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)


if __name__ == "__main__":
    main()
