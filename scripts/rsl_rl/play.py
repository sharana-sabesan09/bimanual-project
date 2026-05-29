"""
Unified inference script for single-arm and dual-arm ball-balance tasks.
Usage: python scripts/rsl_rl/play.py --env {single,dual} --checkpoint <path> [options]
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        choices=["single", "dual"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model_*.pt checkpoint file")
    parser.add_argument("-n", "--num_envs",  type=int,   default=4)
    parser.add_argument("--headless",        action="store_true", default=False)
    parser.add_argument("--action_delta",    type=float, default=0.05,
                        help="Action delta for dual env (ignored for single)")
    args = parser.parse_args()

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper
    if args.env == "single":
        from source.tasks.single.env import SingleArmBallBalanceEnv as EnvClass
    else:
        from source.tasks.double.env import DualArmBallBalanceEnv as EnvClass

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg_path = checkpoint.parent / "train_cfg.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_cfg.pkl not found next to checkpoint: {cfg_path}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    with open(cfg_path, "rb") as f:
        train_cfg = pickle.load(f)

    env_kwargs = {}
    if args.env == "dual":
        env_kwargs["action_delta"] = args.action_delta

    raw_env = EnvClass(n_envs=args.num_envs, show_viewer=not args.headless, **env_kwargs)
    env = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    runner.load(checkpoint)
    print(f"Loaded {checkpoint}")
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)


if __name__ == "__main__":
    main()
