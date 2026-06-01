"""
Unified training launcher.
Usage: python scripts/rsl_rl/train.py --task <gym_env_id> [options]

Available tasks (defined in source/tasks/*/\_\_init\_\_.py):
  BallBalance-SingleArm-v0
  BallBalance-DualArm-v0
  BallBalance-DualArm-LSTM-v0

Adding a new task requires only a gym.register() call — no changes here.
"""

import argparse
import importlib
import os
import torch
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from importlib import metadata
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

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
    parser.add_argument("-e", "--exp_name", type=str,   default=None,
                        help="Optional label appended to the run dir: logs/<task>/<timestamp>-<exp_name>")
    parser.add_argument("-n", "--num_envs", type=int,   default=512)
    parser.add_argument("--max_iterations", type=int,   default=1000)
    parser.add_argument("--headless",       action="store_true", default=False)
    parser.add_argument("--checkpoint",     type=Path, default=None)
    parser.add_argument("--action_delta",   type=float, default=0.3)
    parser.add_argument("--debug",          type=bool, default=False)
    args = parser.parse_args()

    # importing source triggers all gym.register() calls
    import gymnasium as gym
    import source  # noqa: F401

    env_spec = gym.spec(args.task)
    EnvClass    = _resolve(env_spec.entry_point)
    get_train_cfg = _resolve(env_spec.kwargs["rsl_rl_cfg_entry_point"])

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        print("Initializing Genesis with GPU backend...", flush=True)
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    except Exception as e:
        print(f"GPU init failed ({e}). Falling back to CPU...", flush=True)
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}-{args.exp_name}" if args.exp_name else timestamp
    log_dir = Path("logs") / args.task / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_cfg(run_name)
    with open(log_dir / "train_cfg.pkl", "wb") as f:
        pickle.dump(train_cfg, f)

    raw_env = EnvClass(n_envs=args.num_envs, show_viewer=not args.headless,
                       debug=args.debug
                       #action_delta=args.action_delta
                       )
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    checkpoint = args.checkpoint
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        # TODO: Remove this before pushing to main
        # checkpoint.map_location = torch.device('cpu')
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        runner.load(checkpoint)

    print(f"Training: task={args.task} run={run_name} device={gs.device} num_envs={args.num_envs}", flush=True)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
