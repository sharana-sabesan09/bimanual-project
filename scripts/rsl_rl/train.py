"""
Unified training launcher for single-arm and dual-arm ball-balance tasks.
Usage: python scripts/rsl_rl/train.py --env {single,dual} [options]
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

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",            choices=["single", "dual"], required=True)
    parser.add_argument("-e", "--exp_name", type=str,   default="ball_balance")
    parser.add_argument("-n", "--num_envs", type=int,   default=512)
    parser.add_argument("--max_iterations", type=int,   default=1000)
    parser.add_argument("--headless",       action="store_true", default=False)
    parser.add_argument("--action_delta",   type=float, default=0.05,
                        help="Action delta for dual env (ignored for single)")
    args = parser.parse_args()

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper
    if args.env == "single":
        from source.tasks.single.env import SingleArmBallBalanceEnv as EnvClass
        from source.tasks.single.agents.rsl_rl_ppo_cfg import get_train_cfg
    else:
        from source.tasks.double.env import DualArmBallBalanceEnv as EnvClass
        from source.tasks.double.agents.rsl_rl_ppo_cfg import get_train_cfg

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

    log_dir = Path("logs") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_cfg(args.exp_name)
    with open(log_dir / "train_cfg.pkl", "wb") as f:
        pickle.dump(train_cfg, f)

    env_kwargs = {}
    if args.env == "dual":
        env_kwargs["action_delta"] = args.action_delta

    raw_env = EnvClass(n_envs=args.num_envs, show_viewer=not args.headless, **env_kwargs)
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    print(f"Training: env={args.env} device={gs.device} num_envs={args.num_envs}", flush=True)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
