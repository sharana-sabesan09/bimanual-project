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

from source.tasks.double.agents.rsl_rl_ppo_cfg_attention import get_train_cfg

try:
    from importlib import metadata
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

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
    parser.add_argument("--profile",        type=int,   default=0,
                        help="If >0, profile this many PPO iterations and write "
                             "chrome trace to logs/<task>/<run>/profile.json")
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
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
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
                       debug=args.debug, ball_pushing=True
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

    if args.profile > 0:
        from torch.profiler import profile, ProfilerActivity, schedule
        trace_path = str(log_dir / "profile.json")
        # `--profile N` captures N env steps of active data after a 10-step warmup.
        # The schedule fires on raw_env._profiler.step() called inside BaseVecEnv.step(),
        # so the granularity is one env step (not one PPO iteration).
        # N=30 produces a ~5-20 MB trace that chrome://tracing and Perfetto can open.
        prof_schedule = schedule(wait=0, warmup=10, active=args.profile, repeat=1)

        def _on_trace_ready(p):
            p.export_chrome_trace(trace_path)
            print(f"Profile trace written to: {trace_path}", flush=True)
            print("Open in chrome://tracing or https://ui.perfetto.dev", flush=True)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            raw_env._profiler = prof
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        raw_env._profiler = None
    else:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
