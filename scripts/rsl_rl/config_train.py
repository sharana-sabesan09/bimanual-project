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
    training = {
        "task": "BallBalance-DualArm-Attention-v0",
        "name": "dual-baseline",
        "num_envs": 2048,
        "max_iterations": 500,
        "force_multiple": 1,
        "ball_mass": 0.1,
        "model_path": "training_files/density_models/g1_dual_arm_baseline.xml",
    }

    try:
        print("Initializing Genesis with GPU backend...", flush=True)
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
    except Exception as e:
        print(f"GPU init failed ({e}). Falling back to CPU...", flush=True)
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    import gymnasium as gym

    env_spec = gym.spec(training["task"])
    EnvClass = _resolve(env_spec.entry_point)
    get_train_cfg = _resolve(env_spec.kwargs["rsl_rl_cfg_entry_point"])
    
    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    

    run_name = training["name"]

    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_cfg(run_name)
    with open(log_dir / "train_cfg.pkl", "wb") as f:
        pickle.dump(train_cfg, f)

    raw_env = EnvClass(n_envs=training["num_envs"], show_viewer=False,
                debug=False, ball_pushing=True, ball_mass = training["ball_mass"],
                force_multiple=training["force_multiple"], model_path = training["model_path"]
                )
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    print(f"Training: task={training['name']} run={run_name} device={gs.device} num_envs={training['num_envs']}", flush=True)
    runner.learn(num_learning_iterations=training["max_iterations"], init_at_random_ep_len=True)
    # except Exception as e:
    #     print(e)
    #     continue
    print("FINISHED TRAINING")

if __name__ == "__main__":
    main()