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
import statistics
import json
import numpy as np


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


def find_goal_settle_time(distances, settle_time=15, settle_range = 0.03):
    for i in range(len(distances)-settle_time):
        tmp = distances[i:i+settle_time]
        if np.sum(np.abs(tmp))/settle_time < settle_range:
            return i
    return -1

def main():
    training = {
        "task": "BallBalance-SingleArm-v0",
        "name": "single-baseline",
        "category": "mass",
        "num_envs": 10,
        "num_iterations": 100,
        "force_multiple": 1,
        "ball_mass": .1,
        "model_path": "training_files/density_models/g1_single_arm_baseline.xml",
        "checkpoint": "evals/single-baseline/model_499.pt" # relative path
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
                debug=True, ball_pushing=True, ball_mass = training["ball_mass"],
                force_multiple=training["force_multiple"], model_path = training["model_path"]
                )
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    checkpoint = Path(training["checkpoint"]).resolve()
    runner.load(checkpoint, map_location=torch.device("cpu"))
    print(f"Loaded {checkpoint}")
    policy = runner.get_inference_policy(device = gs.device)
    obs = env.reset()
    num_iterations = training["num_iterations"]
    iterations = 0
    distances_from_goal_list = []
    distances_from_goal_switch_list = []
    full_trials = 0
    with torch.no_grad():
        while (iterations<num_iterations):
            actions = policy(obs)
            pre_step = env.episode_length_buf
            obs, _, dones, _ = env.step(actions)
            if torch.any(env.episode_length_buf == 499):
                full_trials += 1
            iterations += dones.sum(dim=-1)
            done_indices = dones.nonzero()
            # print(f"Done Indices: {done_indices}")
            for env_idx in done_indices:
                distances_from_goal_list.append(env._env._return_and_reset_debug(env_idx))
    
    successes = 0
    total_trials = 0
    settle_times = []

    for dist_list in distances_from_goal_list:
        goal_switch_period = env._env.goal_switch_period
        i = 0
        while i < len(dist_list):
            if i+goal_switch_period < len(dist_list):
                settle_time = find_goal_settle_time(dist_list[i:i+goal_switch_period])
                #distances_from_goal_switch_list.append(dist_list[i:i+goal_switch_period].copy())
            else:
                settle_time = find_goal_settle_time(dist_list[i:])
            total_trials += 1
            if settle_time != -1:
                successes += 1
            settle_times.append(settle_time/100 if settle_time != -1 else goal_switch_period/100)
            i += goal_switch_period
    
    average_dists = [statistics.fmean(distances) for distances in distances_from_goal_list]
    avg_dists_after_1_sec = [statistics.fmean(distances[100:]) for distances in distances_from_goal_list if len(distances) > 100]
    avg_iteration_length = statistics.mean(len(distances) for distances in distances_from_goal_list)

    eval_dict = {
        "category": training["category"],
        "full_trial_rate": float(full_trials/iterations),
        "avg_dist_from_goal": statistics.fmean(average_dists),
        "avg_dist_from_goal_1_sec": statistics.fmean(avg_dists_after_1_sec),
        "avg_itn_len": float(avg_iteration_length),
        "avg_settle_time": statistics.mean(settle_times),
        "success_rate": float(successes/total_trials)
    }

    with open(checkpoint.parent/"eval.json", "w") as f:
        f.write(json.dumps(eval_dict))

    print(f"EVALUATION STATS: \
          \n--------------------------------\
    \n\Full trial rate: {full_trials/iterations:.6f} \
    \nAverage distance from goal: {statistics.fmean(average_dists):.6f} \
    \nAverage distance from goal after one second: {statistics.fmean(avg_dists_after_1_sec):.6f}\
    \nAverage iteration length: {avg_iteration_length:.6f}\
    \nAverage settle time: {statistics.mean(settle_times):.6f} seconds \
    \nSuccess Rate: {successes/total_trials:.6f} \
    ")

if __name__ == "__main__":
    main()
