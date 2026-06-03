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
import numpy as np
import statistics
import json

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

def find_goal_settle_time(distances, settle_time=15, settle_range = 0.03):
    for i in range(len(distances)-settle_time):
        tmp = distances[i:i+settle_time]
        if np.sum(np.abs(tmp))/settle_time < settle_range:
            return i
    return -1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,   required=True,
                        help="Gym env ID, e.g. BallBalance-DualArm-v0")
    parser.add_argument("--checkpoint",     type=Path,  required=True,
                        help="Path to model_*.pt checkpoint file")
    parser.add_argument("-n", "--num_envs", type=int,   default=4)
    parser.add_argument("--headless",       action="store_true", default=False)
    parser.add_argument("--action_delta",   type=float, default=0.3)
    parser.add_argument("--num_iterations", type=int, default=12)
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
                       action_delta=args.action_delta, ball_pushing=True,
                       ball_vel_range = 0.0,
                       debug = True)
    env = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    
    runner.load(checkpoint, map_location=torch.device("cpu"), strict=False)
    #runner.load(checkpoint)
    print(f"Loaded {checkpoint}")
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    num_iterations = args.num_iterations
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
            # print(f"Dones is {dones}")
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
