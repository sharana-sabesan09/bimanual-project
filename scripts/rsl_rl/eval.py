"""
Evaluation script — per-goal-switch settle time, success rate, and episode metrics.
Usage: python scripts/rsl_rl/eval.py --task <gym_env_id> --checkpoint <path> [options]
"""

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import statistics

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

GOAL_SWITCH_PERIOD = 100  # steps between goal changes — must match env constant


def _resolve(entry_point: str):
    module_path, attr = entry_point.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def find_settle_time(distances, window=30, threshold=0.02):
    """
    First index i where the next `window` steps all stay within `threshold` metres
    of distances[i]. Returns -1 if ball never settles within the segment.
    """
    distances = np.array(distances)
    for i in range(len(distances) - window):
        if np.max(np.abs(distances[i:i + window] - distances[i])) < threshold:
            return i
    return -1


def per_goal_metrics(episode_distances):
    """
    Slice an episode's distance trace into GOAL_SWITCH_PERIOD-length segments
    (one per goal change) and compute settle time + success for each.

    Returns:
        settle_times  — list of settle times in seconds for successful segments
        success_flags — list of bools, one per goal segment
        raw_steps     — list of raw settle step indices (-1 = DNF), one per segment
    """
    settle_times = []
    success_flags = []
    raw_steps = []

    n = len(episode_distances)
    # Walk through each goal-window; skip the final partial window if < 10 steps
    for start in range(0, n, GOAL_SWITCH_PERIOD):
        segment = episode_distances[start: start + GOAL_SWITCH_PERIOD]
        if len(segment) < 10:
            continue
        idx = find_settle_time(segment)
        raw_steps.append(idx)
        if idx != -1:
            settle_times.append(idx / 50.0)   # 50 Hz (dt=0.02) → seconds
            success_flags.append(True)
        else:
            success_flags.append(False)

    return settle_times, success_flags, raw_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,  required=True)
    parser.add_argument("--checkpoint",     type=Path, required=True)
    parser.add_argument("-n", "--num_envs", type=int,  default=4)
    parser.add_argument("--headless",       action="store_true", default=False)
    parser.add_argument("--action_delta",   type=float, default=0.3)
    parser.add_argument("--num_episodes",   type=int,  default=50,
                        help="Total completed episodes to collect before stopping")
    args = parser.parse_args()

    import gymnasium as gym
    import source  # noqa: F401

    env_spec = gym.spec(args.task)
    EnvClass = _resolve(env_spec.entry_point)

    from scripts.rsl_rl.vec_env import RslRlVecEnvWrapper

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg_path = checkpoint.parent / "train_cfg.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_cfg.pkl not found: {cfg_path}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    with open(cfg_path, "rb") as f:
        train_cfg = pickle.load(f)

    raw_env = EnvClass(n_envs=args.num_envs, show_viewer=not args.headless,
                       action_delta=args.action_delta, debug=True)
    env = RslRlVecEnvWrapper(raw_env)

    runner = OnPolicyRunner(env, train_cfg, str(checkpoint.parent), device=gs.device)
    runner.load(checkpoint, map_location=torch.device("cpu"))
    print(f"Loaded {checkpoint}")
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    completed = 0
    all_episode_distances = []

    with torch.no_grad():
        while completed < args.num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            for env_idx in dones.nonzero():
                ep_dists = env._env._return_and_reset_debug(env_idx)
                all_episode_distances.append(ep_dists)
                completed += 1
                if completed >= args.num_episodes:
                    break

    # ── Episode-level metrics ──────────────────────────────────────────────
    avg_dist_per_ep   = [statistics.fmean(d) for d in all_episode_distances]
    after_2s_per_ep   = [statistics.fmean(d[100:]) for d in all_episode_distances if len(d) > 100]
    ep_lengths        = [len(d) for d in all_episode_distances]

    # ── Per-goal-switch metrics ────────────────────────────────────────────
    all_settle_times  = []   # seconds, successful segments only
    all_success_flags = []   # one bool per goal segment across all episodes

    for ep_dists in all_episode_distances:
        settle_times, success_flags, _ = per_goal_metrics(ep_dists)
        all_settle_times.extend(settle_times)
        all_success_flags.extend(success_flags)

    total_segments  = len(all_success_flags)
    total_successes = sum(all_success_flags)
    success_rate    = total_successes / total_segments if total_segments > 0 else float("nan")

    # Mean settle time with DNF penalty = full window (1.0 s)
    settle_with_penalty = all_settle_times + [1.0] * (total_segments - total_successes)

    print(
        f"\nEVALUATION — {args.task}"
        f"\n  Checkpoint : {checkpoint}"
        f"\n  Episodes   : {completed}"
        f"\n{'─'*52}"
        f"\n  EPISODE-LEVEL"
        f"\n    Avg distance from goal        : {statistics.fmean(avg_dist_per_ep):.4f} m"
        f"\n    Avg distance after 2 s        : {statistics.fmean(after_2s_per_ep):.4f} m"
        f"\n    Avg episode length            : {statistics.fmean(ep_lengths):.1f} steps"
        f"\n{'─'*52}"
        f"\n  PER-GOAL-SWITCH  (window={GOAL_SWITCH_PERIOD} steps, {total_segments} segments)"
        f"\n    Success rate                  : {success_rate*100:.1f}%  ({total_successes}/{total_segments})"
    )
    if all_settle_times:
        print(
            f"    Median settle time (success) : {statistics.median(all_settle_times):.3f} s"
            f"\n    Mean settle time (success)  : {statistics.fmean(all_settle_times):.3f} s"
            f"\n    Mean settle (DNF=1.0 s)     : {statistics.fmean(settle_with_penalty):.3f} s"
        )
    else:
        print("    No successful settlements recorded.")
    print()

    return {
        "task": args.task,
        "avg_dist": statistics.fmean(avg_dist_per_ep),
        "avg_dist_after_2s": statistics.fmean(after_2s_per_ep) if after_2s_per_ep else float("nan"),
        "avg_ep_length": statistics.fmean(ep_lengths),
        "success_rate": success_rate,
        "median_settle_s": statistics.median(all_settle_times) if all_settle_times else float("nan"),
        "mean_settle_s": statistics.fmean(all_settle_times) if all_settle_times else float("nan"),
        "mean_settle_with_penalty_s": statistics.fmean(settle_with_penalty) if settle_with_penalty else float("nan"),
        "n_segments": total_segments,
    }


if __name__ == "__main__":
    main()
