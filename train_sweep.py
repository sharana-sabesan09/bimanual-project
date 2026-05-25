"""
Structured hyperparameter sweep for ball-balance training.

Each experiment is a named override on top of the base config.
All runs land in logs/<sweep_name>/<exp_name>/ so you can compare them
side by side in TensorBoard:

    tensorboard --logdir logs/<sweep_name>

Results summary is written to logs/<sweep_name>/results.csv when all runs finish.
"""

import argparse
import csv
import time
import pickle
from pathlib import Path

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

import importlib

# ── base config ───────────────────────────────────────────────────────────────

BASE_TRAIN = dict(
    learning_rate=3e-4,
    num_steps_per_env=48,
)

BASE_ENV = dict(
    n_envs=1024,
    action_delta=0.3,
    ball_vel_range=0.0,
)

# ── experiments ───────────────────────────────────────────────────────────────
# (name, train_overrides, env_overrides)
# Only specify what differs from base.

EXPERIMENTS = [
    ("baseline",   {},                       {}),
    ("lr_1e-4",    {"learning_rate": 1e-4},  {}),
    ("lr_1e-3",    {"learning_rate": 1e-3},  {}),
    ("delta_0.15", {},                       {"action_delta": 0.15}),
    ("delta_0.5",  {},                       {"action_delta": 0.5}),
]

# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(name, train_ov, env_ov, sweep_dir, max_iterations, get_train_cfg, VecEnvClass, chunk_size):
    train_kw = {**BASE_TRAIN, **train_ov}
    env_kw   = {**BASE_ENV,   **env_ov}

    cfg = get_train_cfg(name)
    cfg["algorithm"]["learning_rate"] = train_kw["learning_rate"]
    cfg["num_steps_per_env"]          = train_kw["num_steps_per_env"]

    log_dir = sweep_dir / name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.pkl", "wb") as f:
        pickle.dump({"train": train_kw, "env": env_kw}, f)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  lr={train_kw['learning_rate']:.0e}  "
            f"steps/env={train_kw['num_steps_per_env']}  "
            f"delta={env_kw['action_delta']}  "
            f"ball_vel_range={env_kw.get('ball_vel_range', 0.0)}")
    print(f"{'='*60}\n")

    env = VecEnvClass(
        n_envs=env_kw["n_envs"],
        show_viewer=False,
        action_delta=env_kw["action_delta"],
        ball_vel_range=env_kw.get("ball_vel_range", 0.0),
    )

    runner = OnPolicyRunner(env, cfg, str(log_dir), device=gs.device)

    iterations_done = 0
    t0 = time.time()
    while iterations_done < max_iterations:
        to_run = min(chunk_size, max_iterations - iterations_done)
        iter_t0 = time.time()
        runner.learn(num_learning_iterations=to_run, init_at_random_ep_len=(iterations_done == 0))
        iter_elapsed = time.time() - iter_t0
        iterations_done += to_run
        total_elapsed = time.time() - t0
        avg_per_iter = total_elapsed / iterations_done
        remaining = max_iterations - iterations_done
        eta = remaining * avg_per_iter
        print(f"[{time.strftime('%H:%M:%S')}] Experiment={name} "
              f"{iterations_done}/{max_iterations} (+{to_run}) "
              f"chunk_time={iter_elapsed:.1f}s "
              f"total={total_elapsed:.1f}s "
              f"ETA={eta/60:.1f}m")
        try:
            import sys
            sys.stdout.flush()
        except Exception:
            pass

    return time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sweep_name", default="sweep_01")
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="number of iterations per progress printout")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="override the number of parallel environments for this sweep")
    parser.add_argument("--action_delta", type=float, default=None,
                        help="override action_delta for all experiments")
    parser.add_argument("--ball_vel_range", type=float, default=None,
                        help="override initial ball velocity magnitude for all experiments")
    parser.add_argument("--experiments", nargs="+",
                        help="subset of experiment names to run (default: all)")
    parser.add_argument("--mode", choices=["single", "double"], default="single",
                        help="which policy/env to use: single (right arm) or double (both arms)")
    args = parser.parse_args()

    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    except Exception:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    sweep_dir = Path("logs") / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # dynamically import the chosen training module
    mod_name = "train_single_hand" if args.mode == "single" else "train_double_hand"
    mod = importlib.import_module(mod_name)
    get_train_cfg = getattr(mod, "get_train_cfg")
    VecEnvClass = getattr(mod, "BallBalanceVecEnv")

    # apply CLI overrides to the environment config
    if args.n_envs is not None:
        BASE_ENV["n_envs"] = args.n_envs
    if args.action_delta is not None:
        BASE_ENV["action_delta"] = args.action_delta
    if args.ball_vel_range is not None:
        BASE_ENV["ball_vel_range"] = args.ball_vel_range

    exps = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        exps  = [(n, t, e) for n, t, e in EXPERIMENTS if n in names]
        if not exps:
            raise ValueError(f"No match. Available: {[n for n, _, _ in EXPERIMENTS]}")

    rows = []
    for name, train_ov, env_ov in exps:
        elapsed = run_experiment(name, train_ov, env_ov, sweep_dir, args.max_iterations,
                                 get_train_cfg, VecEnvClass, args.chunk_size)
        rows.append({
            "experiment":    name,
            "learning_rate": {**BASE_TRAIN, **train_ov}["learning_rate"],
            "action_delta":  {**BASE_ENV,   **env_ov}["action_delta"],
            "ball_vel_range": {**BASE_ENV,   **env_ov}["ball_vel_range"],
            "elapsed_s":     f"{elapsed:.0f}",
        })

    csv_path = sweep_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAll done. Compare runs:")
    print(f"  tensorboard --logdir {sweep_dir}\n")


if __name__ == "__main__":
    main()