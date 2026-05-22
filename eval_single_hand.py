import argparse
import pickle
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from train_single_hand import BallBalanceVecEnv


def load_policy(env, train_cfg: dict, checkpoint: Path):
    log_dir = checkpoint.parent
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(checkpoint)
    print(f"Loaded {checkpoint}")
    return runner.get_inference_policy(device=gs.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_*.pt checkpoint file")
    parser.add_argument("-n", "--num_envs", type=int, default=4)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    with open(checkpoint.parent / "train_cfg.pkl", "rb") as f:
        train_cfg = pickle.load(f)

    env = BallBalanceVecEnv(n_envs=args.num_envs, show_viewer=True)
    policy = load_policy(env, train_cfg, checkpoint)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)


if __name__ == "__main__":
    main()
