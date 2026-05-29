import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import source  # noqa: F401 — triggers all gym.register() calls

envs = [spec for spec in gym.envs.registry.values() if spec.id.startswith("BallBalance")]

for spec in sorted(envs, key=lambda s: s.id):
    cfg_ep = spec.kwargs.get("rsl_rl_cfg_entry_point", "—")
    print(f"{spec.id}")
    print(f"  env : {spec.entry_point}")
    print(f"  cfg : {cfg_ep}")
