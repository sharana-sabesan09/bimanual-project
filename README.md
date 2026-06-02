# Bimanual Ball Balance

Unitree G1 humanoid (fixed base) balancing a ball on a tray. Supports single-arm (7 DOF) and dual-arm (14 DOF). PPO training via rsl-rl-lib >= 5, simulation via Genesis.

## Setup

```bash
pip install -r requirements.txt
python -m nbstripout --install --attributes .gitattributes  # strips notebook outputs before every commit
```

## Project structure

```
source/tasks/
  base_env.py              — BaseVecEnv(gym.Env): scene lifecycle, step/reset flow, episode tracking
  single/
    env.py                 — SingleArmBallBalanceEnv  (7 DOF, 23-dim obs)
    agents/
      rsl_rl_ppo_cfg.py    — PPO hyperparameters
  double/
    env.py                 — DualArmBallBalanceEnv    (14 DOF, 37-dim obs)
    agents/
      rsl_rl_ppo_cfg.py        — PPO hyperparameters (default MLP)
      rsl_rl_ppo_cfg_custom.py — PPO hyperparameters (LSTM actor + deep critic)

scripts/
  list_envs.py             — print all registered gym task IDs
  zero_agent.py            — all-zeros rollout
  random_agent.py          — random action rollout
  rsl_rl/
    train.py               — PPO training launcher
    play.py                — policy inference
    vec_env.py             — thin rsl-rl adapter (RslRlVecEnvWrapper)

RL_lib/
  custom_actor_critic.py   — LSTMActor, DeepMLPCritic (drop-in rsl-rl model replacements)
  example_train_cfg.py     — example cfg using the custom architectures

assets/mujoco_menagerie/unitree_g1/
  g1_single_arm.xml        — right wrist tray
  g1_dual_arm.xml          — both wrists welded
```

All scripts resolve the env class and PPO config from the gym registry via `--task`. To add a new variant, add a `gym.register()` call in `source/tasks/*/\_\_init\_\_.py` — no changes to scripts needed.

## Usage

```bash
python scripts/list_envs.py

python scripts/zero_agent.py   --task BallBalance-DualArm-v0
python scripts/random_agent.py --task BallBalance-SingleArm-v0 --num_envs 4

python scripts/rsl_rl/train.py --task BallBalance-DualArm-v0 -n 512 --headless
python scripts/rsl_rl/train.py --task BallBalance-DualArm-LSTM-v0 -n 512 --headless -e my_run
python scripts/rsl_rl/play.py  --task BallBalance-DualArm-v0 --checkpoint logs/.../model_1000.pt
python scripts/rsl_rl/eval.py --task BallBalance-DualArm-Attention-v0 --checkpoint logs/.../model_1000.py --num_iterations 100 -n 4

tensorboard --logdir logs/
```

Logs are saved to `logs/<task>/<timestamp>[-<label>]/`. Viewer is shown by default — pass `--headless` to disable.

## Kaggle

Open `kaggle_training.ipynb`, set `TASK` and `EXP_NAME` in cell 1, run all cells. Detects Kaggle vs local automatically (`N_ENVS=4`, `MAX_ITERATIONS=5` locally for a quick smoke test).
