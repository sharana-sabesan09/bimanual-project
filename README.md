# Bimanual Ball Balance

Unitree G1 humanoid (fixed base) balancing a ball on a tray.  
Supports single-arm (7 DOF, right wrist) and dual-arm (14 DOF, both wrists).

## Setup

```bash
conda activate <your_env>
pip install -r requirements.txt
```

After cloning, configure the notebook git filter (strips outputs before every commit):

```bash
python -m nbstripout --install --attributes .gitattributes
```

## Project structure

```
source/tasks/single/        — SingleArmBallBalanceEnv + PPO config
source/tasks/double/        — DualArmBallBalanceEnv + PPO config
scripts/
  zero_agent.py             — sends all-zero actions
  random_agent.py           — sends uniform-random actions
  rsl_rl/
    train.py                — PPO training launcher
    play.py                 — policy rollout / eval
RL_lib/                     — custom actor/critic architectures (LSTM + deep MLP)
assets/mujoco_menagerie/    — MJCF robot files
```

## Running

**Zero / random agents:**
```bash
python scripts/zero_agent.py   --env {single,dual}
python scripts/random_agent.py --env {single,dual}
```

**Train:**
```bash
python scripts/rsl_rl/train.py --env {single,dual}
python scripts/rsl_rl/train.py --env dual -n 2048 --max_iterations 1000 --action_delta 0.3
```
Viewer is shown by default. Pass `--headless` to disable.  
Checkpoints and TensorBoard logs are saved to `logs/<exp_name>/`.

**Eval:**
```bash
python scripts/rsl_rl/play.py --env {single,dual} --checkpoint logs/<exp_name>/model_1000.pt
```

**TensorBoard:**
```bash
tensorboard --logdir logs/
```

## Kaggle training

Open `kaggle_training.ipynb`. Set `ENV`, `EXP_NAME`, and `N_ENVS` in cell 1, then run all cells.  
The notebook detects Kaggle vs local automatically — locally it uses `N_ENVS=4` and `MAX_ITERATIONS=5` for a quick smoke test.
