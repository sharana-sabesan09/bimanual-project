# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# List all registered gym environments
python scripts/list_envs.py

# Rollout agents (viewer on by default)
python scripts/zero_agent.py   --task BallBalance-DualArm-v0 --num_envs 4
python scripts/random_agent.py --task BallBalance-SingleArm-v0 --num_envs 4

# Training (GPU→CPU fallback is automatic)
python scripts/rsl_rl/train.py --task BallBalance-DualArm-v0 -n 512 --max_iterations 1000 --headless
python scripts/rsl_rl/train.py --task BallBalance-DualArm-v0 -n 16 --max_iterations 5 --headless -e smoke   # quick smoke test

# Inference
python scripts/rsl_rl/play.py --task BallBalance-DualArm-v0 --checkpoint logs/BallBalance-DualArm-v0/<run>/model_1000.pt

# TensorBoard
tensorboard --logdir logs/
```

Pass `--headless` to suppress the Genesis viewer. Omit it to show it.

## Architecture

### Environment hierarchy

```
source/tasks/base_env.py       BaseVecEnv(gym.Env)
source/tasks/single/env.py     SingleArmBallBalanceEnv(BaseVecEnv)   — 6-dim EE action (IK→7 DOF), 23-dim obs
source/tasks/double/env.py     DualArmBallBalanceEnv(BaseVecEnv)     — 14 DOF, 37-dim obs
```

`BaseVecEnv` owns the Genesis scene lifecycle, episode tracking (`episode_length_buf`, `terminated`, `truncated`), and the concrete `step()`/`reset()` flow. It knows nothing about robots or rewards. Subclasses implement seven abstract methods: `_build_scene`, `_post_build_init`, `_reset_env`, `_apply_action`, `get_obs`, `get_termination`, `_compute_reward`.

**Critical init order in `BaseVecEnv.__init__`:** `self.n_envs` is stored from the constructor arg *before* anything Genesis touches. Do not use `self.scene.n_envs` to size tensors — Genesis does not populate it correctly until after the first `scene.step()`. All tensor allocations (including `prev_actions` in subclasses) must use `self.n_envs`.

**Everything must be batched.** All tensors, actions, observations, and sim calls must carry the leading `n_envs` dimension at all times — even when `n_envs=1`. Never unsqueeze or broadcast silently inside the env or base class to paper over an unbatched input; callers (scripts, agents, teleop) are required to pass correctly shaped `(n_envs, ...)` arrays. This invariant is what makes single-env debugging and multi-env parallel training use identical code paths.

**Step flow:** `_apply_action` → `scene.step` → `episode_length_buf += 1` → `get_obs` → `get_termination` (sets `self.terminated`/`self.truncated`) → `_compute_reward` (may read `self.terminated`) → auto-reset done envs → return 5-tuple `(obs, reward, terminated, truncated, info)`.

### Task registration and the `--task` flag

`import source` (done inside `main()` of every script) triggers `source/__init__.py` → `import_tasks()` → imports every subpackage under `source/tasks/`, firing all `gym.register()` calls. Scripts resolve the env class and PPO config entirely from the registry:

```python
env_spec     = gym.spec(args.task)
EnvClass     = _resolve(env_spec.entry_point)          # "module:Class"
get_train_cfg = _resolve(env_spec.kwargs["rsl_rl_cfg_entry_point"])
```

**To add a new task variant** (different architecture, reward, hyperparams): add a `gym.register()` call in the relevant `source/tasks/*/\_\_init\_\_.py` pointing to a new `rsl_rl_ppo_cfg_*.py`. No changes to any script needed.

### PPO config files

Each task folder has `agents/rsl_rl_ppo_cfg.py` containing only `get_train_cfg(exp_name) -> dict`. The dict is passed directly to rsl-rl's `OnPolicyRunner`. `class_name` values in `"actor"` and `"critic"` are resolved by rsl-rl's `resolve_callable` — use a dotted module path (e.g. `"RL_lib.custom_actor_critic.LSTMActor"`) to reference classes outside the rsl-rl package without patching.

### rsl-rl adapter

`scripts/rsl_rl/vec_env.py::RslRlVecEnvWrapper` is a pure interface adapter (~30 lines). It wraps a `BaseVecEnv` and exposes the fields rsl-rl's `OnPolicyRunner` expects (`num_envs`, `num_actions`, `max_episode_length`, `episode_length_buf` as a pass-through property with setter). All episode logic lives in the env.

### Custom architectures

`RL_lib/custom_actor_critic.py` contains `LSTMActor` (subclass of rsl-rl's `RNNModel`) and `DeepMLPCritic` (subclass of `MLPModel`). Reference them via dotted path in a cfg file. `RL_lib/example_train_cfg.py` and `source/tasks/double/agents/rsl_rl_ppo_cfg_custom.py` show the pattern.

### Log directory structure

`logs/<task_id>/<timestamp>[-<exp_name>]/` — multiple runs of the same task never collide. `train_cfg.pkl` is saved alongside checkpoints so `play.py` can reconstruct the runner without needing the original cfg file.

### Notebook

`kaggle_training.ipynb` runs on both Kaggle and locally. `ON_KAGGLE = os.path.exists('/kaggle')` gates the clone/install cells. Locally it defaults to `N_ENVS=4`, `MAX_ITERATIONS=5` for a quick smoke test. `nbstripout` is configured as a git filter (`.gitattributes`) — teammates must run `python -m nbstripout --install --attributes .gitattributes` once after cloning.

### MJCF assets

`assets/mujoco_menagerie/unitree_g1/g1_single_arm.xml` — right wrist tray only.  
`assets/mujoco_menagerie/unitree_g1/g1_dual_arm.xml` — both wrists welded.
