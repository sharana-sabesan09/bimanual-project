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

---

## Genesis high-performance API patterns

### Init — always use `performance_mode=True` for training

```python
gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
```

Bakes static tensor shapes into compiled CUDA kernels (~26% faster sim). Safe for RL training because n_envs is fixed. Only downside: requires kernel recompile if scene config changes between runs (handled by Quadrants cache).

### Scene — tune RigidOptions

Pass `rigid_options` to `gs.Scene()`. Current tuned values for this project:

```python
gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=1),
    rigid_options=gs.options.RigidOptions(
        iterations=10,      # Newton solver iterations (default 50 — overkill for this task)
        ls_iterations=5,    # line-search iterations per Newton step (default 50)
    ),
)
```

**substeps:** each `scene.step()` runs the full physics pipeline `substeps` times. Genesis's own benchmarks always use `substeps=1`. The project started at 4, reduced to 1 — roughly 4x sim throughput. Increase if contacts become unstable (tray phasing through fingers). `use_contact_island=True` is documented but crashes on Genesis 0.4.7 (Quadrants compilation bug).

**iterations/ls_iterations:** the constraint solver exits early via `tolerance=1e-5` when converged, so the budget is a ceiling not a fixed cost. 10/5 is sufficient for this manipulation task; go lower only for locomotion on flat ground.

### State reads — zero-copy DLPack views (85x faster than API)

The high-level getters (`robot.get_dofs_position()`, `entity.get_pos()`, etc.) call `qd_to_torch(..., copy=True)` internally, allocating a new tensor on every call. Replace with cached DLPack views set up once in `_post_build_init()`:

```python
from genesis.utils.misc import qd_to_torch

# All rigid entities share one solver
_solver = self.robot._solver

# Shapes after transpose: (n_envs, n_total_dofs) and (n_envs, n_total_links, 3/4)
# These are live views — scene.step() writes into the same memory automatically.
self._dof_pos_tc     = qd_to_torch(_solver.dofs_state.pos,   transpose=True, copy=False)
self._dof_vel_tc     = qd_to_torch(_solver.dofs_state.vel,   transpose=True, copy=False)
self._links_pos_tc   = qd_to_torch(_solver.links_state.pos,  transpose=True, copy=False)
self._links_quat_tc  = qd_to_torch(_solver.links_state.quat, transpose=True, copy=False)
self._links_cdvel_tc = qd_to_torch(_solver.links_state.cd_vel, transpose=True, copy=False)

# Global DOF indices (robot._dof_start + local_dof_idx)
self._action_global = torch.tensor(
    [self.robot._dof_start + d for d in self._action_dofs],
    dtype=torch.long, device=gs.device,
)
# Global link indices (plain ints)
self._tray_link_idx = self.tray.base_link_idx
self._ball_link_idx = self.ball.base_link_idx
```

Then in `_post_physics_step()`, replace all API reads with:

```python
_all_pos = self._dof_pos_tc[:, self._action_global]   # GPU gather, no alloc
_all_vel = self._dof_vel_tc[:, self._action_global]
self.arm_pos.copy_(_all_pos[:, :self.n_arm_dofs])
self.hand_pos.copy_(_all_pos[:, self.n_arm_dofs:])
self.arm_vel.copy_(_all_vel[:, :self.n_arm_dofs])
self.hand_vel.copy_(_all_vel[:, self.n_arm_dofs:])
self.tray_pos.copy_(self._links_pos_tc[:,  self._tray_link_idx])
self.tray_quat.copy_(self._links_quat_tc[:, self._tray_link_idx])
self.ball_pos.copy_(self._links_pos_tc[:,  self._ball_link_idx])
self.ball_vel.copy_(self._links_cdvel_tc[:, self._ball_link_idx])
```

**Key facts:**
- `gs.use_zerocopy=True` by default on both CUDA and CPU backends — no config needed.
- `qd_to_torch(..., copy=False)` caches the DLPack view as `field._T_tc` on first call. Subsequent calls return the cached tensor with no overhead.
- Quadrants stores fields as `(n_entities, n_envs)` internally; `transpose=True` gives PyTorch-convention `(n_envs, n_entities)`.
- DOF indices in the scene-level array are **not contiguous** for a multi-joint robot — use gather (`[:, index_tensor]`), not a plain slice.
- `links_state.cd_vel` equals `entity.get_vel()` exactly for free single-link rigid bodies (tray, ball) because their link origin coincides with the COM. Validated: max error = 0.0.
- The views are live — after `scene.step()` values update in-place. Never hold a view across steps as a snapshot; always `.copy_()` into pre-allocated buffers immediately.
- `copy=False` raises `GenesisException` if `gs.use_zerocopy=False` (only if `GS_ENABLE_ZEROCOPY=0` is set explicitly).

### Genesis benchmark methodology (important context)

Genesis's published speed numbers (including the "430,000x" claim) are measured as raw `scene.step()` throughput with `substeps=1`, `enable_self_collision=False`, simple 7–12 DOF robots, and no state reads, reward, or obs computation. They do not reflect RL training throughput. The genesis-speed-benchmark repo (`github.com/zhouxian/genesis-speed-benchmark`) has the exact scripts.

### Per-link collision disabling — not possible via Python API

Genesis `collision_mesh_prim_patterns` / `visual_mesh_prim_patterns` on the USD morph match against `prim.GetName()` (leaf name only, not full path). Because every link in the G1 USD has its collision Xform named `collisions`, you cannot distinguish leg links from arm links using these patterns. The only options are: rename the collision Xforms in the USD asset itself, or convert to MJCF and set `contype="0"` per body in the XML.

### USD prim structure (G1 flattened USD)

Every link follows: `/<root>/<link_name>/visuals` (visual mesh) and `/<root>/<link_name>/collisions` (collision mesh). Joints are under `/<root>/joints/<joint_name>`. Run `conda run -n env_isaaclab python scripts/print_usd_prims.py` to print the full prim tree.
