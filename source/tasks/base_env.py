"""
Generic vectorised Genesis gym env base class.
Subclasses provide all domain logic — this file knows nothing about balls, robots, or rewards.
"""

import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from abc import abstractmethod


class BaseVecEnv(gym.Env):

    def __init__(self, show_viewer: bool = True, n_envs: int = 1, max_episode_steps: int = 500):
        super().__init__()
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, -2.5, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=45,
                max_FPS=60,
            ),
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=0.02),
        )

        self._build_scene(n_envs)      # add entities + scene.build()
        self._post_build_init()        # cache indices, set n_arm_dofs, init spaces + buffers

        self.episode_length_buf = torch.zeros(self.n_envs, dtype=torch.int32, device=gs.device)
        self.terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=gs.device)
        self.truncated  = torch.zeros(self.n_envs, dtype=torch.bool, device=gs.device)
        self.extras = {"log": {}}

        self.reset()

    # ------------------------------------------------------------------ #
    # Abstract — subclass must implement                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_scene(self, n_envs: int):
        """Add all entities and call scene.build(n_envs=n_envs, ...)."""

    @abstractmethod
    def _post_build_init(self):
        """Post-build setup: cache DOF indices, set n_arm_dofs, init observation_space /
        action_space, and any subclass-specific buffers (e.g. prev_actions)."""

    @abstractmethod
    def _reset_env(self, envs_idx=None):
        """Reset robot joints, objects, and any subclass buffers for the given env indices
        (or all envs if None)."""

    @abstractmethod
    def _apply_action(self, action_tensor: torch.Tensor):
        """Apply a (n_envs, n_arm_dofs) action tensor to the simulation."""

    @abstractmethod
    def get_obs(self) -> torch.Tensor:
        """Assemble and return observation tensor of shape (n_envs, obs_dim) from cached state."""

    @abstractmethod
    def get_termination(self):
        """Compute termination signals from cached state.
        Must set self.terminated and self.truncated and return (terminated, truncated)."""

    @abstractmethod
    def _compute_reward(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-env reward from cached state. Called after get_termination so
        self.terminated / self.truncated are available for shaping."""

    # ------------------------------------------------------------------ #
    # Concrete                                                             #
    # ------------------------------------------------------------------ #

    def _post_physics_step(self):
        """Update all cached physics state after each scene.step(). Override in subclass."""

    def reset(self, envs_idx=None, seed=None, options=None):
        self._reset_env(envs_idx)
        if envs_idx is None:
            self.episode_length_buf.zero_()
        else:
            self.episode_length_buf[envs_idx] = 0
        self.scene.step()
        self._post_physics_step()
        return self.get_obs(), {}

    def step(self, action: torch.Tensor | np.ndarray):
        action_tensor = action
        if isinstance(action, np.ndarray):
            action_tensor = torch.tensor(action, device=gs.device, dtype=torch.float32)
        self._apply_action(action_tensor)
        self.scene.step()
        self._post_physics_step()

        self.episode_length_buf += 1
        obs = self.get_obs()

        self.get_termination()
        reward = self._compute_reward(action_tensor)

        terminated = self.terminated.clone()
        truncated  = self.truncated.clone()

        done_idx = torch.where(terminated | truncated)[0]
        if len(done_idx) > 0:
            obs, _ = self.reset(envs_idx=done_idx)

        return obs, reward, terminated, truncated, self.extras

    def render(self):
        pass

    def close(self):
        pass
