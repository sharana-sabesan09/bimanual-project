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

        n = self.scene.n_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.int32, device=gs.device)
        self.terminated = torch.zeros(n, dtype=torch.bool, device=gs.device)
        self.truncated  = torch.zeros(n, dtype=torch.bool, device=gs.device)

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
        """Return observation tensor of shape (n_envs, obs_dim)."""

    @abstractmethod
    def get_termination(self, obs: torch.Tensor):
        """Compute termination signals from obs.
        Must set self.terminated and self.truncated and return (terminated, truncated)."""

    @abstractmethod
    def _compute_reward(self, obs: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-env reward. Called after get_termination so self.terminated /
        self.truncated are available for shaping."""

    # ------------------------------------------------------------------ #
    # Concrete                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, envs_idx=None, seed=None, options=None):
        self._reset_env(envs_idx)
        if envs_idx is None:
            self.episode_length_buf.zero_()
        else:
            self.episode_length_buf[envs_idx] = 0
        self.scene.step()
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        action_tensor = torch.tensor(action, device=gs.device, dtype=torch.float32)
        self._apply_action(action_tensor)
        self.scene.step()

        self.episode_length_buf += 1
        obs = self.get_obs()

        self.get_termination(obs)
        reward = self._compute_reward(obs, action_tensor)

        terminated = self.terminated.clone()
        truncated  = self.truncated.clone()

        done_idx = torch.where(terminated | truncated)[0]
        if len(done_idx) > 0:
            self._reset_env(envs_idx=done_idx)
            self.episode_length_buf[done_idx] = 0
            self.scene.step()
            obs = self.get_obs()

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
