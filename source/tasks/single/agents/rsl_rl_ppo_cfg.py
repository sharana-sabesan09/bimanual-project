"""RSL-RL PPO config and VecEnv wrapper for the single-arm ball-balance task."""

import torch
from tensordict import TensorDict
import genesis as gs

from ..env import SingleArmBallBalanceEnv

NUM_OBS           = 23
NUM_ACTIONS       = 7
MAX_EPISODE_STEPS = 500


class BallBalanceVecEnv:
    """Thin wrapper adapting SingleArmBallBalanceEnv to the rsl-rl VecEnv interface."""

    def __init__(self, n_envs: int, show_viewer: bool = False, **env_kwargs):
        self._env = SingleArmBallBalanceEnv(show_viewer=show_viewer, n_envs=n_envs, **env_kwargs)
        self.num_envs          = n_envs
        self.num_actions       = NUM_ACTIONS
        self.max_episode_length = MAX_EPISODE_STEPS
        self.device            = gs.device
        self.cfg               = {}
        self.episode_length_buf = torch.zeros(n_envs, dtype=torch.int32, device=self.device)

    def get_observations(self) -> TensorDict:
        obs = self._env.get_obs().to(self.device)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:
        self.episode_length_buf.zero_()
        obs = self._env.reset().to(self.device)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple:
        obs, rewards, dones, _ = self._env.step(actions.cpu().numpy())
        obs     = obs.to(self.device)
        rewards = rewards.to(self.device)
        dones   = dones.to(self.device)

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones     = dones | time_outs

        reset_idx = torch.where(dones)[0]
        if len(reset_idx) > 0:
            self._env.reset(envs_idx=reset_idx)
            obs = self._env.get_obs().to(self.device)
        self.episode_length_buf[dones] = 0

        extras = {"time_outs": time_outs.float().to(self.device)}
        return TensorDict({"policy": obs}, batch_size=[self.num_envs]), rewards, dones, extras


def get_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 48,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
