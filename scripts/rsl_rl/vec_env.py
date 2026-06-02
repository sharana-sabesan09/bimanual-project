import torch
from tensordict import TensorDict
import genesis as gs


class RslRlVecEnvWrapper:
    """Adapts a BaseVecEnv to the rsl-rl OnPolicyRunner interface."""

    def __init__(self, env):
        self._env = env
        self.num_envs         = env.n_envs
        self.num_actions      = getattr(env, "n_action_dofs", env.n_arm_dofs)
        self.max_episode_length = env.max_episode_steps
        self.device           = gs.device
        self.cfg              = {}

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        obs = self._env.get_obs().to(self.device)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:
        obs, _ = self._env.reset()
        return TensorDict({"policy": obs.to(self.device)}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple:
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        extras = {"time_outs": truncated.float().to(self.device)}
        if "log" in info:
            extras["log"] = info["log"]
        return (
            TensorDict({"policy": obs.to(self.device)}, batch_size=[self.num_envs]),
            rewards.to(self.device),
            (terminated | truncated).to(self.device),
            extras,
        )
