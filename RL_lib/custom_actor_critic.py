"""
Custom actor/critic architectures for rsl-rl-lib >= 5.

Use dotted module paths in train_cfg to reference these without touching site-packages:
    "actor": {"class_name": "RL_lib.custom_actor_critic.LSTMActor", ...}
    "critic": {"class_name": "RL_lib.custom_actor_critic.DeepMLPCritic", ...}
"""

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.rnn_model import RNNModel


class LSTMActor(RNNModel):
    """LSTM-based actor. Thin wrapper around RNNModel that fixes rnn_type='lstm'
    and exposes LSTM-specific kwargs (hidden_size, num_layers)."""

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_size: int = 256,
                 num_layers: int = 1,
                 hidden_dims: tuple = (256, 128),
                 activation: str = "elu",
                 distribution_cfg: dict | None = None,
                 **kwargs):
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            distribution_cfg=distribution_cfg,
            rnn_type="lstm",
            rnn_hidden_dim=hidden_size,
            rnn_num_layers=num_layers,
            **kwargs,
        )


class DeepMLPCritic(MLPModel):
    """Deeper MLP critic (4 layers vs the default 3).
    Useful when the value function needs more capacity than the actor."""

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims: tuple = (512, 512, 256, 128),
                 activation: str = "elu",
                 **kwargs):
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            **kwargs,
        )
