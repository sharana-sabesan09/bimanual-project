import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from RL_lib.crossattention import CrossAttention

class AttentionActor(MLPModel):
    def __init__(self,
                 obs,
                 obs_groups,
                 obs_set,
                 output_dim,
                 hidden_dims=(256, 128),
                 num_heads=4,
                 activation="elu",
                 distribution_cfg=None,
                 use_arm_comm=True,
                 **kwargs):

        self.token_dim = hidden_dims[0]
        self.right_dim, self.left_dim, self.ball_dim = 14, 14, 9
        self.use_arm_comm = use_arm_comm

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims[1:],
            activation=activation,
            distribution_cfg=distribution_cfg,
            **kwargs
        )

        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()

        self.left_embed = nn.Sequential(
            nn.Linear(self.left_dim, self.token_dim),
            act_fn
        )

        self.right_embed = nn.Sequential(
            nn.Linear(self.right_dim, self.token_dim),
            act_fn
        )

        self.ball_embed = nn.Sequential(
            nn.Linear(self.ball_dim, self.token_dim),
            act_fn
        )

        self.left_to_ball = CrossAttention(self.token_dim, num_heads)
        self.right_to_ball = CrossAttention(self.token_dim, num_heads)

        if self.use_arm_comm:
            self.left_to_right = CrossAttention(self.token_dim, num_heads)
            self.right_to_left = CrossAttention(self.token_dim, num_heads)

            self.comm_gate = nn.Sequential(
                nn.Linear(2 * self.token_dim, self.token_dim),
                act_fn,
                nn.Linear(self.token_dim, 1)
            )

    def _get_latent_dim(self) -> int:
        return 3 * self.token_dim

    def get_latent(self,
                   obs: TensorDict,
                   masks: torch.Tensor | None = None,
                   hidden_state: torch.Tensor | None = None) -> torch.Tensor:

        x = super().get_latent(obs, masks, hidden_state)

        right_obs = x[:, :14]
        left_obs = x[:, 14:28]
        ball_obs = x[:, 28:37]

        left = self.left_embed(left_obs).unsqueeze(1)
        right = self.right_embed(right_obs).unsqueeze(1)
        ball = self.ball_embed(ball_obs).unsqueeze(1)

        left_ball = self.left_to_ball(left, ball, ball)
        right_ball = self.right_to_ball(right, ball, ball)

        left_final = left_ball
        right_final = right_ball

        if self.use_arm_comm:
            left_from_right = self.left_to_right(left, right, right)
            right_from_left = self.right_to_left(right, left, left)

            gate_input = torch.cat([left.squeeze(1), right.squeeze(1)], dim=-1)
            gate = torch.sigmoid(self.comm_gate(gate_input))

            left_final = left_final + gate.unsqueeze(-1) * left_from_right
            right_final = right_final + gate.unsqueeze(-1) * right_from_left

        return torch.cat([
            left_final.squeeze(1),
            right_final.squeeze(1),
            ball.squeeze(1)
        ], dim=-1)

class AttentionCritic(MLPModel):
    def __init__(self,
                 obs,
                 obs_groups,
                 obs_set,
                 output_dim,
                 hidden_dims=(512, 256, 128),
                 num_heads=4,
                 activation="elu",
                 **kwargs):
        
        self.token_dim = hidden_dims[0]
        self.right_dim, self.left_dim, self.ball_dim = 14, 14, 9

        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=hidden_dims[1:],
                         activation=activation,
                         **kwargs)

        self.left_embed = nn.Sequential(
            nn.Linear(self.left_dim, self.token_dim),
            nn.ELU() if activation == "elu" else nn.ReLU()
        )
        self.right_embed = nn.Sequential(
            nn.Linear(self.right_dim, self.token_dim),
            nn.ELU() if activation == "elu" else nn.ReLU()
        )
        self.ball_embed = nn.Sequential(
            nn.Linear(self.ball_dim, self.token_dim),
            nn.ELU() if activation == "elu" else nn.ReLU()
        )

        self.cross_left = CrossAttention(self.token_dim, num_heads)
        self.cross_right = CrossAttention(self.token_dim, num_heads)

    def _get_latent_dim(self) -> int:
        return 3 * self.token_dim

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> torch.Tensor:
        x = super().get_latent(obs, masks, hidden_state)

        right_obs = x[:, :14]
        left_obs = x[:, 14:28]
        ball_obs = x[:, 28:37]

        left = self.left_embed(left_obs).unsqueeze(1)
        right = self.right_embed(right_obs).unsqueeze(1)
        ball = self.ball_embed(ball_obs).unsqueeze(1)

        left_out = self.cross_left(left, ball, ball)
        right_out = self.cross_right(right, ball, ball)

        return torch.cat([
            left_out.squeeze(1),
            right_out.squeeze(1),
            ball.squeeze(1)
        ], dim=-1)