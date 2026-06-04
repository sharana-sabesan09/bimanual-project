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
                 **kwargs):

        self.token_dim = hidden_dims[0]

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

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        self.left_dim, self.right_dim, self.ball_dim = 14, 14, 9

        self.left_embed = nn.Sequential(
            nn.Linear(self.left_dim, self.token_dim),
            act,
            nn.Linear(self.token_dim, self.token_dim)
        )

        self.right_embed = nn.Sequential(
            nn.Linear(self.right_dim, self.token_dim),
            act,
            nn.Linear(self.token_dim, self.token_dim)
        )

        self.ball_embed = nn.Sequential(
            nn.Linear(self.ball_dim, self.token_dim),
            act,
            nn.Linear(self.token_dim, self.token_dim)
        )

        self.l_to_r = CrossAttention(self.token_dim, num_heads)
        self.r_to_l = CrossAttention(self.token_dim, num_heads)

        self.l_to_b = CrossAttention(self.token_dim, num_heads)
        self.r_to_b = CrossAttention(self.token_dim, num_heads)

        self.b_to_l = CrossAttention(self.token_dim, num_heads)
        self.b_to_r = CrossAttention(self.token_dim, num_heads)

        self.norm_l = nn.LayerNorm(self.token_dim)
        self.norm_r = nn.LayerNorm(self.token_dim)
        self.norm_b = nn.LayerNorm(self.token_dim)

        self.head = nn.Sequential(
            nn.Linear(3 * self.token_dim, 256),
            act,
            nn.Linear(256, output_dim)
        )

    def _tok(self, x):
        return x.unsqueeze(1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        x = super().get_latent(obs, masks, hidden_state)

        left_obs = x[:, :self.left_dim]
        right_obs = x[:, self.left_dim:self.left_dim + self.right_dim]
        ball_obs = x[:, self.left_dim + self.right_dim:
                        self.left_dim + self.right_dim + self.ball_dim]

        left = self._tok(self.left_embed(left_obs))
        right = self._tok(self.right_embed(right_obs))
        ball = self._tok(self.ball_embed(ball_obs))

        left = left + self.l_to_r(left, right, right) + self.l_to_b(left, ball, ball)
        right = right + self.r_to_l(right, left, left) + self.r_to_b(right, ball, ball)
        ball = ball + self.b_to_l(ball, left, left) + self.b_to_r(ball, right, right)

        left = self.norm_l(left)
        right = self.norm_r(right)
        ball = self.norm_b(ball)

        z = torch.cat([
            left.squeeze(1),
            right.squeeze(1),
            ball.squeeze(1)
        ], dim=-1)

        return self.head(z)

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