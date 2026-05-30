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
        
        # Calculate internal dimensions before calling super()
        self.token_dim = hidden_dims[0]
        self.right_dim, self.left_dim, self.ball_dim = 14, 14, 9

        # Initialize parent: use the rest of hidden_dims for the post-attention MLP head
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=hidden_dims[1:], 
                         activation=activation,
                         distribution_cfg=distribution_cfg,
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

        self.self_attn = CrossAttention(self.token_dim, num_heads)

        # Learned type embeddings to help attention distinguish tokens
        self.type_embed = nn.Parameter(torch.randn(1, 3, self.token_dim) * 0.02)

    def _get_latent_dim(self) -> int:
        """Informs the parent MLP how many features to expect from get_latent."""
        return 3 * self.token_dim

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> torch.Tensor:
        """Transforms raw obs into the fused attention vector used as MLP input."""
        # Parent handles group concatenation and normalization automatically
        x = super().get_latent(obs, masks, hidden_state)

        # Slice DualArmBallBalanceEnv observations (37 dims)
        right_obs = x[:, :14]
        left_obs = x[:, 14:28]
        ball_obs = x[:, 28:37]

        left = self.left_embed(left_obs).unsqueeze(1)
        right = self.right_embed(right_obs).unsqueeze(1)
        ball = self.ball_embed(ball_obs).unsqueeze(1)

        # Construct token sequence: [Left Arm, Right Arm, Ball/Goal]
        # Shape: (num_envs, 3, token_dim)
        tokens = torch.cat([left, right, ball], dim=1)

        # Add semantic identity to tokens
        tokens = tokens + self.type_embed

        # Self-attention allows each part to look at the others (e.g. Arm-to-Arm coordination)
        tokens_out = self.self_attn(tokens, tokens, tokens)

        return tokens_out.flatten(start_dim=1)

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

        self.self_attn = CrossAttention(self.token_dim, num_heads)

        # Learned type embeddings to help attention distinguish tokens
        self.type_embed = nn.Parameter(torch.randn(1, 3, self.token_dim) * 0.02)

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

        tokens = torch.cat([left, right, ball], dim=1)
        
        # Add semantic identity to tokens
        tokens = tokens + self.type_embed
        
        tokens_out = self.self_attn(tokens, tokens, tokens)

        return tokens_out.flatten(start_dim=1)