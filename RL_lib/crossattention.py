# uncomment this one when evaluating felix's trainings
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # IMPORTANT for (B, T, D)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            attn_out, _ = self.attn(query, key, value, need_weights=False)
        return self.norm(query + self.dropout(attn_out))
    
# uncomment this one otherwise

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.1):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scale = math.sqrt(self.head_dim)

#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.attn_dropout = dropout

#     def forward(self, query, key, value):
#         B, Tq, _ = query.shape
#         Tk = key.shape[1]
#         H, D = self.num_heads, self.head_dim

#         q = self.q_proj(query).view(B, Tq, H, D).transpose(1, 2)  # (B, H, Tq, D)
#         k = self.k_proj(key).view(B, Tk, H, D).transpose(1, 2)
#         v = self.v_proj(value).view(B, Tk, H, D).transpose(1, 2)

#         # vanilla scaled dot-product — no flash-attention kernel, no batch-size limit
#         attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, Tq, Tk)
#         attn_weights = F.softmax(attn_weights, dim=-1)
#         if self.training and self.attn_dropout > 0:
#             attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
#         attn_out = torch.matmul(attn_weights, v)  # (B, H, Tq, D)

#         attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, H * D)
#         attn_out = self.out_proj(attn_out)
#         return self.norm(query + self.dropout(attn_out))