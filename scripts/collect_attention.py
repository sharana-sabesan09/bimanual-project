"""
Collect attention weights from Felix's cross-attention-between-arms architecture.

The 6 attention relationships captured:
    l_to_r  — left arm  → right arm   (how much left cares about right)
    l_to_b  — left arm  → ball
    r_to_l  — right arm → left arm    (how much right cares about left)
    r_to_b  — right arm → ball
    b_to_l  — ball      → left arm
    b_to_r  — ball      → right arm

Each is a single scalar per step (1 query token × 1 key token, averaged over heads).

Usage:
    python scripts/collect_attention.py --checkpoint logs/.../model_500.pt --label 0.1kg
    python scripts/collect_attention.py --checkpoint logs/.../model_500.pt --label 0.3kg
    python scripts/collect_attention.py --checkpoint logs/.../model_500.pt --label 0.5kg

Output: attn_data/<label>.npz
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Keep scripts/rsl_rl/ from shadowing the installed rsl_rl package
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir in sys.path:
    sys.path.remove(_scripts_dir)
sys.path.insert(0, os.path.dirname(_scripts_dir))

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ #
# Observation constants                                                #
# ------------------------------------------------------------------ #
RIGHT_ARM_HOLD = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD  = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
TRAY_Z         = 1.5
TRAY_W, TRAY_H = 0.36, 0.26

# The 6 attention relationships in Felix's architecture
ATTN_LABELS = ["l_to_r", "l_to_b", "r_to_l", "r_to_b", "b_to_l", "b_to_r"]
ATTN_DESCRIPTIONS = {
    "l_to_r": "Left arm → Right arm",
    "l_to_b": "Left arm → Ball",
    "r_to_l": "Right arm → Left arm",
    "r_to_b": "Right arm → Ball",
    "b_to_l": "Ball → Left arm",
    "b_to_r": "Ball → Right arm",
}


# ------------------------------------------------------------------ #
# Minimal self-contained model matching Felix's architecture           #
# ------------------------------------------------------------------ #

class FelixAttentionExtractor(nn.Module):
    """
    Replicates only the embedding + attention layers from Felix's AttentionActor.
    Uses hooks to capture attention weights since CrossAttention uses need_weights=False.
    """
    def __init__(self, token_dim: int, num_heads: int):
        super().__init__()
        self.token_dim = token_dim
        act = nn.ELU()

        # 2-layer embeddings matching Felix's architecture
        self.left_embed  = nn.Sequential(nn.Linear(14, token_dim), act, nn.Linear(token_dim, token_dim))
        self.right_embed = nn.Sequential(nn.Linear(14, token_dim), act, nn.Linear(token_dim, token_dim))
        self.ball_embed  = nn.Sequential(nn.Linear(9,  token_dim), act, nn.Linear(token_dim, token_dim))

        # 6 attention modules — plain nn.MultiheadAttention (no CrossAttention wrapper)
        def make_attn():
            return nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.l_to_r = make_attn()
        self.r_to_l = make_attn()
        self.l_to_b = make_attn()
        self.r_to_b = make_attn()
        self.b_to_l = make_attn()
        self.b_to_r = make_attn()

    def forward(self, obs: torch.Tensor):
        """
        obs: (B, 37)
        Returns dict of attention weights, each shape (B, heads, 1, 1) → scalar per head
        """
        left_obs  = obs[:, :14]
        right_obs = obs[:, 14:28]
        ball_obs  = obs[:, 28:37]

        left  = self.left_embed(left_obs).unsqueeze(1)   # (B, 1, D)
        right = self.right_embed(right_obs).unsqueeze(1) # (B, 1, D)
        ball  = self.ball_embed(ball_obs).unsqueeze(1)   # (B, 1, D)

        weights = {}
        for name, q, k in [
            ("l_to_r", left,  right),
            ("l_to_b", left,  ball),
            ("r_to_l", right, left),
            ("r_to_b", right, ball),
            ("b_to_l", ball,  left),
            ("b_to_r", ball,  right),
        ]:
            attn_mod = getattr(self, name)
            _, w = attn_mod(q, k, k, need_weights=True, average_attn_weights=False)
            # w shape: (B, heads, 1, 1) — 1 query token attending to 1 key token
            weights[name] = w.squeeze(-1).squeeze(-1).mean(dim=-1)  # (B,) — mean over heads

        return weights  # dict of 6 tensors, each (B,)


# ------------------------------------------------------------------ #
# Checkpoint loading                                                   #
# ------------------------------------------------------------------ #

def load_model(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Extract actor weights
    if "actor_state_dict" in state:
        actor_state = state["actor_state_dict"]
    elif "model_state_dict" in state:
        full = state["model_state_dict"]
        actor_state = {k[len("actor."):]: v for k, v in full.items() if k.startswith("actor.")}
    else:
        actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}

    if not actor_state:
        raise KeyError("Could not find actor weights in checkpoint.")

    # Verify this is Felix's architecture by checking for l_to_r keys
    if not any("l_to_r" in k for k in actor_state):
        raise ValueError(
            "This checkpoint does not appear to be from Felix's cross-attention architecture "
            "(no 'l_to_r' keys found). Make sure you trained on the cross-attention-between-arms branch."
        )

    # Detect token_dim from left_embed
    token_dim = actor_state["left_embed.0.weight"].shape[0]
    num_heads  = 4  # fixed in all configs

    print(f"  Detected: token_dim={token_dim}, num_heads={num_heads}")
    print(f"  Architecture: Felix's bidirectional cross-attention (6 attention modules)")

    model = FelixAttentionExtractor(token_dim, num_heads)

    # Map checkpoint keys to model keys
    # Checkpoint: "l_to_r.attn.in_proj_weight" → model: "l_to_r.in_proj_weight"
    remap = {}
    for k, v in actor_state.items():
        new_k = k
        for name in ATTN_LABELS:
            new_k = new_k.replace(f"{name}.attn.", f"{name}.")
        # Drop norm/dropout/head/embed_extra keys not in our minimal model
        if any(x in new_k for x in [".norm.", ".dropout.", "norm_l", "norm_r", "norm_b", "head.", "mlp.", "obs_norm", "distribution"]):
            continue
        remap[new_k] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (e.g. {missing[0]})")

    model.eval()
    return model


# ------------------------------------------------------------------ #
# Observation generator                                                #
# ------------------------------------------------------------------ #

def make_obs(batch_size: int, noise: float = 0.05) -> torch.Tensor:
    B = batch_size
    r_pos = torch.tensor(RIGHT_ARM_HOLD).unsqueeze(0).repeat(B, 1) + torch.randn(B, 7) * noise
    r_vel = torch.randn(B, 7) * noise * 0.5
    l_pos = torch.tensor(LEFT_ARM_HOLD).unsqueeze(0).repeat(B, 1)  + torch.randn(B, 7) * noise
    l_vel = torch.randn(B, 7) * noise * 0.5
    bx = (torch.rand(B, 1) - 0.5) * TRAY_W
    by = (torch.rand(B, 1) - 0.5) * TRAY_H
    bz = torch.full((B, 1), TRAY_Z + 0.02)
    ball_pos = torch.cat([bx, by, bz], dim=-1)
    ball_vel = torch.randn(B, 3) * 0.3
    gx = (torch.rand(B, 1) - 0.5) * TRAY_W * 0.7
    gy = (torch.rand(B, 1) - 0.5) * TRAY_H * 0.7
    gz = torch.full((B, 1), TRAY_Z + 0.02)
    goal_pos = torch.cat([gx, gy, gz], dim=-1)
    return torch.cat([r_pos, r_vel, l_pos, l_vel, ball_pos, ball_vel, goal_pos], dim=-1)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label",      type=str,  required=True, help="e.g. '0.1kg'")
    parser.add_argument("--steps",      type=int,  default=500)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--output_dir", type=str,  default="attn_data")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading {checkpoint}...")
    model = load_model(checkpoint)
    print("Model ready.")

    # Accumulate per-relationship weights: dict of lists
    accum = {name: [] for name in ATTN_LABELS}

    print(f"Running {args.steps} forward passes (batch={args.batch_size})...")
    with torch.no_grad():
        for step in range(args.steps):
            obs = make_obs(args.batch_size)
            weights = model(obs)   # dict of (B,) tensors
            for name in ATTN_LABELS:
                accum[name].append(weights[name].mean().item())  # scalar per step

            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}")

    # Stack into arrays: (steps,) per relationship
    arrays = {name: np.array(accum[name]) for name in ATTN_LABELS}

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.npz"
    np.savez(out_path, label=args.label, **arrays)

    print(f"\nSaved to {out_path}")
    print("\n=== Mean Attention Weights ===")
    for name in ATTN_LABELS:
        desc = ATTN_DESCRIPTIONS[name]
        mean = arrays[name].mean()
        print(f"  {desc:25s}: {mean:.4f}")


if __name__ == "__main__":
    main()
