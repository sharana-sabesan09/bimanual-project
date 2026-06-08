"""
Collect attention INFLUENCE from Felix's cross-attention architecture.

Since each entity is a single token, attention weights are always 1.0.
Instead we measure the L2 norm of each cross-attention output vector,
which tells us how much each source (ball/other arm) influences each entity.

Normalized within each entity so results are interpretable as percentages:
- left arm: what fraction of its update comes from right arm vs ball?
- right arm: what fraction comes from left arm vs ball?
- ball: what fraction comes from left arm vs right arm?

Usage:
    python scripts/collect_attention.py --checkpoint logs/.../model_499.pt --label 0.1kg

Output: attn_data/<label>.npz
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir in sys.path:
    sys.path.remove(_scripts_dir)
sys.path.insert(0, os.path.dirname(_scripts_dir))

import numpy as np
import torch
import torch.nn as nn

RIGHT_ARM_HOLD = np.array([-0.7, -0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
LEFT_ARM_HOLD  = np.array([-0.7,  0.2, 0.0, 1.2, 0.0, -0.6, 0.0], dtype=np.float32)
TRAY_Z         = 1.5
TRAY_W, TRAY_H = 0.36, 0.26

ATTN_LABELS = ["l_to_r", "l_to_b", "r_to_l", "r_to_b", "b_to_l", "b_to_r"]
ATTN_DESCRIPTIONS = {
    "l_to_r": "Left arm → Right arm",
    "l_to_b": "Left arm → Ball",
    "r_to_l": "Right arm → Left arm",
    "r_to_b": "Right arm → Ball",
    "b_to_l": "Ball → Left arm",
    "b_to_r": "Ball → Right arm",
}


class FelixInfluenceExtractor(nn.Module):
    """Measures the L2 norm of each cross-attention output — a proxy for influence."""
    def __init__(self, token_dim: int, num_heads: int):
        super().__init__()
        self.token_dim = token_dim
        act = nn.ELU()

        self.left_embed  = nn.Sequential(nn.Linear(14, token_dim), act, nn.Linear(token_dim, token_dim))
        self.right_embed = nn.Sequential(nn.Linear(14, token_dim), act, nn.Linear(token_dim, token_dim))
        self.ball_embed  = nn.Sequential(nn.Linear(9,  token_dim), act, nn.Linear(token_dim, token_dim))

        def make_attn():
            return nn.MultiheadAttention(token_dim, num_heads, batch_first=True, dropout=0.0)

        self.l_to_r = make_attn()
        self.r_to_l = make_attn()
        self.l_to_b = make_attn()
        self.r_to_b = make_attn()
        self.b_to_l = make_attn()
        self.b_to_r = make_attn()

    def forward(self, obs: torch.Tensor):
        left_obs  = obs[:, :14]
        right_obs = obs[:, 14:28]
        ball_obs  = obs[:, 28:37]

        left  = self.left_embed(left_obs).unsqueeze(1)
        right = self.right_embed(right_obs).unsqueeze(1)
        ball  = self.ball_embed(ball_obs).unsqueeze(1)

        influences = {}
        for name, q, k in [
            ("l_to_r", left,  right),
            ("l_to_b", left,  ball),
            ("r_to_l", right, left),
            ("r_to_b", right, ball),
            ("b_to_l", ball,  left),
            ("b_to_r", ball,  right),
        ]:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                attn_out, _ = getattr(self, name)(q, k, k, need_weights=False)
            influences[name] = attn_out.squeeze(1).norm(dim=-1).mean().item()

        return influences


def load_model(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "actor_state_dict" in state:
        actor_state = state["actor_state_dict"]
    elif "model_state_dict" in state:
        full = state["model_state_dict"]
        actor_state = {k[len("actor."):]: v for k, v in full.items() if k.startswith("actor.")}
    else:
        actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}

    if not actor_state:
        raise KeyError("Could not find actor weights in checkpoint.")

    if not any("l_to_r" in k for k in actor_state):
        raise ValueError("Not Felix's architecture — no 'l_to_r' keys found.")

    token_dim = actor_state["left_embed.0.weight"].shape[0]
    num_heads  = 4

    print(f"  Detected: token_dim={token_dim}, num_heads={num_heads}")

    model = FelixInfluenceExtractor(token_dim, num_heads)

    remap = {}
    for k, v in actor_state.items():
        new_k = k
        for name in ATTN_LABELS:
            new_k = new_k.replace(f"{name}.attn.", f"{name}.")
        if any(x in new_k for x in [".norm.", ".dropout.", "norm_l", "norm_r", "norm_b",
                                      "head.", "mlp.", "obs_norm", "distribution"]):
            continue
        remap[new_k] = v

    missing, _ = model.load_state_dict(remap, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")

    model.eval()
    return model


def make_obs(batch_size: int, noise: float = 0.05) -> torch.Tensor:
    B = batch_size
    r_pos = torch.tensor(RIGHT_ARM_HOLD).unsqueeze(0).repeat(B, 1) + torch.randn(B, 7) * noise
    r_vel = torch.randn(B, 7) * noise * 0.5
    l_pos = torch.tensor(LEFT_ARM_HOLD).unsqueeze(0).repeat(B, 1) + torch.randn(B, 7) * noise
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label",      type=str,  required=True)
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

    accum = {name: [] for name in ATTN_LABELS}

    print(f"Running {args.steps} forward passes (batch={args.batch_size})...")
    with torch.no_grad():
        for step in range(args.steps):
            obs = make_obs(args.batch_size)
            influences = model(obs)
            for name in ATTN_LABELS:
                accum[name].append(influences[name])
            if (step + 1) % 100 == 0:
                print(f"  step {step + 1}/{args.steps}")

    arrays = {name: np.array(accum[name]) for name in ATTN_LABELS}

    left_total  = arrays["l_to_r"].mean() + arrays["l_to_b"].mean()
    right_total = arrays["r_to_l"].mean() + arrays["r_to_b"].mean()
    ball_total  = arrays["b_to_l"].mean() + arrays["b_to_r"].mean()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.npz"
    np.savez(out_path, label=args.label, **arrays)

    print(f"\nSaved to {out_path}")
    print("\n=== Attention Influence (L2 norm of cross-attn output) ===")
    for name in ATTN_LABELS:
        print(f"  {ATTN_DESCRIPTIONS[name]:25s}: {arrays[name].mean():.4f}")

    print("\n=== Normalized Influence per Entity ===")
    print(f"  Left arm:  {arrays['l_to_r'].mean()/left_total*100:.1f}% from right arm, "
          f"{arrays['l_to_b'].mean()/left_total*100:.1f}% from ball")
    print(f"  Right arm: {arrays['r_to_l'].mean()/right_total*100:.1f}% from left arm, "
          f"{arrays['r_to_b'].mean()/right_total*100:.1f}% from ball")
    print(f"  Ball:      {arrays['b_to_l'].mean()/ball_total*100:.1f}% from left arm, "
          f"{arrays['b_to_r'].mean()/ball_total*100:.1f}% from right arm")


if __name__ == "__main__":
    main()
