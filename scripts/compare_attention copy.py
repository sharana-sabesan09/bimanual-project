"""
Visualize attention influence from Felix's cross-attention architecture.

Usage:
    python scripts/compare_attention.py --data_dir attn_data --output_dir attn_plots

Produces:
    attn_plots/coordination_over_training.png  — arm-to-arm vs ball influence over iterations
    attn_plots/influence_breakdown.png         — stacked bar per iteration for each arm
    attn_plots/raw_influence.png               — raw L2 norms over training
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    raise ImportError("Run: pip install matplotlib")

ATTN_LABELS = ["l_to_r", "l_to_b", "r_to_l", "r_to_b", "b_to_l", "b_to_r"]
ATTN_DESCRIPTIONS = {
    "l_to_r": "Left → Right arm",
    "l_to_b": "Left → Ball",
    "r_to_l": "Right → Left arm",
    "r_to_b": "Right → Ball",
    "b_to_l": "Ball → Left arm",
    "b_to_r": "Ball → Right arm",
}

ARM_COLOR   = "#4C72B0"   # blue — arm-to-arm
BALL_COLOR  = "#DD8452"   # orange — arm-to-ball


def load_all(data_dir):
    """Load all .npz files, return sorted by iteration if labelled iter*."""
    runs = {}
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        label = str(d["label"])
        runs[label] = {name: float(d[name].mean()) for name in ATTN_LABELS if name in d}
    return runs


def extract_iter(label):
    """Extract iteration number from label like '0.1kg-iter200' → 200."""
    if "iter" in label:
        return int(label.split("iter")[-1])
    return None


def plot_coordination_over_training(runs, output_dir):
    """
    Line plot showing arm-to-arm influence % vs training iteration.
    One line for left arm, one for right arm.
    """
    # Filter to iteration-labelled runs and sort
    iter_runs = {k: v for k, v in runs.items() if extract_iter(k) is not None}
    if not iter_runs:
        print("No iteration-labelled runs found. Skipping coordination plot.")
        return

    iters = sorted(iter_runs.keys(), key=extract_iter)
    x = [extract_iter(k) for k in iters]

    left_arm_pct  = []
    right_arm_pct = []

    for k in iters:
        d = iter_runs[k]
        left_total  = d["l_to_r"] + d["l_to_b"]
        right_total = d["r_to_l"] + d["r_to_b"]
        left_arm_pct.append(d["l_to_r"] / left_total * 100)
        right_arm_pct.append(d["r_to_l"] / right_total * 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, left_arm_pct,  marker="o", color=ARM_COLOR,  linewidth=2.5,
            label="Left arm: influence from right arm")
    ax.plot(x, right_arm_pct, marker="s", color=BALL_COLOR, linewidth=2.5,
            label="Right arm: influence from left arm")

    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Arm-to-Arm Influence (%)", fontsize=12)
    ax.set_title("Cross-Arm Coordination Develops Over Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylim(0, 60)

    plt.tight_layout()
    out = Path(output_dir) / "coordination_over_training.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_influence_breakdown(runs, output_dir):
    """
    Stacked horizontal bar chart per iteration showing arm vs ball influence.
    """
    iter_runs = {k: v for k, v in runs.items() if extract_iter(k) is not None}
    if not iter_runs:
        # fall back to all runs
        iter_runs = runs

    iters = sorted(iter_runs.keys(), key=lambda k: extract_iter(k) or 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("What Each Arm Pays Attention To", fontsize=13, fontweight="bold")

    for ax, (arm, arm_key, other_key) in zip(axes, [
        ("Left Arm",  "l_to_r", "l_to_b"),
        ("Right Arm", "r_to_l", "r_to_b"),
    ]):
        labels = [k.replace("0.1kg-", "") for k in iters]
        arm_pct  = []
        ball_pct = []

        for k in iters:
            d = iter_runs[k]
            total = d[arm_key] + d[other_key]
            arm_pct.append(d[arm_key]  / total * 100)
            ball_pct.append(d[other_key] / total * 100)

        y = np.arange(len(labels))
        ax.barh(y, arm_pct,  color=ARM_COLOR,  alpha=0.85, label="Other arm")
        ax.barh(y, ball_pct, left=arm_pct, color=BALL_COLOR, alpha=0.85, label="Ball")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Influence (%)")
        ax.set_title(arm, fontsize=11, fontweight="bold")
        ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Add percentage labels
        for i, (a, b) in enumerate(zip(arm_pct, ball_pct)):
            ax.text(a / 2,           i, f"{a:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
            ax.text(a + b / 2,       i, f"{b:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "influence_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_raw_influence(runs, output_dir):
    """Raw L2 norm values for all 6 relationships over training."""
    iter_runs = {k: v for k, v in runs.items() if extract_iter(k) is not None}
    if not iter_runs:
        print("No iteration-labelled runs. Skipping raw influence plot.")
        return

    iters = sorted(iter_runs.keys(), key=extract_iter)
    x = [extract_iter(k) for k in iters]

    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#55A868", "#55A868"]
    styles = ["-", "--", "-", "--", "-", "--"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(ATTN_LABELS):
        y = [iter_runs[k][name] for k in iters]
        ax.plot(x, y, marker="o", color=colors[i], linestyle=styles[i],
                linewidth=2, label=ATTN_DESCRIPTIONS[name])

    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("L2 Norm of Attention Output", fontsize=12)
    ax.set_title("Raw Attention Influence Over Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "raw_influence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_summary(runs):
    print("\n=== Influence Summary ===")
    for label, d in sorted(runs.items(), key=lambda x: extract_iter(x[0]) or 0):
        left_total  = d["l_to_r"] + d["l_to_b"]
        right_total = d["r_to_l"] + d["r_to_b"]
        print(f"\n{label}:")
        print(f"  Left arm:  {d['l_to_r']/left_total*100:.1f}% arm, {d['l_to_b']/left_total*100:.1f}% ball")
        print(f"  Right arm: {d['r_to_l']/right_total*100:.1f}% arm, {d['r_to_b']/right_total*100:.1f}% ball")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="attn_data")
    parser.add_argument("--output_dir", type=str, default="attn_plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runs = load_all(args.data_dir)
    if not runs:
        print(f"No .npz files found in {args.data_dir}.")
        return

    print_summary(runs)
    plot_coordination_over_training(runs, args.output_dir)
    plot_influence_breakdown(runs, args.output_dir)
    plot_raw_influence(runs, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
