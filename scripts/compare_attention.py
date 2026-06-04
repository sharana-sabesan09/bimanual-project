"""
Compare attention maps across multiple checkpoints (e.g. different ball masses).

After running collect_attention.py for each checkpoint:
    python scripts/compare_attention.py --data_dir attn_data --output_dir attn_plots

Produces:
    attn_plots/comparison_bars.png     — all runs side by side per relationship
    attn_plots/arm_focus.png           — for each run: how much each arm focuses on the other arm vs ball
    attn_plots/heatmap_<label>.png     — attention over time for each run
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError("Run: pip install matplotlib")

ATTN_LABELS = ["l_to_r", "l_to_b", "r_to_l", "r_to_b", "b_to_l", "b_to_r"]
ATTN_DESCRIPTIONS = {
    "l_to_r": "Left → Right",
    "l_to_b": "Left → Ball",
    "r_to_l": "Right → Left",
    "r_to_b": "Right → Ball",
    "b_to_l": "Ball → Left",
    "b_to_r": "Ball → Right",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_all(data_dir):
    runs = {}
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        label = str(d["label"])
        runs[label] = {name: d[name] for name in ATTN_LABELS if name in d}
        print(f"Loaded {label}")
    return runs


def plot_comparison_bars(runs, output_dir):
    """Side-by-side bar chart comparing all runs across the 6 attention relationships."""
    labels    = list(runs.keys())
    n_runs    = len(labels)
    n_rel     = len(ATTN_LABELS)
    x         = np.arange(n_rel)
    width     = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Cross-Attention Weights by Relationship", fontsize=14, fontweight="bold")

    for i, label in enumerate(labels):
        means  = [runs[label][name].mean() for name in ATTN_LABELS]
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label,
               color=COLORS[i % len(COLORS)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([ATTN_DESCRIPTIONS[n] for n in ATTN_LABELS], rotation=30, ha="right")
    ax.set_ylabel("Mean Attention Weight")
    ax.legend(title="Ball Mass")
    ax.grid(axis="y", alpha=0.3)

    # Shade arm-to-arm vs arm-to-ball groups
    ax.axvspan(-0.5, 1.5, alpha=0.05, color="blue",  label="_arm↔arm")
    ax.axvspan(1.5,  3.5, alpha=0.05, color="green", label="_arm↔ball")
    ax.axvspan(3.5,  5.5, alpha=0.05, color="orange",label="_ball↔arm")

    ax.text(0.5,  ax.get_ylim()[1] * 0.97, "arm ↔ arm",  ha="center", fontsize=8, color="blue")
    ax.text(2.5,  ax.get_ylim()[1] * 0.97, "arm ↔ ball", ha="center", fontsize=8, color="green")
    ax.text(4.5,  ax.get_ylim()[1] * 0.97, "ball ↔ arm", ha="center", fontsize=8, color="darkorange")

    plt.tight_layout()
    out = Path(output_dir) / "comparison_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_arm_focus(runs, output_dir):
    """
    For each run: stacked bar showing how each arm splits attention
    between the other arm vs the ball.
    """
    labels = list(runs.keys())
    n_runs = len(labels)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]

    fig.suptitle("Arm Attention: Other Arm vs Ball", fontsize=13, fontweight="bold")

    for ax, label in zip(axes, labels):
        data = runs[label]
        left_to_right = data["l_to_r"].mean()
        left_to_ball  = data["l_to_b"].mean()
        right_to_left = data["r_to_l"].mean()
        right_to_ball = data["r_to_b"].mean()

        arms  = ["Left Arm", "Right Arm"]
        to_partner = [left_to_right, right_to_left]
        to_ball    = [left_to_ball,  right_to_ball]

        x = np.arange(2)
        ax.bar(x, to_partner, label="→ Other arm", color="#4C72B0", alpha=0.85)
        ax.bar(x, to_ball,    bottom=to_partner,   label="→ Ball",       color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_title(label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean Attention Weight")
    plt.tight_layout()
    out = Path(output_dir) / "arm_focus.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_heatmaps(runs, output_dir):
    """Attention over time for each run."""
    for label, data in runs.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.suptitle(f"Attention Over Time — {label}", fontsize=13, fontweight="bold")

        arr = np.stack([data[name] for name in ATTN_LABELS], axis=1)  # (steps, 6)

        if arr.shape[0] > 200:
            arr = arr[::arr.shape[0] // 200]

        im = ax.imshow(arr.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(6))
        ax.set_yticklabels([ATTN_DESCRIPTIONS[n] for n in ATTN_LABELS])
        ax.set_xlabel("Timestep")
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        out = Path(output_dir) / f"heatmap_{label}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()


def print_summary(runs):
    print("\n=== Attention Summary ===")
    for label, data in runs.items():
        print(f"\n{label}:")
        for name in ATTN_LABELS:
            print(f"  {ATTN_DESCRIPTIONS[name]:25s}: {data[name].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="attn_data")
    parser.add_argument("--output_dir", type=str, default="attn_plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runs = load_all(args.data_dir)
    if not runs:
        print(f"No .npz files found in {args.data_dir}. Run collect_attention.py first.")
        return

    print_summary(runs)
    plot_comparison_bars(runs, args.output_dir)
    plot_arm_focus(runs, args.output_dir)
    plot_heatmaps(runs, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
