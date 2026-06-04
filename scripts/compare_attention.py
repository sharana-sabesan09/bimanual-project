"""
Compare attention maps across multiple checkpoints (e.g. different ball masses).

After running collect_attention.py for each checkpoint:
    python scripts/compare_attention.py --data_dir attn_data --output_dir attn_plots

Produces:
    attn_plots/comparison_bars.png   — mean attention per ball token, all runs side by side
    attn_plots/heatmap_<label>.png   — attention over time for each run
    attn_plots/left_vs_right.png     — left arm vs right arm per run
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError("Run: pip install matplotlib")

BALL_TOKEN_LABELS = [
    "ball_x", "ball_y", "ball_z",
    "vel_x",  "vel_y",  "vel_z",
    "goal_x", "goal_y", "goal_z",
]

TOKEN_GROUPS = {
    "Ball Position": [0, 1, 2],
    "Ball Velocity": [3, 4, 5],
    "Goal Position": [6, 7, 8],
}

COLORS = ["#4C72B0", "#DD8452", "#55A868"]  # blue, orange, green


def load_all(data_dir):
    runs = {}
    for path in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        label = str(d["label"])
        runs[label] = {
            "left":  d["left"],   # (steps, 9)
            "right": d["right"],  # (steps, 9)
        }
        print(f"Loaded {label}: {d['left'].shape[0]} steps")
    return runs


def plot_comparison_bars(runs, output_dir):
    """Side-by-side bar chart: mean attention per ball token for all runs."""
    labels = list(runs.keys())
    n_runs = len(labels)
    n_tokens = 9
    x = np.arange(n_tokens)
    width = 0.8 / n_runs

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle("Mean Cross-Attention Weights by Ball Token", fontsize=14, fontweight="bold")

    for arm_idx, arm in enumerate(["left", "right"]):
        ax = axes[arm_idx]
        for i, (label, data) in enumerate(runs.items()):
            means = data[arm].mean(axis=0)  # (9,)
            offset = (i - n_runs / 2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, label=label,
                          color=COLORS[i % len(COLORS)], alpha=0.85)

        # shade token groups
        group_colors = ["#f0f0f0", "#e0e8f0", "#e8f0e0"]
        for g_idx, (gname, indices) in enumerate(TOKEN_GROUPS.items()):
            ax.axvspan(indices[0] - 0.5, indices[-1] + 0.5,
                       alpha=0.15, color=group_colors[g_idx], label=f"_{gname}")
            ax.text(np.mean(indices), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.15,
                    gname, ha="center", va="bottom", fontsize=8, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(BALL_TOKEN_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title(f"{arm.capitalize()} Arm")
        ax.legend(title="Ball Mass")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "comparison_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_heatmaps(runs, output_dir):
    """One heatmap per run: timestep (rows) x ball token (cols), for each arm."""
    for label, data in runs.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Attention Over Time — {label}", fontsize=13, fontweight="bold")

        for arm_idx, arm in enumerate(["left", "right"]):
            ax = axes[arm_idx]
            arr = data[arm]  # (steps, 9)

            # downsample rows if too many
            if arr.shape[0] > 200:
                step = arr.shape[0] // 200
                arr = arr[::step]

            im = ax.imshow(arr, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=arr.max())
            ax.set_xticks(range(9))
            ax.set_xticklabels(BALL_TOKEN_LABELS, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Timestep")
            ax.set_title(f"{arm.capitalize()} Arm")
            plt.colorbar(im, ax=ax, label="Attention Weight")

            # vertical lines between token groups
            for boundary in [2.5, 5.5]:
                ax.axvline(boundary, color="white", linewidth=1.5, linestyle="--")

        plt.tight_layout()
        out = Path(output_dir) / f"heatmap_{label}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()


def plot_left_vs_right(runs, output_dir):
    """For each run: left vs right arm attention on the same axes."""
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]

    fig.suptitle("Left vs Right Arm Attention", fontsize=13, fontweight="bold")
    x = np.arange(9)
    width = 0.35

    for ax, (label, data) in zip(axes, runs.items()):
        left_mean  = data["left"].mean(axis=0)
        right_mean = data["right"].mean(axis=0)

        ax.bar(x - width/2, left_mean,  width, label="Left arm",  color="#4C72B0", alpha=0.85)
        ax.bar(x + width/2, right_mean, width, label="Right arm", color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(BALL_TOKEN_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for boundary in [2.5, 5.5]:
            ax.axvline(boundary, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Mean Attention Weight")
    plt.tight_layout()
    out = Path(output_dir) / "left_vs_right.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_summary(runs):
    print("\n=== Attention Summary ===")
    for label, data in runs.items():
        print(f"\n{label}:")
        for arm in ["left", "right"]:
            means = data[arm].mean(axis=0)
            top = np.argsort(means)[::-1][:3]
            top_str = ", ".join(f"{BALL_TOKEN_LABELS[i]}({means[i]:.3f})" for i in top)
            print(f"  {arm:5s} arm top tokens: {top_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="attn_data",
                        help="Directory with .npz files from collect_attention.py")
    parser.add_argument("--output_dir", type=str, default="attn_plots")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runs = load_all(args.data_dir)
    if not runs:
        print(f"No .npz files found in {args.data_dir}. Run collect_attention.py first.")
        return

    print_summary(runs)
    plot_comparison_bars(runs, args.output_dir)
    plot_heatmaps(runs, args.output_dir)
    plot_left_vs_right(runs, args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
