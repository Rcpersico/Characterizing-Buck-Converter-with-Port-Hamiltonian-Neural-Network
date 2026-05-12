"""
Step 2 diagnostics – loads the generated training dataset and produces
three diagnostic plots plus summary statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generate_dataset import load_dataset

DATA_PATH = Path(__file__).parent.parent / "data" / "train.npz"
OUT_DIR = Path(__file__).parent


def main() -> None:
    trajs = load_dataset(DATA_PATH)
    n = len(trajs)

    # -----------------------------------------------------------------------
    # Aggregate all timesteps across trajectories
    # -----------------------------------------------------------------------
    all_iL = np.concatenate([t["i_L"] for t in trajs])
    all_VC = np.concatenate([t["V_C"] for t in trajs])
    all_alpha = np.concatenate([t["alpha"] for t in trajs])
    all_Vin = np.concatenate([t["V_in"] for t in trajs])
    all_io = np.concatenate([t["i_o"] for t in trajs])
    total_steps = len(all_iL)

    profile_counts = Counter(t["profile_type"] for t in trajs)

    # -----------------------------------------------------------------------
    # Summary statistics (printed to stdout)
    # -----------------------------------------------------------------------
    print("=" * 50)
    print("Dataset summary")
    print("=" * 50)
    print(f"  Trajectories       : {n}")
    print(f"  Total timesteps    : {total_steps:,}")
    print(f"  i_L  range  [A]    : {all_iL.min():.4f}  ..  {all_iL.max():.4f}")
    print(f"  V_C  range  [V]    : {all_VC.min():.4f}  ..  {all_VC.max():.4f}")
    print(f"  alpha range        : {all_alpha.min():.4f}  ..  {all_alpha.max():.4f}")
    print(f"  V_in  range [V]    : {all_Vin.min():.4f}  ..  {all_Vin.max():.4f}")
    print(f"  i_o   range [A]    : {all_io.min():.4f}  ..  {all_io.max():.4f}")
    print("  Profile type fractions:")
    for pt, cnt in sorted(profile_counts.items()):
        print(f"    {pt:<15s}: {cnt / n * 100:.1f}%  ({cnt} trajectories)")
    print("=" * 50)

    # -----------------------------------------------------------------------
    # Plot 1 – Phase-space scatter of all (i_L, V_C) points
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_iL, all_VC, s=0.4, alpha=0.25, color="steelblue", rasterized=True)
    ax.set_xlabel("$i_L$ [A]")
    ax.set_ylabel("$V_C$ [V]")
    ax.set_title(
        f"Phase-space coverage — {total_steps:,} points, {n} trajectories",
        fontsize=11,
    )
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_phase_coverage.png", dpi=150)
    plt.close(fig)
    print("Saved → tests/dataset_phase_coverage.png")

    # -----------------------------------------------------------------------
    # Plot 2 – Input distribution histograms (alpha, i_o, V_in)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        (all_alpha, r"$\alpha$ (duty cycle)", "tomato"),
        (all_io,    "$i_o$ [A]",              "seagreen"),
        (all_Vin,   "$V_{in}$ [V]",           "mediumpurple"),
    ]
    for ax, (data, label, color) in zip(axes, specs):
        ax.hist(data, bins=60, color=color, edgecolor="none", alpha=0.85)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}", fontsize=10)
        ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_input_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved → tests/dataset_input_distribution.png")

    # -----------------------------------------------------------------------
    # Plot 3 – Five random sample trajectories
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(7)
    chosen = rng.choice(n, size=min(5, n), replace=False)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    for plot_idx, traj_idx in enumerate(chosen):
        traj = trajs[int(traj_idx)]
        t_ms = traj["t"] * 1e3
        label = f"traj {int(traj_idx)} ({traj['profile_type']})"
        c = colors[plot_idx % len(colors)]
        axes[0].plot(t_ms, traj["i_L"], color=c, lw=0.9, label=label)
        axes[1].plot(t_ms, traj["V_C"], color=c, lw=0.9, label=label)

    axes[0].set_ylabel("$i_L$ [A]", fontsize=11)
    axes[0].set_title("Sample trajectories — inductor current $i_L(t)$", fontsize=11)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, linewidth=0.4)
    axes[1].set_xlabel("time [ms]", fontsize=11)
    axes[1].set_ylabel("$V_C$ [V]", fontsize=11)
    axes[1].set_title("Sample trajectories — capacitor voltage $V_C(t)$", fontsize=11)
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_sample_trajectories.png", dpi=150)
    plt.close(fig)
    print("Saved → tests/dataset_sample_trajectories.png")


if __name__ == "__main__":
    main()
