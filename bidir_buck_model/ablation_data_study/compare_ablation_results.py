"""Compare BidirBuckPHNN ablation results across dataset sizes (multi-seed version).

Loads all ablation_*_results.json files from ablation_data_study/results/,
prints a summary table, and saves two plots:

    plots/ablation_summary_mean.png
        val_loss vs n_traj with ±std error bars (across 3 seeds),
        plus rmse_unidir and rmse_bidir as median lines with IQR shaded.

    plots/ablation_summary_distribution.png
        Box plots of per-trajectory RMSE on the fixed test set,
        one box per (n_traj, regime) combination — visualises the full
        distribution across all 3 seeds × 20 trajectories.

Files produced by the single-seed format (no "aggregate" key) are loaded in
a backward-compatible way: they are shown in the summary table but skipped
in the distribution plot (no per-trajectory data available).

Regime terminology:
  - Unidirectional: i_L >= 0 — standard forward conduction.
  - Bidirectional:  i_L < -0.05 A — reverse current through the synchronous
                    MOSFET (what a freewheeling-diode converter clamps to DCM).

Run from bidir_buck_model/:
    python ablation_data_study/compare_ablation_results.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATION_DIR = Path(__file__).resolve().parent   # ablation_data_study/
RESULTS_DIR = ABLATION_DIR / "results"
PLOTS_DIR = ABLATION_DIR / "plots"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    """Load all ablation_*_results.json files, sorted by n_traj."""
    result_files = sorted(RESULTS_DIR.glob("ablation_*_results.json"))
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}")
        return []

    records: list[dict] = []
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        # Infer n_traj from filename when not stored in the JSON
        if "n_traj" not in data:
            stem = f.stem  # e.g. "ablation_060_results"
            try:
                data["n_traj"] = int(stem.split("_")[1])
            except (IndexError, ValueError):
                data["n_traj"] = -1
        records.append(data)

    records.sort(key=lambda r: r.get("n_traj", -1))
    return records


def _is_multiseed(r: dict) -> bool:
    """Return True if the record was produced by the multi-seed training scripts."""
    return "aggregate" in r and "test_set" in r


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _nan(val) -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def _fmt(val, fmt: str = ".4e") -> str:
    if _nan(val):
        return "nan"
    return format(val, fmt)


def _fmt_ms(mean, std, fmt: str = ".4e") -> str:
    """Format mean ± std string."""
    if _nan(mean):
        return "nan"
    if _nan(std):
        return format(mean, fmt)
    return f"{format(mean, fmt)} ± {format(std, fmt)}"


def _fmt_med_iqr(median, iqr, fmt: str = ".4e") -> str:
    """Format median [IQR] string."""
    if _nan(median):
        return "nan"
    if _nan(iqr):
        return format(median, fmt)
    return f"{format(median, fmt)} [{format(iqr, fmt)}]"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(records: list[dict]) -> None:
    """Print a formatted table: n_traj | val_loss mean±std | R11 | R22 | RMSE med[IQR]."""
    # Column widths
    cw_n = 8
    cw_vl = 26
    cw_r = 20
    cw_rmse = 28

    header = (
        f"{'n_traj':>{cw_n}}  "
        f"{'val_loss (mean±std)':>{cw_vl}}  "
        f"{'R11 (mean±std)':>{cw_r}}  "
        f"{'R22 (mean±std)':>{cw_r}}  "
        f"{'rmse_unidir med[IQR]':>{cw_rmse}}  "
        f"{'rmse_bidir med[IQR]':>{cw_rmse}}"
    )
    print(header)
    print("-" * len(header))

    for r in records:
        n = r.get("n_traj", "?")

        if _is_multiseed(r):
            agg = r["aggregate"]
            ts = r["test_set"]
            vl = _fmt_ms(agg.get("val_loss_mean"), agg.get("val_loss_std"))
            r11 = _fmt_ms(agg.get("R11_mean"), agg.get("R11_std"), ".5f")
            r22 = _fmt_ms(agg.get("R22_mean"), agg.get("R22_std"), ".5f")
            ru = _fmt_med_iqr(ts["unidir"].get("median"), ts["unidir"].get("iqr"))
            rb = _fmt_med_iqr(ts["bidir"].get("median"), ts["bidir"].get("iqr"))
        else:
            # Legacy single-seed format
            vl = _fmt(r.get("best_val_loss"))
            r11 = _fmt(r.get("R11"), ".5f")
            r22 = _fmt(r.get("R22"), ".5f")
            ru = _fmt(r.get("rmse_unidir"))
            rb = _fmt(r.get("rmse_bidir"))

        print(
            f"{n!s:>{cw_n}}  "
            f"{vl:>{cw_vl}}  "
            f"{r11:>{cw_r}}  "
            f"{r22:>{cw_r}}  "
            f"{ru:>{cw_rmse}}  "
            f"{rb:>{cw_rmse}}"
        )


# ---------------------------------------------------------------------------
# Plot 1: mean + error bars / median + IQR shaded
# ---------------------------------------------------------------------------

def _plot_summary_mean(records: list[dict]) -> Path:
    """Save ablation_summary_mean.png.

    Top subplot : val_loss mean ± std (error bars, log scale)
    Bottom subplot : RMSE bidir and unidir median ± IQR (shaded, log scale)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    multi = [r for r in records if _is_multiseed(r)]
    if not multi:
        print("No multi-seed results available for summary_mean plot.")
        return PLOTS_DIR / "ablation_summary_mean.png"

    n_trajs = np.array([r["n_traj"] for r in multi], dtype=float)

    val_mean = np.array([r["aggregate"]["val_loss_mean"] for r in multi])
    val_std = np.array([r["aggregate"]["val_loss_std"] for r in multi])

    unidir_med = np.array([r["test_set"]["unidir"]["median"] for r in multi])
    unidir_iqr = np.array([r["test_set"]["unidir"]["iqr"] for r in multi])
    bidir_med = np.array([r["test_set"]["bidir"]["median"] for r in multi])
    bidir_iqr = np.array([r["test_set"]["bidir"]["iqr"] for r in multi])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- Val loss with error bars ---
    mask_vl = ~np.isnan(val_mean)
    if mask_vl.any():
        ax1.errorbar(
            n_trajs[mask_vl], val_mean[mask_vl],
            yerr=val_std[mask_vl],
            fmt="^-", color="tab:gray", capsize=4, lw=1.8,
            label="val loss (mean ± std)",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Regime-weighted rollout MSE")
    ax1.set_title("BidirBuckPHNN — Data-Size Ablation\nValidation Loss")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    # --- RMSE bidir / unidir with IQR shaded ---
    def _plot_med_iqr(ax, x, med, iqr, color, marker, label):
        mask = ~np.isnan(med)
        if not mask.any():
            return
        ax.plot(x[mask], med[mask], f"{marker}-", color=color, lw=1.8, label=label)
        lo = med[mask] - 0.5 * iqr[mask]
        hi = med[mask] + 0.5 * iqr[mask]
        # Clamp lower bound to a small positive so log scale doesn't blow up
        lo = np.maximum(lo, med[mask] * 1e-3)
        ax.fill_between(x[mask], lo, hi, color=color, alpha=0.20)

    _plot_med_iqr(ax2, n_trajs, unidir_med, unidir_iqr,
                  "tab:blue", "o", "RMSE unidirectional (median ± ½IQR)")
    _plot_med_iqr(ax2, n_trajs, bidir_med, bidir_iqr,
                  "tab:orange", "s", "RMSE bidirectional (median ± ½IQR)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of training trajectories")
    ax2.set_ylabel("RMSE on fixed test set  (i_L + V_C combined) [A / V]")
    ax2.set_title("Test-Set RMSE by Regime")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = PLOTS_DIR / "ablation_summary_mean.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2: box plots of per-trajectory RMSE distribution
# ---------------------------------------------------------------------------

def _plot_summary_distribution(records: list[dict]) -> Path:
    """Save ablation_summary_distribution.png.

    One pair of boxes per dataset size (n_traj):
        left box  — RMSE on unidirectional samples
        right box — RMSE on bidirectional samples
    Each box pools all (n_seeds × 20) per-trajectory RMSE values.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    multi = [r for r in records if _is_multiseed(r)]
    if not multi:
        print("No multi-seed results available for distribution plot.")
        return PLOTS_DIR / "ablation_summary_distribution.png"

    n_sizes = len(multi)
    labels: list[str] = []
    unidir_data: list[np.ndarray] = []
    bidir_data: list[np.ndarray] = []

    for r in multi:
        n = r["n_traj"]
        labels.append(str(n))
        # Filter out None values
        raw_u = [v for v in r["test_set"]["all_rmse_unidir"] if v is not None]
        raw_b = [v for v in r["test_set"]["all_rmse_bidir"] if v is not None]
        unidir_data.append(np.array(raw_u, dtype=float) if raw_u else np.array([]))
        bidir_data.append(np.array(raw_b, dtype=float) if raw_b else np.array([]))

    # Two boxes per n_traj: unidir and bidir side by side
    fig, ax = plt.subplots(figsize=(max(8, n_sizes * 2), 5))

    positions_u = np.arange(n_sizes) * 3.0 + 0.0   # left box
    positions_b = np.arange(n_sizes) * 3.0 + 1.2   # right box

    def _boxplot(ax, data_list, positions, color, label):
        nonempty = [(pos, d) for pos, d in zip(positions, data_list) if len(d) > 0]
        if not nonempty:
            return
        pos_list = [p for p, _ in nonempty]
        d_list = [d for _, d in nonempty]
        bp = ax.boxplot(
            d_list,
            positions=pos_list,
            widths=0.9,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            boxprops=dict(facecolor=color, alpha=0.55),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, color=color),
        )
        # Invisible bar for legend
        ax.bar(
            [-999], [0], color=color, alpha=0.55, label=label
        )

    _boxplot(ax, unidir_data, positions_u, "tab:blue", "Unidirectional")
    _boxplot(ax, bidir_data, positions_b, "tab:orange", "Bidirectional")

    tick_positions = (positions_u + positions_b) / 2.0
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of training trajectories")
    ax.set_ylabel("Per-trajectory RMSE on fixed test set  (combined i_L + V_C)")
    ax.set_title(
        "BidirBuckPHNN — RMSE Distribution on Fixed Test Set\n"
        f"(3 seeds × 20 trajectories = {3 * 20} evaluations per box)"
    )
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = PLOTS_DIR / "ablation_summary_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = _load_results()
    if not records:
        return

    print("=" * 110)
    print("BidirBuckPHNN — Data-Size Ablation Summary")
    print("=" * 110)
    _print_summary_table(records)

    n_multi = sum(1 for r in records if _is_multiseed(r))
    n_legacy = len(records) - n_multi
    print(f"\n{len(records)} result file(s) loaded: "
          f"{n_multi} multi-seed, {n_legacy} legacy single-seed.")

    if n_multi > 0:
        _plot_summary_mean(records)
        _plot_summary_distribution(records)
    else:
        print(
            "\nNo multi-seed results found.  Run training scripts first:\n"
            "  python ablation_data_study/train_ablation_004.py\n"
            "  ... (008, 015, 030, 060, 120)\n"
            "Then re-run this script."
        )


if __name__ == "__main__":
    main()
