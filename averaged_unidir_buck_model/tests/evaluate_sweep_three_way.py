"""
Three-way comparison on the held-out sweep test set.

Models evaluated:
  1. Analytical averaged model — textbook ODEs, true datasheet parameters
  2. Physics-only PHNN       — models/sweep_buck_physics_only.pt (learned R_L, R_C)
  3. Full PHNN               — models/sweep_buck_phnn.pt (+ H_NN correction)

For each test trajectory the script:
  - Rolls out all three models with RK4 from the true initial condition
  - Computes overall RMSE and transient RMSE (first 2 ms after each step)
  - Groups results by alpha band (low 0.2-0.4, mid 0.4-0.6, high 0.6-0.8)
  - Prints a summary table
  - Saves overlay plots for 5 representative trajectories
  - Saves a bar chart of transient RMSE for all three models

Usage:
    python tests/evaluate_sweep_three_way.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_dataset import load_dataset
from phnn import SimulationBuckPHNN


ROOT = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = ROOT / "data" / "plecs_sweep_test.npz"
MODEL_PHNN_PATH = ROOT / "models" / "sweep_buck_phnn.pt"
MODEL_PHYS_PATH = ROOT / "models" / "sweep_buck_physics_only.pt"
OUT_OVERLAY = ROOT / "tests" / "test_images" / "sweep_three_way_comparison.png"
OUT_BARPLOT = ROOT / "tests" / "test_images" / "sweep_three_way_transient_bar.png"

# True analytical model parameters
L_TRUE = 100e-6
C_TRUE = 100e-6
R_TRUE = 0.1

TRANSIENT_WINDOW_S = 2e-3  # first 2 ms after each step
N_OVERLAY_TRAJS = 5


# ---------------------------------------------------------------------------
# Analytical model
# ---------------------------------------------------------------------------


def analytical_rhs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Textbook averaged buck ODE with true L, C, R."""
    i_L, V_C = x
    aVin, i_o = u
    di_L = (aVin - R_TRUE * i_L - V_C) / L_TRUE
    dV_C = (i_L - i_o) / C_TRUE
    return np.array([di_L, dV_C], dtype=np.float64)


def rk4_rollout_analytical(
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
) -> np.ndarray:
    """RK4 rollout using the analytical averaged model."""
    states = np.zeros((len(t), 2), dtype=np.float64)
    states[0] = x0.astype(np.float64)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k = u_eff[k].astype(np.float64)
        k1 = analytical_rhs(states[k], u_k)
        k2 = analytical_rhs(states[k] + 0.5 * dt_k * k1, u_k)
        k3 = analytical_rhs(states[k] + 0.5 * dt_k * k2, u_k)
        k4 = analytical_rhs(states[k] + dt_k * k3, u_k)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# PHNN rollout
# ---------------------------------------------------------------------------


def _phnn_rhs(
    model: SimulationBuckPHNN,
    x: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout_phnn(
    model: SimulationBuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k = u_eff[k]
        k1 = _phnn_rhs(model, states[k], u_k, device)
        k2 = _phnn_rhs(model, states[k] + 0.5 * dt_k * k1, u_k, device)
        k3 = _phnn_rhs(model, states[k] + 0.5 * dt_k * k2, u_k, device)
        k4 = _phnn_rhs(model, states[k] + dt_k * k3, u_k, device)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def find_step_indices(
    alpha: np.ndarray,
    io: np.ndarray,
    alpha_tol: float = 0.02,
    io_tol: float = 0.05,
) -> np.ndarray:
    """Return sample indices where an alpha or load step occurs."""
    d_alpha = np.abs(np.diff(alpha))
    d_io = np.abs(np.diff(io))
    hits = np.where((d_alpha > alpha_tol) | (d_io > io_tol))[0] + 1
    return hits


def transient_mask(
    t: np.ndarray,
    step_indices: np.ndarray,
    window_s: float = TRANSIENT_WINDOW_S,
) -> np.ndarray:
    """Boolean mask: True for samples within window_s after any step."""
    mask = np.zeros(len(t), dtype=bool)
    for si in step_indices:
        t_step = t[si]
        mask |= (t >= t_step) & (t <= t_step + window_s)
    return mask


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Return (RMSE_iL, RMSE_VC) over masked (or all) samples."""
    if mask is not None and mask.sum() == 0:
        return float("nan"), float("nan")
    err = pred - true
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(err[:, 0] ** 2))), float(np.sqrt(np.mean(err[:, 1] ** 2)))


def alpha_band(alpha_val: float) -> str:
    if alpha_val < 0.4:
        return "low (0.2-0.4)"
    elif alpha_val < 0.6:
        return "mid (0.4-0.6)"
    else:
        return "high (0.6-0.8)"


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_trajectory(
    gt: dict,
    model_phys: SimulationBuckPHNN,
    model_full: SimulationBuckPHNN,
    device: torch.device,
) -> dict:
    t = gt["t"]
    x0 = np.array([gt["i_L"][0], gt["V_C"][0]])
    u_eff = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
    gt_states = np.column_stack((gt["i_L"], gt["V_C"]))

    pred_anal = rk4_rollout_analytical(x0, t, u_eff)
    pred_phys = rk4_rollout_phnn(model_phys, x0, t, u_eff, device)
    pred_full = rk4_rollout_phnn(model_full, x0, t, u_eff, device)

    step_idx = find_step_indices(gt["alpha"], gt["i_o"])
    tmask = transient_mask(t, step_idx)

    overall_anal = rmse(pred_anal, gt_states)
    overall_phys = rmse(pred_phys.astype(np.float64), gt_states)
    overall_full = rmse(pred_full.astype(np.float64), gt_states)

    trans_anal = rmse(pred_anal, gt_states, tmask)
    trans_phys = rmse(pred_phys.astype(np.float64), gt_states, tmask)
    trans_full = rmse(pred_full.astype(np.float64), gt_states, tmask)

    # Per-alpha-band: compute per-point errors, then average by band
    band_results: dict[str, dict] = {}
    alpha_arr = gt["alpha"]
    for band in ["low (0.2-0.4)", "mid (0.4-0.6)", "high (0.6-0.8)"]:
        if band.startswith("low"):
            bmask = alpha_arr < 0.4
        elif band.startswith("mid"):
            bmask = (alpha_arr >= 0.4) & (alpha_arr < 0.6)
        else:
            bmask = alpha_arr >= 0.6
        band_results[band] = {
            "anal": rmse(pred_anal, gt_states, bmask) if bmask.any() else (float("nan"), float("nan")),
            "phys": rmse(pred_phys.astype(np.float64), gt_states, bmask) if bmask.any() else (float("nan"), float("nan")),
            "full": rmse(pred_full.astype(np.float64), gt_states, bmask) if bmask.any() else (float("nan"), float("nan")),
        }

    return {
        "t": t,
        "gt": gt_states,
        "pred_anal": pred_anal,
        "pred_phys": pred_phys,
        "pred_full": pred_full,
        "alpha": alpha_arr,
        "step_idx": step_idx,
        "overall": {"anal": overall_anal, "phys": overall_phys, "full": overall_full},
        "transient": {"anal": trans_anal, "phys": trans_phys, "full": trans_full},
        "band": band_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary_table(results: list[dict]) -> None:
    def mean_of(key: str, model: str, idx: int) -> float:
        vals = [r[key][model][idx] for r in results if not np.isnan(r[key][model][idx])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE — mean RMSE across all test trajectories")
    print("=" * 80)
    header = f"{'Metric':<32}  {'Analytical':>12}  {'Physics-only':>14}  {'Full PHNN':>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Overall RMSE i_L [A]", "overall", 0),
        ("Overall RMSE V_C [V]", "overall", 1),
        ("Transient RMSE i_L [A]", "transient", 0),
        ("Transient RMSE V_C [V]", "transient", 1),
    ]
    for label, key, idx in rows:
        a = mean_of(key, "anal", idx)
        p = mean_of(key, "phys", idx)
        f = mean_of(key, "full", idx)
        print(f"  {label:<30}  {a:12.4e}  {p:14.4e}  {f:12.4e}")

    print()
    print("Per-alpha-band transient RMSE i_L [A]:")
    for band in ["low (0.2-0.4)", "mid (0.4-0.6)", "high (0.6-0.8)"]:
        band_results = [r["band"][band] for r in results]
        a = float(np.nanmean([b["anal"][0] for b in band_results]))
        p = float(np.nanmean([b["phys"][0] for b in band_results]))
        f = float(np.nanmean([b["full"][0] for b in band_results]))
        print(f"  {band:<30}  {a:12.4e}  {p:14.4e}  {f:12.4e}")

    print("=" * 80)

    # Print acceptance verdict
    t_full_iL = mean_of("transient", "full", 0)
    t_anal_iL = mean_of("transient", "anal", 0)
    t_phys_iL = mean_of("transient", "phys", 0)

    if t_full_iL < t_anal_iL and t_full_iL < t_phys_iL:
        improvement = (1.0 - t_full_iL / t_anal_iL) * 100.0
        if improvement >= 20.0:
            verdict = f"BEST CASE: Full PHNN beats analytical by {improvement:.1f}% on transient i_L RMSE."
        else:
            verdict = (
                f"MIDDLE CASE: Full PHNN beats analytical by {improvement:.1f}% on transient i_L RMSE "
                "(< 20% — modest improvement)."
            )
    else:
        verdict = (
            "NULL CASE: Full PHNN does NOT outperform analytical model on transient RMSE. "
            "The averaged model is sufficient; PHNN value is passivity guarantee + learned R."
        )

    print(f"\nVerdict: {verdict}")


def save_overlay_plots(results: list[dict], n_plots: int = N_OVERLAY_TRAJS) -> None:
    chosen = results[:n_plots]
    OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 3.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, res in enumerate(chosen):
        t_ms = res["t"] * 1e3
        alpha_str = f"{res['alpha'].min():.2f}–{res['alpha'].max():.2f}"
        title = f"α∈[{alpha_str}]"

        for col, (state_idx, ylabel) in enumerate([(0, "i_L [A]"), (1, "V_C [V]")]):
            ax = axes[row, col]
            ax.plot(t_ms, res["gt"][:, state_idx], "k--", lw=1.2, label="PLECS", zorder=3)
            ax.plot(t_ms, res["pred_anal"][:, state_idx], color="tab:green", lw=1.0, label="Analytical", alpha=0.85)
            ax.plot(t_ms, res["pred_phys"][:, state_idx], color="tab:orange", lw=1.0, label="Physics-only", alpha=0.85)
            ax.plot(t_ms, res["pred_full"][:, state_idx], color="tab:blue", lw=1.0, label="Full PHNN", alpha=0.85)

            for si in res["step_idx"]:
                ax.axvline(res["t"][si] * 1e3, color="gray", ls=":", lw=0.8, alpha=0.6)

            ax.set_ylabel(ylabel)
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            if row == len(chosen) - 1:
                ax.set_xlabel("Time [ms]")
            if col == 0:
                ax.set_title(title)

    fig.suptitle("Three-Way Comparison: PLECS vs Analytical vs Physics-only vs Full PHNN", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY, dpi=200)
    plt.close(fig)
    print(f"Saved overlay plot : {OUT_OVERLAY}")


def save_barplot(results: list[dict]) -> None:
    OUT_BARPLOT.parent.mkdir(parents=True, exist_ok=True)

    def mean_trans(model: str, idx: int) -> float:
        vals = [r["transient"][model][idx] for r in results if not np.isnan(r["transient"][model][idx])]
        return float(np.mean(vals)) if vals else 0.0

    categories = ["i_L transient\nRMSE [A]", "V_C transient\nRMSE [V]"]
    models = ["anal", "phys", "full"]
    labels = ["Analytical", "Physics-only", "Full PHNN"]
    colors = ["tab:green", "tab:orange", "tab:blue"]

    data = [
        [mean_trans(m, 0) for m in models],  # i_L
        [mean_trans(m, 1) for m in models],  # V_C
    ]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = [data[j][i] for j in range(len(categories))]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Transient RMSE Comparison — Sweep Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_BARPLOT, dpi=200)
    plt.close(fig)
    print(f"Saved bar chart    : {OUT_BARPLOT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_phnn(model_path: Path, device: torch.device) -> SimulationBuckPHNN:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train it first with: python src/train_sweep_buck.py"
        )
    ckpt = torch.load(model_path, map_location=device)
    model = SimulationBuckPHNN(
        L=ckpt["fixed_params"]["L"],
        C=ckpt["fixed_params"]["C"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    p = model.get_params()
    no_nn = ckpt.get("no_nn", False)
    print(
        f"  Loaded {'physics-only' if no_nn else 'full PHNN'} from {model_path.name}: "
        f"R_L={p['R_L']:.5f} Ω  R_C={p['R_C']:.5f} Ω  H_NN_norm={p['H_NN_norm']:.3e}"
    )
    return model


if __name__ == "__main__":
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}\n"
            "Generate it with: python src/generate_plecs_sweep_dataset.py"
        )

    trajectories = load_dataset(TEST_DATA_PATH)
    print(f"Loaded {len(trajectories)} test trajectories from {TEST_DATA_PATH.name}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading models:")
    model_phys = load_phnn(MODEL_PHYS_PATH, device)
    model_full = load_phnn(MODEL_PHNN_PATH, device)

    print(f"\nAnalytical model: L={L_TRUE*1e6:.0f} µH  C={C_TRUE*1e6:.0f} µF  R={R_TRUE:.3f} Ω\n")
    print("Rolling out all three models on test trajectories ...")

    results: list[dict] = []
    for k, gt in enumerate(trajectories):
        res = evaluate_trajectory(gt, model_phys, model_full, device)
        results.append(res)
        alpha_rng = f"{gt['alpha'].min():.2f}–{gt['alpha'].max():.2f}"
        print(
            f"  [{k + 1:2d}/{len(trajectories)}]  α∈[{alpha_rng}]  "
            f"Overall RMSE i_L: anal={res['overall']['anal'][0]:.4e}  "
            f"phys={res['overall']['phys'][0]:.4e}  "
            f"full={res['overall']['full'][0]:.4e}"
        )

    print_summary_table(results)
    save_overlay_plots(results)
    save_barplot(results)
