"""
Comprehensive evaluation of BidirBuckPHNN on the held-out plecs_bidir_test.npz dataset.

Regime convention: This synchronous buck (no freewheeling diode) allows i_L < 0.
  - Unidirectional: i_L ≥ 0 throughout — forward conduction only.
  - Bidirectional:  i_L goes negative (reverse current). A converter with a diode
                    would clamp to zero (classical DCM); here the current flows
                    backwards.

Three evaluations:
  1. Per-regime RMSE breakdown (unidirectional vs bidirectional trajectories)
  2. Steady-state V_C accuracy vs R_load scatter
  3. Side-by-side comparison with the unidirectional-trained model (sweep_buck_phnn.pt)

Acceptance criterion for (1): bidirectional RMSE <= 3 × unidirectional RMSE.

Usage:
    python dcm_buck_model/tests/evaluate_bidir.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_dataset import load_dataset
from phnn_bidir import BidirBuckPHNN

# Import the unidirectional (CCM-trained) model for comparison
_UNIDIR_SRC = Path(__file__).resolve().parent.parent.parent / "averaged_ccm_buck_model" / "src"
sys.path.insert(0, str(_UNIDIR_SRC))
from phnn import SimulationBuckPHNN  # noqa: E402 — intentional late import


ROOT = Path(__file__).resolve().parent.parent

BIDIR_MODEL_PATH = ROOT / "models" / "bidir_buck_phnn.pt"
UNIDIR_MODEL_PATH = ROOT.parent / "averaged_ccm_buck_model" / "models" / "sweep_buck_phnn.pt"
TEST_DATA_PATH = ROOT / "data" / "plecs_bidir_test.npz"

L_TRUE = 100e-6
C_TRUE = 100e-6

BIDIR_I_THRESHOLD = -0.05   # A — below this = reverse/bidirectional current (not zero-clamped)
BIDIR_REGIME_PCT = 0.05     # >5% of samples with i_L < threshold → bidirectional trajectory


# ---------------------------------------------------------------------------
# Shared rollout helper
# ---------------------------------------------------------------------------


def _rhs_generic(model, x: np.ndarray, u: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout(model, x0: np.ndarray, t: np.ndarray, u_eff: np.ndarray,
                device: torch.device) -> np.ndarray:
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k = u_eff[k]
        k1 = _rhs_generic(model, states[k], u_k, device)
        k2 = _rhs_generic(model, states[k] + 0.5 * dt_k * k1, u_k, device)
        k3 = _rhs_generic(model, states[k] + 0.5 * dt_k * k2, u_k, device)
        k4 = _rhs_generic(model, states[k] + dt_k * k3, u_k, device)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def classify_trajectories(trajectories: list[dict]) -> tuple[list[int], list[int]]:
    """Split trajectory indices into unidirectional and bidirectional groups."""
    unidir_idx, bidir_idx = [], []
    for i, td in enumerate(trajectories):
        regime = float(td.get("regime", np.array([0.0]))[0])
        if regime > BIDIR_REGIME_PCT:
            bidir_idx.append(i)
        else:
            unidir_idx.append(i)
    return unidir_idx, bidir_idx


def compute_rmse_per_traj(model, trajectories: list[dict], indices: list[int],
                          device: torch.device) -> tuple[list[float], list[float]]:
    """Return (rmse_iL, rmse_vC) for each selected trajectory."""
    rmse_iL, rmse_vC = [], []
    for i in indices:
        gt = trajectories[i]
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
        pred = rk4_rollout(model, x0, gt["t"], u, device)
        rmse_iL.append(float(np.sqrt(np.mean((pred[:, 0] - gt["i_L"]) ** 2))))
        rmse_vC.append(float(np.sqrt(np.mean((pred[:, 1] - gt["V_C"]) ** 2))))
    return rmse_iL, rmse_vC


def steady_state_vc(arr: np.ndarray, frac: float = 0.1) -> float:
    """Mean of the last `frac` fraction of a 1D array (steady-state proxy)."""
    n = max(1, int(len(arr) * frac))
    return float(np.mean(arr[-n:]))


def get_rload_per_traj(trajectories: list[dict], indices: list[int]) -> list[float]:
    """Estimate median R_load for each trajectory from i_o and V_C."""
    r_loads = []
    for i in indices:
        gt = trajectories[i]
        # R_load = V_terminal / i_o ≈ V_C / i_o in steady state
        io_ss = steady_state_vc(gt["i_o"])
        vc_ss = steady_state_vc(gt["V_C"])
        r_loads.append(vc_ss / max(io_ss, 1e-3))
    return r_loads


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main() -> None:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}\n"
            "Generate it first with:\n"
            "  python dcm_buck_model/src/generate_plecs_bidir_dataset.py"
        )
    if not BIDIR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Bidir model not found: {BIDIR_MODEL_PATH}\n"
            "Train it first with:\n"
            "  python dcm_buck_model/src/train_bidir_buck.py"
        )
    if not UNIDIR_MODEL_PATH.exists():
        print(f"WARNING: Unidirectional model not found at {UNIDIR_MODEL_PATH}")
        print("         Side-by-side comparison will be skipped.")
        unidir_available = False
    else:
        unidir_available = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load test data
    trajectories = load_dataset(TEST_DATA_PATH)
    print(f"Loaded {len(trajectories)} test trajectories from {TEST_DATA_PATH.name}")
    unidir_idx, bidir_idx = classify_trajectories(trajectories)
    print(f"  Unidirectional trajectories (regime <= {BIDIR_REGIME_PCT*100:.0f}%): {len(unidir_idx)}")
    print(f"  Bidirectional trajectories  (regime  > {BIDIR_REGIME_PCT*100:.0f}%): {len(bidir_idx)}")
    all_idx = list(range(len(trajectories)))

    # Load bidirectional model
    ckpt_bidir = torch.load(str(BIDIR_MODEL_PATH), map_location="cpu", weights_only=False)
    bidir_model = BidirBuckPHNN(L=L_TRUE, C=C_TRUE).to(device)
    bidir_model.load_state_dict(ckpt_bidir["model_state_dict"])
    bidir_model.eval()

    # Load unidirectional (CCM-trained) model for comparison (if available)
    unidir_model = None
    if unidir_available:
        ckpt_unidir = torch.load(str(UNIDIR_MODEL_PATH), map_location="cpu", weights_only=False)
        unidir_model = SimulationBuckPHNN(L=L_TRUE, C=C_TRUE).to(device)
        unidir_model.load_state_dict(ckpt_unidir["model_state_dict"])
        unidir_model.eval()

    # ---------------------------------------------------------------
    # (1) Per-regime RMSE breakdown
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(1) Per-regime RMSE breakdown")
    print("=" * 70)

    bidir_rmse_iL, bidir_rmse_vC = compute_rmse_per_traj(bidir_model, trajectories, all_idx, device)

    unidir_trajs_rmse_iL = [bidir_rmse_iL[i] for i in unidir_idx]
    unidir_trajs_rmse_vC = [bidir_rmse_vC[i] for i in unidir_idx]
    bidir_trajs_rmse_iL = [bidir_rmse_iL[i] for i in bidir_idx]
    bidir_trajs_rmse_vC = [bidir_rmse_vC[i] for i in bidir_idx]

    mean_unidir_iL = float(np.mean(unidir_trajs_rmse_iL)) if unidir_trajs_rmse_iL else float("nan")
    mean_unidir_vC = float(np.mean(unidir_trajs_rmse_vC)) if unidir_trajs_rmse_vC else float("nan")
    mean_bidir_iL = float(np.mean(bidir_trajs_rmse_iL)) if bidir_trajs_rmse_iL else float("nan")
    mean_bidir_vC = float(np.mean(bidir_trajs_rmse_vC)) if bidir_trajs_rmse_vC else float("nan")

    print(f"\n  {'Regime':<16} {'N':>4} {'RMSE i_L [A]':>14} {'RMSE V_C [V]':>14}")
    print(f"  {'-'*16} {'-'*4} {'-'*14} {'-'*14}")
    print(f"  {'Unidirectional':<16} {len(unidir_idx):>4} {mean_unidir_iL:>14.4e} {mean_unidir_vC:>14.4e}")
    print(f"  {'Bidirectional':<16} {len(bidir_idx):>4} {mean_bidir_iL:>14.4e} {mean_bidir_vC:>14.4e}")
    if unidir_trajs_rmse_iL and bidir_trajs_rmse_iL:
        ratio_iL = mean_bidir_iL / mean_unidir_iL
        ratio_vC = mean_bidir_vC / mean_unidir_vC
        print(f"\n  Bidir/Unidir ratio  i_L: {ratio_iL:.2f}x   V_C: {ratio_vC:.2f}x")
        threshold = 3.0
        passed_iL = ratio_iL <= threshold
        passed_vC = ratio_vC <= threshold
        print(f"  Acceptance criterion (bidirectional RMSE <= {threshold}x unidirectional RMSE):")
        print(f"    i_L: {'PASS' if passed_iL else 'FAIL'}  V_C: {'PASS' if passed_vC else 'FAIL'}")

    # ---------------------------------------------------------------
    # (2) Steady-state accuracy vs R_load
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(2) Steady-state V_C accuracy vs R_load")
    print("=" * 70)

    r_loads_all = get_rload_per_traj(trajectories, all_idx)
    ss_errors_bidir, ss_vc_plecs, ss_vc_phnn = [], [], []

    for i in all_idx:
        gt = trajectories[i]
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
        pred = rk4_rollout(bidir_model, x0, gt["t"], u, device)
        vc_plecs_ss = steady_state_vc(gt["V_C"])
        vc_phnn_ss = steady_state_vc(pred[:, 1])
        ss_errors_bidir.append(vc_phnn_ss - vc_plecs_ss)
        ss_vc_plecs.append(vc_plecs_ss)
        ss_vc_phnn.append(vc_phnn_ss)

    ss_errors_unidir_model: list[float] = []
    if unidir_model is not None:
        for i in all_idx:
            gt = trajectories[i]
            x0 = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
            u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
            pred_uni = rk4_rollout(unidir_model, x0, gt["t"], u, device)
            vc_uni_ss = steady_state_vc(pred_uni[:, 1])
            ss_errors_unidir_model.append(vc_uni_ss - ss_vc_plecs[i])

    # ---------------------------------------------------------------
    # (3) Bidir vs unidirectional model comparison
    # ---------------------------------------------------------------
    if unidir_model is not None:
        print("\n" + "=" * 70)
        print("(3) Bidir model vs unidirectional model on bidirectional trajectories")
        print("=" * 70)

        if bidir_idx:
            unidir_rmse_iL_bidir, unidir_rmse_vC_bidir = compute_rmse_per_traj(
                unidir_model, trajectories, bidir_idx, device
            )
            mean_unidir_on_bidir_iL = float(np.mean(unidir_rmse_iL_bidir))
            mean_unidir_on_bidir_vC = float(np.mean(unidir_rmse_vC_bidir))
            print(f"\n  {'Model':<26} {'RMSE i_L [A]':>14} {'RMSE V_C [V]':>14}")
            print(f"  {'-'*26} {'-'*14} {'-'*14}")
            print(f"  {'Bidir model':<26} {mean_bidir_iL:>14.4e} {mean_bidir_vC:>14.4e}")
            print(f"  {'Unidirectional (baseline)':<26} {mean_unidir_on_bidir_iL:>14.4e} {mean_unidir_on_bidir_vC:>14.4e}")
            if mean_bidir_iL < mean_unidir_on_bidir_iL and mean_bidir_vC < mean_unidir_on_bidir_vC:
                print("\n  Result: Bidir model is better than unidirectional baseline on bidirectional trajectories.")
            else:
                print("\n  Result: Bidir model did NOT outperform unidirectional baseline on bidirectional trajectories.")
                print("          The experiment may have failed — consider separate models per regime.")
        else:
            print("  No bidirectional trajectories in test set — comparison skipped.")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot A: 6 representative trajectories (unidir, boundary, bidir) ---
    sorted_idx = sorted(all_idx, key=lambda i: float(trajectories[i].get("regime", np.array([0.0]))[0]))
    n_plots = min(6, len(sorted_idx))
    plot_indices = [sorted_idx[int(i * (len(sorted_idx) - 1) / (n_plots - 1))]
                    for i in range(n_plots)] if n_plots > 1 else sorted_idx[:1]

    fig_traj, axes = plt.subplots(n_plots, 2, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(plot_indices):
        gt = trajectories[idx]
        t_ms = gt["t"] * 1e3
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
        pred_b = rk4_rollout(bidir_model, x0, gt["t"], u, device)
        regime = float(gt.get("regime", np.array([0.0]))[0])
        alpha_str = f"{gt['alpha'].min():.2f}–{gt['alpha'].max():.2f}"
        title = f"Traj {idx}  regime={regime:.2f}  α∈[{alpha_str}]"

        axes[row, 0].plot(t_ms, gt["i_L"], "k--", lw=1.0, label="PLECS")
        axes[row, 0].plot(t_ms, pred_b[:, 0], color="tab:blue", lw=1.2, label="Bidir model")
        if unidir_model is not None:
            pred_u = rk4_rollout(unidir_model, x0, gt["t"], u, device)
            axes[row, 0].plot(t_ms, pred_u[:, 0], color="tab:red", lw=1.0, ls="--", alpha=0.7,
                              label="Unidir model")
        axes[row, 0].axhline(0, color="gray", lw=0.5, ls=":")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"], "k--", lw=1.0, label="PLECS")
        axes[row, 1].plot(t_ms, pred_b[:, 1], color="tab:orange", lw=1.2, label="Bidir model")
        if unidir_model is not None:
            axes[row, 1].plot(t_ms, pred_u[:, 1], color="tab:red", lw=1.0, ls="--", alpha=0.7,
                              label="Unidir model")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig_traj.suptitle("Bidir Evaluation — 6 Representative Trajectories", fontsize=11)
    fig_traj.tight_layout()
    traj_plot = out_dir / "bidir_eval_trajectories.png"
    fig_traj.savefig(traj_plot, dpi=200)
    plt.close(fig_traj)
    print(f"\nSaved trajectory plot: {traj_plot}")

    # --- Plot B: Steady-state error vs R_load ---
    fig_ss, ax_ss = plt.subplots(figsize=(8, 5))
    r_arr = np.array(r_loads_all)
    err_arr = np.array(ss_errors_bidir)
    ax_ss.scatter(r_arr, err_arr, c="tab:blue", alpha=0.7, s=40, label="Bidir model", zorder=3)
    if ss_errors_unidir_model:
        err_uni_arr = np.array(ss_errors_unidir_model)
        ax_ss.scatter(
            r_arr, err_uni_arr, c="tab:red", alpha=0.7, s=40, marker="^",
            label="Unidir model", zorder=3
        )
    ax_ss.axhline(0, color="gray", lw=1.0, ls="--")
    ax_ss.axvline(5.0, color="orange", lw=1.0, ls=":", alpha=0.7, label="Unidir/bidir boundary (~5 Ω)")
    ax_ss.set_xlabel("Estimated R_load [Ω]")
    ax_ss.set_ylabel("Steady-state V_C error: PHNN − PLECS [V]")
    ax_ss.set_title("Steady-State V_C Error vs Operating Regime")
    ax_ss.legend()
    ax_ss.grid(True, alpha=0.3)
    fig_ss.tight_layout()
    ss_plot = out_dir / "bidir_eval_steady_state.png"
    fig_ss.savefig(ss_plot, dpi=200)
    plt.close(fig_ss)
    print(f"Saved steady-state plot: {ss_plot}")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
