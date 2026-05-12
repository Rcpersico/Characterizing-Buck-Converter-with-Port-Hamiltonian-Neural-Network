"""Diagnostic verification for the rollout-trained BuckPHNN.

Checks:
  a. Parameter recovery (L, C, R)
  b. Trajectory rollout on 4 held-out test trajectories
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import load_dataset
from phnn import BuckPHNN
from simulator import phi_q_to_iL_vC


DATA_PATH = ROOT / "data" / "train.npz"
MODEL_PATH = ROOT / "models" / "buck_phnn_rollout.pt"
TRAJ_PLOT = ROOT / "tests" / "buck_phnn_rollout_trajectories.png"

TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_R = 0.1
SEED = 42
N_TEST = 4

# Tolerances for rollout training
L_C_TOL_PCT = 5.0
R_TOL_PCT = 10.0
RMSE_I_TOL = 0.2
RMSE_V_TOL = 0.2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> tuple[BuckPHNN, dict]:
    """Load the trained BuckPHNN from disk."""
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = BuckPHNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# RK4 rollout with time-varying input
# ---------------------------------------------------------------------------

def _rhs(
    model: BuckPHNN,
    x: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate the learned vector field at one (x, u) point."""
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    dxdt = model(xt, ut, create_graph=False)
    return dxdt.detach().cpu().numpy()[0]


def rollout(
    model: BuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """RK4 rollout from x0 using ground-truth u_eff(t) at each step.

    Args:
        x0:    Initial state shape [2].
        t:     Time grid shape [T].
        u_eff: Input array shape [T, 2] — zero-order hold within each step.
        device: Torch device.

    Returns:
        State array shape [T, 2].
    """
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        x_k = states[k]
        u_k = u_eff[k]
        k1 = _rhs(model, x_k, u_k, device)
        k2 = _rhs(model, x_k + 0.5 * dt * k1, u_k, device)
        k3 = _rhs(model, x_k + 0.5 * dt * k2, u_k, device)
        k4 = _rhs(model, x_k + dt * k3, u_k, device)
        states[k + 1] = x_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Main diagnostic routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run rollout diagnostic checks and print the summary."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(device)
    trajectories = load_dataset(DATA_PATH)

    val_indices = np.asarray(ckpt["val_indices"], dtype=int)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(val_indices, size=N_TEST, replace=False)

    # ------------------------------------------------------------------ (a)
    print("=" * 66)
    print("(a) Parameter recovery")
    print("=" * 66)

    params = model.get_params()
    L_hat, C_hat, R_hat = params["L"], params["C"], params["R_eff"]

    L_err = abs(L_hat - TRUE_L) / TRUE_L * 100
    C_err = abs(C_hat - TRUE_C) / TRUE_C * 100
    R_err = abs(R_hat - TRUE_R) / TRUE_R * 100

    print(f"  Learned L     : {L_hat:.6e} H   (true: {TRUE_L:.4e} H,  error: {L_err:.3f}%)")
    print(f"  Learned C     : {C_hat:.6e} F   (true: {TRUE_C:.4e} F,  error: {C_err:.3f}%)")
    print(f"  Learned R     : {R_hat:.6e} Ω   (true: {TRUE_R:.4e} Ω,  error: {R_err:.3f}%)")

    # ------------------------------------------------------------------ (b)
    print("\n" + "=" * 66)
    print("(b) Trajectory rollout on held-out test trajectories")
    print("=" * 66)

    mse_iL: list[float] = []
    mse_vC: list[float] = []
    pred_list: list[np.ndarray] = []
    gt_list: list[dict] = []

    for idx in test_idx:
        gt = trajectories[idx]
        t = gt["t"]
        x0 = np.array([gt["Phi"][0], gt["q"][0]], dtype=np.float32)
        u_eff_traj = np.column_stack(
            (gt["alpha"] * gt["V_in"], gt["i_o"])
        ).astype(np.float32)

        pred = rollout(model, x0, t, u_eff_traj, device)
        pred_list.append(pred)
        gt_list.append(gt)

        iL_hat, vC_hat = phi_q_to_iL_vC(pred[:, 0], pred[:, 1], TRUE_L, TRUE_C)
        err_iL = float(np.mean((iL_hat - gt["i_L"]) ** 2))
        err_vC = float(np.mean((vC_hat - gt["V_C"]) ** 2))
        mse_iL.append(err_iL)
        mse_vC.append(err_vC)
        print(
            f"  traj {idx:3d}: RMSE i_L = {err_iL**0.5:.4e} A, "
            f"RMSE V_C = {err_vC**0.5:.4e} V   "
            f"profile={gt['profile_type']}"
        )

    # Trajectory plots
    fig, axes = plt.subplots(N_TEST, 2, figsize=(12, 3 * N_TEST))
    for row, (gt, pred, idx) in enumerate(zip(gt_list, pred_list, test_idx)):
        iL_hat, vC_hat = phi_q_to_iL_vC(pred[:, 0], pred[:, 1], TRUE_L, TRUE_C)
        t_ms = gt["t"] * 1e3
        axes[row, 0].plot(t_ms, gt["i_L"], "k--", lw=1.0, label="truth")
        axes[row, 0].plot(t_ms, iL_hat, color="tab:blue", lw=1.2, label="PHNN")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(f"Traj {idx} ({gt['profile_type']})")
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"], "k--", lw=1.0, label="truth")
        axes[row, 1].plot(t_ms, vC_hat, color="tab:orange", lw=1.2, label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(f"Traj {idx} ({gt['profile_type']})")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle("BuckPHNN Rollout: Predicted vs Ground-Truth Trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(TRAJ_PLOT, dpi=200)
    plt.close(fig)
    print(f"  Saved: {TRAJ_PLOT}")

    # ------------------------------------------------------------------ Summary
    mean_rmse_iL = float(np.mean([m ** 0.5 for m in mse_iL]))
    mean_rmse_vC = float(np.mean([m ** 0.5 for m in mse_vC]))

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(
        f"  Learned L      : {L_hat:.4E} H   "
        f"(true: {TRUE_L:.4E} H,  error: {L_err:.2f}%)"
    )
    print(
        f"  Learned C      : {C_hat:.4E} F   "
        f"(true: {TRUE_C:.4E} F,  error: {C_err:.2f}%)"
    )
    print(
        f"  Learned R      : {R_hat:.4E} Ω   "
        f"(true: {TRUE_R:.4E} Ω,  error: {R_err:.2f}%)"
    )
    print(f"  Mean RMSE i_L  : {mean_rmse_iL:.4E} A")
    print(f"  Mean RMSE V_C  : {mean_rmse_vC:.4E} V")

    # Assertions
    checks = [
        (L_err < L_C_TOL_PCT,   f"L error {L_err:.2f}% >= {L_C_TOL_PCT}%"),
        (C_err < L_C_TOL_PCT,   f"C error {C_err:.2f}% >= {L_C_TOL_PCT}%"),
        (R_err < R_TOL_PCT,     f"R error {R_err:.2f}% >= {R_TOL_PCT}%"),
        (mean_rmse_iL < RMSE_I_TOL, f"Mean RMSE i_L={mean_rmse_iL:.4e} A >= {RMSE_I_TOL} A"),
        (mean_rmse_vC < RMSE_V_TOL, f"Mean RMSE V_C={mean_rmse_vC:.4e} V >= {RMSE_V_TOL} V"),
    ]
    failures = [msg for passed, msg in checks if not passed]

    if failures:
        print("\n  FAILED assertions:")
        for msg in failures:
            print(f"    - {msg}")
        print("\n  All parameter assertions: FAILED")
        print("=" * 66)
        raise AssertionError(f"{len(failures)} assertion(s) failed — see above.")

    print("\n  All parameter assertions: PASSED")
    print("=" * 66)


if __name__ == "__main__":
    main()
