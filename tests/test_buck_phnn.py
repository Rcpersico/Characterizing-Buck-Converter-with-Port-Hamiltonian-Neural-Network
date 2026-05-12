"""Diagnostic verification for the trained BuckPHNN (Step 5).

Four checks:
  a. Parameter recovery (L, C, R, ||delta_G||_F)
  b. Trajectory rollout on 4 held-out test trajectories
  c. Energy behaviour under input
  d. Input-coupling matrix check at 5 operating points
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
MODEL_PATH = ROOT / "models" / "buck_phnn.pt"
TRAJ_PLOT = ROOT / "tests" / "buck_phnn_trajectories.png"
ENERGY_PLOT = ROOT / "tests" / "buck_phnn_energy.png"

TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_R = 0.1
SEED = 42
N_TEST = 4
PARAM_TOL_PCT = 5.0   # 5 % tolerance on L, C, R
DELTA_G_TOL = 0.1
G_ENTRY_TOL = 0.1


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
# Hamiltonian evaluation
# ---------------------------------------------------------------------------

def eval_hamiltonian(
    model: BuckPHNN, states: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return H_theta over a state trajectory (no grad needed)."""
    x = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        h = model.hamiltonian(x)
    return h.cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Main diagnostic routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all four diagnostic checks and print the summary."""
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
    G_eff: torch.Tensor = params["G_eff"]
    G_phys_t = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    delta_G_mean = G_eff - G_phys_t
    delta_G_norm = float(torch.norm(delta_G_mean, p="fro"))

    L_err = abs(L_hat - TRUE_L) / TRUE_L * 100
    C_err = abs(C_hat - TRUE_C) / TRUE_C * 100
    R_err = abs(R_hat - TRUE_R) / TRUE_R * 100

    print(f"  Learned L     : {L_hat:.6e} H   (true: {TRUE_L:.4e} H,  error: {L_err:.3f}%)")
    print(f"  Learned C     : {C_hat:.6e} F   (true: {TRUE_C:.4e} F,  error: {C_err:.3f}%)")
    print(f"  Learned R     : {R_hat:.6e} Ω   (true: {TRUE_R:.4e} Ω,  error: {R_err:.3f}%)")
    print(f"  ||delta_G||_F : {delta_G_norm:.4e}   (should be < {DELTA_G_TOL})")

    if L_err > 1.0 or C_err > 1.0 or R_err > 1.0:
        print(
            "  NOTE: Parameter error > 1% — possible warm-start or "
            "input-coupling interference with H parameterisation."
        )

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
    fig.suptitle("BuckPHNN: Predicted vs Ground-Truth Trajectories (Step 5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(TRAJ_PLOT, dpi=200)
    plt.close(fig)
    print(f"  Saved: {TRAJ_PLOT}")

    # ------------------------------------------------------------------ (c)
    print("\n" + "=" * 66)
    print("(c) Energy behaviour under input")
    print("=" * 66)

    gt0 = gt_list[0]
    pred0 = pred_list[0]
    t0 = gt0["t"]
    h0 = eval_hamiltonian(model, pred0, device)
    iL0, vC0 = phi_q_to_iL_vC(pred0[:, 0], pred0[:, 1], TRUE_L, TRUE_C)

    P_in = gt0["alpha"] * gt0["V_in"] * iL0
    P_R = TRUE_R * iL0 ** 2
    P_out = vC0 * gt0["i_o"]
    dH_dt = np.gradient(h0, t0)

    h_rising = bool(np.any(np.diff(h0) > 0))
    power_balance_corr = float(
        np.corrcoef(dH_dt[1:-1], (P_in - P_R - P_out)[1:-1])[0, 1]
    )
    print(f"  H rises at least once : {h_rising} (expected True — input injects power)")
    print(f"  Corr(dH/dt, P_in-P_R-P_out) : {power_balance_corr:.4f} (expected > 0.9)")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t0 * 1e3, h0, color="tab:blue")
    axes[0].set_ylabel("H_theta [J]")
    axes[0].set_title(
        f"Traj {test_idx[0]}: Learned Hamiltonian along predicted rollout (with input)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t0 * 1e3, P_in, label="P_in = α·V_in·i_L", color="tab:green")
    axes[1].plot(t0 * 1e3, P_R, label="P_R = R·i_L²", color="tab:red")
    axes[1].plot(t0 * 1e3, P_out, label="P_out = V_C·i_o", color="tab:orange")
    axes[1].plot(
        t0 * 1e3, dH_dt, "--", color="tab:blue", alpha=0.7,
        label="dH/dt (numerical diff)"
    )
    axes[1].set_ylabel("Power [W]")
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_title("Energy budget: dH/dt ≈ P_in − P_R − P_out")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ENERGY_PLOT, dpi=200)
    plt.close(fig)
    print(f"  Saved: {ENERGY_PLOT}")

    # ------------------------------------------------------------------ (d)
    print("\n" + "=" * 66)
    print("(d) Input-coupling matrix G_theta at 5 operating points")
    print("=" * 66)

    op_points = [
        (0.0,             5.0 * TRUE_C,   "zero current, mid V_C"),
        (1.0 * TRUE_L,    5.0 * TRUE_C,   "nominal (1 A, 5 V)"),
        (2.0 * TRUE_L,    8.0 * TRUE_C,   "high load (2 A, 8 V)"),
        (0.5 * TRUE_L,    3.0 * TRUE_C,   "light load (0.5 A, 3 V)"),
        (1.5 * TRUE_L,    7.0 * TRUE_C,   "medium (1.5 A, 7 V)"),
    ]
    G_phys_np = np.array([[1.0, 0.0], [0.0, -1.0]])
    max_entry_dev = 0.0
    flagged_entries = False

    for phi, q, label in op_points:
        x = torch.tensor([[phi, q]], dtype=torch.float32, device=device)
        with torch.no_grad():
            dg_flat = model.delta_G_net(x)
        dg = dg_flat.cpu().numpy().reshape(2, 2)
        G_full = G_phys_np + dg
        dev = float(np.max(np.abs(dg)))
        max_entry_dev = max(max_entry_dev, dev)
        flag = " *** FLAGGED" if dev > G_ENTRY_TOL else ""
        print(f"  [{label}]")
        print(
            f"    G = [[{G_full[0,0]:+.4f}, {G_full[0,1]:+.4f}],"
            f" [{G_full[1,0]:+.4f}, {G_full[1,1]:+.4f}]]"
            f"  max|delta_G|={dev:.4e}{flag}"
        )
        if dev > G_ENTRY_TOL:
            flagged_entries = True

    print(f"  Max |delta_G| entry across all points: {max_entry_dev:.4e}")

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
    print(f"  ||delta_G||_F  : {delta_G_norm:.4E}     (should be < {DELTA_G_TOL})")
    print(f"  Mean RMSE i_L  : {mean_rmse_iL:.4E} A")
    print(f"  Mean RMSE V_C  : {mean_rmse_vC:.4E} V")

    if flagged_entries:
        print(
            f"  WARNING: G entries deviate > {G_ENTRY_TOL} from G_physics "
            f"at some operating points."
        )

    # Assertions
    checks = [
        (L_err < PARAM_TOL_PCT,   f"L error {L_err:.2f}% exceeds {PARAM_TOL_PCT}%"),
        (C_err < PARAM_TOL_PCT,   f"C error {C_err:.2f}% exceeds {PARAM_TOL_PCT}%"),
        (R_err < PARAM_TOL_PCT,   f"R error {R_err:.2f}% exceeds {PARAM_TOL_PCT}%"),
        (delta_G_norm < DELTA_G_TOL, f"||delta_G||_F={delta_G_norm:.4e} >= {DELTA_G_TOL}"),
        (mean_rmse_iL < 0.1,      f"Mean RMSE i_L={mean_rmse_iL:.4e} A >= 0.1 A"),
        (mean_rmse_vC < 0.1,      f"Mean RMSE V_C={mean_rmse_vC:.4e} V >= 0.1 V"),
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
