"""Train and evaluate BuckPHNN on PLECS-generated switching-simulation data.

Mirrors train_buck_phnn_rollout.py but targets plecs_train.npz and relaxes the
parameter-recovery tolerances to account for switching ripple in the PLECS data.

Usage:
    python src/train_plecs_phnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import load_dataset
from phnn import BuckPHNN
from simulator import phi_q_to_iL_vC


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "plecs_train.npz"
RLC_CKPT_PATH = ROOT / "models" / "rlc_phnn.pt"
MODEL_PATH = ROOT / "models" / "plecs_phnn.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "plecs_phnn_loss.png"
TRAJ_PLOT_PATH = ROOT / "tests" / "plecs_phnn_trajectories.png"

# Fixed circuit parameters used to generate the PLECS dataset
TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_ESR_L = 0.1   # maps to R in the PH model

# Training hyper-parameters
SEED = 42
EPOCHS = 3000
LR = 1e-3
TRAIN_FRACTION = 0.8
EARLY_STOPPING_PATIENCE = 500
SCHEDULER_PATIENCE = 100
SCHEDULER_FACTOR = 0.5

# Evaluation tolerances — looser than the analytical dataset because switching
# ripple in PLECS outputs adds noise to the averaged-model fit
L_C_TOL_PCT = 10.0
R_TOL_PCT = 25.0
RMSE_I_TOL = 0.3
RMSE_V_TOL = 0.3
N_TEST = 4


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def stack_trajectories_rollout(
    trajectories: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack trajectory dicts into one-step (x_k, u_k) -> x_{k+1} pairs."""
    x_k_list, u_k_list, x_next_list, dt_list = [], [], [], []
    for td in trajectories:
        x = np.column_stack((td["Phi"], td["q"]))
        u_eff = np.column_stack((td["alpha"] * td["V_in"], td["i_o"]))
        t = td["t"]
        x_k_list.append(x[:-1])
        u_k_list.append(u_eff[:-1])
        x_next_list.append(x[1:])
        dt_list.append(np.diff(t).reshape(-1, 1))
    return (
        np.concatenate(x_k_list, axis=0).astype(np.float32),
        np.concatenate(u_k_list, axis=0).astype(np.float32),
        np.concatenate(x_next_list, axis=0).astype(np.float32),
        np.concatenate(dt_list, axis=0).astype(np.float32),
    )


def split_by_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """80/20 trajectory-level train/val split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    return idx[:n_train], idx[n_train:]


def compute_state_scale(x: np.ndarray) -> np.ndarray:
    return np.maximum(x.std(axis=0), 1e-12).astype(np.float32)


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def rk4_step(
    model: BuckPHNN,
    x: torch.Tensor,
    u: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    if dt.ndim == 1:
        dt = dt.view(-1, 1)
    k1 = model(x, u)
    k2 = model(x + 0.5 * dt * k1, u)
    k3 = model(x + 0.5 * dt * k2, u)
    k4 = model(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rhs(model: BuckPHNN, x: np.ndarray, u: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rollout(
    model: BuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """RK4 rollout from x0 using ground-truth u_eff at each step."""
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        u_k = u_eff[k]
        k1 = _rhs(model, states[k], u_k, device)
        k2 = _rhs(model, states[k] + 0.5 * dt * k1, u_k, device)
        k3 = _rhs(model, states[k] + 0.5 * dt * k2, u_k, device)
        k4 = _rhs(model, states[k] + dt * k3, u_k, device)
        states[k + 1] = states[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    trajectories: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    """Train BuckPHNN with RK4 rollout loss on the PLECS dataset."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]

    x_k_tr, u_tr, x_next_tr, dt_tr = stack_trajectories_rollout(train_trajs)
    x_k_va, u_va, x_next_va, dt_va = stack_trajectories_rollout(val_trajs)

    x_all_tr = np.vstack((x_k_tr, x_next_tr[-1:]))
    state_scale_np = compute_state_scale(x_all_tr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Trajectories total : {len(trajectories)}")
    print(f"Train points       : {x_k_tr.shape[0]}")
    print(f"Val   points       : {x_k_va.shape[0]}")
    print(f"State scale        : {state_scale_np}")

    ds = torch.tensor(state_scale_np, dtype=torch.float32, device=device)
    tk_tr = torch.tensor(x_k_tr, device=device)
    tu_tr = torch.tensor(u_tr, device=device)
    tn_tr = torch.tensor(x_next_tr, device=device)
    td_tr = torch.tensor(dt_tr, device=device)
    tk_va = torch.tensor(x_k_va, device=device)
    tu_va = torch.tensor(u_va, device=device)
    tn_va = torch.tensor(x_next_va, device=device)
    td_va = torch.tensor(dt_va, device=device)

    model = BuckPHNN(
        initial_l=TRUE_L,
        initial_c=TRUE_C,
        initial_r=0.05,
        r_epsilon=1e-6,
    ).to(device)

    # Warm-start physical parameters from the RLC checkpoint if available
    if RLC_CKPT_PATH.exists():
        rlc_ckpt = torch.load(RLC_CKPT_PATH, map_location="cpu", weights_only=False)
        rlc_state = rlc_ckpt["model_state_dict"]
        state = model.state_dict()
        for key in ("log_inv_l", "log_inv_c", "r_factor_raw"):
            if key in rlc_state:
                state[key] = rlc_state[key].clone()
        model.load_state_dict(state)
        print(f"Warm-started (log_inv_l, log_inv_c, r_factor_raw) from {RLC_CKPT_PATH.name}")
        p_init = model.get_params()
        assert abs(p_init['L'] - TRUE_L) / TRUE_L < 0.05, \
            f"Warm-start L={p_init['L']:.2e} is >5% off from TRUE_L={TRUE_L:.2e}"
        assert abs(p_init['C'] - TRUE_C) / TRUE_C < 0.05, \
            f"Warm-start C={p_init['C']:.2e} is >5% off from TRUE_C={TRUE_C:.2e}"
        assert abs(p_init['R_eff'] - TRUE_ESR_L) / TRUE_ESR_L < 0.30, \
            f"Warm-start R={p_init['R_eff']:.2e} is >30% off from TRUE_ESR_L={TRUE_ESR_L:.2e}"
        print(f"  Verified: L={p_init['L']:.2e}  C={p_init['C']:.2e}  R={p_init['R_eff']:.2e}")
    else:
        print("RLC checkpoint not found — training from cold start.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )
    mse = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    no_improve = 0

    print(
        f"\n{'Epoch':>6}  {'train':>12}  {'val':>12}  "
        f"{'lr':>8}  {'L[H]':>10}  {'C[F]':>10}  {'R[Ω]':>8}"
    )
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = rk4_step(model, tk_tr, tu_tr, td_tr)
        loss = mse(pred_tr / ds, tn_tr / ds)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_va = rk4_step(model, tk_va, tu_va, td_va)
            val_loss = mse(pred_va / ds, tn_va / ds)

        tr_val = float(loss.detach().cpu())
        va_val = float(val_loss.detach().cpu())
        train_losses.append(tr_val)
        val_losses.append(va_val)

        scheduler.step(va_val)

        if va_val < best_val:
            best_val = va_val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 100 == 0:
            p = model.get_params()
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d}  {tr_val:12.6e}  {va_val:12.6e}  "
                f"{lr_now:8.2e}  {p['L']:10.4e}  {p['C']:10.4e}  {p['R_eff']:8.5f}"
            )

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} — "
                f"no val improvement for {EARLY_STOPPING_PATIENCE} epochs "
                f"(best val={best_val:.6e} at epoch {best_epoch})."
            )
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    p = model.get_params()

    print(f"\nBest epoch : {best_epoch}   best val loss : {best_val:.6e}")
    print(
        f"Learned    : L={p['L']:.6e} H  C={p['C']:.6e} F  R_eff={p['R_eff']:.6e} Ω"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "state_scale": state_scale_np,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_losses": np.array(train_losses, dtype=np.float64),
        "val_losses": np.array(val_losses, dtype=np.float64),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "seed": SEED,
        "epochs_run": len(train_losses),
        "learning_rate": LR,
        "true_params": {"L": TRUE_L, "C": TRUE_C, "ESR_L": TRUE_ESR_L},
        "learned_params": {"L": p["L"], "C": p["C"], "R_eff": p["R_eff"]},
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved model : {MODEL_PATH}")

    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ep = len(train_losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, n_ep + 1), train_losses, label="train", alpha=0.8)
    ax.semilogy(range(1, n_ep + 1), val_losses, label="validation", alpha=0.8)
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
               label=f"best val (ep {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised transition MSE")
    ax.set_title("PLECS BuckPHNN — Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return checkpoint


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: BuckPHNN,
    ckpt: dict,
    trajectories: list[dict],
    device: torch.device,
) -> None:
    """Evaluate the trained model: parameter recovery and trajectory rollout."""
    model.eval()
    val_idx = np.asarray(ckpt["val_indices"], dtype=int)
    rng = np.random.default_rng(SEED)
    n_test = min(N_TEST, len(val_idx))
    test_idx = rng.choice(val_idx, size=n_test, replace=False)

    # ------------------------------------------------------------------  (a)
    print("\n" + "=" * 66)
    print("(a) Parameter recovery")
    print("=" * 66)

    p = model.get_params()
    L_hat, C_hat, R_hat = p["L"], p["C"], p["R_eff"]

    L_err = abs(L_hat - TRUE_L) / TRUE_L * 100
    C_err = abs(C_hat - TRUE_C) / TRUE_C * 100
    R_err = abs(R_hat - TRUE_ESR_L) / TRUE_ESR_L * 100

    print(f"  Learned L    : {L_hat:.6e} H   (true: {TRUE_L:.4e} H,  error: {L_err:.2f}%)")
    print(f"  Learned C    : {C_hat:.6e} F   (true: {TRUE_C:.4e} F,  error: {C_err:.2f}%)")
    print(f"  Learned R    : {R_hat:.6e} Ω   (true: {TRUE_ESR_L:.4e} Ω,  error: {R_err:.2f}%)")

    # ------------------------------------------------------------------  (b)
    print("\n" + "=" * 66)
    print("(b) Trajectory rollout on held-out validation trajectories")
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

        R_load_mean = float(np.mean(gt["V_in"]))  # V_in per traj for display
        print(
            f"  traj {idx:3d}: RMSE i_L = {err_iL**0.5:.4e} A, "
            f"RMSE V_C = {err_vC**0.5:.4e} V   "
            f"alpha={gt['alpha'][0]:.2f}  V_in={gt['V_in'][0]:.1f}V"
        )

    # Trajectory plots
    fig, axes = plt.subplots(n_test, 2, figsize=(12, 3 * n_test))
    if n_test == 1:
        axes = axes[np.newaxis, :]
    for row, (gt, pred, idx) in enumerate(zip(gt_list, pred_list, test_idx)):
        iL_hat, vC_hat = phi_q_to_iL_vC(pred[:, 0], pred[:, 1], TRUE_L, TRUE_C)
        t_ms = gt["t"] * 1e3
        # Infer R_load from steady-state (last sample) to avoid divide-by-zero
        # at t=0 when V_C starts from zero
        io_ss = gt["i_o"][-1]
        vc_ss = gt["V_C"][-1]
        r_load_str = f"  R_load={vc_ss / io_ss:.1f}Ω" if io_ss > 1e-6 else ""
        title = f"Traj {idx}  α={gt['alpha'][0]:.2f}  V_in={gt['V_in'][0]:.1f}V{r_load_str}"

        axes[row, 0].plot(t_ms, gt["i_L"], "k--", lw=1.0, label="PLECS truth")
        axes[row, 0].plot(t_ms, iL_hat, color="tab:blue", lw=1.2, label="PHNN")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"], "k--", lw=1.0, label="PLECS truth")
        axes[row, 1].plot(t_ms, vC_hat, color="tab:orange", lw=1.2, label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle("PLECS BuckPHNN — Predicted vs PLECS Ground Truth", fontsize=11)
    fig.tight_layout()
    TRAJ_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAJ_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {TRAJ_PLOT_PATH}")

    # ------------------------------------------------------------------  Summary
    mean_rmse_iL = float(np.mean([m ** 0.5 for m in mse_iL]))
    mean_rmse_vC = float(np.mean([m ** 0.5 for m in mse_vC]))

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  Learned L      : {L_hat:.4E} H   (true: {TRUE_L:.4E} H,  error: {L_err:.2f}%)")
    print(f"  Learned C      : {C_hat:.4E} F   (true: {TRUE_C:.4E} F,  error: {C_err:.2f}%)")
    print(f"  Learned R      : {R_hat:.4E} Ω   (true: {TRUE_ESR_L:.4E} Ω,  error: {R_err:.2f}%)")
    print(f"  Mean RMSE i_L  : {mean_rmse_iL:.4E} A")
    print(f"  Mean RMSE V_C  : {mean_rmse_vC:.4E} V")

    checks = [
        (L_err < L_C_TOL_PCT,       f"L error {L_err:.2f}% >= {L_C_TOL_PCT}%"),
        (C_err < L_C_TOL_PCT,       f"C error {C_err:.2f}% >= {L_C_TOL_PCT}%"),
        (R_err < R_TOL_PCT,         f"R error {R_err:.2f}% >= {R_TOL_PCT}%"),
        (mean_rmse_iL < RMSE_I_TOL, f"Mean RMSE i_L={mean_rmse_iL:.4e} A >= {RMSE_I_TOL} A"),
        (mean_rmse_vC < RMSE_V_TOL, f"Mean RMSE V_C={mean_rmse_vC:.4e} V >= {RMSE_V_TOL} V"),
    ]
    failures = [msg for passed, msg in checks if not passed]
    if failures:
        print("\n  FAILED assertions:")
        for msg in failures:
            print(f"    - {msg}")
        print("\n  Result: FAILED")
        print("=" * 66)
        raise AssertionError(f"{len(failures)} assertion(s) failed — see above.")

    print("\n  Result: PASSED")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"PLECS dataset not found at {DATA_PATH}\n"
            "Run: python src/generate_plecs_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BuckPHNN(
        initial_l=TRUE_L, initial_c=TRUE_C, initial_r=0.05, r_epsilon=1e-6
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device)
