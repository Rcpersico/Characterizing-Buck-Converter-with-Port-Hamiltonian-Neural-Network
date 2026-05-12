"""
Train PhysicalBuckModel on PLECS data using physical coordinates (i_L, V_C).

No prior knowledge of L or C is required. The model learns L, C, and R from
measurable signals: i_L, V_C, alpha, V_in, and i_o.

Usage:
    python src/train_physical_buck.py
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
from phnn import PhysicalBuckModel


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "plecs_physical_train.npz"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "data" / "plecs_train.npz"

MODEL_PATH = ROOT / "models" / "physical_buck.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "physical_buck_loss.png"
TRAJ_PLOT_PATH = ROOT / "tests" / "physical_buck_trajectories.png"

TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_R = 0.1

INIT_L = 200e-6
INIT_C = 200e-6
INIT_R = 0.5

SEED = 42
EPOCHS = 500000
LR = 1e-3
TRAIN_FRACTION = 0.8
EARLY_STOPPING_PATIENCE = 600
SCHEDULER_PATIENCE = 150
SCHEDULER_FACTOR = 0.5
N_TEST = 4

L_TOL_PCT = 10.0
C_TOL_PCT = 10.0
R_TOL_PCT = 25.0
RMSE_I_TOL = 0.3
RMSE_V_TOL = 0.3


def stack_rollout_pairs(
    trajectories: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack (x_k, u_k, x_{k+1}, dt_k) pairs from trajectories."""
    xk, uk, xn, dt = [], [], [], []
    for td in trajectories:
        x = np.column_stack((td["i_L"], td["V_C"]))
        u = np.column_stack((td["alpha"] * td["V_in"], td["i_o"]))
        t = td["t"]
        xk.append(x[:-1])
        uk.append(u[:-1])
        xn.append(x[1:])
        dt.append(np.diff(t).reshape(-1, 1))
    return (
        np.concatenate(xk).astype(np.float32),
        np.concatenate(uk).astype(np.float32),
        np.concatenate(xn).astype(np.float32),
        np.concatenate(dt).astype(np.float32),
    )


def split_trajectories(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    return idx[:n_train], idx[n_train:]


def rk4_step(
    model: PhysicalBuckModel,
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


def _rhs(
    model: PhysicalBuckModel,
    x: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut).detach().cpu().numpy()[0]


def rk4_rollout(
    model: PhysicalBuckModel,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Full trajectory rollout from x0 = [i_L0, V_C0]."""
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k = u_eff[k]
        k1 = _rhs(model, states[k], u_k, device)
        k2 = _rhs(model, states[k] + 0.5 * dt_k * k1, u_k, device)
        k3 = _rhs(model, states[k] + 0.5 * dt_k * k2, u_k, device)
        k4 = _rhs(model, states[k] + dt_k * k3, u_k, device)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


def train(
    trajectories: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_trajs = [trajectories[i] for i in train_idx]
    va_trajs = [trajectories[i] for i in val_idx]

    xk_tr, uk_tr, xn_tr, dt_tr = stack_rollout_pairs(tr_trajs)
    xk_va, uk_va, xn_va, dt_va = stack_rollout_pairs(va_trajs)

    all_x = np.vstack((xk_tr, xn_tr))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Trajectories total : {len(trajectories)}")
    print(f"Train points       : {xk_tr.shape[0]}")
    print(f"Val   points       : {xk_va.shape[0]}")
    print(f"State scale [A, V] : {scale_np}")
    print(
        "Initial guess      : "
        f"L={INIT_L * 1e6:.0f} uH  C={INIT_C * 1e6:.0f} uF  R={INIT_R:.2f} Ohm"
    )

    sc = torch.tensor(scale_np, device=device)
    tXk = torch.tensor(xk_tr, device=device)
    tUk = torch.tensor(uk_tr, device=device)
    tXn = torch.tensor(xn_tr, device=device)
    tDt = torch.tensor(dt_tr, device=device)
    vXk = torch.tensor(xk_va, device=device)
    vUk = torch.tensor(uk_va, device=device)
    vXn = torch.tensor(xn_va, device=device)
    vDt = torch.tensor(dt_va, device=device)

    model = PhysicalBuckModel(
        initial_L=INIT_L,
        initial_C=INIT_C,
        initial_R=INIT_R,
    ).to(device)

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
        f"\n{'Epoch':>6}  {'train':>12}  {'val':>12}  {'lr':>8}  "
        f"{'L[uH]':>8}  {'C[uF]':>8}  {'R[Ohm]':>8}"
    )
    print("-" * 78)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = rk4_step(model, tXk, tUk, tDt)
        loss = mse(pred_tr / sc, tXn / sc)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_va = rk4_step(model, vXk, vUk, vDt)
            val_loss = mse(pred_va / sc, vXn / sc)

        tr_f = float(loss.detach().cpu())
        va_f = float(val_loss.detach().cpu())
        train_losses.append(tr_f)
        val_losses.append(va_f)
        scheduler.step(va_f)

        if va_f < best_val:
            best_val = va_f
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 100 == 0:
            p = model.get_params()
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d}  {tr_f:12.6e}  {va_f:12.6e}  {lr_now:8.2e}  "
                f"{p['L'] * 1e6:8.2f}  {p['C'] * 1e6:8.2f}  {p['R']:8.5f}"
            )

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(best val={best_val:.6e} at epoch {best_epoch})."
            )
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    p = model.get_params()
    print(f"\nBest epoch : {best_epoch}   best val loss : {best_val:.6e}")
    print(
        f"Learned    : L={p['L'] * 1e6:.3f} uH  "
        f"C={p['C'] * 1e6:.3f} uF  R={p['R']:.5f} Ohm"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "state_scale": scale_np,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_losses": np.array(train_losses, dtype=np.float64),
        "val_losses": np.array(val_losses, dtype=np.float64),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "true_params": {"L": TRUE_L, "C": TRUE_C, "R": TRUE_R},
        "learned_params": p,
        "init_params": {"L": INIT_L, "C": INIT_C, "R": INIT_R},
    }
    torch.save(ckpt, MODEL_PATH)
    print(f"Saved model : {MODEL_PATH}")

    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="train", alpha=0.8)
    ax.semilogy(epochs, val_losses, label="validation", alpha=0.8)
    ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised transition MSE")
    ax.set_title("PhysicalBuckModel Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return ckpt


def evaluate(
    model: PhysicalBuckModel,
    ckpt: dict,
    trajectories: list[dict],
    device: torch.device,
) -> None:
    model.eval()
    val_idx = np.asarray(ckpt["val_indices"], dtype=int)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(val_idx, size=min(N_TEST, len(val_idx)), replace=False)

    print("\n" + "=" * 66)
    print("(a) Parameter recovery")
    print("=" * 66)
    p = model.get_params()
    L_hat, C_hat, R_hat = p["L"], p["C"], p["R"]
    L_err = abs(L_hat - TRUE_L) / TRUE_L * 100
    C_err = abs(C_hat - TRUE_C) / TRUE_C * 100
    R_err = abs(R_hat - TRUE_R) / TRUE_R * 100
    print(f"  Init L       : {INIT_L * 1e6:.1f} uH")
    print(f"  Learned L    : {L_hat * 1e6:.3f} uH   error: {L_err:.2f}%")
    print(f"  Init C       : {INIT_C * 1e6:.1f} uF")
    print(f"  Learned C    : {C_hat * 1e6:.3f} uF   error: {C_err:.2f}%")
    print(f"  Init R       : {INIT_R:.2f} Ohm")
    print(f"  Learned R    : {R_hat:.5f} Ohm    error: {R_err:.2f}%")

    print("\n" + "=" * 66)
    print("(b) Trajectory rollout on held-out validation trajectories")
    print("=" * 66)

    mse_iL: list[float] = []
    mse_vC: list[float] = []
    preds: list[np.ndarray] = []
    gts: list[dict] = []
    for idx in test_idx:
        gt = trajectories[idx]
        t = gt["t"]
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], dtype=np.float32)
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)

        pred = rk4_rollout(model, x0, t, u, device)
        preds.append(pred)
        gts.append(gt)

        e_iL = float(np.mean((pred[:, 0] - gt["i_L"]) ** 2))
        e_vC = float(np.mean((pred[:, 1] - gt["V_C"]) ** 2))
        mse_iL.append(e_iL)
        mse_vC.append(e_vC)
        print(
            f"  traj {idx:3d}: RMSE i_L={e_iL ** 0.5:.4e} A  "
            f"RMSE V_C={e_vC ** 0.5:.4e} V  "
            f"alpha={gt['alpha'][0]:.2f}  V_in={gt['V_in'][0]:.1f}V"
        )

    fig, axes = plt.subplots(len(test_idx), 2, figsize=(12, 3 * len(test_idx)))
    if len(test_idx) == 1:
        axes = axes[np.newaxis, :]
    for row, (gt, pred, idx) in enumerate(zip(gts, preds, test_idx)):
        t_ms = gt["t"] * 1e3
        io_ss = gt["i_o"][-1]
        vc_ss = gt["V_C"][-1]
        r_str = f"  R_load={vc_ss / io_ss:.1f} Ohm" if io_ss > 1e-6 else ""
        title = (
            f"Traj {idx}  alpha={gt['alpha'][0]:.2f}  "
            f"V_in={gt['V_in'][0]:.1f}V{r_str}"
        )
        axes[row, 0].plot(t_ms, gt["i_L"], "k--", lw=1.0, label="PLECS")
        axes[row, 0].plot(t_ms, pred[:, 0], color="tab:blue", lw=1.2, label="Model")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"], "k--", lw=1.0, label="PLECS")
        axes[row, 1].plot(t_ms, pred[:, 1], color="tab:orange", lw=1.2, label="Model")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle("PhysicalBuckModel - Predicted vs PLECS Ground Truth", fontsize=11)
    fig.tight_layout()
    TRAJ_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAJ_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {TRAJ_PLOT_PATH}")

    mean_rmse_iL = float(np.mean([m ** 0.5 for m in mse_iL]))
    mean_rmse_vC = float(np.mean([m ** 0.5 for m in mse_vC]))
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  Learned L      : {L_hat * 1e6:.3f} uH  error: {L_err:.2f}%")
    print(f"  Learned C      : {C_hat * 1e6:.3f} uF  error: {C_err:.2f}%")
    print(f"  Learned R      : {R_hat:.5f} Ohm   error: {R_err:.2f}%")
    print(f"  Mean RMSE i_L  : {mean_rmse_iL:.4e} A")
    print(f"  Mean RMSE V_C  : {mean_rmse_vC:.4e} V")

    checks = [
        (L_err < L_TOL_PCT, f"L error {L_err:.2f}% >= {L_TOL_PCT}%"),
        (C_err < C_TOL_PCT, f"C error {C_err:.2f}% >= {C_TOL_PCT}%"),
        (R_err < R_TOL_PCT, f"R error {R_err:.2f}% >= {R_TOL_PCT}%"),
        (mean_rmse_iL < RMSE_I_TOL, f"Mean RMSE i_L={mean_rmse_iL:.4e} A"),
        (mean_rmse_vC < RMSE_V_TOL, f"Mean RMSE V_C={mean_rmse_vC:.4e} V"),
    ]
    failures = [msg for ok, msg in checks if not ok]
    if failures:
        print("\n  FAILED assertions:")
        for msg in failures:
            print(f"    - {msg}")
        print("\n  Result: FAILED")
        print("=" * 66)
        raise AssertionError(f"{len(failures)} assertion(s) failed.")

    print("\n  Result: PASSED")
    print("=" * 66)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run: python src/generate_plecs_dataset.py\n"
            "  or: python src/convert_to_physical_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    train_idx, val_idx = split_trajectories(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalBuckModel(INIT_L, INIT_C, INIT_R).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device)
