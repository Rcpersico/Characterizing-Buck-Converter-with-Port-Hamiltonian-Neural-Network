"""
Train SelfIdentifyingBuckPHNN on PLECS data in physical coordinates.

No prior knowledge of L or C is required. The model learns L, C, inductor-branch
resistance R_L, and capacitor ESR R_C from measurable signals while preserving
the full Port-Hamiltonian structure in energy coordinates.

Usage:
    python src/train_si_buck_phnn.py
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
from phnn import SelfIdentifyingBuckPHNN


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "plecs_physical_train.npz"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "data" / "plecs_train.npz"

MODEL_PATH = ROOT / "models" / "si_buck_phnn.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "si_buck_phnn_loss.png"
TRAJ_PLOT_PATH = ROOT / "tests" / "si_buck_phnn_trajectories.png"

TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_R_L = 0.10
TRUE_R_C = 0.05

INIT_L = 200e-6
INIT_C = 200e-6
INIT_R_L = 0.30
INIT_R_C = 0.20
INIT_V_OFFSET = 0.5

SEED = 42
EPOCHS = 8000
LR = 1e-3
TRAIN_FRACTION = 0.8
DT_TRAIN = 50e-6
MULTISTEP_N = 20
N_STEPS = MULTISTEP_N
EARLY_STOPPING_PATIENCE = 1000
SCHEDULER_PATIENCE = 150
SCHEDULER_FACTOR = 0.5
N_TEST = 4

L_TOL_PCT = 10.0
C_TOL_PCT = 10.0
RL_TOL_PCT = 25.0
RC_TOL_PCT = 30.0
RMSE_I_TOL = 0.2
RMSE_V_TOL = 0.3


def stack_multistep_rollout(
    trajectories: list[dict],
    n_steps: int = MULTISTEP_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack (x_k, u_{k:k+N}, x_{k+1:k+N+1}, dt) windows.

    Returns:
        x0: [M, 2] initial states.
        u:  [M, N, 3] input sequences [alpha * V_in, alpha, i_o].
        xn: [M, N, 2] target state sequences.
        dt: [M] scalar timestep per window.
    """
    x0_list, u_list, xn_list, dt_list = [], [], [], []
    for td in trajectories:
        iL = td["i_L"].astype(np.float32)
        VC = td["V_C"].astype(np.float32)
        alpha = td["alpha"].astype(np.float32)
        Vin = td["V_in"].astype(np.float32)
        io = td["i_o"].astype(np.float32)
        t = td["t"]

        x = np.stack([iL, VC], axis=1)
        u = np.stack([alpha * Vin, alpha, io], axis=1)
        dt = float(np.median(np.diff(t)))

        stride = max(1, n_steps // 2)
        for start in range(0, len(t) - n_steps - 1, stride):
            x0_list.append(x[start])
            u_list.append(u[start : start + n_steps])
            xn_list.append(x[start + 1 : start + n_steps + 1])
            dt_list.append(dt)

    if not x0_list:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, n_steps, 3), dtype=np.float32),
            np.empty((0, n_steps, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.array(x0_list, dtype=np.float32),
        np.array(u_list, dtype=np.float32),
        np.array(xn_list, dtype=np.float32),
        np.array(dt_list, dtype=np.float32),
    )


def split_by_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    return idx[:n_train], idx[n_train:]


def split_trajectories(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return split_by_trajectory(n, seed)


def multistep_rk4(
    model: SelfIdentifyingBuckPHNN,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float | torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Roll out N RK4 steps from x0 using the input sequence u_seq."""
    if torch.is_tensor(dt):
        dt_t = dt.to(device=x0.device, dtype=x0.dtype)
        if dt_t.ndim == 1:
            dt_t = dt_t.view(-1, 1)
    else:
        dt_t = torch.tensor(float(dt), device=x0.device, dtype=x0.dtype)

    preds: list[torch.Tensor] = []
    x = x0
    for k in range(u_seq.shape[1]):
        u_k = u_seq[:, k, :]
        k1 = model(x, u_k, create_graph=create_graph)
        if not create_graph:
            k1 = k1.detach()
        k2 = model(x + 0.5 * dt_t * k1, u_k, create_graph=create_graph)
        if not create_graph:
            k2 = k2.detach()
        k3 = model(x + 0.5 * dt_t * k2, u_k, create_graph=create_graph)
        if not create_graph:
            k3 = k3.detach()
        k4 = model(x + dt_t * k3, u_k, create_graph=create_graph)
        if not create_graph:
            k4 = k4.detach()
        x = x + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not create_graph:
            x = x.detach()
        preds.append(x)

    return torch.stack(preds, dim=1)


def _rhs(
    model: SelfIdentifyingBuckPHNN,
    x: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout(
    model: SelfIdentifyingBuckPHNN,
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

    x0_tr, u_tr, xn_tr, dt_tr = stack_multistep_rollout(tr_trajs, N_STEPS)
    x0_va, u_va, xn_va, dt_va = stack_multistep_rollout(va_trajs, N_STEPS)
    if x0_tr.size == 0 or x0_va.size == 0:
        raise ValueError(
            f"Need trajectories longer than {N_STEPS + 1} samples for "
            "multi-step rollout training."
        )

    all_x = np.vstack((x0_tr, xn_tr.reshape(-1, 2)))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)
    data_dt = float(np.median(np.concatenate((dt_tr, dt_va))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Trajectories total : {len(trajectories)}")
    print(f"Train windows      : {x0_tr.shape[0]} x {N_STEPS} steps")
    print(f"Val   windows      : {x0_va.shape[0]} x {N_STEPS} steps")
    print(f"Training dt        : {DT_TRAIN * 1e6:.1f} us")
    print(f"Dataset dt median  : {data_dt * 1e6:.1f} us")
    print(f"State scale [A, V] : {scale_np}")
    print(
        "Initial guess      : "
        f"L={INIT_L * 1e6:.0f} uH  C={INIT_C * 1e6:.0f} uF  "
        f"R_L={INIT_R_L:.2f} Ohm  R_C={INIT_R_C:.2f} Ohm  "
        f"V_offset={INIT_V_OFFSET:.2f} V"
    )

    sc = torch.tensor(scale_np, device=device)
    tx0_tr = torch.tensor(x0_tr, device=device)
    tu_tr = torch.tensor(u_tr, device=device)
    txn_tr = torch.tensor(xn_tr, device=device)
    tx0_va = torch.tensor(x0_va, device=device)
    tu_va = torch.tensor(u_va, device=device)
    txn_va = torch.tensor(xn_va, device=device)

    model = SelfIdentifyingBuckPHNN(
        initial_L=INIT_L,
        initial_C=INIT_C,
        initial_R=INIT_R_L,
        initial_R_C=INIT_R_C,
        initial_V_offset=INIT_V_OFFSET,
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
        f"{'L[uH]':>8}  {'C[uF]':>8}  {'R_L':>8}  {'R_C':>8}  "
        f"{'Voff':>8}  {'|dG|':>8}"
    )
    print("-" * 108)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = multistep_rk4(model, tx0_tr, tu_tr, DT_TRAIN, create_graph=True)
        loss = mse(pred_tr / sc, txn_tr / sc)
        loss.backward()
        optimizer.step()

        model.eval()
        pred_va = multistep_rk4(model, tx0_va, tu_va, DT_TRAIN, create_graph=False)
        val_loss = mse(pred_va / sc, txn_va / sc)

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
                f"{p['L'] * 1e6:8.2f}  {p['C'] * 1e6:8.2f}  "
                f"{p['R_L']:8.5f}  {p['R_C']:8.5f}  "
                f"{p['V_offset']:8.4f}  {p['delta_G_norm']:8.4f}"
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
        f"C={p['C'] * 1e6:.3f} uF  R_L={p['R_L']:.5f} Ohm  "
        f"R_C={p['R_C']:.5f} Ohm  R_eff={p['R_eff']:.5f} Ohm  "
        f"V_offset={p['V_offset']:.4f} V  "
        f"delta_G_norm={p['delta_G_norm']:.4f}"
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
        "dt_train": DT_TRAIN,
        "multistep_n": N_STEPS,
        "true_params": {"L": TRUE_L, "C": TRUE_C, "R_L": TRUE_R_L, "R_C": TRUE_R_C},
        "learned_params": p,
        "init_params": {
            "L": INIT_L,
            "C": INIT_C,
            "R_L": INIT_R_L,
            "R_C": INIT_R_C,
            "V_offset": INIT_V_OFFSET,
        },
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
    ax.set_ylabel(f"Normalised {N_STEPS}-step rollout MSE")
    ax.set_title("SelfIdentifyingBuckPHNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return ckpt


def evaluate(
    model: SelfIdentifyingBuckPHNN,
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
    L_hat, C_hat = p["L"], p["C"]
    RL_hat, RC_hat = p["R_L"], p["R_C"]
    R_eff_hat = p["R_eff"]
    V_offset = p["V_offset"]
    delta_G_norm = p["delta_G_norm"]
    L_err = abs(L_hat - TRUE_L) / TRUE_L * 100
    C_err = abs(C_hat - TRUE_C) / TRUE_C * 100
    RL_err = abs(RL_hat - TRUE_R_L) / TRUE_R_L * 100
    RC_err = abs(RC_hat - TRUE_R_C) / TRUE_R_C * 100
    print(f"  Init L       : {INIT_L * 1e6:.1f} uH")
    print(f"  Learned L    : {L_hat * 1e6:.3f} uH   error: {L_err:.2f}%")
    print(f"  Init C       : {INIT_C * 1e6:.1f} uF")
    print(f"  Learned C    : {C_hat * 1e6:.3f} uF   error: {C_err:.2f}%")
    print(f"  Init R_L     : {INIT_R_L:.2f} Ohm")
    print(f"  Learned R_L  : {RL_hat:.5f} Ohm    error: {RL_err:.2f}%")
    print(f"  Init R_C     : {INIT_R_C:.2f} Ohm")
    print(f"  Learned R_C  : {RC_hat:.5f} Ohm    error: {RC_err:.2f}%")
    print(f"  Learned R_eff: {R_eff_hat:.5f} Ohm")
    print(f"  Learned V_offset : {V_offset:.4f} V")
    print(f"  delta_G_norm : {delta_G_norm:.4f}")
    print(f"  G_eff:\n{p['G_eff']}")

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
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["alpha"], gt["i_o"])).astype(
            np.float32
        )

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
        title = (
            f"Traj {idx}  alpha={gt['alpha'][0]:.2f}  "
            f"V_in={gt['V_in'][0]:.1f}V  {gt['profile_type']}"
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
    fig.suptitle("SelfIdentifyingBuckPHNN - Predicted vs PLECS", fontsize=11)
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
    print(f"  Learned R_L    : {RL_hat:.5f} Ohm   error: {RL_err:.2f}%")
    print(f"  Learned R_C    : {RC_hat:.5f} Ohm   error: {RC_err:.2f}%")
    print(f"  Learned R_eff  : {R_eff_hat:.5f} Ohm")
    print(f"  Learned V_offset : {V_offset:.4f} V")
    print(f"  delta_G_norm   : {delta_G_norm:.4f}")
    print(f"  Mean RMSE i_L  : {mean_rmse_iL:.4e} A")
    print(f"  Mean RMSE V_C  : {mean_rmse_vC:.4e} V")

    checks = [
        (L_err < L_TOL_PCT, f"L error {L_err:.2f}% >= {L_TOL_PCT}%"),
        (C_err < C_TOL_PCT, f"C error {C_err:.2f}% >= {C_TOL_PCT}%"),
        (RL_err < RL_TOL_PCT, f"R_L error {RL_err:.2f}% >= {RL_TOL_PCT}%"),
        (RC_err < RC_TOL_PCT, f"R_C error {RC_err:.2f}% >= {RC_TOL_PCT}%"),
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
            "Run: & \"$env:USERPROFILE\\anaconda3\\Scripts\\conda.exe\" "
            "run -n nilm-gpu python src/generate_plecs_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfIdentifyingBuckPHNN(
        initial_L=INIT_L,
        initial_C=INIT_C,
        initial_R=INIT_R_L,
        initial_R_C=INIT_R_C,
        initial_V_offset=INIT_V_OFFSET,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device)
