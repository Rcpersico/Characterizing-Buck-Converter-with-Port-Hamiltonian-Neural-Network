"""
Train SwitchedBuckPHNN on raw 1 MHz PLECS switched waveforms.

Two training modes controlled by TRAINING_MODE at the top of this file:

  "fast"  — N_STEPS=4, 500 epochs. 4 µs per window = ~half a switching cycle.
            Enough for multi-step errors to accumulate so correction networks
            get real gradient signal. N=1 was tried and stalls: physics baseline
            already gives near-optimal 1-step MSE, so R_NN stays zero and H_NN
            grows without improving loss. ~2.5x cheaper per epoch than "full".

  "full"  — N_STEPS=10, 3000 epochs. 10 µs = 1 full switching cycle per window.
            Use once the fast mode has confirmed the model is learning.

Usage:
    & "$env:USERPROFILE\\anaconda3\\Scripts\\conda.exe" run -n nilm-gpu python switchingBuckModeling/train_switched_buck.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import load_dataset
from phnn import SwitchedBuckPHNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = ROOT / "data" / "plecs_switched_train.npz"
MODEL_PATH = ROOT / "models" / "switched_buck_phnn.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "test_images" / "switched_buck_phnn_loss.png"
TRAJ_PLOT_PATH = ROOT / "tests" / "test_images" / "switched_buck_phnn_trajectories.png"

# ---------------------------------------------------------------------------
# Hyperparameters (per research spec)
# ---------------------------------------------------------------------------

# Fixed — known from design
L_TRUE = 100e-6    # 100 µH
C_TRUE = 220e-6    # 220 µF

# Resistance initial guesses
INIT_R_L = 0.20    # Ohm (will be learned)
INIT_R_C = 0.05    # Ohm (will be learned)

# ---------------------------------------------------------------------------
# Toggle here — "fast" for iteration, "full" for final run
# ---------------------------------------------------------------------------
TRAINING_MODE = "fast"   # "fast" | "full"

SEED = 42
LR = 5e-4
TRAIN_FRACTION = 0.8
DT_TRAIN = 1e-6        # 1 µs — matches the dataset sampling rate

if TRAINING_MODE == "fast":
    # 4-step RK4: 4 µs ≈ half switching cycle — enough accumulated error for
    # correction networks to learn; ~2.5x cheaper per epoch than "full"
    EPOCHS = 500
    MULTISTEP_N = 4
    EARLY_STOPPING_PATIENCE = 50
    SCHEDULER_PATIENCE = 30
else:
    # 10-step RK4: 10 µs = 1 full switching cycle per window
    EPOCHS = 3000
    MULTISTEP_N = 10
    EARLY_STOPPING_PATIENCE = 300
    SCHEDULER_PATIENCE = 100

N_STEPS = MULTISTEP_N
STRIDE = max(1, N_STEPS // 2)

BATCH_SIZE = 256
SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0   # clamp gradient norm; stabilises training with square-wave v_sw
N_TEST = 4

# Tolerances for final RMSE check (higher than averaged because ripple is included)
RMSE_I_TOL = 0.50   # A
RMSE_V_TOL = 0.50   # V


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def stack_multistep_windows(
    trajectories: list[dict],
    n_steps: int = N_STEPS,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack overlapping (x0, u_seq, x_target_seq, dt) windows from trajectories.

    Uses v_sw as the first input channel, not alpha*V_in. This is the key
    difference from the averaged training script.

    Returns:
        x0: [M, 2]       initial physical states [i_L, V_C]
        u:  [M, N, 2]    input sequences [v_sw, i_o]
        xn: [M, N, 2]    target next-state sequences [i_L, V_C]
        dt: [M]          timestep per window (all equal to DT_TRAIN)
    """
    x0_list, u_list, xn_list, dt_list = [], [], [], []
    for td in trajectories:
        iL = td["i_L"].astype(np.float32)
        VC = td["V_C"].astype(np.float32)
        v_sw = td["v_sw"].astype(np.float32)
        io = td["i_o"].astype(np.float32)
        t = td["t"]

        x = np.stack([iL, VC], axis=1)          # [T, 2]
        u = np.stack([v_sw, io], axis=1)         # [T, 2]
        dt = float(np.median(np.diff(t)))

        end = len(t) - n_steps - 1
        for start in range(0, max(1, end), stride):
            x0_list.append(x[start])
            u_list.append(u[start : start + n_steps])
            xn_list.append(x[start + 1 : start + n_steps + 1])
            dt_list.append(dt)

    if not x0_list:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, n_steps, 2), dtype=np.float32),
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


# ---------------------------------------------------------------------------
# RK4 rollout
# ---------------------------------------------------------------------------


def multistep_rk4(
    model: SwitchedBuckPHNN,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    create_graph: bool = True,
) -> torch.Tensor:
    """Roll out N RK4 steps.  Returns [batch, N, 2] predicted states."""
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
    return torch.stack(preds, dim=1)  # [batch, N, 2]


def _rhs(
    model: SwitchedBuckPHNN,
    x: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout(
    model: SwitchedBuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Full trajectory rollout from x0 = [i_L0, V_C0] using u_eff = [v_sw, i_o]."""
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    trajectories: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_trajs = [trajectories[i] for i in train_idx]
    va_trajs = [trajectories[i] for i in val_idx]

    x0_tr, u_tr, xn_tr, dt_tr = stack_multistep_windows(tr_trajs, N_STEPS, STRIDE)
    x0_va, u_va, xn_va, dt_va = stack_multistep_windows(va_trajs, N_STEPS, STRIDE)

    if x0_tr.size == 0 or x0_va.size == 0:
        raise ValueError(
            f"Need trajectories longer than {N_STEPS + 1} samples for "
            "multi-step rollout training."
        )

    # State scale: normalise loss so [A] and [V] errors are weighted equally
    all_x = np.vstack((x0_tr, xn_tr.reshape(-1, 2)))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)
    data_dt = float(np.median(np.concatenate((dt_tr, dt_va))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Trajectories total : {len(trajectories)}")
    print(f"Train windows      : {x0_tr.shape[0]} x {N_STEPS} steps  (stride={STRIDE})")
    print(f"Val   windows      : {x0_va.shape[0]} x {N_STEPS} steps")
    print(f"Training dt        : {DT_TRAIN * 1e6:.1f} us  (dataset dt: {data_dt * 1e6:.1f} us)")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"State scale [A, V] : {scale_np}")
    print(
        f"Fixed L={L_TRUE * 1e6:.0f} uH  C={C_TRUE * 1e6:.0f} uF  "
        f"Init R_L={INIT_R_L:.2f} Ohm  Init R_C={INIT_R_C:.2f} Ohm"
    )

    n_train = x0_tr.shape[0]
    sc = torch.tensor(scale_np, device=device)
    tx0_tr = torch.tensor(x0_tr, device=device)
    tu_tr = torch.tensor(u_tr, device=device)
    txn_tr = torch.tensor(xn_tr, device=device)
    tx0_va = torch.tensor(x0_va, device=device)
    tu_va = torch.tensor(u_va, device=device)
    txn_va = torch.tensor(xn_va, device=device)

    model = SwitchedBuckPHNN(
        L=L_TRUE,
        C=C_TRUE,
        initial_R_L=INIT_R_L,
        initial_R_C=INIT_R_C,
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
        f"{'R_L':>8}  {'R_C':>8}  {'|R_NN|':>8}  {'|H_NN|':>8}"
    )
    print("-" * 90)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Mini-batch training with random permutation each epoch
        perm = torch.randperm(n_train, device=device)
        epoch_loss_sum = 0.0

        for batch_start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[batch_start : batch_start + BATCH_SIZE]
            x0_b = tx0_tr[batch_idx]
            u_b = tu_tr[batch_idx]
            xn_b = txn_tr[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            pred_b = multistep_rk4(model, x0_b, u_b, DT_TRAIN, create_graph=True)
            loss_b = mse(pred_b / sc, xn_b / sc)
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss_sum += float(loss_b.detach()) * len(batch_idx)

        tr_f = epoch_loss_sum / n_train

        model.eval()
        with torch.no_grad():
            pred_va = multistep_rk4(model, tx0_va, tu_va, DT_TRAIN, create_graph=False)
            va_f = float(mse(pred_va / sc, txn_va / sc))

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

        if epoch == 1 or epoch % 50 == 0:
            p = model.get_params()
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d}  {tr_f:12.6e}  {va_f:12.6e}  {lr_now:8.2e}  "
                f"{p['R_L']:8.5f}  {p['R_C']:8.5f}  {p['R_NN_norm']:8.4f}  "
                f"{p['H_NN_norm']:8.4f}"
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
        f"Learned    : R_L={p['R_L']:.5f} Ohm  "
        f"R_C={p['R_C']:.5f} Ohm  "
        f"R_NN_norm={p['R_NN_norm']:.4f}  "
        f"H_NN_norm={p['H_NN_norm']:.4f}"
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
        "fixed_params": {"L": L_TRUE, "C": C_TRUE},
        "learned_params": p,
        "init_params": {"R_L": INIT_R_L, "R_C": INIT_R_C},
    }
    torch.save(ckpt, MODEL_PATH)
    print(f"Saved model : {MODEL_PATH}")

    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_ax = range(1, len(train_losses) + 1)
    ax.semilogy(epochs_ax, train_losses, label="train", alpha=0.8)
    ax.semilogy(epochs_ax, val_losses, label="validation", alpha=0.8)
    ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5, label=f"best @ {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Normalised {N_STEPS}-step rollout MSE")
    ax.set_title(f"SwitchedBuckPHNN Training Loss  [{TRAINING_MODE} mode, N={N_STEPS}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return ckpt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: SwitchedBuckPHNN,
    ckpt: dict,
    trajectories: list[dict],
    device: torch.device,
) -> None:
    """Roll out SwitchedBuckPHNN on held-out trajectories and plot first 0.5 ms."""
    model.eval()
    val_idx = np.asarray(ckpt["val_indices"], dtype=int)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(val_idx, size=min(N_TEST, len(val_idx)), replace=False)

    print("\n" + "=" * 70)
    print("(a) Learned parameters")
    print("=" * 70)
    p = model.get_params()
    print(f"  Fixed L          : {p['L'] * 1e6:.1f} uH")
    print(f"  Fixed C          : {p['C'] * 1e6:.1f} uF")
    print(f"  Init R_L         : {INIT_R_L:.2f} Ohm")
    print(f"  Learned R_L      : {p['R_L']:.5f} Ohm")
    print(f"  Init R_C         : {INIT_R_C:.2f} Ohm")
    print(f"  Learned R_C      : {p['R_C']:.5f} Ohm")
    print(f"  R_NN_norm        : {p['R_NN_norm']:.4f}")
    print(f"  H_NN_norm        : {p['H_NN_norm']:.4f}")

    print("\n" + "=" * 70)
    print("(b) Trajectory rollout on held-out validation trajectories")
    print("=" * 70)

    mse_iL: list[float] = []
    mse_vC: list[float] = []
    preds: list[np.ndarray] = []
    gts: list[dict] = []
    for idx in test_idx:
        gt = trajectories[idx]
        t = gt["t"]
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], dtype=np.float32)
        u = np.column_stack((gt["v_sw"], gt["i_o"])).astype(np.float32)

        pred = rk4_rollout(model, x0, t, u, device)
        preds.append(pred)
        gts.append(gt)

        e_iL = float(np.mean((pred[:, 0] - gt["i_L"]) ** 2))
        e_vC = float(np.mean((pred[:, 1] - gt["V_C"]) ** 2))
        mse_iL.append(e_iL)
        mse_vC.append(e_vC)
        print(
            f"  traj {idx:3d}: RMSE i_L={e_iL**0.5:.4e} A  "
            f"RMSE V_C={e_vC**0.5:.4e} V  "
            f"alpha={gt['alpha'][0]:.2f}  V_in={gt['V_in'][0]:.1f}V"
        )

    # Plot only the first 0.5 ms to make ripple visible
    limit_idx = int(0.5e-3 / DT_TRAIN)

    fig, axes = plt.subplots(len(test_idx), 2, figsize=(12, 3 * len(test_idx)))
    if len(test_idx) == 1:
        axes = axes[np.newaxis, :]
    for row, (gt, pred, idx) in enumerate(zip(gts, preds, test_idx)):
        t_us = gt["t"][:limit_idx] * 1e6
        title = (
            f"Traj {idx}  alpha={gt['alpha'][0]:.2f}  "
            f"V_in={gt['V_in'][0]:.1f}V  {gt['profile_type']}"
        )
        axes[row, 0].plot(t_us, gt["i_L"][:limit_idx], "k-", lw=1.5, alpha=0.5, label="PLECS")
        axes[row, 0].plot(t_us, pred[:limit_idx, 0], color="tab:blue", lw=1.2, ls="--", label="PHNN")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_us, gt["V_C"][:limit_idx], "k-", lw=1.5, alpha=0.5, label="PLECS")
        axes[row, 1].plot(t_us, pred[:limit_idx, 1], color="tab:orange", lw=1.2, ls="--", label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [µs]")
    fig.suptitle("SwitchedBuckPHNN — Predicted vs PLECS (first 500 µs)", fontsize=11)
    fig.tight_layout()
    TRAJ_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAJ_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {TRAJ_PLOT_PATH}")

    mean_rmse_iL = float(np.mean([m**0.5 for m in mse_iL]))
    mean_rmse_vC = float(np.mean([m**0.5 for m in mse_vC]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Fixed  L         : {p['L'] * 1e6:.1f} uH")
    print(f"  Fixed  C         : {p['C'] * 1e6:.1f} uF")
    print(f"  Learned R_L      : {p['R_L']:.5f} Ohm")
    print(f"  Learned R_C      : {p['R_C']:.5f} Ohm")
    print(f"  R_NN_norm        : {p['R_NN_norm']:.4f}")
    print(f"  H_NN_norm        : {p['H_NN_norm']:.4f}")
    print(f"  Mean RMSE i_L    : {mean_rmse_iL:.4e} A  (tol: {RMSE_I_TOL})")
    print(f"  Mean RMSE V_C    : {mean_rmse_vC:.4e} V  (tol: {RMSE_V_TOL})")

    failures = []
    if mean_rmse_iL >= RMSE_I_TOL:
        failures.append(f"Mean RMSE i_L={mean_rmse_iL:.4e} >= {RMSE_I_TOL}")
    if mean_rmse_vC >= RMSE_V_TOL:
        failures.append(f"Mean RMSE V_C={mean_rmse_vC:.4e} >= {RMSE_V_TOL}")

    if failures:
        print("\n  FAILED assertions:")
        for msg in failures:
            print(f"    - {msg}")
        print("\n  Result: FAILED")
        print("=" * 70)
        raise AssertionError(f"{len(failures)} assertion(s) failed.")

    print("\n  Result: PASSED")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run with PLECS active:\n"
            "  & \"$env:USERPROFILE\\anaconda3\\Scripts\\conda.exe\" run -n nilm-gpu "
            "python switchingBuckModeling/generate_plecs_switched_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    # Verify v_sw field exists (requires the switched dataset, not the averaged one)
    if "v_sw" not in trajectories[0]:
        raise KeyError(
            "Dataset is missing the 'v_sw' field. "
            "Re-run generate_plecs_switched_dataset.py to regenerate the switched dataset."
        )

    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwitchedBuckPHNN(
        L=L_TRUE,
        C=C_TRUE,
        initial_R_L=INIT_R_L,
        initial_R_C=INIT_R_C,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device)
