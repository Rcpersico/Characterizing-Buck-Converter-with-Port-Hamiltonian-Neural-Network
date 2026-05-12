"""Train the full-input BuckPHNN on the 100-trajectory buck converter dataset.

Step 5 of the Port-Hamiltonian neural network pipeline.  Extends the damped
RLC identifier (Step 4) to handle external inputs u_eff = [alpha*V_in, i_o].
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


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.npz"
RLC_CKPT_PATH = ROOT / "models" / "rlc_phnn.pt"
MODEL_PATH = ROOT / "models" / "buck_phnn.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "buck_phnn_loss.png"

SEED = 42
EPOCHS = 3000
LR = 1e-3
TRAIN_FRACTION = 0.8
EARLY_STOPPING_PATIENCE = 500
SCHEDULER_PATIENCE = 100
SCHEDULER_FACTOR = 0.5

TRUE_PARAMS = {"L": 100e-6, "C": 100e-6, "R": 0.1}


def stack_trajectories(
    trajectories: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack trajectory dicts into (state, input, derivative) arrays.

    Returns:
        x:     shape [N, 2]  — (Phi, q) states
        u_eff: shape [N, 2]  — (alpha*V_in, i_o) inputs
        xdot:  shape [N, 2]  — (dPhi/dt, dq/dt) derivative labels
    """
    x = np.concatenate(
        [np.column_stack((t["Phi"], t["q"])) for t in trajectories], axis=0
    )
    u_eff = np.concatenate(
        [np.column_stack((t["alpha"] * t["V_in"], t["i_o"])) for t in trajectories],
        axis=0,
    )
    xdot = np.concatenate(
        [np.column_stack((t["dPhi_dt"], t["dq_dt"])) for t in trajectories], axis=0
    )
    return x.astype(np.float32), u_eff.astype(np.float32), xdot.astype(np.float32)


def split_by_trajectory(
    n_trajectories: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 trajectory-level train/val split (deterministic)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_trajectories)
    n_train = int(round(TRAIN_FRACTION * n_trajectories))
    return indices[:n_train], indices[n_train:]


def compute_derivative_scale(xdot_train: np.ndarray) -> np.ndarray:
    """Per-channel derivative std from the training set."""
    return np.maximum(xdot_train.std(axis=0), 1e-12).astype(np.float32)


def warm_start_from_rlc(model: BuckPHNN, ckpt_path: Path) -> None:
    """Copy log_inv_l, log_inv_c, r_factor_raw from the Step 4 checkpoint."""
    rlc_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rlc_state = rlc_ckpt["model_state_dict"]
    state = model.state_dict()
    for key in ("log_inv_l", "log_inv_c", "r_factor_raw"):
        if key in rlc_state:
            state[key] = rlc_state[key].clone()
    model.load_state_dict(state)
    print(f"Warm-started {('log_inv_l', 'log_inv_c', 'r_factor_raw')} from {ckpt_path.name}.")


def train() -> dict:
    """Train BuckPHNN and save the best-val-loss checkpoint.

    Returns:
        The saved checkpoint dict.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trajectories = load_dataset(DATA_PATH)
    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]

    x_tr_np, u_tr_np, y_tr_np = stack_trajectories(train_trajs)
    x_va_np, u_va_np, y_va_np = stack_trajectories(val_trajs)
    deriv_scale_np = compute_derivative_scale(y_tr_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Train points       : {x_tr_np.shape[0]}")
    print(f"Val   points       : {x_va_np.shape[0]}")
    print(f"Derivative scale   : {deriv_scale_np}")

    ds = torch.tensor(deriv_scale_np, dtype=torch.float32, device=device)
    x_tr = torch.tensor(x_tr_np, device=device)
    u_tr = torch.tensor(u_tr_np, device=device)
    y_tr = torch.tensor(y_tr_np, device=device)
    x_va = torch.tensor(x_va_np, device=device)
    u_va = torch.tensor(u_va_np, device=device)
    y_va = torch.tensor(y_va_np, device=device)

    model = BuckPHNN(
        initial_l=TRUE_PARAMS["L"],
        initial_c=TRUE_PARAMS["C"],
        initial_r=0.05,
        r_epsilon=1e-6,
    ).to(device)

    warm_start_from_rlc(model, RLC_CKPT_PATH)

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
        pred_tr = model(x_tr, u_tr, create_graph=True)
        loss = mse(pred_tr / ds, y_tr / ds)
        loss.backward()
        optimizer.step()

        model.eval()
        pred_va = model(x_va, u_va, create_graph=False)
        val_loss = mse(pred_va / ds, y_va / ds)

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
        f"Learned    : L={p['L']:.6e} H  C={p['C']:.6e} F  "
        f"R_eff={p['R_eff']:.6e} Ω"
    )
    delta_G_norm = float(
        torch.norm(p["G_eff"] - torch.tensor([[1.0, 0.0], [0.0, -1.0]]), p="fro")
    )
    print(f"||delta_G||_F (mean over grid) : {delta_G_norm:.4e}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "derivative_scale": deriv_scale_np,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_losses": np.array(train_losses, dtype=np.float64),
        "val_losses": np.array(val_losses, dtype=np.float64),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "seed": SEED,
        "epochs_run": len(train_losses),
        "learning_rate": LR,
        "true_params": TRUE_PARAMS,
        "learned_params": {"L": p["L"], "C": p["C"], "R_eff": p["R_eff"]},
        "loss_units": "MSE of (predicted xdot - true xdot) / derivative_scale",
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved model : {MODEL_PATH}")

    # Loss curve
    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ep = len(train_losses)
    ep_axis = range(1, n_ep + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(ep_axis, train_losses, label="train", alpha=0.8)
    ax.semilogy(ep_axis, val_losses, label="validation", alpha=0.8)
    ax.axvline(
        best_epoch, color="gray", linestyle="--", alpha=0.5,
        label=f"best val (ep {best_epoch})"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised derivative MSE")
    ax.set_title("BuckPHNN Training Loss (Step 5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return checkpoint


if __name__ == "__main__":
    train()