"""Train the full-input BuckPHNN on the 100-trajectory buck converter dataset
using a rollout (state-transition) loss instead of analytic derivatives.
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
MODEL_PATH = ROOT / "models" / "buck_phnn_rollout.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "buck_phnn_rollout_loss.png"

SEED = 42
EPOCHS = 3000
LR = 1e-3
TRAIN_FRACTION = 0.8
EARLY_STOPPING_PATIENCE = 500
SCHEDULER_PATIENCE = 100
SCHEDULER_FACTOR = 0.5

TRUE_PARAMS = {"L": 100e-6, "C": 100e-6, "R": 0.1}


def stack_trajectories_rollout(
    trajectories: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack trajectory dicts into one-step transition pairs.

    Returns:
        x_k:    shape [N, 2]  — (Phi_k, q_k) states
        u_k:    shape [N, 2]  — (alpha_k*V_in_k, i_o_k) inputs
        x_next: shape [N, 2]  — (Phi_{k+1}, q_{k+1}) next states
        dt:     shape [N, 1]  — time steps
    """
    x_k_list, u_k_list, x_next_list, dt_list = [], [], [], []
    for t_dict in trajectories:
        x = np.column_stack((t_dict["Phi"], t_dict["q"]))
        u_eff = np.column_stack((t_dict["alpha"] * t_dict["V_in"], t_dict["i_o"]))
        t = t_dict["t"]
        
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


def split_by_trajectory(
    n_trajectories: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 trajectory-level train/val split (deterministic)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_trajectories)
    n_train = int(round(TRAIN_FRACTION * n_trajectories))
    return indices[:n_train], indices[n_train:]


def compute_state_scale(x_train: np.ndarray) -> np.ndarray:
    """Per-channel state std from the training set."""
    return np.maximum(x_train.std(axis=0), 1e-12).astype(np.float32)


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


def rk4_step(model: BuckPHNN, x: torch.Tensor, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Perform one step of RK4 integration for batched tensors."""
    if dt.ndim == 1:
        dt = dt.view(-1, 1)

    k1 = model(x, u)
    k2 = model(x + 0.5 * dt * k1, u)
    k3 = model(x + 0.5 * dt * k2, u)
    k4 = model(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def train() -> dict:
    """Train BuckPHNN using rollout loss and save the best-val-loss checkpoint.

    Returns:
        The saved checkpoint dict.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trajectories = load_dataset(DATA_PATH)
    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]

    x_k_tr, u_tr, x_next_tr, dt_tr = stack_trajectories_rollout(train_trajs)
    x_k_va, u_va, x_next_va, dt_va = stack_trajectories_rollout(val_trajs)
    
    # We compute scale over all states (k and k+1) in the training set
    x_all_tr = np.vstack((x_k_tr, x_next_tr[-1:]))
    state_scale_np = compute_state_scale(x_all_tr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
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
        "state_scale": state_scale_np,
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
        "loss_units": "MSE of (predicted x_next - true x_next) / state_scale",
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
    ax.set_ylabel("Normalised transition MSE")
    ax.set_title("BuckPHNN Rollout Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {LOSS_PLOT_PATH}")

    return checkpoint


if __name__ == "__main__":
    train()