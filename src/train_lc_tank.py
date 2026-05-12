"""Train a minimal PHNN on the conservative LC-tank dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from generate_lc_tank_data import load_lc_tank_dataset
from phnn import LCTankPHNN


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "lc_tank.npz"
MODEL_PATH = ROOT / "models" / "lc_tank_phnn.pt"
LOSS_PLOT_PATH = ROOT / "tests" / "lc_tank_loss.png"
SEED = 42
EPOCHS = 2000
LR = 1e-3
TRAIN_FRACTION = 0.8


def stack_trajectories(trajectories: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack trajectory dictionaries into state and derivative arrays.

    Args:
        trajectories: List of LC-tank trajectories.

    Returns:
        Tuple ``(x, xdot)`` with shapes ``[n_samples, 2]``.
    """
    x = np.concatenate(
        [np.column_stack((traj["Phi"], traj["q"])) for traj in trajectories],
        axis=0,
    )
    xdot = np.concatenate(
        [np.column_stack((traj["dPhi_dt"], traj["dq_dt"])) for traj in trajectories],
        axis=0,
    )
    return x.astype(np.float32), xdot.astype(np.float32)


def split_by_trajectory(n_trajectories: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create an 80/20 trajectory-level train/validation split."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_trajectories)
    n_train = int(round(TRAIN_FRACTION * n_trajectories))
    return indices[:n_train], indices[n_train:]


def compute_scales(x_train: np.ndarray, xdot_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-coordinate normalization scales from training data."""
    state_scale = x_train.std(axis=0)
    derivative_scale = xdot_train.std(axis=0)
    state_scale = np.maximum(state_scale, 1e-12).astype(np.float32)
    derivative_scale = np.maximum(derivative_scale, 1e-12).astype(np.float32)
    return state_scale, derivative_scale


def train() -> dict:
    """Train the LC-tank PHNN and save weights, metadata, and loss plot."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trajectories = load_lc_tank_dataset(DATA_PATH)
    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]

    x_train_np, y_train_np = stack_trajectories(train_trajs)
    x_val_np, y_val_np = stack_trajectories(val_trajs)
    state_scale_np, derivative_scale_np = compute_scales(x_train_np, y_train_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_scale = torch.tensor(state_scale_np, dtype=torch.float32, device=device)
    derivative_scale = torch.tensor(derivative_scale_np, dtype=torch.float32, device=device)

    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
    x_val = torch.tensor(x_val_np, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)

    model = LCTankPHNN(state_scale=state_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"Device: {device}")
    print(f"State scale [Phi, q]: {state_scale_np}")
    print(f"Derivative scale [dPhi/dt, dq/dt]: {derivative_scale_np}")
    print("Loss is MSE on derivatives normalized by derivative_scale.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_train = model(x_train, create_graph=True)
        loss = mse(pred_train / derivative_scale, y_train / derivative_scale)
        loss.backward()
        optimizer.step()

        model.eval()
        pred_val = model(x_val, create_graph=False)
        val_loss = mse(pred_val / derivative_scale, y_val / derivative_scale)

        train_loss_value = float(loss.detach().cpu())
        val_loss_value = float(val_loss.detach().cpu())
        train_losses.append(train_loss_value)
        val_losses.append(val_loss_value)

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d}/{EPOCHS}  "
                f"train={train_loss_value:.6e}  val={val_loss_value:.6e}"
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "state_scale": state_scale_np,
        "derivative_scale": derivative_scale_np,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_losses": np.array(train_losses, dtype=np.float64),
        "val_losses": np.array(val_losses, dtype=np.float64),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "loss_units": "MSE of (predicted xdot - true xdot) / derivative_scale",
    }
    torch.save(checkpoint, MODEL_PATH)

    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(train_losses, label="train")
    ax.semilogy(val_losses, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized derivative MSE")
    ax.set_title("LC Tank PHNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close(fig)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved loss plot: {LOSS_PLOT_PATH}")
    print(f"Final training loss: {train_losses[-1]:.6e}")
    print(f"Final validation loss: {val_losses[-1]:.6e}")
    return checkpoint


if __name__ == "__main__":
    train()
