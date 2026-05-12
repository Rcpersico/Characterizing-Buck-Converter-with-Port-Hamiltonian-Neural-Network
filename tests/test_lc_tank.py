"""Diagnostic checks for the trained conservative LC-tank PHNN."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from generate_lc_tank_data import load_lc_tank_dataset
from phnn import LCTankPHNN
from simulator import phi_q_to_iL_vC


DATA_PATH = ROOT / "data" / "lc_tank.npz"
MODEL_PATH = ROOT / "models" / "lc_tank_phnn.pt"
ENERGY_PLOT_PATH = ROOT / "tests" / "lc_tank_energy_conservation.png"
ORBITS_PLOT_PATH = ROOT / "tests" / "lc_tank_orbits.png"
CONTOUR_PLOT_PATH = ROOT / "tests" / "lc_tank_hamiltonian_contour.png"
L = 100e-6
C = 100e-6
DT = 10e-6
SEED = 42


def load_model(device: torch.device) -> tuple[LCTankPHNN, dict]:
    """Load the trained PHNN checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_scale = torch.tensor(checkpoint["state_scale"], dtype=torch.float32, device=device)
    model = LCTankPHNN(state_scale=state_scale).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def phnn_rhs(model: LCTankPHNN, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Evaluate the learned vector field at a single state."""
    x_tensor = torch.tensor(x[None, :], dtype=torch.float32, device=device)
    dxdt = model(x_tensor, create_graph=False)
    return dxdt.detach().cpu().numpy()[0]


def rk4_integrate(
    model: LCTankPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Integrate the learned vector field with fixed-step RK4."""
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        x = states[k]
        k1 = phnn_rhs(model, x, device)
        k2 = phnn_rhs(model, x + 0.5 * dt * k1, device)
        k3 = phnn_rhs(model, x + 0.5 * dt * k2, device)
        k4 = phnn_rhs(model, x + dt * k3, device)
        states[k + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return states


def evaluate_hamiltonian(
    model: LCTankPHNN,
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate ``H_theta`` over a state array."""
    x = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        h = model.hamiltonian(x)
    return h.detach().cpu().numpy().reshape(-1)


def select_validation_trajectories(
    trajectories: list[dict],
    val_indices: np.ndarray,
) -> list[int]:
    """Pick three reproducible random validation trajectories."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(val_indices, size=3, replace=False)
    return [int(i) for i in chosen]


def plot_energy(
    t: np.ndarray,
    h_values: list[np.ndarray],
    out_path: Path,
) -> None:
    """Plot learned Hamiltonian values along predicted trajectories."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, h in enumerate(h_values, start=1):
        ax.plot(t * 1e3, h, label=f"trajectory {idx}")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Learned H_theta")
    ax.set_title("PHNN Energy Along Predicted LC-Tank Trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_orbits(
    predicted: list[np.ndarray],
    truth: list[dict],
    out_path: Path,
) -> None:
    """Plot predicted and ground-truth phase-space orbits."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, (pred_states, gt, color) in enumerate(zip(predicted, truth, colors), start=1):
        pred_i_l, pred_v_c = phi_q_to_iL_vC(pred_states[:, 0], pred_states[:, 1], L, C)
        ax.plot(gt["i_L"], gt["V_C"], color=color, linestyle="--", alpha=0.75, label=f"truth {idx}")
        ax.plot(pred_i_l, pred_v_c, color=color, linewidth=1.5, label=f"PHNN {idx}")
    ax.set_xlabel("i_L [A]")
    ax.set_ylabel("V_C [V]")
    ax.set_title("LC Tank Phase-Space Orbits")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hamiltonian_contour(
    model: LCTankPHNN,
    trajectories: list[dict],
    device: torch.device,
    out_path: Path,
) -> None:
    """Evaluate and plot ``H_theta`` on a 100 x 100 ``(Phi, q)`` grid."""
    all_phi = np.concatenate([traj["Phi"] for traj in trajectories])
    all_q = np.concatenate([traj["q"] for traj in trajectories])
    phi_pad = 0.05 * (all_phi.max() - all_phi.min())
    q_pad = 0.05 * (all_q.max() - all_q.min())
    phi_grid = np.linspace(all_phi.min() - phi_pad, all_phi.max() + phi_pad, 100)
    q_grid = np.linspace(all_q.min() - q_pad, all_q.max() + q_pad, 100)
    phi_mesh, q_mesh = np.meshgrid(phi_grid, q_grid)
    grid_states = np.column_stack((phi_mesh.ravel(), q_mesh.ravel())).astype(np.float32)
    h_grid = evaluate_hamiltonian(model, grid_states, device).reshape(phi_mesh.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(phi_mesh, q_mesh, h_grid, levels=24)
    ax.clabel(contour, inline=True, fontsize=7)
    ax.set_xlabel("Phi [Wb]")
    ax.set_ylabel("q [C]")
    ax.set_title("Learned Hamiltonian Contours")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the three requested diagnostics and print numerical metrics."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(device)
    trajectories = load_lc_tank_dataset(DATA_PATH)
    val_indices = np.asarray(checkpoint["val_indices"], dtype=int)
    chosen_indices = select_validation_trajectories(trajectories, val_indices)
    chosen_truth = [trajectories[i] for i in chosen_indices]

    predicted_states: list[np.ndarray] = []
    h_values: list[np.ndarray] = []
    energy_drifts: list[float] = []
    trajectory_mses: list[float] = []

    for gt in chosen_truth:
        t = gt["t"]
        x0 = np.array([gt["Phi"][0], gt["q"][0]], dtype=np.float32)
        pred = rk4_integrate(model, x0, t, device)
        predicted_states.append(pred)

        h = evaluate_hamiltonian(model, pred, device)
        h_values.append(h)
        denominator = max(float(np.max(np.abs(h))), 1e-12)
        energy_drifts.append(float(np.max(np.abs(h - h[0])) / denominator))

        pred_i_l, pred_v_c = phi_q_to_iL_vC(pred[:, 0], pred[:, 1], L, C)
        gt_physical = np.column_stack((gt["i_L"], gt["V_C"]))
        pred_physical = np.column_stack((pred_i_l, pred_v_c))
        trajectory_mses.append(float(np.mean((pred_physical - gt_physical) ** 2)))

    plot_energy(chosen_truth[0]["t"], h_values, ENERGY_PLOT_PATH)
    plot_orbits(predicted_states, chosen_truth, ORBITS_PLOT_PATH)
    plot_hamiltonian_contour(model, trajectories, device, CONTOUR_PLOT_PATH)

    print(f"Final training loss: {checkpoint['final_train_loss']:.6e}")
    print(f"Final validation loss: {checkpoint['final_val_loss']:.6e}")
    for idx, (traj_idx, drift, mse) in enumerate(
        zip(chosen_indices, energy_drifts, trajectory_mses),
        start=1,
    ):
        print(
            f"Trajectory {idx} (dataset index {traj_idx}): "
            f"max relative H drift = {drift:.6e}, "
            f"trajectory MSE (i_L, V_C) = {mse:.6e}"
        )
    print(f"Saved energy conservation plot: {ENERGY_PLOT_PATH}")
    print(f"Saved orbit plot: {ORBITS_PLOT_PATH}")
    print(f"Saved Hamiltonian contour plot: {CONTOUR_PLOT_PATH}")


if __name__ == "__main__":
    main()
