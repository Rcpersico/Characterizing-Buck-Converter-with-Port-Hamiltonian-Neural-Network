"""Diagnostics for the trained damped RLC PHNN."""

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

from generate_rlc_data import PARAMS, load_rlc_dataset
from phnn import RLCBuckPHNN
from simulator import phi_q_to_iL_vC


DATA_PATH = ROOT / "data" / "rlc_no_input.npz"
MODEL_PATH = ROOT / "models" / "rlc_phnn.pt"
ENERGY_PLOT_PATH = ROOT / "tests" / "rlc_phnn_energy_decay.png"
ORBITS_PLOT_PATH = ROOT / "tests" / "rlc_phnn_spiral_orbits.png"
DT = 10e-6
SEED = 123


def load_model(device: torch.device) -> tuple[RLCBuckPHNN, dict]:
    """Load the trained RLC PHNN checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = RLCBuckPHNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def phnn_rhs(model: RLCBuckPHNN, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Evaluate the learned vector field at one state."""
    x_tensor = torch.tensor(x[None, :], dtype=torch.float32, device=device)
    dxdt = model(x_tensor, create_graph=False)
    return dxdt.detach().cpu().numpy()[0]


def rk4_integrate(
    model: RLCBuckPHNN,
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
    model: RLCBuckPHNN,
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate ``H_theta`` over a state array."""
    x = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        h = model.hamiltonian(x)
    return h.detach().cpu().numpy().reshape(-1)


def select_validation_trajectories(val_indices: np.ndarray) -> list[int]:
    """Pick three reproducible validation trajectories."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(val_indices, size=3, replace=False)
    return [int(i) for i in chosen]


def plot_energy(t: np.ndarray, h_values: list[np.ndarray], out_path: Path) -> None:
    """Plot learned Hamiltonian values along predicted damped trajectories."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, h in enumerate(h_values, start=1):
        ax.plot(t * 1e3, h, label=f"trajectory {idx}")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Learned H_theta")
    ax.set_title("RLC PHNN Energy Decay")
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
    """Plot predicted and ground-truth spirals in physical coordinates."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for idx, (pred_states, gt, color) in enumerate(zip(predicted, truth, colors), start=1):
        pred_i_l, pred_v_c = phi_q_to_iL_vC(
            pred_states[:, 0],
            pred_states[:, 1],
            PARAMS["L"],
            PARAMS["C"],
        )
        ax.plot(gt["i_L"], gt["V_C"], color=color, linestyle="--", alpha=0.7, label=f"truth {idx}")
        ax.plot(pred_i_l, pred_v_c, color=color, linewidth=1.5, label=f"PHNN {idx}")
        ax.scatter(pred_i_l[0], pred_v_c[0], color=color, s=16, marker="o")
        ax.scatter(pred_i_l[-1], pred_v_c[-1], color=color, s=16, marker="x")
    ax.set_xlabel("i_L [A]")
    ax.set_ylabel("V_C [V]")
    ax.set_title("Damped RLC Phase-Space Spirals")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run Stage 2 diagnostics and assertions."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(device)
    trajectories = load_rlc_dataset(DATA_PATH)
    val_indices = np.asarray(checkpoint["val_indices"], dtype=int)
    chosen_indices = select_validation_trajectories(val_indices)
    chosen_truth = [trajectories[i] for i in chosen_indices]

    predicted_states: list[np.ndarray] = []
    h_values: list[np.ndarray] = []
    max_energy_increases: list[float] = []
    final_energy_ratios: list[float] = []
    trajectory_mses: list[float] = []

    for gt in chosen_truth:
        t = gt["t"]
        x0 = np.array([gt["Phi"][0], gt["q"][0]], dtype=np.float32)
        pred = rk4_integrate(model, x0, t, device)
        predicted_states.append(pred)

        h = evaluate_hamiltonian(model, pred, device)
        h_values.append(h)
        max_energy_increases.append(float(np.max(np.diff(h))))
        final_energy_ratios.append(float(h[-1] / max(h[0], 1e-12)))

        pred_i_l, pred_v_c = phi_q_to_iL_vC(pred[:, 0], pred[:, 1], PARAMS["L"], PARAMS["C"])
        gt_physical = np.column_stack((gt["i_L"], gt["V_C"]))
        pred_physical = np.column_stack((pred_i_l, pred_v_c))
        trajectory_mses.append(float(np.mean((pred_physical - gt_physical) ** 2)))

    plot_energy(chosen_truth[0]["t"], h_values, ENERGY_PLOT_PATH)
    plot_orbits(predicted_states, chosen_truth, ORBITS_PLOT_PATH)

    learned = model.learned_physical_parameters()
    r11_error = abs(learned["R_11"] - PARAMS["R"])

    print(f"Final training loss: {checkpoint['final_train_loss']:.6e}")
    print(f"Final validation loss: {checkpoint['final_val_loss']:.6e}")
    print(
        "Learned parameters: "
        f"L={learned['L']:.8e} H, C={learned['C']:.8e} F, "
        f"R11={learned['R_11']:.8e} Ohm, "
        f"R12={learned['R_12']:.8e}, R22={learned['R_22']:.8e}"
    )
    print(f"True R: {PARAMS['R']:.8e} Ohm, |R11 - R|: {r11_error:.8e}")
    for idx, (traj_idx, increase, ratio, mse) in enumerate(
        zip(chosen_indices, max_energy_increases, final_energy_ratios, trajectory_mses),
        start=1,
    ):
        print(
            f"Trajectory {idx} (dataset index {traj_idx}): "
            f"max Delta H = {increase:.6e}, "
            f"H_final/H_initial = {ratio:.6e}, "
            f"trajectory MSE (i_L, V_C) = {mse:.6e}"
        )
    print(f"Saved energy decay plot: {ENERGY_PLOT_PATH}")
    print(f"Saved spiral orbit plot: {ORBITS_PLOT_PATH}")

    assert r11_error < 2e-2, f"R11 is not close to true R: {learned['R_11']}"
    assert learned["R_22"] < 1e-4, f"R22 should be near zero: {learned['R_22']}"
    assert max(max_energy_increases) <= 1e-8, "Learned H should be nonincreasing."
    assert max(final_energy_ratios) < 0.2, "Damped trajectories should spiral inward."


if __name__ == "__main__":
    main()
