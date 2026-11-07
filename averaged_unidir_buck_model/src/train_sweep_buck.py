"""
Train SimulationBuckPHNN on the operating-point-rich sweep dataset.

Two modes:
  default      Full PHNN — learns R_L, R_C, plus the H_NN correction.
               Saved to models/sweep_buck_phnn.pt
  --no-nn      Physics-only baseline — H_NN frozen to zero, only R_L and
               R_C are updated.  Saved to models/sweep_buck_physics_only.pt

Usage:
    python src/train_sweep_buck.py               # full PHNN
    python src/train_sweep_buck.py --no-nn       # physics-only baseline
"""

from __future__ import annotations

import argparse
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
from phnn import SimulationBuckPHNN


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "plecs_sweep_train.npz"

# Fixed — known from design (sweep dataset uses these values)
L_TRUE = 100e-6
C_TRUE = 100e-6

INIT_R_L = 0.20
INIT_R_C = 0.05

SEED = 42
EPOCHS = 8000
LR = 1e-3
TRAIN_FRACTION = 0.8
DT_TRAIN = 50e-6
MULTISTEP_N = 40
EARLY_STOPPING_PATIENCE = 1000
SCHEDULER_PATIENCE = 150
SCHEDULER_FACTOR = 0.5
N_TEST = 4

RMSE_I_TOL = 0.20  # A — looser than sim-only tolerance given real PLECS noise
RMSE_V_TOL = 0.30  # V


# Representative operating points for H_NN instrumentation
_OP_LOW = {
    "label": "low-alpha (iL=1A, VC=3V)",
    "x_phys": [1.0, 3.0],
    "u_eff": [0.25 * 12.0, 1.0],  # alpha=0.25, Vin=12V, io=1A
}
_OP_HIGH = {
    "label": "high-alpha (iL=3A, VC=9V)",
    "x_phys": [3.0, 9.0],
    "u_eff": [0.75 * 12.0, 3.0],  # alpha=0.75, Vin=12V, io=3A
}


def _h_nn_at(
    model: SimulationBuckPHNN,
    x_phys: list[float],
    u_eff: list[float],
    device: torch.device,
) -> float:
    """Evaluate H_NN correction at a single operating point."""
    L = float(model.L)
    C = float(model.C)
    x_ph = torch.tensor(
        [[L * x_phys[0], C * x_phys[1]]], dtype=torch.float32, device=device
    )
    u_t = torch.tensor([u_eff], dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(model._h_nn(x_ph, u_t).abs())


def stack_multistep_rollout(
    trajectories: list[dict],
    n_steps: int = MULTISTEP_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack (x0, u_{0:N}, x_{1:N+1}, dt) windows for multi-step rollout loss."""
    x0_list, u_list, xn_list, dt_list = [], [], [], []
    for td in trajectories:
        iL = td["i_L"].astype(np.float32)
        VC = td["V_C"].astype(np.float32)
        alpha = td["alpha"].astype(np.float32)
        Vin = td["V_in"].astype(np.float32)
        io = td["i_o"].astype(np.float32)
        t = td["t"]

        x = np.stack([iL, VC], axis=1)
        u = np.stack([alpha * Vin, io], axis=1)
        dt = float(np.median(np.diff(t)))

        stride = max(1, n_steps // 2)
        for start in range(0, len(t) - n_steps - 1, stride):
            x0_list.append(x[start])
            u_list.append(u[start : start + n_steps])
            xn_list.append(x[start + 1 : start + n_steps + 1])
            dt_list.append(dt)

    if not x0_list:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0,), np.float32),
        )

    return (
        np.array(x0_list, np.float32),
        np.array(u_list, np.float32),
        np.array(xn_list, np.float32),
        np.array(dt_list, np.float32),
    )


def split_by_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    return idx[:n_train], idx[n_train:]


def multistep_rk4(
    model: SimulationBuckPHNN,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    create_graph: bool = True,
) -> torch.Tensor:
    """Roll out N RK4 steps from x0 using the input sequence u_seq."""
    dt_t = torch.tensor(float(dt), device=x0.device, dtype=x0.dtype)
    preds: list[torch.Tensor] = []
    x = x0
    for k in range(u_seq.shape[1]):
        u_k = u_seq[:, k, :]
        k1 = model(x, u_k, create_graph=create_graph)
        k2 = model(x + 0.5 * dt_t * k1, u_k, create_graph=create_graph)
        k3 = model(x + 0.5 * dt_t * k2, u_k, create_graph=create_graph)
        k4 = model(x + dt_t * k3, u_k, create_graph=create_graph)
        x = x + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not create_graph:
            x = x.detach()
        preds.append(x)
    return torch.stack(preds, dim=1)


def _rhs(
    model: SimulationBuckPHNN, x: np.ndarray, u: np.ndarray, device: torch.device
) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout(
    model: SimulationBuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Full trajectory RK4 rollout from x0 = [i_L0, V_C0]."""
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
    no_nn: bool,
    model_path: Path,
    loss_plot_path: Path,
    traj_plot_path: Path,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_trajs = [trajectories[i] for i in train_idx]
    va_trajs = [trajectories[i] for i in val_idx]

    x0_tr, u_tr, xn_tr, dt_tr = stack_multistep_rollout(tr_trajs, MULTISTEP_N)
    x0_va, u_va, xn_va, dt_va = stack_multistep_rollout(va_trajs, MULTISTEP_N)
    if x0_tr.size == 0 or x0_va.size == 0:
        raise ValueError(
            f"Need trajectories longer than {MULTISTEP_N + 1} samples for "
            "multi-step rollout training."
        )

    all_x = np.vstack((x0_tr, xn_tr.reshape(-1, 2)))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)
    data_dt = float(np.median(np.concatenate((dt_tr, dt_va))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_tag = "physics-only (--no-nn)" if no_nn else "full PHNN"
    print(f"Mode               : {mode_tag}")
    print(f"Device             : {device}")
    print(f"Trajectories total : {len(trajectories)}")
    print(f"Train windows      : {x0_tr.shape[0]} x {MULTISTEP_N} steps")
    print(f"Val   windows      : {x0_va.shape[0]} x {MULTISTEP_N} steps")
    print(f"Training dt        : {DT_TRAIN * 1e6:.1f} µs")
    print(f"Dataset dt median  : {data_dt * 1e6:.1f} µs")
    print(f"State scale [A, V] : {scale_np}")

    sc = torch.tensor(scale_np, device=device)
    tx0_tr = torch.tensor(x0_tr, device=device)
    tu_tr = torch.tensor(u_tr, device=device)
    txn_tr = torch.tensor(xn_tr, device=device)
    tx0_va = torch.tensor(x0_va, device=device)
    tu_va = torch.tensor(u_va, device=device)
    txn_va = torch.tensor(xn_va, device=device)

    model = SimulationBuckPHNN(
        L=L_TRUE,
        C=C_TRUE,
        initial_R_L=INIT_R_L,
        initial_R_C=INIT_R_C,
    ).to(device)

    if no_nn:
        for param in model.H_NN.parameters():
            param.data.zero_()
            param.requires_grad_(False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
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

    header = (
        f"{'Epoch':>6}  {'train':>12}  {'val':>12}  {'lr':>8}  "
        f"{'R_L':>8}  {'R_C':>8}  {'|H_NN|lo':>10}  {'|H_NN|hi':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = multistep_rk4(model, tx0_tr, tu_tr, DT_TRAIN, create_graph=True)
        loss = mse(pred_tr / sc, txn_tr / sc)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
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
            h_lo = _h_nn_at(model, _OP_LOW["x_phys"], _OP_LOW["u_eff"], device)
            h_hi = _h_nn_at(model, _OP_HIGH["x_phys"], _OP_HIGH["u_eff"], device)
            print(
                f"{epoch:6d}  {tr_f:12.6e}  {va_f:12.6e}  {lr_now:8.2e}  "
                f"{p['R_L']:8.5f}  {p['R_C']:8.5f}  {h_lo:10.4e}  {h_hi:10.4e}"
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
    h_lo = _h_nn_at(model, _OP_LOW["x_phys"], _OP_LOW["u_eff"], device)
    h_hi = _h_nn_at(model, _OP_HIGH["x_phys"], _OP_HIGH["u_eff"], device)

    print(f"\nBest epoch : {best_epoch}   best val loss : {best_val:.6e}")
    print(
        f"Learned R_L={p['R_L']:.5f} Ω  R_C={p['R_C']:.5f} Ω  "
        f"H_NN_norm={p['H_NN_norm']:.4e}"
    )
    print(
        f"H_NN at {_OP_LOW['label']}  : {h_lo:.4e}")
    print(
        f"H_NN at {_OP_HIGH['label']} : {h_hi:.4e}"
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "state_scale": scale_np,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_losses": np.array(train_losses, np.float64),
        "val_losses": np.array(val_losses, np.float64),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "dt_train": DT_TRAIN,
        "multistep_n": MULTISTEP_N,
        "fixed_params": {"L": L_TRUE, "C": C_TRUE},
        "learned_params": p,
        "no_nn": no_nn,
    }
    torch.save(ckpt, model_path)
    print(f"Saved model : {model_path}")

    # Loss curve
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="train", alpha=0.8)
    ax.semilogy(epochs, val_losses, label="validation", alpha=0.8)
    ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5, label=f"best ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Normalised {MULTISTEP_N}-step rollout MSE")
    ax.set_title(f"Sweep PHNN Training Loss ({mode_tag})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {loss_plot_path}")

    return ckpt


def evaluate(
    model: SimulationBuckPHNN,
    ckpt: dict,
    trajectories: list[dict],
    device: torch.device,
    traj_plot_path: Path,
) -> None:
    model.eval()
    val_idx = np.asarray(ckpt["val_indices"], dtype=int)
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(val_idx, size=min(N_TEST, len(val_idx)), replace=False)

    print("\n" + "=" * 70)
    print("Trajectory rollout on held-out validation trajectories")
    print("=" * 70)
    p = model.get_params()
    print(f"  Fixed L     : {p['L'] * 1e6:.0f} µH")
    print(f"  Fixed C     : {p['C'] * 1e6:.0f} µF")
    print(f"  Learned R_L : {p['R_L']:.5f} Ω  (true: 0.10000 Ω)")
    print(f"  Learned R_C : {p['R_C']:.5f} Ω")
    print(f"  H_NN_norm   : {p['H_NN_norm']:.4e}")
    print()

    mse_iL: list[float] = []
    mse_vC: list[float] = []
    preds: list[np.ndarray] = []
    gts: list[dict] = []

    for idx in test_idx:
        gt = trajectories[idx]
        t = gt["t"]
        x0 = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
        u = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)

        pred = rk4_rollout(model, x0, t, u, device)
        preds.append(pred)
        gts.append(gt)

        e_iL = float(np.mean((pred[:, 0] - gt["i_L"]) ** 2))
        e_vC = float(np.mean((pred[:, 1] - gt["V_C"]) ** 2))
        mse_iL.append(e_iL)
        mse_vC.append(e_vC)

        alpha_range = f"{gt['alpha'].min():.2f}–{gt['alpha'].max():.2f}"
        print(
            f"  traj {idx:3d}: RMSE i_L={e_iL**0.5:.4e} A  "
            f"RMSE V_C={e_vC**0.5:.4e} V  alpha∈[{alpha_range}]"
        )

    # Trajectory plots
    traj_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(test_idx), 2, figsize=(12, 3 * len(test_idx)))
    if len(test_idx) == 1:
        axes = axes[np.newaxis, :]
    for row, (gt, pred, idx) in enumerate(zip(gts, preds, test_idx)):
        t_ms = gt["t"] * 1e3
        alpha_str = f"{gt['alpha'].min():.2f}–{gt['alpha'].max():.2f}"
        title = f"Traj {idx}  α∈[{alpha_str}]  V_in={gt['V_in'][0]:.1f}V"
        axes[row, 0].plot(t_ms, gt["i_L"], "k--", lw=1.0, label="PLECS")
        axes[row, 0].plot(t_ms, pred[:, 0], color="tab:blue", lw=1.2, label="PHNN")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"], "k--", lw=1.0, label="PLECS")
        axes[row, 1].plot(t_ms, pred[:, 1], color="tab:orange", lw=1.2, label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    mode_tag = "physics-only" if ckpt["no_nn"] else "full PHNN"
    fig.suptitle(f"Sweep PHNN ({mode_tag}) — Predicted vs PLECS", fontsize=11)
    fig.tight_layout()
    fig.savefig(traj_plot_path, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {traj_plot_path}")

    mean_rmse_iL = float(np.mean([m**0.5 for m in mse_iL]))
    mean_rmse_vC = float(np.mean([m**0.5 for m in mse_vC]))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mean RMSE i_L : {mean_rmse_iL:.4e} A  (tol: {RMSE_I_TOL} A)")
    print(f"  Mean RMSE V_C : {mean_rmse_vC:.4e} V  (tol: {RMSE_V_TOL} V)")

    checks = [
        (mean_rmse_iL < RMSE_I_TOL, f"Mean RMSE i_L={mean_rmse_iL:.4e} >= {RMSE_I_TOL}"),
        (mean_rmse_vC < RMSE_V_TOL, f"Mean RMSE V_C={mean_rmse_vC:.4e} >= {RMSE_V_TOL}"),
    ]
    failures = [msg for ok, msg in checks if not ok]
    if failures:
        print("\n  FAILED assertions:")
        for msg in failures:
            print(f"    - {msg}")
        print("\n  Result: FAILED")
        print("=" * 70)
        raise AssertionError(f"{len(failures)} assertion(s) failed.")

    print("\n  Result: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-nn",
        action="store_true",
        help="Freeze H_NN (trains physics-only baseline: learned R_L, R_C only).",
    )
    args = parser.parse_args()

    if args.no_nn:
        MODEL_PATH = ROOT / "models" / "sweep_buck_physics_only.pt"
        LOSS_PLOT = ROOT / "tests" / "test_images" / "sweep_buck_physics_only_loss.png"
        TRAJ_PLOT = ROOT / "tests" / "test_images" / "sweep_buck_physics_only_traj.png"
    else:
        MODEL_PATH = ROOT / "models" / "sweep_buck_phnn.pt"
        LOSS_PLOT = ROOT / "tests" / "test_images" / "sweep_buck_phnn_loss.png"
        TRAJ_PLOT = ROOT / "tests" / "test_images" / "sweep_buck_phnn_traj.png"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Generate it first with: python src/generate_plecs_sweep_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx, args.no_nn, MODEL_PATH, LOSS_PLOT, TRAJ_PLOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimulationBuckPHNN(L=L_TRUE, C=C_TRUE, initial_R_L=INIT_R_L, initial_R_C=INIT_R_C).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device, TRAJ_PLOT)
