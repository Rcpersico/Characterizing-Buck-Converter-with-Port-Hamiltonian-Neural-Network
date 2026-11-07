"""
Train VRMBuckPHNN on the 12V->1V VRM sweep dataset.

Regime convention (synchronous buck, no freewheeling diode):
  - Unidirectional: i_L >= 0  (forward conduction throughout)
  - Bidirectional:  i_L <  0  (reverse current through synchronous MOSFET)

Key differences from train_bidir_buck.py (100uH / 100kHz model):
  - L_TRUE = 440e-9,  C_TRUE = 200e-6  (VRM power stage)
  - DT_TRAIN = 4.69e-6  (cycle-averaged sample spacing at 1.67 MHz)
  - MULTISTEP_N = 64  (~300 us window, covers ~5 LC periods at f_LC~17 kHz)
  - Model: VRMBuckPHNN from phnn_vrm.py  (V_typical=1V, I_typical=10A)
  - RMSE_V_TOL = 0.10 V  (tighter: V_out ~ 1V, so 0.5V would be 50% error)
  - RMSE_I_TOL = 2.0 A   (looser: large transients during alpha/load steps)

Usage (from bidir_buck_model/ as working directory):
    conda run -n nilm-gpu python src/train_vrm_buck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import load_dataset
from phnn_vrm import VRMBuckPHNN


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "plecs_vrm_train.npz"

L_TRUE = 440e-9
C_TRUE = 200e-6

SEED   = 42
EPOCHS = 12000
LR     = 5e-4
TRAIN_FRACTION = 0.8
DT_TRAIN       = 4.69e-6   # matches dataset dt
MULTISTEP_N    = 32        # ~150 us = ~2.5 LC periods at f_LC ~ 17 kHz
                           # was 64: too many backprop steps through a fast oscillator
                           # caused gradient explosion that destroyed the model in epoch 1
EARLY_STOPPING_PATIENCE = 1500
SCHEDULER_PATIENCE      = 200
SCHEDULER_FACTOR        = 0.5
N_TEST = 6

RMSE_I_TOL = 2.0   # A  -- loose: large alpha-step transients in dataset
RMSE_V_TOL = 0.10  # V  -- tighter than bidir model: V_out ~ 1 V

BIDIR_I_THRESHOLD = -0.05  # A
# VRM: "bidirectional" windows are alpha-step transients where i_L briefly goes
# negative -- NOT a separate steady-state regime to upweight.  Equal weighting
# avoids doubling the gradient contribution from the worst-gradient windows.
BIDIR_LOSS_WEIGHT = 1.0

# Clip gradient norm before each optimizer step.  Without clipping, one batch
# containing a 30A transient window drives a gradient of norm >> 1000, making
# the LR=5e-4 update catastrophically large.
GRAD_CLIP = 1.0


def stack_multistep_rollout(
    trajectories: list[dict],
    n_steps: int = MULTISTEP_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0_list, u_list, xn_list, dt_list, bidir_list = [], [], [], [], []
    for td in trajectories:
        iL    = td["i_L"].astype(np.float32)
        VC    = td["V_C"].astype(np.float32)
        alpha = td["alpha"].astype(np.float32)
        Vin   = td["V_in"].astype(np.float32)
        io    = td["i_o"].astype(np.float32)
        t     = td["t"]

        x  = np.stack([iL, VC], axis=1)
        u  = np.stack([alpha * Vin, io], axis=1)
        dt = float(np.median(np.diff(t)))

        stride = max(1, n_steps // 2)
        for start in range(0, len(t) - n_steps - 1, stride):
            window_iL = iL[start : start + n_steps]
            is_bidir  = bool(np.any(window_iL < BIDIR_I_THRESHOLD))
            x0_list.append(x[start])
            u_list.append(u[start : start + n_steps])
            xn_list.append(x[start + 1 : start + n_steps + 1])
            dt_list.append(dt)
            bidir_list.append(is_bidir)

    if not x0_list:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), bool),
        )
    return (
        np.array(x0_list, np.float32),
        np.array(u_list,  np.float32),
        np.array(xn_list, np.float32),
        np.array(dt_list, np.float32),
        np.array(bidir_list, bool),
    )


def split_by_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    return idx[:n_train], idx[n_train:]


def multistep_rk4(
    model: VRMBuckPHNN,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    create_graph: bool = True,
) -> torch.Tensor:
    dt_t  = torch.tensor(float(dt), device=x0.device, dtype=x0.dtype)
    preds: list[torch.Tensor] = []
    x = x0
    for k in range(u_seq.shape[1]):
        u_k = u_seq[:, k, :]
        k1 = model(x, u_k, create_graph=create_graph)
        k2 = model(x + 0.5 * dt_t * k1, u_k, create_graph=create_graph)
        k3 = model(x + 0.5 * dt_t * k2, u_k, create_graph=create_graph)
        k4 = model(x + dt_t * k3,        u_k, create_graph=create_graph)
        x  = x + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not create_graph:
            x = x.detach()
        preds.append(x)
    return torch.stack(preds, dim=1)


def regime_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_bidir: torch.Tensor,
    scale: torch.Tensor,
    bidir_weight: float = BIDIR_LOSS_WEIGHT,
) -> torch.Tensor:
    sq_err     = ((pred - target) / scale).pow(2)
    per_window = sq_err.mean(dim=[1, 2])
    weights    = torch.where(is_bidir, bidir_weight, 1.0)
    return (weights * per_window).mean()


def _rhs(model: VRMBuckPHNN, x: np.ndarray, u: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def rk4_rollout(
    model: VRMBuckPHNN,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k  = u_eff[k]
        k1 = _rhs(model, states[k],               u_k, device)
        k2 = _rhs(model, states[k] + 0.5*dt_k*k1, u_k, device)
        k3 = _rhs(model, states[k] + 0.5*dt_k*k2, u_k, device)
        k4 = _rhs(model, states[k] + dt_k*k3,     u_k, device)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return states


def _regime_diag_tensors(
    model: VRMBuckPHNN,
    x0_va: torch.Tensor,
    u_va: torch.Tensor,
    is_bidir_va: torch.Tensor,
    device: torch.device,
) -> dict:
    unidir_mask = ~is_bidir_va
    bidir_mask  =  is_bidir_va
    out = {k: float("nan") for k in
           ("H_NN_norm_unidir", "H_NN_norm_bidir", "dG_norm_unidir", "dG_norm_bidir")}
    for label, mask in [("unidir", unidir_mask), ("bidir", bidir_mask)]:
        if mask.sum() == 0:
            continue
        x_sel = x0_va[mask][:64]
        u_sel = u_va[mask, 0, :][:64]
        diag  = model.get_regime_diagnostics(x_sel, u_sel, x_sel, u_sel)
        out[f"H_NN_norm_{label}"] = diag[f"H_NN_norm_{label}"]
        out[f"dG_norm_{label}"]   = diag[f"dG_norm_{label}"]
    return out


def train(
    trajectories: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_path: Path,
    loss_plot_path: Path,
    traj_plot_path: Path,
    wandb_run=None,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_trajs = [trajectories[i] for i in train_idx]
    va_trajs = [trajectories[i] for i in val_idx]

    x0_tr, u_tr, xn_tr, dt_tr, bidir_tr = stack_multistep_rollout(tr_trajs, MULTISTEP_N)
    x0_va, u_va, xn_va, dt_va, bidir_va = stack_multistep_rollout(va_trajs, MULTISTEP_N)

    if x0_tr.size == 0 or x0_va.size == 0:
        raise ValueError(
            f"Need trajectories longer than {MULTISTEP_N + 1} samples for training."
        )

    all_x    = np.vstack((x0_tr, xn_tr.reshape(-1, 2)))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)
    data_dt  = float(np.median(np.concatenate((dt_tr, dt_va))))

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_bidir_tr  = int(bidir_tr.sum())
    n_bidir_va  = int(bidir_va.sum())
    print(f"Device              : {device}")
    print(f"Trajectories total  : {len(trajectories)}")
    print(f"Train windows       : {x0_tr.shape[0]} x {MULTISTEP_N} steps  "
          f"(bidirectional: {n_bidir_tr}, unidirectional: {x0_tr.shape[0] - n_bidir_tr})")
    print(f"Val   windows       : {x0_va.shape[0]} x {MULTISTEP_N} steps  "
          f"(bidirectional: {n_bidir_va}, unidirectional: {x0_va.shape[0] - n_bidir_va})")
    print(f"Training dt         : {DT_TRAIN * 1e6:.2f} us")
    print(f"Dataset dt median   : {data_dt * 1e6:.2f} us")
    print(f"State scale [A, V]  : {scale_np}")

    sc = torch.tensor(scale_np, device=device)

    tx0_tr  = torch.tensor(x0_tr,  device=device)
    tu_tr   = torch.tensor(u_tr,   device=device)
    txn_tr  = torch.tensor(xn_tr,  device=device)
    tbidir_tr = torch.tensor(bidir_tr, device=device)

    tx0_va  = torch.tensor(x0_va,  device=device)
    tu_va   = torch.tensor(u_va,   device=device)
    txn_va  = torch.tensor(xn_va,  device=device)
    tbidir_va = torch.tensor(bidir_va, device=device)

    model     = VRMBuckPHNN(L=L_TRUE, C=C_TRUE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val   = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    no_improve = 0

    header = (
        f"{'Epoch':>6}  {'train':>12}  {'val':>12}  {'lr':>8}  "
        f"{'R11':>8}  {'R22':>8}  "
        f"{'|H|uni':>10}  {'|H|bi':>10}  "
        f"{'|dG|uni':>10}  {'|dG|bi':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = multistep_rk4(model, tx0_tr, tu_tr, DT_TRAIN, create_graph=True)
        loss    = regime_weighted_loss(pred_tr, txn_tr, tbidir_tr, sc, BIDIR_LOSS_WEIGHT)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_va  = multistep_rk4(model, tx0_va, tu_va, DT_TRAIN, create_graph=False)
            val_loss = regime_weighted_loss(pred_va, txn_va, tbidir_va, sc, BIDIR_LOSS_WEIGHT)

        tr_f = float(loss.detach().cpu())
        va_f = float(val_loss.detach().cpu())
        train_losses.append(tr_f)
        val_losses.append(va_f)
        scheduler.step(va_f)

        if va_f < best_val:
            best_val   = va_f
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(
                {"model_state_dict": best_state, "epoch": epoch, "val_loss": best_val},
                model_path,
            )
        else:
            no_improve += 1

        if epoch == 1 or epoch % 100 == 0:
            p      = model.get_params()
            lr_now = optimizer.param_groups[0]["lr"]
            model.eval()
            with torch.no_grad():
                rdiag = _regime_diag_tensors(model, tx0_va, tu_va, tbidir_va, device)
            print(
                f"{epoch:6d}  {tr_f:12.6e}  {va_f:12.6e}  {lr_now:8.2e}  "
                f"{p['R_11']:8.5f}  {p['R_22']:8.5f}  "
                f"{rdiag['H_NN_norm_unidir']:10.4e}  {rdiag['H_NN_norm_bidir']:10.4e}  "
                f"{rdiag['dG_norm_unidir']:10.4e}  {rdiag['dG_norm_bidir']:10.4e}"
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
    print(f"\nBest epoch  : {best_epoch}   best val loss : {best_val:.6e}")
    print(
        f"R_theta at rep. pt: R11={p['R_11']:.5f}  R12={p['R_12']:.5f}  R22={p['R_22']:.5f}"
    )
    print(f"delta_G_norm={p['delta_G_norm']:.4e}  H_NN_norm={p['H_NN_norm']:.4e}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "state_scale":      scale_np,
        "train_indices":    train_idx,
        "val_indices":      val_idx,
        "train_losses":     np.array(train_losses, np.float64),
        "val_losses":       np.array(val_losses,   np.float64),
        "best_val_loss":    best_val,
        "best_epoch":       best_epoch,
        "dt_train":         DT_TRAIN,
        "multistep_n":      MULTISTEP_N,
        "fixed_params":     {"L": L_TRUE, "C": C_TRUE},
        "learned_params":   p,
        "bidir_loss_weight": BIDIR_LOSS_WEIGHT,
    }
    torch.save(ckpt, model_path)
    print(f"Saved model : {model_path}")

    # Loss curve
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_r = range(1, len(train_losses) + 1)
    ax.semilogy(epochs_r, train_losses, label="train",      alpha=0.8)
    ax.semilogy(epochs_r, val_losses,   label="validation", alpha=0.8)
    ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5, label=f"best ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Regime-weighted {MULTISTEP_N}-step rollout MSE")
    ax.set_title("VRM Buck PHNN Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot : {loss_plot_path}")

    return ckpt


def evaluate(
    model: VRMBuckPHNN,
    ckpt: dict,
    trajectories: list[dict],
    device: torch.device,
    traj_plot_path: Path,
) -> None:
    model.eval()
    val_idx = np.asarray(ckpt["val_indices"], dtype=int)
    rng     = np.random.default_rng(SEED)
    test_idx = (
        rng.choice(val_idx, size=N_TEST, replace=False)
        if len(val_idx) >= N_TEST else val_idx
    )

    print("\n" + "=" * 70)
    print("Trajectory rollout on held-out validation trajectories")
    print("=" * 70)
    p = model.get_params()
    print(f"  Fixed L          : {p['L']*1e9:.0f} nH")
    print(f"  Fixed C          : {p['C']*1e6:.0f} uF")
    print(f"  R11 (rep pt)     : {p['R_11']:.5f} Ohm")
    print(f"  R22 (rep pt)     : {p['R_22']:.6f} Ohm")
    print(f"  delta_G_norm     : {p['delta_G_norm']:.4e}")
    print(f"  H_NN_norm        : {p['H_NN_norm']:.4e}")
    print()

    mse_iL: list[float] = []
    mse_vC: list[float] = []
    preds:  list[np.ndarray] = []
    gts:    list[dict] = []

    for idx in test_idx:
        gt     = trajectories[idx]
        t      = gt["t"]
        x0     = np.array([gt["i_L"][0], gt["V_C"][0]], np.float32)
        u      = np.column_stack((gt["alpha"] * gt["V_in"], gt["i_o"])).astype(np.float32)
        regime = float(gt.get("regime", np.array([0.0]))[0])

        pred = rk4_rollout(model, x0, t, u, device)
        preds.append(pred)
        gts.append(gt)

        e_iL = float(np.mean((pred[:, 0] - gt["i_L"]) ** 2))
        e_vC = float(np.mean((pred[:, 1] - gt["V_C"]) ** 2))
        mse_iL.append(e_iL)
        mse_vC.append(e_vC)

        alpha_r = f"{gt['alpha'].min():.2f}-{gt['alpha'].max():.2f}"
        print(
            f"  traj {idx:3d}: RMSE i_L={e_iL**0.5:.4e} A  "
            f"RMSE V_C={e_vC**0.5:.4e} V  "
            f"alpha=[{alpha_r}]  regime={regime:.3f}"
        )

    # Trajectory plots
    traj_plot_path.parent.mkdir(parents=True, exist_ok=True)
    n_plots = len(test_idx)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes[np.newaxis, :]
    for row, (gt, pred, idx) in enumerate(zip(gts, preds, test_idx)):
        t_ms   = gt["t"] * 1e3
        regime = float(gt.get("regime", np.array([0.0]))[0])
        label  = "bidir" if regime > 0.05 else "unidir"
        title  = f"Traj {idx}  [{label} regime={regime:.2f}]  alpha=[{gt['alpha'].min():.2f}-{gt['alpha'].max():.2f}]"

        axes[row, 0].plot(t_ms, gt["i_L"],    "k--", lw=1.0, label="PLECS")
        axes[row, 0].plot(t_ms, pred[:, 0],   color="tab:blue",   lw=1.2, label="PHNN")
        axes[row, 0].axhline(0, color="gray", lw=0.5, ls=":")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(title)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, gt["V_C"],    "k--", lw=1.0, label="PLECS")
        axes[row, 1].plot(t_ms, pred[:, 1],   color="tab:orange", lw=1.2, label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(title)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle("VRM Buck PHNN -- Predicted vs PLECS", fontsize=11)
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
    failures = []
    if mean_rmse_iL >= RMSE_I_TOL:
        failures.append(f"Mean RMSE i_L={mean_rmse_iL:.4e} >= {RMSE_I_TOL}")
    if mean_rmse_vC >= RMSE_V_TOL:
        failures.append(f"Mean RMSE V_C={mean_rmse_vC:.4e} >= {RMSE_V_TOL}")
    if failures:
        print("\n  FAILED:")
        for msg in failures:
            print(f"    - {msg}")
    else:
        print("\n  Result: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    MODEL_PATH  = ROOT / "models" / "vrm_buck_phnn.pt"
    LOSS_PLOT   = ROOT / "tests"  / "vrm_buck_phnn_loss.png"
    TRAJ_PLOT   = ROOT / "tests"  / "vrm_buck_phnn_traj.png"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Generate it first with:\n"
            "  conda run -n nilm-gpu python src/generate_plecs_vrm_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    regimes  = [float(t.get("regime", np.array([0.0]))[0]) for t in trajectories]
    n_bidir  = sum(1 for r in regimes if r > 0.05)
    print(f"  Bidirectional (regime>5%): {n_bidir}  Unidirectional: {len(trajectories)-n_bidir}")
    print(f"  Mean regime fraction     : {np.mean(regimes):.3f}\n")

    train_idx, val_idx = split_by_trajectory(len(trajectories), SEED)
    print(f"Train: {len(train_idx)} trajectories  |  Val: {len(val_idx)} trajectories\n")

    ckpt = train(trajectories, train_idx, val_idx, MODEL_PATH, LOSS_PLOT, TRAJ_PLOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VRMBuckPHNN(L=L_TRUE, C=C_TRUE).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device, TRAJ_PLOT)
