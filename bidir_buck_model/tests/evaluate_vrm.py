"""
VRM PHNN evaluation: held-out PLECS test set + real GPU power profile.

Plots produced:
  vrm_eval_test_trajectories.png  -- i_L(t) and V_C(t): PHNN vs PLECS
  vrm_eval_test_phase.png         -- V_C vs i_L phase portrait: PHNN vs PLECS
  vrm_eval_gpu_trajectory.png     -- i_L(t) and V_C(t) over GPU power profile
  vrm_eval_gpu_phase.png          -- V_C vs i_L phase portrait over GPU profile

Usage (from bidir_buck_model/):
    conda run -n nilm-gpu python tests/evaluate_vrm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_dataset import load_dataset
from phnn_vrm import VRMBuckPHNN

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "vrm_buck_phnn.pt"
TEST_PATH  = ROOT / "data"   / "plecs_vrm_test.npz"
GPU_CSV    = ROOT.parent / "GPU_Profile" / "wandb_export_2025-06-07T02_42_05.439-07_00.csv"
OUT_DIR    = Path(__file__).parent

L_TRUE = 440e-9
C_TRUE = 200e-6
V_IN   = 12.0
DT     = 4.69e-6


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> VRMBuckPHNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first:  conda run -n nilm-gpu python src/train_vrm_buck.py"
        )
    ckpt  = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    model = VRMBuckPHNN(L=L_TRUE, C=C_TRUE).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    p = model.get_params()
    print(f"  Loaded model")
    print(f"  Fixed  L = {p['L']*1e9:.0f} nH    C = {p['C']*1e6:.0f} uF")
    print(f"  Learned R11 = {p['R_11']*1e3:.3f} mOhm")
    print(f"  H_NN_norm = {p['H_NN_norm']:.3e}   delta_G_norm = {p['delta_G_norm']:.3e}")
    return model


def rk4_rollout(model: VRMBuckPHNN, x0: np.ndarray, t: np.ndarray,
                u_eff: np.ndarray, device: torch.device) -> np.ndarray:
    states    = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k  = u_eff[k].astype(np.float32)
        with torch.no_grad():
            def f(x):
                xt = torch.tensor(x[None], dtype=torch.float32, device=device)
                ut = torch.tensor(u_k[None], dtype=torch.float32, device=device)
                return model(xt, ut, create_graph=False).cpu().numpy()[0]
            k1 = f(states[k])
            k2 = f(states[k] + 0.5 * dt_k * k1)
            k3 = f(states[k] + 0.5 * dt_k * k2)
            k4 = f(states[k] + dt_k * k3)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Evaluation 1: held-out PLECS test set
# ---------------------------------------------------------------------------

def evaluate_test_set(model: VRMBuckPHNN, device: torch.device) -> None:
    if not TEST_PATH.exists():
        print(f"  Test dataset not found: {TEST_PATH} -- skipping.")
        return

    trajs = load_dataset(TEST_PATH)
    n     = len(trajs)
    print(f"\n  Loaded {n} test trajectories from {TEST_PATH.name}")

    all_preds, all_rmse_iL, all_rmse_vC = [], [], []
    for td in trajs:
        x0   = np.array([td["i_L"][0], td["V_C"][0]], np.float32)
        u    = np.column_stack((td["alpha"] * td["V_in"], td["i_o"])).astype(np.float32)
        pred = rk4_rollout(model, x0, td["t"], u, device)
        all_preds.append(pred)
        all_rmse_iL.append(float(np.sqrt(np.mean((pred[:, 0] - td["i_L"])**2))))
        all_rmse_vC.append(float(np.sqrt(np.mean((pred[:, 1] - td["V_C"])**2))))
        alpha_r = f"{td['alpha'].min():.2f}-{td['alpha'].max():.2f}"
        print(f"    traj {len(all_preds)-1:2d}: RMSE i_L={all_rmse_iL[-1]:.4f} A  "
              f"RMSE V_C={all_rmse_vC[-1]:.4f} V  alpha=[{alpha_r}]")

    print(f"\n  Mean RMSE i_L : {np.mean(all_rmse_iL):.4f} A")
    print(f"  Mean RMSE V_C : {np.mean(all_rmse_vC):.4f} V")

    n_plot = min(3, n)

    # ── Trajectory plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_plot, 2, figsize=(13, 3.5 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_plot):
        td   = trajs[row]
        pred = all_preds[row]
        t_ms = td["t"] * 1e3

        axes[row, 0].plot(t_ms, td["i_L"],  "k--", lw=1.0, label="PLECS")
        axes[row, 0].plot(t_ms, pred[:, 0], color="tab:blue", lw=1.2, label="PHNN")
        axes[row, 0].axhline(0, color="gray", lw=0.5, ls=":")
        axes[row, 0].set_ylabel("i_L [A]")
        axes[row, 0].set_title(
            f"Traj {row}  alpha=[{td['alpha'].min():.2f}-{td['alpha'].max():.2f}]  "
            f"RMSE={all_rmse_iL[row]:.3f} A"
        )
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t_ms, td["V_C"],  "k--", lw=1.0, label="PLECS")
        axes[row, 1].plot(t_ms, pred[:, 1], color="tab:orange", lw=1.2, label="PHNN")
        axes[row, 1].set_ylabel("V_C [V]")
        axes[row, 1].set_title(f"RMSE V_C={all_rmse_vC[row]:.4f} V")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle("VRM PHNN vs PLECS — Held-out Test Set (trajectories)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "vrm_eval_test_trajectories.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\n  Saved: {out}")

    # ── Phase plots: V_C vs i_L ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 5))
    if n_plot == 1:
        axes = [axes]

    for col in range(n_plot):
        td   = trajs[col]
        pred = all_preds[col]
        ax   = axes[col]

        # Colour by time so you can see direction of travel
        t_norm = np.linspace(0, 1, len(td["t"]))

        sc_plecs = ax.scatter(td["i_L"], td["V_C"], c=t_norm, cmap="Blues",
                              s=4, alpha=0.7, label="PLECS")
        sc_phnn  = ax.scatter(pred[:, 0], pred[:, 1], c=t_norm, cmap="Oranges",
                              s=4, alpha=0.7, label="PHNN")

        # Mark start and end
        ax.plot(td["i_L"][0],  td["V_C"][0],  "b^", ms=8, zorder=5)
        ax.plot(pred[0, 0],    pred[0, 1],    "r^", ms=8, zorder=5)
        ax.plot(td["i_L"][-1], td["V_C"][-1], "bs", ms=8, zorder=5)
        ax.plot(pred[-1, 0],   pred[-1, 1],   "rs", ms=8, zorder=5)

        ax.set_xlabel("i_L [A]")
        ax.set_ylabel("V_C [V]")
        ax.set_title(
            f"Traj {col}  alpha=[{td['alpha'].min():.2f}-{td['alpha'].max():.2f}]\n"
            f"triangle=start  square=end"
        )
        ax.legend(handles=[
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',  ms=8, label='PLECS'),
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='darkorange', ms=8, label='PHNN'),
        ], fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VRM PHNN vs PLECS — Phase Portrait (V_C vs i_L)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "vrm_eval_test_phase.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Evaluation 2: real GPU power profile
# ---------------------------------------------------------------------------

def load_gpu_profile(dt: float = DT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, clean, interpolate and scale the GPU power CSV to VRM load current.

    Returns (t, alpha_arr, io_arr) on the PHNN 4.69-us grid.
    """
    if not GPU_CSV.exists():
        raise FileNotFoundError(f"GPU CSV not found: {GPU_CSV}")

    df = pd.read_csv(GPU_CSV)
    df['time_s'] = pd.to_numeric(df['Relative Time (Process)'], errors='coerce')
    df = df.dropna(subset=['time_s'])

    # Melt all four workload columns into one combined power series
    COLS = {
        'SoftUpdateBigTau': 'SoftUpdateBigTau - system/gpu.0.powerWatts',
        'HardUpdateBigC'  : 'HardUpdateBigC - system/gpu.0.powerWatts',
        'HardUpdate'      : 'HardUpdate - system/gpu.0.powerWatts',
        'DQN'             : 'DQN - system/gpu.0.powerWatts',
    }
    frames = []
    for label, col in COLS.items():
        sub = df[['time_s', col]].dropna(subset=[col]).copy()
        sub = sub.rename(columns={col: 'power_W'})
        frames.append(sub)

    combined = pd.concat(frames).sort_values('time_s').reset_index(drop=True)

    # Use only the first 90 minutes to keep simulation manageable
    MAX_S = 90 * 60
    combined = combined[combined['time_s'] <= MAX_S]
    t_raw = combined['time_s'].values
    p_raw = combined['power_W'].values

    # Dense time grid at PHNN sample rate
    t_dense = np.arange(t_raw[0], t_raw[-1], dt)

    # Linear interpolation from 15-s samples onto microsecond grid
    p_dense = np.interp(t_dense, t_raw, p_raw)

    # Scale power to VRM load-current range [2, 14] A
    P_min, P_max = p_raw.min(), p_raw.max()
    I_min, I_max = 2.0, 14.0
    io_arr = I_min + (p_dense - P_min) / (P_max - P_min) * (I_max - I_min)

    # Fixed duty cycle: target V_out = 1 V from V_in = 12 V
    alpha_arr = np.full_like(t_dense, 1.0 / V_IN)

    # Shift time to start at 0
    t_dense = t_dense - t_dense[0]

    print(f"  GPU CSV    : {len(t_raw)} samples over {t_raw[-1]/60:.1f} min")
    print(f"  PHNN grid  : {len(t_dense):,} steps at dt={dt*1e6:.2f} us")
    print(f"  Power range: {P_min:.2f} – {P_max:.2f} W")
    print(f"  i_o range  : {io_arr.min():.2f} – {io_arr.max():.2f} A")

    return t_dense, alpha_arr, io_arr


def evaluate_gpu_profile(model: VRMBuckPHNN, device: torch.device) -> None:
    """Evaluate PHNN response at each GPU sample point.

    Simulating 90 minutes at dt=4.69us is 1.15 billion RK4 steps — infeasible.
    Instead, at each 15s GPU sample we run a short 8ms window (1706 steps):
      - First 2ms: warm-up from current state to get the transient response
      - Last 6ms: steady-state at that load
    This gives the full load-step dynamics at every operating point in seconds.
    """
    if not GPU_CSV.exists():
        print(f"  GPU CSV not found: {GPU_CSV} -- skipping.")
        return

    df = pd.read_csv(GPU_CSV)
    df['time_s'] = pd.to_numeric(df['Relative Time (Process)'], errors='coerce')
    df = df.dropna(subset=['time_s'])

    COLS = {
        'SoftUpdateBigTau': 'SoftUpdateBigTau - system/gpu.0.powerWatts',
        'HardUpdateBigC'  : 'HardUpdateBigC - system/gpu.0.powerWatts',
        'HardUpdate'      : 'HardUpdate - system/gpu.0.powerWatts',
        'DQN'             : 'DQN - system/gpu.0.powerWatts',
    }
    frames = []
    for label, col in COLS.items():
        sub = df[['time_s', col]].dropna(subset=[col]).copy()
        sub = sub.rename(columns={col: 'power_W'})
        frames.append(sub)

    combined = (pd.concat(frames)
                .sort_values('time_s')
                .reset_index(drop=True))

    MAX_S = 90 * 60
    combined = combined[combined['time_s'] <= MAX_S]
    t_samples = combined['time_s'].values
    p_samples = combined['power_W'].values

    # Scale power to VRM load-current range [2, 14] A
    P_min, P_max = p_samples.min(), p_samples.max()
    io_samples = 2.0 + (p_samples - P_min) / (P_max - P_min) * 12.0

    alpha_nom = 1.0 / V_IN

    # Short simulation window at each sample point
    WINDOW_S  = 8e-3    # 8 ms total per sample
    n_steps   = max(2, round(WINDOW_S / DT))
    t_window  = np.arange(n_steps) * DT

    # Storage for per-sample results
    ss_iL, ss_VC = [], []   # steady-state (last 25%) values
    all_iL, all_VC = [], []  # full windows for plotting

    # Initial steady state from first sample
    io_now = float(io_samples[0])
    VC_now = alpha_nom * V_IN - 3e-3 * io_now
    x_now  = np.array([io_now, VC_now], dtype=np.float32)

    print(f"  Running {len(io_samples)} x {WINDOW_S*1e3:.0f} ms windows "
          f"({n_steps} steps each) ...")

    for k, io_k in enumerate(io_samples):
        u_k = np.column_stack((
            np.full(n_steps, alpha_nom * V_IN),
            np.full(n_steps, io_k),
        )).astype(np.float32)

        pred  = rk4_rollout(model, x_now, t_window, u_k, device)
        all_iL.append(pred[:, 0])
        all_VC.append(pred[:, 1])

        # Steady-state = mean of last 25% of window
        n_ss  = max(1, n_steps // 4)
        ss_iL.append(float(pred[-n_ss:, 0].mean()))
        ss_VC.append(float(pred[-n_ss:, 1].mean()))

        # Carry final state forward as initial state for next window
        x_now = pred[-1].astype(np.float32)

        if (k + 1) % 20 == 0 or k == 0:
            print(f"    sample {k+1:3d}/{len(io_samples)}: "
                  f"i_o={io_k:.2f}A  ss_iL={ss_iL[-1]:.3f}A  "
                  f"ss_VC={ss_VC[-1]:.4f}V")

    ss_iL = np.array(ss_iL)
    ss_VC = np.array(ss_VC)
    V_target = alpha_nom * V_IN   # ideal DC output
    t_min    = t_samples / 60.0

    # ── Trajectory plot (steady-state per sample) ─────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t_min, io_samples, color="tab:green", lw=1.2, marker="o",
                 ms=3, label="i_o demand (scaled GPU power)")
    axes[0].set_ylabel("Load current i_o [A]")
    axes[0].set_title("VRM PHNN — Steady-state Response to GPU Power Profile")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, io_samples, "k--", lw=0.8, alpha=0.5, label="i_o demand")
    axes[1].plot(t_min, ss_iL,  color="tab:blue", lw=1.2, marker="o",
                 ms=3, label="i_L steady-state (PHNN)")
    axes[1].axhline(0, color="gray", lw=0.5, ls=":")
    axes[1].set_ylabel("Current [A]")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].axhline(V_target, color="k", ls="--", lw=0.8, label=f"Target {V_target:.3f} V")
    axes[2].plot(t_min, ss_VC, color="tab:orange", lw=1.2, marker="o",
                 ms=3, label="V_C steady-state (PHNN)")
    axes[2].set_ylabel("V_C [V]")
    axes[2].set_xlabel("Time [minutes]")
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "vrm_eval_gpu_trajectory.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\n  Saved: {out}")

    # ── Per-perturbation phase spirals ────────────────────────────────────────
    # A "perturbation" = any window where the load changed from the previous.
    # We rank by |Δi_o| and show the top N in a grid, each as a smooth spiral.
    step_sizes = np.abs(np.diff(io_samples))           # size of each load step
    # indices into all_iL / all_VC where a load step occurred (index k means
    # the window AFTER the step, starting at the old steady-state)
    all_step_idx = np.argsort(step_sizes)[::-1]        # largest first

    N_PANELS = 12    # number of perturbations to show
    top_idx  = all_step_idx[:N_PANELS]

    n_cols = 4
    n_rows = int(np.ceil(N_PANELS / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes = axes.flatten()

    t_ms = t_window * 1e3   # time axis for the window in ms

    for panel, k in enumerate(top_idx):
        ax  = axes[panel]
        iL  = all_iL[k]       # i_L trajectory over 8ms window
        VC  = all_VC[k]       # V_C trajectory over 8ms window
        io_prev = io_samples[k - 1] if k > 0 else io_samples[k]
        io_new  = io_samples[k]
        step_dir = "↑" if io_new > io_prev else "↓"

        # Build smooth coloured line: colour maps from blue (t=0) to red (t=8ms)
        points = np.array([iL, VC]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        norm   = plt.Normalize(0, len(iL) - 1)
        lc     = LineCollection(segs, cmap="coolwarm", norm=norm,
                                linewidth=1.4, alpha=0.9, zorder=2)
        lc.set_array(np.arange(len(iL) - 1))
        ax.add_collection(lc)
        ax.autoscale()

        # Mark start (old equilibrium) and end (new equilibrium)
        ax.plot(iL[0],  VC[0],  "b^", ms=9, zorder=5, label="start")
        ax.plot(iL[-1], VC[-1], "rs", ms=9, zorder=5, label="end")

        # Draw the new equilibrium lines for reference
        ax.axvline(io_new,  color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(V_target, color="gray", lw=0.7, ls=":", alpha=0.6)

        step_mag = step_sizes[k]
        t_event  = t_min[k]
        ax.set_title(
            f"{step_dir}  {io_prev:.2f}→{io_new:.2f} A  (Δ={step_mag:.2f} A)\n"
            f"t = {t_event:.1f} min",
            fontsize=8,
        )
        ax.set_xlabel("i_L [A]", fontsize=7)
        ax.set_ylabel("V_C [V]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Hide any unused panels
    for panel in range(N_PANELS, len(axes)):
        axes[panel].set_visible(False)

    # Shared colorbar showing time within the 8ms window
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(0, t_ms[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:N_PANELS].tolist(),
                        shrink=0.6, pad=0.02)
    cbar.set_label("Time within 8ms window [ms]  (blue=start, red=end)",
                   fontsize=8)

    fig.suptitle(
        "VRM PHNN — Phase Spirals for Top-12 GPU Load Perturbations\n"
        "Each panel: 8ms PHNN rollout after a load step  (V_C vs i_L)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    out = OUT_DIR / "vrm_eval_gpu_phase_spirals.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\n  Saved: {out}")

    # Summary
    V_err = ss_VC - V_target
    i_err = ss_iL - io_samples
    print(f"\n  Steady-state voltage error (V_C vs {V_target:.3f} V target):")
    print(f"    Mean : {V_err.mean():.4f} V    RMS : {np.sqrt(np.mean(V_err**2)):.4f} V"
          f"    Max : {np.abs(V_err).max():.4f} V")
    print(f"\n  Steady-state current tracking (i_L vs i_o):")
    print(f"    Mean : {i_err.mean():.4f} A    RMS : {np.sqrt(np.mean(i_err**2)):.4f} A"
          f"    Max : {np.abs(i_err).max():.4f} A")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading VRM PHNN model ...")
    model = load_model(device)

    print("\n" + "=" * 64)
    print("Evaluation 1 — Held-out PLECS test set")
    print("=" * 64)
    evaluate_test_set(model, device)

    print("\n" + "=" * 64)
    print("Evaluation 2 — Real GPU power profile")
    print("=" * 64)
    evaluate_gpu_profile(model, device)

    print("\n" + "=" * 64)
    print("Plots saved to tests/:")
    print("  vrm_eval_test_trajectories.png    -- i_L(t), V_C(t) vs PLECS")
    print("  vrm_eval_test_phase.png           -- V_C vs i_L phase portrait vs PLECS")
    print("  vrm_eval_gpu_trajectory.png       -- steady-state tracking over GPU profile")
    print("  vrm_eval_gpu_phase_spirals.png    -- per-perturbation phase spirals (top 12)")
    print("=" * 64)
