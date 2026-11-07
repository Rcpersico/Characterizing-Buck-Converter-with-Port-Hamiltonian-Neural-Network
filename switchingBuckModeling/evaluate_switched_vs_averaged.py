"""
Three-way comparison: PLECS ground truth vs. analytical averaged model vs. SwitchedBuckPHNN.

Tests the key research claim: a PHNN trained on raw switched waveforms should
predict transient behaviour more accurately than the textbook averaged model.

Test scenarios (from the research spec):
  1. Alpha step 0.3 -> 0.7  at V_in=12V, R_load=3 Ohm
  2. Alpha step 0.5 -> 0.2  at V_in=14V, R_load=4 Ohm
  3. Load step 5 Ohm -> 2 Ohm  at alpha=0.5, V_in=12V

For each scenario:
  - Run PLECS at 1 MHz to get the ground truth switched waveform.
  - Simulate the textbook averaged model (RK4 on the averaged ODE).
  - Roll out SwitchedBuckPHNN (RK4 at 1 MHz with v_sw(t) as input).
  - Compute RMSE over the first 1 ms (transient region).
  - Generate overlay plots.

Acceptance criterion (criterion 3 from the spec):
  SwitchedBuckPHNN RMSE < analytical averaged model RMSE for BOTH i_L and V_C
  in at least 2 out of 3 scenarios.

Usage:
    & "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n nilm-gpu python switchingBuckModeling/evaluate_switched_vs_averaged.py

Requires:
  - PLECS running with the buck_converter.plecs model loaded.
  - models/switched_buck_phnn.pt produced by train_switched_buck.py.
"""

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

from generate_plecs_switched_dataset import PLECSRPCSimulator, _interpolate_raw
from phnn import SwitchedBuckPHNN

# ---------------------------------------------------------------------------
# Paths and physical constants
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "models" / "switched_buck_phnn.pt"
PLECS_MODEL_PATH = ROOT / "models" / "buck_converter.plecs"
OUT_PLOT_PATH = ROOT / "tests" / "test_images" / "switched_vs_averaged_alpha_step.png"

L_TRUE = 100e-6    # H
C_TRUE = 220e-6    # F
R_L_TRUE = 0.1     # Ohm  (inductor series resistance)
R_C_TRUE = 0.05    # Ohm  (capacitor ESR)
F_SW = 100e3       # Hz
T_SW = 1.0 / F_SW  # s
DT_EVAL = 1e-6     # s  (1 MHz evaluation grid)
T_TRANSIENT = 1e-3 # s  (evaluate RMSE over first 1 ms after step)
T_STEADY = 5e-3    # s  (pre-simulate to reach steady state before step)


# ---------------------------------------------------------------------------
# Analytical averaged model
# ---------------------------------------------------------------------------


def simulate_textbook_averaged(
    t: np.ndarray,
    x0: np.ndarray,
    alpha_seq: np.ndarray,
    V_in_seq: np.ndarray,
    R_load_seq: np.ndarray,
) -> np.ndarray:
    """Textbook averaged buck ODE integrated with RK4.

    ODE:
        di_L/dt = (alpha*V_in - R_L*i_L - V_C) / L
        dV_C/dt = (i_L - V_C/R_load) / C

    Returns state array [len(t), 2] with columns [i_L, V_C].
    """
    states = np.zeros((len(t), 2), dtype=float)
    states[0] = x0

    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        a = float(alpha_seq[k])
        Vin = float(V_in_seq[k])
        Rload = float(R_load_seq[k])

        def f(x: np.ndarray) -> np.ndarray:
            i_L, V_C = x[0], x[1]
            i_o = V_C / Rload
            diL_dt = (a * Vin - R_L_TRUE * i_L - V_C) / L_TRUE
            dVC_dt = (i_L - i_o) / C_TRUE
            return np.array([diL_dt, dVC_dt])

        xk = states[k]
        k1 = f(xk)
        k2 = f(xk + 0.5 * dt * k1)
        k3 = f(xk + 0.5 * dt * k2)
        k4 = f(xk + dt * k3)
        states[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return states


# ---------------------------------------------------------------------------
# SwitchedBuckPHNN rollout
# ---------------------------------------------------------------------------


def simulate_switched_phnn(
    model: SwitchedBuckPHNN,
    t: np.ndarray,
    x0: np.ndarray,
    alpha_seq: np.ndarray,
    V_in_seq: np.ndarray,
    R_load_seq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Roll out SwitchedBuckPHNN with v_sw(t) computed from the PWM pattern.

    v_sw(k) = V_in(k)   if t(k) mod T_sw < alpha(k) * T_sw
              0           otherwise

    i_o is approximated as V_C / R_load at each step (closed-loop resistive
    load approximation; ESR is small so V_terminal ≈ V_C).
    """
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)

    # Pre-compute v_sw for all timesteps
    t_mod = t % T_SW
    v_sw = np.where(t_mod < alpha_seq * T_SW + 1e-10, V_in_seq, 0.0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        for k in range(len(t) - 1):
            dt_k = float(t[k + 1] - t[k])
            V_C_k = float(states[k, 1])
            i_o_k = float(V_C_k / R_load_seq[k])
            u_k = np.array([v_sw[k], i_o_k], dtype=np.float32)

            xt = torch.tensor(states[k][None], dtype=torch.float32, device=device)
            ut = torch.tensor(u_k[None], dtype=torch.float32, device=device)

            def f_model(x_t: torch.Tensor) -> np.ndarray:
                return model(x_t, ut, create_graph=False).cpu().numpy()[0]

            k1 = f_model(xt)
            k2 = f_model(xt + torch.tensor(0.5 * dt_k * k1, device=device))
            k3 = f_model(xt + torch.tensor(0.5 * dt_k * k2, device=device))
            k4 = f_model(xt + torch.tensor(dt_k * k3, device=device))
            states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return states


# ---------------------------------------------------------------------------
# Per-scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    sim: PLECSRPCSimulator,
    scenario: dict,
    model: SwitchedBuckPHNN,
    device: torch.device,
) -> dict:
    """Run one comparison scenario. Returns a result dict."""
    name = scenario["name"]
    alpha_start = float(scenario["alpha_start"])
    alpha_end = float(scenario["alpha_end"])
    R_start = float(scenario["R_start"])
    R_end = float(scenario["R_end"])
    V_in = float(scenario["V_in"])

    print(f"\nScenario: {name}")

    # Step 1: settle to steady state at starting conditions
    model_vars_pre = {
        "V_in": V_in,
        "L": L_TRUE,
        "C": C_TRUE,
        "ESR_L": R_L_TRUE,
        "ESR_C": R_C_TRUE,
        "R_load": R_start,
        "alpha": alpha_start,
        "cycle_frequency": F_SW,
    }
    sim.simulate(model_vars_pre, T_STEADY)
    initial_state = sim.get_system_state()

    # Step 2: simulate the transient at post-step conditions
    model_vars_post = dict(model_vars_pre)
    model_vars_post["alpha"] = alpha_end
    model_vars_post["R_load"] = R_end

    raw = sim.simulate(model_vars_post, T_TRANSIENT, initial_system_state=initial_state)

    t_out = np.arange(0.0, T_TRANSIENT + DT_EVAL * 0.5, DT_EVAL)
    iL_gt, Vterm_gt = _interpolate_raw(raw, t_out, iL_idx=0, VC_idx=1)
    i_o_gt = Vterm_gt / R_end
    i_cap_gt = iL_gt - i_o_gt
    VC_gt = Vterm_gt - R_C_TRUE * i_cap_gt

    x0 = np.array([iL_gt[0], VC_gt[0]])
    alpha_seq = np.full_like(t_out, alpha_end)
    Vin_seq = np.full_like(t_out, V_in)
    Rload_seq = np.full_like(t_out, R_end)

    # Analytical averaged model
    states_avg = simulate_textbook_averaged(t_out, x0, alpha_seq, Vin_seq, Rload_seq)

    # SwitchedBuckPHNN
    states_phnn = simulate_switched_phnn(model, t_out, x0, alpha_seq, Vin_seq, Rload_seq, device)

    rmse_avg_iL = float(np.sqrt(np.mean((states_avg[:, 0] - iL_gt) ** 2)))
    rmse_avg_VC = float(np.sqrt(np.mean((states_avg[:, 1] - VC_gt) ** 2)))
    rmse_phnn_iL = float(np.sqrt(np.mean((states_phnn[:, 0] - iL_gt) ** 2)))
    rmse_phnn_VC = float(np.sqrt(np.mean((states_phnn[:, 1] - VC_gt) ** 2)))

    phnn_wins_iL = rmse_phnn_iL < rmse_avg_iL
    phnn_wins_VC = rmse_phnn_VC < rmse_avg_VC

    print(f"  Averaged RMSE : i_L={rmse_avg_iL:.4f} A   V_C={rmse_avg_VC:.4f} V")
    print(f"  PHNN     RMSE : i_L={rmse_phnn_iL:.4f} A   V_C={rmse_phnn_VC:.4f} V")
    status = "SUCCESS" if (phnn_wins_iL and phnn_wins_VC) else "PARTIAL" if (phnn_wins_iL or phnn_wins_VC) else "FAILED"
    print(f"  Result: {status}  (PHNN wins i_L={phnn_wins_iL}, V_C={phnn_wins_VC})")

    return {
        "name": name,
        "t": t_out,
        "iL_gt": iL_gt,
        "VC_gt": VC_gt,
        "iL_avg": states_avg[:, 0],
        "VC_avg": states_avg[:, 1],
        "iL_phnn": states_phnn[:, 0],
        "VC_phnn": states_phnn[:, 1],
        "rmse_avg_iL": rmse_avg_iL,
        "rmse_avg_VC": rmse_avg_VC,
        "rmse_phnn_iL": rmse_phnn_iL,
        "rmse_phnn_VC": rmse_phnn_VC,
        "phnn_wins_iL": phnn_wins_iL,
        "phnn_wins_VC": phnn_wins_VC,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Run train_switched_buck.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SwitchedBuckPHNN on {device} ...")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = SwitchedBuckPHNN(L=L_TRUE, C=C_TRUE, initial_R_L=0.20, initial_R_C=0.05).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    p = model.get_params()
    print(f"  Learned R_L={p['R_L']:.4f} Ohm  R_C={p['R_C']:.4f} Ohm")
    print(f"  Best val loss={ckpt['best_val_loss']:.4e}  epoch={ckpt['best_epoch']}")

    if not PLECS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PLECS model not found at {PLECS_MODEL_PATH}. Ensure buck_converter.plecs exists."
        )
    sim = PLECSRPCSimulator(PLECS_MODEL_PATH)

    scenarios = [
        {
            "name": "Alpha step 0.3->0.7  V_in=12V  R=3Ω",
            "alpha_start": 0.3, "alpha_end": 0.7,
            "R_start": 3.0,   "R_end": 3.0,
            "V_in": 12.0,
        },
        {
            "name": "Alpha step 0.5->0.2  V_in=14V  R=4Ω",
            "alpha_start": 0.5, "alpha_end": 0.2,
            "R_start": 4.0,   "R_end": 4.0,
            "V_in": 14.0,
        },
        {
            "name": "Load step 5->2Ω  alpha=0.5  V_in=12V",
            "alpha_start": 0.5, "alpha_end": 0.5,
            "R_start": 5.0,   "R_end": 2.0,
            "V_in": 12.0,
        },
    ]

    results = [run_scenario(sim, sc, model, device) for sc in scenarios]

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY — transient RMSE over first 1 ms after step")
    print("=" * 80)
    print(f"{'Scenario':<45}  {'Avg i_L':>8}  {'PHNN i_L':>8}  {'Avg V_C':>8}  {'PHNN V_C':>8}  {'Result':>8}")
    print("-" * 80)
    n_wins = 0
    for r in results:
        status = "WIN" if (r["phnn_wins_iL"] and r["phnn_wins_VC"]) else "partial" if (r["phnn_wins_iL"] or r["phnn_wins_VC"]) else "loss"
        if r["phnn_wins_iL"] and r["phnn_wins_VC"]:
            n_wins += 1
        print(
            f"  {r['name']:<43}  "
            f"{r['rmse_avg_iL']:>8.4f}  {r['rmse_phnn_iL']:>8.4f}  "
            f"{r['rmse_avg_VC']:>8.4f}  {r['rmse_phnn_VC']:>8.4f}  {status:>8}"
        )
    print("-" * 80)
    print(f"  PHNN wins (both states): {n_wins}/{len(results)}")
    if n_wins >= 2:
        print("  OVERALL RESULT: PASSED — SwitchedBuckPHNN beats averaged model on transients.")
    else:
        print("  OVERALL RESULT: FAILED — See notes.md for failure analysis and next steps.")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 4 * len(results)))
    t_ms = results[0]["t"] * 1e3

    for row, r in enumerate(results):
        # i_L panel
        ax = axes[row, 0]
        ax.plot(t_ms, r["iL_gt"],   "k",          lw=1.5, alpha=0.55, label="PLECS (raw)")
        ax.plot(t_ms, r["iL_avg"],  "tab:red",    lw=1.5, ls="--",
                label=f"Averaged  RMSE={r['rmse_avg_iL']:.3f} A")
        ax.plot(t_ms, r["iL_phnn"], "tab:blue",   lw=1.5, ls=":",
                label=f"PHNN     RMSE={r['rmse_phnn_iL']:.3f} A")
        ax.set_title(r["name"])
        ax.set_ylabel("i_L  [A]")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # V_C panel
        ax = axes[row, 1]
        ax.plot(t_ms, r["VC_gt"],   "k",          lw=1.5, alpha=0.55, label="PLECS (raw)")
        ax.plot(t_ms, r["VC_avg"],  "tab:red",    lw=1.5, ls="--",
                label=f"Averaged  RMSE={r['rmse_avg_VC']:.3f} V")
        ax.plot(t_ms, r["VC_phnn"], "tab:orange", lw=1.5, ls=":",
                label=f"PHNN     RMSE={r['rmse_phnn_VC']:.3f} V")
        ax.set_title(r["name"])
        ax.set_ylabel("V_C  [V]")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")

    fig.suptitle(
        "SwitchedBuckPHNN vs Analytical Averaged Model — 1 ms transient RMSE comparison",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nSaved comparison plot -> {OUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
