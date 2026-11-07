"""
Evaluate SwitchedBuckPHNN vs Analytical Averaged Model vs PLECS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generate_plecs_switched_dataset import PLECSRPCSimulator, _interpolate_raw
from phnn import SwitchedBuckPHNN

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "switched_buck_phnn.pt"
PLECS_MODEL_PATH = ROOT / "models" / "buck_converter.plecs"
OUT_PLOT_PATH = ROOT / "tests" / "switched_vs_averaged_comparison.png"

# Known parameters
L_TRUE = 100e-6
C_TRUE = 220e-6
R_L_TRUE = 0.1
R_C_TRUE = 0.05
F_SW = 100e3
T_SW = 1.0 / F_SW
DT_EVAL = 1e-6


def simulate_textbook_averaged(
    t: np.ndarray,
    x0: np.ndarray,
    alpha_seq: np.ndarray,
    V_in_seq: np.ndarray,
    R_load_seq: np.ndarray,
) -> np.ndarray:
    """Simulate textbook averaged model with RK4."""
    states = np.zeros((len(t), 2), dtype=float)
    states[0] = x0
    for k in range(len(t) - 1):
        dt = t[k+1] - t[k]
        a = alpha_seq[k]
        Vin = V_in_seq[k]
        Rload = R_load_seq[k]
        
        def f(x):
            i_L, V_C = x[0], x[1]
            i_o = V_C / Rload # simplistic resistive load
            diL_dt = (a * Vin - R_L_TRUE * i_L - V_C) / L_TRUE
            dVC_dt = (i_L - i_o) / C_TRUE
            return np.array([diL_dt, dVC_dt])

        x_k = states[k]
        k1 = f(x_k)
        k2 = f(x_k + 0.5 * dt * k1)
        k3 = f(x_k + 0.5 * dt * k2)
        k4 = f(x_k + dt * k3)
        states[k+1] = x_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return states


def simulate_switched_phnn(
    model: SwitchedBuckPHNN,
    t: np.ndarray,
    x0: np.ndarray,
    alpha_seq: np.ndarray,
    V_in_seq: np.ndarray,
    R_load_seq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Roll out SwitchedBuckPHNN."""
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    
    # Generate v_sw
    t_mod = t % T_SW
    v_sw = np.zeros_like(t)
    for k in range(len(t)):
        if t_mod[k] < (alpha_seq[k] * T_SW + 1e-10):
            v_sw[k] = V_in_seq[k]

    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        
        # Calculate expected output current. Here we just use the current V_C, 
        # which isn't perfect since V_terminal = V_C + ESR * i_C, 
        # but for small ESR, it's a good approximation.
        V_C_k = states[k, 1]
        i_o_k = V_C_k / R_load_seq[k]
        
        u_k = np.array([v_sw[k], i_o_k], dtype=np.float32)
        
        xt = torch.tensor(states[k][None], dtype=torch.float32, device=device)
        ut = torch.tensor(u_k[None], dtype=torch.float32, device=device)
        
        def f(x_val):
            return model(x_val, ut, create_graph=False).detach().cpu().numpy()[0]
            
        k1 = f(xt)
        k2 = f(xt + 0.5 * dt_k * torch.tensor(k1, device=device))
        k3 = f(xt + 0.5 * dt_k * torch.tensor(k2, device=device))
        k4 = f(xt + dt_k * torch.tensor(k3, device=device))
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return states


def run_scenario(
    sim: PLECSRPCSimulator,
    scenario_name: str,
    alpha_start: float,
    alpha_end: float,
    R_start: float,
    R_end: float,
    V_in: float,
    model: SwitchedBuckPHNN,
    device: torch.device,
):
    print(f"Running scenario: {scenario_name}")
    
    # Simulate steady state first to get initial conditions
    t_steady = 5e-3
    model_vars = {
        "V_in": V_in,
        "L": L_TRUE,
        "C": C_TRUE,
        "ESR_L": R_L_TRUE,
        "ESR_C": R_C_TRUE,
        "R_load": R_start,
        "alpha": alpha_start,
        "cycle_frequency": F_SW,
    }
    
    # Get to steady state
    sim.simulate(model_vars, t_steady)
    initial_state = sim.get_system_state()
    
    # Now simulate the transient
    t_transient = 1e-3 # 1ms transient
    model_vars["alpha"] = alpha_end
    model_vars["R_load"] = R_end
    
    raw = sim.simulate(model_vars, t_transient, initial_system_state=initial_state)
    
    t_out = np.arange(0.0, t_transient + DT_EVAL * 0.5, DT_EVAL)
    iL_gt, Vterm_gt = _interpolate_raw(raw, t_out, iL_idx=0, VC_idx=1)
    
    i_o_gt = Vterm_gt / R_end
    i_cap_gt = iL_gt - i_o_gt
    VC_gt = Vterm_gt - R_C_TRUE * i_cap_gt
    
    x0 = np.array([iL_gt[0], VC_gt[0]])
    
    alpha_seq = np.full_like(t_out, alpha_end)
    Vin_seq = np.full_like(t_out, V_in)
    Rload_seq = np.full_like(t_out, R_end)
    
    # Simulate Averaged
    states_avg = simulate_textbook_averaged(t_out, x0, alpha_seq, Vin_seq, Rload_seq)
    
    # Simulate Switched PHNN
    states_phnn = simulate_switched_phnn(model, t_out, x0, alpha_seq, Vin_seq, Rload_seq, device)
    
    return t_out, (iL_gt, VC_gt), (states_avg[:, 0], states_avg[:, 1]), (states_phnn[:, 0], states_phnn[:, 1])


def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}")
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = SwitchedBuckPHNN(
        L=L_TRUE,
        C=C_TRUE,
        initial_R_L=0.20,
        initial_R_C=0.05,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    sim = PLECSRPCSimulator(PLECS_MODEL_PATH)
    
    scenarios = [
        {
            "name": "Alpha step 0.3 -> 0.7 (V_in=12V, R_load=3Ω)",
            "alpha_start": 0.3, "alpha_end": 0.7,
            "R_start": 3.0, "R_end": 3.0,
            "V_in": 12.0
        },
        {
            "name": "Alpha step 0.5 -> 0.2 (V_in=14V, R_load=4Ω)",
            "alpha_start": 0.5, "alpha_end": 0.2,
            "R_start": 4.0, "R_end": 4.0,
            "V_in": 14.0
        },
        {
            "name": "Load step 5Ω -> 2Ω (alpha=0.5, V_in=12V)",
            "alpha_start": 0.5, "alpha_end": 0.5,
            "R_start": 5.0, "R_end": 2.0,
            "V_in": 12.0
        }
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for row, sc in enumerate(scenarios):
        t, (iL_gt, VC_gt), (iL_avg, VC_avg), (iL_phnn, VC_phnn) = run_scenario(
            sim, sc["name"], sc["alpha_start"], sc["alpha_end"], 
            sc["R_start"], sc["R_end"], sc["V_in"], model, device
        )
        
        rmse_avg_iL = np.sqrt(np.mean((iL_avg - iL_gt)**2))
        rmse_avg_VC = np.sqrt(np.mean((VC_avg - VC_gt)**2))
        
        rmse_phnn_iL = np.sqrt(np.mean((iL_phnn - iL_gt)**2))
        rmse_phnn_VC = np.sqrt(np.mean((VC_phnn - VC_gt)**2))
        
        print(f"  Averaged RMSE: i_L={rmse_avg_iL:.4f} A, V_C={rmse_avg_VC:.4f} V")
        print(f"  PHNN     RMSE: i_L={rmse_phnn_iL:.4f} A, V_C={rmse_phnn_VC:.4f} V")
        if rmse_phnn_iL < rmse_avg_iL and rmse_phnn_VC < rmse_avg_VC:
            print("  -> SUCCESS: PHNN beats analytical model on both states!")
        else:
            print("  -> FAILED: PHNN did not beat analytical model on both states.")
            
        t_ms = t * 1000
        
        # i_L plot
        ax = axes[row, 0]
        ax.plot(t_ms, iL_gt, 'k', label='PLECS (Raw)', alpha=0.6)
        ax.plot(t_ms, iL_avg, 'tab:red', label=f'Avg (RMSE={rmse_avg_iL:.3f})', ls='--')
        ax.plot(t_ms, iL_phnn, 'tab:blue', label=f'PHNN (RMSE={rmse_phnn_iL:.3f})', ls=':')
        ax.set_title(f"{sc['name']} - i_L")
        ax.set_ylabel("Current [A]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # V_C plot
        ax = axes[row, 1]
        ax.plot(t_ms, VC_gt, 'k', label='PLECS (Raw)', alpha=0.6)
        ax.plot(t_ms, VC_avg, 'tab:red', label=f'Avg (RMSE={rmse_avg_VC:.3f})', ls='--')
        ax.plot(t_ms, VC_phnn, 'tab:blue', label=f'PHNN (RMSE={rmse_phnn_VC:.3f})', ls=':')
        ax.set_title(f"{sc['name']} - V_C")
        ax.set_ylabel("Voltage [V]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        if row == 2:
            axes[row, 0].set_xlabel("Time [ms]")
            axes[row, 1].set_xlabel("Time [ms]")
            
    fig.tight_layout()
    fig.savefig(OUT_PLOT_PATH, dpi=200)
    print(f"\nSaved plot to {OUT_PLOT_PATH}")

if __name__ == "__main__":
    main()
