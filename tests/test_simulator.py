"""
Sanity-check simulation for the averaged Port-Hamiltonian buck converter.

Circuit parameters
------------------
    L = 100 uH,  C = 100 uF,  R = 0.1 Ω
    V_in = 12 V,  alpha = 0.5 (constant),  i_o = 1 A (constant)

Steady-state analysis
---------------------
At steady state dPhi/dt = 0  →  alpha * V_in = R * i_L + V_C
               dq/dt   = 0  →  i_L = i_o

So:  V_C* = alpha * V_in - R * i_o = 0.5*12 - 0.1*1 = 5.9 V

Resonant frequency
------------------
The natural frequency of the LC tank (undamped):
    omega_0 = 1 / sqrt(L * C)
"""

import sys
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Allow running from anywhere in the project tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator import (
    simulate_buck,
    iL_vC_to_phi_q,
    phi_q_to_iL_vC,
    compute_xdot,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
params = {
    "L": 100e-6,   # 100 µH
    "C": 100e-6,   # 100 µF
    "R": 0.1,      # 0.1 Ω
}
V_IN = 12.0
ALPHA = 0.5
I_O = 1.0
T_END = 10e-3   # 10 ms
DT = 1e-6       # 1 µs output grid

# ---------------------------------------------------------------------------
# Analytical targets
# ---------------------------------------------------------------------------
V_C_STAR = ALPHA * V_IN - params["R"] * I_O          # 5.9 V
OMEGA_0 = 1.0 / math.sqrt(params["L"] * params["C"]) # rad/s


def u_fn(t: float) -> np.ndarray:
    return np.array([ALPHA, V_IN, I_O])


def run_simulation():
    Phi0, q0 = iL_vC_to_phi_q(0.0, 0.0, params["L"], params["C"])
    x0 = np.array([Phi0, q0])

    result = simulate_buck(
        x0=x0,
        u_fn=u_fn,
        t_span=(0.0, T_END),
        params=params,
        dt=DT,
    )
    return result


def test_steady_state(result: dict) -> None:
    """V_C should settle to V_C* = alpha*V_in - R*i_o within 0.5 %."""
    # Use the final 20 % of the simulation as the "settled" window
    t = result["t"]
    V_C = result["V_C"]
    mask = t >= 0.8 * T_END
    V_C_mean = V_C[mask].mean()

    print(f"\n--- Steady-state V_C ---")
    print(f"  Expected : {V_C_STAR:.4f} V")
    print(f"  Simulated: {V_C_mean:.4f} V")

    tol = 0.005 * abs(V_C_STAR)   # 0.5 % tolerance
    err = abs(V_C_mean - V_C_STAR)
    assert err < tol, (
        f"Steady-state V_C mismatch: expected {V_C_STAR:.4f} V, "
        f"got {V_C_mean:.4f} V (err={err:.4f} V > tol={tol:.4f} V)"
    )
    print(f"  PASSED  (err={err:.6f} V, tol={tol:.6f} V)")


def test_resonant_frequency(result: dict) -> None:
    """Dominant oscillation frequency of i_L(t) must match omega_0 within 3 %."""
    t = result["t"]
    i_L = result["i_L"]

    # FFT of i_L on the uniform grid
    dt_actual = t[1] - t[0]
    n = len(i_L)
    freqs = np.fft.rfftfreq(n, d=dt_actual)  # Hz
    spectrum = np.abs(np.fft.rfft(i_L - i_L.mean()))

    # Dominant frequency (exclude DC bin 0)
    dominant_hz = freqs[1 + np.argmax(spectrum[1:])]
    dominant_omega = 2.0 * math.pi * dominant_hz

    print(f"\n--- Resonant frequency ---")
    print(f"  Expected omega_0 : {OMEGA_0:.2f} rad/s  ({OMEGA_0/(2*math.pi):.2f} Hz)")
    print(f"  Simulated omega  : {dominant_omega:.2f} rad/s  ({dominant_hz:.2f} Hz)")

    tol = 0.03 * OMEGA_0   # 3 % tolerance
    err = abs(dominant_omega - OMEGA_0)
    assert err < tol, (
        f"Resonant frequency mismatch: expected {OMEGA_0:.2f} rad/s, "
        f"got {dominant_omega:.2f} rad/s (err={err:.2f} > tol={tol:.2f})"
    )
    print(f"  PASSED  (err={err:.2f} rad/s, tol={tol:.2f} rad/s)")


def test_compute_xdot() -> None:
    """compute_xdot should return zero at steady state."""
    i_L_ss = I_O
    V_C_ss = V_C_STAR
    Phi_ss, q_ss = iL_vC_to_phi_q(i_L_ss, V_C_ss, params["L"], params["C"])
    x_ss = np.array([Phi_ss, q_ss])
    u_ss = np.array([ALPHA, V_IN, I_O])

    xdot = compute_xdot(x_ss, u_ss, params)
    print(f"\n--- xdot at steady state ---")
    print(f"  dPhi/dt = {xdot[0]:.2e}  (should be ~0)")
    print(f"  dq/dt   = {xdot[1]:.2e}  (should be ~0)")
    assert np.allclose(xdot, 0.0, atol=1e-12), (
        f"compute_xdot nonzero at steady state: {xdot}"
    )
    print(f"  PASSED")


def save_plot(result: dict) -> None:
    out_path = os.path.join(os.path.dirname(__file__), "sanity_check.png")
    t_ms = result["t"] * 1e3  # convert to ms

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(t_ms, result["i_L"], color="steelblue", linewidth=1.5)
    ax1.axhline(I_O, color="steelblue", linestyle="--", linewidth=0.8, label=f"SS target {I_O} A")
    ax1.set_ylabel("$i_L$ [A]")
    ax1.set_title("Buck Converter — Averaged Port-Hamiltonian Simulation")
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.4)

    ax2.plot(t_ms, result["V_C"], color="darkorange", linewidth=1.5)
    ax2.axhline(V_C_STAR, color="darkorange", linestyle="--", linewidth=0.8,
                label=f"SS target {V_C_STAR} V")
    ax2.set_ylabel("$V_C$ [V]")
    ax2.set_xlabel("Time [ms]")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Buck Converter — Step 1 Sanity Check")
    print("=" * 55)
    print(f"  L={params['L']*1e6:.0f} µH  C={params['C']*1e6:.0f} µF  "
          f"R={params['R']} Ω")
    print(f"  V_in={V_IN} V  alpha={ALPHA}  i_o={I_O} A")
    print(f"  t_end={T_END*1e3:.0f} ms  dt={DT*1e6:.0f} µs")

    result = run_simulation()
    print(f"\n  Solver returned {len(result['t'])} time points.")

    test_steady_state(result)
    test_resonant_frequency(result)
    test_compute_xdot()
    save_plot(result)

    print("\n" + "=" * 55)
    print("  All assertions PASSED.")
    print("=" * 55)
