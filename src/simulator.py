"""
Averaged Port-Hamiltonian model of a synchronous buck converter.

State variables (energy-based coordinates):
    Phi  [V·s = Wb]  — inductor flux linkage:  Phi = L * i_L
    q    [A·s = C ]  — capacitor charge:        q   = C * V_C

Hamiltonian (total stored energy):
    H(Phi, q) = Phi^2 / (2L) + q^2 / (2C)

Input vector  u = [alpha, V_in, i_o]:
    alpha   [dimensionless, 0..1]  — average duty cycle
    V_in    [V]                    — input voltage
    i_o     [A]                    — output (load) current
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable


def phi_q_to_iL_vC(
    Phi: np.ndarray,
    q: np.ndarray,
    L: float,
    C: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flux-charge coordinates to physical currents/voltages.

    Args:
        Phi: Inductor flux linkage [Wb].
        q:   Capacitor charge [C].
        L:   Inductance [H].
        C:   Capacitance [F].

    Returns:
        (i_L [A], V_C [V])
    """
    i_L = Phi / L
    V_C = q / C
    return i_L, V_C


def iL_vC_to_phi_q(
    i_L: np.ndarray,
    V_C: np.ndarray,
    L: float,
    C: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical currents/voltages to flux-charge coordinates.

    Args:
        i_L: Inductor current [A].
        V_C: Capacitor voltage [V].
        L:   Inductance [H].
        C:   Capacitance [F].

    Returns:
        (Phi [Wb], q [C])
    """
    Phi = L * i_L
    q = C * V_C
    return Phi, q


def compute_xdot(
    x: np.ndarray,
    u: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Evaluate the Port-Hamiltonian ODE analytically.

    Dynamics (averaged model):
        dPhi/dt = alpha * V_in - R * i_L - V_C
        dq/dt   = i_L - i_o

    where  i_L = Phi / L,  V_C = q / C.

    Args:
        x:      State vector [Phi [Wb], q [C]].
        u:      Input vector [alpha, V_in [V], i_o [A]].
        params: Dict with keys 'L' [H], 'C' [F], 'R' [Ω].

    Returns:
        dx/dt as shape-(2,) array [dPhi/dt, dq/dt].
    """
    Phi, q = x
    alpha, V_in, i_o = u
    L = params["L"]
    C = params["C"]
    R = params["R"]

    i_L = Phi / L
    V_C = q / C

    dPhi_dt = alpha * V_in - R * i_L - V_C
    dq_dt = i_L - i_o

    return np.array([dPhi_dt, dq_dt])


def simulate_buck(
    x0: np.ndarray,
    u_fn: Callable[[float], np.ndarray],
    t_span: tuple[float, float],
    params: dict,
    dt: float | None = None,
) -> dict:
    """Simulate the averaged Port-Hamiltonian buck converter ODE.

    The ODE is integrated with RK45 at tight tolerances.  An optional
    uniform output grid can be requested via *dt*; otherwise the adaptive
    solver chooses the time points.

    Args:
        x0:     Initial state [Phi0 [Wb], q0 [C]].
        u_fn:   Callable u_fn(t) → array([alpha, V_in, i_o]).
        t_span: (t_start, t_end) in seconds.
        params: Dict with keys 'L' [H], 'C' [F], 'R' [Ω].
        dt:     If given, output is evaluated on a uniform grid with
                spacing dt [s].

    Returns:
        Dict with keys:
            't'     — time array [s]
            'Phi'   — flux linkage [Wb]
            'q'     — capacitor charge [C]
            'i_L'   — inductor current [A]
            'V_C'   — capacitor voltage [V]
            'alpha' — duty cycle at each time
            'V_in'  — input voltage at each time [V]
            'i_o'   — load current at each time [A]
    """
    x0 = np.asarray(x0, dtype=float)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        u = np.asarray(u_fn(t), dtype=float)
        return compute_xdot(x, u, params)

    t_eval = None
    if dt is not None:
        t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)

    sol = solve_ivp(
        rhs,
        t_span,
        x0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    t = sol.t
    Phi = sol.y[0]
    q = sol.y[1]

    L = params["L"]
    C = params["C"]
    i_L, V_C = phi_q_to_iL_vC(Phi, q, L, C)

    # Evaluate inputs at every output time step
    u_arr = np.array([u_fn(ti) for ti in t])  # shape (N, 3)
    alpha_arr = u_arr[:, 0]
    V_in_arr = u_arr[:, 1]
    i_o_arr = u_arr[:, 2]

    return {
        "t": t,
        "Phi": Phi,
        "q": q,
        "i_L": i_L,
        "V_C": V_C,
        "alpha": alpha_arr,
        "V_in": V_in_arr,
        "i_o": i_o_arr,
    }
