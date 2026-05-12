"""
LC-tank dataset generator for Step 3 PHNN verification.

Conservative system: R = 0, u(t) = [0, 0, 0].
Dynamics reduce to a pure LC oscillator with closed phase-space orbits.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from simulator import simulate_buck, compute_xdot, iL_vC_to_phi_q

# Physical parameters — same as training dataset
PARAMS: dict = {"L": 100e-6, "C": 100e-6, "R": 0.0}

# Dataset configuration
N_TRAJECTORIES: int = 50
DT: float = 10e-6          # 100 kHz sampling
T_END_RAW: float = 5e-3    # 5 ms per trajectory

# Snap t_end to an exact dt multiple (avoids np.arange overshoot in simulate_buck)
N_STEPS: int = round(T_END_RAW / DT)
T_END: float = float(N_STEPS * DT)

# Initial-condition ranges
IL_RANGE: tuple[float, float] = (-2.0, 2.0)   # A
VC_RANGE: tuple[float, float] = (-10.0, 10.0)  # V

SEED: int = 42


def _zero_input(t: float) -> np.ndarray:
    """Zero input: no duty cycle, no supply, no load."""
    return np.array([0.0, 0.0, 0.0])


def generate_lc_tank_dataset(
    n_trajectories: int = N_TRAJECTORIES,
    params: dict = PARAMS,
    dt: float = DT,
    t_end: float = T_END,
    il_range: tuple[float, float] = IL_RANGE,
    vc_range: tuple[float, float] = VC_RANGE,
    seed: int = SEED,
) -> list[dict]:
    """Simulate the conservative LC tank and return trajectory list.

    Args:
        n_trajectories: Number of trajectories to generate.
        params:         Physical parameters dict (L, C, R). R should be 0.
        dt:             Output sampling interval [s].
        t_end:          Trajectory duration [s].
        il_range:       (min, max) for initial i_L [A].
        vc_range:       (min, max) for initial V_C [V].
        seed:           RNG seed for reproducibility.

    Returns:
        List of dicts, each with keys:
            t, Phi, q, i_L, V_C, dPhi_dt, dq_dt
    """
    rng = np.random.default_rng(seed)
    L = params["L"]
    C = params["C"]

    trajectories: list[dict] = []

    for _ in range(n_trajectories):
        # Sample ICs; reject if both are exactly zero (extremely unlikely)
        while True:
            iL0 = rng.uniform(*il_range)
            VC0 = rng.uniform(*vc_range)
            if abs(iL0) > 1e-9 or abs(VC0) > 1e-9:
                break

        Phi0, q0 = iL_vC_to_phi_q(iL0, VC0, L, C)
        x0 = np.array([Phi0, q0])

        sol = simulate_buck(
            x0=x0,
            u_fn=_zero_input,
            t_span=(0.0, t_end),
            params=params,
            dt=dt,
        )

        # Analytical derivatives — vectorised over time axis
        n = len(sol["t"])
        u_zero = np.array([0.0, 0.0, 0.0])
        derivs = np.stack(
            [compute_xdot(np.array([sol["Phi"][k], sol["q"][k]]), u_zero, params)
             for k in range(n)]
        )  # shape (n, 2)

        trajectories.append({
            "t":        sol["t"],
            "Phi":      sol["Phi"],
            "q":        sol["q"],
            "i_L":      sol["i_L"],
            "V_C":      sol["V_C"],
            "dPhi_dt":  derivs[:, 0],
            "dq_dt":    derivs[:, 1],
        })

    return trajectories


def save_lc_tank_dataset(trajectories: list[dict], path: str | Path) -> None:
    """Save trajectory list to .npz file.

    Args:
        trajectories: List of trajectory dicts from generate_lc_tank_dataset.
        path:         Output path (will be created if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for i, traj in enumerate(trajectories):
        for key, val in traj.items():
            arrays[f"traj_{i}_{key}"] = np.asarray(val)
    arrays["n_trajectories"] = np.array(len(trajectories))

    np.savez(path, **arrays)
    print(f"Saved {len(trajectories)} trajectories to {path}")


def load_lc_tank_dataset(path: str | Path) -> list[dict]:
    """Load trajectory list from .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        List of trajectory dicts.
    """
    data = np.load(path)
    n = int(data["n_trajectories"])
    keys = ["t", "Phi", "q", "i_L", "V_C", "dPhi_dt", "dq_dt"]
    return [
        {k: data[f"traj_{i}_{k}"] for k in keys}
        for i in range(n)
    ]


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "data" / "lc_tank.npz"
    trajs = generate_lc_tank_dataset()

    # Quick summary
    all_iL = np.concatenate([t["i_L"] for t in trajs])
    all_VC = np.concatenate([t["V_C"] for t in trajs])
    total_pts = sum(len(t["t"]) for t in trajs)
    print(f"Trajectories : {len(trajs)}")
    print(f"Total points : {total_pts}")
    print(f"i_L  range   : {all_iL.min():.3f} .. {all_iL.max():.3f} A")
    print(f"V_C  range   : {all_VC.min():.3f} .. {all_VC.max():.3f} V")
    print(f"Duration     : {T_END*1e3:.1f} ms per trajectory")

    save_lc_tank_dataset(trajs, out_path)
