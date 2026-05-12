"""RLC no-input dataset generator for Stage 2 PHNN verification.

Damped system: R > 0, u(t) = [0, 0, 0].  Dynamics reduce to the
unforced buck/RLC oscillator with inductor branch dissipation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from simulator import compute_xdot, iL_vC_to_phi_q, simulate_buck


PARAMS: dict = {"L": 100e-6, "C": 100e-6, "R": 0.1}
N_TRAJECTORIES: int = 60
DT: float = 10e-6
T_END_RAW: float = 8e-3
N_STEPS: int = round(T_END_RAW / DT)
T_END: float = float(N_STEPS * DT)
IL_RANGE: tuple[float, float] = (-2.0, 2.0)
VC_RANGE: tuple[float, float] = (-10.0, 10.0)
SEED: int = 123


def _zero_input(t: float) -> np.ndarray:
    """Zero input: no duty cycle, no supply, no load."""
    return np.array([0.0, 0.0, 0.0])


def generate_rlc_dataset(
    n_trajectories: int = N_TRAJECTORIES,
    params: dict = PARAMS,
    dt: float = DT,
    t_end: float = T_END,
    il_range: tuple[float, float] = IL_RANGE,
    vc_range: tuple[float, float] = VC_RANGE,
    seed: int = SEED,
) -> list[dict]:
    """Simulate damped zero-input RLC trajectories.

    Returns trajectory dicts with ``t, Phi, q, i_L, V_C, dPhi_dt, dq_dt``.
    Initial conditions are nonzero, and all inputs are identically zero.
    """
    rng = np.random.default_rng(seed)
    l_value = params["L"]
    c_value = params["C"]

    trajectories: list[dict] = []
    for _ in range(n_trajectories):
        while True:
            i_l0 = float(rng.uniform(*il_range))
            v_c0 = float(rng.uniform(*vc_range))
            if abs(i_l0) > 1e-6 or abs(v_c0) > 1e-6:
                break

        phi0, q0 = iL_vC_to_phi_q(i_l0, v_c0, l_value, c_value)
        sol = simulate_buck(
            x0=np.array([phi0, q0]),
            u_fn=_zero_input,
            t_span=(0.0, t_end),
            params=params,
            dt=dt,
        )

        n = len(sol["t"])
        u_zero = np.array([0.0, 0.0, 0.0])
        derivs = np.stack(
            [
                compute_xdot(np.array([sol["Phi"][k], sol["q"][k]]), u_zero, params)
                for k in range(n)
            ],
            axis=0,
        )

        trajectories.append(
            {
                "t": sol["t"],
                "Phi": sol["Phi"],
                "q": sol["q"],
                "i_L": sol["i_L"],
                "V_C": sol["V_C"],
                "dPhi_dt": derivs[:, 0],
                "dq_dt": derivs[:, 1],
            }
        )

    return trajectories


def save_rlc_dataset(trajectories: list[dict], path: str | Path) -> None:
    """Save trajectory list to a single ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for i, traj in enumerate(trajectories):
        for key, value in traj.items():
            arrays[f"traj_{i}_{key}"] = np.asarray(value)
    arrays["n_trajectories"] = np.array(len(trajectories))
    np.savez(path, **arrays)
    print(f"Saved {len(trajectories)} trajectories to {path}")


def load_rlc_dataset(path: str | Path) -> list[dict]:
    """Load trajectory list from a ``.npz`` file."""
    data = np.load(path)
    n = int(data["n_trajectories"])
    keys = ["t", "Phi", "q", "i_L", "V_C", "dPhi_dt", "dq_dt"]
    return [{key: data[f"traj_{i}_{key}"] for key in keys} for i in range(n)]


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "data" / "rlc_no_input.npz"
    trajs = generate_rlc_dataset()

    all_i_l = np.concatenate([traj["i_L"] for traj in trajs])
    all_v_c = np.concatenate([traj["V_C"] for traj in trajs])
    total_points = sum(len(traj["t"]) for traj in trajs)
    print(f"Trajectories : {len(trajs)}")
    print(f"Total points : {total_points}")
    print(f"i_L range    : {all_i_l.min():.3f} .. {all_i_l.max():.3f} A")
    print(f"V_C range    : {all_v_c.min():.3f} .. {all_v_c.max():.3f} V")
    print(f"R            : {PARAMS['R']:.3f} Ohm")
    print(f"Duration     : {T_END * 1e3:.1f} ms per trajectory")

    save_rlc_dataset(trajs, out_path)
