"""
Step 2 – Training data generation for the Port-Hamiltonian buck converter PHNN.

Produces a dataset of simulated buck converter trajectories with diverse
initial conditions, input profiles, and durations.  Analytical state
derivatives (dx/dt) are stored alongside noisy state observations so that
Step 4 can use a derivative-matching loss.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from simulator import simulate_buck, compute_xdot, iL_vC_to_phi_q


# ---------------------------------------------------------------------------
# Input profile factory
# ---------------------------------------------------------------------------

def make_input_profile(
    profile_type: str,
    V_in: float,
    t_span: tuple[float, float],
    rng: np.random.Generator,
    alpha_range: tuple[float, float] = (0.2, 0.8),
    io_range: tuple[float, float] = (0.0, 2.0),
) -> Callable[[float], np.ndarray]:
    """Return a callable u_fn(t) -> np.ndarray([alpha, V_in, i_o]).

    Args:
        profile_type: One of "constant", "alpha_step", "load_step",
                      "alpha_ramp", "random_pwc".
        V_in:         Input voltage, constant within this trajectory [V].
        t_span:       (t_start, t_end) of the trajectory [s].
        rng:          NumPy random generator (consumed in-place).
        alpha_range:  (min, max) for duty-cycle samples.
        io_range:     (min, max) for load-current samples [A].

    Returns:
        Callable u_fn(t) -> shape-(3,) array [alpha, V_in, i_o].
    """
    t0, t1 = t_span
    a_lo, a_hi = alpha_range
    io_lo, io_hi = io_range

    if profile_type == "constant":
        alpha = float(rng.uniform(a_lo, a_hi))
        i_o = float(rng.uniform(io_lo, io_hi))

        def u_fn(t: float) -> np.ndarray:
            return np.array([alpha, V_in, i_o])

    elif profile_type == "alpha_step":
        alpha1 = float(rng.uniform(a_lo, a_hi))
        alpha2 = float(rng.uniform(a_lo, a_hi))
        t_sw = float(rng.uniform(t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0)))
        i_o = float(rng.uniform(io_lo, io_hi))

        def u_fn(t: float) -> np.ndarray:
            return np.array([alpha1 if t < t_sw else alpha2, V_in, i_o])

    elif profile_type == "load_step":
        alpha = float(rng.uniform(a_lo, a_hi))
        io1 = float(rng.uniform(io_lo, io_hi))
        io2 = float(rng.uniform(io_lo, io_hi))
        t_sw = float(rng.uniform(t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0)))

        def u_fn(t: float) -> np.ndarray:
            return np.array([alpha, V_in, io1 if t < t_sw else io2])

    elif profile_type == "alpha_ramp":
        alpha_start = float(rng.uniform(a_lo, a_hi))
        alpha_end = float(rng.uniform(a_lo, a_hi))
        i_o = float(rng.uniform(io_lo, io_hi))
        duration = t1 - t0

        def u_fn(t: float) -> np.ndarray:
            frac = float(np.clip((t - t0) / duration, 0.0, 1.0))
            return np.array([alpha_start + frac * (alpha_end - alpha_start), V_in, i_o])

    elif profile_type == "random_pwc":
        # 2–4 piecewise-constant segments; both alpha and i_o switch.
        n_seg = int(rng.integers(2, 5))
        if n_seg > 1:
            switch_times = np.sort(
                rng.uniform(t0 + 0.1 * (t1 - t0), t0 + 0.9 * (t1 - t0), size=n_seg - 1)
            )
        else:
            switch_times = np.array([])
        alpha_levels = rng.uniform(a_lo, a_hi, size=n_seg)
        io_levels = rng.uniform(io_lo, io_hi, size=n_seg)
        # boundaries[1:] are the right edges of each segment
        boundaries = np.concatenate([[t0], switch_times, [t1 + 1.0]])

        def u_fn(t: float) -> np.ndarray:
            idx = int(np.searchsorted(boundaries[1:], t, side="right"))
            idx = min(idx, n_seg - 1)
            return np.array([alpha_levels[idx], V_in, io_levels[idx]])

    else:
        raise ValueError(f"Unknown profile_type: {profile_type!r}")

    return u_fn


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Simulate n_trajectories buck converter trajectories for PHNN training.

    Each trajectory uses randomly sampled initial conditions, a random input
    profile, and a random duration.  Analytical state derivatives (dPhi/dt,
    dq/dt) are stored as training labels.  Optional Gaussian measurement
    noise is added to the physical outputs i_L and V_C.

    Args:
        n_trajectories: Number of trajectories to simulate.
        params:         Dict with keys 'L' [H], 'C' [F], 'R' [Ω].
        config:         Dict with keys:
                            dt            [s] – output sampling interval
                            t_span_range  (t_min, t_max) [s]
                            iL_range      (min, max) initial i_L [A]
                            VC_range      (min, max) initial V_C [V]
                            Vin_range     (min, max) V_in per trajectory [V]
                            alpha_range   (min, max) duty cycle
                            io_range      (min, max) load current [A]
                            sigma_iL      measurement noise std on i_L [A]
                            sigma_VC      measurement noise std on V_C [V]
                            profile_weights  {profile_type: weight}
        seed:           Integer seed for full reproducibility.

    Returns:
        List of dicts, each with keys:
            t, Phi, q, i_L, V_C, alpha, V_in, i_o,
            dPhi_dt, dq_dt, profile_type
    """
    rng = np.random.default_rng(seed)
    L, C, R = params["L"], params["C"], params["R"]

    dt = config["dt"]
    t_lo, t_hi = config["t_span_range"]
    iL_lo, iL_hi = config["iL_range"]
    VC_lo, VC_hi = config["VC_range"]
    Vin_lo, Vin_hi = config["Vin_range"]
    alpha_range: tuple[float, float] = tuple(config["alpha_range"])  # type: ignore[assignment]
    io_range: tuple[float, float] = tuple(config["io_range"])  # type: ignore[assignment]
    sigma_iL: float = float(config.get("sigma_iL", 0.0))
    sigma_VC: float = float(config.get("sigma_VC", 0.0))

    pw: dict[str, float] = config.get(
        "profile_weights",
        {"constant": 0.4, "alpha_step": 0.2, "load_step": 0.2,
         "alpha_ramp": 0.1, "random_pwc": 0.1},
    )
    profile_types = list(pw.keys())
    weights = np.array([pw[k] for k in profile_types], dtype=float)
    weights /= weights.sum()

    trajectories: list[dict] = []

    for _ in range(n_trajectories):
        # --- initial conditions ---
        i_L_0 = float(rng.uniform(iL_lo, iL_hi))
        V_C_0 = float(rng.uniform(VC_lo, VC_hi))
        Phi_0, q_0 = iL_vC_to_phi_q(
            np.array([i_L_0]), np.array([V_C_0]), L, C
        )
        x0 = np.array([float(Phi_0[0]), float(q_0[0])])

        # --- exogenous parameters ---
        t_end_raw = float(rng.uniform(t_lo, t_hi))
        # Snap to an exact dt multiple so np.arange in simulate_buck never
        # overshoots t_span[1] due to floating-point rounding.
        n_steps = max(1, round(t_end_raw / dt))
        t_end = float(n_steps * dt)
        V_in = float(rng.uniform(Vin_lo, Vin_hi))
        profile_type = str(rng.choice(profile_types, p=weights))

        u_fn = make_input_profile(
            profile_type=profile_type,
            V_in=V_in,
            t_span=(0.0, t_end),
            rng=rng,
            alpha_range=alpha_range,
            io_range=io_range,
        )

        result = simulate_buck(x0, u_fn, (0.0, t_end), params, dt=dt)

        Phi = result["Phi"]
        q = result["q"]
        alpha_arr = result["alpha"]
        V_in_arr = result["V_in"]
        i_o_arr = result["i_o"]
        i_L_clean = result["i_L"]   # Phi / L, no noise
        V_C_clean = result["V_C"]   # q / C, no noise

        # Vectorised analytical derivatives (equivalent to looped compute_xdot)
        dPhi_dt = alpha_arr * V_in_arr - R * i_L_clean - V_C_clean
        dq_dt = i_L_clean - i_o_arr

        # Measurement noise on physical observations only
        N = len(Phi)
        i_L = i_L_clean + (rng.standard_normal(N) * sigma_iL if sigma_iL > 0 else 0.0)
        V_C = V_C_clean + (rng.standard_normal(N) * sigma_VC if sigma_VC > 0 else 0.0)

        trajectories.append({
            "t": result["t"],
            "Phi": Phi,
            "q": q,
            "i_L": i_L,
            "V_C": V_C,
            "alpha": alpha_arr,
            "V_in": V_in_arr,
            "i_o": i_o_arr,
            "dPhi_dt": dPhi_dt,
            "dq_dt": dq_dt,
            "profile_type": profile_type,
        })

    return trajectories


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_dataset(trajectories: list[dict], path: str | Path) -> None:
    """Serialise a list of trajectory dicts to a single .npz file.

    Each array is stored as 'traj_{i}_{key}'.  Metadata (count, profile
    types) is stored as separate arrays.

    Args:
        trajectories: Output of generate_dataset.
        path:         Destination path (suffix .npz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    profile_types_list: list[str] = []

    required_keys = ("t", "Phi", "q", "i_L", "V_C", "alpha", "V_in", "i_o")
    optional_keys = ("dPhi_dt", "dq_dt")
    _array_keys = required_keys + tuple(
        key for key in optional_keys if all(key in traj for traj in trajectories)
    )

    for i, traj in enumerate(trajectories):
        prefix = f"traj_{i}"
        for key in _array_keys:
            arrays[f"{prefix}_{key}"] = traj[key]
        profile_types_list.append(str(traj["profile_type"]))

    arrays["profile_types"] = np.array(profile_types_list)
    arrays["n_trajectories"] = np.array(len(trajectories))
    arrays["array_keys"] = np.array(_array_keys)

    np.savez(path, **arrays)


def load_dataset(path: str | Path) -> list[dict]:
    """Load a dataset written by save_dataset.

    Args:
        path: Path to the .npz file.

    Returns:
        List of trajectory dicts with the same schema as generate_dataset.
    """
    data = np.load(Path(path), allow_pickle=False)

    n = int(data["n_trajectories"])
    profile_types = [str(pt) for pt in data["profile_types"]]

    if "array_keys" in data:
        _array_keys = tuple(str(key) for key in data["array_keys"])
    else:
        legacy_keys = ("t", "Phi", "q", "i_L", "V_C", "alpha", "V_in", "i_o",
                       "dPhi_dt", "dq_dt")
        _array_keys = tuple(
            key for key in legacy_keys if f"traj_0_{key}" in data.files
        )

    trajectories: list[dict] = []
    for i in range(n):
        prefix = f"traj_{i}"
        traj: dict = {key: data[f"{prefix}_{key}"] for key in _array_keys}
        traj["profile_type"] = profile_types[i]
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = {"L": 100e-6, "C": 100e-6, "R": 0.1}

    config: dict = {
        "dt": 10e-6,
        "t_span_range": (5e-3, 20e-3),
        "iL_range": (-1.0, 3.0),
        "VC_range": (0.0, 10.0),
        "Vin_range": (10.0, 14.0),
        "alpha_range": (0.2, 0.8),
        "io_range": (0.0, 2.0),
        "sigma_iL": 0.02,
        "sigma_VC": 0.05,
        "profile_weights": {
            "constant": 0.40,
            "alpha_step": 0.20,
            "load_step": 0.20,
            "alpha_ramp": 0.10,
            "random_pwc": 0.10,
        },
    }

    print("Generating 100 trajectories …")
    trajs = generate_dataset(n_trajectories=100, params=params, config=config, seed=42)

    out = Path(__file__).parent.parent / "data" / "train.npz"
    save_dataset(trajs, out)

    total = sum(len(t["t"]) for t in trajs)
    print(f"Done.  {len(trajs)} trajectories, {total:,} total timesteps.")
    print(f"Saved → {out}")
