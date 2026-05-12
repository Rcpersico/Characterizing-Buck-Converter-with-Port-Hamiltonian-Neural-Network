"""
PLECS RPC dataset generation for PHNN training.

Produces switching-simulation trajectories with the same output schema as
generate_dataset.py so both can feed the same PHNN training pipeline.

Expected PLECS model variables:
    V_in            Input voltage [V]
    L               Inductance [H]
    C               Capacitance [F]
    ESR_L           Inductor series resistance [Ohm]
    ESR_C           Capacitor ESR [Ohm]
    R_load          Load resistance [Ohm]
    alpha           Duty cycle [0..1]
    cycle_frequency Switching frequency [Hz]

Expected logged signals, by index:
    0: inductor current i_L [A]
    1: output terminal voltage V_terminal [V]

The saved ``V_C`` state is the ideal capacitor voltage, not the output
terminal voltage. With capacitor ESR, PLECS measures
``V_terminal = V_C + ESR_C * i_C``; the generator subtracts this ESR drop so
the dataset state matches the averaged PH model.

The load-step generator chains multiple PLECS simulation segments by restoring
the final SystemState from one segment as the initial state of the next. This
creates real mid-trajectory load transients for identifying L.
"""

from __future__ import annotations

import sys
import xmlrpc.client
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import load_dataset, save_dataset  # noqa: F401


# ---------------------------------------------------------------------------
# PLECS RPC interface
# ---------------------------------------------------------------------------


class PLECSRPCSimulator:
    def __init__(self, model_path: str | Path, port: int = 1080) -> None:
        self.model_path = str(Path(model_path).resolve()).replace("\\", "/")
        self.model_name = Path(model_path).stem
        self.server = xmlrpc.client.ServerProxy(f"http://localhost:{port}/RPC2")
        try:
            self.server.plecs.load(self.model_path)
            print(f"Loaded {self.model_name}")
        except xmlrpc.client.Fault as e:
            if "already" not in e.faultString.lower():
                raise
        self.server.plecs.set(f"{self.model_name}/V_in", "V", "V_in")

    def simulate(
        self,
        model_vars: dict,
        t_end: float,
        initial_system_state: dict | None = None,
    ) -> dict:
        dt_out = 0.5e-6  # 20 points per switching cycle

        solver_opts = {
            'Solver':  'auto',
            'MaxStep': 1e-7,
        }

        if initial_system_state is None:
            # First segment: time runs 0 → t_end, OutputTimes is easy
            t_out_forced = np.arange(0.0, t_end + dt_out * 0.5, dt_out).tolist()
            solver_opts['StopTime'] = float(t_end)
            solver_opts['OutputTimes'] = t_out_forced
        else:
            # Subsequent segments: let PLECS use natural output
            # MaxStep=1e-7 already ensures ≥100 points per switching cycle
            t_start_abs = float(initial_system_state['StopTime'])
            solver_opts['TimeSpan'] = float(t_end)
            solver_opts['InitialSystemState'] = initial_system_state
            # No OutputTimes — PLECS outputs at MaxStep resolution naturally

        opt_struct = {
            'ModelVars': {k: float(v) for k, v in model_vars.items()},
            'SolverOpts': solver_opts,
        }
        return self.server.plecs.simulate(self.model_name, opt_struct)

    def get_system_state(self) -> dict:
        """Return the final PLECS system state from the latest simulation."""
        return self.server.plecs.get(self.model_name, "SystemState")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _resample(
    raw: dict,
    t_out: np.ndarray,
    iL_idx: int,
    VC_idx: int,
    f_switch: float = 100e3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true cycle-averaged i_L and V_C from raw PLECS output.

    Integrates over each switching period using trapezoidal quadrature on the
    raw non-uniform PLECS timesteps — equivalent to a PLECS Mean Value block,
    but computed in Python without changing the PLECS model.

    No interpolation before averaging: this is the key difference from the
    previous moving-average approach that produced L=208 µH instead of 100 µH.
    """
    T_sw = 1.0 / f_switch
    t_raw = np.asarray(raw["Time"], dtype=float)
    if t_raw.size < 2:
        raise ValueError("PLECS returned fewer than two time samples.")
    # Normalise to zero so t_raw aligns with t_out (always 0-based per segment).
    t_raw = t_raw - t_raw[0]

    iL_raw = np.asarray(raw["Values"][iL_idx], dtype=float)
    VC_raw = np.asarray(raw["Values"][VC_idx], dtype=float)

    t_start = t_raw[0]   # 0.0 after normalisation
    t_end = t_raw[-1]

    n_cycles = int(np.floor((t_end - t_start) / T_sw))
    if n_cycles < 2:
        # Fallback for very short segments.
        i_L = np.interp(t_out, t_raw, iL_raw)
        V_C = np.interp(t_out, t_raw, VC_raw)
        return i_L, V_C

    t_boundaries = t_start + np.arange(n_cycles + 1) * T_sw
    iL_avg = np.zeros(n_cycles)
    VC_avg = np.zeros(n_cycles)
    t_mid = np.zeros(n_cycles)

    for k in range(n_cycles):
        ta, tb = t_boundaries[k], t_boundaries[k + 1]
        t_mid[k] = 0.5 * (ta + tb)

        mask = (t_raw >= ta) & (t_raw <= tb)
        t_seg = np.concatenate([[ta], t_raw[mask], [tb]])
        iL_seg = np.concatenate(
            [[np.interp(ta, t_raw, iL_raw)], iL_raw[mask], [np.interp(tb, t_raw, iL_raw)]]
        )
        VC_seg = np.concatenate(
            [[np.interp(ta, t_raw, VC_raw)], VC_raw[mask], [np.interp(tb, t_raw, VC_raw)]]
        )

        iL_avg[k] = np.trapz(iL_seg, t_seg) / T_sw
        VC_avg[k] = np.trapz(VC_seg, t_seg) / T_sw

    # Cycle-averaged signals are smooth — safe to interpolate onto t_out.
    i_L = np.interp(t_out, t_mid, iL_avg)
    V_C = np.interp(t_out, t_mid, VC_avg)
    return i_L, V_C


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_plecs_dataset_with_steps(
    simulator: PLECSRPCSimulator,
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Generate trajectories with mid-trajectory load steps for rich L ID.

    Each trajectory consists of 2-3 segments with different R_load values.
    The load steps create persistent transients where di_L/dt is large,
    giving the optimizer strong signal to identify L.

    The output schema is identical to generate_plecs_dataset() so the same
    training pipeline can be used.
    """
    rng = np.random.default_rng(seed)
    L = float(params["L"])
    C = float(params["C"])
    ESR_L = float(params["ESR_L"])
    ESR_C = float(params.get("ESR_C", 0.0))
    cycle_freq = float(params["cycle_frequency"])

    dt = float(config["dt"])
    Vin_lo, Vin_hi = config["Vin_range"]
    a_lo, a_hi = config["alpha_range"]
    Rl_lo, Rl_hi = config["Rload_range"]
    sigma_iL = float(config.get("sigma_iL", 0.0))
    sigma_VC = float(config.get("sigma_VC", 0.0))
    iL_idx = int(config.get("iL_signal_idx", 0))
    VC_idx = int(config.get("VC_signal_idx", 1))
    n_steps = int(config.get("n_load_steps", 2))
    seg_dur_range = config.get("segment_duration_range", (5e-3, 15e-3))

    trajectories: list[dict] = []

    for k in range(n_trajectories):
        V_in = float(rng.uniform(Vin_lo, Vin_hi))
        alpha = float(rng.uniform(a_lo, a_hi))

        R_loads = [float(rng.uniform(Rl_lo, Rl_hi)) for _ in range(n_steps + 1)]

        seg_durs = []
        for _ in range(n_steps + 1):
            t_raw = float(rng.uniform(*seg_dur_range))
            n_samp = max(1, round(t_raw / dt))
            seg_durs.append(float(n_samp * dt))

        t_all, iL_all, Vterm_all = [], [], []
        t_offset = 0.0
        initial_state: dict | None = None
        sim_ok = True

        for seg_idx, (R_load, seg_t) in enumerate(zip(R_loads, seg_durs)):
            model_vars = {
                "V_in": V_in,
                "L": L,
                "C": C,
                "ESR_L": ESR_L,
                "ESR_C": ESR_C,
                "R_load": R_load,
                "alpha": alpha,
                "cycle_frequency": cycle_freq,
            }

            try:
                raw = simulator.simulate(
                    model_vars,
                    seg_t,
                    initial_system_state=initial_state,
                )
                initial_state = simulator.get_system_state()
                if not initial_state:
                    raise RuntimeError(
                        "PLECS did not return a SystemState for segment chaining."
                    )
            except Exception as exc:
                print(f"  [{k + 1}/{n_trajectories}] seg {seg_idx} error: {exc}")
                sim_ok = False
                break

            t_seg = np.arange(0.0, seg_t + dt * 0.5, dt)
            iL_seg, Vterm_seg = _resample(raw, t_seg, iL_idx, VC_idx, f_switch=cycle_freq)

            if seg_idx == 0:
                t_all.append(t_seg + t_offset)
                iL_all.append(iL_seg)
                Vterm_all.append(Vterm_seg)
            else:
                t_all.append(t_seg[1:] + t_offset)
                iL_all.append(iL_seg[1:])
                Vterm_all.append(Vterm_seg[1:])

            t_offset += seg_t

        if not sim_ok:
            continue

        t_out = np.concatenate(t_all)
        i_L_clean = np.concatenate(iL_all)
        V_terminal_clean = np.concatenate(Vterm_all)
        N = len(t_out)

        segment_edges = np.cumsum(seg_durs)
        seg_idx_for_sample = np.searchsorted(segment_edges[:-1], t_out, side="right")
        R_load_arr = np.asarray(R_loads, dtype=float)[seg_idx_for_sample]
        i_o_arr = V_terminal_clean / R_load_arr
        i_cap_clean = i_L_clean - i_o_arr
        V_C_clean = V_terminal_clean - ESR_C * i_cap_clean

        pct_negative = float(np.mean(i_L_clean < -0.1)) * 100
        if pct_negative > 5.0:
            print(
                f"  WARNING: traj {k + 1} has {pct_negative:.1f}% negative i_L - "
                f"possible DCM. alpha={alpha:.2f}, R_loads={R_loads}"
            )

        alpha_arr = np.full(N, alpha)
        V_in_arr = np.full(N, V_in)

        i_L = i_L_clean + (
            rng.standard_normal(N) * sigma_iL if sigma_iL > 0 else 0.0
        )
        V_C = V_C_clean + (
            rng.standard_normal(N) * sigma_VC if sigma_VC > 0 else 0.0
        )

        trajectories.append(
            {
                "t": t_out,
                "i_L": i_L,
                "V_C": V_C,
                "V_terminal": V_terminal_clean,
                "alpha": alpha_arr,
                "V_in": V_in_arr,
                "i_o": i_o_arr,
                "profile_type": "load_step",
                "Phi": L * i_L_clean,
                "q": C * V_C_clean,
            }
        )

        if (k + 1) % 10 == 0 or k == 0:
            r_str = " -> ".join(f"{r:.1f} Ohm" for r in R_loads)
            print(
                f"  [{k + 1}/{n_trajectories}]  V_in={V_in:.1f}V  "
                f"alpha={alpha:.3f}  R_load: {r_str}  "
                f"t_total={t_offset * 1e3:.1f}ms  N={N}"
            )

    return trajectories


def generate_plecs_dataset(
    simulator: PLECSRPCSimulator,
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Simulate fixed-load PLECS trajectories.

    This legacy path is retained for comparison experiments. New PHNN
    identification should use generate_plecs_dataset_with_steps().
    """
    rng = np.random.default_rng(seed)
    L = float(params["L"])
    C = float(params["C"])
    ESR_L = float(params["ESR_L"])
    ESR_C = float(params.get("ESR_C", 0.0))
    cycle_freq = float(params["cycle_frequency"])

    dt = float(config["dt"])
    t_lo, t_hi = config["t_span_range"]
    Vin_lo, Vin_hi = config["Vin_range"]
    a_lo, a_hi = config["alpha_range"]
    Rl_lo, Rl_hi = config["Rload_range"]
    iL_idx = int(config.get("iL_signal_idx", 0))
    VC_idx = int(config.get("VC_signal_idx", 1))

    trajectories: list[dict] = []

    for k in range(n_trajectories):
        V_in = float(rng.uniform(Vin_lo, Vin_hi))
        alpha = float(rng.uniform(a_lo, a_hi))
        R_load = float(rng.uniform(Rl_lo, Rl_hi))
        n_steps = max(1, round(float(rng.uniform(t_lo, t_hi)) / dt))
        t_end = float(n_steps * dt)

        model_vars = {
            "V_in": V_in,
            "L": L,
            "C": C,
            "ESR_L": ESR_L,
            "ESR_C": ESR_C,
            "R_load": R_load,
            "alpha": alpha,
            "cycle_frequency": cycle_freq,
        }

        try:
            raw = simulator.simulate(model_vars, t_end)
        except Exception as exc:
            print(f"  [{k + 1}/{n_trajectories}] simulation error: {exc}")
            continue

        t_out = np.arange(0.0, t_end + dt * 0.5, dt)
        i_L_clean, V_terminal_clean = _resample(raw, t_out, iL_idx, VC_idx, f_switch=cycle_freq)
        i_o_clean = V_terminal_clean / R_load
        i_cap_clean = i_L_clean - i_o_clean
        V_C_clean = V_terminal_clean - ESR_C * i_cap_clean

        N = len(t_out)
        trajectories.append(
            {
                "t": t_out,
                "i_L": i_L_clean,
                "V_C": V_C_clean,
                "V_terminal": V_terminal_clean,
                "alpha": np.full(N, alpha),
                "V_in": np.full(N, V_in),
                "i_o": i_o_clean,
                "profile_type": "constant",
                "Phi": L * i_L_clean,
                "q": C * V_C_clean,
            }
        )

        if (k + 1) % 10 == 0 or k == 0:
            print(
                f"  [{k + 1}/{n_trajectories}]  V_in={V_in:.1f}V  "
                f"alpha={alpha:.3f}  R_load={R_load:.1f} Ohm  "
                f"t_end={t_end * 1e3:.1f}ms"
            )

    return trajectories


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "buck_converter.plecs"
    sim = PLECSRPCSimulator(model_path)

    params = {
        "L": 50e-6,
        "C": 200e-6,
        "ESR_L": 0.1,
        "ESR_C": 0.05,
        "cycle_frequency": 100e3,
    }

    config = {
        "dt": 50e-6,
        "Vin_range": (10.0, 14.0),
        "alpha_range": (0.2, 0.8),
        "Rload_range": (2.0, 5.0),
        "sigma_iL": 0.02,
        "sigma_VC": 0.05,
        "iL_signal_idx": 0,
        "VC_signal_idx": 1,
        "n_load_steps": 2,
        "segment_duration_range": (10e-3, 15e-3),
    }

    print("Generating 100 PLECS trajectories with load steps ...")
    trajs = generate_plecs_dataset_with_steps(sim, 100, params, config, seed=42)

    out = Path(__file__).parent.parent / "data" / "plecs_physical_train.npz"
    save_dataset(trajs, str(out))

    total = sum(len(t["t"]) for t in trajs)
    print(f"\nDone. {len(trajs)} trajectories, {total:,} total timesteps.")
    print(f"Saved -> {out}")
