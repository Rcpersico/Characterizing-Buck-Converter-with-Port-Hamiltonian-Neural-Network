"""
PLECS RPC dataset generation for Switched PHNN training.

Produces high-resolution (1 MHz) switching-simulation trajectories 
retaining the full switching ripple and discontinuous dynamics.

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
        dt_out = 1e-6  # 1 MHz uniform grid

        solver_opts = {
            'Solver':  'auto',
            'MaxStep': 1e-7, # Tighter MaxStep for switched resolution
        }

        if initial_system_state is None:
            t_out_forced = np.arange(0.0, t_end + dt_out * 0.5, dt_out).tolist()
            solver_opts['StopTime'] = float(t_end)
            solver_opts['OutputTimes'] = t_out_forced
        else:
            solver_opts['TimeSpan'] = float(t_end)
            solver_opts['InitialSystemState'] = initial_system_state

        opt_struct = {
            'ModelVars': {k: float(v) for k, v in model_vars.items()},
            'SolverOpts': solver_opts,
        }
        return self.server.plecs.simulate(self.model_name, opt_struct)

    def get_system_state(self) -> dict:
        return self.server.plecs.get(self.model_name, "SystemState")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _interpolate_raw(
    raw: dict,
    t_out: np.ndarray,
    iL_idx: int,
    VC_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate raw PLECS output onto a uniform high-rate grid.

    No cycle averaging. Keeps all switching ripple.
    """
    t_raw = np.asarray(raw["Time"], dtype=float)
    if t_raw.size < 2:
        raise ValueError("PLECS returned fewer than two time samples.")
    t_raw = t_raw - t_raw[0]

    iL_raw = np.asarray(raw["Values"][iL_idx], dtype=float)
    VC_raw = np.asarray(raw["Values"][VC_idx], dtype=float)

    i_L = np.interp(t_out, t_raw, iL_raw)
    V_C = np.interp(t_out, t_raw, VC_raw)
    return i_L, V_C


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_plecs_switched_dataset(
    simulator: PLECSRPCSimulator,
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Generate raw switched trajectories with mid-trajectory load steps."""
    rng = np.random.default_rng(seed)
    L = float(params["L"])
    C = float(params["C"])
    ESR_L = float(params["ESR_L"])
    ESR_C = float(params.get("ESR_C", 0.0))
    cycle_freq = float(params["cycle_frequency"])
    T_sw = 1.0 / cycle_freq

    dt = float(config["dt"])
    Vin_lo, Vin_hi = config["Vin_range"]
    a_lo, a_hi = config["alpha_range"]
    Rl_lo, Rl_hi = config["Rload_range"]
    sigma_iL = float(config.get("sigma_iL", 0.0))
    sigma_VC = float(config.get("sigma_VC", 0.0))
    iL_idx = int(config.get("iL_signal_idx", 0))
    VC_idx = int(config.get("VC_signal_idx", 1))
    n_steps = int(config.get("n_load_steps", 1))
    seg_dur_range = config.get("segment_duration_range", (1e-3, 2.5e-3))

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
                    raise RuntimeError("PLECS did not return a SystemState.")
            except Exception as exc:
                print(f"  [{k + 1}/{n_trajectories}] seg {seg_idx} error: {exc}")
                sim_ok = False
                break

            t_seg = np.arange(0.0, seg_t + dt * 0.5, dt)
            iL_seg, Vterm_seg = _interpolate_raw(raw, t_seg, iL_idx, VC_idx)

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
        
        # Calculate instantaneous v_sw
        v_sw_arr = np.zeros(N)
        t_mod = t_out % T_sw
        # Add small epsilon to avoid float rounding errors exactly on the edge
        v_sw_arr[t_mod < (alpha * T_sw + 1e-10)] = V_in

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
                "v_sw": v_sw_arr,
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
        "dt": 1e-6,
        "Vin_range": (10.0, 14.0),
        "alpha_range": (0.2, 0.8),
        "Rload_range": (2.0, 5.0),
        "sigma_iL": 0.02, # same noise std as averaged to be comparable, though maybe could be larger
        "sigma_VC": 0.05,
        "iL_signal_idx": 0,
        "VC_signal_idx": 1,
        "n_load_steps": 1,
        "segment_duration_range": (1e-3, 2.5e-3),
    }

    print("Generating 50 raw PLECS trajectories ...")
    trajs = generate_plecs_switched_dataset(sim, 50, params, config, seed=123)

    out = Path(__file__).parent.parent / "data" / "plecs_switched_train.npz"
    save_dataset(trajs, str(out))

    total = sum(len(t["t"]) for t in trajs)
    print(f"\nDone. {len(trajs)} trajectories, {total:,} total timesteps.")
    print(f"Saved -> {out}")
