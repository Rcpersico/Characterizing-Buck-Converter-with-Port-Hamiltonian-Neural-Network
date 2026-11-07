"""
PLECS RPC dataset generation for SwitchedBuckPHNN training.

Produces high-resolution (1 MHz) switching-simulation trajectories retaining
the full switching ripple, dead-time effects, and startup transients.

Key differences from the averaged dataset generator:
  - No cycle averaging. Raw PLECS output is interpolated onto a uniform 1 MHz grid.
  - Sample rate: 1 MHz (dt = 1e-6 s), giving 10 samples per switching cycle at
    100 kHz. Enough to resolve ripple while remaining tractable.
  - PLECS MaxStep = 1e-7 (100 ns) for accurate switched waveforms.
  - Shorter trajectories: 1–2.5 ms each (50x less wall time per trajectory than
    the averaged version, but 50x more samples per ms).
  - Saves v_sw(t) = V_in during on-time, 0 during off-time, computed from the
    PWM pattern for each sample. This is the instantaneous switch state used as
    input to SwitchedBuckPHNN.

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

Output schema (same dict structure as generate_plecs_dataset.py):
    t, i_L, V_C, V_terminal, alpha, V_in, v_sw, i_o, profile_type, Phi, q

Saved to data/plecs_switched_train.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow imports from src/ (generate_dataset.save_dataset / load_dataset)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generate_dataset import load_dataset, save_dataset  # noqa: F401

try:
    import xmlrpc.client
    _HAS_XMLRPC = True
except ImportError:
    _HAS_XMLRPC = False


# ---------------------------------------------------------------------------
# PLECS RPC interface
# ---------------------------------------------------------------------------


class PLECSRPCSimulator:
    """Thin wrapper around the PLECS XML-RPC API."""

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

    def simulate(
        self,
        model_vars: dict,
        t_end: float,
        initial_system_state: dict | None = None,
    ) -> dict:
        dt_out = 1e-6  # 1 MHz uniform output grid

        solver_opts: dict = {
            "Solver": "auto",
            "MaxStep": 1e-7,  # 100 ns — resolves switching transitions
        }

        if initial_system_state is None:
            t_out_forced = np.arange(0.0, t_end + dt_out * 0.5, dt_out).tolist()
            solver_opts["StopTime"] = float(t_end)
            solver_opts["OutputTimes"] = t_out_forced
        else:
            solver_opts["TimeSpan"] = float(t_end)
            solver_opts["InitialSystemState"] = initial_system_state

        opt_struct = {
            "ModelVars": {k: float(v) for k, v in model_vars.items()},
            "SolverOpts": solver_opts,
        }
        return self.server.plecs.simulate(self.model_name, opt_struct)

    def get_system_state(self) -> dict:
        """Return the final PLECS system state from the latest simulation."""
        return self.server.plecs.get(self.model_name, "SystemState")


# ---------------------------------------------------------------------------
# Raw-data resampling (no cycle averaging)
# ---------------------------------------------------------------------------


def _interpolate_raw(
    raw: dict,
    t_out: np.ndarray,
    iL_idx: int,
    VC_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate raw PLECS output onto a uniform high-rate grid.

    No cycle averaging is applied. All switching ripple is preserved.
    """
    t_raw = np.asarray(raw["Time"], dtype=float)
    if t_raw.size < 2:
        raise ValueError("PLECS returned fewer than two time samples.")
    t_raw = t_raw - t_raw[0]  # normalise to 0-based

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
    """Generate raw switched PLECS trajectories with mid-trajectory load steps.

    Each trajectory consists of ``n_load_steps + 1`` segments at different
    R_load values, chained via PLECS SystemState. The load steps create
    persistent transients with large di_L/dt that are invisible to the averaged
    model but must be captured by SwitchedBuckPHNN.

    Returns a list of trajectory dicts with the same schema as
    generate_plecs_dataset.py so the same training pipeline can be reused.
    The extra key ``v_sw`` contains the instantaneous switch-node voltage [V].
    """
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
            # Snap to nearest dt multiple to avoid np.arange overshoot
            n_samp = max(1, round(t_raw / dt))
            seg_durs.append(float(n_samp * dt))

        t_all: list[np.ndarray] = []
        iL_all: list[np.ndarray] = []
        Vterm_all: list[np.ndarray] = []
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
                raw = simulator.simulate(model_vars, seg_t, initial_system_state=initial_state)
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
                # Drop the duplicated boundary sample
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

        # Assign R_load per sample based on segment boundaries
        segment_edges = np.cumsum(seg_durs)
        seg_idx_for_sample = np.searchsorted(segment_edges[:-1], t_out, side="right")
        R_load_arr = np.asarray(R_loads, dtype=float)[seg_idx_for_sample]
        i_o_arr = V_terminal_clean / R_load_arr
        i_cap_clean = i_L_clean - i_o_arr
        V_C_clean = V_terminal_clean - ESR_C * i_cap_clean

        # DCM check: warn if more than 5% of inductor current is negative
        pct_negative = float(np.mean(i_L_clean < -0.1)) * 100
        if pct_negative > 5.0:
            print(
                f"  WARNING: traj {k + 1} has {pct_negative:.1f}% negative i_L "
                f"— possible DCM. alpha={alpha:.2f}, R_loads={R_loads}"
            )

        # Instantaneous switch-node voltage: V_in during on-time, 0 during off-time.
        # t_out is the global time so t_out % T_sw gives the PWM phase correctly.
        t_mod = t_out % T_sw
        v_sw_arr = np.where(t_mod < alpha * T_sw + 1e-10, V_in, 0.0)

        alpha_arr = np.full(N, alpha)
        V_in_arr = np.full(N, V_in)

        # Optional measurement noise (applied only to observables, not to v_sw)
        i_L = i_L_clean + (rng.standard_normal(N) * sigma_iL if sigma_iL > 0 else 0.0)
        V_C = V_C_clean + (rng.standard_normal(N) * sigma_VC if sigma_VC > 0 else 0.0)

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
            r_str = " -> ".join(f"{r:.1f}" for r in R_loads)
            print(
                f"  [{k + 1}/{n_trajectories}]  V_in={V_in:.1f}V  "
                f"alpha={alpha:.3f}  R_load: {r_str} Ohm  "
                f"t_total={t_offset * 1e3:.1f}ms  N={N}"
            )

    return trajectories


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    model_path = ROOT / "models" / "buck_converter.plecs"
    sim = PLECSRPCSimulator(model_path)

    # Physical parameters matching the PLECS model
    params = {
        "L": 100e-6,       # 100 µH
        "C": 220e-6,       # 220 µF  (match actual PLECS model)
        "ESR_L": 0.1,      # 0.1 Ohm inductor series resistance
        "ESR_C": 0.05,     # 0.05 Ohm capacitor ESR
        "cycle_frequency": 100e3,  # 100 kHz switching frequency
    }

    # Config: 1 MHz sampling, short trajectories, CCM-safe R_load range
    config = {
        "dt": 1e-6,                          # 1 MHz — 10 samples per switching cycle
        "Vin_range": (10.0, 14.0),
        "alpha_range": (0.2, 0.8),
        "Rload_range": (2.0, 5.0),           # CCM-restricted (R_max ≈ 5 Ohm at worst case)
        "sigma_iL": 0.02,                    # 20 mA measurement noise
        "sigma_VC": 0.05,                    # 50 mV measurement noise
        "iL_signal_idx": 0,
        "VC_signal_idx": 1,
        "n_load_steps": 1,                   # 2 segments per trajectory (1 load step)
        "segment_duration_range": (1e-3, 2.5e-3),  # 1–2.5 ms per segment
    }

    # 60 trajectories: 50× more samples/ms than averaged, kept small for tractability
    n_traj = 60
    print(f"Generating {n_traj} raw switched PLECS trajectories ...")
    trajs = generate_plecs_switched_dataset(sim, n_traj, params, config, seed=123)

    out = ROOT / "data" / "plecs_switched_train.npz"
    save_dataset(trajs, str(out))

    total = sum(len(t["t"]) for t in trajs)
    print(f"\nDone. {len(trajs)} trajectories, {total:,} total timesteps.")
    print(f"Saved -> {out}")
