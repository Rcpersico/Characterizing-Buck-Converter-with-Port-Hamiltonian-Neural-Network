"""
PLECS dataset generator for the 12V->1V VRM operating point.

Models a SINGLE-PHASE synchronous buck converter at data-center VRM conditions,
using power-stage parameters from:
    Qian/Yan/Cui, COMPEL 2025, Table II
    (two-phase series-capacitor buck topology -- SCB VRM)

This generator models the CONVENTIONAL single-phase synchronous buck at the
same operating point (L=440 nH, C=200 uF, R_L=3 mOhm, V_in=12 V, f_sw=1.67 MHz)
as a baseline study. It does NOT implement the two-phase series-capacitor
topology from that paper.

Regime convention: synchronous buck, no freewheeling diode -- inductor current
can go negative under light load.
  - Unidirectional: i_L >= 0  (forward conduction throughout)
  - Bidirectional:  i_L <  0  (reverse current through synchronous MOSFET)
    In a converter WITH a diode this would clamp to classical DCM; here
    current actually flows backwards rather than clamping to zero.

Key differences from generate_plecs_bidir_dataset.py (100uH / 100kHz model):
  - L:   100 uH  -> 440 nH            (~227x smaller inductance)
  - C:   100 uF  -> 200 uF            (2x larger capacitance)
  - ESR_L: 0.1 Ohm -> 3 mOhm         (between R_ds=2.2mOhm and R_co=5mOhm, Table II)
  - ESR_C: 0.05 Ohm -> 0.0            (set to zero, same as bidir model)
  - V_in:  8-16 V sweep -> 12.0 V     (fixed data-center input rail)
  - f_sw:  100 kHz -> 1.67 MHz        (~16.7x faster switching)
  - alpha_range: (0.2, 0.8) -> (0.05, 0.20)  (nominal duty 1/12 ~ 0.083 for 1V)
  - Rload_range: (0.1, 50) Ohm -> (0.07, 0.5) Ohm  (i_o ~ 2-14 A at V_out~1V)
  - Timescales rescaled ~10.7x faster to match VRM LC dynamics
    (f_LC: ~1.6 kHz -> ~17.0 kHz)

Timing rationale:
  - f_LC_old ~ 1/(2*pi*sqrt(100uH * 100uF)) ~ 1592 Hz
  - f_LC_new ~ 1/(2*pi*sqrt(440nH * 200uF)) ~ 17000 Hz  -> ratio ~ 10.7
  - dt: 50 us -> 4.69 us   (~7.8 switching cycles per sample at 1.67 MHz)
  - OutputTimes spacing: 0.5 us -> 60 ns  (~10 output pts per switching cycle)
  - MaxStep: 100 ns -> 30 ns              (~20 internal solver steps per cycle)
  - segment_duration: 15-25 ms -> 1.4-2.3 ms  (~24-39 LC periods each)

IMPORTANT -- VRMPLECSSimulator subclass:
  The base PLECSRPCSimulator.simulate() hard-codes dt_out=0.5e-6 and MaxStep=1e-7.
  At 1.67 MHz (T_sw ~ 600 ns) those values give <1 output point per switching
  cycle, which would break the trapz cycle-averaging in _resample(). This file
  defines VRMPLECSSimulator, a subclass that overrides simulate() with the
  VRM-appropriate timing. No existing files are modified.

Usage (from bidir_buck_model/ as working directory):
    conda run -n nilm-gpu python src/generate_plecs_vrm_dataset.py --train-n 5
    conda run -n nilm-gpu python src/generate_plecs_vrm_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import save_dataset
from generate_plecs_dataset import PLECSRPCSimulator, _resample


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# VRM-timing PLECS simulator
# ---------------------------------------------------------------------------


class VRMPLECSSimulator(PLECSRPCSimulator):
    """PLECSRPCSimulator subclass with VRM-appropriate output and solver timing.

    The base class hard-codes dt_out=0.5 us and MaxStep=100 ns, suitable for
    a 100 kHz converter (T_sw=10 us). At 1.67 MHz (T_sw ~ 600 ns), those
    values leave fewer than 1 output point per switching cycle -- _resample()
    would produce garbage cycle averages.

    This subclass overrides simulate() to use:
      DT_OUT   = 60 ns   (~10 output points per T_sw at 1.67 MHz)
      MAX_STEP = 30 ns   (~20 internal solver steps per T_sw)

    All other behaviour (SystemState chaining, RPC connection, model loading,
    and _resample()) is inherited unchanged from PLECSRPCSimulator.
    """

    DT_OUT: float = 60e-9    # output spacing: ~10 pts/T_sw at 1.67 MHz
    MAX_STEP: float = 30e-9  # solver max step: ~20 steps/T_sw

    def simulate(
        self,
        model_vars: dict,
        t_end: float,
        initial_system_state: dict | None = None,
    ) -> dict:
        solver_opts: dict = {
            "Solver": "auto",
            "MaxStep": self.MAX_STEP,
        }

        if initial_system_state is None:
            # First segment: force output at DT_OUT spacing so _resample()
            # has ~10 raw points per switching cycle for trapz integration.
            # Clip to <= t_end: np.arange accumulates float error over ~30k
            # steps and can overshoot by a few ns, which PLECS rejects.
            t_out_arr = np.arange(0.0, t_end + self.DT_OUT * 0.5, self.DT_OUT)
            t_out_forced = t_out_arr[t_out_arr <= float(t_end)].tolist()
            solver_opts["StopTime"] = float(t_end)
            solver_opts["OutputTimes"] = t_out_forced
        else:
            # Continuation segments: let PLECS output at MaxStep=30 ns
            # resolution (~20 pts/T_sw), which is sufficient for _resample().
            solver_opts["TimeSpan"] = float(t_end)
            solver_opts["InitialSystemState"] = initial_system_state

        opt_struct = {
            "ModelVars": {k: float(v) for k, v in model_vars.items()},
            "SolverOpts": solver_opts,
        }
        return self.server.plecs.simulate(self.model_name, opt_struct)


# ---------------------------------------------------------------------------
# Alpha sampling helpers (unchanged from generate_plecs_bidir_dataset.py)
# ---------------------------------------------------------------------------


def _round_alpha(alpha: float, decimals: int = 2) -> float:
    return round(alpha, decimals)


def _sample_diverse_alpha(
    rng: np.random.Generator,
    prev_alpha: float,
    a_lo: float,
    a_hi: float,
    delta_min: float = 0.05,
    decimals: int = 2,
) -> float:
    # NOTE: delta_min default is 0.05 here (not 0.2 as in the bidir generator).
    # The VRM alpha range (0.05, 0.20) is only 0.15 wide vs the old range 0.6 wide.
    # Proportional scaling: delta_min_new = 0.2 * (0.15/0.6) = 0.05.
    # Using 0.2 here would make BOTH exclusion windows cover the entire range,
    # causing the fallback to only ever return a_lo or a_hi.
    lo1, hi1 = a_lo, prev_alpha - delta_min
    lo2, hi2 = prev_alpha + delta_min, a_hi

    valid: list[tuple[float, float]] = []
    if lo1 < hi1:
        valid.append((lo1, hi1))
    if lo2 < hi2:
        valid.append((lo2, hi2))

    if not valid:
        return _round_alpha(float(rng.choice([a_lo, a_hi])), decimals)

    widths = [r[1] - r[0] for r in valid]
    total = sum(widths)
    u = float(rng.uniform(0.0, total))
    cumulative = 0.0
    for (lo, hi), w in zip(valid, widths):
        if u <= cumulative + w:
            return _round_alpha(float(rng.uniform(lo, hi)), decimals)
        cumulative += w
    return _round_alpha(float(rng.uniform(*valid[-1])), decimals)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


def generate_plecs_vrm_dataset(
    simulator: VRMPLECSSimulator,
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Generate 12V->1V VRM sweep trajectories (alpha + load steps).

    Operating-point sweep: alpha in [0.05, 0.20] (nominal 1/12 ~ 0.083 for 1V
    output) and R_load in [0.07, 0.5] Ohm (i_o ~ 2-14 A at V_out ~ 1V),
    covering the load range of the COMPEL 2025 SCB VRM paper.

    Each returned trajectory dict contains a 'regime' field (float in [0, 1])
    equal to the fraction of samples where i_L < -0.05 A.  A nonzero regime
    means the converter entered bidirectional conduction (reverse current
    through the synchronous MOSFET).  In a converter WITH a freewheeling diode
    the same operating point would produce classical DCM (current clamped to
    zero); here the current actually goes negative.

    Args:
        simulator: Configured VRMPLECSSimulator instance.
        n_trajectories: Number of trajectories to attempt.
        params: PLECS model parameters (L, C, ESR_L, ESR_C, cycle_frequency).
        config: Dataset configuration dict (see driver for keys).
        seed: RNG seed for reproducibility.

    Returns:
        List of trajectory dicts with standard schema plus 'regime' field.
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
    delta_alpha_min = float(config.get("delta_alpha_min", 0.05))
    n_segs_lo = int(config.get("n_segs_range", (3, 4))[0])
    n_segs_hi = int(config.get("n_segs_range", (3, 4))[1])
    seg_dur_lo, seg_dur_hi = config.get("segment_duration_range", (1.4e-3, 2.3e-3))
    sigma_iL = float(config.get("sigma_iL", 0.0))
    sigma_VC = float(config.get("sigma_VC", 0.0))
    iL_idx = int(config.get("iL_signal_idx", 0))
    VC_idx = int(config.get("VC_signal_idx", 1))

    trajectories: list[dict] = []

    for k in range(n_trajectories):
        V_in = float(rng.uniform(Vin_lo, Vin_hi))  # nominally always 12.0 V for VRM
        n_segs = int(rng.integers(n_segs_lo, n_segs_hi + 1))

        alphas: list[float] = [_round_alpha(float(rng.uniform(a_lo, a_hi)))]
        for _ in range(n_segs - 1):
            alphas.append(
                _sample_diverse_alpha(rng, alphas[-1], a_lo, a_hi, delta_alpha_min)
            )

        R_loads = [float(rng.uniform(Rl_lo, Rl_hi)) for _ in range(n_segs)]

        seg_durs: list[float] = []
        for _ in range(n_segs):
            t_raw = float(rng.uniform(seg_dur_lo, seg_dur_hi))
            n_samp = max(1, round(t_raw / dt))
            seg_durs.append(float(n_samp * dt))

        t_all: list[np.ndarray] = []
        iL_all: list[np.ndarray] = []
        Vterm_all: list[np.ndarray] = []
        alpha_seg_all: list[np.ndarray] = []

        t_offset = 0.0
        sim_ok = True

        # Burn-in: run 500 us at the initial operating point so the converter
        # starts in steady state rather than from V_C=0, i_L=0.  Without this,
        # the VRM's tiny L=440nH and ESR=3mOhm (char. impedance ~47mOhm) cause
        # a huge underdamped current spike at startup (>40A in a 2-14A circuit).
        # 500 us ~ 6-7 x tau_settle (tau = 2L/R_L ~ 77 us) -> <1% residual error.
        WARMUP_T = 500e-6
        warmup_vars = {
            "V_in": V_in, "L": L, "C": C, "ESR_L": ESR_L, "ESR_C": ESR_C,
            "R_load": R_loads[0], "alpha": alphas[0],
            "cycle_frequency": cycle_freq,
        }
        try:
            simulator.simulate(warmup_vars, WARMUP_T, initial_system_state=None)
            initial_state: dict | None = simulator.get_system_state()
            if not initial_state:
                raise RuntimeError("PLECS did not return a SystemState after warmup.")
        except Exception as exc:
            print(f"  [{k + 1}/{n_trajectories}] warmup error: {exc}")
            continue

        for seg_idx in range(n_segs):
            alpha_seg = alphas[seg_idx]
            R_load = R_loads[seg_idx]
            seg_t = seg_durs[seg_idx]

            model_vars = {
                "V_in": V_in,
                "L": L,
                "C": C,
                "ESR_L": ESR_L,
                "ESR_C": ESR_C,
                "R_load": R_load,
                "alpha": alpha_seg,
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
            iL_seg, Vterm_seg = _resample(raw, t_seg, iL_idx, VC_idx, f_switch=cycle_freq)

            if seg_idx == 0:
                t_all.append(t_seg + t_offset)
                iL_all.append(iL_seg)
                Vterm_all.append(Vterm_seg)
                alpha_seg_all.append(np.full(len(t_seg), alpha_seg, dtype=np.float64))
            else:
                t_all.append(t_seg[1:] + t_offset)
                iL_all.append(iL_seg[1:])
                Vterm_all.append(Vterm_seg[1:])
                alpha_seg_all.append(
                    np.full(len(t_seg) - 1, alpha_seg, dtype=np.float64)
                )

            t_offset += seg_t

        if not sim_ok:
            continue

        t_out = np.concatenate(t_all)
        i_L_clean = np.concatenate(iL_all)
        V_terminal_clean = np.concatenate(Vterm_all)
        alpha_arr = np.concatenate(alpha_seg_all)
        N = len(t_out)

        seg_edges = np.cumsum(seg_durs)
        seg_idx_for_sample = np.searchsorted(seg_edges[:-1], t_out, side="right")
        R_load_arr = np.asarray(R_loads, dtype=float)[seg_idx_for_sample]

        i_o_arr = V_terminal_clean / R_load_arr
        i_cap_clean = i_L_clean - i_o_arr
        V_C_clean = V_terminal_clean - ESR_C * i_cap_clean

        # Regime: fraction of samples with significantly negative i_L.
        # In this synchronous buck (no freewheeling diode), negative i_L means
        # the synchronous MOSFET is carrying reverse current (bidirectional
        # conduction). A converter WITH a diode would clamp to classical DCM.
        regime = float(np.mean(i_L_clean < -0.05))

        pct_negative = regime * 100.0
        alpha_str = " -> ".join(f"{a:.2f}" for a in alphas)
        R_str = " -> ".join(f"{r:.3f}" for r in R_loads)
        if pct_negative > 5.0:
            print(
                f"  Bidirectional: traj {k + 1} has {pct_negative:.1f}% reverse current "
                f"(regime={regime:.3f}).  alpha: {alpha_str}  R_load: {R_str} Ohm"
            )

        V_in_arr = np.full(N, V_in, dtype=np.float64)

        i_L_noisy = (
            i_L_clean + (rng.standard_normal(N) * sigma_iL if sigma_iL > 0 else 0.0)
        )
        V_C_noisy = (
            V_C_clean + (rng.standard_normal(N) * sigma_VC if sigma_VC > 0 else 0.0)
        )

        trajectories.append(
            {
                "t": t_out,
                "i_L": i_L_noisy,
                "V_C": V_C_noisy,
                "V_terminal": V_terminal_clean,
                "alpha": alpha_arr,
                "V_in": V_in_arr,
                "i_o": i_o_arr,
                "profile_type": "alpha_load_sweep_vrm",
                "Phi": L * i_L_clean,
                "q": C * V_C_clean,
                "regime": np.array([regime], dtype=np.float64),
            }
        )

        if (k + 1) % 10 == 0 or k == 0:
            print(
                f"  [{k + 1:3d}/{n_trajectories}]  V_in={V_in:.1f}V  "
                f"alpha: {alpha_str}  R: {R_str} Ohm  "
                f"regime={regime:.3f}  t={t_offset * 1e3:.2f}ms  N={N}"
            )

    return trajectories


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PLECS 12V->1V VRM sweep dataset."
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "buck_converter.plecs"),
        help="Path to the PLECS model file.",
    )
    parser.add_argument(
        "--train-n", type=int, default=120, help="Number of training trajectories."
    )
    parser.add_argument(
        "--test-n", type=int, default=20, help="Number of test trajectories."
    )
    parser.add_argument("--port", type=int, default=1080, help="PLECS RPC port.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # VRM converter parameters
    # Source: Qian/Yan/Cui, COMPEL 2025, Table II (SCB VRM power stage).
    # This generator models a SINGLE-PHASE CONVENTIONAL synchronous buck at
    # the same operating point -- NOT the two-phase series-capacitor topology.
    # -----------------------------------------------------------------------
    params = {
        "L": 440e-9,          # 440 nH inductor (Table II)
        "C": 200e-6,          # 200 uF output capacitor (Table II)
        # ESR_L = 3 mOhm: midpoint between R_ds=2.2 mOhm and R_co=5 mOhm from Table II.
        # A single lumped series resistance approximating MOSFET + winding losses.
        "ESR_L": 3e-3,
        "ESR_C": 0.0,         # capacitor ESR zero -- keeps V_C = V_terminal exactly
        "cycle_frequency": 1.67e6,  # 1.67 MHz switching frequency (Table II)
    }

    # Rload_range: (0.07, 0.5) Ohm -- chosen so that at nominal V_out ~ 1 V
    # (alpha = 1/12 ~ 0.083, V_in = 12 V):
    #   i_o_max = 1.0 V / 0.07 Ohm ~ 14.3 A  (rated heavy load, COMPEL 2025)
    #   i_o_min = 1.0 V / 0.50 Ohm =  2.0 A  (light load)
    # This spans the 2-14 A operating range from the paper.
    config = {
        "dt": 4.69e-6,               # cycle-averaged sample spacing (seconds)
        "Vin_range": (12.0, 12.0),   # fixed 12 V data-center input rail
        "alpha_range": (0.05, 0.20), # nominal 1/12 ~ 0.083 for 1V output
        "Rload_range": (0.07, 0.5),  # load current ~ 2-14 A at V_out ~ 1V (see above)
        # delta_alpha_min: proportionally scaled from 0.2 (bidir, range width 0.6)
        # to the VRM range width 0.15: 0.2 * (0.15/0.6) = 0.05.
        # Using 0.2 here would make both exclusion windows span the entire 0.15-wide
        # range and force every alpha change to the boundary values only.
        "delta_alpha_min": 0.05,
        "n_segs_range": (3, 4),
        # segment_duration: 1.4-2.3 ms ~ 24-39 LC periods at f_LC ~ 17 kHz
        # (RL damping time constant tau = 2L/R_L ~ 293 us; segments ~ 5-8 tau)
        "segment_duration_range": (1.4e-3, 2.3e-3),
        "sigma_iL": 0.0,
        "sigma_VC": 0.0,
        "iL_signal_idx": 0,
        "VC_signal_idx": 1,
    }

    out_train = ROOT / "data" / "plecs_vrm_train.npz"
    out_test  = ROOT / "data" / "plecs_vrm_test.npz"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 2: Feasibility self-check (printed before connecting to PLECS)
    # -----------------------------------------------------------------------
    print("=" * 66)
    print("VRM PARAMETER FEASIBILITY CHECK")
    print("  (COMPEL 2025 SCB VRM Table II -- single-phase conventional buck)")
    print("=" * 66)

    L_chk  = params["L"]
    C_chk  = params["C"]
    RL_chk = params["ESR_L"]
    fsw    = params["cycle_frequency"]
    a_lo_c, a_hi_c = config["alpha_range"]
    Rl_lo_c, Rl_hi_c = config["Rload_range"]
    seg_lo_c, seg_hi_c = config["segment_duration_range"]

    # DC operating point at nominal duty (1V output from 12V input)
    V_in_nom    = 12.0
    alpha_nom   = 1.0 / V_in_nom          # 0.0833... for exactly 1V out
    V_out_nom   = alpha_nom * V_in_nom     # 1.000 V
    R_mid       = 0.5 * (Rl_lo_c + Rl_hi_c)
    i_o_nom     = V_out_nom / R_mid
    IR_drop_nom = i_o_nom * RL_chk

    print(f"  DC operating point (nominal 1V output):")
    print(f"    alpha = 1 / V_in = 1 / {V_in_nom:.0f} = {alpha_nom:.4f}")
    print(f"    V_out = alpha * V_in = {V_out_nom:.3f} V")
    print(f"    R_load_mid = {R_mid:.4f} Ohm  =>  i_o = {i_o_nom:.2f} A")
    print(f"    IR drop = i_o * ESR_L = {i_o_nom:.2f} A * {RL_chk*1e3:.1f} mOhm"
          f" = {IR_drop_nom*1e3:.2f} mV")
    assert IR_drop_nom < 0.1 * V_out_nom, (
        f"FAIL: IR drop {IR_drop_nom:.5f} V exceeds 10% of V_out {V_out_nom:.4f} V"
    )
    print(f"    IR drop / V_out = {IR_drop_nom / V_out_nom * 100:.3f}%  => PASS")

    # Load current range
    i_o_max = V_out_nom / Rl_lo_c
    i_o_min = V_out_nom / Rl_hi_c
    print()
    print(f"  Load current range at V_out = {V_out_nom:.1f} V:")
    print(f"    R_min = {Rl_lo_c:.3f} Ohm  =>  i_o_max = {i_o_max:.1f} A  (heavy load)")
    print(f"    R_max = {Rl_hi_c:.3f} Ohm  =>  i_o_min = {i_o_min:.1f} A  (light load)")

    # LC resonance and timescale relationships
    f_LC      = 1.0 / (2.0 * np.pi * np.sqrt(L_chk * C_chk))
    T_LC      = 1.0 / f_LC
    n_sw_per_LC = fsw / f_LC
    # RL envelope damping time constant: tau = 2L / R_L
    tau_RL    = 2.0 * L_chk / RL_chk

    print()
    print(f"  LC resonant frequency : {f_LC:.1f} Hz  ({f_LC/1e3:.3f} kHz)")
    print(f"  LC period             : {T_LC*1e6:.2f} us")
    print(f"  Switching cycles / LC period : {n_sw_per_LC:.1f}  (brief target: ~98)")
    print(f"  RL damping time const : tau = 2L/R_L = {tau_RL*1e6:.0f} us")
    print(f"  Expected settling     : ~3-5 tau ~ {3*tau_RL*1e6:.0f}-{5*tau_RL*1e6:.0f} us")
    print(f"  Segment duration      : {seg_lo_c*1e3:.1f}-{seg_hi_c*1e3:.1f} ms")
    print(f"    = {seg_lo_c/T_LC:.0f}-{seg_hi_c/T_LC:.0f} LC periods")
    print(f"    = {seg_lo_c/tau_RL:.1f}-{seg_hi_c/tau_RL:.1f} RL time constants")
    print()
    print(f"  Solver timing (VRMPLECSSimulator):")
    print(f"    dt_out  = {VRMPLECSSimulator.DT_OUT*1e9:.0f} ns"
          f"  ({1/(fsw*VRMPLECSSimulator.DT_OUT):.1f} output pts / T_sw)")
    print(f"    MaxStep = {VRMPLECSSimulator.MAX_STEP*1e9:.0f} ns"
          f"  ({1/(fsw*VRMPLECSSimulator.MAX_STEP):.1f} solver steps / T_sw)")
    print(f"    Steps per 2ms segment (approx): {int(2e-3/VRMPLECSSimulator.MAX_STEP):,}")
    print("=" * 66)

    # -----------------------------------------------------------------------
    # STEP 3: Workflow instructions (stop here -- do NOT train yet)
    # -----------------------------------------------------------------------
    print("""
BEFORE RUNNING THE FULL DATASET -- READ THIS
============================================================

PLECS SCHEMATIC: WHICH VALUES NEED MANUAL CHANGES IN THE GUI?
--------------------------------------------------------------
This generator sends L, C, ESR_L, ESR_C, V_in, R_load, alpha,
and cycle_frequency as ModelVars over RPC. This works ONLY if
those names are declared as workspace variables in your .plecs
schematic and each component references the variable (e.g., the
inductor's Inductance field is "L", not a hard-coded number).

Open buck_converter.plecs in the PLECS GUI and verify / change:

  Workspace var / Component    Old value     New VRM value
  ---------------------------------------------------------
  L   (inductor inductance)    100e-6 H      440e-9 H
  C   (capacitor capacitance)  100e-6 F      200e-6 F
  ESR_L (inductor ESR)         0.1 Ohm       3e-3 Ohm
  ESR_C (capacitor ESR)        0.05 Ohm      0.0 Ohm
  V_in (voltage source)        8-16 V range  12 V fixed
  cycle_frequency (PWM block)  100e3 Hz      1.67e6 Hz

  *** PWM / sawtooth carrier frequency is especially likely to
  *** be hard-coded in a PWM Generator block rather than a
  *** workspace variable. Check the PWM block directly in the
  *** schematic -- this is the most common catch.

If the schematic already uses workspace variables for all of
these, Python controls them automatically via ModelVars and you
only need to update the default values in the workspace panel
(for sanity, not for correctness -- Python overwrites them at
runtime).

RECOMMENDED WORKFLOW
--------------------------------------------------------------
1. Verify / update the PLECS schematic as described above.

2. Smoke test with 5 trajectories (one segment ~ 2ms ~ 67,000
   solver steps at MaxStep=30ns -- expect ~3-4x slower than the
   100kHz model):
     conda run -n nilm-gpu python src/generate_plecs_vrm_dataset.py --train-n 5 --test-n 2

3. Inspect one trajectory. Expect:
     - V_out settling near alpha * 12 V  (~1V at alpha=0.083)
     - i_L in the 2-14 A range; possibly going slightly negative
       under light load (bidirectional conduction)
     - Visible but damped LC ringing after alpha / load steps
     - No numerical blowup, NaN, or unrealistic values

4. If the smoke test looks correct, run the full dataset:
     conda run -n nilm-gpu python src/generate_plecs_vrm_dataset.py

Output files:
     data/plecs_vrm_train.npz   (120 trajectories)
     data/plecs_vrm_test.npz    ( 20 trajectories)

NOTE FOR TRAINING LATER:
  BidirBuckPHNN in phnn_bidir.py uses I_typical=5.0 A and
  V_typical=12.0 V for internal normalisation. At V_out ~ 1V,
  the voltage normalisation is 12x too large. A new model file
  (e.g. phnn_vrm.py) with V_typical=1.0 and I_typical=10.0 is
  recommended before training on this dataset.
============================================================
""")

    # -----------------------------------------------------------------------
    # Generate datasets
    # -----------------------------------------------------------------------
    if args.train_n <= 10:
        print(f"[SMOKE TEST] Running {args.train_n} training + {args.test_n} test trajectories ...")
    else:
        print(f"Generating {args.train_n} training trajectories (seed=42) ...")

    sim = VRMPLECSSimulator(args.model, port=args.port)

    trajs_train = generate_plecs_vrm_dataset(sim, args.train_n, params, config, seed=42)
    save_dataset(trajs_train, str(out_train))
    total_train = sum(len(t["t"]) for t in trajs_train)
    n_bidir_tr = sum(1 for t in trajs_train if float(t["regime"][0]) > 0.05)
    print(
        f"Saved {len(trajs_train)} train trajectories, {total_train:,} timesteps"
        f"  ({n_bidir_tr} bidirectional, {len(trajs_train) - n_bidir_tr} unidirectional)"
        f" -> {out_train}\n"
    )

    print(f"Generating {args.test_n} test trajectories (seed=123) ...")
    trajs_test = generate_plecs_vrm_dataset(sim, args.test_n, params, config, seed=123)
    save_dataset(trajs_test, str(out_test))
    total_test = sum(len(t["t"]) for t in trajs_test)
    n_bidir_te = sum(1 for t in trajs_test if float(t["regime"][0]) > 0.05)
    print(
        f"Saved {len(trajs_test)} test trajectories, {total_test:,} timesteps"
        f"  ({n_bidir_te} bidirectional, {len(trajs_test) - n_bidir_te} unidirectional)"
        f" -> {out_test}"
    )
