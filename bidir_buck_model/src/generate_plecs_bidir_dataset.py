"""
PLECS dataset generator for the bidirectional-current extension experiment.

Regime convention: This PLECS model is a synchronous buck with no freewheeling
diode, so inductor current can go negative under light load.
  - Unidirectional: i_L ≥ 0 — standard forward conduction.
  - Bidirectional:  i_L < 0 — reverse current flows through the synchronous
                    MOSFET. In a converter WITH a diode the same condition
                    produces classical DCM (zero clamping); here the current
                    actually goes negative.

Extends the unidirectional sweep generator to cover the full R_load range
[2, 50] Ω so that the dataset spans heavy unidirectional, light unidirectional,
the unidirectional/bidirectional boundary, and deep reverse-current operation.
Bidirectional trajectories are intentionally kept (they are the point of this
experiment).

Key differences from generate_plecs_sweep_dataset.py:
  - Rload_range: (2.0, 50.0) instead of (2.0, 5.0)
  - segment_duration_range: (15e-3, 25e-3) — reverse-current LC ringing settles
    more slowly than unidirectional steady-state
  - 120 training trajectories (more data for richer dynamics)
  - Each saved trajectory carries a 'regime' field: fraction of samples with
    i_L < -0.05 A (0.0 = pure unidirectional, 1.0 = fully reversed)
  - Reverse-current trajectories are printed but NOT skipped

Usage:
    python dcm_buck_model/src/generate_plecs_bidir_dataset.py
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
# Alpha sampling helper 
# ---------------------------------------------------------------------------


def _round_alpha(alpha: float, decimals: int = 2) -> float:
    return round(alpha, decimals)


def _sample_diverse_alpha(
    rng: np.random.Generator,
    prev_alpha: float,
    a_lo: float,
    a_hi: float,
    delta_min: float = 0.2,
    decimals: int = 2,
) -> float:
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


def generate_plecs_bidir_dataset(
    simulator: PLECSRPCSimulator,
    n_trajectories: int,
    params: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    """Generate bidirectional-current sweep trajectories.

    R_load spans [2, 50] Ω so the dataset covers unidirectional (i_L ≥ 0),
    the unidirectional/bidirectional boundary, and deep reverse-current
    operation (i_L < 0). Each returned trajectory dict contains a 'regime'
    field (float in [0, 1]) equal to the fraction of samples where
    i_L < -0.05 A.

    Args:
        simulator: Configured PLECSRPCSimulator instance.
        n_trajectories: Number of trajectories to attempt.
        params: PLECS model parameters (L, C, ESR_L, ESR_C, cycle_frequency).
        config: Dataset configuration dict (see driver for keys).
        seed: RNG seed for reproducibility.

    Returns:
        List of trajectory dicts with the standard schema plus 'regime' field.
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
    delta_alpha_min = float(config.get("delta_alpha_min", 0.2))
    n_segs_lo = int(config.get("n_segs_range", (3, 4))[0])
    n_segs_hi = int(config.get("n_segs_range", (3, 4))[1])
    seg_dur_lo, seg_dur_hi = config.get("segment_duration_range", (15e-3, 25e-3))
    sigma_iL = float(config.get("sigma_iL", 0.0))
    sigma_VC = float(config.get("sigma_VC", 0.0))
    iL_idx = int(config.get("iL_signal_idx", 0))
    VC_idx = int(config.get("VC_signal_idx", 1))

    trajectories: list[dict] = []

    for k in range(n_trajectories):
        V_in = float(rng.uniform(Vin_lo, Vin_hi))
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
        initial_state: dict | None = None
        sim_ok = True

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

        # Regime label: fraction of samples with significantly negative i_L.
        # Positive regime → bidirectional (reverse-current) conduction.
        # In a converter with a freewheeling diode this would be classical DCM
        # (current clamped to zero); here the current actually goes negative.
        regime = float(np.mean(i_L_clean < -0.05))

        pct_negative = regime * 100.0
        alpha_str = " → ".join(f"{a:.2f}" for a in alphas)
        R_str = " → ".join(f"{r:.1f}" for r in R_loads)
        if pct_negative > 5.0:
            print(
                f"  Bidirectional: traj {k + 1} has {pct_negative:.1f}% reverse current "
                f"(regime={regime:.3f}). alpha: {alpha_str}, R_load: {R_str}"
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
                "profile_type": "alpha_load_sweep_bidir",
                "Phi": L * i_L_clean,
                "q": C * V_C_clean,
                "regime": np.array([regime], dtype=np.float64),
            }
        )

        if (k + 1) % 10 == 0 or k == 0:
            print(
                f"  [{k + 1:3d}/{n_trajectories}]  V_in={V_in:.1f}V  "
                f"alpha: {alpha_str}  R: {R_str}  "
                f"regime={regime:.3f}  t={t_offset * 1e3:.1f}ms  N={N}"
            )

    return trajectories


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PLECS bidirectional-current sweep dataset."
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

    sim = PLECSRPCSimulator(args.model, port=args.port)

    # L and C match L_TRUE / C_TRUE in train_bidir_buck.py so model priors
    # are consistent with the physical circuit being simulated.
    params = {
        "L": 100e-6,
        "C": 100e-6,
        "ESR_L": 0.1,
        "ESR_C": 0.05,
        "cycle_frequency": 100e3,
    }

    config = {
        "dt": 50e-6,
        "Vin_range": (8.0, 16.0),
        "alpha_range": (0.2, 0.8),
        # Full R_load range: covers unidirectional (2–5 Ω), boundary (~5–15 Ω),
        # bidirectional/reverse-current (>15 Ω — where i_L goes negative)
        "Rload_range": (0.1, 50.0),
        "delta_alpha_min": 0.2,
        "n_segs_range": (3, 4),
        # Longer segments: reverse-current LC ringing settles more slowly
        "segment_duration_range": (15e-3, 25e-3),
        "sigma_iL": 0.0,
        "sigma_VC": 0.0,
        "iL_signal_idx": 0,
        "VC_signal_idx": 1,
    }

    out_train = ROOT / "data" / "plecs_bidir_train2.npz"
    out_test = ROOT / "data" / "plecs_bidir_test2.npz"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.train_n} training trajectories (seed=42) ...")
    trajs_train = generate_plecs_bidir_dataset(sim, args.train_n, params, config, seed=42)
    save_dataset(trajs_train, str(out_train))
    total_train = sum(len(t["t"]) for t in trajs_train)
    n_bidir = sum(1 for t in trajs_train if float(t["regime"][0]) > 0.05)
    print(
        f"Saved {len(trajs_train)} train trajectories, {total_train:,} timesteps"
        f"  ({n_bidir} bidirectional, {len(trajs_train) - n_bidir} unidirectional)"
        f" → {out_train}\n"
    )

    print(f"Generating {args.test_n} test trajectories (seed=123) ...")
    trajs_test = generate_plecs_bidir_dataset(sim, args.test_n, params, config, seed=123)
    save_dataset(trajs_test, str(out_test))
    total_test = sum(len(t["t"]) for t in trajs_test)
    n_bidir_test = sum(1 for t in trajs_test if float(t["regime"][0]) > 0.05)
    print(
        f"Saved {len(trajs_test)} test trajectories, {total_test:,} timesteps"
        f"  ({n_bidir_test} bidirectional, {len(trajs_test) - n_bidir_test} unidirectional)"
        f" → {out_test}"
    )

