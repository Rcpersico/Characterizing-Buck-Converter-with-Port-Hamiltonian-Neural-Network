from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import load_dataset


DATA_PATH = ROOT / "data" / "plecs_physical_train.npz"
TRUE_L = 100e-6
TRUE_C = 100e-6
TRUE_R = 0.1


def implied_l_stats(trajectories: list[dict], use_clean_coordinates: bool) -> dict:
    implied_l = []
    inverse_ratio = []
    n_mask = 0
    n_total = 0

    for td in trajectories:
        t = np.asarray(td["t"], dtype=float)
        if use_clean_coordinates and "Phi" in td and "q" in td:
            i_L = np.asarray(td["Phi"], dtype=float) / TRUE_L
            V_C = np.asarray(td["q"], dtype=float) / TRUE_C
        else:
            i_L = np.asarray(td["i_L"], dtype=float)
            V_C = np.asarray(td["V_C"], dtype=float)

        alpha = np.asarray(td["alpha"], dtype=float)
        V_in = np.asarray(td["V_in"], dtype=float)
        V_L = alpha * V_in - TRUE_R * i_L - V_C

        diL_dt = np.diff(i_L) / np.diff(t)
        V_L_mid = 0.5 * (V_L[:-1] + V_L[1:])
        mask = (
            (np.abs(V_L_mid) > 0.1)
            & (np.abs(diL_dt) > 1e-9)
            & np.isfinite(V_L_mid)
            & np.isfinite(diL_dt)
        )

        n_mask += int(mask.sum())
        n_total += mask.size
        if np.any(mask):
            # Physical relation: V_L = L * di_L/dt.
            implied_l.append(V_L_mid[mask] / diL_dt[mask])
            inverse_ratio.append(diL_dt[mask] / V_L_mid[mask])

    implied_l_arr = np.concatenate(implied_l)
    inverse_ratio_arr = np.concatenate(inverse_ratio)
    return {
        "n_mask": n_mask,
        "n_total": n_total,
        "median_uH": float(np.median(implied_l_arr) * 1e6),
        "mean_uH": float(np.mean(implied_l_arr) * 1e6),
        "std_uH": float(np.std(implied_l_arr) * 1e6),
        "p10_uH": float(np.percentile(implied_l_arr, 10) * 1e6),
        "p90_uH": float(np.percentile(implied_l_arr, 90) * 1e6),
        "inverse_median_per_H": float(np.median(inverse_ratio_arr)),
    }


if __name__ == "__main__":
    trajectories = load_dataset(DATA_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {DATA_PATH.name}")

    for label, use_clean in (
        ("observed i_L/V_C", False),
        ("clean Phi/q coordinates", True),
    ):
        stats = implied_l_stats(trajectories, use_clean_coordinates=use_clean)
        print(f"\n=== {label} ===")
        print(f"Mask points: {stats['n_mask']:,}/{stats['n_total']:,}")
        print(f"implied L = {stats['median_uH']:.1f} uH")
        print(
            "implied L mean/std = "
            f"{stats['mean_uH']:.1f}/{stats['std_uH']:.1f} uH"
        )
        print(
            "implied L p10/p90 = "
            f"{stats['p10_uH']:.1f}/{stats['p90_uH']:.1f} uH"
        )
        print(
            "literal diL_dt/V_L median = "
            f"{stats['inverse_median_per_H']:.3e} 1/H"
        )
