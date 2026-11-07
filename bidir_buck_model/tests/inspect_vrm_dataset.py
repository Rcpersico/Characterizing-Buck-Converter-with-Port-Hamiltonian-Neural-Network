"""Quick visual inspection of the VRM smoke-test dataset.

Usage (from bidir_buck_model/):
    conda run -n nilm-gpu python tests/inspect_vrm_dataset.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generate_dataset import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "plecs_vrm_train.npz"
OUT  = ROOT / "tests" / "vrm_smoke_inspect.png"

trajs = load_dataset(DATA)
print(f"Loaded {len(trajs)} trajectories from {DATA.name}")

n_plot = min(len(trajs), 3)
fig, axes = plt.subplots(n_plot, 2, figsize=(13, 3.5 * n_plot))
if n_plot == 1:
    axes = axes[np.newaxis, :]

for row, td in enumerate(trajs[:n_plot]):
    t_ms   = td["t"] * 1e3
    i_L    = td["i_L"]
    V_C    = td["V_C"]
    alpha  = td["alpha"]
    V_in   = td["V_in"]
    V_out_expected = alpha * V_in   # cycle-averaged expected output
    regime = float(td.get("regime", np.array([0.0]))[0])

    ax_i, ax_v = axes[row, 0], axes[row, 1]

    ax_i.plot(t_ms, i_L, lw=0.8, color="tab:blue", label="i_L")
    ax_i.axhline(0, color="gray", lw=0.5, ls=":")
    ax_i.set_ylabel("i_L [A]")
    ax_i.set_title(
        f"Traj {row}  regime={regime:.3f}  "
        f"alpha {alpha.min():.2f}–{alpha.max():.2f}"
    )
    ax_i.grid(True, alpha=0.3)
    ax_i.legend(fontsize=8)

    ax_v.plot(t_ms, V_C, lw=0.8, color="tab:orange", label="V_C (measured)")
    ax_v.plot(t_ms, V_out_expected, lw=0.8, ls="--", color="gray",
              label="alpha·V_in (expected DC)")
    ax_v.set_ylabel("V_C [V]")
    ax_v.set_title(
        f"Traj {row}  V_C range [{V_C.min():.3f}, {V_C.max():.3f}] V"
    )
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(fontsize=8)

    # Print per-trajectory sanity numbers
    print(f"\nTraj {row}:")
    print(f"  alpha range : {alpha.min():.2f} – {alpha.max():.2f}")
    print(f"  V_C range   : {V_C.min():.4f} – {V_C.max():.4f} V")
    print(f"  i_L range   : {i_L.min():.3f} – {i_L.max():.3f} A")
    print(f"  regime      : {regime:.4f}  ({regime*100:.1f}% reverse current)")
    # At each segment, expected steady-state V_C ~ alpha * V_in
    # Check that measured V_C tracks expected within 20%
    err_frac = np.abs(V_C - V_out_expected) / (V_out_expected + 1e-9)
    print(f"  |V_C - expected| / expected: mean={err_frac.mean():.3f}, max={err_frac.max():.3f}")

for ax in axes[-1, :]:
    ax.set_xlabel("Time [ms]")

fig.suptitle("VRM Smoke-Test Dataset — PLECS Trajectories", fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180)
plt.close(fig)
print(f"\nSaved -> {OUT}")
