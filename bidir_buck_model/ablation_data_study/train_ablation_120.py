"""Ablation study: BidirBuckPHNN trained on the full training pool, 3 seeds.

Trains three independent runs (seeds = [42, 43, 44]) using the same
hyperparameters (LR, rollout length, loss) as train_bidir_buck.py.
No early stopping — each run trains for exactly 3000 epochs.

Dataset note
------------
The full training dataset (plecs_bidir_train2.npz) has 120 trajectories.
After build_fixed_test_set.py reserves 20 for the permanent held-out test
set, 100 trajectories remain as the training pool.  This script uses all
100 (the maximum available), so N_TRAJ = 100.  The script is named "120"
to reflect the total dataset size (120 trajectories generated in total).

Prerequisites:
    python ablation_data_study/build_fixed_test_set.py   (once, before any training)

Run from bidir_buck_model/:
    python ablation_data_study/train_ablation_120.py

Outputs (per seed):
    models/ablation_100_seed{SEED}.pt
    plots/ablation_100_seed{SEED}_loss.png

Combined output:
    results/ablation_100_results.json

Regime terminology:
  - Unidirectional: i_L >= 0 — standard forward conduction.
  - Bidirectional:  i_L < -0.05 A — reverse current through the synchronous
                    MOSFET (what a freewheeling-diode converter clamps to DCM).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # bidir_buck_model/
ABLATION_DIR = Path(__file__).resolve().parent  # ablation_data_study/

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ABLATION_DIR))

from utils_ablation import SEEDS, save_combined_results, train_single_seed  # noqa: E402

# All available training trajectories after reserving the fixed 20-trajectory test set.
# The source dataset has 120 trajectories; 120 - 20 test = 100 available for training.
N_TRAJ = 100


def main() -> None:
    per_seed_results = []
    for k, seed in enumerate(SEEDS, 1):
        print(
            f"\n=== ABLATION: {N_TRAJ} trajectories (full pool) | seed={seed} "
            f"(run {k} of {len(SEEDS)}) ===\n"
        )
        result = train_single_seed(N_TRAJ, seed, k, len(SEEDS))
        per_seed_results.append(result)

    save_combined_results(N_TRAJ, per_seed_results)


if __name__ == "__main__":
    main()
