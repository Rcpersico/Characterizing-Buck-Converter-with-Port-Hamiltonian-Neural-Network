"""Ablation study: BidirBuckPHNN trained on 60 trajectories, 3 seeds.

Trains three independent runs (seeds = [42, 43, 44]) using the same
hyperparameters (LR, rollout length, loss) as train_bidir_buck.py, varying
only the dataset size.  No early stopping — each run trains for exactly
3000 epochs.

Prerequisites:
    python ablation_data_study/build_fixed_test_set.py   (once, before any training)

Run from bidir_buck_model/:
    python ablation_data_study/train_ablation_060.py

Outputs (per seed):
    models/ablation_060_seed{SEED}.pt
    plots/ablation_060_seed{SEED}_loss.png

Combined output:
    results/ablation_060_results.json

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

N_TRAJ = 60


def main() -> None:
    per_seed_results = []
    for k, seed in enumerate(SEEDS, 1):
        print(
            f"\n=== ABLATION: {N_TRAJ} trajectories | seed={seed} "
            f"(run {k} of {len(SEEDS)}) ===\n"
        )
        result = train_single_seed(N_TRAJ, seed, k, len(SEEDS))
        per_seed_results.append(result)

    save_combined_results(N_TRAJ, per_seed_results)


if __name__ == "__main__":
    main()
