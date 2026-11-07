---
name: project-direction1-sweep
description: Direction 1 sweep PHNN experiment — operating-point-rich dataset, three-way comparison vs analytical model, scripts written awaiting PLECS+GPU run
metadata:
  type: project
---

Direction 1 tests whether SimulationBuckPHNN trained on alpha+load sweep data outperforms the textbook analytical averaged model.

**Why:** The current PLECS dataset used fixed alpha per trajectory (only load steps). The PHNN's H_NN and delta_G converged to near-zero because the training data never exposed operating-point-dependent corrections. Sweeping alpha within trajectories forces the model to learn plant gain variation across the full alpha range (0.2–0.8).

**Files created (2026-05-14):**
- `src/generate_plecs_sweep_dataset.py` — new generator; 3–4 segments per trajectory, both alpha AND R_load change at each boundary, consecutive alpha must differ by ≥ 0.2. Saves `data/plecs_sweep_train.npz` (90 trajs, seed=42) and `data/plecs_sweep_test.npz` (20 trajs, seed=123). Imports PLECSRPCSimulator and _resample from generate_plecs_dataset.
- `src/train_sweep_buck.py` — trains SimulationBuckPHNN (L=100µH, C=100µF fixed). `--no-nn` flag freezes H_NN to zero → physics-only baseline (learned R_L, R_C only). Saves `models/sweep_buck_phnn.pt` or `models/sweep_buck_physics_only.pt`. Logs H_NN value at two operating points each epoch.
- `tests/evaluate_sweep_three_way.py` — three-way RK4 rollout comparison: (1) analytical averaged model (true L/C/R), (2) physics-only PHNN, (3) full PHNN. Computes overall and transient (2 ms window after each step) RMSE, per-alpha-band breakdown. Prints summary table and saves overlay plots + bar chart.

**PLECS model parameters for sweep dataset:** L=100µH, C=100µF, ESR_L=0.1Ω, ESR_C=0.0, f_sw=100kHz.

**Run order (needs PLECS running + GPU):**
1. `python src/generate_plecs_sweep_dataset.py` — generates both train and test sets
2. `python src/train_sweep_buck.py` — full PHNN
3. `python src/train_sweep_buck.py --no-nn` — physics-only baseline
4. `python tests/evaluate_sweep_three_way.py` — three-way comparison

**Expected outcome:** one of three results: >20% transient RMSE improvement (best case), slight improvement (middle), or no improvement (null — document as legitimate negative result).

**How to apply:** [[project-switched-phnn]] is Direction 2 (parallel experiment). Check both when reporting results.
