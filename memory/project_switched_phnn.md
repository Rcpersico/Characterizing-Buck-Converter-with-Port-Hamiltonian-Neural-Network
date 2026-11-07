---
name: project-switched-phnn
description: Direction 2 switched PHNN experiment — all scripts in switchingBuckModeling/, dataset exists, training pending
metadata:
  type: project
---

Direction 2 is the switched-data PHNN experiment (Step 8 in notes.md). Goal: train `SwitchedBuckPHNN` on raw 1 MHz PLECS switched waveforms and beat the analytical averaged model on 1 ms transient RMSE.

**Why:** Averaged model structurally cannot represent switching ripple or accurate startup transients. The PHNN trained on switched data should capture these effects.

**Current state (as of 2026-05-13):**
- `switchingBuckModeling/generate_plecs_switched_dataset.py` — dataset generator, 1 MHz, no averaging, `v_sw(t)` field
- `switchingBuckModeling/train_switched_buck.py` — training script, 5000 epochs, 100-step rollout, mini-batch 256, grad clip 1.0
- `switchingBuckModeling/evaluate_switched_vs_averaged.py` — 3-way comparison (PLECS / averaged ODE / PHNN) on alpha step and load step transients
- `src/phnn.py::SwitchedBuckPHNN` — model with state-dependent R_NN (Cholesky PSD) and H_NN correction
- `data/plecs_switched_train.npz` — dataset exists (60 trajectories, 1 MHz, CCM, with `v_sw` field)
- `models/switched_buck_phnn.pt` — NOT YET TRAINED

**How to apply:** When user asks about the switched PHNN, check these files. Training needs PLECS running + GPU. Run order: (1) dataset gen if needed, (2) training, (3) evaluation.
