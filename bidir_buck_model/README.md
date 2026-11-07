# Bidir Buck PHNN — Exploratory Bidirectional-Current Extension

## Regime convention

This PLECS model is a **synchronous buck with no freewheeling diode**, so the
inductor current is not clamped at zero under light load.

| Regime | i_L behaviour | What happens with a real diode |
|--------|--------------|-------------------------------|
| **CCM (unidirectional)** | i_L ≥ 0 throughout | Same — standard CCM |
| **Bidirectional ("DCM")** | i_L goes negative | Diode would clamp to zero (classical DCM); here current actually reverses |

The code uses **bidirectional** to label what *would* be classical DCM in a
diode-based converter.  The key distinction: detection uses
`i_L < -0.05 A` (current is significantly negative), not `i_L == 0` (zero-clamped).

---

## Goal

Train a single `BidirBuckPHNN` that accurately characterises the averaged synchronous
buck converter across the full R_load range:
- **Heavy CCM** (R_load 2–5 Ω): the well-understood regime from the CCM model
- **Light CCM / boundary** (R_load ~5–15 Ω): marginal CCM, may show ripple
- **Bidirectional / reverse-current** (R_load > ~15 Ω): synchronous MOSFET allows
  i_L < 0, which pumps V_C above α·V_in — impossible under naive CCM assumptions

---

## Hypothesis

Three model extensions give the PHNN enough capacity to capture regime-dependent
behaviour without losing the physical Port-Hamiltonian structure:

1. **State-dependent dissipation R_theta(x)**: Forward losses (MOSFET on-resistance)
   and reverse losses (body-diode or synchronous conduction when i_L < 0) are
   fundamentally different. A state-dependent PSD dissipation matrix can represent
   this, whereas two fixed scalars cannot.

2. **Richer H_NN correction** (4→64→64→64→1 vs 4→32→32→1 in CCM): The effective
   energy storage changes with operating regime — the averaged L and C seen by the
   slow dynamics shift as the converter spends time in reverse-current conduction.
   More network capacity lets H_NN absorb these non-quadratic corrections.

3. **State-dependent input coupling delta_G(x)** (2→128→128→128→2): The plant
   gain ∂V_C/∂α varies significantly between CCM (standard) and DCM (can be
   non-monotone). A diagonal input-coupling correction lets the model adapt.

---

## Risk

The averaged Port-Hamiltonian model assumes a smooth, continuous ODE. The transition
from unidirectional to bidirectional conduction is associated with a regime shift in
loss characteristics (the synchronous MOSFET conducts differently when i_L < 0), which
the averaged model only captures approximately. If the transition nonlinearity is too
sharp, smooth neural corrections may not be sufficient and we may need:

- Separate CCM and DCM models with a switching condition
- A hybrid or piecewise-smooth model architecture
- A fundamentally different modelling approach (e.g., energy-based hybrid automaton)

**If this experiment fails, that is a valid scientific finding.** It would mean
the averaged Port-Hamiltonian structure is insufficient for bidirectional operation and motivate the
hybrid approach.

---

## File Layout

```
dcm_buck_model/
├── src/
│   ├── phnn_bidir.py                    BidirBuckPHNN model class
│   ├── generate_plecs_bidir_dataset.py  Dataset generator (R_load 2–50 Ω, bidir included)
│   ├── train_bidir_buck.py              Trainer with regime-weighted loss
│   ├── generate_dataset.py              save_dataset / load_dataset (copied from root src/)
│   ├── generate_plecs_dataset.py        PLECSRPCSimulator (copied from root src/)
│   └── simulator.py                     ODE integrator (copied from root src/)
├── models/
│   ├── bidir_buck_phnn.pt               Trained bidir model (after training)
│   └── buck_converter.plecs             PLECS schematic
├── data/
│   ├── plecs_bidir_train.npz            Bidir training dataset (120 trajs, seed=42)
│   └── plecs_bidir_test.npz             Bidir test dataset (20 trajs, seed=123)
└── tests/
    └── evaluate_bidir.py                Comprehensive 3-part evaluation script
```

No files in this folder import from `averaged_ccm_buck_model/` during training.
`evaluate_bidir.py` loads the unidirectional-trained model for comparison but handles its absence
gracefully — the evaluation still runs without it.

---

## How to Run

Run commands with:
```
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n nilm-gpu python <script>
```

### Step 1 — Generate the bidir dataset (requires PLECS + RPC connection)
```
python dcm_buck_model/src/generate_plecs_bidir_dataset.py
```
Produces:
- `data/plecs_bidir_train.npz` (120 trajectories, seed=42)
- `data/plecs_bidir_test.npz` (20 trajectories, seed=123)

Config differences from CCM sweep:
- `Rload_range: (2.0, 50.0)` — covers full CCM-to-bidirectional range
- `segment_duration_range: (15e-3, 25e-3)` — longer segments for reverse-current settling
- Each trajectory carries a `regime` field (fraction of samples with i_L < −0.05 A,
  i.e. significantly in reverse — not zero-clamped as in classical diode DCM)

### Step 2 — Train the bidir PHNN
```
python dcm_buck_model/src/train_bidir_buck.py
```
Key hyperparameters:
- LR = 5e-4, EPOCHS = 12000, early-stopping patience = 1500
- MULTISTEP_N = 60 (3 ms rollout window — captures reverse-current ringing)
- Bidirectional windows (any i_L < −0.05 A) get 2× loss weight
- Per-epoch logging of |H_NN| and |delta_G| split by unidirectional vs bidirectional windows

Saves best model to `models/bidir_buck_phnn.pt`.

### Step 3 — Evaluate
```
python dcm_buck_model/tests/evaluate_bidir.py
```
Produces three outputs in `tests/`:
- `bidir_eval_trajectories.png` — 6 representative plots covering unidirectional, boundary, and bidirectional
- `bidir_eval_steady_state.png` — steady-state V_C error vs R_load scatter

Console output includes:
- Per-regime RMSE table (unidirectional vs bidirectional)
- Acceptance criterion check (bidirectional RMSE ≤ 3× unidirectional RMSE)
- Side-by-side comparison with the unidirectional-trained model on bidirectional trajectories

---

## Architecture: BidirBuckPHNN

```
ẋ = (J - R_theta(x)) ∇H_total(x, u) + (G_physics + delta_G(x)) u

H_total = H_physics(x) + H_NN(x, u)   # H_NN: 4→64→64→64→1
R_theta(x) = L_R(x) @ L_R(x)^T + ε·I  # L_R from MLP: 2→64→64→3
delta_G(x) = diag(dg(x))               # dg: 2→128→128→128→2
G_physics = diag(1, -1)                 # fixed from physics
```

Initialisation:
- H_NN output layer: zeros (starts as pure physics)
- R_chol_net bias: [√0.10, 0.0, √0.05] so R_theta(0) ≈ diag(0.10, 0.05) at origin
- delta_G output layer: zeros (starts with G = G_physics)

---

## What Success Looks Like

1. Bidirectional RMSE ≤ 3× unidirectional RMSE on the held-out test set
2. The H_NN and delta_G norms grow meaningfully larger in bidirectional windows
   than unidirectional windows (proving the neural components are learning regime-specific
   corrections, not just fitting noise)
3. The bidirectional-trained model is measurably better than the CCM-trained model
   on light-load trajectories where i_L goes negative

If all three criteria are met, the averaged PHNN with state-dependent dissipation
is a viable unified model for buck converter characterisation across both regimes.
