# Averaged CCM Buck PHNN — Working CCM Model Archive

This folder is a **locked archive** of the publication-ready Port-Hamiltonian Neural Network
that characterizes a DC-DC buck converter in Continuous Conduction Mode (CCM). Do not modify
any files here — this is the baseline against which the DCM extension is compared.

---

## Goal

Train a `SimulationBuckPHNN` on cycle-averaged PLECS switching-simulation data so that the
model generalizes across the full operating-point range (α ∈ [0.15, 0.85]) while maintaining
physical passivity guarantees from the Port-Hamiltonian structure.

---

## What Was Built

A **`SimulationBuckPHNN`** (see `src/phnn.py`) that:

- Fixes L = 100 µH and C = 100 µF (known from design)
- Learns effective inductor resistance R_L and capacitor ESR R_C
- Learns a neural Hamiltonian correction H_NN(x, u_eff) that absorbs switching
  losses, diode drop, dead-time, and other non-ideal effects
- Is trained on an **operating-point-rich sweep dataset** where each trajectory
  has 3–4 segments with consecutive α values differing by at least 0.2

---

## Key Results

| Metric | Value |
|--------|-------|
| Mean RMSE i_L (validation) | < 0.20 A |
| Mean RMSE V_C (validation) | < 0.30 V |
| Alpha-step extrapolation | Correct transient direction and settling time |
| DCM regime (R_load > 5Ω) | Graceful degradation — predicts wrong steady state but doesn't blow up |

**Alpha-step extrapolation success:** The model was trained on α ∈ [0.2, 0.8] with
R_load ∈ [2, 5] Ω (CCM only). When evaluated on an alpha step from 0.15 → 0.85 (outside
training range), it correctly predicted the transient direction and approximate settling
behaviour, demonstrating good operating-point generalization from the sweep training strategy.

**DCM degradation:** When R_load is increased past the CCM boundary (~5 Ω), the model
predicts a steady-state V_C that exceeds α·V_in — the same violation that the averaged
model makes — but the rollout remains numerically stable. This is the expected failure mode,
not a numerical instability.

---

## File Layout

```
averaged_ccm_buck_model/
├── src/
│   ├── phnn.py                        SimulationBuckPHNN model class
│   ├── simulator.py                   Ground-truth ODE integrator (used by generate_dataset)
│   ├── generate_dataset.py            save_dataset / load_dataset utilities
│   ├── generate_plecs_dataset.py      PLECSRPCSimulator + _resample (used by sweep generator)
│   ├── generate_plecs_sweep_dataset.py  Operating-point-rich sweep dataset generator
│   ├── train_simulation_buck.py       Original single-operating-point trainer
│   └── train_sweep_buck.py            ← ENTRY POINT: sweep trainer (produced best results)
├── models/
│   ├── sweep_buck_phnn.pt             Best model checkpoint (full PHNN)
│   ├── sweep_buck_physics_only.pt     Ablation baseline (physics only, no H_NN)
│   ├── sim_buck_phnn.pt               Original single-point trained model
│   └── buck_converter.plecs           PLECS schematic for RPC simulation
├── data/
│   ├── plecs_sweep_train.npz          Sweep training dataset (90 trajectories, seed=42)
│   ├── plecs_sweep_test.npz           Sweep test dataset (20 trajectories, seed=123)
│   └── plecs_physical_train.npz       Physical-coordinate training dataset (original)
├── tests/
│   ├── test_images/
│   │   ├── sweep_buck_phnn_loss.png   Training loss curve for sweep PHNN
│   │   └── sweep_buck_phnn_traj.png   Trajectory rollout plots (predicted vs PLECS)
│   ├── alpha_step_rk4.png             Alpha-step response from RK4 rollout
│   ├── alpha_step_vs_plecs.png        Alpha-step comparison: PHNN vs PLECS
│   ├── sweep_alpha_step_test.png      Sweep model alpha-step extrapolation test
│   ├── dcm_test.png                   DCM degradation test (light load)
│   └── evaluate_sweep_three_way.py    3-way comparison: PLECS vs averaged vs PHNN
└── notes.md                           Full project documentation (copy of root notes.md at archive time)
```

---

## How to Reproduce

All scripts use `ROOT = Path(__file__).resolve().parent.parent` so they resolve paths
relative to this folder automatically. Run from the project root or any directory.

Run commands with:
```
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n nilm-gpu python <script>
```

### Step 1 — Generate the sweep dataset (requires PLECS + RPC connection)
```
python averaged_ccm_buck_model/src/generate_plecs_sweep_dataset.py
```
Produces:
- `data/plecs_sweep_train.npz` (90 trajectories, seed=42)
- `data/plecs_sweep_test.npz` (20 trajectories, seed=123)

Config: α ∈ [0.2, 0.8], R_load ∈ [2, 5] Ω (CCM only), consecutive α differ by ≥ 0.2,
segment duration 8–15 ms, 3–4 segments per trajectory.

### Step 2 — Train the sweep PHNN
```
python averaged_ccm_buck_model/src/train_sweep_buck.py
```
Saves best model to `models/sweep_buck_phnn.pt`. Also generates:
- `tests/test_images/sweep_buck_phnn_loss.png`
- `tests/test_images/sweep_buck_phnn_traj.png`

### Step 3 — Train physics-only ablation baseline (optional)
```
python averaged_ccm_buck_model/src/train_sweep_buck.py --no-nn
```
Saves to `models/sweep_buck_physics_only.pt`.

### Step 4 — Evaluate
```
python averaged_ccm_buck_model/tests/evaluate_sweep_three_way.py
```

---

## Design Rationale

**Why sweep dataset?** A fixed-α load-step dataset doesn't expose the operating-point
dependent plant gain ∂V_C/∂α. The sweep forces the model to generalize across α values
within a single training trajectory.

**Why CCM-only data?** The averaged Port-Hamiltonian model assumes i_L > 0. At light loads
(R_load > ~5 Ω), the synchronous buck allows negative inductor current (reverse-current
DCM), making V_C_ss > α·V_in — impossible under the CCM averaged model. Restricting to
R_load ∈ [2, 5] Ω ensures all training data is consistent with the model structure.

**Why H_NN in the Hamiltonian?** Switching losses (diode drops, dead-time, ESR variations)
are energy-storage non-idealities. Absorbing them in the Hamiltonian H = H_physics + H_NN
is structurally correct: the PH vector field is `ẋ = (J - R) ∇H + Gu`, so H_NN corrections
appear as state-dependent energy corrections that feed directly into the ODE through the
gradient. This preserves passivity — the correction can only shift energy levels, not
violate power balance.
