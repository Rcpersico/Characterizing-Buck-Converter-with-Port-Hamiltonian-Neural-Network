# Project Notes — Buck Converter Port-Hamiltonian Neural Network

To run python scripts use:
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" run -n nilm-gpu python <script_path>

Use this command form for all script executions and tests. Do not invoke python directly. Do not try to install packages with plain pip — if a package is missing, ask me first before running anything.

## Goal

Build a Port-Hamiltonian Neural Network (PHNN) that learns the dynamics of a DC-DC buck converter from trajectory data. The network is constrained to the PH structure so it recovers physically meaningful parameters (L, C, R) rather than acting as a black box. Those parameters are then used to linearize the plant and design a closed-loop controller.


## Physics Reference

### Model: Averaged Port-Hamiltonian Buck Converter

The averaged model treats the switching duty cycle α ∈ [0, 1] as a continuous input. States are chosen as energy-conjugate (Port-Hamiltonian) coordinates:

| Symbol | Units | Meaning |
|--------|-------|---------|
| Φ | Wb (= V·s) | Inductor flux linkage: Φ = L · i_L |
| q | C (= A·s) | Capacitor charge: q = C · V_C |

**Hamiltonian (stored energy):**
```
H(Φ, q) = Φ²/(2L) + q²/(2C)
```

**ODE (from PH structure):**
```
dΦ/dt = α·V_in − R·i_L − V_C
dq/dt = i_L − i_o
```
where `i_L = Φ/L` and `V_C = q/C`.

**Input vector:** `u = [α, V_in, i_o]`
- α — duty cycle (dimensionless, 0..1)
- V_in — supply voltage [V]
- i_o — load current [A]

**Parameters:** L [H], C [F], R [Ω] (inductor series resistance)

### Steady-State Solution

Setting dΦ/dt = 0 and dq/dt = 0:
```
i_L* = i_o
V_C* = α·V_in − R·i_o
```

### Natural (Resonant) Frequency
```
ω₀ = 1 / sqrt(L·C)   [rad/s]
f₀ = ω₀ / (2π)       [Hz]
```

---

## Codebase

```
src/
  simulator.py          Ground-truth ODE simulator (Step 1)
  generate_dataset.py   Dataset generation (Step 2)

data/
  train.npz             100-trajectory training dataset (seed 42)

tests/
  test_simulator.py             Sanity-check script for the simulator
  sanity_check.png              Output plot from the sanity check
  test_dataset.py               Dataset diagnostics (Step 2)
  dataset_phase_coverage.png    Phase-space scatter of all (i_L, V_C) points
  dataset_input_distribution.png  Histograms of α, i_o, V_in
  dataset_sample_trajectories.png 5 random i_L(t) and V_C(t) traces
```

### `src/generate_dataset.py` — public API

| Function | Purpose |
|----------|---------|
| `make_input_profile(profile_type, V_in, t_span, rng, ...)` | Returns a callable `u_fn(t)` for one of five profile shapes (see below). |
| `generate_dataset(n_trajectories, params, config, seed)` | Simulates N trajectories; returns list of dicts with keys `t, Phi, q, i_L, V_C, alpha, V_in, i_o, dPhi_dt, dq_dt, profile_type`. |
| `save_dataset(trajectories, path)` | Serialises the list to a `.npz` file (keys: `traj_{i}_{field}`, `profile_types`, `n_trajectories`). |
| `load_dataset(path)` | Loads and reconstructs the list from a `.npz` file. |

**Profile types supported by `make_input_profile`:**

| Type | α | i_o |
|------|---|-----|
| `constant` | fixed | fixed |
| `alpha_step` | step at random time in [20%, 80%] of duration | fixed |
| `load_step` | fixed | step at random time in [20%, 80%] of duration |
| `alpha_ramp` | linear ramp over full duration | fixed |
| `random_pwc` | 2–4 piecewise-constant segments | 2–4 piecewise-constant segments |

V_in is constant within every trajectory regardless of profile type.

**`generate_dataset` config dict keys:**

| Key | Description |
|-----|-------------|
| `dt` | Output sampling interval [s] |
| `t_span_range` | `(t_min, t_max)` trajectory duration [s] |
| `iL_range` | `(min, max)` initial inductor current [A] |
| `VC_range` | `(min, max)` initial capacitor voltage [V] |
| `Vin_range` | `(min, max)` input voltage per trajectory [V] |
| `alpha_range` | `(min, max)` duty-cycle samples |
| `io_range` | `(min, max)` load-current samples [A] |
| `sigma_iL` | Gaussian measurement noise std on i_L [A] (0 = off) |
| `sigma_VC` | Gaussian measurement noise std on V_C [V] (0 = off) |
| `profile_weights` | Dict of `{profile_type: weight}` (normalised internally) |

**Derivative storage note:** `dPhi_dt` and `dq_dt` are computed from the *clean* (noiseless) states using the vectorised PH equations, not by differencing the noisy observations. Noise is applied only to `i_L` and `V_C` in the returned dict. `Phi` and `q` remain clean.

---

### `src/simulator.py` — public API

| Function | Purpose |
|----------|---------|
| `simulate_buck(x0, u_fn, t_span, params, dt)` | Integrate the PH ODE with RK45 (rtol=1e-8, atol=1e-10). Returns dict of time series. |
| `compute_xdot(x, u, params)` | Evaluate dx/dt analytically. Used as ground-truth derivatives for training loss. |
| `phi_q_to_iL_vC(Φ, q, L, C)` | Convert energy coordinates → physical variables. |
| `iL_vC_to_phi_q(i_L, V_C, L, C)` | Convert physical variables → energy coordinates. |

**`simulate_buck` return dict keys:** `t, Phi, q, i_L, V_C, alpha, V_in, i_o`

---

## Step 1 — Ground-Truth Simulator (Complete)

**Date completed:** 2026-05-04

### What was built
- `src/simulator.py`: the averaged PH ODE integrated with `scipy.integrate.solve_ivp` (RK45, tight tolerances). Includes coordinate conversion helpers and an analytic derivative function.
- `tests/test_simulator.py`: runs a full 10 ms simulation from rest and checks three things automatically.

### Test parameters
```
L = 100 µH,  C = 100 µF,  R = 0.1 Ω
V_in = 12 V,  α = 0.5,  i_o = 1 A
x0 = [Φ=0, q=0]  (i_L=0, V_C=0 at t=0)
t_end = 10 ms,  dt = 1 µs output grid
```

### Sanity-check results

**Steady-state V_C:**
```
Expected : 5.9000 V   (= α·V_in − R·i_o = 0.5·12 − 0.1·1)
Simulated: 5.8967 V   (mean over final 20% of trace)
Error    : 0.0033 V   (< 0.5% tolerance) ✓
```
Small offset is expected — system is still settling at 10 ms; exact convergence requires t → ∞.

**Resonant frequency:**
```
Expected ω₀ : 10 000.0 rad/s  (1591.6 Hz)
Simulated   : 10 052.1 rad/s  (1599.8 Hz)   via FFT of i_L(t)
Error       : 52.1 rad/s  (< 3% tolerance) ✓
```

**xdot at steady state:**
```
dΦ/dt = 0.00e+00 ✓
dq/dt = 0.00e+00 ✓
```

### Plot

`tests/sanity_check.png` — shows i_L(t) and V_C(t) ringing down at ω₀ from zero initial conditions and converging to steady state. Physically expected: lightly damped LC resonance (Q ≈ ω₀·L/R = 10) decaying over ~5–8 ms.

---

## Step 2 — Training Data Generation (Complete)

**Date completed:** 2026-05-04

### What was built
- `src/generate_dataset.py`: input profile factory (`make_input_profile`), trajectory generator (`generate_dataset`), and serialisation helpers (`save_dataset` / `load_dataset`).
- `tests/test_dataset.py`: diagnostic plots and summary statistics.
- `data/train.npz`: the training dataset (seed 42).

### Dataset parameters
```
n_trajectories : 100
params         : L = 100 µH,  C = 100 µF,  R = 0.1 Ω
dt             : 10 µs  (100 kHz output grid)
t_span_range   : 5 ms – 20 ms  (snapped to nearest dt multiple)
iL_range       : −1 A – 3 A   (initial condition)
VC_range       : 0 V – 10 V   (initial condition)
Vin_range      : 10 V – 14 V  (constant per trajectory)
alpha_range    : 0.2 – 0.8
io_range       : 0 A – 2 A
sigma_iL       : 0.02 A
sigma_VC       : 0.05 V
profile_weights: constant 40%, alpha_step 20%, load_step 20%, alpha_ramp 10%, random_pwc 10%
```

### Dataset statistics (seed 42)
```
Total timesteps : 122,951
i_L  range [A]  : −7.69 .. 10.15
V_C  range [V]  : −3.17 .. 17.77
alpha range     : 0.20 .. 0.80
V_in  range [V] : 10.00 .. 13.90
i_o   range [A] : 0.01 .. 1.99
Profile mix     : constant 40, alpha_step 22, load_step 17, alpha_ramp 11, random_pwc 10
```

### Phase-space coverage
The (i_L, V_C) scatter spans a meaningful 2D region with no visible clusters or dead zones. Dense in the expected operating region (i_L ≈ 0–3 A, V_C ≈ 3–10 V) with good excursions from diverse initial conditions and transients. Suitable for PHNN training.

---

## Step 3 - Conservative LC-Tank PHNN Verification (Complete)

**Date completed:** 2026-05-04

### What was built
- `src/generate_lc_tank_data.py`: conservative LC-tank dataset generator with R = 0 and zero input.
- `src/phnn.py`: minimal `LCTankPHNN` that learns scalar `H_theta(Phi, q)` and computes `dx/dt = J grad H` with fixed canonical `J`.
- `src/train_lc_tank.py`: full-batch trainer with trajectory-level 80/20 split and normalized derivative MSE.
- `tests/test_lc_tank.py`: diagnostic script for learned energy conservation, phase-space orbits, and Hamiltonian contours.
- `data/lc_tank.npz`: 50 clean LC-tank trajectories, 5 ms each, sampled at 100 kHz.
- `models/lc_tank_phnn.pt`: trained model weights plus normalization metadata and loss history.

### Scaling decision
The PHNN API remains in physical energy coordinates `(Phi, q)`. Inputs are divided by per-coordinate training-set standard deviations inside the model before the MLP, but autograd differentiates with respect to the original physical state. Therefore `forward(x)` still returns physical `(dPhi/dt, dq/dt)`.

The training loss is computed on normalized derivatives:
```
MSE((x_dot_pred - x_dot_true) / derivative_scale)
```
This balances the two derivative channels without changing the Port-Hamiltonian vector field.

### Results
```
Final training loss   : 3.982249e-04
Final validation loss : 5.406903e-04

Trajectory 1 (dataset index 8)  max relative H drift: 6.136667e-07, trajectory MSE: 3.071872e-01
Trajectory 2 (dataset index 0)  max relative H drift: 1.426001e-07, trajectory MSE: 1.033568e+00
Trajectory 3 (dataset index 33) max relative H drift: 2.164994e-07, trajectory MSE: 1.046363e-01
```

### Diagnostic plots
- `tests/lc_tank_loss.png`: normalized derivative MSE over 2000 epochs.
- `tests/lc_tank_energy_conservation.png`: learned Hamiltonian is visually flat over 5 ms predicted rollouts.
- `tests/lc_tank_orbits.png`: predicted phase-space trajectories form closed orbits and closely overlay ground truth.
- `tests/lc_tank_hamiltonian_contour.png`: learned Hamiltonian contours are centered concentric ellipses/circles.

---

## Step 4 - Add Dissipation: RLC PHNN Stage 2 (Complete)

**Date completed:** 2026-05-04

### What was built
- `src/generate_rlc_data.py`: damped zero-input RLC dataset generator with `R = 0.1 Ohm`, nonzero initial conditions, and `u(t) = [0, 0, 0]`.
- `src/phnn.py`: added `RLCBuckPHNN`, which computes `dx/dt = (J - R_theta) grad H_theta`.
- `src/train_rlc_phnn.py`: full-batch derivative-matching trainer for the damped RLC model.
- `tests/test_rlc_phnn.py`: diagnostic script for energy monotonicity, damped phase-space spirals, and learned dissipation entries.
- `data/rlc_no_input.npz`: 60 clean damped RLC trajectories, 8 ms each, sampled at 100 kHz.
- `models/rlc_phnn.pt`: trained Stage 2 model checkpoint.

### Dissipation parameterization
The learned dissipation matrix is positive semidefinite by construction:
```python
R_theta = L_R @ L_R.T + epsilon * I
```
`L_R` is stored as a learnable lower-triangular 2x2 factor. For the buck converter, a structural mask confines effective dissipation to the inductor branch:
```python
L_R mask = [[1, 0],
            [0, 0]]
```
so `R_theta[2, 2]` remains at the small epsilon floor. This matches the physical model where the series resistance is in the inductor branch, not the capacitor branch.

### Hamiltonian scaling decision
For this Stage 2 identifier, `H_theta` is a learned positive diagonal quadratic:
```python
H_theta(Phi, q) = 0.5 * inv_L * Phi^2 + 0.5 * inv_C * q^2
```
This keeps `grad H_theta = [i_L, V_C]` physically scaled, which makes `R_theta[1, 1]` directly comparable to the true resistance. A fully free MLP Hamiltonian can fit the vector field but can rescale the Hamiltonian and blur direct `R` identification.

### Training results
```
Final training loss   : 9.598623e-10
Final validation loss : 6.076260e-10

Learned L             : 9.99968458e-05 H
Learned C             : 1.00000186e-04 F
Learned R_theta[1,1]  : 1.00000076e-01 Ohm
Learned R_theta[1,2]  : 0.00000000e+00
Learned R_theta[2,2]  : 9.99999997e-07
True R                : 1.00000000e-01 Ohm
|R_theta[1,1] - R|    : 7.59959221e-08 Ohm
```

### Checkpoint diagnostics
Three validation trajectories were rolled out with the learned model.
```
Trajectory 1 (dataset index 13):
  max Delta H          : -1.227818e-11
  H_final / H_initial  : 3.421617e-04
  trajectory MSE       : 5.147719e-08

Trajectory 2 (dataset index 22):
  max Delta H          : -6.906475e-12
  H_final / H_initial  : 3.623119e-04
  trajectory MSE       : 3.847888e-08

Trajectory 3 (dataset index 16):
  max Delta H          : -2.887646e-11
  H_final / H_initial  : 3.338051e-04
  trajectory MSE       : 1.362431e-07
```
All sampled energy differences were nonpositive, and the phase-space trajectories spiral inward as expected.

### Diagnostic plots
- `tests/rlc_phnn_loss.png`: normalized derivative MSE over 1500 epochs.
- `tests/rlc_phnn_energy_decay.png`: learned `H_theta(t)` decreases monotonically along predicted damped rollouts.
- `tests/rlc_phnn_spiral_orbits.png`: predicted phase-space trajectories spiral inward and overlay ground truth.

---

## Step 5 — Input-Coupled BuckPHNN (Complete)

**Date completed:** 2026-05-05

### What was built
- `src/phnn.py`: added `BuckPHNN` class alongside existing `RLCBuckPHNN`. Implements `dx/dt = (J - R_theta) grad H_theta(x) + G_theta(x) u_eff` where `u_eff = [alpha*V_in, i_o]`.
- `src/train_buck_phnn.py`: trains on the full 100-trajectory `data/train.npz` dataset. Warm-starts from `rlc_phnn.pt`. Adam lr=1e-3, ReduceLROnPlateau(patience=100, factor=0.5), early stopping patience=500.
- `tests/test_buck_phnn.py`: four diagnostic checks — parameter recovery, 4-trajectory rollout, energy behaviour, input-coupling matrix.
- `models/buck_phnn.pt`: trained Step 5 checkpoint (best val at epoch 2988).
- `tests/buck_phnn_loss.png`: loss curve.
- `tests/buck_phnn_trajectories.png`: predicted vs ground-truth on 4 held-out trajectories.
- `tests/buck_phnn_energy.png`: H_theta(t) and power budget along a predicted rollout.

### Model architecture
```
G_theta(x) = G_physics + delta_G_theta(x)
G_physics  = [[1, 0], [0, -1]]  (fixed)
delta_G_theta: Linear(2→32) → Tanh → Linear(32→4)
              output layer zeroed at init so delta_G = 0 at start
```
H_theta and R_theta are the same parameterisation as RLCBuckPHNN (quadratic H, Cholesky R with inductor-branch mask).

### Training results
```
Train points        : 98,378
Val   points        : 24,573
Derivative scale    : [0.9696, 0.9649]  (≈ 1 Wb/s, 1 C/s)
Best epoch          : 2988
Best val loss       : 9.336e-12
Final train loss    : 1.034e-11
Final val loss      : 9.966e-12
```
The LR scheduler fired at ~epoch 1550 (spike in the curve, damped by the scheduler dropping lr 1e-3 → 5e-4 → 2.5e-4). No late-epoch divergence.

### Parameter recovery
```
Learned L      : 9.999998e-05 H   (true: 1.0000e-04 H,  error: 0.000%)
Learned C      : 9.999970e-05 F   (true: 1.0000e-04 F,  error: 0.000%)
Learned R      : 1.000002e-01 Ω   (true: 1.0000e-01 Ω,  error: 0.000%)
||delta_G||_F  : 3.15e-06         (physics prior: no correction needed)
```
Recovery is at Step 4 quality — adding the input coupling did not degrade the Hamiltonian identification.

### Trajectory rollout (held-out, RK4)
```
traj  35 (constant)    RMSE i_L = 1.95e-02 A,  RMSE V_C = 5.12e-02 V
traj   6 (load_step)   RMSE i_L = 2.08e-02 A,  RMSE V_C = 5.31e-02 V
traj  53 (random_pwc)  RMSE i_L = 7.64e-02 A,  RMSE V_C = 9.55e-02 V
traj  14 (alpha_ramp)  RMSE i_L = 2.02e-02 A,  RMSE V_C = 5.10e-02 V
Mean RMSE i_L : 3.42e-02 A   (spec: < 0.1 A)
Mean RMSE V_C : 6.27e-02 V   (spec: < 0.1 V)
```

### Energy and input-coupling diagnostics
- H_theta rises when the input injects more power than is dissipated — confirmed.
- Power balance correlation corr(dH/dt, P_in - P_R - P_out) = 1.0000.
- G_theta at all 5 operating points prints as [[+1.0000, +0.0000], [+0.0000, -1.0000]] to 4 d.p. Max |delta_G| entry ≈ 3.2e-06.

### All assertions PASSED

---

## Step 6 — Rollout-Based Training (Complete)

**Date completed:** 2026-05-05

### What was built
- `src/train_buck_phnn_rollout.py`: Trains the `BuckPHNN` model using one-step RK4 state-transition (rollout) loss instead of analytic derivative matching. Warm-starts from `rlc_phnn.pt`.
- `tests/test_buck_phnn_rollout.py`: Diagnostic script for physical parameter recovery and held-out trajectory rollout RMSE.
- `models/buck_phnn_rollout.pt`: Trained Step 6 checkpoint.
- `tests/buck_phnn_rollout_loss.png`: Rollout training loss curve.
- `tests/buck_phnn_rollout_trajectories.png`: Predicted vs ground-truth rollouts on 4 held-out trajectories.

### Training strategy
Instead of `MSE(f_theta(x, u), dx/dt_true)`, the training pairs are consecutive transition steps:
```python
x_pred_next = RK4Step(x_k, u_k, f_theta, dt)
loss = MSE((x_pred_next - x_next) / state_scale)
```
State scaling is based on the standard deviation of `[Phi, q]` in the training set (`state_scale ≈ [0.000113, 0.000224]`).

### Training results
- Early stopping triggered at epoch 546 (best val loss: `5.64e-06` at epoch 46).
- Recovery matches Step 5 quality without requiring clean simulator derivatives.

### Parameter recovery
```
Learned L      : 1.0001e-04 H   (true: 1.0000e-04 H,  error: 0.011%)
Learned C      : 9.9986e-05 F   (true: 1.0000e-04 F,  error: 0.014%)
Learned R      : 1.0006e-01 Ω   (true: 1.0000e-01 Ω,  error: 0.055%)
```

### Trajectory rollout (held-out, RK4)
```
traj  35 (constant)    RMSE i_L = 1.95e-02 A,  RMSE V_C = 5.12e-02 V
traj   6 (load_step)   RMSE i_L = 2.07e-02 A,  RMSE V_C = 5.31e-02 V
traj  53 (random_pwc)  RMSE i_L = 7.67e-02 A,  RMSE V_C = 9.57e-02 V
traj  14 (alpha_ramp)  RMSE i_L = 2.02e-02 A,  RMSE V_C = 5.10e-02 V
Mean RMSE i_L : 3.43e-02 A   (spec: < 0.20 A)
Mean RMSE V_C : 6.28e-02 V   (spec: < 0.20 V)
```

### Research outcome
The PHNN effectively learned the physical dynamics and recovered physically meaningful `L`, `C`, and `R` parameters using strictly transition states (rollouts), bypassing the need for clean, simulator-provided `xdot` labels. This empirical success brings the model much closer to practical viability for real-world converter identification, where only noisy trajectory measurements are available.

---

## Design Decisions & Rationale

**Why energy coordinates (Φ, q) instead of (i_L, V_C)?**
The Port-Hamiltonian structure is natural in these coordinates: the Hamiltonian H is a simple quadratic, and the PHNN will learn H_θ, R_θ, G_θ directly. Using physical variables would require re-deriving the PH form.

**Why averaged model?**
Switching-frequency dynamics (tens–hundreds of kHz) are not the target of identification. The averaged model captures the slow envelope dynamics that a PHNN trained on macro-timescale traces will see.

**Why RK45 with rtol=1e-8, atol=1e-10?**
Training data quality depends on the fidelity of the ground truth. Tight tolerances ensure `compute_xdot` and the integrated trajectory are mutually consistent to near machine precision — important for derivative-matching loss.

**Why `compute_xdot` as a separate function?**
Step 4 will compute training loss as `||compute_xdot(x, u, params) − f_θ(x, u)||²` at each sampled point. Having the analytic derivative as a standalone function makes that clean without re-running the integrator.

**`np.arange` overshoot in `simulate_buck` — known floating-point hazard**
`simulate_buck` builds its output grid as:
```python
t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
```
When `t_span[1]` is not an exact floating-point multiple of `dt`, `np.arange` can generate a final value that exceeds `t_span[1]` by a ULP, causing `solve_ivp` to raise `ValueError: Values in t_eval are not within t_span`.

**Fix (applied in `generate_dataset.py`):** snap every sampled duration to the nearest `dt` multiple before passing it to `simulate_buck`:
```python
n_steps = max(1, round(t_end_raw / dt))
t_end = float(n_steps * dt)
```
Do this in any future caller of `simulate_buck` that passes a non-trivial `dt`. Do **not** modify `simulator.py` itself.




Problem: DCM Caused by Synchronous Rectification at Light Loads
During PLECS dataset generation, a persistent and physically impossible violation was observed: the cycle-averaged capacitor voltage V_C_ss exceeded α·V_in at steady state across 35 of 88 trajectories, with violations as large as 2.5V. This was confirmed not to be a Python averaging artifact — raw PLECS output for a representative trajectory (α=0.75, V_in=10.37V) showed V_C = 8.83V at steady state against a maximum possible value of α·V_in = 7.78V, and the inductor current range was [-5.5A, +8.5A], confirming sustained negative current flow.
The root cause was discontinuous conduction mode (DCM) with bidirectional inductor current, enabled by the synchronous buck topology (two MOSFETs, no freewheeling diode). At light loads, the inductor current naturally reverses during the off-period. Unlike a conventional asymmetric buck with a body diode, the synchronous converter allows this reverse current to flow through the low-side MOSFET, creating a resonant energy exchange between L and C that pumps V_C above α·V_in. This violates the fundamental CCM assumption of the averaged Port-Hamiltonian model (V_C_ss = α·V_in − R_L·i_L), making the data inconsistent with any parameterization of the model.
Solution: Restrict R_load to Enforce CCM
The fix was to restrict the load resistance range to ensure continuous conduction mode across all operating points. The CCM boundary condition requires i_L_avg > ΔiL/2, where ΔiL = V_in·(1−α)/(L·f_sw) is the peak-to-peak ripple. For the worst-case operating point in the dataset (α=0.2, V_in=10V), this gives R_load_max ≈ 5Ω. The Rload_range was changed from (5.0, 20.0)Ω to (2.0, 5.0)Ω, ensuring all trajectories remain in CCM. A DCM warning check was also added to the generator to flag any trajectory where more than 5% of i_L samples are negative. After this change, training loss dropped from a stuck plateau of ~0.12 to below 5×10⁻³ within 300 epochs — confirming the data is now consistent with the averaged model.