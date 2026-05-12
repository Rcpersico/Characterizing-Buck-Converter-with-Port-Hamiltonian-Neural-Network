import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from generate_dataset import load_dataset

# Load dataset
trajs = load_dataset('data/plecs_train.npz')
print(f"Loaded {len(trajs)} trajectories\n")

# Pick the first trajectory
t0 = trajs[0]
t = t0['t']
iL = t0['i_L']
VC = t0['V_C']
Phi = t0['Phi']
q = t0['q']
alpha = t0['alpha'][0]
Vin = t0['V_in'][0]
io = t0['i_o']

L = 100e-6
C = 100e-6
R = 0.1

# Check 1: Are Phi and q consistent with i_L and V_C?
Phi_expected = L * iL
q_expected = C * VC
print("=== Check 1: Energy coordinate consistency ===")
print(f"max|Phi - L*iL|: {np.max(np.abs(Phi - Phi_expected)):.2e} Wb")
print(f"max|q - C*VC|:   {np.max(np.abs(q - q_expected)):.2e} C")
print()

# Check 2: Are the derivatives correct?
dPhi_dt_stored = t0['dPhi_dt']
dq_dt_stored = t0['dq_dt']
dPhi_dt_expected = alpha * Vin - R * iL - VC
dq_dt_expected = iL - io

print("=== Check 2: Derivative labels ===")
print(f"max|dPhi/dt error|: {np.max(np.abs(dPhi_dt_stored - dPhi_dt_expected)):.2e}")
print(f"max|dq/dt error|:   {np.max(np.abs(dq_dt_stored - dq_dt_expected)):.2e}")
print()

# Check 3: Numerical derivative vs stored derivative
dt_vec = np.diff(t)
dPhi_dt_numeric = np.diff(Phi) / dt_vec
dq_dt_numeric = np.diff(q) / dt_vec

print("=== Check 3: Stored vs numerical derivatives ===")
print(f"Stored dPhi/dt range: [{dPhi_dt_stored.min():.3f}, {dPhi_dt_stored.max():.3f}]")
print(f"Numeric dPhi/dt range: [{dPhi_dt_numeric.min():.3f}, {dPhi_dt_numeric.max():.3f}]")
print(f"RMS difference: {np.sqrt(np.mean((dPhi_dt_stored[:-1] - dPhi_dt_numeric)**2)):.3e}")
print()
print(f"Stored dq/dt range: [{dq_dt_stored.min():.3f}, {dq_dt_stored.max():.3f}]")
print(f"Numeric dq/dt range: [{dq_dt_numeric.min():.3f}, {dq_dt_numeric.max():.3f}]")
print(f"RMS difference: {np.sqrt(np.mean((dq_dt_stored[:-1] - dq_dt_numeric)**2)):.3e}")
print()

# Check 4: Look at the actual waveforms
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Row 1: States
axes[0, 0].plot(t * 1e3, iL, 'b-', lw=1.5, label='i_L')
axes[0, 0].set_ylabel('i_L [A]')
axes[0, 0].set_title(f'Traj 0: α={alpha:.2f}, V_in={Vin:.1f}V')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(t * 1e3, VC, 'r-', lw=1.5, label='V_C')
axes[0, 1].set_ylabel('V_C [V]')
axes[0, 1].set_title(f'i_o range: [{io.min():.2f}, {io.max():.2f}] A')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Row 2: Derivatives (stored)
axes[1, 0].plot(t * 1e3, dPhi_dt_stored, 'b-', lw=1.5, label='stored dΦ/dt')
axes[1, 0].set_ylabel('dΦ/dt [Wb/s]')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(t * 1e3, dq_dt_stored, 'r-', lw=1.5, label='stored dq/dt')
axes[1, 1].set_ylabel('dq/dt [C/s]')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# Row 3: Derivative comparison
axes[2, 0].plot(t[:-1] * 1e3, dPhi_dt_numeric, 'g-', lw=1.5, alpha=0.7, label='numeric dΦ/dt')
axes[2, 0].plot(t * 1e3, dPhi_dt_stored, 'b--', lw=1.0, label='stored dΦ/dt')
axes[2, 0].set_ylabel('dΦ/dt [Wb/s]')
axes[2, 0].set_xlabel('Time [ms]')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

axes[2, 1].plot(t[:-1] * 1e3, dq_dt_numeric, 'g-', lw=1.5, alpha=0.7, label='numeric dq/dt')
axes[2, 1].plot(t * 1e3, dq_dt_stored, 'r--', lw=1.0, label='stored dq/dt')
axes[2, 1].set_ylabel('dq/dt [C/s]')
axes[2, 1].set_xlabel('Time [ms]')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend()

plt.tight_layout()
plt.savefig('tests/plecs_data_diagnostic.png', dpi=150)
print(f"Saved diagnostic plot: tests/plecs_data_diagnostic.png")

# Check 5: FFT to see if filtering worked
from scipy.fft import rfft, rfftfreq
dt_median = np.median(np.diff(t))
fs = 1.0 / dt_median
fft_iL = np.abs(rfft(iL))
freqs = rfftfreq(len(iL), dt_median)

fig2, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(freqs / 1e3, fft_iL, 'b-', lw=1.5)
ax.axvline(1.59, color='green', linestyle='--', lw=2, label='LC resonance ~1.6 kHz')
ax.axvline(20.0, color='orange', linestyle='--', lw=2, label='i_L filter cutoff 20 kHz')
ax.axvline(100.0, color='red', linestyle='--', lw=2, label='Switching freq 100 kHz')
ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('|FFT(i_L)|')
ax.set_title(f'i_L spectrum (dt={dt_median*1e6:.1f} µs, fs={fs/1e3:.1f} kHz)')
ax.set_xlim(0, min(150, 10*fs/2/1e3))
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('tests/plecs_spectrum_check.png', dpi=150)
print(f"Saved spectrum plot: tests/plecs_spectrum_check.png")
print(f"\nIf you see a spike at 100 kHz → filter didn't work")
print("If i_L spectrum drops off sharply after 20 kHz, filter worked")
