"""Verify that the four plotted trajectories are genuinely held-out."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ckpt = torch.load(ROOT / "models" / "buck_phnn.pt", map_location="cpu", weights_only=False)
train_idx = np.asarray(ckpt["train_indices"])
val_idx   = np.asarray(ckpt["val_indices"])

print(f"Train trajectories ({len(train_idx)}): {sorted(train_idx)}")
print(f"Val   trajectories ({len(val_idx)}):   {sorted(val_idx)}")
print()

# Reproduce exactly the selection in test_buck_phnn.py
rng = np.random.default_rng(42)
test_chosen = rng.choice(val_idx, size=4, replace=False)
print(f"Trajectories plotted by test script: {test_chosen}")
print()
all_held_out = True
for i in test_chosen:
    in_tr = int(i) in train_idx
    in_va = int(i) in val_idx
    status = "IN VAL only" if (in_va and not in_tr) else "*** PROBLEM ***"
    print(f"  index {i:3d}:  in_train={in_tr}  in_val={in_va}  → {status}")
    if in_tr:
        all_held_out = False

print()
print("All four trajectories are truly held-out:", all_held_out)
