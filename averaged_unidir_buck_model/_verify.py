"""Quick smoke-test: verify averaged_ccm_buck_model pipeline loads without errors."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from generate_dataset import load_dataset
from phnn import SimulationBuckPHNN
import torch

DATA = HERE / "data" / "plecs_sweep_train.npz"
MODEL = HERE / "models" / "sweep_buck_phnn.pt"

assert DATA.exists(), f"Dataset not found: {DATA}"
assert MODEL.exists(), f"Model not found: {MODEL}"

trajs = load_dataset(DATA)
print(f"Loaded {len(trajs)} trajectories from {DATA.name}")
print(f"First traj keys: {list(trajs[0].keys())}")
print(f"First traj length: {len(trajs[0]['t'])} samples")

ckpt = torch.load(str(MODEL), map_location="cpu", weights_only=False)
model = SimulationBuckPHNN(L=100e-6, C=100e-6)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
p = model.get_params()
print(f"Model loaded: R_L={p['R_L']:.5f} Ohm  R_C={p['R_C']:.5f} Ohm  H_NN_norm={p['H_NN_norm']:.4e}")
print("CCM pipeline verification: PASSED")
