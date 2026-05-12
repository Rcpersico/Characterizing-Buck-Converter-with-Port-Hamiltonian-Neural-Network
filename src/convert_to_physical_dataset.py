"""Convert existing PLECS dataset to the physical-coordinate schema."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import load_dataset, save_dataset


ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "data" / "plecs_train.npz"
DST_PATH = ROOT / "data" / "plecs_physical_train.npz"

TRUE_L = 100e-6
TRUE_C = 100e-6


if __name__ == "__main__":
    trajectories = load_dataset(SRC_PATH)
    print(f"Loaded {len(trajectories)} trajectories from {SRC_PATH.name}")

    physical_trajectories = []
    for td in trajectories:
        i_L = td["Phi"] / TRUE_L if "Phi" in td else td["i_L"]
        V_C = td["q"] / TRUE_C if "q" in td else td["V_C"]
        physical_trajectories.append(
            {
                "t": td["t"],
                "i_L": i_L,
                "V_C": V_C,
                "alpha": td["alpha"],
                "V_in": td["V_in"],
                "i_o": td["i_o"],
                "profile_type": td.get("profile_type", "constant"),
                "Phi": td["Phi"],
                "q": td["q"],
            }
        )

    save_dataset(physical_trajectories, DST_PATH)
    print(f"Saved {len(physical_trajectories)} trajectories to {DST_PATH.name}")
    total = sum(len(t["t"]) for t in physical_trajectories)
    print(f"Total timesteps: {total:,}")
