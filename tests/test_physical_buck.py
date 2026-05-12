"""Evaluate a trained PhysicalBuckModel checkpoint on the physical PLECS dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import load_dataset
from phnn import PhysicalBuckModel
from train_physical_buck import (
    DATA_PATH,
    INIT_C,
    INIT_L,
    INIT_R,
    MODEL_PATH,
    evaluate,
)


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}\n"
            "Run: python src/train_physical_buck.py"
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run: python src/generate_plecs_dataset.py or "
            "python src/convert_to_physical_dataset.py"
        )

    trajectories = load_dataset(DATA_PATH)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicalBuckModel(INIT_L, INIT_C, INIT_R).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluate(model, ckpt, trajectories, device)
