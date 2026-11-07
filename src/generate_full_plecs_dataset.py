"""
Multi-converter PLECS dataset generation.

Generates 100 trajectories (with load steps) for each of 10 distinct buck
converter configurations and saves them to data/fullPlecsDataset/ as
buckdata1.npz … buckdata10.npz.

Converter configurations span a realistic design space:
  - Inductance:    50 µH … 500 µH
  - Capacitance:   47 µF … 470 µF
  - ESR_L:         0.05 Ω … 0.5 Ω
  - ESR_C:         0.01 Ω … 0.1 Ω
  - Switching freq: 50 kHz … 200 kHz
  - V_in range:    varies per converter

Running this script requires PLECS Standalone to be open and the RPC server
to be active on port 1080.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_plecs_dataset import (
    PLECSRPCSimulator,
    generate_plecs_dataset_with_steps,
)
from generate_dataset import save_dataset  # noqa: F401


# ---------------------------------------------------------------------------
# Converter configurations
# ---------------------------------------------------------------------------

CONVERTERS = [
    {
        "name": "buckdata1",
        "description": "Baseline 12 V→5 V, medium L/C",
        "params": {
            "L": 100e-6,
            "C": 220e-6,
            "ESR_L": 0.10,
            "ESR_C": 0.05,
            "cycle_frequency": 100e3,
        },
        "config": {
            "dt": 50e-6,
            "Vin_range": (10.0, 14.0),
            "alpha_range": (0.2, 0.8),
            "Rload_range": (2.0, 6.0),
            "sigma_iL": 0.02,
            "sigma_VC": 0.05,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (10e-3, 15e-3),
        },
    },
    {
        "name": "buckdata2",
        "description": "High-frequency 24 V input, small L",
        "params": {
            "L": 50e-6,
            "C": 100e-6,
            "ESR_L": 0.08,
            "ESR_C": 0.03,
            "cycle_frequency": 200e3,
        },
        "config": {
            "dt": 25e-6,
            "Vin_range": (20.0, 28.0),
            "alpha_range": (0.2, 0.7),
            "Rload_range": (3.0, 8.0),
            "sigma_iL": 0.02,
            "sigma_VC": 0.05,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (5e-3, 10e-3),
        },
    },
    {
        "name": "buckdata3",
        "description": "Low-frequency, large inductance, heavy load",
        "params": {
            "L": 500e-6,
            "C": 470e-6,
            "ESR_L": 0.30,
            "ESR_C": 0.08,
            "cycle_frequency": 50e3,
        },
        "config": {
            "dt": 100e-6,
            "Vin_range": (10.0, 14.0),
            "alpha_range": (0.3, 0.7),
            "Rload_range": (1.0, 4.0),
            "sigma_iL": 0.03,
            "sigma_VC": 0.05,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (20e-3, 30e-3),
        },
    },
    {
        "name": "buckdata4",
        "description": "High-voltage input 48 V, medium freq",
        "params": {
            "L": 200e-6,
            "C": 150e-6,
            "ESR_L": 0.20,
            "ESR_C": 0.05,
            "cycle_frequency": 100e3,
        },
        "config": {
            "dt": 50e-6,
            "Vin_range": (42.0, 54.0),
            "alpha_range": (0.1, 0.5),
            "Rload_range": (5.0, 15.0),
            "sigma_iL": 0.03,
            "sigma_VC": 0.08,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (10e-3, 15e-3),
        },
    },
    {
        "name": "buckdata5",
        "description": "Low ESR, low noise design",
        "params": {
            "L": 150e-6,
            "C": 330e-6,
            "ESR_L": 0.05,
            "ESR_C": 0.01,
            "cycle_frequency": 150e3,
        },
        "config": {
            "dt": 33e-6,
            "Vin_range": (10.0, 14.0),
            "alpha_range": (0.2, 0.8),
            "Rload_range": (2.0, 7.0),
            "sigma_iL": 0.01,
            "sigma_VC": 0.02,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (10e-3, 15e-3),
        },
    },
    {
        "name": "buckdata6",
        "description": "High ESR, lossy components",
        "params": {
            "L": 100e-6,
            "C": 220e-6,
            "ESR_L": 0.50,
            "ESR_C": 0.10,
            "cycle_frequency": 100e3,
        },
        "config": {
            "dt": 50e-6,
            "Vin_range": (10.0, 14.0),
            "alpha_range": (0.2, 0.8),
            "Rload_range": (2.0, 6.0),
            "sigma_iL": 0.03,
            "sigma_VC": 0.07,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (10e-3, 15e-3),
        },
    },
    {
        "name": "buckdata7",
        "description": "Automotive 12 V input, tight duty cycle",
        "params": {
            "L": 75e-6,
            "C": 180e-6,
            "ESR_L": 0.12,
            "ESR_C": 0.04,
            "cycle_frequency": 120e3,
        },
        "config": {
            "dt": 42e-6,
            "Vin_range": (10.5, 13.5),
            "alpha_range": (0.3, 0.6),
            "Rload_range": (1.5, 5.0),
            "sigma_iL": 0.02,
            "sigma_VC": 0.05,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 3,
            "segment_duration_range": (8e-3, 12e-3),
        },
    },
    {
        "name": "buckdata8",
        "description": "Wide input range, three load steps",
        "params": {
            "L": 250e-6,
            "C": 100e-6,
            "ESR_L": 0.15,
            "ESR_C": 0.06,
            "cycle_frequency": 80e3,
        },
        "config": {
            "dt": 63e-6,
            "Vin_range": (8.0, 16.0),
            "alpha_range": (0.2, 0.8),
            "Rload_range": (2.0, 10.0),
            "sigma_iL": 0.03,
            "sigma_VC": 0.06,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 3,
            "segment_duration_range": (12e-3, 20e-3),
        },
    },
    {
        "name": "buckdata9",
        "description": "Telecom 48 V→12 V, low ripple",
        "params": {
            "L": 400e-6,
            "C": 390e-6,
            "ESR_L": 0.25,
            "ESR_C": 0.07,
            "cycle_frequency": 75e3,
        },
        "config": {
            "dt": 67e-6,
            "Vin_range": (44.0, 52.0),
            "alpha_range": (0.15, 0.45),
            "Rload_range": (4.0, 12.0),
            "sigma_iL": 0.02,
            "sigma_VC": 0.05,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (15e-3, 25e-3),
        },
    },
    {
        "name": "buckdata10",
        "description": "Very high frequency GaN-class design",
        "params": {
            "L": 25e-6,
            "C": 47e-6,
            "ESR_L": 0.05,
            "ESR_C": 0.02,
            "cycle_frequency": 500e3,
        },
        "config": {
            "dt": 10e-6,
            "Vin_range": (10.0, 14.0),
            "alpha_range": (0.2, 0.8),
            "Rload_range": (2.0, 6.0),
            "sigma_iL": 0.01,
            "sigma_VC": 0.03,
            "iL_signal_idx": 0,
            "VC_signal_idx": 1,
            "n_load_steps": 2,
            "segment_duration_range": (4e-3, 8e-3),
        },
    },
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _save_notes(out_dir: Path) -> None:
    """Write converter_notes.json alongside the .npz files."""
    notes = {}
    for conv in CONVERTERS:
        p = conv["params"]
        c = conv["config"]
        notes[conv["name"]] = {
            "description": conv["description"],
            "params": {
                "L_uH": p["L"] * 1e6,
                "C_uF": p["C"] * 1e6,
                "ESR_L_ohm": p["ESR_L"],
                "ESR_C_ohm": p["ESR_C"],
                "switching_frequency_kHz": p["cycle_frequency"] / 1e3,
            },
            "operating_conditions": {
                "Vin_range_V": list(c["Vin_range"]),
                "alpha_range": list(c["alpha_range"]),
                "Rload_range_ohm": list(c["Rload_range"]),
                "dt_us": c["dt"] * 1e6,
                "n_load_steps": c["n_load_steps"],
                "segment_duration_range_ms": [
                    c["segment_duration_range"][0] * 1e3,
                    c["segment_duration_range"][1] * 1e3,
                ],
                "noise_sigma_iL_A": c["sigma_iL"],
                "noise_sigma_VC_V": c["sigma_VC"],
            },
        }

    notes_path = out_dir / "converter_notes.json"
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"Notes saved -> {notes_path}")


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "buck_converter.plecs"
    out_dir = Path(__file__).parent.parent / "data" / "fullPlecsDataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = PLECSRPCSimulator(model_path)

    for idx, conv in enumerate(CONVERTERS, start=1):
        print(f"\n{'='*60}")
        print(f"Converter {idx}/10: {conv['name']} — {conv['description']}")
        print(f"  L={conv['params']['L']*1e6:.0f} µH  "
              f"C={conv['params']['C']*1e6:.0f} µF  "
              f"ESR_L={conv['params']['ESR_L']:.3f} Ω  "
              f"ESR_C={conv['params']['ESR_C']:.3f} Ω  "
              f"f_sw={conv['params']['cycle_frequency']/1e3:.0f} kHz")
        print(f"{'='*60}")

        trajs = generate_plecs_dataset_with_steps(
            sim,
            n_trajectories=100,
            params=conv["params"],
            config=conv["config"],
            seed=42 + idx,
        )

        out_path = out_dir / f"{conv['name']}.npz"
        save_dataset(trajs, str(out_path))
        total_steps = sum(len(t["t"]) for t in trajs)
        print(f"  Saved {len(trajs)} trajectories ({total_steps:,} timesteps) -> {out_path}")

    _save_notes(out_dir)

    print(f"\nAll 10 converters done. Files in {out_dir}")
