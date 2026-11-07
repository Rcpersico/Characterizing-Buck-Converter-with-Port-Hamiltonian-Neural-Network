"""Shared utilities for the BidirBuckPHNN data-size ablation study.

All ablation scripts are run from bidir_buck_model/ as the working directory:
    python ablation_data_study/train_ablation_120.py

Regime terminology (synchronous buck, no freewheeling diode):
  - Unidirectional: i_L >= 0 — standard forward conduction.
  - Bidirectional:  i_L < BIDIR_I_THRESHOLD — the synchronous MOSFET allows
                    reverse current that a freewheeling diode would block.
                    A converter WITH a diode would produce classical DCM
                    (current clamped to zero); here current actually goes negative.

Public API (called by training scripts):
    load_and_subset_dataset(npz_path, n_traj, seed)
        Excludes the fixed 20-trajectory test set before subsetting.
        Raises FileNotFoundError if test_indices.json does not exist
        (run build_fixed_test_set.py first).

    evaluate_on_fixed_test_set(model, device)
        Rolls out the trained model on the 20 fixed test trajectories with
        full-trajectory RK4 and returns per-trajectory + aggregate RMSE
        split by regime (bidirectional / unidirectional).

    train_single_seed(n_traj, seed, run_k, n_runs)
        Complete single-seed training run: loads data, trains for EPOCHS
        epochs (no early stopping), saves model + loss plot, evaluates on
        the fixed test set, and returns a result dict.

    save_combined_results(n_traj, per_seed_results)
        Pools per-seed results, computes aggregate statistics, and writes
        results/ablation_{N:03d}_results.json.

    save_results, plot_loss_curve, compute_regime_rmse,
    plot_val_trajectories, stack_multistep_rollout, multistep_rk4,
    regime_weighted_loss
        Unchanged helpers kept for backward compatibility.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — runs once on import
# ---------------------------------------------------------------------------

_ABLATION_DIR = Path(__file__).resolve().parent         # ablation_data_study/
_ROOT = _ABLATION_DIR.parent                            # bidir_buck_model/
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generate_dataset import load_dataset, save_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIDIR_I_THRESHOLD = -0.05  # A — below this i_L is counted as bidirectional

# Paths used by load_and_subset_dataset and evaluate_on_fixed_test_set
_DATA_PATH = _ROOT / "data" / "plecs_bidir_train2.npz"
_TEST_IDX_PATH = _ABLATION_DIR / "data" / "test_indices.json"
_TEST_SET_PATH = _ABLATION_DIR / "data" / "plecs_test_fixed.npz"

# Training hyperparameters — DO NOT CHANGE.
# The ablation studies only the effect of dataset size; all other knobs are fixed.
_EPOCHS = 3000
_LR = 5e-4
_MULTISTEP_N = 30
_DT_TRAIN = 50e-6       # 50 µs — matches dataset sample interval
_BIDIR_LOSS_WEIGHT = 2.0
_SCHEDULER_PATIENCE = 100
_SCHEDULER_FACTOR = 0.5
_L_TRUE = 100e-6        # H — fixed physical parameter
_C_TRUE = 100e-6        # F — fixed physical parameter

SEEDS = [42, 43, 44]    # Three training seeds used by every ablation script


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_stats(values: list) -> dict:
    """Return mean, median, IQR, min, max for a list of floats (None → skipped).

    Always returns a dict with the five keys.  If no valid values are
    available, all entries are float('nan') and n_valid == 0.
    """
    arr = np.array(
        [v for v in values if v is not None and not np.isnan(float(v))],
        dtype=np.float64,
    )
    if len(arr) == 0:
        nan = float("nan")
        return {"n_valid": 0, "mean": nan, "median": nan,
                "iqr": nan, "min": nan, "max": nan}
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return {
        "n_valid": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _rhs(
    model: torch.nn.Module,
    x_np: np.ndarray,
    u_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    xt = torch.tensor(x_np[None], dtype=torch.float32, device=device)
    ut = torch.tensor(u_np[None], dtype=torch.float32, device=device)
    return model(xt, ut, create_graph=False).detach().cpu().numpy()[0]


def _rk4_rollout_np(
    model: torch.nn.Module,
    x0: np.ndarray,
    t: np.ndarray,
    u_eff: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Full RK4 trajectory rollout from x0 = [i_L0, V_C0] in numpy."""
    states = np.zeros((len(t), 2), dtype=np.float32)
    states[0] = x0.astype(np.float32)
    for k in range(len(t) - 1):
        dt_k = float(t[k + 1] - t[k])
        u_k = u_eff[k]
        k1 = _rhs(model, states[k], u_k, device)
        k2 = _rhs(model, states[k] + 0.5 * dt_k * k1, u_k, device)
        k3 = _rhs(model, states[k] + 0.5 * dt_k * k2, u_k, device)
        k4 = _rhs(model, states[k] + dt_k * k3, u_k, device)
        states[k + 1] = states[k] + (dt_k / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


# ---------------------------------------------------------------------------
# Spec-required public API — load_and_subset_dataset (modified)
# ---------------------------------------------------------------------------

def load_and_subset_dataset(
    npz_path: str | Path,
    n_traj: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Load dataset, exclude fixed test set, subset to n_traj, return 80/20 split.

    Reads test_indices.json produced by build_fixed_test_set.py and
    permanently excludes those 20 trajectories from the available pool.
    The remaining pool is then randomly subsampled to n_traj trajectories
    using `seed`, and split 80 / 20 into train and val.

    Args:
        npz_path : Path to the full training dataset (plecs_bidir_train2.npz).
        n_traj   : Number of trajectories to use; must be <= pool size.
        seed     : RNG seed — controls both subset selection and 80/20 ordering.

    Returns:
        (train_trajs, val_trajs) — lists of trajectory dicts.

    Raises:
        FileNotFoundError  if test_indices.json does not exist.
        ValueError         if n_traj exceeds the available training pool.
    """
    if not _TEST_IDX_PATH.exists():
        raise FileNotFoundError(
            f"Test indices file not found: {_TEST_IDX_PATH}\n"
            "Run build_fixed_test_set.py first:\n"
            "  python ablation_data_study/build_fixed_test_set.py"
        )

    with open(_TEST_IDX_PATH) as fh:
        idx_data = json.load(fh)

    test_set: set[int] = set(idx_data["test_indices"])

    all_trajs = load_dataset(Path(npz_path))
    n_total = len(all_trajs)

    # Build pool: every index NOT reserved for the fixed test set
    pool_indices = [i for i in range(n_total) if i not in test_set]
    pool_trajs = [all_trajs[i] for i in pool_indices]
    n_pool = len(pool_trajs)

    if n_traj > n_pool:
        raise ValueError(
            f"Requested {n_traj} trajectories but only {n_pool} are available "
            f"after excluding {len(test_set)} fixed test trajectories "
            f"from {n_total} total."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)[:n_traj]
    subset = [pool_trajs[int(i)] for i in perm]

    n_train = int(round(0.8 * n_traj))
    return subset[:n_train], subset[n_train:]


# ---------------------------------------------------------------------------
# Spec-required public API — evaluate_on_fixed_test_set (new)
# ---------------------------------------------------------------------------

def evaluate_on_fixed_test_set(
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate a trained model on the 20 fixed held-out test trajectories.

    For each trajectory:
      - Rolls out the full trajectory with RK4 integration.
      - Computes RMSE on i_L and V_C split by regime:
          bidirectional mask : i_L < BIDIR_I_THRESHOLD (-0.05 A)
          unidirectional mask: i_L >= BIDIR_I_THRESHOLD
      - Skips a trajectory's regime if its mask is empty (no samples).

    Args:
        model  : Trained BidirBuckPHNN in eval mode (set internally).
        device : Torch device to use for inference.

    Returns:
        Dict with two top-level keys:

        "per_trajectory" : list of 20 dicts, each containing
            traj_idx           (int)
            frac_bidir         (float) — fraction of samples with i_L < threshold
            rmse_iL_bidir      (float | None)
            rmse_vC_bidir      (float | None)
            rmse_combined_bidir (float | None)  — (rmse_iL + rmse_vC) / 2
            rmse_iL_unidir     (float | None)
            rmse_vC_unidir     (float | None)
            rmse_combined_unidir (float | None)

        "aggregate" : dict with keys "bidir" and "unidir", each a stats dict
            containing n_valid, mean, median, iqr, min, max
            computed over the non-None per-trajectory combined RMSE values.

    Raises:
        FileNotFoundError if plecs_test_fixed.npz does not exist.
    """
    if not _TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Fixed test set not found: {_TEST_SET_PATH}\n"
            "Run build_fixed_test_set.py first:\n"
            "  python ablation_data_study/build_fixed_test_set.py"
        )

    test_trajs = load_dataset(_TEST_SET_PATH)
    model.eval()

    per_trajectory: list[dict] = []

    with torch.no_grad():
        for traj_idx, td in enumerate(test_trajs):
            t = td["t"]
            iL_gt = np.asarray(td["i_L"], dtype=np.float32)
            VC_gt = np.asarray(td["V_C"], dtype=np.float32)
            x0 = np.array([iL_gt[0], VC_gt[0]], dtype=np.float32)
            u_eff = np.column_stack((
                np.asarray(td["alpha"], dtype=np.float32)
                * np.asarray(td["V_in"], dtype=np.float32),
                np.asarray(td["i_o"], dtype=np.float32),
            ))

            states = _rk4_rollout_np(model, x0, t, u_eff, device)

            bidir_mask = iL_gt < BIDIR_I_THRESHOLD
            unidir_mask = ~bidir_mask
            frac_bidir = float(np.mean(bidir_mask))

            result: dict = {"traj_idx": traj_idx, "frac_bidir": frac_bidir}

            # Bidirectional regime
            if bidir_mask.any():
                iL_err = float(np.sqrt(np.mean(
                    (states[bidir_mask, 0] - iL_gt[bidir_mask]) ** 2
                )))
                vC_err = float(np.sqrt(np.mean(
                    (states[bidir_mask, 1] - VC_gt[bidir_mask]) ** 2
                )))
                result["rmse_iL_bidir"] = iL_err
                result["rmse_vC_bidir"] = vC_err
                result["rmse_combined_bidir"] = (iL_err + vC_err) / 2.0
            else:
                result["rmse_iL_bidir"] = None
                result["rmse_vC_bidir"] = None
                result["rmse_combined_bidir"] = None

            # Unidirectional regime
            if unidir_mask.any():
                iL_err = float(np.sqrt(np.mean(
                    (states[unidir_mask, 0] - iL_gt[unidir_mask]) ** 2
                )))
                vC_err = float(np.sqrt(np.mean(
                    (states[unidir_mask, 1] - VC_gt[unidir_mask]) ** 2
                )))
                result["rmse_iL_unidir"] = iL_err
                result["rmse_vC_unidir"] = vC_err
                result["rmse_combined_unidir"] = (iL_err + vC_err) / 2.0
            else:
                result["rmse_iL_unidir"] = None
                result["rmse_vC_unidir"] = None
                result["rmse_combined_unidir"] = None

            per_trajectory.append(result)

    bidir_combined = [r["rmse_combined_bidir"] for r in per_trajectory]
    unidir_combined = [r["rmse_combined_unidir"] for r in per_trajectory]

    return {
        "per_trajectory": per_trajectory,
        "aggregate": {
            "bidir": _aggregate_stats(bidir_combined),
            "unidir": _aggregate_stats(unidir_combined),
        },
    }


# ---------------------------------------------------------------------------
# Training helper — single seed
# ---------------------------------------------------------------------------

def train_single_seed(
    n_traj: int,
    seed: int,
    run_k: int,
    n_runs: int,
) -> dict:
    """Complete single-seed training run for the ablation study.

    Trains BidirBuckPHNN for exactly _EPOCHS epochs (no early stopping),
    saves the final model and a loss plot, evaluates on the fixed test set,
    and returns a result dict.

    Args:
        n_traj  : Number of trajectories to use from the training pool.
        seed    : RNG seed for dataset subsetting, weight initialisation,
                  and numpy random state.
        run_k   : 1-based index of this run (used for display only).
        n_runs  : Total number of seeds in this script (used for display).

    Returns:
        Dict with keys:
            seed            (int)
            val_loss        (float) — final-epoch validation loss
            R11, R22        (float) — dissipation matrix entries at rep. point
            test_results    (dict)  — output of evaluate_on_fixed_test_set()
    """
    # Local import so utils_ablation can be imported without phnn_bidir on path
    from phnn_bidir import BidirBuckPHNN  # noqa: PLC0415

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading {n_traj} trajectories (seed={seed}) ...")
    train_trajs, val_trajs = load_and_subset_dataset(_DATA_PATH, n_traj, seed)
    print(f"  Train trajectories : {len(train_trajs)}")
    print(f"  Val   trajectories : {len(val_trajs)}")

    x0_tr, u_tr, xn_tr, dt_tr, bidir_tr = stack_multistep_rollout(
        train_trajs, _MULTISTEP_N
    )
    x0_va, u_va, xn_va, dt_va, bidir_va = stack_multistep_rollout(
        val_trajs, _MULTISTEP_N
    )

    if x0_tr.size == 0 or x0_va.size == 0:
        raise ValueError(
            f"Trajectories are too short for {_MULTISTEP_N}-step rollout windows."
        )

    all_x = np.vstack((x0_tr, xn_tr.reshape(-1, 2)))
    scale_np = np.maximum(all_x.std(axis=0), 1e-6).astype(np.float32)

    n_bidir_tr = int(bidir_tr.sum())
    n_bidir_va = int(bidir_va.sum())
    print(
        f"  Train windows : {x0_tr.shape[0]}  "
        f"(bidirectional: {n_bidir_tr}, unidirectional: {x0_tr.shape[0] - n_bidir_tr})"
    )
    print(
        f"  Val   windows : {x0_va.shape[0]}  "
        f"(bidirectional: {n_bidir_va}, unidirectional: {x0_va.shape[0] - n_bidir_va})"
    )
    print(f"  State scale [A, V] : {scale_np}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}\n")

    sc = torch.tensor(scale_np, device=device)
    tx0_tr = torch.tensor(x0_tr, device=device)
    tu_tr = torch.tensor(u_tr, device=device)
    txn_tr = torch.tensor(xn_tr, device=device)
    tbidir_tr = torch.tensor(bidir_tr, device=device)
    tx0_va = torch.tensor(x0_va, device=device)
    tu_va = torch.tensor(u_va, device=device)
    txn_va = torch.tensor(xn_va, device=device)
    tbidir_va = torch.tensor(bidir_va, device=device)

    model = BidirBuckPHNN(L=_L_TRUE, C=_C_TRUE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=_SCHEDULER_PATIENCE,
        factor=_SCHEDULER_FACTOR,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    header = (
        f"{'Epoch':>6}  {'train':>12}  {'val':>12}  {'lr':>8}  "
        f"{'R11':>8}  {'R22':>8}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, _EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_tr = multistep_rk4(model, tx0_tr, tu_tr, _DT_TRAIN, create_graph=True)
        loss = regime_weighted_loss(
            pred_tr, txn_tr, tbidir_tr, sc, _BIDIR_LOSS_WEIGHT
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_va = multistep_rk4(
                model, tx0_va, tu_va, _DT_TRAIN, create_graph=False
            )
            val_loss = regime_weighted_loss(
                pred_va, txn_va, tbidir_va, sc, _BIDIR_LOSS_WEIGHT
            )

        tr_f = float(loss.detach().cpu())
        va_f = float(val_loss.detach().cpu())
        train_losses.append(tr_f)
        val_losses.append(va_f)
        scheduler.step(va_f)

        if epoch == 1 or epoch % 100 == 0:
            p = model.get_params()
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d}  {tr_f:12.6e}  {va_f:12.6e}  {lr_now:8.2e}  "
                f"{p['R_11']:8.5f}  {p['R_22']:8.5f}"
            )

    # ---- Save model (final epoch — no early stopping) ----
    model_path = (
        _ABLATION_DIR / "models"
        / f"ablation_{n_traj:03d}_seed{seed}.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": _EPOCHS, "seed": seed},
        model_path,
    )
    print(f"\n  Saved model : {model_path}")

    # ---- Loss plot ----
    loss_plot_path = (
        _ABLATION_DIR / "plots"
        / f"ablation_{n_traj:03d}_seed{seed}_loss.png"
    )
    best_epoch = int(np.argmin(val_losses)) + 1
    plot_loss_curve(loss_plot_path, train_losses, val_losses, best_epoch)
    print(f"  Saved loss plot : {loss_plot_path}")

    # ---- Evaluate on fixed test set ----
    print("\n  Evaluating on fixed test set (20 held-out trajectories) ...")
    model.eval()
    test_results = evaluate_on_fixed_test_set(model, device)

    p = model.get_params()
    final_val_loss = float(val_losses[-1])

    # ---- Per-seed summary ----
    print(f"\n  --- Seed {seed} summary (run {run_k} of {n_runs}) ---")
    print(f"  Final val loss   : {final_val_loss:.6e}")
    print(f"  R11={p['R_11']:.5f}  R12={p['R_12']:.5f}  R22={p['R_22']:.5f}")
    for regime in ("bidir", "unidir"):
        a = test_results["aggregate"][regime]
        if a["n_valid"] > 0:
            print(
                f"  Test {regime:6s} RMSE: "
                f"mean={a['mean']:.4e}  median={a['median']:.4e}  "
                f"IQR={a['iqr']:.4e}  "
                f"(n={a['n_valid']} trajectories)"
            )
        else:
            print(f"  Test {regime:6s} RMSE: no trajectories of this regime")

    return {
        "seed": seed,
        "val_loss": final_val_loss,
        "R11": float(p["R_11"]),
        "R22": float(p["R_22"]),
        "test_results": test_results,
    }


# ---------------------------------------------------------------------------
# Aggregate results across seeds and save combined JSON
# ---------------------------------------------------------------------------

def save_combined_results(
    n_traj: int,
    per_seed_results: list[dict],
) -> Path:
    """Pool per-seed results, compute aggregate statistics, and save JSON.

    Combines the 3 per-seed result dicts returned by train_single_seed into
    a single results file read by compare_ablation_results.py.

    The JSON layout:
        n_traj       — dataset size for this ablation run
        seeds        — list of seeds used
        per_seed     — list of per-seed result dicts (raw output of train_single_seed)
        aggregate    — mean ± std of val_loss, R11, R22 across seeds
        test_set     — pooled per-trajectory RMSE statistics across all seeds:
                           bidir   : {n_valid, mean, median, iqr, min, max}
                           unidir  : {n_valid, mean, median, iqr, min, max}
                           all_rmse_bidir   : list of n_seeds×20 floats/None
                           all_rmse_unidir  : list of n_seeds×20 floats/None
                           n_evaluations    : total rollouts (n_seeds × 20)

    Args:
        n_traj           : Dataset size label.
        per_seed_results : List of dicts from train_single_seed.

    Returns:
        Path to the saved JSON file.
    """
    val_losses = [r["val_loss"] for r in per_seed_results]
    R11s = [r["R11"] for r in per_seed_results]
    R22s = [r["R22"] for r in per_seed_results]

    n = len(per_seed_results)
    std_kw = dict(ddof=1) if n > 1 else dict(ddof=0)

    # Pool all per-trajectory RMSE values across seeds (None = regime absent)
    all_bidir: list = []
    all_unidir: list = []
    for r in per_seed_results:
        for pt in r["test_results"]["per_trajectory"]:
            all_bidir.append(pt["rmse_combined_bidir"])
            all_unidir.append(pt["rmse_combined_unidir"])

    combined = {
        "n_traj": n_traj,
        "seeds": [r["seed"] for r in per_seed_results],
        "per_seed": per_seed_results,
        "aggregate": {
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses, **std_kw)),
            "R11_mean": float(np.mean(R11s)),
            "R11_std": float(np.std(R11s, **std_kw)),
            "R22_mean": float(np.mean(R22s)),
            "R22_std": float(np.std(R22s, **std_kw)),
        },
        "test_set": {
            "bidir": _aggregate_stats(all_bidir),
            "unidir": _aggregate_stats(all_unidir),
            "all_rmse_bidir": all_bidir,    # list of None | float
            "all_rmse_unidir": all_unidir,
            "n_evaluations": n * 20,
        },
    }

    save_path = _ABLATION_DIR / "results" / f"ablation_{n_traj:03d}_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nSaved combined results : {save_path}")

    # Aggregate summary print
    agg = combined["aggregate"]
    ts = combined["test_set"]
    print("=== AGGREGATE SUMMARY ===")
    print(
        f"  val_loss : {agg['val_loss_mean']:.4e} ± {agg['val_loss_std']:.4e}"
    )
    print(
        f"  R11      : {agg['R11_mean']:.5f} ± {agg['R11_std']:.5f}"
    )
    print(
        f"  R22      : {agg['R22_mean']:.5f} ± {agg['R22_std']:.5f}"
    )
    for regime in ("bidir", "unidir"):
        a = ts[regime]
        if a["n_valid"] > 0:
            print(
                f"  test {regime:6s} RMSE: "
                f"median={a['median']:.4e}  IQR={a['iqr']:.4e}  "
                f"(n={a['n_valid']})"
            )
        else:
            print(f"  test {regime:6s} RMSE: no valid trajectories")

    return save_path


# ---------------------------------------------------------------------------
# save_results — legacy helper (single-seed format, kept for backward compat)
# ---------------------------------------------------------------------------

def save_results(
    save_path: str | Path,
    val_losses: list[float],
    learned_params: dict,
    rmse_bidir: float,
    rmse_unidir: float,
    n_traj: int | None = None,
) -> Path:
    """Save a single-seed results.json summary (kept for backward compatibility).

    Args:
        save_path:      Full path to the output .json file.
        val_losses:     Per-epoch validation losses.
        learned_params: Dict from model.get_params() — must contain R_11, R_22.
        rmse_bidir:     Mean RMSE on bidirectional validation trajectories.
        rmse_unidir:    Mean RMSE on unidirectional validation trajectories.
        n_traj:         Number of trajectories used (stored for the comparison script).

    Returns:
        Path to the saved file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "best_val_loss": float(min(val_losses)),
        "best_epoch": int(np.argmin(val_losses) + 1),
        "R11": float(learned_params.get("R_11", float("nan"))),
        "R22": float(learned_params.get("R_22", float("nan"))),
        "rmse_unidir": float(rmse_unidir),
        "rmse_bidir": float(rmse_bidir),
        "learned_params": {k: float(v) for k, v in learned_params.items()},
    }
    if n_traj is not None:
        results["n_traj"] = int(n_traj)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    return save_path


# ---------------------------------------------------------------------------
# plot_loss_curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    save_path: str | Path,
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
) -> Path:
    """Save a semilogy training/validation loss curve to save_path.

    Args:
        save_path:    Full path to the output .png file.
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        best_epoch:   Epoch with the minimum validation loss (informational;
                      no early stopping is used).

    Returns:
        Path to the saved file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="train", alpha=0.8)
    ax.semilogy(epochs, val_losses, label="validation", alpha=0.8)
    ax.axvline(
        best_epoch, color="gray", ls="--", alpha=0.5,
        label=f"min val (epoch {best_epoch})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regime-weighted rollout MSE")
    ax.set_title("Ablation: Training Loss Curve (3000 epochs, no early stopping)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# stack_multistep_rollout
# ---------------------------------------------------------------------------

def stack_multistep_rollout(
    trajectories: list[dict],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows for multi-step rollout training.

    A window is flagged as bidirectional (reverse-current) if any sample in
    the window has i_L < BIDIR_I_THRESHOLD.  Stride = n_steps // 2 (50% overlap).

    Returns:
        x0       : [M, 2]           initial states [i_L, V_C]
        u        : [M, n_steps, 2]  input sequences [alpha*V_in, i_o]
        xn       : [M, n_steps, 2]  target state sequences
        dt       : [M]              timestep per window [s]
        is_bidir : [M] bool         True = window contains bidirectional samples
    """
    x0_list, u_list, xn_list, dt_list, bidir_list = [], [], [], [], []
    for td in trajectories:
        iL = td["i_L"].astype(np.float32)
        VC = td["V_C"].astype(np.float32)
        alpha = td["alpha"].astype(np.float32)
        Vin = td["V_in"].astype(np.float32)
        io = td["i_o"].astype(np.float32)
        t = td["t"]
        x = np.stack([iL, VC], axis=1)
        u = np.stack([alpha * Vin, io], axis=1)
        dt = float(np.median(np.diff(t)))
        stride = max(1, n_steps // 2)
        for start in range(0, len(t) - n_steps - 1, stride):
            is_bidir = bool(np.any(iL[start: start + n_steps] < BIDIR_I_THRESHOLD))
            x0_list.append(x[start])
            u_list.append(u[start: start + n_steps])
            xn_list.append(x[start + 1: start + n_steps + 1])
            dt_list.append(dt)
            bidir_list.append(is_bidir)
    if not x0_list:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0, n_steps, 2), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), bool),
        )
    return (
        np.array(x0_list, np.float32),
        np.array(u_list, np.float32),
        np.array(xn_list, np.float32),
        np.array(dt_list, np.float32),
        np.array(bidir_list, bool),
    )


# ---------------------------------------------------------------------------
# multistep_rk4
# ---------------------------------------------------------------------------

def multistep_rk4(
    model: torch.nn.Module,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    create_graph: bool = True,
) -> torch.Tensor:
    """Roll out n_steps of RK4 integration from x0 using the input sequence u_seq.

    Args:
        model:        BidirBuckPHNN instance.
        x0:           [batch, 2] initial states.
        u_seq:        [batch, N, 2] input sequences [alpha*V_in, i_o].
        dt:           Integration timestep [s].
        create_graph: True during training, False during eval.

    Returns:
        [batch, N, 2] predicted states at steps 1..N.
    """
    dt_t = torch.tensor(float(dt), device=x0.device, dtype=x0.dtype)
    preds: list[torch.Tensor] = []
    x = x0
    for k in range(u_seq.shape[1]):
        u_k = u_seq[:, k, :]
        k1 = model(x, u_k, create_graph=create_graph)
        k2 = model(x + 0.5 * dt_t * k1, u_k, create_graph=create_graph)
        k3 = model(x + 0.5 * dt_t * k2, u_k, create_graph=create_graph)
        k4 = model(x + dt_t * k3, u_k, create_graph=create_graph)
        x = x + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not create_graph:
            x = x.detach()
        preds.append(x)
    return torch.stack(preds, dim=1)  # [batch, N, 2]


# ---------------------------------------------------------------------------
# regime_weighted_loss
# ---------------------------------------------------------------------------

def regime_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_bidir: torch.Tensor,
    scale: torch.Tensor,
    bidir_weight: float = 2.0,
) -> torch.Tensor:
    """Multi-step MSE with 2× weight on bidirectional (reverse-current) windows.

    Args:
        pred:         [M, N, 2] predicted states.
        target:       [M, N, 2] target states.
        is_bidir:     [M] bool — True = window contains i_L < BIDIR_I_THRESHOLD.
        scale:        [2] normalisation scale for [i_L, V_C].
        bidir_weight: Multiplier for reverse-current windows (default 2.0).

    Returns:
        Scalar loss.
    """
    sq_err = ((pred - target) / scale).pow(2)  # [M, N, 2]
    per_window = sq_err.mean(dim=[1, 2])        # [M]
    weights = torch.where(is_bidir, bidir_weight, 1.0)
    return (weights * per_window).mean()


# ---------------------------------------------------------------------------
# compute_regime_rmse — validation-set RMSE split by regime
# ---------------------------------------------------------------------------

def compute_regime_rmse(
    model: torch.nn.Module,
    val_trajs: list[dict],
    device: torch.device,
) -> tuple[float, float]:
    """Compute trajectory-level RMSE split by regime on the validation set.

    Regime classification per trajectory:
      Bidirectional  — any sample has i_L < BIDIR_I_THRESHOLD (reverse current).
      Unidirectional — i_L stays >= BIDIR_I_THRESHOLD throughout.

    The per-trajectory scalar RMSE is the mean of RMSE_iL and RMSE_VC.

    Args:
        model:     Trained BidirBuckPHNN.
        val_trajs: Validation trajectory dicts from load_and_subset_dataset.
        device:    Torch device.

    Returns:
        (rmse_bidir, rmse_unidir) — means over trajectories of each regime.
        Returns float('nan') if no trajectories of that regime are present.
    """
    model.eval()
    rmse_bidir_list: list[float] = []
    rmse_unidir_list: list[float] = []

    with torch.no_grad():
        for td in val_trajs:
            t = td["t"]
            iL_gt = td["i_L"].astype(np.float32)
            VC_gt = td["V_C"].astype(np.float32)
            x0 = np.array([iL_gt[0], VC_gt[0]], dtype=np.float32)
            u_eff = np.column_stack((
                td["alpha"].astype(np.float32) * td["V_in"].astype(np.float32),
                td["i_o"].astype(np.float32),
            ))

            states = _rk4_rollout_np(model, x0, t, u_eff, device)

            bidir_mask = iL_gt < BIDIR_I_THRESHOLD
            unidir_mask = ~bidir_mask

            if bidir_mask.any():
                rmse_iL = float(np.sqrt(np.mean(
                    (states[bidir_mask, 0] - iL_gt[bidir_mask]) ** 2
                )))
                rmse_vC = float(np.sqrt(np.mean(
                    (states[bidir_mask, 1] - VC_gt[bidir_mask]) ** 2
                )))
                rmse_bidir_list.append((rmse_iL + rmse_vC) / 2.0)

            if unidir_mask.any():
                rmse_iL = float(np.sqrt(np.mean(
                    (states[unidir_mask, 0] - iL_gt[unidir_mask]) ** 2
                )))
                rmse_vC = float(np.sqrt(np.mean(
                    (states[unidir_mask, 1] - VC_gt[unidir_mask]) ** 2
                )))
                rmse_unidir_list.append((rmse_iL + rmse_vC) / 2.0)

    rmse_bidir = float(np.mean(rmse_bidir_list)) if rmse_bidir_list else float("nan")
    rmse_unidir = float(np.mean(rmse_unidir_list)) if rmse_unidir_list else float("nan")
    return rmse_bidir, rmse_unidir


# ---------------------------------------------------------------------------
# plot_val_trajectories
# ---------------------------------------------------------------------------

def plot_val_trajectories(
    save_path: str | Path,
    model: torch.nn.Module,
    val_trajs: list[dict],
    device: torch.device,
    max_plots: int = 6,
) -> Path:
    """Roll out the model on validation trajectories and plot predicted vs PLECS.

    Args:
        save_path: Full path to the output .png file.
        model:     Trained BidirBuckPHNN.
        val_trajs: Validation trajectory dicts from load_and_subset_dataset.
        device:    Torch device.
        max_plots: Maximum number of trajectories to plot (default 6).

    Returns:
        Path to the saved file.
    """
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_plots = min(len(val_trajs), max_plots)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, td in enumerate(val_trajs[:n_plots]):
            t = td["t"]
            iL_gt = td["i_L"].astype(np.float32)
            VC_gt = td["V_C"].astype(np.float32)
            x0 = np.array([iL_gt[0], VC_gt[0]], dtype=np.float32)
            u_eff = np.column_stack((
                td["alpha"].astype(np.float32) * td["V_in"].astype(np.float32),
                td["i_o"].astype(np.float32),
            ))

            states = _rk4_rollout_np(model, x0, t, u_eff, device)

            rmse_iL = float(np.sqrt(np.mean((states[:, 0] - iL_gt) ** 2)))
            rmse_vC = float(np.sqrt(np.mean((states[:, 1] - VC_gt) ** 2)))
            is_bidir = bool(np.any(iL_gt < BIDIR_I_THRESHOLD))
            regime_label = "bidirectional" if is_bidir else "unidirectional"
            t_ms = t * 1e3

            axes[row, 0].plot(t_ms, iL_gt, "k--", lw=1.0, label="PLECS")
            axes[row, 0].plot(t_ms, states[:, 0], color="tab:blue", lw=1.2, label="PHNN")
            axes[row, 0].axhline(0, color="gray", lw=0.5, ls=":")
            axes[row, 0].set_ylabel("i_L [A]")
            axes[row, 0].set_title(
                f"Val traj {row}  [{regime_label}]  RMSE i_L={rmse_iL:.3e} A"
            )
            axes[row, 0].legend(fontsize=8)
            axes[row, 0].grid(True, alpha=0.3)

            axes[row, 1].plot(t_ms, VC_gt, "k--", lw=1.0, label="PLECS")
            axes[row, 1].plot(t_ms, states[:, 1], color="tab:orange", lw=1.2, label="PHNN")
            axes[row, 1].set_ylabel("V_C [V]")
            axes[row, 1].set_title(
                f"Val traj {row}  [{regime_label}]  RMSE V_C={rmse_vC:.3e} V"
            )
            axes[row, 1].legend(fontsize=8)
            axes[row, 1].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [ms]")
    fig.suptitle(
        "Ablation: Validation Trajectories — Predicted vs PLECS", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
