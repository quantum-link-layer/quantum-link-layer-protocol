#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hybrid_qecc_figs.py

Task: threshold-like scan over p_trans, comparing THREE estimators:
  - SyndromeWeightEstimator  (baseline)
  - ZCosetGapEstimator       (advanced)
  - NNFilterDecoder          (NN-based filter+decoder; uses NN soft for filter, NN hard for decode)

For each (estimator, target_accept), we:
  - Fix p_local, sweep p_trans across a grid and multiple distances (e.g., d in {3,5,7,9}).
  - Calibrate threshold tau (quantile on scores from calibration shots).
  - Evaluate conditional logical error only on accepted shots.
  - Save to CSV with ORIGINAL schema:
        estimator,target_accept,p_local,d,p_trans,condL
  - If CSV exists, recompute ONLY the missing/bad points, so you can delete rows to force per-point refresh.

Assumed APIs:
  from surface import State_Encoding
  from estimator_z import ZCosetGapEstimator
  from estimator_weight import SyndromeWeightEstimator
  from estimator_NN import NNFilterDecoder
"""

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Imports from your project
# ---------------------------
try:
    from surface import State_Encoding
    from estimator_z import ZCosetGapEstimator
    from estimator_weight import SyndromeWeightEstimator
    from estimator_NN import NNFilterDecoder
except Exception as e:
    raise RuntimeError(
        "Failed to import project modules. Run from repo root or set PYTHONPATH.\n"
        f"Import error: {e}"
    )

# ---------------------------
# Utilities
# ---------------------------

def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

@dataclass
class ExperimentConfig:
    d: int
    p_local: float
    p_trans: float
    shots_cal: int
    shots_eval: int
    num_layer: int = 1
    seed: int = 12345

def build_circuit_and_sampler(cfg: ExperimentConfig):
    circuit = State_Encoding(
        d=cfg.d,
        num_layer=cfg.num_layer,
        p_local=cfg.p_local,
        p_trans=cfg.p_trans
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()
    return circuit, dem, sampler

# Factory: instantiate estimator by name
def make_estimator(est_name: str, circuit, dem, cfg: ExperimentConfig, args) -> object:
    if est_name == "SyndromeWeight":
        return SyndromeWeightEstimator(circuit, dem=dem)
    elif est_name == "ZCosetGap":
        return ZCosetGapEstimator(circuit, dem=dem)
    elif est_name == "NNFilter":
        model_dir = args.nn_model_dir or "./Model"
        model_path = os.path.join(model_dir, f"model_d_{cfg.d}.pt")
        return NNFilterDecoder(
            circuit=circuit,
            model_path=model_path,
            d=cfg.d,
            num_layer=cfg.num_layer,
            batch_size=args.nn_batch_size,      # try 4096 if memory allows
            device=args.nn_device,
            cal_shots_cap=args.nn_cal_shots_cap # e.g., 10000
        )
    else:
        raise ValueError(f"Unknown estimator name: {est_name}")

def run_once_for_estimator(est_name: str, cfg: ExperimentConfig, target_accept_p: float, args) -> Tuple[float, float, float]:
    """Run calibration + evaluation once. Returns (accept_rate, cond_L, tau)."""
    circuit, dem, sampler = build_circuit_and_sampler(cfg)
    est = make_estimator(est_name, circuit, dem, cfg, args)

    # Calibration
    synd_cal, _ = sampler.sample(cfg.shots_cal, separate_observables=True)
    scores_cal = est.score_batch(synd_cal)
    tau = est.quantile_threshold(scores_cal, target_accept_p)

    # Evaluation
    synd_eval, obs_eval = sampler.sample(cfg.shots_eval, separate_observables=True)
    acc_rate, cond_L = est.evaluate_batch(synd_eval, obs_eval, tau)

    return float(acc_rate), float(cond_L), float(tau)

def _round_key(x: float, ndigits: int = 12) -> float:
    """Round float for stable dict keys / comparisons."""
    return float(np.round(x, ndigits))

def _needed_keys(d_list: List[int], p_trans_grid: List[float]) -> List[Tuple[int, float]]:
    return [(int(d), _round_key(pt)) for d in d_list for pt in p_trans_grid]

def _bad_condL(v) -> bool:
    try:
        fv = float(v)
    except Exception:
        return True
    return (not np.isfinite(fv)) or (fv <= 0.0)

# ---------------------------
# CSV write/load with per-point recompute
# ---------------------------

def write_threshold_scan_csv(csv_path: str,
                             est_name: str,
                             target_accept_p: float,
                             p_local: float,
                             d_list: List[int],
                             p_trans_grid: List[float],
                             shots_cal: int,
                             shots_eval: int,
                             num_layer: int,
                             seed: int,
                             args) -> pd.DataFrame:
    """Build full dataset and write with ORIGINAL schema."""
    rows = []
    for d in d_list:
        for pT in p_trans_grid:
            cfg = ExperimentConfig(
                d=d,
                p_local=p_local,
                p_trans=pT,
                shots_cal=shots_cal,
                shots_eval=shots_eval,
                num_layer=num_layer,
                seed=seed,
            )
            _, condL, _ = run_once_for_estimator(est_name, cfg, target_accept_p, args)
            rows.append({
                "estimator": est_name,
                "target_accept": target_accept_p,
                "p_local": p_local,
                "d": d,
                "p_trans": pT,
                "condL": condL
            })
    df = pd.DataFrame(rows, columns=["estimator","target_accept","p_local","d","p_trans","condL"])
    df.to_csv(csv_path, index=False)
    return df

def load_fix_threshold_scan(csv_path: str,
                            est_name: str,
                            target_accept_p: float,
                            p_local: float,
                            d_list: List[int],
                            p_trans_grid: List[float],
                            shots_cal: int,
                            shots_eval: int,
                            num_layer: int,
                            seed: int,
                            force: bool,
                            args) -> Tuple[pd.DataFrame, int]:
    """
    Load CSV if exists, then recompute only missing/bad (d,p_trans) points
    for the given (estimator, target_accept, p_local).
    Returns (df, n_recomputed).
    If CSV doesn't exist or --force, rebuild the full grid.
    """
    needed = set(_needed_keys(d_list, p_trans_grid))

    if not os.path.exists(csv_path) or force:
        df = write_threshold_scan_csv(csv_path, est_name, target_accept_p, p_local,
                                      d_list, p_trans_grid, shots_cal, shots_eval,
                                      num_layer, seed, args)
        return df, len(needed)

    # Load existing and subset to our config
    df_all = pd.read_csv(csv_path)
    required_cols = ["estimator","target_accept","p_local","d","p_trans","condL"]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df = df_all[(df_all["estimator"] == est_name) &
                (np.isclose(df_all["target_accept"], target_accept_p)) &
                (np.isclose(df_all["p_local"], p_local))].copy()

    # Build a dict of existing good points
    have: Dict[Tuple[int, float], float] = {}
    if not df.empty:
        for _, row in df.iterrows():
            key = (int(row["d"]), _round_key(float(row["p_trans"])))
            if key in needed and not _bad_condL(row["condL"]):
                have[key] = float(row["condL"])

    # Identify work: missing or bad points
    to_compute = [k for k in sorted(needed) if k not in have]
    n_recomputed = 0

    if to_compute:
        new_rows = []
        for d, pt_key in to_compute:
            pT = float(pt_key)
            cfg = ExperimentConfig(
                d=d,
                p_local=p_local,
                p_trans=pT,
                shots_cal=shots_cal,
                shots_eval=shots_eval,
                num_layer=num_layer,
                seed=seed,
            )
            _, condL, _ = run_once_for_estimator(est_name, cfg, target_accept_p, args)
            new_rows.append({
                "estimator": est_name,
                "target_accept": target_accept_p,
                "p_local": p_local,
                "d": d,
                "p_trans": pT,
                "condL": condL
            })
        n_recomputed = len(new_rows)

        # Upsert: drop old rows for those keys (if any), then append new
        if not df_all.empty:
            # normalize p_trans to rounded keys for matching
            df_all["_pt_key"] = df_all["p_trans"].apply(lambda x: _round_key(float(x)))
            drop_keys = set(to_compute)
            drop_mask = (
                (df_all["estimator"] == est_name) &
                (np.isclose(df_all["target_accept"], target_accept_p)) &
                (np.isclose(df_all["p_local"], p_local)) &
                df_all.apply(lambda r: (int(r["d"]), float(r["_pt_key"])) in drop_keys, axis=1)
            )
            df_all = df_all.loc[~drop_mask, required_cols]  # keep only required cols
        else:
            df_all = pd.DataFrame(columns=required_cols)

        df_add = pd.DataFrame(new_rows, columns=required_cols)
        df_all = pd.concat([df_all, df_add], ignore_index=True)

        # Keep ONLY the rows for this estimator+config in this CSV (like your original behavior)
        df_all = df_all[(df_all["estimator"] == est_name) &
                        (np.isclose(df_all["target_accept"], target_accept_p)) &
                        (np.isclose(df_all["p_local"], p_local))]

        # Sort for readability
        df_all = df_all.sort_values(by=["d","p_trans"]).reset_index(drop=True)
        df_all.to_csv(csv_path, index=False)

        return df_all, n_recomputed

    # Nothing to recompute; ensure df has only required cols and sorted
    df = df[required_cols].sort_values(by=["d","p_trans"]).reset_index(drop=True)
    return df, 0

# ---------------------------
# Command: threshold-scan
# ---------------------------

def cmd_threshold_scan(args):
    """
    For each (target_accept, estimator), ensure we have a complete/clean grid in CSV,
    recomputing only missing/bad points, then plot.
    """
    d_list = [int(x) for x in args.d_list.split(",")]
    p_trans_grid = parse_float_list(args.p_trans_grid) or [0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20]
    accept_list = parse_float_list(args.accept_list) or [0.10, 0.20, 0.40, 0.80]

    os.makedirs(args.outdir, exist_ok=True)

    # Add NNFilter to the sweep list
    estimator_names = ["SyndromeWeight", "ZCosetGap", "NNFilter"]

    for target_accept_p in accept_list:
        for est_name in estimator_names:
            csv_path = os.path.join(
                args.outdir,
                f"threshold_scan_{est_name}_acc{int(target_accept_p*100)}.csv"
            )

            df, nfix = load_fix_threshold_scan(
                csv_path=csv_path,
                est_name=est_name,
                target_accept_p=target_accept_p,
                p_local=args.p_local,
                d_list=d_list,
                p_trans_grid=p_trans_grid,
                shots_cal=args.shots_cal,
                shots_eval=args.shots_eval,
                num_layer=args.num_layer,
                seed=args.seed,
                force=args.force,
                args=args
            )

# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Hybrid QECC figures (Task 2: threshold-like scan over p_trans)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Defaults (match your figs_threshold setup)
    DEFAULT_OUTDIR = "figs_threshold"
    DEFAULT_ACCEPT_GRID = "0.10,0.20,0.40,0.80"

    # threshold-scan only
    p2 = sub.add_parser("threshold-scan", help="Task 2: Threshold-like scan over p_trans")
    p2.add_argument("--d-list", type=str, default="3,5,7,9",
                    help="Comma-separated distances to plot, e.g. '3,5,7,9'")
    p2.add_argument("--num-layer", type=int, default=1, help="Circuit layers for State_Encoding (also for NN input depth)")
    p2.add_argument("--p-local", type=float, required=True, help="Fixed local error probability")
    p2.add_argument("--p-trans-grid", type=str,
                    default="0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20",
                    help="Comma-separated sweep for p_trans")
    p2.add_argument("--accept-list", type=str, default=DEFAULT_ACCEPT_GRID,
                    help="Comma-separated target accept rates; a separate figure will be generated for each")
    p2.add_argument("--shots-cal", type=int, default=50_000, help="Calibration shots per (d, p_trans)")
    p2.add_argument("--shots-eval", type=int, default=50_000, help="Evaluation shots per (d, p_trans)")
    p2.add_argument("--seed", type=int, default=12345)
    p2.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    p2.add_argument("--force", action="store_true",
                    help="Force recomputation of ALL points even if CSV exists")
    # NN options
    p2.add_argument("--nn-cal-shots-cap", type=int, default=30000,
                help="Max shots used to calibrate NN quantile (tau). None to disable cap.")
    p2.add_argument("--nn-model-dir", type=str, default="./Model",
                    help="Directory containing model_d_{d}.pt files")
    p2.add_argument("--nn-batch-size", type=int, default=512, help="NN inference batch size")
    p2.add_argument("--nn-device", type=str, default=None, help="e.g., 'cuda:0' or 'cpu'")
    # NN risk-model training options
    p2.add_argument("--risk-shots", type=int, default=300000,
                    help="Number of shots to train the NN risk model per (d, p_trans).")
    p2.add_argument("--risk-epochs", type=int, default=5,
                    help="Number of training epochs for the NN risk model.")
    p2.add_argument("--risk-lr", type=float, default=1e-3,
                    help="Learning rate for the NN risk model.")


    p2.set_defaults(func=cmd_threshold_scan)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
