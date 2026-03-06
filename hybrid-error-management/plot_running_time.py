#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_qecc_runtime.py

Benchmark the runtime of different filters (no calibration, no decoding).

For each (d, p_trans):
  - Generate one circuit
  - Sample a fixed number of syndromes
  - Run each estimator's score_batch() multiple times
  - Record average CPU time (seconds per call)

Output:
  figs_runtime/runtime_results.csv
  figs_runtime/runtime_bar_accXX.png
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Callable

# Project imports
try:
    from surface import State_Encoding
    from estimator_weight import SyndromeWeightEstimator
    from estimator_z import ZCosetGapEstimator
    from estimator_NN import NNFilterDecoder
except Exception as e:
    raise RuntimeError(f"Import failed: {e}")

# -------------------------
# Configurable parameters
# -------------------------
OUTDIR = "figs_runtime"
CSV_PATH = os.path.join(OUTDIR, "runtime_results.csv")

N_SYND = 500             # number of sampled syndromes per run
D_LIST = [5, 7, 9, 11]
P_LOCAL = 0.003
P_TRANS_GRID = [0.08, 0.12, 0.16, 0.20]
FILTER_BASE_COLOR = {
    "SW + MWPM": "tab:blue",
    "DS + MWPM":      "tab:red",
    "DS + NN":       "tab:green"
}


NN_DEVICE = "cuda:0"    # or "cpu"
NN_BATCH_SIZE = 4096

os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# Utility
# -------------------------
@dataclass
class ExperimentConfig:
    d: int
    p_local: float
    p_trans: float
    num_layer: int = 1
    seed: int = 12345


def build_sampler(cfg: ExperimentConfig):
    circuit = State_Encoding(d=cfg.d, num_layer=cfg.num_layer,
                             p_local=cfg.p_local, p_trans=cfg.p_trans)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()
    return circuit, dem, sampler

def make_est_ctor(name: str, cfg: ExperimentConfig):
    def ctor(circuit, dem):
        if name == "SW + MWPM":
            return SyndromeWeightEstimator(circuit, dem)
        elif name == "DS + MWPM":
            return ZCosetGapEstimator(circuit, dem)
        elif name == "DS + NN":
            return NNFilterDecoder(
                circuit=circuit, model_path=f"./Model/model_d_{cfg.d}.pt",
                d=cfg.d, num_layer=cfg.num_layer,
                batch_size=NN_BATCH_SIZE, device=NN_DEVICE)
        else:
            raise ValueError(name)
    return ctor



def measure_node_processing_time(est_ctor, circuit, dem, sampler, shots):
    """
    Measure *average* processing latency for one shot:
        t_filter_avg = total_filter_time / N_shots
        t_decode_avg = total_decode_time / N_accepted
        t_node = t_filter_avg + t_decode_avg
    """
    est = est_ctor(circuit, dem=dem)
    synd, obs = sampler.sample(shots, separate_observables=True)

    # --- Filtering ---
    t0 = time.time()
    scores = est.score_batch(synd)
    tau = 0.5 if isinstance(est, NNFilterDecoder) else np.median(scores)
    acc_mask = est.accept_mask(scores, tau)
    t1 = time.time()

    acc_rate = acc_mask.mean()

    # --- Decoding on accepted shots ---
    t2 = time.time()
    if acc_rate > 0:
        if hasattr(est, "matcher_std"):
            est.matcher_std.decode_batch(synd[acc_mask])
        else:
            E = est._fill_event_tensor(synd[acc_mask])
            est._predict_hard_from_soft(est._predict_soft(E))
    t3 = time.time()

    # --- Normalize per shot ---
    t_filter_total = t1 - t0
    t_decode_total = t3 - t2

    t_filter_avg = t_filter_total / shots
    t_decode_avg = t_decode_total / max(1, acc_mask.sum())
    t_node_us = (t_filter_avg + t_decode_avg)*1e6

    return t_node_us





# -------------------------
# Benchmark core
# -------------------------
def run_benchmark() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["estimator","d","p_trans","t_node"])

    results = []
    for d in D_LIST:
        for pT in P_TRANS_GRID:
            cfg = ExperimentConfig(d=d, p_local=P_LOCAL, p_trans=pT)
            keyset = {(r["estimator"], int(r["d"]), round(r["p_trans"], 4)) for _, r in df.iterrows()}
            for name in ["SW + MWPM", "DS + MWPM", "DS + NN"]:
                if (name, d, round(pT,4)) in keyset:
                    print(f"[skip] {name} d={d} pT={pT}")
                    continue
                circuit, dem, sampler = build_sampler(cfg)
                ctor = make_est_ctor(name, cfg)
                t_node_us = measure_node_processing_time(
                    ctor, circuit, dem, sampler, shots=N_SYND
                )
                results.append({
                    "estimator": name,
                    "d": d,
                    "p_trans": pT,
                    "t_node": t_node_us
                })
                print(f"{name:15s} d={d:2d} pT={pT:.3f} → t_node={t_node_us:.3f}us")

                # flush every run (so if it crashes, data stays)
                df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
                df.to_csv(CSV_PATH, index=False)
                results = []  # reset buffer
    return pd.read_csv(CSV_PATH)


# -------------------------
# Plot
# -------------------------
def plot_runtime(df: pd.DataFrame):
    filters = ["SW + MWPM", "DS + MWPM", "DS + NN"]

    d_vals = sorted(df["d"].unique())
    pT_vals = sorted(df["p_trans"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.18
    x = np.arange(len(pT_vals))

    # function to darken with code distance
    def shade(base_color, frac):
        import matplotlib.colors as mcolors
        rgb = np.array(mcolors.to_rgb(base_color))
        return np.clip(rgb * frac + (1 - frac), 0, 1)

    for j, filt in enumerate(filters):
        for i_d, d in enumerate(d_vals):
            frac = 0.5 + 0.4 * (i_d / (len(d_vals) - 1))
            sub = df[(df["d"] == d) & (df["estimator"] == filt)].sort_values("p_trans")
            if sub.empty:
                continue
            col = shade(FILTER_BASE_COLOR[filt], frac)
            xpos = x + (i_d - (len(d_vals) - 1) / 2) * width  # slight offset by d
            ax.bar(xpos, sub["t_node"],
                   width=width * 0.9, color=col, alpha=0.9, zorder = 2 - j,
                   label=f"{filt} (d={d})")
            

    # axes, labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pt:.2f}" for pt in pT_vals])
    ax.set_xlabel("Transmission Error Rate", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (µs)", fontweight="bold")
    # ax.set_title("Filter runtime comparison")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Legend 1: code distance ---
    handles, labels = ax.get_legend_handles_labels()
    gray_handles = []
    gray_cmap = plt.cm.Greys
    n = len(handles)
    for i, (h, lab) in enumerate(zip(handles, labels)):
        gray_shade = gray_cmap(0.4 + 1.5 * (i / max(1, n - 1)))  # 0.4–0.9 light→dark
        patch = plt.Rectangle((0, 0), 1, 1, color=gray_shade, label=lab)
        gray_handles.append(patch)

    leg1 = ax.legend(
        gray_handles, [f"d={d}" for d in sorted(df['d'].unique())],
        title="Code distance",
        loc="upper right", frameon=True, bbox_to_anchor=(1.28, 1.01)
    )
    ax.add_artist(leg1)

    # --- Legend 2: filter type ---
    from matplotlib.patches import Patch
    style_handles = [
        Patch(facecolor=FILTER_BASE_COLOR[name], label=name)
        for name in ["SW + MWPM", "DS + MWPM", "DS + NN"]
    ]

    ax.legend(
        handles=style_handles, title="Filter + Decoder",
        loc="upper right", frameon=True, bbox_to_anchor=(1.34, 0.6)
    )


    plt.tight_layout()
    out_pdf = os.path.join(OUTDIR, "runtime.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved {out_pdf}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    df = run_benchmark()
    plt.rcParams.update({
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
    })
    plot_runtime(df)
