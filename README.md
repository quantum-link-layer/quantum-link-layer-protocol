# Quantum Link-Layer Protocol Simulation Code

This repository contains the simulation code accompanying our paper on **quantum link-layer protocols**. It provides reference circuit implementations for multiple quantum codes, scripts to reproduce the experiments reported in the paper, and a hybrid error management framework that combines filtering and decoding strategies.

> **At a glance**
>
> * **Codes implemented:** Surface code, Steane code, Repetition code
> * **Protocols:** Entanglement Distillation (ED), Entanglement Encoding (EE/LEE), QECC Transmission (SE), 2nd-Generation
> * **Hybrid error management:** Syndrome-weight filter, Z-coset gap filter (MWPM's soft-output), neural-network filter/decoder 
> * **Artifacts:** Logical-error calculators, plotting scripts, experiment drivers, pre-trained NN models

---

## Repository layout

```
.
├── three-protocols/                         # Protocol comparison experiments
│   ├── circuits/
│   │   ├── DEJMPS.py                        # Entanglement distillation (DEJMPS) circuits
│   │   ├── steane.py                        # Steane [[7,1,3]] code circuits
│   │   └── surface.py                       # Surface code circuits
│   ├── cals/
│   │   └── cal_logical_error.py             # Logical error estimation utilities
│   ├── experiments/
│   │   ├── error_correction_of_three_protocols/
│   │   │   └── plot_column_chart.py         # Comparing the three protocols (ED, EE-T, SE) under error correction
│   │   ├── error_detection_of_three_codes/
│   │   │   └── plot_column_chart.py         # Comparing different codes (repetition, surface, Steane) under error detection
│   │   └── EE-T_and_2G/
│   │       └── compare_correction.py        # Compare EE-T vs. 2nd-Gen correction
│   └── README.md
│
├── hybrid-error-management/                 # Hybrid filtering + decoding experiments
│   ├── Model/
│   │   ├── model.py                         # Transformer-based NN architecture
│   │   └── model_d_{3,5,7,9,11}.pt         # Pre-trained model weights
│   ├── surface.py                           # Surface code state encoding with parameterized noise
│   ├── estimator_weight.py                  # SW filter + PM decoder
│   ├── estimator_z.py                       # DS filter + PM decoder
│   ├── estimator_NN.py                      # DS filter + NN decoder
│   ├── estimator_swNN.py                    # SW filter + NN decoder
│   ├── near_term_smallcode.py               # Small code (d=3) performance comparison
│   ├── threshold_scan.py                    # Threshold scan across multiple code distances
│   ├── plot_3d_landscape.py                 # 3D landscape visualization
│   └── plot_running_time.py                 # Runtime benchmarking
│
└── requirements.txt                         # Python dependencies
```

---

## Installation

**Python version.** Python 3.10+ is recommended.

```bash
# 1) Create a clean environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n qlink python=3.10 -y
conda activate qlink
pip install -r requirements.txt
```

**Note on PyTorch.** The `hybrid-error-management` experiments require PyTorch. If you need GPU support, install the appropriate CUDA variant following the [PyTorch installation guide](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Part 1: Three Protocols (`three-protocols/`)

This folder compares three quantum link-layer protocols across different codes.

### Circuits (`circuits/`)

* **`surface.py`** -- Surface code circuits, including logical state preparation and stabilizer measurement.
* **`steane.py`** -- Steane [[7,1,3]] code circuits.
* **`DEJMPS.py`** -- Entanglement distillation primitives (DEJMPS protocol).

### Logical error calculators (`cals/`)

* **`cal_logical_error.py`** -- Estimates logical error rates under different protocol and noise settings. Supports PyMatching (MWPM) for the surface code and BP-LSD for the Steane code.

### Experiments (`experiments/`)

* **`error_correction_of_three_protocols/`** -- Reproduces error correction comparison figures across protocols.
* **`error_detection_of_three_codes/`** -- Reproduces error detection comparison figures across codes.
* **`EE-T_and_2G/`** -- Compares Localized Entanglement Encoding (LEE) against second-generation protocols.

**Usage.** Navigate to the relevant experiment folder and run the Python script. Generated figures are saved locally as PDF files.

---

## Part 2: Hybrid Error Management (`hybrid-error-management/`)

This folder implements and evaluates hybrid error management strategies that combine a **filter** (to discard high-risk shots) with a **decoder** (to correct remaining errors).

### Filter / estimator modules

| Module | Method | Description |
|---|---|---|
| `estimator_weight.py` | SW + PM | Accepts shots with normalized syndrome weight below a threshold. Decoder-agnostic. |
| `estimator_z.py` | DS + PM | Computes the exact coset gap via DEM lifting. Accepts shots with large gap. |
| `estimator_NN.py` | DS + NN | Uses a pre-trained transformer model to predict logical error probability. Can serve as both filter and decoder. |
| `estimator_swNN.py` | SW + NN | Applies syndrome-weight filtering first, then decodes accepted shots with the NN. |

### Neural network model (`Model/`)

* **`model.py`** -- Transformer-based architecture with convolutional embedding, multi-head self-attention, and dilated 2D convolutions. Input shape: `(B, 3, d+1, d+1)`.
* **`model_d_{3,5,7,9,11}.pt`** -- Pre-trained weights for code distances 3 through 11.

### Experiment scripts

#### `near_term_smallcode.py` -- Small code performance

Compares all filter + decoder combinations at code distance d = 3 across a grid of local and transmission error rates and target acceptance rates.

```bash
cd hybrid-error-management
python near_term_smallcode.py
```

#### `threshold_scan.py` -- Threshold scan

Scans over multiple code distances and transmission error rates to identify threshold-like crossings for each estimator.

```bash
cd hybrid-error-management
python threshold_scan.py --p-local 0.003 --d-list 3,5,7,9 --p-trans-grid 0.01,0.02,0.03,...,0.20
```

Key CLI arguments:
* `--d-list` -- Code distances to sweep (default: `3,5,7,9`)
* `--p-local` -- Fixed local error rate (required)
* `--p-trans-grid` -- Transmission error rate grid
* `--accept-list` -- Target acceptance rates (default: `0.10,0.20,0.40,0.80`)
* `--nn-device` -- `cuda:0` or `cpu`
* `--shots-cal`, `--shots-eval` -- Calibration and evaluation shot counts

#### `plot_3d_landscape.py` -- 3D visualization

Generates 3D surface/line plots of the (p_trans, acceptance rate, logical error rate) landscape from cached CSV data.

```bash
cd hybrid-error-management
python plot_3d_landscape.py --data_dir <cache_dir> --d 3 --mode surface_lines
```

#### `plot_running_time.py` -- Runtime benchmarking

Benchmarks per-shot filter scoring and decoding time for SW+MWPM, DS+MWPM, and DS+NN across multiple code distances.

```bash
cd hybrid-error-management
python plot_running_time.py
```

---

## Reproducibility checklist

1. Install dependencies from `requirements.txt` in a clean environment.
2. Use the same commit/tag corresponding to the paper.
3. For `three-protocols`: navigate to each experiment subfolder and run its script.
4. For `hybrid-error-management`: run each experiment script as described above. Results are cached in CSV files for incremental computation.

---

## Contact

For questions and reproducibility issues, please open a GitHub issue or contact the authors.
