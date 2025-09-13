# Quantum Link‑Layer Protocol Simulation Code

This repository contains the simulation code accompanying our paper on **quantum link layer protocols**. It provides reference circuit implementations for multiple quantum codes and scripts to reproduce the three main experiments reported in the paper (error detection, error correction, and LEE vs. 2G comparisons).

> **At a glance**
>
> * **Codes implemented:** Surface code, Steane code, Repetition code
> * **Artifacts:** Logical‑error calculators, plotting scripts, experiment drivers

---

## Repository layout

```
├── cals
│   └── cal_logical_error.py         # Logical error estimation utilities
├── circuits
│   ├── DEJMPS.py                    # Entanglement distillation / DEJMPS circuit(s)
│   ├── steane.py                    # Steane [[7,1,3]] code circuits
│   └── surface.py                   # Surface‑code circuits
├── experiments
│   ├── error_correction_of_three_protocols
│   │   └── plot_column_chart.py     # Reproduce EC comparison figure(s)
│   ├── error_detection_of_three_codes
│   │   └── plot_column_chart.py     # Reproduce ED comparison figure(s)
│   └── LEE_and_2G
│       └── compare_correction.py    # Compare LEE vs. 2nd‑Gen correction
└── requirements.txt                 # Python dependencies
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

---

## Core components

### Circuits (`circuits/`)

This folder defines the building blocks of quantum link‑layer protocols under different codes.

* **`surface.py`** — Implements the surface code circuits, including logical state preparation and stabilizer measurement.
* **`steane.py`** — Implements the Steane \[\[7,1,3]] code circuits.
* **`DEJMPS.py`** — Implements entanglement distillation primitives (DEJMPS protocol).

Together, these files define how each code encodes entanglement and performs error detection/correction at the circuit level.

### Logical error calculators (`cals/`)

* **`cal_logical_error.py`** — Provides routines for estimating logical error rates. It connects the abstract code definitions with protocol/noise settings, enabling evaluation of error performance under different physical error models.

### Experiments (`experiments/`)

Each subfolder corresponds to one experiment in the paper. They use the circuits and calculators to generate results.

* **`error_correction_of_three_protocols/`** — Contains scripts for reproducing error correction comparison figures.
* **`error_detection_of_three_codes/`** — Contains scripts for reproducing error detection comparison figures.
* **`LEE_and_2G/`** — Contains scripts comparing Localized Entanglement Encoding (LEE) against second‑generation protocols.

---

## Usage

The experiment scripts are directly runnable. They import the circuits and calculators internally, and produce plots (e.g., column charts) for comparison.

To reproduce results:

1. Navigate to the relevant experiment folder.
2. Run the provided Python script (e.g., `plot_column_chart.py`).
3. Generated figures will be saved locally (PDF files).

---

## Reproducibility checklist

* Install dependencies from `requirements.txt` in a clean environment.
* Use the same commit/tag corresponding to the paper.
* Run each experiment script as described above to regenerate figures.

---

## Contact

For questions and reproducibility issues, please open a GitHub issue or contact the authors.
