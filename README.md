# 🌿 GreenGauge: Green AI Experiment Tracker

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/emissions-CodeCarbon-green.svg)](https://codecarbon.io/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GreenGauge** is a lightweight, reproducible experiment-tracking and benchmarking framework designed to quantify, analyze, and minimize compute energy consumption and carbon emissions ($CO_2e$) across artificial intelligence and machine learning pipelines.

Rather than optimizing for raw accuracy alone, GreenGauge enables developers and researchers to evaluate models through a **Green AI lens** — balancing predictive performance against computational and environmental cost.

---

## 📁 Repository Structure

```text
GreenGauge/
│
├── run_experiments.py      # ⚡ Unified experiment runner CLI (run all or individual samples)
├── requirements.txt        # Core project dependencies (FastAPI, CodeCarbon, Scikit-Learn, etc.)
├── .gitignore              # Ignored files (venv, pycache, temporary logs)
├── README.md               # Project documentation & run guide
│
├── backend/
│   ├── app/                # Core application configuration and utilities
│   ├── experiments/        # Experiment scripts (Phases 2 through 7)
│   │   ├── sample_01.py    # Phase 2: Baseline CodeCarbon quickstart computation
│   │   ├── sample_02.py    # Phase 4: Three controlled workloads (CPU, NumPy, Random Forest)
│   │   ├── sample_03.py    # Phase 5: 4-model ML comparison on same dataset
│   │   ├── sample_04.py    # Phase 6: Repeated runs (5 trials) for statistical robustness
│   │   ├── sample_05.py    # Phase 7: Structured CSV persistence & database schema prep
│   │   ├── benchmark_results.csv # Generated output data
│   │   └── emissions.csv   # CodeCarbon emissions log
│   ├── models/             # Database models and data schemas
│   ├── services/           # Carbon tracking and estimation logic
│   └── main.py             # FastAPI entrypoint (Phase 8)
│
└── results/                # Output metrics, comparison tables, and benchmark logs
    └── benchmark_results.csv
```

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to set up and run the experiments on any system (Windows, macOS, or Linux).

### Step 1: Clone the Repository

```bash
git clone https://github.com/ananta-devs/GreenGauge.git
cd GreenGauge
```

---

### Step 2: Create and Activate a Virtual Environment

It is strongly recommended to use a clean virtual environment to prevent package version conflicts.

#### On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*(If you see an execution policy error in PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run activate).*

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Verify Hardware Detection

CodeCarbon automatically inspects your hardware (CPU, GPU, RAM) to apply the correct energy estimation model.

Run:
```bash
python run_experiments.py --detect
```
*(Or directly: `codecarbon detect`)*

**Sample Output:**
```text
[codecarbon INFO] >>> Tracker's metadata:
  Platform system: Windows-11
  Python version: 3.14.x (or 3.10+)
  CodeCarbon version: 3.3.x
  Available RAM : 15.68 GB
  CPU model: 12th Gen Intel(R) Core(TM) i5-1235U
  CPU count: 12 thread(s) in 1 physical CPU(s)
  GPU count: 0 (or GPU model if NVIDIA GPU is present)
```

---

## 🧪 Running the Experiments

You can run the experiment suite using either the **unified CLI runner** (recommended) or by running individual sample scripts.

### Method A: Unified Experiment Runner (Recommended)

Run all experiments sequentially:
```bash
python run_experiments.py --all
```

Run a specific experiment:
```bash
python run_experiments.py --sample 1    # Phase 2: Simple arithmetic loop
python run_experiments.py --sample 2    # Phase 4: CPU vs NumPy vs ML
python run_experiments.py --sample 3    # Phase 5: 4-Model ML comparison
python run_experiments.py --sample 4    # Phase 6: 5 repeated runs with Mean ± Std
python run_experiments.py --sample 5    # Phase 7: Save benchmark to CSV
```

Interactive menu mode:
```bash
python run_experiments.py
```

---

### Method B: Running Scripts Directly

You can also run each script directly inside `backend/experiments/`:

#### 1. Phase 2 — Baseline Quickstart (`sample_01.py`)
Runs $10\text{M}$ integer additions inside `EmissionsTracker` to verify energy tracking works.
```bash
cd backend/experiments
python sample_01.py
```

#### 2. Phase 4 — Three Controlled Workloads (`sample_02.py`)
Evaluates three distinct computational loads:
- **Workload A (Pure CPU)**: Arithmetic integer squaring loop ($20\text{M}$ operations).
- **Workload B (Data-processing)**: NumPy $4000 \times 4000$ matrix multiplication.
- **Workload C (Machine Learning)**: Random Forest Classifier trained on Breast Cancer data.
```bash
python sample_02.py
```

#### 3. Phase 5 — Model Comparison on Identical Dataset (`sample_03.py`)
Compares 4 classical ML models under identical conditions on the Breast Cancer Wisconsin dataset:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest** (100 estimators)
4. **Support Vector Machine** (SVM with RBF kernel)
```bash
python sample_03.py
```

#### 4. Phase 6 — Repeated Experiments & Statistical Aggregation (`sample_04.py`)
Performs 5 independent trials per model to calculate **Mean $\pm$ Standard Deviation** for Accuracy, Training Time, Energy Consumed, and $CO_2e$.
```bash
python sample_04.py
```

#### 5. Phase 7 — Structured CSV Storage (`sample_05.py`)
Captures all experiment metrics and exports structured tabular data to `results/benchmark_results.csv` and `backend/experiments/benchmark_results.csv`.
```bash
python sample_05.py
```

---

## 📊 Sample Results (Real Hardware Benchmark)

*Hardware: 12th Gen Intel Core i5-1235U, 16 GB RAM, Windows 11, CodeCarbon process mode.*

### 4-Model Comparison (Phase 5)

| Model | Accuracy (%) | Training Runtime | Energy Consumed (kWh) | Estimated $CO_2e$ (kg) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **$98.25\%$** | $0.0126\text{ s}$ | $< 10^{-7}\text{ kWh}$ | $2.30 \times 10^{-8}\text{ kg}$ |
| **Decision Tree** | $92.11\%$ | $0.0071\text{ s}$ | $< 10^{-7}\text{ kWh}$ | $3.50 \times 10^{-8}\text{ kg}$ |
| **Random Forest (100 trees)**| $95.61\%$ | $0.2481\text{ s}$ | $3.10 \times 10^{-6}\text{ kWh}$ | $2.22 \times 10^{-6}\text{ kg}$ |
| **SVM (RBF Kernel)** | **$98.25\%$** | **$0.0043\text{ s}$** | $< 10^{-7}\text{ kWh}$ | $3.40 \times 10^{-8}\text{ kg}$ |

> 💡 **Green AI Takeaway**: Logistic Regression and SVM both achieved the highest accuracy ($98.25\%$) while training in a fraction of a second and consuming negligible energy. Random Forest consumed over **$100\times$ more energy and emissions** for a lower accuracy ($95.61\%$).

---

## 🛠️ Troubleshooting & FAQs

### 1. `ModuleNotFoundError: No module named 'codecarbon'` or `'sklearn'`
**Cause**: The virtual environment is either not activated or dependencies are not installed.  
**Fix**:
```bash
# Verify which python is running:
where python    # On Windows
which python    # On macOS/Linux

# Ensure venv is activated and reinstall:
pip install -r requirements.txt
```

### 2. `PermissionError: [Errno 13] Permission denied: '...emissions.csv'`
**Cause**: `emissions.csv` is currently opened in Microsoft Excel or another program that locks files on Windows.  
**Fix**: Close the CSV file in Excel or other viewers before running the script.

### 3. `Multiple instances of codecarbon are allowed to run at the same time`
**Context**: This is a harmless CodeCarbon informational warning stating that parallel trackers can run simultaneously. The scripts will proceed normally.

### 4. `No GPU found`
**Context**: If your system does not have an NVIDIA GPU (or CUDA is not configured), CodeCarbon defaults to CPU tracking via the Windows Energy Meter Interface (RAPL) or TDP power modeling. Experiments will run on the CPU as intended.

---

## 🗺️ Next Steps: Phase 8 Backend Roadmap

The next phase transitions these experiment scripts into a **FastAPI backend** connected to a SQLite database:

```text
POST /experiments           # Create experiment configuration
POST /experiments/{id}/run  # Trigger model run & emissions tracker
GET  /experiments           # List past experiments
GET  /experiments/{id}      # View granular energy and metric breakdown
GET  /experiments/compare   # Compare multiple models side-by-side
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
