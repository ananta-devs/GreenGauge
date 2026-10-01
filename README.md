# 🌿 GreenGauge: Green AI Experiment Tracker

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/emissions-CodeCarbon-green.svg)](https://codecarbon.io/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GreenGauge** is a lightweight, reproducible experiment-tracking and benchmarking framework designed to quantify, analyze, and minimize compute energy consumption and carbon emissions ($CO_2e$) across artificial intelligence, machine learning, and multi-modal data processing pipelines.

Rather than optimizing for raw accuracy alone, GreenGauge enables developers and researchers to evaluate models through a **Green AI lens** — balancing predictive performance and algorithmic throughput against computational and environmental cost.

---

## 📁 Repository Structure

```text
GreenGauge/
│
├── run_experiments.py          # ⚡ Unified experiment runner CLI (run all or individual samples)
├── requirements.txt            # Core dependencies (FastAPI, CodeCarbon, Scikit-Learn, Pillow, imageio)
├── .gitignore                  # Ignored files (venv, pycache, temporary logs)
├── README.md                   # Project documentation & run guide
│
├── backend/
│   ├── app/                    # Core application configuration and utilities
│   ├── experiments/            # Experiment scripts
│   │   │   # --- Baseline & ML Model Benchmarks ---
│   │   ├── sample_01.py        # Phase 2: Baseline CodeCarbon quickstart computation
│   │   ├── sample_02.py        # Phase 4: Three controlled workloads (CPU, NumPy, Random Forest)
│   │   ├── sample_03.py        # Phase 5: 4-model ML comparison on same dataset
│   │   ├── sample_04.py        # Phase 6: Repeated runs (5 trials) for statistical robustness
│   │   ├── sample_05.py        # Phase 7: Structured CSV persistence & database schema prep
│   │   │
│   │   │   # --- Multi-Modal Media Benchmarks ---
│   │   ├── exp_image.py        # 🖼️ Image: Spatial filtering, Sobel edge convolution, Histogram EQ
│   │   ├── exp_audio.py        # 🎵 Audio: FFT spectrum, STFT spectrogram, Butterworth IIR filter
│   │   ├── exp_video.py        # 🎬 Video: Temporal frame differencing, motion energy, keyframes
│   │   ├── exp_csv.py          # 📊 Tabular: Cleaning, feature engineering, GroupBy, RandomForest
│   │   │
│   │   ├── data/               # Small, self-contained sample media assets (< 500 KB each)
│   │   │   ├── sample_image.png
│   │   │   ├── sample_audio.wav
│   │   │   ├── sample_video.gif
│   │   │   └── sample_data.csv
│   │   │
│   │   ├── benchmark_results.csv # Generated output data
│   │   └── emissions.csv       # CodeCarbon emissions log
│   ├── models/                 # Database models and data schemas
│   ├── services/               # Carbon tracking and estimation logic
│   └── main.py                 # FastAPI entrypoint (Phase 8)
│
└── results/                    # Output metrics, comparison tables, and benchmark logs
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

You can run the experiment suite using either the **unified CLI runner** (from the project root) or by running individual sample scripts directly inside `backend/experiments/`.

### Method A: Unified Experiment Runner (Recommended)

Run all experiments sequentially:
```bash
python run_experiments.py --all
```

Run a specific experiment:
```bash
# Baseline & ML Benchmarks
python run_experiments.py --sample 1      # Phase 2: Simple arithmetic loop
python run_experiments.py --sample 2      # Phase 4: CPU vs NumPy vs ML
python run_experiments.py --sample 3      # Phase 5: 4-Model ML comparison
python run_experiments.py --sample 4      # Phase 6: 5 repeated runs with Mean ± Std
python run_experiments.py --sample 5      # Phase 7: Save benchmark to CSV

# Multi-Modal Media Benchmarks
python run_experiments.py --sample image  # Image: Gaussian blur, Sobel edge filter
python run_experiments.py --sample audio  # Audio: FFT, STFT spectrogram, bandpass filter
python run_experiments.py --sample video  # Video: Motion differencing & keyframes
python run_experiments.py --sample csv    # Tabular: Feature engineering & modeling
```

Interactive menu mode:
```bash
python run_experiments.py
```

---

### Method B: Running Scripts Directly (`cd backend/experiments`)

Each experiment is completely self-contained and automatically generates its small test dataset if missing:

```bash
cd backend/experiments
```

#### 1. Baseline Quickstart (`sample_01.py`)
Runs $10\text{M}$ integer additions inside `EmissionsTracker` to verify energy tracking works.
```bash
python sample_01.py
```

#### 2. Three Controlled Workloads (`sample_02.py`)
Evaluates three distinct computational loads: Pure CPU integer math, NumPy $4000 \times 4000$ matrix multiplication, and Random Forest training.
```bash
python sample_02.py
```

#### 3. Model Comparison on Identical Dataset (`sample_03.py`)
Compares Logistic Regression, Decision Tree, Random Forest, and SVM on Breast Cancer Wisconsin.
```bash
python sample_03.py
```

#### 4. Repeated Experiments (`sample_04.py`)
Performs 5 independent trials per model to calculate **Mean $\pm$ Standard Deviation** for Accuracy, Time, Energy, and $CO_2e$.
```bash
python sample_04.py
```

#### 5. Structured CSV Storage (`sample_05.py`)
Exports structured tabular data to `results/benchmark_results.csv` and `backend/experiments/benchmark_results.csv`.
```bash
python sample_05.py
```

#### 6. 🖼️ Image Processing Experiment (`exp_image.py`)
Loads `data/sample_image.png` (~6 KB, 512x512) and performs:
- Grayscale conversion & luminance weighting
- 2D Gaussian blur convolution
- Sobel horizontal & vertical gradient edge detection
- Cumulative histogram equalization
- Multi-scale resizing and rotation transformations
```bash
python exp_image.py
```

#### 7. 🎵 Audio Signal Processing Experiment (`exp_audio.py`)
Loads `data/sample_audio.wav` (~258 KB, 3s @ 44.1 kHz) and performs:
- Waveform amplitude normalization
- Fast Fourier Transform (FFT) frequency spectrum analysis
- Short-Time Fourier Transform (STFT) spectrogram calculation
- Acoustic feature extraction: RMS energy, Zero-Crossing Rate, Spectral Centroid, Spectral Rolloff
- 6th-order Butterworth digital bandpass filtering (300 Hz - 3400 Hz)
```bash
python exp_audio.py
```

#### 8. 🎬 Video Motion Analytics Experiment (`exp_video.py`)
Loads `data/sample_video.gif` (~67 KB, 60 frames @ 128x128) and performs:
- Frame sequence decoding into 4D tensor ($T \times H \times W \times C$)
- Temporal frame differencing ($|F_t - F_{t-1}|$) to compute motion energy
- 3-frame moving-average temporal smoothing (denoising)
- Spatial Sobel edge filtering per frame
- Automated keyframe extraction based on peak motion thresholds
```bash
python exp_video.py
```

#### 9. 📊 Tabular / CSV Processing Experiment (`exp_csv.py`)
Loads `data/sample_data.csv` (~413 KB, 5,000 rows x 10 columns) and performs:
- Missing value median/mode imputation
- Interquartile Range (IQR) outlier capping
- Advanced feature engineering (rolling windows, log transforms, interaction terms)
- Multi-column GroupBy aggregations
- Supervised classification using `RandomForestClassifier` with test accuracy evaluation
```bash
python exp_csv.py
```

---

## 📊 Benchmark Results (Real Hardware Measurements)

*Hardware: 12th Gen Intel Core i5-1235U (10 cores / 12 threads), 16 GB RAM, Windows 11, CodeCarbon process mode.*

### Multi-Modal Workload Comparison

| Modality | Script | File Size | Execution Time | Energy Consumed | Estimated $CO_2e$ | Throughput / Metric |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image (.png)** | `exp_image.py` | $6.0\text{ KB}$ | $0.220\text{ s}$ | $1.47 \times 10^{-6}\text{ kWh}$ | $1.05 \times 10^{-6}\text{ kg}$ | $9,535\text{ kPixels/s}$ |
| **Audio (.wav)** | `exp_audio.py` | $258.4\text{ KB}$ | $0.332\text{ s}$ | $2.25 \times 10^{-6}\text{ kWh}$ | $1.60 \times 10^{-6}\text{ kg}$ | $5,980\text{ kSamples/s}$ |
| **Video (.gif)** | `exp_video.py` | $66.8\text{ KB}$ | $0.342\text{ s}$ | $2.61 \times 10^{-6}\text{ kWh}$ | $1.86 \times 10^{-6}\text{ kg}$ | $525.5\text{ FPS}$ |
| **Tabular (.csv)**| `exp_csv.py` | $413.6\text{ KB}$ | $0.532\text{ s}$ | $3.15 \times 10^{-6}\text{ kWh}$ | $2.25 \times 10^{-6}\text{ kg}$ | $9,391\text{ rows/s}$ ($100\%$ Acc) |

### 4-Model ML Comparison (Phase 5)

| Model | Accuracy (%) | Training Runtime | Energy Consumed (kWh) | Estimated $CO_2e$ (kg) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **$98.25\%$** | $0.0126\text{ s}$ | $< 10^{-7}\text{ kWh}$ | $2.30 \times 10^{-8}\text{ kg}$ |
| **Decision Tree** | $92.11\%$ | $0.0071\text{ s}$ | $< 10^{-7}\text{ kWh}$ | $3.50 \times 10^{-8}\text{ kg}$ |
| **Random Forest (100 trees)**| $95.61\%$ | $0.2481\text{ s}$ | $3.10 \times 10^{-6}\text{ kWh}$ | $2.22 \times 10^{-6}\text{ kg}$ |
| **SVM (RBF Kernel)** | **$98.25\%$** | **$0.0043\text{ s}$** | $< 10^{-7}\text{ kWh}$ | $3.40 \times 10^{-8}\text{ kg}$ |

> 💡 **Green AI Takeaway**: Different data modalities have distinct energy signatures. While 2D spatial image convolutions are highly cache-friendly, multi-frame video analytics and multi-stage tabular feature engineering demand sustained CPU utilization, leading to proportional increases in energy consumed per unit time.

---

## 📈 Visualizing Energy & Carbon Emissions (Matplotlib)

GreenGauge provides automated Matplotlib data visualization modules to generate high-resolution comparison charts and multi-run interval analytics:

### 1. Overall Experiments Comparison Chart
Generates a publication-grade side-by-side comparison of **Energy Consumed (mWh)** and **Carbon Emissions (mg $CO_2e$)** across all 11 project workloads (Baseline compute, classical ML models, and multi-modal media):
```bash
python run_experiments.py --plot
# Or directly:
python backend/experiments/plot_experiments_comparison.py
```
- **Saved Output**: `results/experiments_comparison.png`

### 2. Multi-Run Interval Comparison Plots
Executes multiple trials per sample separated by a cooldown interval to evaluate measurement repeatability, variance, and temporal stability:
```bash
python run_experiments.py --plot-intervals
# Or directly:
python backend/experiments/plot_multi_run_intervals.py
```
- **Master Multi-Panel Plot**: `results/multi_run_intervals_comparison.png` (3x2 grid with separate dual-axis subplots for each sample)
- **Individual Sample Charts**: `results/intervals/` (`interval_image.png`, `interval_audio.png`, `interval_video.png`, `interval_csv.png`, `interval_random_forest.png`, `interval_svm.png`)
- **Raw Multi-Run Dataset**: `results/multi_run_interval_data.csv`

---

## 🛠️ Troubleshooting & FAQs

### 1. `ModuleNotFoundError: No module named 'codecarbon'` or `'PIL'`
**Cause**: The virtual environment is either not activated or dependencies are not installed.  
**Fix**:
```bash
# Activate venv:
venv\Scripts\activate          # On Windows
source venv/bin/activate       # On macOS/Linux

# Install/update dependencies:
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
GET  /experiments/compare   # Compare multiple models & modalities side-by-side
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
