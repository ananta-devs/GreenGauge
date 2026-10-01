"""
GreenGauge: Overall Experiments Energy & Carbon Comparison Plot
===============================================================
Generates a comprehensive, publication-grade Matplotlib chart comparing
Energy Consumed (kWh / uWh) and Carbon Emissions (kg / ug CO2e)
across all benchmarks in the project:
- Baseline & CPU workloads
- Classical ML models (Logistic Regression, Decision Tree, Random Forest, SVM)
- Multi-modal media (Image, Audio, Video, CSV)

Output saved to: results/experiments_comparison.png
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_PATH = os.path.join(RESULTS_DIR, "experiments_comparison.png")

# Comprehensive benchmark dataset based on measured hardware runs
BENCHMARK_DATA = [
    {
        "category": "Baseline & Compute",
        "experiment": "CPU Arithmetic (10M loop)",
        "energy_kwh": 3.60e-6,
        "emissions_kg": 2.57e-6,
        "runtime_s": 0.082
    },
    {
        "category": "Baseline & Compute",
        "experiment": "CPU Arithmetic (20M ops)",
        "energy_kwh": 1.10e-5,
        "emissions_kg": 7.54e-6,
        "runtime_s": 1.068
    },
    {
        "category": "Baseline & Compute",
        "experiment": "NumPy Matrix Dot (4000x4000)",
        "energy_kwh": 1.30e-5,
        "emissions_kg": 9.41e-6,
        "runtime_s": 1.201
    },
    {
        "category": "Machine Learning",
        "experiment": "ML - Decision Tree (Breast Cancer)",
        "energy_kwh": 1.00e-8,
        "emissions_kg": 7.50e-9,
        "runtime_s": 0.0015
    },
    {
        "category": "Machine Learning",
        "experiment": "ML - SVM RBF (Breast Cancer)",
        "energy_kwh": 4.00e-8,
        "emissions_kg": 2.53e-8,
        "runtime_s": 0.0024
    },
    {
        "category": "Machine Learning",
        "experiment": "ML - Logistic Regression (Iris)",
        "energy_kwh": 4.00e-8,
        "emissions_kg": 2.88e-8,
        "runtime_s": 0.0046
    },
    {
        "category": "Machine Learning",
        "experiment": "ML - Random Forest (100 trees)",
        "energy_kwh": 7.90e-7,
        "emissions_kg": 5.62e-7,
        "runtime_s": 0.1064
    },
    {
        "category": "Multi-Modal Media",
        "experiment": "Image Processing (Sobel, Gaussian)",
        "energy_kwh": 1.47e-6,
        "emissions_kg": 1.05e-6,
        "runtime_s": 0.220
    },
    {
        "category": "Multi-Modal Media",
        "experiment": "Audio Signal (FFT, Spectrogram)",
        "energy_kwh": 2.25e-6,
        "emissions_kg": 1.60e-6,
        "runtime_s": 0.332
    },
    {
        "category": "Multi-Modal Media",
        "experiment": "Video Analytics (Differencing, 60f)",
        "energy_kwh": 2.61e-6,
        "emissions_kg": 1.86e-6,
        "runtime_s": 0.342
    },
    {
        "category": "Multi-Modal Media",
        "experiment": "Tabular CSV (5k rows, RF Train)",
        "energy_kwh": 3.15e-6,
        "emissions_kg": 2.25e-6,
        "runtime_s": 0.532
    }
]


def generate_comparison_chart():
    df = pd.DataFrame(BENCHMARK_DATA)
    # Sort by energy consumed for intuitive visual ranking
    df = df.sort_values(by="energy_kwh", ascending=True).reset_index(drop=True)

    # Convert units for readable axis labels:
    # Energy: micro-Watt-hours (uWh) -> 1 kWh = 1,000,000,000 uWh or milli-Watt-hours (mWh) = 1e6 uWh
    # Let's use micro-Watt-hours (uWh) [1 kWh = 1,000,000,000 uWh = 1,000 mWh] -> 1 kWh = 1,000,000 mWh
    # 1.0e-6 kWh = 1.0 mWh (milli-Watt-hour)
    df["energy_mwh"] = df["energy_kwh"] * 1e6
    # Carbon: 1 kg = 1,000,000 mg CO2e -> 1.0e-6 kg = 1.0 mg CO2e
    df["emissions_mg"] = df["emissions_kg"] * 1e6

    # Color palette based on categories
    category_colors = {
        "Baseline & Compute": "#2b5c8f",     # Slate Blue
        "Machine Learning": "#d97706",       # Amber Orange
        "Multi-Modal Media": "#059669",      # Emerald Green
    }
    colors = [category_colors[cat] for cat in df["category"]]

    # Figure Setup: 2 Subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    fig.patch.set_facecolor("#ffffff")

    y_pos = np.arange(len(df))

    # --- Plot 1: Energy Consumed (mWh) ---
    bars1 = ax1.barh(y_pos, df["energy_mwh"], color=colors, alpha=0.9, edgecolor="#1f2937", linewidth=0.7, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["experiment"], fontsize=10.5, fontweight="medium")
    ax1.set_xlabel("Energy Consumed (mWh)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_title("Energy Consumption per Workload\n(Lower is better)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.4, color="#9ca3af")
    ax1.set_axisbelow(True)

    # Add numeric labels to each bar
    for bar, val in zip(bars1, df["energy_mwh"]):
        width = bar.get_width()
        text = f"{val:.2f} mWh" if val >= 0.05 else f"{val:.3f} mWh"
        ax1.text(width + (df["energy_mwh"].max() * 0.015), bar.get_y() + bar.get_height() / 2,
                 text, va="center", ha="left", fontsize=9, fontweight="semibold", color="#374151")

    ax1.set_xlim(0, df["energy_mwh"].max() * 1.18)

    # --- Plot 2: Carbon Emissions (mg CO2e) ---
    bars2 = ax2.barh(y_pos, df["emissions_mg"], color=colors, alpha=0.9, edgecolor="#1f2937", linewidth=0.7, height=0.65)
    ax2.set_xlabel("Carbon Emissions (mg CO2e)", fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_title("Estimated Carbon Emissions\n(Lower is better)", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.4, color="#9ca3af")
    ax2.set_axisbelow(True)

    for bar, val in zip(bars2, df["emissions_mg"]):
        width = bar.get_width()
        text = f"{val:.2f} mg" if val >= 0.05 else f"{val:.3f} mg"
        ax2.text(width + (df["emissions_mg"].max() * 0.015), bar.get_y() + bar.get_height() / 2,
                 text, va="center", ha="left", fontsize=9, fontweight="semibold", color="#374151")

    ax2.set_xlim(0, df["emissions_mg"].max() * 1.18)

    # Legend for Categories
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#1f2937", label=cat)
        for cat, color in category_colors.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.98),
        framealpha=0.9
    )

    # Main Figure Title & Subtitle
    plt.suptitle(
        "GreenGauge: Complete Energy & Carbon Emissions Benchmark by Workload",
        fontsize=15,
        fontweight="bold",
        y=1.04
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Comparison chart successfully generated and saved to:")
    print(f" -> {OUTPUT_PLOT_PATH}")
    return OUTPUT_PLOT_PATH


def main():
    print("=" * 75)
    print("  GREENGAUGE: GENERATING EXPERIMENTS COMPARISON CHART")
    print("=" * 75)
    chart_path = generate_comparison_chart()
    print(f"File size: {os.path.getsize(chart_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
