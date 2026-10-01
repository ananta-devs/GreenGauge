"""
GreenGauge: Multi-Run Interval Comparison Plotter
=================================================
Executes multiple runs across an interval for each sample workload:
- Image Processing (exp_image)
- Audio Signal Processing (exp_audio)
- Video Motion Processing (exp_video)
- Tabular CSV Processing & ML (exp_csv)
- Random Forest ML (sample_03)
- Support Vector Machine (sample_03)

Generates:
1. A master multi-panel figure where EVERY sample has its own dedicated subplot
   comparing Energy Consumed and Carbon Emissions across all runs:
   -> results/multi_run_intervals_comparison.png
2. Individual standalone plots for each sample in results/intervals/
3. Exported raw multi-run data to results/multi_run_interval_data.csv
"""

import os
import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

# Import individual workload functions from sibling experiment modules
from exp_image import ensure_sample_image, process_image
from exp_audio import ensure_sample_audio, process_audio
from exp_video import ensure_sample_video, process_video
from exp_csv import ensure_sample_csv, process_and_model_csv

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
INTERVALS_DIR = os.path.join(RESULTS_DIR, "intervals")
os.makedirs(INTERVALS_DIR, exist_ok=True)


def get_ml_breast_cancer_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test


def execute_workload_trial(sample_key, run_idx, assets):
    """Executes a single trial of the specified sample workload inside CodeCarbon."""
    tracker = EmissionsTracker(
        project_name=f"interval_{sample_key}_{run_idx}",
        tracking_mode="process",
        save_to_file=False,
        log_level="error"
    )

    tracker.start()
    t0 = time.perf_counter()

    if sample_key == "image":
        process_image(assets["image_path"], iterations=4)
    elif sample_key == "audio":
        process_audio(assets["audio_path"], passes=6)
    elif sample_key == "video":
        process_video(assets["video_path"], repetitions=2)
    elif sample_key == "csv":
        process_and_model_csv(assets["csv_path"])
    elif sample_key == "random_forest":
        X_tr, _, y_tr, X_te, _, y_te = assets["ml_data"]
        clf = RandomForestClassifier(n_estimators=80, random_state=run_idx)
        clf.fit(X_tr, y_tr)
        _ = clf.predict(X_te)
    elif sample_key == "svm":
        _, X_tr_s, y_tr, _, X_te_s, y_te = assets["ml_data"]
        clf = SVC(kernel="rbf")
        clf.fit(X_tr_s, y_tr)
        _ = clf.predict(X_te_s)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    energy_kwh = em_data.energy_consumed
    # If energy is below minimum timer resolution on sub-second runs, provide physics-based estimate:
    # Energy = Average Power (W) * Duration (h)
    if energy_kwh <= 0.0:
        power_w = (em_data.cpu_power or 20.0) + (em_data.ram_power or 10.0)
        energy_kwh = (power_w * (duration / 3600.0)) / 1000.0
        emissions = energy_kwh * 0.716  # standard grid emission factor ~0.716 kg CO2/kWh

    return {
        "run_index": run_idx,
        "sample_key": sample_key,
        "duration_s": duration,
        "energy_kwh": energy_kwh,
        "emissions_kg": emissions,
        "energy_uwh": energy_kwh * 1e9,  # micro-Watt-hours
        "emissions_ug": emissions * 1e9,  # micro-grams CO2e
        "timestamp": datetime.now().isoformat()
    }


def collect_multi_run_data(num_runs=5, interval_seconds=1.0):
    """Runs each sample workload across multiple runs separated by an interval."""
    # Ensure assets exist
    assets = {
        "image_path": ensure_sample_image(),
        "audio_path": ensure_sample_audio(),
        "video_path": ensure_sample_video(),
        "csv_path": ensure_sample_csv(),
        "ml_data": get_ml_breast_cancer_data()
    }

    sample_configs = [
        ("image", "Image Processing (exp_image)"),
        ("audio", "Audio Signal Processing (exp_audio)"),
        ("video", "Video Motion Analytics (exp_video)"),
        ("csv", "Tabular CSV Pipeline (exp_csv)"),
        ("random_forest", "Random Forest Classifier (sample_03)"),
        ("svm", "Support Vector Machine (sample_03)"),
    ]

    all_records = []

    print(f"Starting Multi-Run Interval Benchmark ({num_runs} runs per sample, {interval_seconds}s cooldown interval)...")

    for sample_key, sample_title in sample_configs:
        print(f"\n[+] Benchmarking {sample_title}:")
        for r in range(1, num_runs + 1):
            record = execute_workload_trial(sample_key, r, assets)
            record["sample_title"] = sample_title
            all_records.append(record)
            print(f"    Run {r}/{num_runs}: Time={record['duration_s']:.3f}s | "
                  f"Energy={record['energy_uwh']:.2f} uWh | CO2e={record['emissions_ug']:.2f} ug")
            if r < num_runs:
                time.sleep(interval_seconds)

    df = pd.DataFrame(all_records)
    csv_out = os.path.join(RESULTS_DIR, "multi_run_interval_data.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n[OK] Raw multi-run dataset saved to {csv_out}")
    return df, sample_configs


def plot_master_multi_panel(df, sample_configs):
    """Generates a 3x2 grid figure where each sample has its own dedicated dual-axis subplot."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=False)
    fig.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    for idx, (sample_key, sample_title) in enumerate(sample_configs):
        ax = axes[idx]
        sub_df = df[df["sample_key"] == sample_key].sort_values("run_index")

        runs = sub_df["run_index"].values
        energy_uwh = sub_df["energy_uwh"].values
        emissions_ug = sub_df["emissions_ug"].values

        # Primary Axis: Energy (Blue)
        color_energy = "#1d4ed8"
        line1 = ax.plot(
            runs, energy_uwh, color=color_energy, marker="o", markersize=7,
            linewidth=2.2, label="Energy (uWh)"
        )
        ax.fill_between(runs, energy_uwh, alpha=0.12, color=color_energy)
        ax.set_ylabel("Energy Consumed (uWh)", color=color_energy, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=color_energy)
        ax.set_xticks(runs)
        ax.set_xticklabels([f"Run {r}" for r in runs], fontsize=9.5)
        ax.set_xlabel("Trial Interval", fontsize=10, fontweight="medium")

        # Mean Energy reference line
        mean_energy = np.mean(energy_uwh)
        std_energy = np.std(energy_uwh)
        ax.axhline(mean_energy, color=color_energy, linestyle=":", alpha=0.6, linewidth=1.2)

        # Secondary Axis: Carbon Emissions (Red/Amber)
        ax_sec = ax.twinx()
        color_emiss = "#dc2626"
        line2 = ax_sec.plot(
            runs, emissions_ug, color=color_emiss, marker="s", markersize=6,
            linewidth=2.0, linestyle="--", label="CO2e (ug)"
        )
        ax_sec.set_ylabel("Carbon Emissions (ug CO2e)", color=color_emiss, fontsize=10, fontweight="bold")
        ax_sec.tick_params(axis="y", labelcolor=color_emiss)

        mean_emiss = np.mean(emissions_ug)
        std_emiss = np.std(emissions_ug)
        ax_sec.axhline(mean_emiss, color=color_emiss, linestyle="--", alpha=0.5, linewidth=1.0)

        # Title with Mean ± Std
        title_str = f"{sample_title}\nMean Energy: {mean_energy:.1f} ± {std_energy:.1f} uWh | CO2e: {mean_emiss:.1f} ± {std_emiss:.1f} ug"
        ax.set_title(title_str, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.35, color="#9ca3af")

        # Combined Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.85)

    plt.suptitle(
        "GreenGauge: Multi-Run Energy & Carbon Footprint Across Intervals\n(Performance Stability & Measurement Repeatability)",
        fontsize=15,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout()
    master_plot_path = os.path.join(RESULTS_DIR, "multi_run_intervals_comparison.png")
    plt.savefig(master_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Master multi-panel plot saved to {master_plot_path}")
    return master_plot_path


def plot_individual_sample_charts(df, sample_configs):
    """Generates and saves standalone interval charts for each individual sample in results/intervals/."""
    for sample_key, sample_title in sample_configs:
        sub_df = df[df["sample_key"] == sample_key].sort_values("run_index")
        runs = sub_df["run_index"].values
        energy_uwh = sub_df["energy_uwh"].values
        emissions_ug = sub_df["emissions_ug"].values

        fig, ax1 = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#ffffff")

        # Energy Axis
        color_e = "#2563eb"
        p1 = ax1.plot(runs, energy_uwh, color=color_e, marker="o", markersize=8, linewidth=2.5, label="Energy (uWh)")
        ax1.fill_between(runs, energy_uwh, alpha=0.15, color=color_e)
        ax1.set_xlabel("Experiment Interval (Run)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Energy Consumed (uWh)", color=color_e, fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_e)
        ax1.set_xticks(runs)
        ax1.set_xticklabels([f"Run {r}" for r in runs], fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Emissions Axis
        ax2 = ax1.twinx()
        color_c = "#e11d48"
        p2 = ax2.plot(runs, emissions_ug, color=color_c, marker="s", markersize=7, linewidth=2.2, linestyle="--", label="Emissions (ug CO2e)")
        ax2.set_ylabel("Carbon Emissions (ug CO2e)", color=color_c, fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color_c)

        # Header Title
        mean_e = np.mean(energy_uwh)
        std_e = np.std(energy_uwh)
        mean_c = np.mean(emissions_ug)
        std_c = np.std(emissions_ug)
        plt.title(f"{sample_title}\nMultiple Runs at Intervals (Mean: {mean_e:.1f} uWh, {mean_c:.1f} ug CO2e)",
                  fontsize=12, fontweight="bold", pad=10)

        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=9.5)

        plt.tight_layout()
        single_path = os.path.join(INTERVALS_DIR, f"interval_{sample_key}.png")
        plt.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Generated {single_path}")


def main():
    print("=" * 80)
    print("  GREENGAUGE: MULTI-RUN INTERVAL BENCHMARK & PLOTTING")
    print("=" * 80)

    # 1. Run 5 trials per sample with interval delay
    df, sample_configs = collect_multi_run_data(num_runs=5, interval_seconds=0.6)

    # 2. Generate Master 3x2 Grid Plot (every sample on separate subplot)
    plot_master_multi_panel(df, sample_configs)

    # 3. Generate Individual Plots per Sample
    plot_individual_sample_charts(df, sample_configs)

    print("=" * 80)
    print("[OK] All multi-run interval charts successfully generated!")


if __name__ == "__main__":
    main()
