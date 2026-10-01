#!/usr/bin/env python3
"""
GreenGauge / Green AI Tracker - Unified Experiment Runner
=========================================================
Run any experiment sample from the root directory with ease:
  python run_experiments.py --all
  python run_experiments.py --sample 1
  python run_experiments.py --sample image
  python run_experiments.py --plot
  python run_experiments.py --plot-intervals
  python run_experiments.py --detect
"""

import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "backend", "experiments")

SAMPLES = {
    1: ("sample_01.py", "Phase 2: Baseline Quickstart (Arithmetic Loop)"),
    2: ("sample_02.py", "Phase 4: Three Controlled Workloads (CPU, NumPy, ML)"),
    3: ("sample_03.py", "Phase 5: 4-Model Comparison (Logistic, Tree, Forest, SVM)"),
    4: ("sample_04.py", "Phase 6: Repeated Experiments (5 Runs, Mean ± Std)"),
    5: ("sample_05.py", "Phase 7: Structured CSV Collection & Persistence"),
    6: ("exp_image.py", "Image Modality: Spatial Filtering & Computer Vision"),
    7: ("exp_audio.py", "Audio Modality: FFT Spectrum & Digital Signal Processing"),
    8: ("exp_video.py", "Video Modality: Motion Differencing & Frame Analytics"),
    9: ("exp_csv.py",   "Tabular Modality: Feature Engineering & Tabular Modeling"),
}

ALIAS_MAP = {
    "image": 6,
    "img": 6,
    "audio": 7,
    "sound": 7,
    "video": 8,
    "vid": 8,
    "csv": 9,
    "tabular": 9,
}


def run_hardware_detection():
    print("\n" + "=" * 70)
    print("  HARDWARE DETECTION (CodeCarbon)")
    print("=" * 70)
    cmd = [sys.executable, "-m", "codecarbon", "detect"]
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_sample(sample_id):
    if sample_id not in SAMPLES:
        print(f"[!] Error: Invalid sample '{sample_id}'. Choose between 1 and 9 or [image, audio, video, csv].")
        return False

    script_name, description = SAMPLES[sample_id]
    script_path = os.path.join(EXPERIMENTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"[!] Error: Script {script_path} not found.")
        return False

    print("\n" + "=" * 75)
    print(f"  RUNNING SAMPLE 0{sample_id}: {description}")
    print(f"  Script: {script_name}")
    print("=" * 75)

    res = subprocess.run([sys.executable, script_path], cwd=EXPERIMENTS_DIR)
    return res.returncode == 0


def run_comparison_plot():
    plot_script = os.path.join(EXPERIMENTS_DIR, "plot_experiments_comparison.py")
    print("\n[+] Generating overall experiments comparison chart...")
    res = subprocess.run([sys.executable, plot_script], cwd=EXPERIMENTS_DIR)
    return res.returncode == 0


def run_interval_plots():
    plot_script = os.path.join(EXPERIMENTS_DIR, "plot_multi_run_intervals.py")
    print("\n[+] Generating multi-run interval charts across all samples...")
    res = subprocess.run([sys.executable, plot_script], cwd=EXPERIMENTS_DIR)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="GreenGauge Experiment Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --detect           # Detect system hardware
  python run_experiments.py --sample 1         # Run sample_01.py
  python run_experiments.py --sample image     # Run exp_image.py
  python run_experiments.py --plot             # Generate overall comparison chart
  python run_experiments.py --plot-intervals   # Generate multi-run interval plots
  python run_experiments.py --all              # Run all experiments in sequence
        """
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Sample number (1-9) or modality name (image, audio, video, csv)"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiment samples sequentially")
    parser.add_argument("--plot", action="store_true", help="Generate Matplotlib comparison chart of all experiments")
    parser.add_argument("--plot-intervals", action="store_true", help="Generate Matplotlib multi-run interval plots for all samples")
    parser.add_argument("--detect", action="store_true", help="Detect hardware used for carbon estimation")

    args = parser.parse_args()

    if args.detect:
        run_hardware_detection()
        return

    if args.plot:
        run_comparison_plot()
        return

    if args.plot_intervals:
        run_interval_plots()
        return

    if args.all:
        print("\nStarting full Green AI experiment pipeline (Samples 01 to 09)...")
        for s_id in sorted(SAMPLES.keys()):
            success = run_sample(s_id)
            if not success:
                print(f"[!] Sample {s_id} encountered an error. Stopping pipeline.")
                sys.exit(1)
        print("\n[OK] All experiment benchmarks completed successfully!")
        return

    if args.sample:
        val = args.sample.lower()
        if val in ALIAS_MAP:
            target_id = ALIAS_MAP[val]
        else:
            try:
                target_id = int(val)
            except ValueError:
                print(f"[!] Invalid sample '{args.sample}'. Use 1-9 or [image, audio, video, csv].")
                sys.exit(1)

        success = run_sample(target_id)
        if not success:
            sys.exit(1)
        return

    # If no argument supplied, show interactive menu
    print("\n" + "=" * 70)
    print("  * GreenGauge -- Experiment Runner")
    print("=" * 70)
    print("  [0] Detect Hardware (codecarbon detect)")
    print("  --- Core Baseline Benchmarks ---")
    for s_id in range(1, 6):
        _, desc = SAMPLES[s_id]
        print(f"  [{s_id}] Sample 0{s_id} -- {desc}")
    print("  --- Multi-Modal Media Benchmarks ---")
    for s_id in range(6, 10):
        _, desc = SAMPLES[s_id]
        print(f"  [{s_id}] Sample 0{s_id} -- {desc}")
    print("  --- Visualization & Plotting ---")
    print("  [P] Overall Experiments Comparison Chart (matplotlib)")
    print("  [M] Multi-Run Interval Comparison Plots (matplotlib)")
    print("  --------------------------------")
    print("  [A] Run All Experiments (01 to 09)")
    print("  [Q] Quit")
    print("=" * 70)

    try:
        choice = input("Select an option: ").strip().upper()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice == "0":
        run_hardware_detection()
    elif choice in [str(i) for i in range(1, 10)]:
        run_sample(int(choice))
    elif choice == "P":
        run_comparison_plot()
    elif choice == "M":
        run_interval_plots()
    elif choice == "A":
        for s_id in sorted(SAMPLES.keys()):
            run_sample(s_id)
    else:
        print("Done.")


if __name__ == "__main__":
    main()
