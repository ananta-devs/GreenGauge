#!/usr/bin/env python3
"""
GreenGauge / Green AI Tracker - Unified Experiment Runner
=========================================================
Run any experiment sample from the root directory with ease:
  python run_experiments.py --all
  python run_experiments.py --sample 1
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
}


def run_hardware_detection():
    print("\n" + "=" * 70)
    print("  HARDWARE DETECTION (CodeCarbon)")
    print("=" * 70)
    cmd = [sys.executable, "-m", "codecarbon", "detect"]
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_sample(sample_id):
    if sample_id not in SAMPLES:
        print(f"[!] Error: Invalid sample {sample_id}. Choose between 1 and 5.")
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


def main():
    parser = argparse.ArgumentParser(
        description="GreenGauge Experiment Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --detect        # Detect system hardware
  python run_experiments.py --sample 1      # Run sample_01.py
  python run_experiments.py --sample 5      # Run Phase 7 CSV benchmark
  python run_experiments.py --all           # Run all 5 experiments in sequence
        """
    )
    parser.add_argument("--sample", type=int, choices=[1, 2, 3, 4, 5], help="Sample number to run (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all experiment samples sequentially")
    parser.add_argument("--detect", action="store_true", help="Detect hardware used for carbon estimation")

    args = parser.parse_args()

    if args.detect:
        run_hardware_detection()
        return

    if args.all:
        print("\nStarting full Green AI experiment pipeline (Samples 01 to 05)...")
        for s_id in sorted(SAMPLES.keys()):
            success = run_sample(s_id)
            if not success:
                print(f"[!] Sample {s_id} encountered an error. Stopping pipeline.")
                sys.exit(1)
        print("\n[✔] All experiment benchmarks completed successfully!")
        return

    if args.sample:
        success = run_sample(args.sample)
        if not success:
            sys.exit(1)
        return

    # If no argument supplied, show interactive menu
    print("\n" + "=" * 70)
    print("  🌿 GreenGauge — Experiment Runner")
    print("=" * 70)
    print("  [0] Detect Hardware (codecarbon detect)")
    for s_id, (_, desc) in SAMPLES.items():
        print(f"  [{s_id}] Sample 0{s_id} — {desc}")
    print("  [A] Run All Experiments (01 to 05)")
    print("  [Q] Quit")
    print("=" * 70)

    try:
        choice = input("Select an option (0-5, A, Q): ").strip().upper()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice == "0":
        run_hardware_detection()
    elif choice in ["1", "2", "3", "4", "5"]:
        run_sample(int(choice))
    elif choice == "A":
        for s_id in sorted(SAMPLES.keys()):
            run_sample(s_id)
    else:
        print("Done.")


if __name__ == "__main__":
    main()
