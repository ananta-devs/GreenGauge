"""
=======================================
Controlled comparison of three distinct computational workloads:
- Experiment A: Pure CPU loop (arithmetic squaring)
- Experiment B: Data-processing workload (NumPy 4000x4000 matrix multiplication)
- Experiment C: Machine Learning workload (RandomForest on Breast Cancer dataset)

Measures: Runtime, Energy (kWh), CO2e (kg), CPU Power (W).
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker


def run_workload_a():
    """Experiment A: Pure CPU Workload"""
    print("\n[1/3] Running Experiment A: Pure CPU Workload...")
    tracker = EmissionsTracker(
        project_name="workload_a_cpu",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )
    tracker.start()
    t0 = time.perf_counter()

    # 20 million operations
    total = 0
    for i in range(20_000_000):
        total += (i % 100) * (i % 100)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    return {
        "workload": "CPU Arithmetic (20M ops)",
        "duration_s": duration,
        "energy_kwh": energy_kwh,
        "emissions_kg": emissions,
        "metric": f"Checksum={total}"
    }


def run_workload_b():
    """Experiment B: Data-Processing Workload (NumPy)"""
    print("[2/3] Running Experiment B: NumPy Matrix Multiplication...")
    tracker = EmissionsTracker(
        project_name="workload_b_numpy",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )
    tracker.start()
    t0 = time.perf_counter()

    # Matrix operations: 4000 x 4000 float64 dot product
    matrix_a = np.random.random((4000, 4000))
    matrix_b = np.random.random((4000, 4000))
    result = np.dot(matrix_a, matrix_b)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    return {
        "workload": "NumPy Matrix Dot (4000x4000)",
        "duration_s": duration,
        "energy_kwh": energy_kwh,
        "emissions_kg": emissions,
        "metric": f"Matrix shape={result.shape}"
    }


def run_workload_c():
    """Experiment C: Machine Learning Workload (Random Forest)"""
    print("[3/3] Running Experiment C: Machine Learning (Random Forest)...")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    tracker = EmissionsTracker(
        project_name="workload_c_ml",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )
    tracker.start()
    t0 = time.perf_counter()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    return {
        "workload": "ML RandomForest (Breast Cancer)",
        "duration_s": duration,
        "energy_kwh": energy_kwh,
        "emissions_kg": emissions,
        "metric": f"Accuracy={acc * 100:.2f}%"
    }


def main():
    print("=" * 75)
    print("  GREEN AI TRACKER - PHASE 4: THREE WORKLOADS BENCHMARK")
    print("=" * 75)

    results = []
    results.append(run_workload_a())
    results.append(run_workload_b())
    results.append(run_workload_c())

    print("\n" + "=" * 75)
    print(f"{'Workload':<32} | {'Duration (s)':<12} | {'Energy (kWh)':<12} | {'CO2e (kg)':<12}")
    print("-" * 75)
    for r in results:
        print(f"{r['workload']:<32} | {r['duration_s']:<12.3f} | {r['energy_kwh']:<12.6f} | {r['emissions_kg']:<12.8f}")
    print("=" * 75)
    print("Summary:")
    for r in results:
        print(f" - {r['workload']}: {r['metric']}")


if __name__ == "__main__":
    main()
