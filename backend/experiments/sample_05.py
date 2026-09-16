"""
Phase 7 — Structured Measurement Storage & CSV Collection
==========================================================
Executes controlled model benchmarks, captures full system and CodeCarbon energy metrics,
and persists them into structured CSV files ready for database persistence (Phase 8 SQLite / FastAPI).

Columns saved:
- id, name, dataset, model, accuracy, runtime, energy_kwh, emissions_kg
- cpu_power, gpu_power, ram_power, cpu_utilization, tracking_mode, created_at
"""

import os
import time
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker


def run_benchmark(experiment_id, name, dataset_name, X, y, model_factory, is_scaled=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if is_scaled:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    tracker = EmissionsTracker(
        project_name=f"exp_{experiment_id}_{dataset_name}",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )

    tracker.start()
    t0 = time.perf_counter()

    model = model_factory()
    model.fit(X_train, y_train)

    runtime = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100

    record = {
        "id": experiment_id,
        "name": name,
        "dataset": dataset_name,
        "model": model.__class__.__name__,
        "accuracy": round(acc, 2),
        "runtime_s": round(runtime, 4),
        "energy_kwh": round(em_data.energy_consumed, 8),
        "emissions_kg": round(emissions, 10),
        "cpu_power_w": round(em_data.cpu_power or 0.0, 3),
        "gpu_power_w": round(em_data.gpu_power or 0.0, 3),
        "ram_power_w": round(em_data.ram_power or 0.0, 3),
        "cpu_utilization": round(em_data.cpu_utilization_percent or 0.0, 2),
        "tracking_mode": "process",
        "cpu_model": em_data.cpu_model or "Unknown",
        "created_at": datetime.now().isoformat()
    }
    return record


def main():
    print("=" * 85)
    print("  GREEN AI TRACKER - PHASE 7: BENCHMARK SUITE & CSV PERSISTENCE")
    print("=" * 85)

    cancer = load_breast_cancer()
    iris = load_iris()

    experiments_to_run = [
        # (id, name, dataset_name, X, y, model_factory, is_scaled)
        (1, "Logistic Regression Iris", "iris", iris.data, iris.target, lambda: LogisticRegression(max_iter=500), True),
        (2, "Decision Tree Iris", "iris", iris.data, iris.target, lambda: DecisionTreeClassifier(max_depth=4), False),
        (3, "Random Forest Breast Cancer", "breast_cancer", cancer.data, cancer.target, lambda: RandomForestClassifier(n_estimators=100), False),
        (4, "SVM Breast Cancer", "breast_cancer", cancer.data, cancer.target, lambda: SVC(kernel="rbf"), True),
    ]

    records = []
    for exp_id, name, dname, X, y, model_fn, scaled in experiments_to_run:
        print(f"Executing [{exp_id}/4] {name} on '{dname}'...")
        rec = run_benchmark(exp_id, name, dname, X, y, model_fn, scaled)
        records.append(rec)

    df = pd.DataFrame(records)

    # Save to both local backend/experiments and results/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path_local = os.path.join(base_dir, "benchmark_results.csv")
    csv_path_results = os.path.join(results_dir, "benchmark_results.csv")

    df.to_csv(csv_path_local, index=False)
    df.to_csv(csv_path_results, index=False)

    print("\n" + "=" * 95)
    print(f"Successfully saved {len(df)} experiment records to:")
    print(f" -> {csv_path_local}")
    print(f" -> {csv_path_results}")
    print("=" * 95)
    print("\nBenchmark Summary Table:")
    cols_to_show = ["id", "name", "dataset", "accuracy", "runtime_s", "energy_kwh", "emissions_kg"]
    print(df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
