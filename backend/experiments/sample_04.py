"""
======================================================
Runs multiple trials (e.g., 5 runs per model) to quantify mean and standard deviation
for runtime, energy consumption, and carbon emissions.

This directly reduces measurement variance as recommended in Green AI literature.
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker


def run_single_experiment(model_fn, X_train, y_train, X_test, y_test, run_idx, model_name):
    tracker = EmissionsTracker(
        project_name=f"repeat_{model_name}_{run_idx}",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )
    
    tracker.start()
    t0 = time.perf_counter()

    model = model_fn()
    model.fit(X_train, y_train)

    runtime = time.perf_counter() - t0
    emissions = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100

    return {
        "accuracy": acc,
        "runtime_s": runtime,
        "energy_kwh": energy_kwh,
        "emissions_kg": emissions
    }


def main():
    print("=" * 85)
    print("  GREEN AI TRACKER - PHASE 6: REPEATED EXPERIMENTS (5 RUNS PER MODEL)")
    print("=" * 85)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_configs = [
        ("Logistic Regression", lambda: LogisticRegression(max_iter=1000), True),
        ("Decision Tree", lambda: DecisionTreeClassifier(max_depth=5), False),
        ("Random Forest", lambda: RandomForestClassifier(n_estimators=100), False),
        ("SVM", lambda: SVC(kernel="rbf"), True),
    ]

    NUM_RUNS = 5
    summary_results = []

    for name, model_fn, use_scaled in model_configs:
        print(f"\nEvaluating '{name}' across {NUM_RUNS} runs:")
        xtr = X_train_scaled if use_scaled else X_train
        xte = X_test_scaled if use_scaled else X_test

        run_records = []
        for i in range(1, NUM_RUNS + 1):
            rec = run_single_experiment(model_fn, xtr, y_train, xte, y_test, i, name.lower().replace(" ", "_"))
            run_records.append(rec)
            print(f"  Run {i}: Acc={rec['accuracy']:.2f}%, Time={rec['runtime_s']:.4f}s, Energy={rec['energy_kwh']:.7f} kWh")

        accuracies = [r["accuracy"] for r in run_records]
        runtimes = [r["runtime_s"] for r in run_records]
        energies = [r["energy_kwh"] for r in run_records]
        emissions = [r["emissions_kg"] for r in run_records]

        summary_results.append({
            "model": name,
            "acc_mean": np.mean(accuracies),
            "acc_std": np.std(accuracies),
            "time_mean": np.mean(runtimes),
            "time_std": np.std(runtimes),
            "energy_mean": np.mean(energies),
            "energy_std": np.std(energies),
            "emissions_mean": np.mean(emissions),
            "emissions_std": np.std(emissions),
        })

    print("\n" + "=" * 95)
    print("  AGGREGATED RESULTS (MEAN ± STD)")
    print("=" * 95)
    header = f"{'Model':<20} | {'Accuracy (%)':<16} | {'Runtime (s)':<16} | {'Energy (kWh)':<20} | {'CO2e (kg)':<16}"
    print(header)
    print("-" * 95)
    for s in summary_results:
        acc_str = f"{s['acc_mean']:.1f} ± {s['acc_std']:.2f}"
        time_str = f"{s['time_mean']:.3f} ± {s['time_std']:.3f}"
        energy_str = f"{s['energy_mean']:.6f} ± {s['energy_std']:.6f}"
        emiss_str = f"{s['emissions_mean']:.8f}"
        print(f"{s['model']:<20} | {acc_str:<16} | {time_str:<16} | {energy_str:<20} | {emiss_str:<16}")
    print("=" * 95)


if __name__ == "__main__":
    main()
