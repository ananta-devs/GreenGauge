"""
Phase 5 — First Real ML Model Comparison
=========================================
Controlled comparison of 4 classical machine learning models on the Breast Cancer dataset:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

Metrics tracked per model:
- Test Accuracy (%)
- Training Time (s)
- Energy Consumed (kWh)
- Estimated CO2e (kg)
- Average CPU & RAM Power (W)
"""

import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    tracker = EmissionsTracker(
        project_name=f"compare_{name.lower().replace(' ', '_')}",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )
    
    tracker.start()
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start_time
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return {
        "model": name,
        "accuracy": accuracy * 100,
        "runtime_s": train_time,
        "energy_kwh": em_data.energy_consumed,
        "emissions_kg": emissions,
        "cpu_power_w": em_data.cpu_power or 0.0,
        "ram_power_w": em_data.ram_power or 0.0
    }


def main():
    print("=" * 80)
    print("  GREEN AI TRACKER - PHASE 5: MODEL COMPARISON (BREAST CANCER)")
    print("=" * 80)

    # 1. Load and prepare dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # Standardize features for gradient/margin-based models (LogisticRegression & SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42), False),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), False),
        ("SVM", SVC(kernel="rbf", random_state=42), True),
    ]

    results = []
    for name, model, use_scaled in models:
        print(f"--> Training and tracking {name}...")
        xtr = X_train_scaled if use_scaled else X_train
        xte = X_test_scaled if use_scaled else X_test
        res = evaluate_model(name, model, xtr, y_train, xte, y_test)
        results.append(res)

    print("\n" + "=" * 80)
    print(f"{'Model':<22} | {'Accuracy (%)':<12} | {'Runtime (s)':<12} | {'Energy (kWh)':<14} | {'CO2e (kg)':<14}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<22} | {r['accuracy']:<12.2f} | {r['runtime_s']:<12.4f} | {r['energy_kwh']:<14.7f} | {r['emissions_kg']:<14.9f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
