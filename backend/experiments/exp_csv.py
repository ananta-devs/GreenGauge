"""
GreenGauge Experiment: Tabular / CSV Processing & Modeling Benchmark
====================================================================
File Modality: Tabular (.csv)
Tracks energy consumption and carbon emissions for data wrangling, feature engineering, and ML modeling.

Pipeline stages:
1. Synthetic CSV generation / loading (5,000 rows, 12 features, mixed types & missing values)
2. Data cleaning & missing value imputation (median/mode)
3. Outlier detection & IQR capping
4. Advanced feature engineering (rolling windows, log transforms, interactions, one-hot encoding)
5. Multi-column categorical GroupBy aggregation
6. Supervised ML modeling (RandomForestClassifier) on the engineered tabular dataset
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "sample_data.csv")


def ensure_sample_csv():
    """Generates a realistic synthetic CSV dataset (5,000 rows, ~180 KB) if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        return CSV_PATH

    print(f"[*] Generating synthetic sample CSV at {CSV_PATH}...")
    np.random.seed(42)
    n_rows = 5000

    user_ids = np.arange(1001, 1001 + n_rows)
    age = np.random.randint(18, 70, size=n_rows).astype(float)
    income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_rows)
    credit_score = np.random.normal(680, 50, size=n_rows)
    tx_amount = np.random.exponential(scale=120, size=n_rows)
    tx_count_30d = np.random.poisson(lam=12, size=n_rows)
    device_type = np.random.choice(["iOS", "Android", "Web", "MacOS", "Windows"], size=n_rows)
    plan_tier = np.random.choice(["Basic", "Standard", "Premium"], size=n_rows, p=[0.5, 0.35, 0.15])
    is_active = np.random.choice([0, 1], size=n_rows, p=[0.25, 0.75])

    # Inject realistic missing values (5%)
    mask_age = np.random.rand(n_rows) < 0.05
    age[mask_age] = np.nan
    mask_income = np.random.rand(n_rows) < 0.05
    income[mask_income] = np.nan

    # Target variable: High-value customer churn risk (0 or 1)
    logit = (
        0.00003 * income
        - 0.03 * age
        - 0.005 * credit_score
        + 0.008 * tx_amount
        - 0.8 * is_active
        + np.random.normal(0, 0.5, size=n_rows)
    )
    churn = (logit > np.median(logit)).astype(int)

    df = pd.DataFrame({
        "user_id": user_ids,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "tx_amount": tx_amount,
        "tx_count_30d": tx_count_30d,
        "device_type": device_type,
        "plan_tier": plan_tier,
        "is_active": is_active,
        "churn": churn
    })

    df.to_csv(CSV_PATH, index=False)
    size_kb = os.path.getsize(CSV_PATH) / 1024
    print(f"[OK] Generated {CSV_PATH} ({size_kb:.1f} KB, {n_rows} rows)")
    return CSV_PATH


def process_and_model_csv(csv_path):
    """Executes a compute-controlled tabular data pipeline & ML training."""
    # 1. Ingest CSV
    df = pd.read_csv(csv_path)
    initial_rows, initial_cols = df.shape

    # 2. Missing Value Imputation
    df["age"] = df["age"].fillna(df["age"].median())
    df["income"] = df["income"].fillna(df["income"].median())

    # 3. Outlier Capping via IQR
    for col in ["income", "tx_amount"]:
        q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    # 4. Feature Engineering
    df["log_income"] = np.log1p(df["income"])
    df["log_tx_amount"] = np.log1p(df["tx_amount"])
    df["tx_per_income"] = df["tx_amount"] / (df["income"] + 1.0)
    df["credit_util_interaction"] = (df["credit_score"] / 850.0) * df["tx_count_30d"]

    # Rolling window statistics
    df["rolling_tx_mean_5"] = df["tx_amount"].rolling(window=5, min_periods=1).mean()
    df["rolling_tx_std_5"] = df["tx_amount"].rolling(window=5, min_periods=1).std().fillna(0)

    # 5. GroupBy Aggregations
    tier_aggregates = df.groupby("plan_tier").agg({
        "tx_amount": ["mean", "sum"],
        "credit_score": "mean",
        "churn": "mean"
    })
    tier_aggregates.columns = ["_".join(c) for c in tier_aggregates.columns]

    # One-Hot Encoding
    encoded_df = pd.get_dummies(df.drop(columns=["user_id"]), columns=["device_type", "plan_tier"], drop_first=True)

    # 6. ML Model Training
    X = encoded_df.drop(columns=["churn"])
    y = encoded_df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return {
        "initial_shape": (initial_rows, initial_cols),
        "engineered_features": X.shape[1],
        "test_accuracy": acc * 100,
        "n_samples": len(df)
    }


def main():
    print("=" * 80)
    print("  GREENGAUGE: TABULAR / CSV EXPERIMENT (CodeCarbon)")
    print("=" * 80)

    csv_path = ensure_sample_csv()
    file_size_kb = os.path.getsize(csv_path) / 1024
    print(f"Target CSV: {csv_path} ({file_size_kb:.2f} KB)")

    tracker = EmissionsTracker(
        project_name="exp_csv_processing",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )

    print("\nStarting tabular benchmark (Cleaning, Feature Engineering, GroupBy, RandomForest)...")
    tracker.start()
    t0 = time.perf_counter()

    metrics = process_and_model_csv(csv_path)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    throughput_rows = metrics["n_samples"] / duration

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: TABULAR / CSV WORKLOAD")
    print("=" * 80)
    print(f"  Dataset Size        : {metrics['n_samples']} rows x {metrics['initial_shape'][1]} cols")
    print(f"  File Size           : {file_size_kb:.2f} KB")
    print(f"  Engineered Features : {metrics['engineered_features']} features")
    print(f"  Test Accuracy       : {metrics['test_accuracy']:.2f}%")
    print(f"  Execution Time      : {duration:.4f} s")
    print(f"  Throughput          : {throughput_rows:.1f} rows/s")
    print(f"  Energy Consumed     : {em_data.energy_consumed:.8f} kWh")
    print(f"  Carbon Emissions    : {emissions:.10f} kg CO2e")
    print(f"  CPU Average Power   : {em_data.cpu_power or 0.0:.2f} W")
    print(f"  RAM Average Power   : {em_data.ram_power or 0.0:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()
