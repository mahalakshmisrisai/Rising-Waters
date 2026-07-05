"""
train_model.py
---------------
Loads flood_data.csv, trains four classifiers (Decision Tree, Random Forest,
KNN, XGBoost), compares their performance, and saves the best-performing
model + scaler to disk for use by the Flask app (app.py).

Run:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this runs headless too
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

DATA_PATH = "flood_data.csv"
FEATURE_COLUMNS = [
    "annual_rainfall",
    "seasonal_rainfall",
    "cloud_visibility",
    "temperature",
    "humidity",
]
TARGET_COLUMN = "flood"

MODEL_OUT = "model/flood_model.pkl"
SCALER_OUT = "model/scaler.pkl"
METRICS_OUT = "model/model_comparison.png"


def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (helps KNN especially; harmless for tree models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        trained_models[name] = model

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))

    # --- Pick and save the best model ------------------------------------
    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]
    best_acc = results[best_name]

    print("\n================ SUMMARY ================")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<15s}: {acc * 100:.2f}%")
    print(f"\nBest model: {best_name} ({best_acc * 100:.2f}% accuracy)")

    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")
    joblib.dump(best_name, "model/best_model_name.pkl")
    print(f"\nSaved best model to '{MODEL_OUT}' and scaler to '{SCALER_OUT}'.")

    # --- Comparison chart --------------------------------------------------
    plt.figure(figsize=(7, 4))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    bars = plt.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.2f}%",
                  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(METRICS_OUT)
    print(f"Saved comparison chart to '{METRICS_OUT}'.")

    # Confusion matrix for the best model
    cm = confusion_matrix(y_test, trained_models[best_name].predict(X_test_scaled))
    print(f"\nConfusion matrix for {best_name}:\n{cm}")


if __name__ == "__main__":
    main()
