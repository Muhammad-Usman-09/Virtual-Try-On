"""
Size Model Training Script
Run this once to train and save the ML size prediction model.

Usage:
    cd ai_engine/size
    python train_model.py

Uses the real ANSUR II Anthropometric Survey dataset (US Army, 2012):
    Source: https://www.openlab.psu.edu/ansur2
    Files:  data/size_charts/ANSUR_II_MALE_Public.csv
            data/size_charts/ANSUR_II_FEMALE_Public.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

# Reuse the exact same size chart used by the rule-based predictor,
# so ML predictions and rule-based fallback stay consistent.
sys.path.insert(0, os.path.dirname(__file__))
from size_predictor import SIZE_CHART

ANSUR_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'size_charts')
MALE_CSV = os.path.join(ANSUR_DIR, 'ANSUR_II_MALE_Public.csv')
FEMALE_CSV = os.path.join(ANSUR_DIR, 'ANSUR_II_FEMALE_Public.csv')

# ANSUR II raw column -> our feature name (all raw values are in mm, mass in hectograms)
ANSUR_COLUMN_MAP = {
    "stature":               "height_cm",
    "weightkg":               "weight_kg",
    "chestcircumference":    "chest_cm",
    "waistcircumference":    "waist_cm",
    "buttockcircumference":  "hips_cm",
    "biacromialbreadth":     "shoulder_cm",
    "crotchheight":          "inseam_cm",
}


def assign_size_label(row, chart):
    """
    Derive a garment size label (XS-XXL) for a real body-measurement row
    using the same chest/waist/hips ranges as the rule-based predictor.
    Returns None if no size range matches closely enough (row is dropped).
    """
    best_size, best_score = None, -1
    for size, ranges in chart.items():
        score, total = 0, 0
        for key, col in (("chest", "chest_cm"), ("waist", "waist_cm"), ("hips", "hips_cm")):
            if key in ranges:
                lo, hi = ranges[key]
                val = row[col]
                if lo <= val <= hi:
                    score += 1
                elif val < lo:
                    score += max(0, 1 - (lo - val) / 10)
                else:
                    score += max(0, 1 - (val - hi) / 10)
                total += 1
        norm = score / total if total else 0
        if norm > best_score:
            best_score, best_size = norm, size
    return best_size


def load_ansur_data():
    """
    Load and combine the real ANSUR II male + female datasets, convert units
    (mm -> cm, hectograms -> kg), and derive a garment size label for each
    person using the existing SIZE_CHART.
    """
    frames = []
    for path, gender in [(MALE_CSV, "men"), (FEMALE_CSV, "women")]:
        raw = pd.read_csv(path, encoding='latin1')
        cols_needed = list(ANSUR_COLUMN_MAP.keys())
        df = raw[cols_needed].rename(columns=ANSUR_COLUMN_MAP).copy()

        # Convert units: mm -> cm (divide by 10), hectograms -> kg (divide by 10)
        for col in ["height_cm", "chest_cm", "waist_cm", "hips_cm", "shoulder_cm", "inseam_cm"]:
            df[col] = df[col] / 10.0
        df["weight_kg"] = df["weight_kg"] / 10.0
        df["gender"] = gender
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Derive size label per gender using the same chart as the rule-based predictor
    labels = []
    for gender, group in df.groupby("gender"):
        chart = SIZE_CHART[gender]
        sizes = group.apply(lambda r: assign_size_label(r, chart), axis=1)
        labels.append(sizes)
    df["size"] = pd.concat(labels).sort_index()
    df = df.dropna(subset=["size"])

    return df


def train_size_model():
    """Train Random Forest + KNN on real ANSUR II data, keep the better one."""
    print("Loading ANSUR II dataset (male + female)...")
    df = load_ansur_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Size distribution:\n{df['size'].value_counts()}\n")

    features = ["height_cm", "weight_kg", "chest_cm", "waist_cm",
                "hips_cm", "shoulder_cm", "inseam_cm"]
    X = df[features].values
    y = df["size"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest ---
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc:.2%}")
    print(classification_report(y_test, rf_pred))

    # --- KNN (needs scaled features since it's distance-based) ---
    print("Training KNN model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=9, weights='distance')
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"KNN Accuracy: {knn_acc:.2%}")
    print(classification_report(y_test, knn_pred))

    # --- Keep whichever performs better as the primary model ---
    if rf_acc >= knn_acc:
        print(f"\nRandom Forest wins ({rf_acc:.2%} vs {knn_acc:.2%}) -> saving as primary model")
        best_model, best_name = rf_model, "random_forest"
    else:
        print(f"\nKNN wins ({knn_acc:.2%} vs {rf_acc:.2%}) -> saving as primary model")
        best_model, best_name = knn_model, "knn"

    model_path = os.path.join(os.path.dirname(__file__), 'size_model.pkl')
    bundle = {
        "model": best_model,
        "model_type": best_name,
        "scaler": scaler if best_name == "knn" else None,
        "features": features,
        "rf_accuracy": rf_acc,
        "knn_accuracy": knn_acc,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {model_path}")
    return best_model


if __name__ == '__main__':
    train_size_model()