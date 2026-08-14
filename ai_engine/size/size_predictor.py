"""
AI Engine — Module 4: Size & Fit Recommender
Rule-based + ML model for clothing size prediction.

For a real ML model, train using ANSUR II dataset:
  https://www.openlab.psu.edu/ansur2

This file includes:
1. Rule-based predictor (works immediately, no training needed)
2. train_model() function (train on real data when available)
3. load_and_predict() for trained sklearn model
"""

import os
import json
import numpy as np

# Path to saved ML model (created by train_model.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'size_model.pkl')


# ── International Size Charts (measurements in cm) ──────────────────────────
SIZE_CHART = {
    "women": {
        "XS":  {"chest": (71, 81),  "waist": (56, 63),  "hips": (79, 88)},
        "S":   {"chest": (81, 86),  "waist": (63, 68),  "hips": (88, 93)},
        "M":   {"chest": (86, 92),  "waist": (68, 74),  "hips": (93, 99)},
        "L":   {"chest": (92, 98),  "waist": (74, 80),  "hips": (99, 105)},
        "XL":  {"chest": (98, 106), "waist": (80, 88),  "hips": (105, 113)},
        "XXL": {"chest": (106, 116),"waist": (88, 98),  "hips": (113, 123)},
    },
    "men": {
        "XS":  {"chest": (79, 86),  "waist": (66, 72)},
        "S":   {"chest": (86, 92),  "waist": (72, 78)},
        "M":   {"chest": (92, 99),  "waist": (78, 84)},
        "L":   {"chest": (99, 107), "waist": (84, 91)},
        "XL":  {"chest": (107, 115),"waist": (91, 99)},
        "XXL": {"chest": (115, 124),"waist": (99, 109)},
    }
}


def rule_based_size(chest_cm, waist_cm, hips_cm, gender="women"):
    """
    Determine size using rule-based chart matching.
    Returns (size, confidence, notes)
    """
    chart = SIZE_CHART.get(gender, SIZE_CHART["women"])
    scores = {}

    for size, ranges in chart.items():
        score = 0
        total = 0

        if "chest" in ranges and chest_cm > 0:
            lo, hi = ranges["chest"]
            if lo <= chest_cm <= hi:
                score += 1
            elif chest_cm < lo:
                score += max(0, 1 - (lo - chest_cm) / 10)
            else:
                score += max(0, 1 - (chest_cm - hi) / 10)
            total += 1

        if "waist" in ranges and waist_cm > 0:
            lo, hi = ranges["waist"]
            if lo <= waist_cm <= hi:
                score += 1
            elif waist_cm < lo:
                score += max(0, 1 - (lo - waist_cm) / 10)
            else:
                score += max(0, 1 - (waist_cm - hi) / 10)
            total += 1

        if "hips" in ranges and hips_cm > 0 and gender == "women":
            lo, hi = ranges["hips"]
            if lo <= hips_cm <= hi:
                score += 1
            elif hips_cm < lo:
                score += max(0, 1 - (lo - hips_cm) / 10)
            else:
                score += max(0, 1 - (hips_cm - hi) / 10)
            total += 1

        scores[size] = score / total if total > 0 else 0

    best_size = max(scores, key=scores.get)
    confidence = scores[best_size]
    return best_size, round(confidence, 2)


def generate_fit_notes(measurements, size, gender):
    """Generate human-readable fit advice"""
    notes = []
    chest = measurements.get("chest_cm", 0)
    waist = measurements.get("waist_cm", 0)
    hips  = measurements.get("hips_cm", 0)

    if gender == "women" and hips > 0 and chest > 0:
        diff = hips - chest
        if diff > 8:
            notes.append("Your hips are wider relative to chest — consider sizing up in bottoms.")
        elif diff < -5:
            notes.append("Your chest is wider relative to hips — consider sizing up in tops.")

    if waist > 0 and chest > 0:
        ratio = waist / chest
        if ratio < 0.75:
            notes.append("You have a defined waist — fitted styles will suit you well.")
        elif ratio > 0.92:
            notes.append("For comfort, consider relaxed or straight-cut styles.")

    if not notes:
        notes.append(f"Size {size} should fit you well across most styles.")

    return " ".join(notes)


def predict_size(height_cm, weight_kg, chest_cm, waist_cm, hips_cm,
                 shoulder_cm=0, inseam_cm=0, gender="women"):
    """
    Main size prediction function.
    Tries to load trained ML model first; falls back to rule-based predictor.

    Returns dict with full recommendation breakdown.
    """
    measurements = {
        "height_cm":   height_cm,
        "weight_kg":   weight_kg,
        "chest_cm":    chest_cm,
        "waist_cm":    waist_cm,
        "hips_cm":     hips_cm,
        "shoulder_cm": shoulder_cm,
        "inseam_cm":   inseam_cm,
    }

    # Try ML model if available (trained on real ANSUR II data via train_model.py)
    if os.path.exists(MODEL_PATH):
        try:
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                bundle = pickle.load(f)

            model = bundle["model"]
            scaler = bundle.get("scaler")

            features = np.array([[height_cm, weight_kg, chest_cm,
                                   waist_cm, hips_cm, shoulder_cm, inseam_cm]])
            if scaler is not None:
                features = scaler.transform(features)

            predicted_size = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)), 2)

            print(f"[size] ML model ({bundle.get('model_type', 'unknown')}) prediction: "
                  f"{predicted_size} ({confidence:.0%})")
            size = predicted_size

        except Exception as e:
            print(f"[size] ML model error, using rule-based: {e}")
            size, confidence = rule_based_size(chest_cm, waist_cm, hips_cm, gender)
    else:
        print("[size] No trained model found — using rule-based predictor")
        size, confidence = rule_based_size(chest_cm, waist_cm, hips_cm, gender)

    # Size breakdown per garment type
    size_order = ["XS", "S", "M", "L", "XL", "XXL"]
    size_idx = size_order.index(size) if size in size_order else 2

    # Slight variations per garment type
    top_size    = size
    bottom_size = size_order[min(size_idx + (1 if hips_cm > chest_cm + 8 else 0), 5)]
    dress_size  = size_order[max(size_idx - (1 if waist_cm < chest_cm - 10 else 0), 0)]

    fit_notes = generate_fit_notes(measurements, size, gender)

    return {
        "recommended_size": size,
        "confidence":       confidence,
        "size_breakdown": {
            "top":    top_size,
            "bottom": bottom_size,
            "dress":  dress_size,
        },
        "fit_notes":        fit_notes,
        "measurements":     measurements,
        "method":           "ml_model" if os.path.exists(MODEL_PATH) else "rule_based"
    }