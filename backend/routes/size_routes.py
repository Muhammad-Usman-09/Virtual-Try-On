"""
Module 4: AI Size & Fit Recommender Routes
POST /api/size/recommend  — Get size recommendation from measurements
POST /api/size/save       — Save user size profile
GET  /api/size/profile    — Get saved profile
"""

from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.database import db, SizeProfile
from ai_engine.size.size_predictor import predict_size

size_bp = Blueprint('size', __name__)


@size_bp.route('/recommend', methods=['POST'])
def recommend_size():
    """
    Predict clothing size from body measurements.

    Expects JSON:
    {
        "height_cm":   170,
        "weight_kg":   65,
        "chest_cm":    90,
        "waist_cm":    75,
        "hips_cm":     95,
        "shoulder_cm": 42,
        "inseam_cm":   78,
        "gender":      "female"   (optional: male/female/unisex)
    }

    Returns:
    {
        "recommended_size": "M",
        "confidence": 0.87,
        "size_breakdown": {"top": "M", "bottom": "L", "dress": "M"},
        "fit_notes": "Slightly wider hips — go L in bottoms"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400

    required = ['height_cm', 'weight_kg', 'chest_cm', 'waist_cm', 'hips_cm']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "success": False,
            "error": f"Missing fields: {', '.join(missing)}"
        }), 400

    try:
        result = predict_size(
            height_cm   = float(data['height_cm']),
            weight_kg   = float(data['weight_kg']),
            chest_cm    = float(data['chest_cm']),
            waist_cm    = float(data['waist_cm']),
            hips_cm     = float(data['hips_cm']),
            shoulder_cm = float(data.get('shoulder_cm', 0)),
            inseam_cm   = float(data.get('inseam_cm', 0)),
            gender      = data.get('gender', 'unisex')
        )

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@size_bp.route('/save', methods=['POST'])
def save_size_profile():
    """Save user's body measurements"""
    data = request.get_json()
    user_id = data.get('user_id', 1)  # Default user for demo

    profile = SizeProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = SizeProfile(user_id=user_id)
        db.session.add(profile)

    profile.height_cm   = data.get('height_cm')
    profile.weight_kg   = data.get('weight_kg')
    profile.chest_cm    = data.get('chest_cm')
    profile.waist_cm    = data.get('waist_cm')
    profile.hips_cm     = data.get('hips_cm')
    profile.shoulder_cm = data.get('shoulder_cm')
    profile.inseam_cm   = data.get('inseam_cm')

    db.session.commit()
    return jsonify({"success": True, "message": "Profile saved!"})


@size_bp.route('/chart', methods=['GET'])
def get_size_chart():
    """Return standard international size chart for reference"""
    chart = {
        "women": {
            "XS": {"chest": "76-81", "waist": "58-63", "hips": "83-88"},
            "S":  {"chest": "81-86", "waist": "63-68", "hips": "88-93"},
            "M":  {"chest": "86-91", "waist": "68-73", "hips": "93-98"},
            "L":  {"chest": "91-97", "waist": "73-79", "hips": "98-104"},
            "XL": {"chest": "97-104","waist": "79-86", "hips": "104-111"},
            "XXL":{"chest": "104-112","waist":"86-94", "hips": "111-119"},
        },
        "men": {
            "XS": {"chest": "81-86", "waist": "68-73"},
            "S":  {"chest": "86-91", "waist": "73-78"},
            "M":  {"chest": "91-97", "waist": "78-83"},
            "L":  {"chest": "97-104","waist": "83-89"},
            "XL": {"chest": "104-112","waist":"89-96"},
            "XXL":{"chest": "112-120","waist":"96-104"},
        }
    }
    return jsonify({"success": True, "size_chart": chart, "unit": "cm"})
