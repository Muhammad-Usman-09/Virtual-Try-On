"""
Module 2: Virtual Makeup Try-On Routes
POST /api/makeup/apply    — Apply makeup on user face photo
GET  /api/makeup/products — List available makeup products
"""

from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.utils.image_utils import save_upload, allowed_file
from backend.models.database import db, Product
from ai_engine.makeup.makeup_processor import process_makeup

makeup_bp = Blueprint('makeup', __name__)


@makeup_bp.route('/products', methods=['GET'])
def get_makeup_products():
    """Return all makeup products grouped by sub-category"""
    products = Product.query.filter_by(category='makeup').all()
    grouped = {}
    for p in products:
        sub = p.sub_category or 'other'
        grouped.setdefault(sub, []).append(p.to_dict())

    return jsonify({
        "success": True,
        "products": grouped,
        "total": len(products)
    })


@makeup_bp.route('/apply', methods=['POST'])
def apply_makeup():
    """
    Apply makeup product to user face photo.
    Expects:
        - image: user's face photo
        - product_id: makeup product to apply
        - intensity: 0.0 to 1.0 (optional, default 0.6)
    Returns:
        - result_image: base64 encoded result
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    product_id  = request.form.get('product_id')
    intensity   = float(request.form.get('intensity', 0.6))

    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    product = Product.query.get(product_id)
    if not product or product.category != 'makeup':
        return jsonify({"success": False, "error": "Makeup product not found"}), 404

    try:
        user_image_path = save_upload(file, 'uploads/makeup')

        result_image = process_makeup(
            user_image_path=user_image_path,
            product_type=product.sub_category,   # lipstick / eyeshadow / blush / foundation
            color_hex=product.color_hex,
            intensity=intensity
        )

        return jsonify({
            "success": True,
            "result_image": result_image,
            "product": product.to_dict(),
            "applied": product.sub_category,
            "color": product.color_hex
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@makeup_bp.route('/apply-multiple', methods=['POST'])
def apply_multiple_makeup():
    """
    Apply multiple makeup products at once (full look).
    Expects JSON body:
        {
          "image_path": "...",
          "products": [{"product_id": 1, "intensity": 0.6}, ...]
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400

    # This would apply products one by one using the same image
    # Implementation left as exercise / future enhancement
    return jsonify({
        "success": False,
        "message": "Full look feature coming soon!"
    }), 501
