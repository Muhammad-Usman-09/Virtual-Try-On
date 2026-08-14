"""
Module 3: Virtual Accessories Try-On Routes
POST /api/accessories/tryon   — Overlay accessory on user photo
GET  /api/accessories/products — List accessories
"""

from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.utils.image_utils import save_upload, allowed_file
from backend.models.database import db, Product
from ai_engine.accessories.accessory_processor import process_accessory

accessory_bp = Blueprint('accessories', __name__)


@accessory_bp.route('/products', methods=['GET'])
def get_accessory_products():
    """Return all accessory products, optionally filtered by type"""
    sub_category = request.args.get('type')  # ?type=glasses

    query = Product.query.filter_by(category='accessories')
    if sub_category:
        query = query.filter_by(sub_category=sub_category)

    products = query.all()
    return jsonify({
        "success": True,
        "products": [p.to_dict() for p in products]
    })


@accessory_bp.route('/tryon', methods=['POST'])
def accessory_tryon():
    """
    Try on an accessory.
    Expects:
        - image: user photo
        - product_id: accessory product
    Returns:
        - result_image: base64 result
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file       = request.files['image']
    product_id = request.form.get('product_id')

    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    product = Product.query.get(product_id)
    if not product or product.category != 'accessories':
        return jsonify({"success": False, "error": "Accessory not found"}), 404

    try:
        user_image_path = save_upload(file, 'uploads/accessories')

        result_image = process_accessory(
            user_image_path=user_image_path,
            accessory_type=product.sub_category,  # glasses / hat / earrings / necklace
            accessory_image_url=product.image_url
        )

        return jsonify({
            "success": True,
            "result_image": result_image,
            "product": product.to_dict()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
