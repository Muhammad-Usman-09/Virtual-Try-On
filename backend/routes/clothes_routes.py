"""
Module 1: Virtual Clothes Try-On Routes
POST /api/clothes/tryon   — Apply garment onto user photo
GET  /api/clothes/products — List available clothing items
"""

from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.image_utils import save_upload, allowed_file, image_to_base64
from models.database import db, Product
from ai_engine.clothes.tryon_processor import process_clothes_tryon

clothes_bp = Blueprint('clothes', __name__)


@clothes_bp.route('/products', methods=['GET'])
def get_clothing_products():
    """Return all clothing products from inventory"""
    products = Product.query.filter_by(category='clothes').all()
    return jsonify({
        "success": True,
        "products": [p.to_dict() for p in products]
    })


@clothes_bp.route('/tryon', methods=['POST'])
def clothes_tryon():
    """
    Virtual clothes try-on endpoint.
    Expects:
        - image: user's full body photo (form-data file)
        - product_id: ID of the clothing item to try on
    Returns:
        - result_image: base64 encoded result image
    """
    # Validate inputs
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    product_id = request.form.get('product_id')

    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid image format. Use JPG or PNG"}), 400

    if not product_id:
        return jsonify({"success": False, "error": "product_id is required"}), 400

    # Get product from DB
    product = Product.query.get(product_id)
    if not product or product.category != 'clothes':
        return jsonify({"success": False, "error": "Product not found"}), 404

    try:
        # Save uploaded user image
        user_image_path = save_upload(file, 'uploads/clothes')

        # Run AI try-on
        result_image = process_clothes_tryon(
            user_image_path=user_image_path,
            garment_image_url=product.image_url,
            product_name=product.name
        )

        return jsonify({
            "success": True,
            "result_image": result_image,
            "product": product.to_dict(),
            "message": f"Try-on complete for {product.name}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
