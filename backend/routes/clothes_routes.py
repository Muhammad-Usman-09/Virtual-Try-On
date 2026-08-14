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


def _resolve_cloth_type(product):
    """
    Map inventory clothing categories to CatVTON categories:
        upper | lower | overall
    """
    sub_category = (product.sub_category or "").strip().lower()
    product_name = (product.name or "").strip().lower()
    text = f"{sub_category} {product_name}"

    lower_keywords = {
        "pant", "pants", "trouser", "trousers", "jean", "jeans",
        "short", "shorts", "skirt", "leggings", "jogger", "joggers",
        "shalwar", "pajama", "pajamas",
    }
    overall_keywords = {
        "dress", "gown", "jumpsuit", "overall", "onepiece", "one-piece",
        "frock", "maxi", "suit", "kameez",
    }
    upper_keywords = {
        "shirt", "tshirt", "t-shirt", "tee", "top", "blouse",
        "kurta", "hoodie", "jacket", "coat", "sweater", "cardigan",
        "waistcoat", "vest",
    }

    if any(keyword in text for keyword in overall_keywords):
        return "overall"
    if any(keyword in text for keyword in lower_keywords):
        return "lower"
    if any(keyword in text for keyword in upper_keywords):
        return "upper"
    return "upper"


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
        cloth_type = _resolve_cloth_type(product)

        # Run AI try-on
        result_image = process_clothes_tryon(
            user_image_path=user_image_path,
            garment_image_url=product.image_url,
            product_name=product.name,
            cloth_type=cloth_type
        )

        return jsonify({
            "success": True,
            "result_image": result_image,
            "product": product.to_dict(),
            "cloth_type": cloth_type,
            "message": f"Try-on complete for {product.name} ({cloth_type})"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
