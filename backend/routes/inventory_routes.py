"""
Inventory Management Routes
POST /api/inventory/upload-csv  — Upload product CSV
GET  /api/inventory/products    — Get all products
POST /api/inventory/seed        — Seed demo data
"""

from flask import Blueprint, request, jsonify
import sys, os, csv, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.database import db, Product, Store

inventory_bp = Blueprint('inventory', __name__)


@inventory_bp.route('/seed', methods=['POST'])
def seed_demo_data():
    """
    Seeds the database with demo store and products.
    Run this once after first launch to populate the app.
    POST /api/inventory/seed
    """
    # Create demo store
    store = Store.query.filter_by(name="Style Hub PK").first()
    if not store:
        store = Store(name="Style Hub PK", category="fashion", website="https://stylehub.pk")
        db.session.add(store)
        db.session.flush()

    # Demo clothing products
    clothes = [
        {"name": "Classic White Shirt",      "sub_category": "shirt",   "price": 1500, "sizes": "S,M,L,XL",
         "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
        {"name": "Black Slim Trousers",       "sub_category": "pants",   "price": 2200, "sizes": "S,M,L,XL,XXL",
         "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
        {"name": "Floral Summer Dress",       "sub_category": "dress",   "price": 3500, "sizes": "XS,S,M,L",
         "image_url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400"},
        {"name": "Denim Jacket",              "sub_category": "jacket",  "price": 4500, "sizes": "S,M,L,XL",
         "image_url": "https://images.unsplash.com/photo-1551537482-f2075a1d41f2?w=400"},
        {"name": "Striped Casual T-Shirt",    "sub_category": "tshirt",  "price": 800,  "sizes": "XS,S,M,L,XL",
         "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400"},
        {"name": "Formal Kurta (Navy)",       "sub_category": "kurta",   "price": 2800, "sizes": "S,M,L,XL,XXL",
         "image_url": "https://images.unsplash.com/photo-1618886614638-80e3c103d465?w=400"},
    ]

    # Demo makeup products
    makeup = [
        {"name": "Red Matte Lipstick",        "sub_category": "lipstick",   "price": 650,  "color_hex": "#C0392B",
         "image_url": "https://images.unsplash.com/photo-1586495777744-4e6232bf5a4f?w=400"},
        {"name": "Nude Pink Lipstick",         "sub_category": "lipstick",   "price": 650,  "color_hex": "#D4A0A0",
         "image_url": "https://images.unsplash.com/photo-1586495777744-4e6232bf5a4f?w=400"},
        {"name": "Berry Bold Lipstick",        "sub_category": "lipstick",   "price": 650,  "color_hex": "#8B1A4A",
         "image_url": "https://images.unsplash.com/photo-1586495777744-4e6232bf5a4f?w=400"},
        {"name": "Rose Gold Eyeshadow",        "sub_category": "eyeshadow",  "price": 1200, "color_hex": "#B76E79",
         "image_url": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=400"},
        {"name": "Smoky Brown Eyeshadow",      "sub_category": "eyeshadow",  "price": 1200, "color_hex": "#4A2C2A",
         "image_url": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=400"},
        {"name": "Peach Blush",                "sub_category": "blush",      "price": 900,  "color_hex": "#FFAD8E",
         "image_url": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=400"},
        {"name": "Warm Foundation (Shade 03)", "sub_category": "foundation", "price": 1800, "color_hex": "#C68B6D",
         "image_url": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400"},
    ]

    # Demo accessories
    accessories = [
        {"name": "Classic Aviator Sunglasses", "sub_category": "glasses",  "price": 1200,
         "image_url": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400"},
        {"name": "Round Black Frames",          "sub_category": "glasses",  "price": 950,
         "image_url": "https://images.unsplash.com/photo-1574258495973-f010dfbb5371?w=400"},
        {"name": "Gold Hoop Earrings",          "sub_category": "earrings", "price": 750,
         "image_url": "https://images.unsplash.com/photo-1535632066927-ab7c9ab60908?w=400"},
        {"name": "Pearl Necklace",              "sub_category": "necklace", "price": 1100,
         "image_url": "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=400"},
        {"name": "Beige Bucket Hat",            "sub_category": "hat",      "price": 850,
         "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
        {"name": "Leather Watch (Brown)",       "sub_category": "watch",    "price": 3500,
         "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    ]

    added = 0
    for item in clothes:
        if not Product.query.filter_by(name=item['name']).first():
            p = Product(store_id=store.id, category='clothes', **item)
            db.session.add(p)
            added += 1

    for item in makeup:
        if not Product.query.filter_by(name=item['name']).first():
            p = Product(store_id=store.id, category='makeup', **item)
            db.session.add(p)
            added += 1

    for item in accessories:
        if not Product.query.filter_by(name=item['name']).first():
            p = Product(store_id=store.id, category='accessories', **item)
            db.session.add(p)
            added += 1

    db.session.commit()

    return jsonify({
        "success": True,
        "message": f"Demo data seeded! Added {added} products.",
        "store": store.name
    })


@inventory_bp.route('/products', methods=['GET'])
def get_all_products():
    """Get all products, optionally filter by category"""
    category = request.args.get('category')
    query = Product.query
    if category:
        query = query.filter_by(category=category)
    products = query.all()
    return jsonify({"success": True, "products": [p.to_dict() for p in products]})


@inventory_bp.route('/upload-csv', methods=['POST'])
def upload_csv():
    """
    Upload a CSV file to bulk-import products.
    CSV columns: name, category, sub_category, price, image_url, color_hex, sizes
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No CSV file uploaded"}), 400

    file = request.files['file']
    store_name = request.form.get('store_name', 'Imported Store')

    store = Store.query.filter_by(name=store_name).first()
    if not store:
        store = Store(name=store_name, category='fashion')
        db.session.add(store)
        db.session.flush()

    content = file.read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))

    added = 0
    for row in reader:
        p = Product(
            store_id     = store.id,
            name         = row.get('name', ''),
            category     = row.get('category', 'clothes'),
            sub_category = row.get('sub_category', ''),
            price        = float(row.get('price', 0)),
            image_url    = row.get('image_url', ''),
            color_hex    = row.get('color_hex', ''),
            sizes        = row.get('sizes', '')
        )
        db.session.add(p)
        added += 1

    db.session.commit()
    return jsonify({"success": True, "message": f"Imported {added} products from CSV"})
