"""
Testimonial Routes — GET list + POST new testimonial
"""
from flask import Blueprint, request, jsonify, session
from models.database import db, Testimonial

testimonial_bp = Blueprint('testimonials', __name__)


@testimonial_bp.route('/', methods=['GET'])
def get_testimonials():
    tests = Testimonial.query.order_by(Testimonial.created_at.desc()).limit(12).all()
    if not tests:
        # Seed demo testimonials if empty
        _seed_demo_testimonials()
        tests = Testimonial.query.all()
    return jsonify({"success": True, "testimonials": [t.to_dict() for t in tests]})


@testimonial_bp.route('/', methods=['POST'])
def add_testimonial():
    data    = request.get_json() or {}
    name    = (data.get('name') or '').strip()
    message = (data.get('message') or '').strip()
    role    = (data.get('role') or '').strip()
    rating  = int(data.get('rating', 5))
    user_id = session.get('user_id')

    if not name or not message:
        return jsonify({"success": False, "error": "Name and message required"}), 400

    t = Testimonial(
        user_id = user_id,
        name    = name,
        role    = role or 'Fashion Shopper',
        message = message,
        rating  = min(5, max(1, rating))
    )
    db.session.add(t)
    db.session.commit()
    return jsonify({"success": True, "testimonial": t.to_dict()}), 201


def _seed_demo_testimonials():
    demos = [
        {"name": "Ayesha Raza",    "role": "Online Shopper, Lahore",   "rating": 5,
         "message": "I used to return 3 out of 5 items because of sizing. AI-FIT showed me exactly how the shalwar kameez would fit — zero returns since!"},
        {"name": "Bilal Chaudhry","role": "E-commerce Seller, Karachi","rating": 5,
         "message": "Integrated this into our store and saw a 30% drop in returns in the first month. Game changer for Pakistani fashion retail."},
        {"name": "Sana Mirza",    "role": "University Student, Islamabad","rating": 4,
         "message": "The makeup try-on is so accurate! Tried 10 lipstick shades before ordering — got exactly what I wanted."},
        {"name": "Farhan Ahmed",  "role": "Fashion Brand Owner",        "rating": 5,
         "message": "Our customers now shop with so much more confidence. The AI size recommender is brilliant — even handles Pakistani brand sizing quirks."},
        {"name": "Zara Malik",    "role": "Influencer, Islamabad",      "rating": 5,
         "message": "Virtual try-on has completely changed how I recommend outfits to my followers. They can try before they buy!"},
        {"name": "Hassan Iqbal",  "role": "Software Engineer",          "rating": 4,
         "message": "Impressive technical work — MediaPipe + CV pipeline is smooth. The analytics dashboard is exactly what a BDA project should show."},
    ]
    for d in demos:
        db.session.add(Testimonial(**d))
    db.session.commit()
