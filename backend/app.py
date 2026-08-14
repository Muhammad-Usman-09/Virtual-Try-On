"""
AI-Driven Virtual Try-On Ecosystem
Main Flask Application Entry Point — Updated with Auth + Analytics + Testimonials
"""

from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from routes.clothes_routes     import clothes_bp
from routes.makeup_routes      import makeup_bp
from routes.size_routes        import size_bp
from routes.inventory_routes   import inventory_bp
from routes.auth_routes        import auth_bp
from routes.analytics_routes   import analytics_bp
from routes.testimonial_routes import testimonial_bp
from models.database import db
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    db.init_app(app)

    app.register_blueprint(clothes_bp,      url_prefix='/api/clothes')
    app.register_blueprint(makeup_bp,       url_prefix='/api/makeup')
    app.register_blueprint(size_bp,         url_prefix='/api/size')
    app.register_blueprint(inventory_bp,    url_prefix='/api/inventory')
    app.register_blueprint(auth_bp,         url_prefix='/api/auth')
    app.register_blueprint(analytics_bp,    url_prefix='/api/analytics')
    app.register_blueprint(testimonial_bp,  url_prefix='/api/testimonials')

    with app.app_context():
        db.create_all()

    @app.route('/')
    def home():
        return jsonify({
            "message": "AI-FIT Try-On Ecosystem API",
            "version": "2.0.0",
            "modules": ["clothes", "makeup", "size", "analytics", "auth", "testimonials"]
        })

    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"})

    return app
    
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
