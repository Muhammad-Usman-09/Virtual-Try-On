"""
Configuration settings for the Flask app.
Reads from .env file — never hardcode secrets!
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

class Config:
    # Flask secret key (used for sessions)
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-in-production-abc123')

    # Database — uses SQLite locally, PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///tryon.db'   # Default: local SQLite file (no setup needed!)
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

    # Allowed image extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    # Cloudinary (for cloud image storage) — optional
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME', '')
    CLOUDINARY_API_KEY    = os.getenv('CLOUDINARY_API_KEY', '')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET', '')

    # Debug mode
    DEBUG = os.getenv('DEBUG', 'True') == 'True'

    # Session secret (required for login sessions)
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE   = False   # True in production (HTTPS)
