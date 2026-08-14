"""
Database Models — SQLAlchemy ORM
Tables: User (with auth), Product, Store, SizeProfile, TryOnSession, Testimonial
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100), nullable=False)
    email        = db.Column(db.String(150), unique=True, nullable=False)
    password_hash= db.Column(db.String(256), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    size_profile = db.relationship('SizeProfile', backref='user', uselist=False)
    sessions     = db.relationship('TryOnSession', backref='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "created_at": self.created_at.strftime('%Y-%m-%d')
        }


class Store(db.Model):
    __tablename__ = 'stores'
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(50))
    website  = db.Column(db.String(200))
    products = db.relationship('Product', backref='store')

    def to_dict(self):
        return {"id": self.id, "name": self.name, "category": self.category}


class Product(db.Model):
    __tablename__ = 'products'
    id           = db.Column(db.Integer, primary_key=True)
    store_id     = db.Column(db.Integer, db.ForeignKey('stores.id'))
    name         = db.Column(db.String(200), nullable=False)
    category     = db.Column(db.String(50))
    sub_category = db.Column(db.String(50))
    price        = db.Column(db.Float)
    image_url    = db.Column(db.String(500))
    color_hex    = db.Column(db.String(10))
    sizes        = db.Column(db.String(200))
    meta         = db.Column(db.Text)

    def to_dict(self):
        return {
            "id":           self.id,
            "name":         self.name,
            "category":     self.category,
            "sub_category": self.sub_category,
            "price":        self.price,
            "image_url":    self.image_url,
            "color_hex":    self.color_hex,
            "sizes":        self.sizes.split(',') if self.sizes else []
        }


class SizeProfile(db.Model):
    __tablename__ = 'size_profiles'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'))
    height_cm   = db.Column(db.Float)
    weight_kg   = db.Column(db.Float)
    chest_cm    = db.Column(db.Float)
    waist_cm    = db.Column(db.Float)
    hips_cm     = db.Column(db.Float)
    shoulder_cm = db.Column(db.Float)
    inseam_cm   = db.Column(db.Float)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "height_cm":   self.height_cm,
            "weight_kg":   self.weight_kg,
            "chest_cm":    self.chest_cm,
            "waist_cm":    self.waist_cm,
            "hips_cm":     self.hips_cm,
            "shoulder_cm": self.shoulder_cm,
            "inseam_cm":   self.inseam_cm
        }


class TryOnSession(db.Model):
    __tablename__ = 'tryon_sessions'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=True)
    module     = db.Column(db.String(20))
    result_url = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "module":     self.module,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M')
        }


class Testimonial(db.Model):
    __tablename__ = 'testimonials'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    name       = db.Column(db.String(100), nullable=False)
    role       = db.Column(db.String(100))
    message    = db.Column(db.Text, nullable=False)
    rating     = db.Column(db.Integer, default=5)
    avatar_url = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "role":       self.role,
            "message":    self.message,
            "rating":     self.rating,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at.strftime('%Y-%m-%d')
        }
