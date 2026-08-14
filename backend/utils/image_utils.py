"""
Image Utility Functions
Used by all AI modules for preprocessing/saving images
"""

import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
import uuid
from flask import current_app


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed = {'png', 'jpg', 'jpeg', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def save_upload(file, subfolder='uploads'):
    """Save uploaded file and return its path"""
    upload_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', subfolder
    )
    os.makedirs(upload_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(upload_dir, filename)

    img = Image.open(file).convert('RGB')
    img.save(filepath, 'JPEG', quality=90)
    return filepath


def load_image_cv2(filepath):
    """Load image as OpenCV BGR array"""
    return cv2.imread(filepath)


def load_image_pil(filepath):
    """Load image as PIL Image"""
    return Image.open(filepath).convert('RGB')


def image_to_base64(image_array):
    """Convert OpenCV image array to base64 string for frontend display"""
    _, buffer = cv2.imencode('.jpg', image_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def pil_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90)
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def resize_image(img, max_size=800):
    """Resize image maintaining aspect ratio"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def apply_color_overlay(image, mask, hex_color, alpha=0.5):
    """
    Apply a color overlay on a masked region.
    Used for makeup: apply lipstick/eyeshadow color.

    Args:
        image: OpenCV BGR image
        mask: Binary mask (same size as image)
        hex_color: Color string like '#FF0000'
        alpha: Transparency (0=invisible, 1=solid)
    Returns:
        Modified image
    """
    # Convert hex to BGR
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    color_bgr = (b, g, r)

    result = image.copy()
    overlay = image.copy()

    # Apply color where mask is active
    overlay[mask > 0] = color_bgr

    # Blend overlay with original
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    return result


def overlay_png_on_image(background, overlay_img_path, x, y, scale=1.0):
    """
    Overlay a PNG (with transparency) onto a background image.
    Used for accessories (glasses, hats, etc.)

    Args:
        background: OpenCV BGR image
        overlay_img_path: Path to PNG file with alpha channel
        x, y: Top-left position to place overlay
        scale: Resize overlay by this factor
    Returns:
        Modified background image
    """
    overlay = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return background

    # Resize
    new_w = int(overlay.shape[1] * scale)
    new_h = int(overlay.shape[0] * scale)
    overlay = cv2.resize(overlay, (new_w, new_h))

    result = background.copy()
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Clamp to image bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    ox1 = x1 - x
    oy1 = y1 - y

    if x2 <= x1 or y2 <= y1:
        return background

    roi = result[y1:y2, x1:x2]
    ov_crop = overlay[oy1:oy1+(y2-y1), ox1:ox1+(x2-x1)]

    if overlay.shape[2] == 4:
        # Has alpha channel
        alpha_mask = ov_crop[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (
                alpha_mask * ov_crop[:, :, c] +
                (1 - alpha_mask) * roi[:, :, c]
            ).astype(np.uint8)
    else:
        roi[:] = ov_crop[:, :, :3]

    result[y1:y2, x1:x2] = roi
    return result
