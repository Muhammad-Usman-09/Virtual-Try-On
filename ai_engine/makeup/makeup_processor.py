"""
AI Engine — Module 2: Virtual Makeup Try-On
============================================
Uses MediaPipe Face Mesh (468 landmarks) for precise facial feature detection.
Applies makeup products with realistic color blending.

COMFYUI NOTE (Sir's recommendation):
--------------------------------------
ComfyUI is a node-based stable diffusion interface that can do photo-realistic
makeup try-on via ControlNet + InPainting. Integration path would be:
  1. Run ComfyUI locally (comfyanonymous/ComfyUI on GitHub)
  2. Use its HTTP API (/prompt endpoint) to send workflow JSON
  3. Pass user face image + makeup mask + color reference

For this FYP version we use MediaPipe-based CV approach which:
  - Runs fully offline (no GPU needed)
  - Produces clean, predictable results
  - Completes in ~1 second
  - Supports lipstick, eyeshadow, blush, foundation

The ComfyUI path can be shown as "Future Enhancement" in panel.
"""

import cv2
import numpy as np
import mediapipe as mp
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.utils.image_utils import image_to_base64, resize_image

mp_face_mesh = mp.solutions.face_mesh

# ── Facial landmark index groups (MediaPipe 468-point mesh) ─────────────────
LIPS_OUTER    = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
LEFT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_CHEEK    = [116,123,147,187,207,206,203,36,47,100,101,50,36]
RIGHT_CHEEK   = [345,352,376,411,427,426,423,266,277,329,330,280,345]
FACE_OVAL     = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]


def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)


def get_pts(face_landmarks, indices, w, h):
    return np.array(
        [[int(face_landmarks.landmark[i].x*w), int(face_landmarks.landmark[i].y*h)]
         for i in indices], dtype=np.int32
    )


def apply_lipstick(image, face_landmarks, color_hex, intensity=0.65):
    h,w = image.shape[:2]
    bgr  = hex_to_bgr(color_hex)
    pts  = get_pts(face_landmarks, LIPS_OUTER, w, h)

    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (5,5), 2)

    result  = image.copy()
    overlay = image.copy()
    overlay[mask>0] = bgr

    alpha = mask.astype(float)/255.0 * intensity
    for c in range(3):
        result[:,:,c] = (alpha*overlay[:,:,c] + (1-alpha)*result[:,:,c]).astype(np.uint8)
    return result


def apply_eyeshadow(image, face_landmarks, color_hex, intensity=0.50):
    h,w = image.shape[:2]
    bgr  = hex_to_bgr(color_hex)
    result = image.copy()

    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        pts = get_pts(face_landmarks, eye_indices, w, h)
        shadow = pts.copy()
        shadow[:,1] -= 18

        combined = np.vstack([pts, shadow[::-1]])
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(mask, [combined], 255)
        mask = cv2.GaussianBlur(mask, (11,11), 5)

        overlay = result.copy()
        overlay[mask>0] = bgr
        alpha = mask.astype(float)/255.0 * intensity
        for c in range(3):
            result[:,:,c] = (alpha*overlay[:,:,c] + (1-alpha)*result[:,:,c]).astype(np.uint8)

    return result


def apply_eyeliner(image, face_landmarks, color_hex, intensity=0.8):
    """Draw a thin eyeliner line along upper lash line"""
    h,w = image.shape[:2]
    bgr  = hex_to_bgr(color_hex)
    result = image.copy()

    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        pts = get_pts(face_landmarks, eye_indices[:8], w, h)
        cv2.polylines(result, [pts], False, bgr, 2, cv2.LINE_AA)
    return result


def apply_blush(image, face_landmarks, color_hex, intensity=0.32):
    h,w = image.shape[:2]
    bgr  = hex_to_bgr(color_hex)
    result = image.copy()

    for cheek_indices in [LEFT_CHEEK, RIGHT_CHEEK]:
        pts  = get_pts(face_landmarks, cheek_indices, w, h)
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (51,51), 20)

        overlay = result.copy()
        overlay[mask>0] = bgr
        alpha = mask.astype(float)/255.0 * intensity
        for c in range(3):
            result[:,:,c] = (alpha*overlay[:,:,c] + (1-alpha)*result[:,:,c]).astype(np.uint8)

    return result


def apply_foundation(image, face_landmarks, color_hex, intensity=0.22):
    h,w = image.shape[:2]
    bgr  = hex_to_bgr(color_hex)

    pts  = get_pts(face_landmarks, FACE_OVAL, w, h)
    hull = cv2.convexHull(pts)

    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.GaussianBlur(mask, (21,21), 10)

    overlay = image.copy()
    overlay[mask>0] = bgr
    result = image.copy()
    alpha  = mask.astype(float)/255.0 * intensity
    for c in range(3):
        result[:,:,c] = (alpha*overlay[:,:,c] + (1-alpha)*result[:,:,c]).astype(np.uint8)

    return result


MAKEUP_FNS = {
    'lipstick':   apply_lipstick,
    'eyeshadow':  apply_eyeshadow,
    'eyeliner':   apply_eyeliner,
    'blush':      apply_blush,
    'foundation': apply_foundation,
}


def process_makeup(user_image_path, product_type, color_hex, intensity=0.6):
    """
    Main function — apply makeup and return base64 result.
    product_type: lipstick / eyeshadow / eyeliner / blush / foundation
    """
    print(f"[makeup] Applying {product_type} ({color_hex}) intensity={intensity}")

    image = cv2.imread(user_image_path)
    if image is None:
        raise ValueError("Could not load user image")

    image = resize_image(image, max_size=720)
    h,w   = image.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True,          # More accurate lip/eye landmarks
        min_detection_confidence=0.5
    ) as mesh:
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected. Please upload a clear, well-lit frontal face photo.")

        lm = results.multi_face_landmarks[0]
        fn = MAKEUP_FNS.get(product_type)
        if fn is None:
            raise ValueError(f"Unknown product type: {product_type}. Options: {list(MAKEUP_FNS)}")

        result = fn(image, lm, color_hex, intensity)

    print(f"[makeup] Done — {product_type} applied")
    return image_to_base64(result)
