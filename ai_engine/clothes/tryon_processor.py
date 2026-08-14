"""
AI Engine — Module 1: Virtual Clothes Try-On
=========================================================
STRATEGY:
-----------------------------------------
1. PRIMARY:  CatVTON (diffusion-based, ICLR 2025) — real garment synthesis
             via catvton_engine.py. Requires a CUDA GPU (~6GB+ VRAM).

2. FALLBACK: If CatVTON/GPU is unavailable (e.g. developing on a laptop
             without a GPU), automatically falls back to the lightweight
             MediaPipe + rembg CV-overlay below, so the app never breaks
             during local development — it just runs the simpler visual
             instead of failing.

TO INSTALL CatVTON: see ai_engine/clothes/CatVTON_Setup_Guide.md
"""

import cv2
import numpy as np
import requests
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.utils.image_utils import image_to_base64, resize_image

import mediapipe as mp
mp_pose    = mp.solutions.pose

# ─────────────────────────────────────────────────────────────
#  Attempt to import CatVTON engine (needs GPU + separate setup)
# ─────────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        # Make sure this folder (ai_engine/clothes) is on sys.path so
        # catvton_engine.py (sitting right next to this file) can be found,
        # regardless of how tryon_processor.py itself was imported.
        _clothes_dir = os.path.dirname(os.path.abspath(__file__))
        if _clothes_dir not in sys.path:
            sys.path.insert(0, _clothes_dir)

        from catvton_engine import generate_tryon as catvton_generate_tryon
        CATVTON_AVAILABLE = True
        print("[clothes] CatVTON + CUDA GPU detected — using diffusion pipeline")
    else:
        CATVTON_AVAILABLE = False
        print("[clothes] No CUDA GPU — falling back to CV-overlay mode")
except Exception as e:
    CATVTON_AVAILABLE = False
    print(f"[clothes] CatVTON not available ({e}) — falling back to CV-overlay mode")

# ─────────────────────────────────────────────────────────────
#  Attempt to import rembg (background removal via u2net ONNX)
# ─────────────────────────────────────────────────────────────
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
    print("[clothes] rembg available — using enhanced overlay mode")
except ImportError:
    REMBG_AVAILABLE = False
    print("[clothes] rembg not installed — using basic overlay (pip install rembg)")


def download_image(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"[clothes] Download failed: {e}")
        return None


def remove_background(image_bgr):
    """
    Remove garment background using u2net (rembg).
    Returns BGRA image with transparent background.
    """
    if not REMBG_AVAILABLE:
        return image_bgr

    try:
        import PIL.Image, io
        pil_img = PIL.Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result_pil = rembg_remove(pil_img)
        result_rgba = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGBA2BGRA)
        return result_rgba
    except Exception as e:
        print(f"[clothes] rembg failed, using raw image: {e}")
        return image_bgr


def detect_body_region(image):
    """Detect torso region using MediaPipe pose landmarks."""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = image.shape[:2]
        lm   = results.pose_landmarks.landmark

        def px(lmk):
            return (int(lmk.x * w), int(lmk.y * h))

        return {
            "left_shoulder":  px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]),
            "right_shoulder": px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            "left_hip":       px(lm[mp_pose.PoseLandmark.LEFT_HIP]),
            "right_hip":      px(lm[mp_pose.PoseLandmark.RIGHT_HIP]),
        }


def tps_warp_garment(garment, src_pts, dst_pts, output_shape):
    """
    Thin Plate Spline warping — bends garment to fit body contour.
    Falls back to affine warp if TPS not available.
    """
    try:
        from cv2 import createThinPlateSplineShapeTransformer
        tps = createThinPlateSplineShapeTransformer()
        src = np.array(src_pts, dtype=np.float32).reshape(1, -1, 2)
        dst = np.array(dst_pts, dtype=np.float32).reshape(1, -1, 2)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        tps.estimateTransformation(dst, src, matches)
        warped = tps.warpImage(garment)
        return cv2.resize(warped, (output_shape[1], output_shape[0]))
    except Exception:
        # Simple resize fallback
        return cv2.resize(garment, (output_shape[1], output_shape[0]))


def overlay_garment_enhanced(user_image, garment_image, body_landmarks):
    """
    Enhanced overlay: removes garment background, then blends onto torso.
    """
    result = user_image.copy()
    h, w   = result.shape[:2]

    ls = body_landmarks["left_shoulder"]
    rs = body_landmarks["right_shoulder"]
    lh = body_landmarks["left_hip"]
    rh = body_landmarks["right_hip"]

    # Compute bounding box with padding
    pad_x = int((max(ls[0], rs[0]) - min(ls[0], rs[0])) * 0.15)
    pad_y_top = 15
    pad_y_bot = 25

    x1 = max(0, min(ls[0], rs[0]) - pad_x)
    x2 = min(w, max(ls[0], rs[0]) + pad_x)
    y1 = max(0, min(ls[1], rs[1]) - pad_y_top)
    y2 = min(h, max(lh[1], rh[1]) + pad_y_bot)

    garment_w = x2 - x1
    garment_h = y2 - y1
    if garment_w <= 0 or garment_h <= 0:
        return result

    # Remove background from garment
    garment_clean = remove_background(garment_image)

    # Warp/resize garment to fit torso region
    garment_resized = cv2.resize(garment_clean, (garment_w, garment_h))

    roi = result[y1:y2, x1:x2]

    if garment_resized.ndim == 3 and garment_resized.shape[2] == 4:
        # Has alpha channel — clean blend
        alpha = garment_resized[:, :, 3:4].astype(float) / 255.0
        garment_rgb = garment_resized[:, :, :3].astype(float)
        roi_f = roi.astype(float)
        blended = (garment_rgb * alpha + roi_f * (1 - alpha)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended
    else:
        # No alpha — use weighted blend
        result[y1:y2, x1:x2] = cv2.addWeighted(
            garment_resized[:, :, :3], 0.88, roi, 0.12, 0
        )

    return result


def process_clothes_tryon(user_image_path, garment_image_url, product_name="", cloth_type="upper"):
    """
    Main function: process clothes try-on and return base64 result.

    cloth_type: "upper" | "lower" | "overall" — only used by CatVTON path,
                ignored by the CV-overlay fallback (which always treats it
                as upper-body torso overlay).
    """
    print(f"[clothes] Processing: {product_name}")

    user_img = cv2.imread(user_image_path)
    if user_img is None:
        raise ValueError("Could not load user image")

    user_img = resize_image(user_img, max_size=720)

    garment_img = download_image(garment_image_url)
    if garment_img is None:
        cv2.putText(user_img, f"[{product_name}]", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 120, 255), 2)
        return image_to_base64(user_img)

    # ── Try CatVTON first (real diffusion-based synthesis) ──────────────
    if CATVTON_AVAILABLE:
        try:
            result_pil = catvton_generate_tryon(user_img, garment_img, cloth_type=cloth_type)
            result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            print("[clothes] Try-on complete — CatVTON (diffusion)")
            return image_to_base64(result_bgr)
        except Exception as e:
            print(f"[clothes] CatVTON failed ({e}) — falling back to CV-overlay")

    # ── Fallback: MediaPipe + rembg CV-overlay ───────────────────────────
    body_landmarks = detect_body_region(user_img)

    if body_landmarks is None:
        print("[clothes] No pose detected — using default region")
        h, w = user_img.shape[:2]
        body_landmarks = {
            "left_shoulder":  (int(w * 0.28), int(h * 0.18)),
            "right_shoulder": (int(w * 0.72), int(h * 0.18)),
            "left_hip":       (int(w * 0.30), int(h * 0.60)),
            "right_hip":      (int(w * 0.70), int(h * 0.60)),
        }

    result = overlay_garment_enhanced(user_img, garment_img, body_landmarks)

    # Subtle watermark
    cv2.putText(result, f"AI-FIT: {product_name}", (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    method = "Enhanced (rembg)" if REMBG_AVAILABLE else "Basic Overlay"
    print(f"[clothes] Try-on complete — {method}")
    return image_to_base64(result)