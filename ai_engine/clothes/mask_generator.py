"""
AI Engine — Module 1 Helper: Garment Mask Generator  (FIXED VERSION)
=========================================================
Generates a binary mask (white = region to replace with new garment,
black = keep original) using MediaPipe pose landmarks.

WHAT WAS WRONG BEFORE:
The old polygon connected shoulder points (top) directly to hip points
(bottom): [ (L-shoulder, top_y), (R-shoulder, top_y), (R-hip, bottom_y),
(L-hip, bottom_y) ].

MediaPipe's LEFT_HIP/RIGHT_HIP landmarks sit on the pelvis bone, which is
naturally much narrower (closer to the body's centerline) than the actual
torso width at that height — especially with a loose/buttoned shirt. So
the mask ended up shaped like a narrow "V" / funnel: wide at the shoulders,
pinching almost to a point near the waist (see mask_debug.png). That mask
never covered the sleeves or the sides of the torso at all.

Effect on CatVTON: the diffusion model only had a thin sliver to repaint.
With blur_factor=9 softening the mask edges further and a modest
guidance_scale, the model just reconstructed something that blended into
the surrounding (unchanged) shirt — so the final result looked almost
identical to the input photo, even though a different garment image was
supplied. That matches catvton_test_result.png exactly.

THE FIX:
Instead of tapering from shoulder-width to hip-width, we use a single
consistent body_width = max(shoulder_width, hip_width), padded generously,
and draw a rectangle-like region (slightly flared at the shoulder line to
also catch the sleeve caps) spanning from the collar down past the hip
line. This mimics how CatVTON's own AutoMasker/VITON-HD "agnostic mask"
works — generous and roughly constant-width, not a pinched joint-to-joint
polygon. A slightly-too-generous mask is far safer for diffusion inpainting
than a too-narrow one.

Supports three cloth types (matches CatVTON's expected categories):
    "upper"   - shirts, t-shirts, jackets (shoulders to hips)
    "lower"   - trousers, jeans, skirts (hips to ankles)
    "overall" - dresses, full outfits (shoulders to ankles)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

mp_pose = mp.solutions.pose


def extract_landmarks(image_bgr):
    """
    Run MediaPipe Pose on a person image and return pixel-coordinate
    landmarks needed for masking. Returns None if no person detected.
    """
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = image_bgr.shape[:2]
        lm = results.pose_landmarks.landmark

        def px(landmark_id):
            p = lm[landmark_id]
            return (int(p.x * w), int(p.y * h))

        P = mp_pose.PoseLandmark
        return {
            "left_shoulder":  px(P.LEFT_SHOULDER),
            "right_shoulder": px(P.RIGHT_SHOULDER),
            "left_elbow":     px(P.LEFT_ELBOW),
            "right_elbow":    px(P.RIGHT_ELBOW),
            "left_hip":       px(P.LEFT_HIP),
            "right_hip":      px(P.RIGHT_HIP),
            "left_knee":      px(P.LEFT_KNEE),
            "right_knee":     px(P.RIGHT_KNEE),
            "left_ankle":     px(P.LEFT_ANKLE),
            "right_ankle":    px(P.RIGHT_ANKLE),
            "nose":           px(P.NOSE),
        }


def _default_landmarks(w, h):
    """Fallback landmarks (rough estimate) when pose detection fails."""
    return {
        "left_shoulder":  (int(w * 0.28), int(h * 0.18)),
        "right_shoulder": (int(w * 0.72), int(h * 0.18)),
        "left_elbow":     (int(w * 0.20), int(h * 0.38)),
        "right_elbow":    (int(w * 0.80), int(h * 0.38)),
        "left_hip":       (int(w * 0.30), int(h * 0.55)),
        "right_hip":      (int(w * 0.70), int(h * 0.55)),
        "left_knee":      (int(w * 0.32), int(h * 0.75)),
        "right_knee":     (int(w * 0.68), int(h * 0.75)),
        "left_ankle":     (int(w * 0.33), int(h * 0.95)),
        "right_ankle":    (int(w * 0.67), int(h * 0.95)),
        "nose":           (int(w * 0.50), int(h * 0.08)),
    }


def generate_garment_mask(image_bgr, cloth_type="upper", landmarks=None, padding_ratio=0.22):
    """
    Build a binary garment mask for CatVTON.

    Args:
        image_bgr: person photo (OpenCV BGR array)
        cloth_type: "upper" | "lower" | "overall"
        landmarks: optional pre-computed landmarks dict (skips re-detection
                   if you already called extract_landmarks once)
        padding_ratio: how much extra room to give around the body region
                       (helps CatVTON blend edges naturally). Raised from
                       0.12 -> 0.22: the old value was too tight even
                       before accounting for the taper bug.

    Returns:
        PIL.Image (mode "L", grayscale) — white=inpaint region, black=keep.
        Same width/height as the input image.
    """
    h, w = image_bgr.shape[:2]

    if landmarks is None:
        landmarks = extract_landmarks(image_bgr)
    if landmarks is None:
        print("[mask] No pose detected — using default region estimate")
        landmarks = _default_landmarks(w, h)

    ls, rs = landmarks["left_shoulder"], landmarks["right_shoulder"]
    le, re = landmarks["left_elbow"], landmarks["right_elbow"]
    lh, rh = landmarks["left_hip"], landmarks["right_hip"]
    lk, rk = landmarks["left_knee"], landmarks["right_knee"]
    la, ra = landmarks["left_ankle"], landmarks["right_ankle"]

    shoulder_width = abs(rs[0] - ls[0])
    hip_width = abs(rh[0] - lh[0])
    # KEY FIX: use ONE consistent width for the whole torso instead of
    # tapering from shoulder points down to hip points. A shirt does not
    # pinch in to the width of the pelvis bone — it stays roughly as wide
    # as the shoulders/ribcage all the way down.
    body_width = max(shoulder_width, hip_width, 1)
    pad = int(body_width * padding_ratio)

    center_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if cloth_type == "upper":
        top_y = min(ls[1], rs[1]) - int(shoulder_width * 0.30)          # up to the collar
        bottom_y = max(lh[1], rh[1]) + int(body_width * 0.35)           # past the hip, to the hem
        left_x = center_x - body_width / 2 - pad
        right_x = center_x + body_width / 2 + pad
        # Main torso column — constant width, full height.
        draw.rounded_rectangle([left_x, top_y, right_x, bottom_y],
                                radius=int(body_width * 0.15), fill=255)

        # Sleeve caps: only widen near the shoulder/chest band (top ~42%
        # of the torso) to also catch short sleeves. Do NOT widen the
        # full height — a hand resting near the hip/pocket must stay
        # OUTSIDE the mask, or it gets redrawn/hallucinated by the
        # diffusion model along with the garment (this was happening
        # before: the tucked-in hand was inside the mask and got
        # regenerated, changing its pose/shape in the result).
        if le and re:
            sleeve_zone_bottom = top_y + int((bottom_y - top_y) * 0.42)
            sleeve_left = min(left_x, le[0] - pad * 0.5)
            sleeve_right = max(right_x, re[0] + pad * 0.5)
            draw.rectangle([sleeve_left, top_y, sleeve_right, sleeve_zone_bottom], fill=255)

    elif cloth_type == "lower":
        # Pants genuinely do taper hip -> ankle, so the trapezoid shape is
        # fine here — just give it more padding than before.
        top_y = min(lh[1], rh[1]) - pad
        bottom_y = max(la[1], ra[1]) + pad
        polygon = [
            (lh[0] - pad, top_y),
            (rh[0] + pad, top_y),
            (ra[0] + pad, bottom_y),
            (la[0] - pad, bottom_y),
        ]
        draw.polygon(polygon, fill=255)

    elif cloth_type == "overall":
        top_y = min(ls[1], rs[1]) - int(shoulder_width * 0.30)
        bottom_y = max(la[1], ra[1]) + pad
        left_x = center_x - body_width / 2 - pad
        right_x = center_x + body_width / 2 + pad
        draw.rounded_rectangle([left_x, top_y, right_x, bottom_y],
                                radius=int(body_width * 0.15), fill=255)

    else:
        raise ValueError(f"Unknown cloth_type: {cloth_type} (use upper/lower/overall)")

    mask = mask.crop((0, 0, w, h))
    return mask


def visualize_mask(image_bgr, mask_pil, alpha=0.5):
    """
    Debug helper: overlay the mask in red on top of the person image so you
    can visually confirm the region looks right before running CatVTON.
    Returns an OpenCV BGR image.
    """
    overlay = image_bgr.copy()
    mask_np = np.array(mask_pil)
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255  # BGR -> red channel
    blended = np.where(mask_np[..., None] > 0,
                        cv2.addWeighted(overlay, 1 - alpha, red, alpha, 0),
                        overlay)
    return blended