"""
AI Engine — Module 3: Virtual Accessories Try-On
Uses MediaPipe Face Mesh for landmark-based accessory positioning.
Overlays PNG accessories (glasses, hats, earrings, necklaces) onto user photo.
"""

import cv2
import numpy as np
import mediapipe as mp
import requests
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.utils.image_utils import image_to_base64, resize_image

mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose


def download_accessory_image(url):
    """Download PNG accessory with transparency"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        # Ensure BGRA
        if img is not None and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    except Exception as e:
        print(f"[accessories] Download failed: {e}")
        return None


def overlay_accessory(background, accessory, x, y, target_w, target_h):
    """
    Place accessory PNG onto background at position (x,y) with given size.
    Handles alpha blending for transparent PNG.
    """
    if accessory is None:
        return background

    # Resize accessory
    acc = cv2.resize(accessory, (target_w, target_h), interpolation=cv2.INTER_AREA)
    result = background.copy()
    bg_h, bg_w = background.shape[:2]

    # Clamp coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + target_w)
    y2 = min(bg_h, y + target_h)

    if x2 <= x1 or y2 <= y1:
        return result

    ax1 = x1 - x
    ay1 = y1 - y
    ax2 = ax1 + (x2 - x1)
    ay2 = ay1 + (y2 - y1)

    roi    = result[y1:y2, x1:x2].astype(float)
    a_crop = acc[ay1:ay2, ax1:ax2]

    if a_crop.shape[2] == 4:
        alpha = a_crop[:, :, 3:4].astype(float) / 255.0
        fg    = a_crop[:, :, :3].astype(float)
        blended = fg * alpha + roi * (1 - alpha)
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        result[y1:y2, x1:x2] = a_crop[:, :, :3]

    return result


def apply_glasses(image, face_landmarks, accessory_img):
    """Position glasses across the eye bridge"""
    h, w = image.shape[:2]
    lm   = face_landmarks.landmark

    # Use nose bridge and outer eye corners for positioning
    nose_top   = lm[6]
    left_eye   = lm[33]   # Right eye outer (from image perspective)
    right_eye  = lm[263]  # Left eye outer

    lx = int(left_eye.x  * w)
    rx = int(right_eye.x * w)
    ny = int(nose_top.y  * h)

    glasses_w = abs(rx - lx) + 60
    glasses_h = int(glasses_w * 0.4)

    x = min(lx, rx) - 30
    y = ny - glasses_h // 2

    return overlay_accessory(image, accessory_img, x, y, glasses_w, glasses_h)


def apply_hat(image, face_landmarks, accessory_img):
    """Position hat above the head"""
    h, w = image.shape[:2]
    lm   = face_landmarks.landmark

    # Forehead top landmark (approx)
    forehead = lm[10]
    left_ear = lm[234]
    right_ear= lm[454]

    lx = int(left_ear.x  * w)
    rx = int(right_ear.x * w)
    fy = int(forehead.y  * h)

    hat_w = abs(rx - lx) + 80
    hat_h = int(hat_w * 0.7)

    x = min(lx, rx) - 40
    y = fy - hat_h + 10   # Sit on top of head

    return overlay_accessory(image, accessory_img, x, y, hat_w, hat_h)


def apply_earrings(image, face_landmarks, accessory_img):
    """Position earrings at ear lobes"""
    h, w = image.shape[:2]
    lm   = face_landmarks.landmark

    # Ear lobe approximate landmarks
    left_ear_lobe  = lm[172]
    right_ear_lobe = lm[397]

    earring_w = 30
    earring_h = 50

    result = image.copy()
    for ear_lm in [left_ear_lobe, right_ear_lobe]:
        ex = int(ear_lm.x * w) - earring_w // 2
        ey = int(ear_lm.y * h)
        result = overlay_accessory(result, accessory_img, ex, ey, earring_w, earring_h)

    return result


def apply_necklace(image, face_landmarks, accessory_img):
    """Position necklace at neck/chest area"""
    h, w = image.shape[:2]
    lm   = face_landmarks.landmark

    # Chin bottom
    chin = lm[152]
    cx   = int(chin.x * w)
    cy   = int(chin.y * h)

    necklace_w = int(w * 0.45)
    necklace_h = int(necklace_w * 0.25)

    x = cx - necklace_w // 2
    y = cy + 20

    return overlay_accessory(image, accessory_img, x, y, necklace_w, necklace_h)


def apply_watch(image, accessory_img):
    """
    Position watch on wrist area.
    Uses a fixed lower-left region (simplified — full pose estimation optional).
    """
    h, w = image.shape[:2]
    watch_w = int(w * 0.15)
    watch_h = watch_w

    x = int(w * 0.15)
    y = int(h * 0.75)

    return overlay_accessory(image, accessory_img, x, y, watch_w, watch_h)


ACCESSORY_FUNCTIONS = {
    'glasses':  apply_glasses,
    'hat':      apply_hat,
    'earrings': apply_earrings,
    'necklace': apply_necklace,
}


def process_accessory(user_image_path, accessory_type, accessory_image_url):
    """
    Main function: apply accessory and return base64 result.

    Args:
        user_image_path: path to user photo
        accessory_type: 'glasses' / 'hat' / 'earrings' / 'necklace' / 'watch'
        accessory_image_url: URL of accessory PNG

    Returns:
        base64 encoded result image
    """
    print(f"[accessories] Applying {accessory_type}")

    image = cv2.imread(user_image_path)
    if image is None:
        raise ValueError("Could not load user image")

    image = resize_image(image, max_size=720)

    # Download accessory
    accessory_img = download_accessory_image(accessory_image_url)

    # Handle watch without face mesh
    if accessory_type == 'watch':
        result = apply_watch(image, accessory_img)
        return image_to_base64(result)

    # All other accessories need face detection
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected. Please upload a clear face photo.")

        face_landmarks = results.multi_face_landmarks[0]
        apply_fn = ACCESSORY_FUNCTIONS.get(accessory_type)

        if apply_fn is None:
            raise ValueError(f"Unknown accessory type: {accessory_type}")

        result = apply_fn(image, face_landmarks, accessory_img)

    print(f"[accessories] {accessory_type} applied!")
    return image_to_base64(result)
