r"""
AI Engine — Module 1: CatVTON Diffusion Pipeline Wrapper
=========================================================
Wraps the CatVTON diffusion model (ICLR 2025) for local inference.

IMPORTANT DESIGN DECISION:
CatVTON's official demo (app.py) uses AutoMasker (DensePose + SCHP) to
generate masks automatically — this needs extra heavy checkpoints and
GPU memory. Their own code confirms a manually-supplied mask is accepted
and takes PRIORITY over AutoMasker. We supply our own mask (built from
MediaPipe pose landmarks in mask_generator.py), so DensePose/SCHP are
never loaded. This significantly reduces setup complexity and VRAM use.

MODEL REQUIREMENTS:
- Base model:  runwayml/stable-diffusion-inpainting  (downloads automatically)
- Attn weights: zhengchong/CatVTON  (downloads automatically via snapshot_download)
- ~6GB+ VRAM recommended (use --mixed_precision fp16 on 6GB cards)

SETUP (one-time, see CatVTON_Setup_Guide.md for full details):
    pip install diffusers transformers accelerate huggingface_hub
    git clone https://github.com/Zheng-Chong/CatVTON.git

HOW THIS FILE FINDS THE CLONED CatVTON REPO:
    1. Environment variable CATVTON_REPO_PATH, if set (highest priority)
       e.g. (PowerShell):  $env:CATVTON_REPO_PATH = "E:\path\to\CatVTON"
    2. A few common relative locations next to this project (auto-tried)
    If none of these actually contain the repo, this file raises a clear
    error telling you exactly what it tried and how to fix it — it will
    NOT silently fall through to an unrelated "utils" package.
"""

import os
import sys
import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────
#  Locate the cloned CatVTON repo (contains `model/` and `utils.py`)
# ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_CANDIDATE_PATHS = []
if os.environ.get("CATVTON_REPO_PATH"):
    _CANDIDATE_PATHS.append(os.environ["CATVTON_REPO_PATH"])

# Common locations relative to ai_engine/clothes/ (this file's folder)
_CANDIDATE_PATHS += [
    os.path.join(_THIS_DIR, "CatVTON"),                    # ai_engine/clothes/CatVTON
    os.path.join(_THIS_DIR, "..", "..", "CatVTON"),         # project_root/CatVTON
    os.path.join(_THIS_DIR, "..", "CatVTON"),               # ai_engine/CatVTON
    os.path.join(_THIS_DIR, "..", "..", "..", "CatVTON"),   # one level above project root
]


def _find_catvton_repo():
    """Return the first candidate path that actually looks like the CatVTON repo."""
    tried = []
    for path in _CANDIDATE_PATHS:
        abs_path = os.path.abspath(path)
        tried.append(abs_path)
        # A real CatVTON clone has a `model/` folder and a `utils.py` file at its root
        if os.path.isdir(os.path.join(abs_path, "model")) and \
           os.path.isfile(os.path.join(abs_path, "utils.py")):
            return abs_path

    tried_list = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        "\n\nCould not find the CatVTON repo (needs a 'model/' folder + 'utils.py' inside it).\n"
        f"Tried these locations:\n{tried_list}\n\n"
        "FIX — do ONE of these:\n"
        "  1) Clone it if you haven't yet:\n"
        "       git clone https://github.com/Zheng-Chong/CatVTON.git\n"
        "     (clone it into your project root, next to 'backend' and 'ai_engine')\n"
        "  2) OR set the environment variable to wherever you already cloned it:\n"
        "       PowerShell:  $env:CATVTON_REPO_PATH = \"E:\\full\\path\\to\\CatVTON\"\n"
    )


_CATVTON_REPO = _find_catvton_repo()
if _CATVTON_REPO not in sys.path:
    sys.path.insert(0, _CATVTON_REPO)   # so `from model.pipeline import ...` and `from utils import ...` resolve here first

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)       # so our own mask_generator.py is importable

from mask_generator import extract_landmarks, generate_garment_mask, visualize_mask

# Lazy-loaded globals — the model is loaded once on first use, not on import,
# so importing this file elsewhere (e.g. Flask app startup) stays fast and
# doesn't require a GPU just to boot the server.
_pipeline = None
_mask_processor = None


def _load_pipeline(mixed_precision="fp16", device="cuda"):
    """
    Load the CatVTON pipeline once and cache it. Downloads checkpoints
    from Hugging Face automatically on first run (~5-10GB, one-time).
    """
    global _pipeline, _mask_processor

    if _pipeline is not None:
        return _pipeline

    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor
    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype

    print("[catvton] Downloading/loading checkpoints (first run only, ~5-10GB)...")
    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    _pipeline = CatVTONPipeline(
        base_ckpt="runwayml/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(mixed_precision),
        use_tf32=True,
        device=device,
    )

    _mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    print("[catvton] Pipeline ready.")
    return _pipeline


def generate_tryon(
    person_image_bgr,
    garment_image_bgr,
    cloth_type="upper",
    width=768,
    height=1024,
    num_inference_steps=45,
    guidance_scale=4.0,
    seed=42,
    mixed_precision="fp16",
    device="cuda",
):
    """
    Main entry point: person photo + garment photo -> try-on result.

    Args:
        person_image_bgr:  OpenCV BGR array (user's uploaded photo)
        garment_image_bgr: OpenCV BGR array (product/catalog image)
        cloth_type: "upper" | "lower" | "overall"
        width/height: CatVTON works best at 768x1024 (paper default).
                      Drop to 512x680 if you hit CUDA out-of-memory on 6GB.
        num_inference_steps: fewer steps = faster but lower quality
                              (20-30 is a good speed/quality tradeoff for a demo)
        guidance_scale: how strongly the garment image is followed (2.0-3.0 typical)

    Returns:
        PIL.Image — the try-on result (RGB)
    """
    from utils import resize_and_crop, resize_and_padding
    import numpy as np
    import cv2

    pipeline = _load_pipeline(mixed_precision=mixed_precision, device=device)

    # Convert OpenCV BGR -> PIL RGB
    person_pil_full = Image.fromarray(cv2.cvtColor(person_image_bgr, cv2.COLOR_BGR2RGB))
    garment_pil_full = Image.fromarray(cv2.cvtColor(garment_image_bgr, cv2.COLOR_BGR2RGB))

    # ── Resize FIRST, then build the mask on the ALREADY-resized image ──
    # (Bug fix: previously the mask was built on the original-size photo
    # then separately .resize()'d to match — but resize_and_crop() also
    # CROPS, so the mask ended up misaligned with the actual photo pixels.
    # Building the mask after resize_and_crop guarantees pixel-perfect
    # alignment between the mask and the image CatVTON actually sees.)
    person_pil = resize_and_crop(person_pil_full, (width, height))
    garment_pil = resize_and_padding(garment_pil_full, (width, height))

    person_resized_bgr = cv2.cvtColor(np.array(person_pil), cv2.COLOR_RGB2BGR)
    landmarks = extract_landmarks(person_resized_bgr)
    mask = generate_garment_mask(person_resized_bgr, cloth_type=cloth_type, landmarks=landmarks)

    # Sanity check — if the mask is basically empty, inpainting will look
    # like "nothing changed" (which is exactly the symptom this fixes).
    mask_coverage = (np.array(mask) > 0).mean()
    print(f"[catvton] Mask covers {mask_coverage:.1%} of the image")
    if mask_coverage < 0.15:
        print("[catvton] WARNING: mask looks too small — pose detection may have "
              "failed or landmarks are misplaced. Check mask_debug.png.")

    # Always save a debug overlay so misalignment is obvious at a glance,
    # instead of only being visible as "nothing changed" in the final result.
    try:
        debug_overlay = visualize_mask(person_resized_bgr, mask)
        debug_path = os.path.join(_THIS_DIR, "mask_debug.png")
        cv2.imwrite(debug_path, debug_overlay)
        print(f"[catvton] Saved mask debug overlay: {debug_path}")
    except Exception as _e:
        pass

    mask = _mask_processor.blur(mask, blur_factor=9)

    generator = torch.Generator(device=device).manual_seed(seed) if seed != -1 else None

    print(f"[catvton] Running inference (steps={num_inference_steps}, cloth_type={cloth_type})...")
    result = pipeline(
        image=person_pil,
        condition_image=garment_pil,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    print("[catvton] Done.")
    return result


if __name__ == "__main__":
    # Quick manual test — run this file directly on the GPU machine:
    #   python catvton_engine.py path/to/person.jpg path/to/garment.jpg upper
    import cv2
    import sys as _sys

    if len(_sys.argv) < 3:
        print("Usage: python catvton_engine.py <person.jpg> <garment.jpg> [upper|lower|overall]")
        _sys.exit(1)

    person_path = _sys.argv[1].strip()
    garment_path = _sys.argv[2].strip()
    cloth_type = _sys.argv[3].strip() if len(_sys.argv) > 3 else "upper"

    print(f"[catvton] Using CatVTON repo at: {_CATVTON_REPO}")
    print(f"[catvton] Person image:  {person_path}")
    print(f"[catvton] Garment image: {garment_path}")
    print(f"[catvton] Cloth type:    {cloth_type}")

    person = cv2.imread(person_path)
    if person is None:
        print(f"ERROR: could not read person image at: {person_path}")
        _sys.exit(1)

    garment = cv2.imread(garment_path)
    if garment is None:
        print(f"ERROR: could not read garment image at: {garment_path}")
        _sys.exit(1)

    result = generate_tryon(person, garment, cloth_type=cloth_type)
    out_path = os.path.join(_THIS_DIR, "catvton_test_result.png")
    result.save(out_path)
    print(f"Saved: {out_path}")