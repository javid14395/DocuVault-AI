"""
DocuVault AI - Built-in Document Studio (studio.py)

Provides document enhancement, formatting, and export utilities:
1. Auto-Perspective Crop ("CamScanner Mode"):
   Detects the 4 corners of an ID card in a cluttered photo, flattens perspective,
   and applies adaptive contrast enhancement.
2. Portal Size Resizer (<50KB / <100KB / <200KB):
   Iteratively performs binary search on JPEG quality and resolution downscaling
   to guarantee an output image strictly under the requested target byte limit.
3. A4 Printable Sheet Generator:
   Neatly places 1 or 2 document cards (e.g., Aadhaar Front and Back) onto a standard
   A4-sized 300 DPI canvas and exports a ready-to-print PDF.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger("docuvault.studio")

BASE_DIR = Path(__file__).resolve().parent
STUDIO_DIR = BASE_DIR / "data" / "studio"
STUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Standard A4 size in pixels at 300 DPI
A4_WIDTH_300DPI = 2480
A4_HEIGHT_300DPI = 3508


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order a list of 4 coordinates: [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perform a perspective warp on `image` given 4 corner points.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - bl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    max_w = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b))

    max_w = max(max_w, 200)
    max_h = max(max_h, 200)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_w, max_h), flags=cv2.INTER_CUBIC)
    return warped


def auto_perspective_crop(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    enhance: bool = True,
) -> Tuple[Image.Image, bool]:
    """
    Detects the 4 corners of an ID card in a cluttered photo, flattens the perspective,
    and applies adaptive contrast enhancement (CamScanner mode).

    Returns `(enhanced_pil_image, quad_detected_boolean)`.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(str(image_input))
        if not p.is_file():
            raise ValueError(f"Image file not found: {image_input}")
        try:
            with Image.open(p) as opened:
                rgb_arr = np.array(ImageOps.exif_transpose(opened).convert("RGB"))
                img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Could not read image from {image_input}: {exc}") from exc
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_h, orig_w = img_bgr.shape[:2]
    quad_detected = False

    # Work on a resized copy for fast contour detection
    scale = 800.0 / float(max(orig_h, orig_w))
    if scale < 1.0:
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr.copy()
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 160)

    # Dilate edges slightly to close small gaps in the boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    target_quad = None
    min_area = (small.shape[0] * small.shape[1]) * 0.12  # At least 12% of total image

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            # Check convexity
            if cv2.isContourConvex(approx):
                target_quad = approx.reshape(4, 2)
                quad_detected = True
                break

    if target_quad is not None:
        # Scale corners back to original image dimensions
        orig_pts = (target_quad / scale).astype("float32")
        warped_bgr = four_point_transform(img_bgr, orig_pts)
    else:
        # Fallback: simple auto-crop bounding box if edges found, else keep full image
        warped_bgr = img_bgr.copy()

    # Orientation normalization: ID cards are typically landscape (width > height)
    wh_ratio = warped_bgr.shape[1] / float(warped_bgr.shape[0])
    if wh_ratio < 0.75:  # Sideways portrait
        warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_90_CLOCKWISE)

    # Adaptive Contrast Enhancement (CLAHE on L-channel of LAB space)
    if enhance:
        lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        warped_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    final_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    pil_res = Image.fromarray(final_rgb)

    # Slight sharpness boost for crisp text
    if enhance:
        enhancer = ImageEnhance.Sharpness(pil_res)
        pil_res = enhancer.enhance(1.25)

    return pil_res, quad_detected


def resize_for_portal(
    image_input: Union[str, Path, Image.Image],
    max_kb: int = 100,
    output_format: str = "JPEG",
) -> Tuple[bytes, int, Tuple[int, int]]:
    """
    Compresses and rescales an image using binary search on JPEG quality and resolution
    to ensure the output file is strictly <= max_kb (e.g., 50KB, 100KB, 200KB)
    for government/banking portal uploads.

    Returns `(compressed_bytes, size_in_bytes, (width, height))`.
    """
    target_bytes = int(max_kb) * 1024

    if isinstance(image_input, (str, Path)):
        p = Path(str(image_input))
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {image_input}")
        with Image.open(p) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = ImageOps.exif_transpose(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    cur_w, cur_h = img.size

    best_bytes: Optional[bytes] = None

    # Step 1: Binary search on quality at current resolution
    low_q, high_q = 20, 95
    while low_q <= high_q:
        mid_q = (low_q + high_q) // 2
        buf = io.BytesIO()
        img.save(buf, format=output_format, quality=mid_q, optimize=True)
        size = buf.tell()

        if size <= target_bytes:
            best_bytes = buf.getvalue()
            low_q = mid_q + 1  # Try higher quality
        else:
            high_q = mid_q - 1

    if best_bytes is not None:
        return best_bytes, len(best_bytes), img.size

    # Step 2: If quality alone is not enough (e.g. huge megapixel image for 50KB),
    # iteratively scale down resolution by 15% and re-attempt binary search
    current_img = img
    for _ in range(8):
        cur_w = max(400, int(cur_w * 0.85))
        cur_h = max(250, int(cur_h * 0.85))
        current_img = img.resize((cur_w, cur_h), Image.Resampling.LANCZOS)

        low_q, high_q = 25, 90
        while low_q <= high_q:
            mid_q = (low_q + high_q) // 2
            buf = io.BytesIO()
            current_img.save(buf, format=output_format, quality=mid_q, optimize=True)
            size = buf.tell()

            if size <= target_bytes:
                best_bytes = buf.getvalue()
                low_q = mid_q + 1
            else:
                high_q = mid_q - 1

        if best_bytes is not None:
            return best_bytes, len(best_bytes), current_img.size

    # Hard fallback: save at minimum acceptable quality
    buf = io.BytesIO()
    current_img.save(buf, format=output_format, quality=25, optimize=True)
    return buf.getvalue(), buf.tell(), current_img.size


def generate_a4_printable_sheet(
    image_inputs: Sequence[Union[str, Path, Image.Image]],
    title: Optional[str] = None,
    output_pdf_path: Optional[Union[str, Path]] = None,
) -> Tuple[bytes, str]:
    """
    Places 1 or 2 document cards (e.g. Aadhaar Front & Back) neatly on a standard
    A4 canvas (300 DPI: 2480 x 3508) ready for printing.

    Returns `(pdf_bytes, output_file_path)`.
    """
    if not image_inputs:
        raise ValueError("At least one document image is required for A4 print generation.")

    sheet = Image.new("RGB", (A4_WIDTH_300DPI, A4_HEIGHT_300DPI), (255, 255, 255))

    loaded_images: List[Image.Image] = []
    for item in image_inputs[:2]:
        if isinstance(item, (str, Path)):
            p = Path(str(item))
            if not p.is_file():
                continue
            with Image.open(p) as opened:
                im = ImageOps.exif_transpose(opened).convert("RGB")
        elif isinstance(item, Image.Image):
            im = ImageOps.exif_transpose(item).convert("RGB")
        else:
            continue
        loaded_images.append(im)

    if not loaded_images:
        raise ValueError("None of the specified document images could be opened for A4 sheet.")

    # Standard ID card print dimension on A4: ~1000px width (approx 3.37 inches / 85.6 mm at 300 DPI)
    max_card_w = 1200
    max_card_h = 760

    if len(loaded_images) == 1:
        card = loaded_images[0]
        card.thumbnail((max_card_w, max_card_h), Image.Resampling.LANCZOS)
        # Center card on upper half of A4 page (standard photocopy placement)
        x = (A4_WIDTH_300DPI - card.width) // 2
        y = (A4_HEIGHT_300DPI // 3) - (card.height // 2)

        # Draw light dashed bounding border for scissors guide
        sheet.paste(card, (x, y))

    elif len(loaded_images) >= 2:
        card1 = loaded_images[0]
        card2 = loaded_images[1]
        card1.thumbnail((max_card_w, max_card_h), Image.Resampling.LANCZOS)
        card2.thumbnail((max_card_w, max_card_h), Image.Resampling.LANCZOS)

        # Top card (Front)
        x1 = (A4_WIDTH_300DPI - card1.width) // 2
        y1 = 600
        sheet.paste(card1, (x1, y1))

        # Bottom card (Back)
        x2 = (A4_WIDTH_300DPI - card2.width) // 2
        y2 = y1 + card1.height + 200
        sheet.paste(card2, (x2, y2))

    buf = io.BytesIO()
    sheet.save(buf, format="PDF", resolution=300.0)
    pdf_bytes = buf.getvalue()

    out_path = ""
    if output_pdf_path is not None:
        target = Path(output_pdf_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pdf_bytes)
        out_path = str(target)

    return pdf_bytes, out_path
