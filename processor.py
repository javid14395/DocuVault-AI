"""
DocuVault AI - Visual Document Inspector, OCR Classifier & Face Grouping Engine (processor.py)

CRITICAL DESIGN RULE:
Never inspect or rely on filenames (`IMG_20250101.jpg`, `WhatsApp Image 2024.jpeg`, `scan001.png`).
Every candidate image and PDF is opened and visually analyzed offline:
  - PDF Page 1 rendering via `pypdfium2`
  - Step A: Dimension pre-check (skip tiny images < 300x300 px)
  - Step B: Offline OCR via `RapidOCR` (`rapidocr_onnxruntime`)
  - Step C: Regex & weighted keyword content classification + ID number extraction
  - Step D: Face detection, 20%-margin face crop (`data/faces/`), normalized LBP+HOG+Color
            embedding clustering against `people`, and document thumbnail generation (`data/thumbnails/`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import site
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Ensure NVIDIA CUDA & cuDNN DLL search paths are registered before loading onnxruntime
import face_engine
from face_engine import compute_cosine_similarity, get_insightface_app  # noqa: F401 (re-exported)
from face_engine import init_nvidia_dll_paths as _init_nvidia_dll_paths

_init_nvidia_dll_paths()

import numpy as np
from PIL import Image, ImageOps

import database
from scanner import extract_exif_date_taken

logger = logging.getLogger("docuvault.processor")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
MODELS_DIR = BASE_DIR / "models"

MIN_IMAGE_WIDTH = 300
MIN_IMAGE_HEIGHT = 300
MIN_OCR_TEXT_LENGTH = 15
MAX_OCR_IMAGE_DIMENSION = 1024
THUMBNAIL_MAX_SIZE = (600, 600)
FACE_THUMBNAIL_MAX_SIZE = (256, 256)
FACE_CROP_MARGIN_RATIO = 0.20

# Similarity threshold for independently standardized spatial + Gradientface + LBP + chroma embeddings
FACE_MATCH_THRESHOLD = 0.72
MIN_DOCUMENT_CLASSIFICATION_SCORE = 8

# Similarity threshold for InsightFace clustering: lowered from 0.50 to 0.38
# Tolerates accessories (sunglasses, glasses), varied lighting, and angles
INSIGHTFACE_CLUSTER_THRESHOLD = face_engine.INITIAL_CLUSTERING_THRESHOLD
_CLUSTER_LOCK = threading.Lock()

# NOTE: InsightFace init, CUDA providers, embeddings & cosine similarity live in face_engine.py

# -----------------------------------------------------------------------------
# Regex Patterns for Indian Identity & Official Documents (Visual OCR Text Only)
# -----------------------------------------------------------------------------

# PAN Card: 5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F)
RE_PAN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")

# Full 12-digit Aadhaar (not part of a 16-digit VID on the same line or longer numeric string)
RE_AADHAAR_FULL = re.compile(
    r"(?<!\d{4} )(?<!\d{4}-)(?<!\d)\b(\d{4}[ -]?\d{4}[ -]?\d{4})\b(?![ -]?\d)"
)
# Spaced 4-4-4 12-digit Aadhaar (strongest structural indicator, e.g. "1234 5678 9012")
RE_AADHAAR_SPACED = re.compile(
    r"(?<!\d{4} )(?<!\d{4}-)(?<!\d)\b(\d{4}[ ]+\d{4}[ ]+\d{4})\b(?![ ]+\d)"
)
# Masked Aadhaar (e.g. "XXXX XXXX 1234" or "**** **** 1234")
RE_AADHAAR_MASKED = re.compile(
    r"\b((?:[Xx\*]{4}[\s\-]?){2}\d{4})\b"
)
# 16-digit Aadhaar Virtual ID (VID)
RE_AADHAAR_VID = re.compile(
    r"VID\s*[:\-]?\s*(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b",
    re.IGNORECASE,
)

# Aadhaar Fast-Track Cascading Filter Definitions
# Gate 1 Master Keys: "GOVERNMENT" (for front) or "UIDAI" (for back) - case-insensitive
GATE1_AADHAAR_MASTER_KEYS: Tuple[str, ...] = ("GOVERNMENT", "UIDAI")

# Gate 2 Secondary Verification regex: \d{4}\s\d{4}\s\d{4}
RE_AADHAAR_444 = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")

# Driving License Number (e.g. MH14 20110062821, MH-14-2011-0062821, DL-0420110149646)
RE_DRIVING_LICENSE = re.compile(
    r"\b([A-Z]{2}[\-\s]?\d{2}[\-\s]?(?:19|20)\d{2}[\-\s]?\d{6,8}|[A-Z]{2}\d{2}\s?\d{11})\b"
)
RE_DL_LABELED = re.compile(
    r"(?:DL\s*(?:No|Number)|Licence\s*No|License\s*No)\.?\s*[:\-]?\s*([A-Z0-9\-\/\s]{8,22})",
    re.IGNORECASE,
)

# Voter ID (EPIC Number: 3 letters + 7 digits, e.g. ABC1234567)
RE_VOTER_EPIC = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")

# Indian Passport Number (1 letter + 7 digits, e.g. Z1234567 or J8362914)
RE_PASSPORT_NO = re.compile(r"\b([A-PR-WY][0-9]{7}|[A-Z][1-9][0-9]{6})\b")
RE_PASSPORT_MRZ = re.compile(r"P<IND[A-Z<]{5,}")

# Certificate / Marksheet Roll or Certificate Number
RE_CERT_NUMBER = re.compile(
    r"(?:Certificate\s*No|Roll\s*No|Registration\s*No|Reg\.?\s*No|Enrollment\s*No|Enrolment\s*No)"
    r"\.?\s*[:\-]?\s*([A-Z0-9\/\-]{4,22})",
    re.IGNORECASE,
)

# Admit Card / Hall Ticket Number
RE_ADMIT_NUMBER = re.compile(
    r"(?:Confirmation\s*(?:No|Number|#)|Roll\s*(?:No|Number|#)|Seat\s*(?:No|Number|#)|Ticket\s*(?:No|Number|#)|Registration\s*(?:No|Number|#))"
    r"\.?\s*[:\-]?\s*([A-Z0-9\/\-]{4,24})",
    re.IGNORECASE,
)

# Bill / Receipt / GSTIN / Invoice Number
RE_GSTIN = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
RE_INVOICE_NO = re.compile(
    r"(?:Invoice\s*(?:No|Number|#)|Receipt\s*(?:No|Number|#)|Bill\s*(?:No|Number|#))"
    r"\.?\s*[:\-]?\s*([A-Z0-9\/\-]{3,24})",
    re.IGNORECASE,
)

# Date of Birth (DOB) - Labeled
RE_DOB_LABELED = re.compile(
    r"(?:(?:Date\s*of\s*Birth|Birth\s*Date|D\.?O\.?B\.?|YOB|Year\s*of\s*Birth|जन्म\s*तारीख|जन्म\s*तिथि)"
    r"[\s\:\-\.\/]*)"
    r"([0-3]?[0-9][\/\-\.][0-1]?[0-9][\/\-\.](?:19|20)\d{2}|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# Generic standalone date: DD/MM/YYYY or DD-MM-YYYY
RE_GENERIC_DATE = re.compile(
    r"\b([0-3][0-9][\/\-\.][0-1][0-9][\/\-\.](?:19|20)\d{2})\b"
)

# Document Expiry Date (Driving License, Passport, ID validity)
RE_EXPIRY_LABELED = re.compile(
    r"(?:(?:Valid\s*(?:Till|Upto|Until|To)|Expiry\s*Date|Exp\.?\s*Date|Date\s*of\s*Expiry|Valid\s*Through|Validity(?:\s*\(?[A-Z]+\)?)?)"
    r"[\s\:\-\.\/]*)"
    r"([0-3]?[0-9][\/\-\.][0-1]?[0-9][\/\-\.](?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Lazy Singletons for Offline Models (RapidOCR & OpenCV Face Detector)
# -----------------------------------------------------------------------------

_OCR_LOCK = threading.Lock()
_OCR_ENGINE: Any = None

_FACE_LOCK = threading.Lock()
_DNN_FACE_DETECTOR: Any = None
_HAAR_DETECTORS: List[Any] = []
_FACE_DETECTORS_INITIALIZED = False


def _ensure_data_dirs() -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def _get_ocr_engine() -> Any:
    """Lazy-load and return the singleton RapidOCR ONNX engine configured with CUDAExecutionProvider."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            _init_nvidia_dll_paths()
            import onnxruntime as ort
            from rapidocr_onnxruntime import RapidOCR

            providers = ort.get_available_providers()
            logger.info("ONNX Runtime available execution providers: %s", providers)

            # Ensure config.yaml enables CUDA across Det, Cls, and Rec
            try:
                import rapidocr_onnxruntime
                import yaml

                pkg_dir = Path(rapidocr_onnxruntime.__file__).parent
                cfg_file = pkg_dir / "config.yaml"
                if cfg_file.is_file():
                    raw = cfg_file.read_text(encoding="utf-8")
                    if "use_cuda: false" in raw:
                        cfg_file.write_text(raw.replace("use_cuda: false", "use_cuda: true"), encoding="utf-8")
            except Exception as cfg_err:
                logger.debug("Automatic config.yaml CUDA sync: %s", cfg_err)

            cuda_ok = "CUDAExecutionProvider" in providers
            if not cuda_ok:
                logger.warning("CUDAExecutionProvider NOT available for RapidOCR; falling back to CPU.")
            # Strictly request CUDA for Det/Cls/Rec (ORT falls back to CPU only if CUDA truly fails to load)
            # (config.yaml `use_cuda: true` for Det/Cls/Rec is enforced above; this rapidocr build's
            # kwargs updater is unreliable, so the YAML is the source of truth.)
            _OCR_ENGINE = RapidOCR()
            det_p = getattr(getattr(getattr(_OCR_ENGINE, "text_detector", None), "infer", None), "session", None)
            rec_p = getattr(getattr(getattr(_OCR_ENGINE, "text_recognizer", None), "session", None), "session", None)
            logger.info(
                "RapidOCR initialized (Det providers: %s, Rec providers: %s)",
                det_p.get_providers() if det_p else "N/A",
                rec_p.get_providers() if rec_p else "N/A",
            )
    return _OCR_ENGINE


def _init_face_detectors() -> Tuple[Any, List[Any]]:
    """
    Initialize OpenCV face detectors once:
      1. OpenCV YuNet ONNX FaceDetectorYN or DNN Caffe/TensorFlow detector if model weights exist in `models/`
      2. Built-in offline OpenCV Haar Cascade (`haarcascade_frontalface_default.xml` + `alt2`)
    """
    global _DNN_FACE_DETECTOR, _HAAR_DETECTORS, _FACE_DETECTORS_INITIALIZED
    if _FACE_DETECTORS_INITIALIZED:
        return _DNN_FACE_DETECTOR, _HAAR_DETECTORS

    import cv2

    # 1. Optional local OpenCV DNN / YuNet model if placed in ./models/
    yunet_candidates = list(MODELS_DIR.glob("*yunet*.onnx")) if MODELS_DIR.is_dir() else []
    if yunet_candidates and hasattr(cv2, "FaceDetectorYN"):
        try:
            detector = cv2.FaceDetectorYN.create(
                str(yunet_candidates[0]), "", (320, 320), 0.65, 0.3, 5000
            )
            # Configure CUDA backend if available, fallback cleanly to CPU
            try:
                if hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
                    detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass
            _DNN_FACE_DETECTOR = ("yunet", detector)
        except Exception as exc:
            logger.debug("YuNet initialization skipped: %s", exc)

    if _DNN_FACE_DETECTOR is None and MODELS_DIR.is_dir():
        proto = MODELS_DIR / "deploy.prototxt"
        caffe = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
        if proto.is_file() and caffe.is_file():
            try:
                net = cv2.dnn.readNetFromCaffe(str(proto), str(caffe))
                try:
                    if hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                except Exception:
                    pass
                _DNN_FACE_DETECTOR = ("ssd", net)
            except Exception as exc:
                logger.debug("DNN Caffe detector initialization skipped: %s", exc)

    # 2. Always load built-in offline OpenCV Haar Cascades
    haar_base = getattr(cv2.data, "haarcascades", "")
    for xml_name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        xml_path = os.path.join(haar_base, xml_name)
        if os.path.isfile(xml_path):
            cascade = cv2.CascadeClassifier(xml_path)
            if not cascade.empty():
                _HAAR_DETECTORS.append(cascade)

    _FACE_DETECTORS_INITIALIZED = True
    return _DNN_FACE_DETECTOR, _HAAR_DETECTORS


# -----------------------------------------------------------------------------
# Image / PDF Loading & OCR Extraction
# -----------------------------------------------------------------------------


class VisualLoadError(Exception):
    """Base exception for visual decoding failures."""
    def __init__(self, message: str, reason: str = "Corrupted or unreadable image"):
        super().__init__(message)
        self.reason = reason


class PasswordProtectedError(VisualLoadError):
    """Raised when a PDF file is encrypted / password protected."""
    def __init__(self, message: str = "Password protected PDF"):
        super().__init__(message, reason="Password protected PDF")


def load_visual_image(
    file_path: Union[str, Path],
    password: Optional[str] = None,
) -> Image.Image:
    """
    Open an image (.jpg, .jpeg, .png, .webp) or render Page 1 of a .pdf via `pypdfium2`
    into an upright RGB `PIL.Image.Image`.

    Raises:
      - `PasswordProtectedError` if the PDF is password-protected and unauthenticated.
      - `VisualLoadError` if the file is corrupted, empty, or unreadable.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.is_file():
        raise VisualLoadError(f"File not found: {path}", reason="File not found")

    if not path.is_file():
        raise VisualLoadError(f"File not found: '{path}'", reason="File not found")
    try:
        if path.stat().st_size == 0:
            raise VisualLoadError(f"File is 0 bytes: '{path.name}'", reason="Zero-byte file")
    except OSError:
        pass

    try:
        if ext == ".pdf":
            import pypdfium2 as pdfium
            import pypdfium2.raw as pdfium_c

            try:
                pdf = pdfium.PdfDocument(str(path), password=password)
            except pdfium.PdfiumError as pe:
                err_code = getattr(pe, "err_code", None)
                if err_code in (pdfium_c.FPDF_ERR_PASSWORD, pdfium_c.FPDF_ERR_SECURITY) or "password" in str(pe).lower():
                    raise PasswordProtectedError(f"Password required for PDF '{path.name}'") from pe
                raise VisualLoadError(f"Corrupted or invalid PDF format: {pe}", reason=f"Corrupt PDF ({pe})") from pe
            except Exception as pe:
                if "password" in str(pe).lower():
                    raise PasswordProtectedError(f"Password required for PDF '{path.name}'") from pe
                raise VisualLoadError(f"Failed to open PDF: {pe}", reason=f"Corrupt PDF ({pe})") from pe

            try:
                if len(pdf) < 1:
                    raise VisualLoadError(f"PDF contains 0 pages: {path.name}", reason="Empty PDF (0 pages)")
                page = pdf[0]
                try:
                    # Dynamically clamp scale so max rendered side <= 2400 to prevent OOM on posters/architectural drawings
                    render_scale = 2.0
                    try:
                        pw, ph = page.get_size()
                        max_dim = max(float(pw), float(ph))
                        if max_dim * render_scale > 2400:
                            render_scale = max(1.0, 2400.0 / max_dim)
                    except Exception:
                        render_scale = 2.0

                    bitmap = page.render(scale=render_scale, rotation=0)
                    pil_img = bitmap.to_pil().convert("RGB")
                    if hasattr(bitmap, "close"):
                        bitmap.close()
                    return pil_img
                finally:
                    if hasattr(page, "close"):
                        page.close()
            finally:
                if hasattr(pdf, "close"):
                    pdf.close()
        else:
            with Image.open(path) as img:
                transposed = ImageOps.exif_transpose(img)
                res = transposed.convert("RGB")
                # Clamp massive raster images > 4096 to prevent runaway OCR allocations
                if max(res.size) > 4096:
                    res.thumbnail((4096, 4096), Image.Resampling.BILINEAR)
                return res
    except (PasswordProtectedError, VisualLoadError):
        raise
    except Exception as exc:
        raise VisualLoadError(f"Could not visually decode file '{path.name}': {exc}", reason=f"Decode error: {exc}") from exc


def _extract_ocr_detailed(
    pil_img: Image.Image,
    ocr_engine: Any = None,
    allow_angle_cls: bool = True,
) -> Tuple[str, float, bool]:
    """
    Run offline RapidOCR on an RGB PIL image (downscaling oversized images for speed)
    and return `(extracted_text, average_confidence, is_vertical_text_layout)`.

    When `allow_angle_cls=False`, RapidOCR's internal 180° crop flipper is temporarily
    disabled so right-side-up orientations cleanly outscore upside-down (180°) ones.
    """
    width, height = pil_img.size
    ocr_target = pil_img
    longest = max(width, height)
    is_resized = False
    if longest > MAX_OCR_IMAGE_DIMENSION:
        scale = MAX_OCR_IMAGE_DIMENSION / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        ocr_target = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        is_resized = True

    rgb_array = np.asarray(ocr_target, dtype=np.uint8)
    if is_resized:
        try:
            ocr_target.close()
        except Exception:
            pass

    bgr_array = rgb_array[:, :, ::-1].copy()

    engine = ocr_engine or _get_ocr_engine()
    with _OCR_LOCK:
        prev_cls = getattr(engine, "use_angle_cls", True)
        try:
            if hasattr(engine, "use_angle_cls"):
                engine.use_angle_cls = allow_angle_cls
            result, _ = engine(bgr_array)
        finally:
            if hasattr(engine, "use_angle_cls"):
                engine.use_angle_cls = prev_cls

    if not result:
        return "", 0.0, False

    extracted_lines: List[str] = []
    confidences: List[float] = []
    vertical_boxes = 0
    horizontal_boxes = 0

    for entry in result:
        if not entry or len(entry) < 2:
            continue
        box = entry[0]
        text_segment = str(entry[1]).strip()
        confidence = float(entry[2]) if len(entry) >= 3 and entry[2] is not None else 1.0
        if text_segment and confidence >= 0.30:
            extracted_lines.append(text_segment)
            confidences.append(confidence)
            if len(text_segment) >= 3 and isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    xs = [float(pt[0]) for pt in box]
                    ys = [float(pt[1]) for pt in box]
                    span_x = max(xs) - min(xs)
                    span_y = max(ys) - min(ys)
                    if span_y > span_x * 1.35:
                        vertical_boxes += 1
                    elif span_x > span_y * 1.15:
                        horizontal_boxes += 1
                except (TypeError, ValueError, IndexError):
                    pass

    if not extracted_lines:
        return "", 0.0, False

    avg_conf = sum(confidences) / float(len(confidences))
    is_vertical_layout = vertical_boxes > horizontal_boxes
    return "\n".join(extracted_lines), round(avg_conf, 4), is_vertical_layout


def extract_ocr_with_confidence(
    pil_img: Image.Image,
    ocr_engine: Any = None,
) -> Tuple[str, float]:
    """
    Run offline RapidOCR on an RGB PIL image (downscaling oversized images for speed)
    and return `(extracted_text, average_confidence)`.
    """
    text, avg_conf, _ = _extract_ocr_detailed(
        pil_img, ocr_engine=ocr_engine, allow_angle_cls=True
    )
    return text, avg_conf


def extract_ocr_text(pil_img: Image.Image, ocr_engine: Any = None) -> str:
    """Backward-compatible wrapper returning extracted OCR text string."""
    text, _ = extract_ocr_with_confidence(pil_img, ocr_engine=ocr_engine)
    return text


LOW_OCR_CONFIDENCE_THRESHOLD = 0.72


def rotate_pil_clockwise(pil_img: Image.Image, degrees_cw: int) -> Image.Image:
    """Losslessly rotate a PIL image by 90, 180, or 270 degrees clockwise."""
    angle = degrees_cw % 360
    if angle == 90:
        return pil_img.transpose(Image.Transpose.ROTATE_270)
    if angle == 180:
        return pil_img.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return pil_img.transpose(Image.Transpose.ROTATE_90)
    return pil_img


def inspect_with_auto_rotation(
    pil_img: Image.Image,
    ocr_engine: Any = None,
) -> Tuple[Image.Image, str, float, bool, str, Optional[str], int]:
    """
    Run OCR + Smart Classification on `pil_img` (already EXIF-transposed).
    If the initial 0° pass yields < 15 characters, low OCR confidence (< 0.72),
    vertical/sideways text bounding boxes, or fails document classification,
    automatically tries 90° and 270° (and 180° when upside-down text is present)
    to recover rotated mobile photos of PAN/Aadhaar/ID cards and save an upright thumbnail.

    Returns `(upright_pil_img, ocr_text, avg_conf, is_document, doc_category, extracted_id, applied_rotation_cw)`.
    """
    text_0, conf_0, is_vert_0 = _extract_ocr_detailed(
        pil_img, ocr_engine=ocr_engine, allow_angle_cls=False
    )
    is_doc_0, cat_0, id_0 = classify_document_text(text_0)
    clean_len_0 = len(text_0.strip())

    # Fast path: upright horizontal-text verified document with strong OCR confidence
    if (
        is_doc_0
        and not is_vert_0
        and clean_len_0 >= MIN_OCR_TEXT_LENGTH
        and conf_0 >= LOW_OCR_CONFIDENCE_THRESHOLD
    ):
        return pil_img, text_0, conf_0, is_doc_0, cat_0, id_0, 0

    candidates: List[Tuple[int, Image.Image, str, float, bool, bool, str, Optional[str]]] = [
        (0, pil_img, text_0, conf_0, is_vert_0, is_doc_0, cat_0, id_0)
    ]

    # Try 90° clockwise
    img_90 = rotate_pil_clockwise(pil_img, 90)
    text_90, conf_90, is_vert_90 = _extract_ocr_detailed(
        img_90, ocr_engine=ocr_engine, allow_angle_cls=False
    )
    is_doc_90, cat_90, id_90 = classify_document_text(text_90)
    clean_len_90 = len(text_90.strip())
    candidates.append((90, img_90, text_90, conf_90, is_vert_90, is_doc_90, cat_90, id_90))

    # Try 270° clockwise
    img_270 = rotate_pil_clockwise(pil_img, 270)
    text_270, conf_270, is_vert_270 = _extract_ocr_detailed(
        img_270, ocr_engine=ocr_engine, allow_angle_cls=False
    )
    is_doc_270, cat_270, id_270 = classify_document_text(text_270)
    clean_len_270 = len(text_270.strip())
    candidates.append((270, img_270, text_270, conf_270, is_vert_270, is_doc_270, cat_270, id_270))

    # If 0°, 90°, and 270° all detected zero characters, the image contains no readable text
    if clean_len_0 == 0 and clean_len_90 == 0 and clean_len_270 == 0:
        return pil_img, text_0, conf_0, False, database.NOT_A_DOCUMENT, None, 0

    # Check 180° if neither 90° nor 270° produced an upright horizontal-text verified document
    has_upright_sideways_doc = (is_doc_90 and not is_vert_90) or (is_doc_270 and not is_vert_270)
    if not has_upright_sideways_doc:
        img_180 = rotate_pil_clockwise(pil_img, 180)
        text_180, conf_180, is_vert_180 = _extract_ocr_detailed(
            img_180, ocr_engine=ocr_engine, allow_angle_cls=False
        )
        is_doc_180, cat_180, id_180 = classify_document_text(text_180)
        candidates.append(
            (180, img_180, text_180, conf_180, is_vert_180, is_doc_180, cat_180, id_180)
        )

    def _orientation_rank(
        item: Tuple[int, Image.Image, str, float, bool, bool, str, Optional[str]]
    ):
        angle, _, txt, conf, is_vert, is_doc, _, ext_id = item
        t_len = len(txt.strip())
        quality = conf * float(min(t_len, 240))
        return (
            1 if (is_doc and not is_vert) else 0,
            1 if is_doc else 0,
            0 if is_vert else 1,
            1 if ext_id else 0,
            round(quality, 3),
            round(conf, 4),
            t_len,
            1 if angle == 0 else 0,
        )

    best_angle, best_img, best_text, best_conf, _, best_is_doc, best_cat, best_id = max(
        candidates, key=_orientation_rank
    )

    # Free memory from unused rotated copies
    for c in candidates:
        cand_img = c[1]
        if cand_img is not best_img and cand_img is not pil_img:
            try:
                cand_img.close()
            except Exception:
                pass

    if best_angle != 0 and best_is_doc:
        logger.info(
            "Auto-rotated document by %d° CW -> classified as %s (conf=%.2f)",
            best_angle,
            best_cat,
            best_conf,
        )
    return best_img, best_text, best_conf, best_is_doc, best_cat, best_id, best_angle


# -----------------------------------------------------------------------------
# Step C: Smart Content Classifier (Pure Visual OCR Analysis)
# -----------------------------------------------------------------------------


def _is_valid_aadhaar_number(raw_num: str) -> bool:
    """
    Validate that a 12-digit string adheres to genuine Aadhaar numbering specifications:
    - Must be exactly 12 numeric digits.
    - Real Aadhaar numbers NEVER start with '0' or '1' (first digit is always 2-9).
    - NEVER match if it starts with '91' or '+91' followed by a mobile digit (6-9) (Indian mobile numbers).
    """
    compact = re.sub(r"[\s\-]+", "", str(raw_num))
    if len(compact) != 12 or not compact.isdigit():
        return False
    # Aadhaar first digit is 2-9
    if compact[0] in ("0", "1"):
        return False
    # Country code + mobile number trap (+91 98765 43210 -> 919876543210)
    if compact.startswith("91") and compact[2] in "6789":
        return False
    return True


def _normalize_aadhaar_display(raw_num: str) -> str:
    """Format a 12-digit or masked Aadhaar string cleanly as 'XXXX XXXX 1234'."""
    compact = re.sub(r"[\s\-]+", "", raw_num).upper()
    if len(compact) == 12:
        return f"{compact[0:4]} {compact[4:8]} {compact[8:12]}"
    return raw_num.strip()


def validate_aadhaar_cascading(
    ocr_text: str,
) -> Tuple[bool, bool, bool, Optional[str]]:
    """
    Cascading Filter (Short-Circuit) Two-Step Validation for Aadhaar Fast-Track Mode:

    1. Gate 1 (Primary Master Keys):
       After the fast OCR pass, check ONLY for the most unique strings:
       "GOVERNMENT" (for front) or "UIDAI" (for back) - case-insensitive.
       If NEITHER of these Gate 1 keywords is found, immediately short-circuit.
       Do not waste time checking other keywords.

    2. Gate 2 (Secondary Verification):
       IF Gate 1 passes (meaning it might be a government ID), THEN and ONLY THEN
       run the secondary checks (looking for "DOB", "Male/Female", or the `\\d{4}\\s\\d{4}\\s\\d{4}` regex)
       to confirm it is an actual Aadhaar card before triggering the heavy InsightFace engine.

    Returns:
       `(is_confirmed, gate1_passed, gate2_passed, extracted_id)`
    """
    if not ocr_text or not ocr_text.strip():
        return False, False, False, None

    upper_text = ocr_text.upper()

    # Gate 1: Check ONLY for the primary master keys: "GOVERNMENT" or "UIDAI" (case-insensitive)
    has_gate1 = ("GOVERNMENT" in upper_text) or ("UIDAI" in upper_text)
    if not has_gate1:
        # Immediate short-circuit: do not evaluate secondary keywords or regexes
        return False, False, False, None

    lower_text = ocr_text.lower()

    # Gate 2: Secondary Verification (DOB, Male/Female, or \\d{4}\\s\\d{4}\\s\\d{4} regex)
    has_dob = (
        "dob" in lower_text
        or "date of birth" in lower_text
        or "year of birth" in lower_text
        or "yob" in lower_text
        or bool(RE_DOB_LABELED.search(ocr_text))
    )
    has_gender = (
        "male" in lower_text
        or "female" in lower_text
        or "/ male" in lower_text
        or "/ female" in lower_text
        or "transgender" in lower_text
    )

    m_444 = RE_AADHAAR_444.search(ocr_text)
    m_spaced = RE_AADHAAR_SPACED.search(ocr_text)
    m_masked = RE_AADHAAR_MASKED.search(ocr_text)
    m_full = RE_AADHAAR_FULL.search(ocr_text)
    m_vid = RE_AADHAAR_VID.search(ocr_text)

    has_number = False
    extracted_id: Optional[str] = None

    if m_444 and _is_valid_aadhaar_number(m_444.group(0)):
        has_number = True
        extracted_id = _normalize_aadhaar_display(m_444.group(0))
    elif m_spaced and _is_valid_aadhaar_number(m_spaced.group(1)):
        has_number = True
        extracted_id = _normalize_aadhaar_display(m_spaced.group(1))
    elif m_masked:
        has_number = True
        extracted_id = _normalize_aadhaar_display(m_masked.group(1))
    elif m_full and _is_valid_aadhaar_number(m_full.group(1)):
        has_number = True
        extracted_id = _normalize_aadhaar_display(m_full.group(1))
    elif m_vid:
        has_number = True
        extracted_id = f"VID: {m_vid.group(1).strip()}"

    gate2_passed = bool(has_dob or has_gender or has_number)
    is_confirmed = bool(has_gate1 and gate2_passed)

    if is_confirmed and not extracted_id and m_444:
        extracted_id = _normalize_aadhaar_display(m_444.group(0))

    return is_confirmed, True, gate2_passed, extracted_id


def normalize_date_string(date_raw: str) -> Optional[str]:
    """
    Normalize raw date strings into standard ISO YYYY-MM-DD.
    Handles formats like DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, or YYYY.
    """
    if not date_raw:
        return None
    clean = re.sub(r"[\s]+", "", str(date_raw)).replace(".", "-").replace("/", "-")
    parts = [p for p in clean.split("-") if p]
    try:
        if len(parts) == 3:
            # If in DD-MM-YYYY format
            if len(parts[2]) == 4 and len(parts[0]) <= 2:
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
                    return f"{y:04d}-{m:02d}-{d:02d}"
            # If in YYYY-MM-DD format
            elif len(parts[0]) == 4 and len(parts[2]) <= 2:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                if 1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
                    return f"{y:04d}-{m:02d}-{d:02d}"
        elif len(parts) == 1 and len(parts[0]) == 4 and parts[0].isdigit():
            y = int(parts[0])
            if 1900 <= y <= 2100:
                return f"{y:04d}-01-01"
    except (ValueError, TypeError):
        pass
    return None


def extract_metadata(ocr_text: str, doc_category: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract Date of Birth (dob) and Document Expiry Date (expiry_date) from OCR text.
    Returns normalized ISO (dob_iso, expiry_iso) strings, e.g. ('1995-08-15', '2031-10-25').
    """
    if not ocr_text:
        return None, None

    dob_val: Optional[str] = None
    expiry_val: Optional[str] = None

    # 1. Date of Birth
    dob_match = RE_DOB_LABELED.search(ocr_text)
    if dob_match:
        dob_val = normalize_date_string(dob_match.group(1))

    # Fallback for PAN cards: DOB is often printed right below Name or Father's Name
    if not dob_val and doc_category == "PAN Card":
        for date_candidate in RE_GENERIC_DATE.findall(ocr_text):
            norm = normalize_date_string(date_candidate)
            if norm:
                try:
                    year = int(norm.split("-")[0])
                    if 1925 <= year <= 2022:
                        dob_val = norm
                        break
                except (ValueError, IndexError):
                    pass

    # 2. Expiry Date (primarily Driving License, Passport, and validity certificates)
    if doc_category in ("Driving License", "Passport", "Certificate/Marksheet"):
        exp_match = RE_EXPIRY_LABELED.search(ocr_text)
        if exp_match:
            expiry_val = normalize_date_string(exp_match.group(1))

    return dob_val, expiry_val


def classify_document_text(ocr_text: str) -> Tuple[bool, str, Optional[str]]:
    """
    Analyze extracted OCR text using case-insensitive domain keywords and regex patterns.

    Returns `(is_document, doc_category, extracted_id_number)` where `doc_category` is one of:
      - 'PAN Card'
      - 'Aadhaar Card'
      - 'Driving License'
      - 'Voter ID'
      - 'Passport'
      - 'Certificate/Marksheet'
      - 'Bill/Receipt'
      - 'Not a Document'
    """
    if not ocr_text:
        return False, database.NOT_A_DOCUMENT, None

    cleaned = ocr_text.strip()
    non_space_chars = re.sub(r"\s+", "", cleaned)
    if len(cleaned) < MIN_OCR_TEXT_LENGTH or len(non_space_chars) < 12:
        return False, database.NOT_A_DOCUMENT, None

    lower_text = cleaned.lower()
    upper_text = cleaned.upper()

    scores: Dict[str, int] = {cat: 0 for cat in database.DOCUMENT_CATEGORIES}
    extracted_ids: Dict[str, Optional[str]] = {cat: None for cat in database.DOCUMENT_CATEGORIES}

    # --- 1. PAN Card ---
    # Exclude 15-char GSTIN substrings automatically via word boundaries in RE_PAN
    gstin_match = RE_GSTIN.search(upper_text)
    pan_matches = [
        m for m in RE_PAN.findall(upper_text)
        if not (gstin_match and m in gstin_match.group(1) and "income tax" not in lower_text)
    ]
    if pan_matches:
        scores["PAN Card"] += 12
        extracted_ids["PAN Card"] = pan_matches[0]

    if "income tax department" in lower_text or "ayakar vibhag" in lower_text:
        scores["PAN Card"] += 14
    if "permanent account number" in lower_text:
        scores["PAN Card"] += 14
    has_govt_india = any(
        kw in lower_text for kw in ("govt. of india", "govt of india", "government of india")
    )
    has_father_name = any(
        kw in lower_text for kw in ("father's name", "fathers name", "father name")
    )
    if has_govt_india and has_father_name:
        scores["PAN Card"] += 11

    # --- 2. Aadhaar Card ---
    # Mandatory Anchor Context: A 12-digit number alone is NEVER enough.
    # It MUST be accompanied by at least one genuine Aadhaar anchor keyword.
    has_aadhaar_anchor = any(
        kw in lower_text
        for kw in (
            "government of india",
            "govt of india",
            "govt. of india",
            "unique identification authority",
            "uidai",
            "mera aadhaar",
            "meri pehchan",
            "aadhaar",
            "aadhar",
            "enrollment no",
            "enrolment no",
            "vid",
            "dob",
            "date of birth",
            "male",
            "female",
        )
    )

    masked_aadhaar = RE_AADHAAR_MASKED.search(cleaned)
    spaced_aadhaar = RE_AADHAAR_SPACED.search(cleaned)
    full_aadhaar = RE_AADHAAR_FULL.search(cleaned)
    vid_match = RE_AADHAAR_VID.search(cleaned)

    # Validate numbers so shop banners, phone numbers, and invalid Aadhaar numbers are rejected
    if masked_aadhaar and has_aadhaar_anchor:
        scores["Aadhaar Card"] += 12
        extracted_ids["Aadhaar Card"] = _normalize_aadhaar_display(masked_aadhaar.group(1))
    elif spaced_aadhaar and _is_valid_aadhaar_number(spaced_aadhaar.group(1)) and has_aadhaar_anchor:
        scores["Aadhaar Card"] += 12
        extracted_ids["Aadhaar Card"] = _normalize_aadhaar_display(spaced_aadhaar.group(1))
    elif full_aadhaar and _is_valid_aadhaar_number(full_aadhaar.group(1)) and has_aadhaar_anchor:
        scores["Aadhaar Card"] += 9
        extracted_ids["Aadhaar Card"] = _normalize_aadhaar_display(full_aadhaar.group(1))

    if vid_match and has_aadhaar_anchor and not extracted_ids["Aadhaar Card"]:
        extracted_ids["Aadhaar Card"] = f"VID: {vid_match.group(1).strip()}"

    if "unique identification authority" in lower_text or "uidai" in lower_text:
        scores["Aadhaar Card"] += 15
    if "mera aadhaar" in lower_text or "meri pehchan" in lower_text:
        scores["Aadhaar Card"] += 13
    elif "aadhaar" in lower_text or "aadhar" in lower_text:
        scores["Aadhaar Card"] += 10
    if "enrollment no" in lower_text or "enrolment no" in lower_text:
        scores["Aadhaar Card"] += 10
    if "vid :" in lower_text or "vid:" in lower_text or (vid_match and has_aadhaar_anchor):
        scores["Aadhaar Card"] += 11
    if "government of india" in lower_text or "govt of india" in lower_text:
        scores["Aadhaar Card"] += 8

    # Cascading Gate 1 + Gate 2 validation
    is_aadhaar_conf, g1_pass, g2_pass, cascaded_id = validate_aadhaar_cascading(cleaned)
    if is_aadhaar_conf:
        scores["Aadhaar Card"] += 15
        if not extracted_ids["Aadhaar Card"] and cascaded_id:
            extracted_ids["Aadhaar Card"] = cascaded_id

    # --- 3. Driving License ---
    for kw, weight in (
        ("driving licence", 15),
        ("driving license", 15),
        ("union of india driving", 14),
        ("motor vehicles", 11),
        ("transport department", 11),
        ("authorization to drive", 13),
        ("authorisation to drive", 13),
        ("dl no", 11),
        ("dl number", 11),
        ("licensing authority", 9),
    ):
        if kw in lower_text:
            scores["Driving License"] += weight

    dl_match = RE_DRIVING_LICENSE.search(upper_text)
    if dl_match:
        scores["Driving License"] += 9
        extracted_ids["Driving License"] = re.sub(r"\s+", " ", dl_match.group(1)).strip()
    else:
        dl_labeled = RE_DL_LABELED.search(cleaned)
        if dl_labeled:
            extracted_ids["Driving License"] = dl_labeled.group(1).strip()

    # --- 4. Voter ID ---
    for kw, weight in (
        ("election commission of india", 16),
        ("elector photo identity card", 16),
        ("elector's name", 13),
        ("electors name", 13),
        ("epic no", 12),
        ("nirvachan", 10),
    ):
        if kw in lower_text:
            scores["Voter ID"] += weight

    epic_match = RE_VOTER_EPIC.search(upper_text)
    if epic_match:
        if scores["Voter ID"] > 0 or "elector" in lower_text or "election" in lower_text:
            scores["Voter ID"] += 10
        extracted_ids["Voter ID"] = epic_match.group(1)

    # --- 5. Passport ---
    if RE_PASSPORT_MRZ.search(upper_text):
        scores["Passport"] += 16
    if "passport" in lower_text:
        scores["Passport"] += 13
    if "republic of india" in lower_text:
        scores["Passport"] += 10
    if "place of issue" in lower_text:
        scores["Passport"] += 9
    if "nationality" in lower_text:
        scores["Passport"] += 8

    passport_no_match = RE_PASSPORT_NO.search(upper_text)
    if passport_no_match:
        extracted_ids["Passport"] = passport_no_match.group(1)

    # --- 6. Certificate / Marksheet ---
    # Negative keywords for marksheets/certificates (textbooks, chapters, admit cards)
    marksheet_negatives = (
        "chapter",
        "lesson",
        "syllabus",
        "hall ticket",
        "call letter",
        "admit card",
        "exercise",
        "textbook",
        "mathematics",
        "maths",
        "philosopher",
        "properties of matter",
    )
    has_marksheet_negative = any(neg in lower_text for neg in marksheet_negatives)

    # A genuine educational/official certificate must have a core anchor:
    # "school" or "university" ALONE in text is not enough to declare an image a certificate!
    cert_anchors = (
        "certificate",
        "marksheet",
        "mark sheet",
        "statement of marks",
        "board of secondary",
        "higher secondary",
        "secondary school leaving",
        "leaving certificate",
        "passing certificate",
        "grade card",
        "transcripts",
        "provisional certificate",
        "migration certificate",
        "transfer certificate",
        "degree",
    )
    has_cert_anchor = any(anchor in lower_text for anchor in cert_anchors)

    if not has_marksheet_negative and has_cert_anchor:
        for kw, weight in (
            ("death certificate", 16),
            ("birth certificate", 16),
            ("statement of marks", 15),
            ("board of secondary", 14),
            ("higher secondary", 12),
            ("certificate of", 12),
            ("municipal corporation", 11),
            ("marksheet", 14),
            ("mark sheet", 14),
            ("provisional certificate", 13),
            ("migration certificate", 13),
            ("transfer certificate", 13),
            ("university", 6),
            ("school", 4),
        ):
            if kw in lower_text:
                scores["Certificate/Marksheet"] += weight

        cert_match = RE_CERT_NUMBER.search(cleaned)
        if cert_match:
            extracted_ids["Certificate/Marksheet"] = cert_match.group(1).strip()

    # --- 7. Admit Card / Hall Ticket ---
    admit_keywords = (
        ("hall ticket", 16),
        ("admit card", 16),
        ("online call letter", 15),
        ("call letter", 14),
        ("confirmation number", 13),
        ("examination admission", 14),
        ("exam admit", 14),
        ("seat no", 10),
        ("roll no", 8),
    )
    for kw, weight in admit_keywords:
        if kw in lower_text:
            scores["Admit Card / Hall Ticket"] += weight

    admit_match = RE_ADMIT_NUMBER.search(cleaned)
    if admit_match and scores["Admit Card / Hall Ticket"] > 0:
        extracted_ids["Admit Card / Hall Ticket"] = admit_match.group(1).strip()

    # --- 8. Bill / Receipt ---
    for kw, weight in (
        ("tax invoice", 14),
        ("bill of supply", 13),
        ("cash memo", 13),
        ("payment receipt", 13),
        ("fee receipt", 13),
        ("electricity bill", 13),
        ("gstin", 10),
        ("invoice no", 10),
        ("receipt no", 10),
        ("cgst", 6),
        ("sgst", 6),
        ("igst", 6),
        ("grand total", 7),
        ("total amount", 6),
        ("amount paid", 6),
        ("subtotal", 5),
    ):
        if kw in lower_text:
            scores["Bill/Receipt"] += weight

    inv_match = RE_INVOICE_NO.search(cleaned)
    if inv_match:
        extracted_ids["Bill/Receipt"] = inv_match.group(1).strip()
    elif gstin_match:
        extracted_ids["Bill/Receipt"] = gstin_match.group(1)

    # Select winning category
    best_category = max(scores, key=lambda cat: scores[cat])
    best_score = scores[best_category]

    if best_score < MIN_DOCUMENT_CLASSIFICATION_SCORE:
        return False, database.NOT_A_DOCUMENT, None

    return True, best_category, extracted_ids.get(best_category)


# -----------------------------------------------------------------------------
# Step D: Face Detection, Feature Embedding, Clustering & Document Thumbnails
# -----------------------------------------------------------------------------


def _standardize_vector(vec: np.ndarray) -> np.ndarray:
    """Zero-mean and L2-normalize a feature sub-vector independently."""
    v = vec.astype(np.float32)
    v = v - float(np.mean(v))
    norm = float(np.linalg.norm(v))
    if norm > 1e-8:
        v = v / norm
    return v


def _compute_lbp_grid_histogram(gray_img: np.ndarray, grid: int = 4, bins: int = 14) -> np.ndarray:
    """
    Compute vectorized 8-neighbor Local Binary Pattern (LBP) spatial grid histograms,
    excluding flat/constant codes (0 and 255) so smooth skin does not create universal spikes.
    """
    padded = np.pad(gray_img.astype(np.int16), pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, 0:-2],
        padded[1:-1, 0:-2],
    ]
    # Small deadzone (+2) suppresses JPEG sensor noise on uniform skin patches
    for bit_idx, neighbor in enumerate(neighbors):
        lbp |= (((neighbor - center) >= 2).astype(np.uint8) << bit_idx)

    h, w = lbp.shape
    cell_h, cell_w = h // grid, w // grid
    hist_parts: List[np.ndarray] = []

    for gy in range(grid):
        for gx in range(grid):
            patch = lbp[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            # Range (1, 255) excludes constant codes 0 and 255
            hist, _ = np.histogram(patch, bins=bins, range=(1, 255))
            hist_f = hist.astype(np.float32)
            hist_f /= float(patch.size) + 1e-6
            hist_parts.append(hist_f)

    return np.concatenate(hist_parts)


def _compute_gradient_grid_histogram(gray_img: np.ndarray, grid: int = 4, bins: int = 8) -> np.ndarray:
    """
    Compute spatial oriented-gradient (HOG-style) histograms over an inner face image.
    """
    gray_f = gray_img.astype(np.float32)
    gx = np.zeros_like(gray_f)
    gy = np.zeros_like(gray_f)
    gx[:, 1:-1] = gray_f[:, 2:] - gray_f[:, :-2]
    gy[1:-1, :] = gray_f[2:, :] - gray_f[:-2, :]

    mag = np.hypot(gx, gy)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0  # Unsigned [0, 180)

    h, w = gray_img.shape
    cell_h, cell_w = h // grid, w // grid
    hist_parts: List[np.ndarray] = []

    for cy in range(grid):
        for cx in range(grid):
            m_patch = mag[cy * cell_h : (cy + 1) * cell_h, cx * cell_w : (cx + 1) * cell_w]
            a_patch = angle[cy * cell_h : (cy + 1) * cell_h, cx * cell_w : (cx + 1) * cell_w]
            hist, _ = np.histogram(a_patch, bins=bins, range=(0.0, 180.0), weights=m_patch)
            hist_f = hist.astype(np.float32)
            norm = float(np.linalg.norm(hist_f))
            if norm > 1e-6:
                hist_f /= norm
            hist_parts.append(hist_f)

    return np.concatenate(hist_parts)


def image_to_base64_thumbnail(pil_img: Image.Image, max_size: Tuple[int, int] = (160, 160), quality: int = 70) -> str:
    """Compress image into a compact Base64 JPEG data URI for instant DB retrieval."""
    import base64
    import io
    thumb = pil_img.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def compute_face_embedding(face_rgb: np.ndarray) -> List[float]:
    """
    Compute a discriminative 512-dimensional facial embedding vector combining:
      1. InsightFace normalized 512-d feature vector if local model exists
      2. Or standardized 512-d spatial Gradientface + LBP + inner-face appearance representation:
         - 196 inner spatial appearance dims
         - 288 horizontal/vertical spatial Gradientface dims
         - 28 spatial LBP/texture dims
    """
    pil_face = Image.fromarray(face_rgb).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    rgb_96 = np.asarray(pil_face, dtype=np.uint8)

    gray_f = (
        0.299 * rgb_96[:, :, 0].astype(np.float32)
        + 0.587 * rgb_96[:, :, 1].astype(np.float32)
        + 0.114 * rgb_96[:, :, 2].astype(np.float32)
    )
    try:
        import cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray_96 = clahe.apply(np.clip(gray_f, 0, 255).astype(np.uint8))
    except ImportError:
        p5, p95 = np.percentile(gray_f, (5, 95))
        stretched = (gray_f - p5) * (255.0 / max(1.0, float(p95 - p5)))
        gray_96 = np.clip(stretched, 0, 255).astype(np.uint8)

    # Crop central 84% inner face [8:88, 8:88] (80x80) to eliminate background edges
    inner_gray = gray_96[8:88, 8:88]
    inner_pil = Image.fromarray(inner_gray)

    # 1. Spatial 14x14 normalized appearance grid (196 dims)
    spatial_14 = np.asarray(
        inner_pil.resize((14, 14), Image.Resampling.BILINEAR), dtype=np.float32
    ).ravel()
    spatial_vec = _standardize_vector(spatial_14)

    # 2. Signed spatial Gradientface 12x12 horizontal & vertical derivatives (288 dims)
    grid_14 = np.asarray(
        inner_pil.resize((14, 14), Image.Resampling.BILINEAR), dtype=np.float32
    )
    signed_gx = (grid_14[1:-1, 2:] - grid_14[1:-1, :-2]).ravel()
    signed_gy = (grid_14[2:, 1:-1] - grid_14[:-2, 1:-1]).ravel()
    grad_vec = _standardize_vector(np.concatenate([signed_gx, signed_gy]))

    # 3. Exactly 28 spatial LBP texture dims (196 + 288 + 28 = exactly 512 dimensions)
    lbp_raw = _compute_lbp_grid_histogram(inner_gray, grid=2, bins=7)
    texture_vec = _standardize_vector(lbp_raw)

    combined = np.concatenate([
        spatial_vec * 0.40,
        grad_vec * 0.40,
        texture_vec * 0.20,
    ]).astype(np.float32)

    combined -= float(np.mean(combined))
    norm = float(np.linalg.norm(combined))
    if norm > 1e-8:
        combined /= norm

    return [round(float(x), 6) for x in combined.tolist()]


def detect_primary_face_box(
    rgb_array: np.ndarray,
    min_resolution: int = face_engine.MIN_FACE_RESOLUTION,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the primary portrait face `(x, y, w, h)` on a verified document image
    using OpenCV DNN (if local weights exist) or OpenCV Haar Cascade.
    Returns None if no valid portrait face is present (e.g. bills/receipts).
    """
    import cv2

    img_h, img_w = rgb_array.shape[:2]
    doc_area = float(img_w * img_h)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    with _FACE_LOCK:
        dnn_detector, haar_detectors = _init_face_detectors()
        candidate_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Try OpenCV DNN detector if available locally
        if dnn_detector is not None:
            kind, model = dnn_detector
            if kind == "yunet":
                model.setInputSize((img_w, img_h))
                _, faces = model.detect(bgr_array)
                if faces is not None:
                    for det in faces:
                        fx, fy, fw, fh = [int(round(v)) for v in det[:4]]
                        candidate_boxes.append((fx, fy, fw, fh))
            elif kind == "ssd":
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(bgr_array, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                model.setInput(blob)
                detections = model.forward()
                for i in range(detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    if conf >= 0.60:
                        box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
                        x1, y1, x2, y2 = box.astype(int)
                        candidate_boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))

        # 2. Offline Haar Cascade detection
        if not candidate_boxes and haar_detectors:
            gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
            eq_gray = cv2.equalizeHist(gray)
            min_side = max(int(min_resolution), int(min(img_w, img_h) * 0.05))

            for cascade in haar_detectors:
                detected = cascade.detectMultiScale(
                    eq_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_side, min_side),
                )
                if detected is not None and len(detected) > 0:
                    for fx, fy, fw, fh in detected:
                        candidate_boxes.append((int(fx), int(fy), int(fw), int(fh)))
                    break

    # Filter out implausible or low-resolution (< 80x80 px) junk faces
    min_box_side = max(1, int(min_resolution))
    valid_boxes: List[Tuple[int, int, int, int]] = []
    for fx, fy, fw, fh in candidate_boxes:
        if fw < min_box_side or fh < min_box_side:
            continue
        fx_c = max(0, fx)
        fy_c = max(0, fy)
        fw_c = min(img_w - fx_c, fw)
        fh_c = min(img_h - fy_c, fh)
        if fw_c < min_box_side or fh_c < min_box_side:
            continue
        aspect = float(fw_c) / float(fh_c)
        area_ratio = float(fw_c * fh_c) / doc_area
        if 0.65 <= aspect <= 1.45 and 0.0015 <= area_ratio <= 0.60:
            valid_boxes.append((fx_c, fy_c, fw_c, fh_c))

    if not valid_boxes:
        return None

    # Return largest detected face (ignores tiny ghost/watermark portraits on IDs)
    return max(valid_boxes, key=lambda b: b[2] * b[3])


def match_or_create_person(
    embedding: Sequence[float],
    face_thumbnail_rel_path: str,
    db_manager: database.DatabaseManager,
    similarity_threshold: float = FACE_MATCH_THRESHOLD,
) -> int:
    """
    Compare `embedding` against all existing `people` in SQLite.
    Assigns the best-matching `person_id` if similarity >= `similarity_threshold`
    (updating the cluster's running centroid), or creates a new Person cluster.
    Thread-safe under concurrent multi-worker scanning.
    """
    with _CLUSTER_LOCK:
        query_vec = np.asarray(embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm > 1e-8:
            query_vec = query_vec / q_norm

        existing_people = db_manager.get_all_people(include_embeddings=True)

        best_person_id: Optional[int] = None
        best_similarity: float = -1.0
        best_existing_vec: Optional[np.ndarray] = None

        for person in existing_people:
            stored_emb = person.get("face_embedding")
            if not stored_emb or len(stored_emb) != len(query_vec):
                continue
            p_vec = np.asarray(stored_emb, dtype=np.float32)
            p_norm = float(np.linalg.norm(p_vec))
            if p_norm <= 1e-8:
                continue
            p_vec = p_vec / p_norm

            similarity = float(np.dot(query_vec, p_vec))
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = int(person["id"])
                best_existing_vec = p_vec

        if best_person_id is not None and best_similarity >= similarity_threshold:
            # Smoothly refine centroid embedding with weighted running average
            if best_existing_vec is not None:
                doc_count = 1
                matched_person = db_manager.get_person(best_person_id, include_embedding=False)
                if matched_person:
                    doc_count = max(1, int(matched_person.get("document_count", 1)))
                updated_vec = (best_existing_vec * float(doc_count) + query_vec) / float(doc_count + 1)
                # Ensure strictly unit L2-normalized vector (preserve absolute directional cosine)
                u_norm = float(np.linalg.norm(updated_vec))
                if u_norm > 1e-8:
                    updated_vec = updated_vec / u_norm
                db_manager.update_person(
                    person_id=best_person_id,
                    face_embedding=[round(float(x), 6) for x in updated_vec.tolist()],
                )
            return best_person_id

        return db_manager.add_person(
            face_embedding=list(embedding),
            face_thumbnail_path=face_thumbnail_rel_path,
        )


def extract_reference_face_embedding(
    image_input: Union[str, Path, Image.Image, np.ndarray],
) -> Dict[str, Any]:
    """
    Extract a 512-d face embedding (and base64 portrait preview) from a user-selected reference image.
    1. Attempts InsightFace 512-d embedding extraction if local InsightFace model is loaded.
    2. Detects the primary face box across upright/rotated orientations (or uses the center portrait crop
       if the user provided an already tightly cropped face image) and computes the 512-d embedding vector.
    """
    if isinstance(image_input, (str, Path)):
        pil_img = load_visual_image(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input).__name__}")

    rgb_array = np.asarray(pil_img, dtype=np.uint8)
    insight_embedding: Optional[List[float]] = None
    detected_box: Optional[Tuple[int, int, int, int]] = None
    best_rgb = rgb_array

    # 1. InsightFace (GPU) via face_engine; retries on a padded canvas for tight Square Cropper reference queries.
    try:
        primary_hit = face_engine.extract_primary_face(rgb_array, is_reference_crop=True)
    except Exception as exc:
        logger.debug("face_engine reference extraction notice: %s", exc)
        primary_hit = None
    if primary_hit is not None:
        pf, used_rgb = primary_hit
        insight_embedding = pf.embedding
        detected_box = pf.box
        rgb_array = used_rgb
        best_rgb = used_rgb

    # 2. Try OpenCV YuNet / DNN / Haar face detector across 0, 90, 180, 270 degrees if needed
    if detected_box is None:
        for angle in (0, 90, 270, 180):
            cand_pil = pil_img if angle == 0 else pil_img.rotate(360 - angle, expand=True)
            cand_rgb = np.asarray(cand_pil, dtype=np.uint8)
            box = detect_primary_face_box(cand_rgb, min_resolution=32)
            if box is not None:
                detected_box = box
                best_rgb = cand_rgb
                break

    img_h, img_w = best_rgb.shape[:2]
    face_detected = detected_box is not None

    if detected_box is not None:
        fx, fy, fw, fh = detected_box
        core_face = best_rgb[fy : fy + fh, fx : fx + fw]
        pad_x = int(round(fw * FACE_CROP_MARGIN_RATIO))
        pad_y = int(round(fh * FACE_CROP_MARGIN_RATIO))
        px1 = max(0, fx - pad_x)
        py1 = max(0, fy - pad_y)
        px2 = min(img_w, fx + fw + pad_x)
        py2 = min(img_h, fy + fh + pad_y)
        preview_crop = Image.fromarray(best_rgb[py1:py2, px1:px2]).convert("RGB")
    else:
        # Fallback if the user selected a pre-cropped passport/portrait photo
        side = min(img_h, img_w)
        cy, cx = img_h // 2, img_w // 2
        half = max(16, int(side * 0.42))
        y1 = max(0, cy - half)
        y2 = min(img_h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(img_w, cx + half)
        core_face = best_rgb[y1:y2, x1:x2]
        preview_crop = Image.fromarray(core_face).convert("RGB")

    native_embedding = compute_face_embedding(core_face)
    primary_embedding = insight_embedding if insight_embedding is not None else native_embedding
    preview_b64 = image_to_base64_thumbnail(preview_crop, max_size=(160, 160), quality=82)

    return {
        "embedding": primary_embedding,
        "native_embedding": native_embedding,
        "insight_embedding": insight_embedding,
        "face_detected": face_detected,
        "reference_face_b64": preview_b64,
    }


def search_by_reference_face(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    db_manager: Optional[database.DatabaseManager] = None,
    similarity_threshold: float = 0.48,
) -> Dict[str, Any]:
    """
    Reverse Face Search:
    1. Extract 512-d face embedding from the reference sample image.
    2. Run Cosine Similarity comparison (threshold ~0.45-0.50, default 0.48) against ALL stored
       face embeddings in SQLite (`people.face_embedding` and `files.face_embedding`, plus backfilling
       any existing face crops in `data/faces/`).
    3. Return all matching verified records (both ID Vault documents and Gallery Photos) sorted
       by highest facial similarity.
    """
    active_db = db_manager or database.db
    ref_info = extract_reference_face_embedding(image_input)
    query_primary = ref_info["embedding"]
    query_native = ref_info["native_embedding"]

    def _best_sim(stored_vec: Optional[Sequence[float]]) -> float:
        if not stored_vec:
            return -1.0
        sim1 = compute_cosine_similarity(query_primary, stored_vec)
        sim2 = compute_cosine_similarity(query_native, stored_vec) if query_native is not query_primary else sim1
        return max(sim1, sim2)

    # 1. Compare against all person clusters in `people`
    people_rows = active_db.get_all_people(include_embeddings=True)
    matched_person_scores: Dict[int, float] = {}
    matched_people_details: List[Dict[str, Any]] = []

    for person in people_rows:
        pid = int(person["id"])
        p_emb = person.get("face_embedding")
        sim = _best_sim(p_emb)

        # Also check person's saved face thumbnail crop if embedding was missing
        if sim < similarity_threshold and person.get("face_thumbnail_path"):
            face_crop_path = BASE_DIR / str(person["face_thumbnail_path"])
            if face_crop_path.is_file():
                try:
                    with Image.open(face_crop_path) as fc_img:
                        fc_rgb = np.asarray(fc_img.convert("RGB"), dtype=np.uint8)
                    h, w = fc_rgb.shape[:2]
                    # Strip 20% margin to get tight core face
                    mx, my = int(w * 0.14), int(h * 0.14)
                    core = fc_rgb[my : max(my + 1, h - my), mx : max(mx + 1, w - mx)]
                    crop_emb = compute_face_embedding(core)
                    sim = max(sim, _best_sim(crop_emb))
                except Exception:
                    pass

        if sim >= similarity_threshold:
            matched_person_scores[pid] = sim
            p_copy = dict(person)
            p_copy.pop("face_embedding", None)
            p_copy["similarity"] = round(sim, 4)
            p_copy["similarity_percent"] = round(min(99.9, max(0.0, sim * 100.0)), 1)
            matched_people_details.append(p_copy)

    # 2. Compare against lightweight face candidates in `files`
    candidates = active_db.get_document_face_candidates()
    matched_candidates: List[Tuple[int, float]] = []

    for item in candidates:
        doc_id = int(item["id"])
        pid = item.get("person_id")
        doc_emb = item.get("face_embedding")

        # If file doesn't have its own face_embedding cached yet, check if `data/faces/face_<hash>.jpg` exists
        if not doc_emb and item.get("file_path"):
            file_hash_id = hashlib.sha1(str(item["file_path"]).lower().encode("utf-8")).hexdigest()[:16]
            saved_face_file = FACES_DIR / f"face_{file_hash_id}.jpg"
            if saved_face_file.is_file():
                try:
                    with Image.open(saved_face_file) as f_img:
                        f_rgb = np.asarray(f_img.convert("RGB"), dtype=np.uint8)
                    fh, fw = f_rgb.shape[:2]
                    mx, my = int(fw * 0.14), int(fh * 0.14)
                    core_f = f_rgb[my : max(my + 1, fh - my), mx : max(mx + 1, fw - mx)]
                    doc_emb = compute_face_embedding(core_f)
                    active_db.update_file_face_embedding(doc_id, doc_emb)
                except Exception:
                    pass

        file_sim = _best_sim(doc_emb) if doc_emb else -1.0
        cluster_sim = matched_person_scores.get(int(pid), -1.0) if pid is not None else -1.0
        best_doc_sim = max(file_sim, cluster_sim)

        if best_doc_sim >= similarity_threshold:
            matched_candidates.append((doc_id, best_doc_sim))

    # Fetch full document records ONLY for the matched IDs
    matched_ids = [m[0] for m in matched_candidates]
    full_docs_map = {int(d["id"]): d for d in active_db.get_files_by_ids(matched_ids)}
    matched_docs: List[Dict[str, Any]] = []

    for doc_id, best_doc_sim in matched_candidates:
        doc = full_docs_map.get(doc_id)
        if not doc:
            continue
        doc_copy = dict(doc)
        doc_copy.pop("face_embedding", None)
        doc_copy["face_similarity"] = round(best_doc_sim, 4)
        doc_copy["face_match_percent"] = round(min(99.9, max(0.0, best_doc_sim * 100.0)), 1)
        matched_docs.append(doc_copy)

    matched_docs.sort(key=lambda d: float(d.get("face_similarity", 0.0)), reverse=True)
    matched_people_details.sort(key=lambda p: float(p.get("similarity", 0.0)), reverse=True)

    return {
        "face_detected": ref_info["face_detected"],
        "reference_face_b64": ref_info["reference_face_b64"],
        "threshold": similarity_threshold,
        "matched_people": matched_people_details,
        "matched_documents": matched_docs,
        "total_matches": len(matched_docs),
    }


def compute_dhash(image: Union[Image.Image, np.ndarray]) -> str:
    """
    Compute a fast 64-bit perceptual difference hash (dHash) using OpenCV/NumPy.
    Resizes grayscale image to 9x8, compares adjacent horizontal pixels (8x8 = 64 bits),
    and returns a 16-character zero-padded hexadecimal string.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)

    try:
        import cv2

        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    except ImportError:
        pil_gray = Image.fromarray(rgb).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        resized = np.asarray(pil_gray, dtype=np.uint8)

    diff = resized[:, 1:] > resized[:, :-1]
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bool(bit))
    return f"{hash_int:016x}"


def _save_document_thumbnail(pil_img: Image.Image, file_hash_id: str) -> str:
    """Save a compressed preview thumbnail of a verified (upright) document to `data/thumbnails/`."""
    _ensure_data_dirs()
    thumb = pil_img.copy()
    try:
        thumb.thumbnail(THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
        filename = f"thumb_{file_hash_id}.jpg"
        dest_path = THUMBNAILS_DIR / filename
        thumb.save(dest_path, format="JPEG", quality=82, optimize=True)
        return f"data/thumbnails/{filename}"
    finally:
        try:
            thumb.close()
        except Exception:
            pass


def _crop_and_save_face(
    rgb_array: np.ndarray,
    face_box: Tuple[int, int, int, int],
    file_hash_id: str,
) -> Tuple[np.ndarray, str]:
    """
    Extract both the tight face core (for clean background-free embedding) and the
    20%-margin portrait crop, saving the 20%-margin crop to `data/faces/`.
    """
    _ensure_data_dirs()
    img_h, img_w = rgb_array.shape[:2]
    fx, fy, fw, fh = face_box

    core_face = rgb_array[fy : fy + fh, fx : fx + fw]

    pad_x = int(round(fw * FACE_CROP_MARGIN_RATIO))
    pad_y = int(round(fh * FACE_CROP_MARGIN_RATIO))
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(img_w, fx + fw + pad_x)
    y2 = min(img_h, fy + fh + pad_y)

    padded_crop = rgb_array[y1:y2, x1:x2]
    pil_crop = Image.fromarray(padded_crop).convert("RGB")
    try:
        pil_crop.thumbnail(FACE_THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
        filename = f"face_{file_hash_id}.jpg"
        dest_path = FACES_DIR / filename
        pil_crop.save(dest_path, format="JPEG", quality=88, optimize=True)
        return core_face, f"data/faces/{filename}"
    finally:
        try:
            pil_crop.close()
        except Exception:
            pass


GALLERY_FACE_MIN_SCORE = 0.55
GALLERY_FACE_MIN_SIDE = face_engine.MIN_FACE_RESOLUTION
GALLERY_DETECT_MAX_SIDE = 1600


def detect_gallery_faces(rgb_array: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect ALL faces >= 80x80 px in a general photo (GPU-accelerated InsightFace when available,
    OpenCV fallback otherwise). Returns a list sorted by face area (largest first):
      {"box": (x, y, w, h), "embedding": List[float] | None, "score": float, "source": "insightface"|"opencv"}
    """
    results: List[Dict[str, Any]] = []

    for df in face_engine.detect_faces(
        rgb_array,
        min_score=GALLERY_FACE_MIN_SCORE,
        min_side=GALLERY_FACE_MIN_SIDE,
        max_side=GALLERY_DETECT_MAX_SIDE,
    ):
        results.append({"box": df.box, "embedding": df.embedding, "score": df.score, "source": "insightface"})

    if not results:
        try:
            box = detect_primary_face_box(rgb_array, min_resolution=GALLERY_FACE_MIN_SIDE)
        except Exception as exc:
            logger.debug("OpenCV face fallback notice: %s", exc)
            box = None
        if box is not None:
            results.append({"box": box, "embedding": None, "score": 0.5, "source": "opencv"})

    results.sort(key=lambda r: r["box"][2] * r["box"][3], reverse=True)
    return results


def _store_gallery_photo(
    active_db: database.DatabaseManager,
    pil_img: Image.Image,
    faces: List[Dict[str, Any]],
    norm_path: str,
    mtime_str: str,
    file_size: int,
    file_hash_id: str,
    b64_thumb: Optional[str],
    ocr_text: Optional[str] = None,
    date_taken: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist an image containing faces as 'Face Gallery Photo' with its face embedding + person cluster."""
    rgb_array = np.asarray(pil_img, dtype=np.uint8)
    primary = faces[0]
    core_face, face_rel_path = _crop_and_save_face(rgb_array, primary["box"], file_hash_id)
    embedding = primary.get("embedding")
    threshold = INSIGHTFACE_CLUSTER_THRESHOLD
    if embedding is None:
        embedding = compute_face_embedding(core_face)
        threshold = FACE_MATCH_THRESHOLD
    person_id = match_or_create_person(
        embedding=embedding,
        face_thumbnail_rel_path=face_rel_path,
        db_manager=active_db,
        similarity_threshold=threshold,
    )
    thumb_path = _save_document_thumbnail(pil_img, file_hash_id)
    dhash_val = compute_dhash(pil_img)
    clean_text = (ocr_text or "").strip()
    note = f"{len(faces)} face(s) detected"
    file_id = active_db.upsert_file(
        file_path=norm_path,
        file_hash_or_mtime=mtime_str,
        file_size=file_size,
        is_document=True,
        doc_category="Face Gallery Photo",
        person_id=person_id,
        thumbnail_path=thumb_path,
        dhash=dhash_val,
        thumbnail_b64=b64_thumb,
        face_embedding=embedding,
        ocr_text=(f"{note}. {clean_text}" if clean_text else note),
        date_taken=date_taken,
    )
    return {
        "file_id": file_id,
        "file_path": norm_path,
        "is_document": True,
        "doc_category": "Face Gallery Photo",
        "person_id": person_id,
        "thumbnail_path": thumb_path,
        "thumbnail_b64": b64_thumb,
        "date_taken": date_taken,
        "face_count": len(faces),
        "status": "face_found",
    }


# -----------------------------------------------------------------------------
# Main File & Batch Processing Pipeline
# -----------------------------------------------------------------------------


def process_file(
    candidate: Union[str, Path, Dict[str, Any]],
    db_manager: Optional[database.DatabaseManager] = None,
    ocr_engine: Any = None,
    pdf_password: Optional[str] = None,
    scan_mode: str = "id_vault",
    force_rescan: bool = False,
    target_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Visually inspect a single candidate image/PDF through Steps A -> B -> C -> D
    and persist the result in `docuvault.db`.

    Supports two scan modes:
    - 'id_vault': Default visual OCR + Indian ID document classification & person tagging.
    - 'face_search': Deep Face Hunter mode. Bypasses OCR rules, extracts 512-d facial embeddings
      from all photos, and clusters faces using cosine similarity into 'Face Gallery Photo'.

    If `target_category` is provided (e.g. 'PAN Card', 'Aadhaar Card'):
      Documents not matching `target_category` are discarded and not saved to the database.
    """
    active_db = db_manager or database.db

    if isinstance(candidate, dict):
        raw_path = candidate["file_path"]
    else:
        raw_path = candidate

    norm_path = database.normalize_file_path(raw_path)

    try:
        st = os.stat(norm_path)
        file_size = int(st.st_size)
        mtime_float = float(st.st_mtime)
        mtime_str = database.normalize_hash_or_mtime(mtime_float)
    except OSError as exc:
        logger.warning("Skipping inaccessible file '%s': %s", norm_path, exc)
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime="0.0",
            file_size=0,
            is_document=False,
            doc_category=database.UNREADABLE_OR_CORRUPT,
            ocr_text=f"File access error: {exc}",
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.UNREADABLE_OR_CORRUPT,
            "status": "inaccessible_file",
            "error": str(exc),
        }

    # Record into Ghost Cache for Incremental Scanning
    active_db.record_scan_history(norm_path, file_size, mtime_float)

    # Skip if already up to date in SQLite (unless unlocking with a password)
    if pdf_password is None and not force_rescan and active_db.is_file_scanned(norm_path, mtime_str):
        existing = active_db.get_file_by_path(norm_path)
        return {
            "file_id": existing["id"] if existing else None,
            "file_path": norm_path,
            "is_document": existing["is_document"] if existing else False,
            "doc_category": existing["doc_category"] if existing else database.NOT_A_DOCUMENT,
            "status": "skipped_already_scanned",
        }

    # Open image or render PDF Page 1 (with password support & robust corruption handling)
    try:
        pil_img = load_visual_image(norm_path, password=pdf_password)
    except PasswordProtectedError as pass_err:
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=True,
            doc_category=database.PASSWORD_PROTECTED,
            ocr_text="Password protected encrypted PDF. Click 'Unlock PDF' to decrypt.",
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": True,
            "doc_category": database.PASSWORD_PROTECTED,
            "status": "password_protected",
            "error": str(pass_err),
        }
    except VisualLoadError as load_err:
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.UNREADABLE_OR_CORRUPT,
            ocr_text=f"Unreadable file: {load_err.reason}",
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.UNREADABLE_OR_CORRUPT,
            "status": "unreadable_corrupt",
            "error": load_err.reason,
        }
    except Exception as unk_err:
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.UNREADABLE_OR_CORRUPT,
            ocr_text=f"Unexpected load error: {unk_err}",
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.UNREADABLE_OR_CORRUPT,
            "status": "unreadable_unknown",
            "error": str(unk_err),
        }

    # Step A: Dimension pre-check (< 300x300 px skipped and recorded as Not a Document)
    width, height = pil_img.size
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.NOT_A_DOCUMENT,
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.NOT_A_DOCUMENT,
            "status": "tiny_dimensions_skipped",
        }

    file_hash_id = hashlib.sha1(norm_path.lower().encode("utf-8")).hexdigest()[:16]
    photo_date_taken = extract_exif_date_taken(
        norm_path,
        fallback_mtime=mtime_float,
        use_mtime_fallback=True,
    )

    # Compute in-memory base64 thumbnail for lightning-fast UI loading
    b64_thumb: Optional[str] = None
    try:
        b64_thumb = image_to_base64_thumbnail(pil_img)
    except Exception as b64_err:
        logger.debug("Base64 thumb conversion notice: %s", b64_err)

    # -------------------------------------------------------------------------
    # DEEP FACE HUNTER MODE: Ignore OCR rules, extract 512-d embeddings from all photos
    # -------------------------------------------------------------------------
    if scan_mode == "face_search":
        rgb_array = np.asarray(pil_img, dtype=np.uint8)
        gallery_faces = detect_gallery_faces(rgb_array)

        if gallery_faces:
            return _store_gallery_photo(
                active_db,
                pil_img,
                gallery_faces,
                norm_path,
                mtime_str,
                file_size,
                file_hash_id,
                b64_thumb,
                date_taken=photo_date_taken,
            )
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.NOT_A_DOCUMENT,
            date_taken=photo_date_taken,
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.NOT_A_DOCUMENT,
            "status": "no_face_in_photo",
        }

    # -------------------------------------------------------------------------
    # ID VAULT MODE: Cascading Filter (Aadhaar Fast-Track) or Standard Inspection
    # -------------------------------------------------------------------------
    clean_target = (
        target_category.strip()
        if (target_category and target_category.strip() and target_category.strip().lower() != "all")
        else None
    )
    is_aadhaar_fast_track = bool(
        (clean_target and "aadhaar" in clean_target.lower())
        or str(scan_mode).strip().lower() in ("aadhaar_fast_track", "aadhaar_fast", "fast_track_aadhaar", "aadhaar")
    )

    if is_aadhaar_fast_track:
        # Pre-flight Fast OCR Pass (0° orientation, no angle classification overhead)
        text_fast, conf_fast, is_vert_fast = _extract_ocr_detailed(
            pil_img, ocr_engine=ocr_engine, allow_angle_cls=False
        )

        is_confirmed, gate1_passed, gate2_passed, extracted_id = validate_aadhaar_cascading(text_fast)

        if not gate1_passed:
            # Immediate Short-Circuit: Neither "GOVERNMENT" nor "UIDAI" found
            # Discard immediately, call del on image tensors, run gc.collect()
            try:
                pil_img.close()
            except Exception:
                pass
            del pil_img
            import gc
            gc.collect()
            return {
                "file_id": None,
                "file_path": norm_path,
                "is_document": False,
                "doc_category": database.NOT_A_DOCUMENT,
                "status": "skipped_aadhaar_gate1_short_circuit",
                "gate1_passed": False,
                "gate2_passed": False,
            }

        # Gate 1 passed! Now check Gate 2 (Secondary Verification)
        oriented_img = pil_img
        applied_rotation = 0
        if not gate2_passed and is_vert_fast:
            # If text is vertical/sideways, test 90° rotation to check if Gate 2 passes
            img_90 = rotate_pil_clockwise(pil_img, 90)
            text_90, conf_90, _ = _extract_ocr_detailed(
                img_90, ocr_engine=ocr_engine, allow_angle_cls=False
            )
            is_conf_90, g1_90, g2_90, id_90 = validate_aadhaar_cascading(text_90)
            if g2_90:
                is_confirmed, gate2_passed, extracted_id = True, True, id_90
                text_fast, conf_fast = text_90, conf_90
                oriented_img = img_90
                applied_rotation = 90
            else:
                try:
                    img_90.close()
                except Exception:
                    pass

        if not gate2_passed:
            # Gate 2 failed: Not an actual Aadhaar card
            try:
                pil_img.close()
            except Exception:
                pass
            if oriented_img is not pil_img:
                try:
                    oriented_img.close()
                except Exception:
                    pass
            del pil_img
            import gc
            gc.collect()
            return {
                "file_id": None,
                "file_path": norm_path,
                "is_document": False,
                "doc_category": database.NOT_A_DOCUMENT,
                "status": "skipped_aadhaar_gate2_failed",
                "gate1_passed": True,
                "gate2_passed": False,
            }

        # Confirmed Aadhaar Card!
        ocr_text = text_fast
        avg_conf = conf_fast
        is_document = True
        doc_category = "Aadhaar Card"
        extracted_id_number = extracted_id

    else:
        # Standard multi-category inspection with auto-rotation & classification
        try:
            (
                oriented_img,
                ocr_text,
                avg_conf,
                is_document,
                doc_category,
                extracted_id_number,
                applied_rotation,
            ) = inspect_with_auto_rotation(pil_img, ocr_engine=ocr_engine)
        except Exception as exc:
            logger.warning("OCR analysis failed on '%s': %s", norm_path, exc)
            file_id = active_db.upsert_file(
                file_path=norm_path,
                file_hash_or_mtime=mtime_str,
                file_size=file_size,
                is_document=False,
                doc_category=database.UNREADABLE_OR_CORRUPT,
                ocr_text=f"Inspection error: {exc}",
                thumbnail_b64=b64_thumb,
                date_taken=photo_date_taken,
            )
            return {
                "file_id": file_id,
                "file_path": norm_path,
                "is_document": False,
                "doc_category": database.UNREADABLE_OR_CORRUPT,
                "status": "inspection_failed",
            }

        # Check targeted document category filter if specified (skips unnecessary face detection if not matching)
        if clean_target is not None:
            if not is_document or doc_category != clean_target:
                # File does not match target category (e.g. searching ONLY PAN Cards) -> discard/skip
                return {
                    "file_id": None,
                    "file_path": norm_path,
                    "is_document": False,
                    "doc_category": database.NOT_A_DOCUMENT,
                    "status": "skipped_not_target_category",
                }

    # If OCR indicates this is not a verified document, detect if this is a general Face Gallery Photo
    gallery_faces: List[Dict[str, Any]] = []
    if not is_document:
        try:
            gallery_faces = detect_gallery_faces(np.asarray(pil_img, dtype=np.uint8))
        except Exception as exc:
            logger.debug("Face check on non-document '%s': %s", norm_path, exc)
            gallery_faces = []

    if not is_document and gallery_faces:
        # No document keywords but faces present -> Face Gallery Photo (with embeddings)
        return _store_gallery_photo(
            active_db,
            pil_img,
            gallery_faces,
            norm_path,
            mtime_str,
            file_size,
            file_hash_id,
            b64_thumb,
            ocr_text,
            date_taken=photo_date_taken,
        )

    if not is_document and len((ocr_text or "").strip()) < MIN_OCR_TEXT_LENGTH:
        # No readable text AND no faces -> Unidentified / Unreadable
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.UNREADABLE_OR_CORRUPT,
            ocr_text="No readable text and no faces detected.",
            thumbnail_b64=b64_thumb,
            date_taken=photo_date_taken,
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.UNREADABLE_OR_CORRUPT,
            "applied_rotation": applied_rotation,
            "status": "unidentified_no_text_no_face",
        }

    if not is_document:
        file_id = active_db.upsert_file(
            file_path=norm_path,
            file_hash_or_mtime=mtime_str,
            file_size=file_size,
            is_document=False,
            doc_category=database.NOT_A_DOCUMENT,
            extracted_id_number=None,
            ocr_text=None,
            person_id=None,
            thumbnail_path=None,
            dhash=None,
            date_taken=photo_date_taken,
        )
        return {
            "file_id": file_id,
            "file_path": norm_path,
            "is_document": False,
            "doc_category": database.NOT_A_DOCUMENT,
            "extracted_id_number": None,
            "person_id": None,
            "thumbnail_path": None,
            "dhash": None,
            "applied_rotation": applied_rotation,
            "status": "classified_non_document",
        }

    # Step D: Verified Document -> Upright Thumbnail + 64-bit dHash + Upright Face Clustering
    thumbnail_rel_path: Optional[str] = None
    doc_dhash: Optional[str] = None
    person_id: Optional[int] = None
    doc_face_embedding: Optional[List[float]] = None

    try:
        thumbnail_rel_path = _save_document_thumbnail(oriented_img, file_hash_id)
    except Exception as exc:
        logger.warning("Failed to save document thumbnail for '%s': %s", norm_path, exc)

    try:
        doc_dhash = compute_dhash(oriented_img)
    except Exception as exc:
        logger.warning("Failed to compute dHash for '%s': %s", norm_path, exc)

    try:
        rgb_array = np.asarray(oriented_img, dtype=np.uint8)
        # Use InsightFace via face_engine if available for reliable 512-d normalized embedding on IDs
        doc_face_hit = face_engine.extract_primary_face(rgb_array) if face_engine.is_available() else None
        if doc_face_hit is not None:
            df, _ = doc_face_hit
            core_face, face_rel_path = _crop_and_save_face(rgb_array, df.box, file_hash_id)
            doc_face_embedding = df.embedding
            person_id = match_or_create_person(
                embedding=doc_face_embedding,
                face_thumbnail_rel_path=face_rel_path,
                db_manager=active_db,
                similarity_threshold=INSIGHTFACE_CLUSTER_THRESHOLD,
            )
        else:
            face_box = detect_primary_face_box(rgb_array)
            if face_box is not None:
                core_face, face_rel_path = _crop_and_save_face(rgb_array, face_box, file_hash_id)
                doc_face_embedding = compute_face_embedding(core_face)
                person_id = match_or_create_person(
                    embedding=doc_face_embedding,
                    face_thumbnail_rel_path=face_rel_path,
                    db_manager=active_db,
                    similarity_threshold=FACE_MATCH_THRESHOLD,
                )
    except Exception as exc:
        logger.warning("Face detection/clustering skipped on '%s': %s", norm_path, exc)

    # Step E: Extract Structured Metadata (DOB & Expiry Date)
    dob, expiry_date = extract_metadata(ocr_text, doc_category)

    file_id = active_db.upsert_file(
        file_path=norm_path,
        file_hash_or_mtime=mtime_str,
        file_size=file_size,
        is_document=True,
        doc_category=doc_category,
        extracted_id_number=extracted_id_number,
        dob=dob,
        expiry_date=expiry_date,
        ocr_text=ocr_text,
        person_id=person_id,
        thumbnail_path=thumbnail_rel_path,
        dhash=doc_dhash,
        thumbnail_b64=b64_thumb,
        face_embedding=doc_face_embedding,
        date_taken=photo_date_taken,
    )

    return {
        "file_id": file_id,
        "file_path": norm_path,
        "is_document": True,
        "doc_category": doc_category,
        "extracted_id_number": extracted_id_number,
        "dob": dob,
        "expiry_date": expiry_date,
        "date_taken": photo_date_taken,
        "person_id": person_id,
        "thumbnail_path": thumbnail_rel_path,
        "thumbnail_b64": b64_thumb,
        "dhash": doc_dhash,
        "applied_rotation": applied_rotation,
        "ocr_confidence": avg_conf,
        "status": "verified_document",
    }


# -----------------------------------------------------------------------------
# Targeted Folder Face Search (scans ONLY the chosen folder, GPU InsightFace)
# -----------------------------------------------------------------------------
FOLDER_SEARCH_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".tif", ".tiff")
_SKIP_DIR_PREFIXES = (".", "$")


def list_images_in_folder(folder_path: Union[str, Path], recursive: bool = True) -> List[str]:
    """Collect image files under `folder_path` (skipping hidden/system folders)."""
    root = database.normalize_file_path(folder_path)
    found: List[str] = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(_SKIP_DIR_PREFIXES)]
        for name in filenames:
            if name.lower().endswith(FOLDER_SEARCH_IMAGE_EXTS):
                found.append(os.path.join(dirpath, name))
        if not recursive:
            break
    return found


def search_folder_for_face(
    folder_path: Union[str, Path],
    query_embedding: Sequence[float],
    similarity_threshold: float = face_engine.DEFAULT_MATCH_THRESHOLD,
    db_manager: Optional[database.DatabaseManager] = None,
    recursive: bool = True,
    progress_callback: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    stop_event: Optional[threading.Event] = None,
    phase_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Targeted Folder Face Search (scans ONLY `folder_path`, recursively):

      Phase 1 - EXTRACT: detect ALL faces (group photos included) in EVERY image and keep their
                512-d embeddings.
      Phase 2 - MATCH:   cosine similarity of ALL extracted faces vs. the target embedding
                (threshold ~0.45); every image with at least one matching face is a hit.
      Phase 3 - INDEX:   upsert hits into SQLite (one row per file path, never duplicated) so global
                person clustering / Face Gallery Photos stay in sync. Already-indexed ID documents
                keep their category.
    """
    active_db = db_manager or database.db
    face_engine.require_app()
    images = list_images_in_folder(folder_path, recursive=recursive)
    total = len(images)
    unreadable = 0
    stopped = lambda: bool(stop_event is not None and stop_event.is_set())  # noqa: E731

    # ---- Phase 1: extract every face from every image ----
    if phase_callback:
        phase_callback("extracting")
    extracted: List[Tuple[str, List[face_engine.DetectedFace]]] = []
    scanned = 0
    for idx, img_path in enumerate(images, start=1):
        if stopped():
            break
        try:
            rgb = face_engine.load_rgb_image(img_path)
            faces = [f for f in face_engine.detect_faces_thorough(rgb) if f.embedding is not None]
            if faces:
                extracted.append((img_path, faces))
            del rgb
        except face_engine.UnreadableImageError as exc:
            unreadable += 1
            logger.debug("Skipping unreadable image: %s", exc)
        except Exception as exc:
            unreadable += 1
            logger.warning("Face extraction error on '%s': %s", img_path, exc)
        scanned = idx
        if idx % 64 == 0:
            import gc
            gc.collect()
        if progress_callback is not None:
            try:
                progress_callback(idx, total, None)
            except Exception:
                pass

    # ---- Phase 2: compare ALL faces against the target ----
    if phase_callback:
        phase_callback("matching")
    hits: List[Tuple[str, float, int]] = []
    seen_paths: set = set()
    for img_path, faces in extracted:
        key = database.normalize_file_path(img_path).lower()
        if key in seen_paths:
            continue
        sim, _ = face_engine.best_match_similarity(query_embedding, faces)
        if sim >= similarity_threshold:
            seen_paths.add(key)
            hits.append((img_path, sim, len(faces)))
    hits.sort(key=lambda h: h[1], reverse=True)

    # ---- Phase 3: index hits without duplicating records ----
    if phase_callback:
        phase_callback("indexing")
    matches: List[Dict[str, Any]] = []
    for img_path, sim, face_count in hits:
        try:
            existing = active_db.get_file_by_path(img_path)
            if existing is None or not existing.get("is_document"):
                process_file(img_path, db_manager=active_db, scan_mode="face_search", force_rescan=True)
        except Exception as idx_exc:
            logger.debug("Indexing matched file failed '%s': %s", img_path, idx_exc)
        record = active_db.get_file_by_path(img_path)
        if record is None:
            continue
        record = dict(record)
        record.pop("face_embedding", None)
        record["face_similarity"] = round(sim, 4)
        record["face_match_percent"] = round(min(99.9, max(0.0, sim * 100.0)), 1)
        record["faces_in_photo"] = face_count
        matches.append(record)
        if progress_callback is not None:
            try:
                progress_callback(scanned, total, record)
            except Exception:
                pass

    return {
        "folder": database.normalize_file_path(folder_path),
        "total_images": total,
        "scanned": scanned,
        "unreadable": unreadable,
        "cancelled": stopped(),
        "matches": matches,
    }


def process_candidates(
    candidate_files: Sequence[Union[str, Path, Dict[str, Any]]],
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
    db_manager: Optional[database.DatabaseManager] = None,
    scan_mode: str = "id_vault",
    target_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a sequence of candidate files with optional progress reporting,
    targeted filtering (including Aadhaar Fast-Track), and graceful cancellation.
    """
    active_db = db_manager or database.db
    total = len(candidate_files)
    processed_count = 0
    documents_found = 0
    non_documents_count = 0
    errors_count = 0

    for idx, candidate in enumerate(candidate_files, start=1):
        if stop_event is not None and stop_event.is_set():
            logger.info("Scan cancelled by stop_event at %d/%d.", idx - 1, total)
            break

        try:
            result = process_file(
                candidate,
                db_manager=active_db,
                scan_mode=scan_mode,
                target_category=target_category,
            )
            processed_count += 1
            if result.get("is_document"):
                documents_found += 1
            else:
                non_documents_count += 1

            # Cascading Filter & Short-Circuit skip handler
            if result.get("status") in (
                "skipped_aadhaar_gate1_short_circuit",
                "skipped_aadhaar_gate2_failed",
                "skipped_not_target_category",
            ):
                if progress_callback is not None:
                    try:
                        progress_callback(idx, total, result)
                    except Exception as cb_exc:
                        logger.debug("Progress callback error: %s", cb_exc)
                continue
        except Exception as exc:
            errors_count += 1
            logger.error("Unhandled error processing '%s': %s", candidate, exc)
            result = {
                "file_id": None,
                "file_path": str(candidate),
                "is_document": False,
                "doc_category": database.NOT_A_DOCUMENT,
                "status": f"error: {exc}",
            }

        if progress_callback is not None:
            try:
                progress_callback(idx, total, result)
            except Exception as cb_exc:
                logger.debug("Progress callback error: %s", cb_exc)

    return {
        "total_candidates": total,
        "processed": processed_count,
        "documents_found": documents_found,
        "non_documents": non_documents_count,
        "errors": errors_count,
    }


if __name__ == "__main__":
    import scanner

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    database.init_db()
    candidates = scanner.find_all_candidate_files()
    print(f"Found {len(candidates)} candidate files to inspect.")
    summary = process_candidates(
        candidates,
        progress_callback=lambda cur, tot, res: print(
            f"[{cur}/{tot}] {res['doc_category']} <- {res['file_path']}"
        ),
    )
    print(f"Scan complete: {summary}")

