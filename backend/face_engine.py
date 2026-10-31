"""
DocuVault AI - Dedicated Face Engine (face_engine.py)

Single home for everything InsightFace related:
  * CUDA / CPU execution-provider selection with automatic GPU -> CPU fallback
  * Lazy, thread-safe `FaceAnalysis` (buffalo_s, detection + recognition only)
  * Robust image loading (EXIF orientation, corrupt/unreadable files)
  * Multi-face detection with 512-d L2-normalised embeddings
  * Cosine similarity helpers

100% offline: models are cached under `models/`. No cloud APIs are used.
"""

from __future__ import annotations

import logging
import os
import site
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("docuvault.face_engine")

# InsightFace still calls scikit-image's deprecated `SimilarityTransform.estimate`; harmless, so keep logs clean.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else BACKEND_DIR
BASE_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "buffalo_s"
EMBEDDING_DIM = 512
DET_SIZE: Tuple[int, int] = (640, 640)
DEFAULT_MIN_SCORE = 0.55
# Strict minimum face bounding-box resolution (80x80 px) to eliminate background crowd / junk face pollution
MIN_FACE_RESOLUTION = 80
DEFAULT_MIN_SIDE = MIN_FACE_RESOLUTION
DEFAULT_MAX_SIDE = 1600
DEFAULT_MATCH_THRESHOLD = 0.45
# Global clustering threshold: lowered from 0.45/0.50 down to 0.38
# Tolerates accessories (sunglasses, spectacles, hats), angles, and lighting changes
INITIAL_CLUSTERING_THRESHOLD = 0.38

ImageInput = Union[str, Path, np.ndarray, "Any"]  # path, RGB ndarray or PIL.Image


# -----------------------------------------------------------------------------
# Errors & data types
# -----------------------------------------------------------------------------
class FaceEngineError(Exception):
    """Base class for face engine failures."""


class FaceEngineUnavailable(FaceEngineError):
    """InsightFace could not be initialised (missing package/model)."""


class UnreadableImageError(FaceEngineError):
    """The image is missing, corrupt or of an unsupported format."""


@dataclass(frozen=True)
class DetectedFace:
    """A detected face: `box` is (x, y, w, h) in original-image pixels."""

    box: Tuple[int, int, int, int]
    score: float
    embedding: Optional[List[float]]

    @property
    def area(self) -> int:
        return int(self.box[2]) * int(self.box[3])


# -----------------------------------------------------------------------------
# GPU / provider handling
# -----------------------------------------------------------------------------
def init_nvidia_dll_paths() -> None:
    """Register pip-installed NVIDIA CUDA/cuDNN DLL folders before onnxruntime loads."""
    try:
        for sp in site.getsitepackages():
            for b in Path(sp).glob("nvidia/*/bin"):
                if not b.is_dir():
                    continue
                b_str = str(b)
                if b_str not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = b_str + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(b_str)
                    except Exception:
                        pass
    except Exception:
        pass


def cuda_available() -> bool:
    """True if onnxruntime exposes the CUDAExecutionProvider."""
    try:
        init_nvidia_dll_paths()
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


_APP: Any = None
_APP_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_INIT_FAILED = False


def _build_app(providers: Sequence[Any], ctx_id: int) -> Any:
    from insightface.app import FaceAnalysis

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(
        name=MODEL_NAME,
        root=str(MODELS_DIR),
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    return app


def get_insightface_app() -> Any:
    """
    Lazily create the singleton InsightFace engine. CUDA is prioritised; if CUDA
    initialisation fails for any reason it falls back to CPU. Returns None if
    InsightFace is unusable (callers then use their OpenCV fallback).
    """
    global _APP, _INIT_FAILED
    if _APP is not None:
        return _APP
    with _APP_LOCK:
        if _APP is not None:
            return _APP
        if _INIT_FAILED:
            return None

        init_nvidia_dll_paths()
        attempts: List[Tuple[Sequence[Any], int]] = []
        if cuda_available():
            # Try configured CUDAExecutionProvider options to bound memory arena
            cuda_opts = {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            }
            attempts.append(([("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"], 0))
            attempts.append((["CUDAExecutionProvider", "CPUExecutionProvider"], 0))
        else:
            logger.warning("CUDAExecutionProvider unavailable for InsightFace; using CPU.")
        attempts.append((["CPUExecutionProvider"], -1))

        last_exc: Optional[Exception] = None
        for providers, ctx_id in attempts:
            try:
                app = _build_app(providers, ctx_id)
                _APP = app
                try:
                    active = {
                        name: m.session.get_providers()[0]
                        for name, m in app.models.items()
                        if getattr(m, "session", None) is not None
                    }
                    logger.info("InsightFace %s ready. Active providers: %s", MODEL_NAME, active)
                except Exception:
                    logger.info("InsightFace %s ready (providers=%s).", MODEL_NAME, providers)
                return _APP
            except Exception as exc:  # GPU init failure -> try next (CPU)
                last_exc = exc
                logger.warning("InsightFace init with %s failed: %s", providers, exc)

        _INIT_FAILED = True
        logger.warning("InsightFace unavailable (%s). OpenCV fallback will be used.", last_exc)
        return None


def is_available() -> bool:
    return get_insightface_app() is not None


def require_app() -> Any:
    app = get_insightface_app()
    if app is None:
        raise FaceEngineUnavailable(
            "InsightFace is not available. Install insightface + onnxruntime and ensure models/buffalo_s exists."
        )
    return app


# -----------------------------------------------------------------------------
# Similarity
# -----------------------------------------------------------------------------
def compute_cosine_similarity(vec_a: Optional[Sequence[float]], vec_b: Optional[Sequence[float]]) -> float:
    """Cosine similarity in [-1, 1]; returns -1.0 for missing/mismatched/zero vectors."""
    if vec_a is None or vec_b is None:
        return -1.0
    if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
        return -1.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return -1.0
    return float(np.dot(a / na, b / nb))


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------
def load_rgb_image(source: ImageInput) -> np.ndarray:
    """Load a path / PIL image / ndarray into an RGB uint8 array (EXIF-corrected)."""
    from PIL import Image, ImageOps

    try:
        if isinstance(source, np.ndarray):
            arr = source
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
        if isinstance(source, Image.Image):
            return np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
        path = Path(str(source))
        if not path.is_file():
            raise UnreadableImageError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise UnreadableImageError(f"Zero-byte image file: {path}")
        with Image.open(path) as opened:
            opened.load()
            return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    except UnreadableImageError:
        raise
    except Exception as exc:
        raise UnreadableImageError(f"Unreadable image '{source if isinstance(source, (str, Path)) else type(source).__name__}': {exc}") from exc


# -----------------------------------------------------------------------------
# Detection & embeddings
# -----------------------------------------------------------------------------
def detect_faces(
    rgb: np.ndarray,
    min_score: float = DEFAULT_MIN_SCORE,
    min_side: int = DEFAULT_MIN_SIDE,
    max_side: int = DEFAULT_MAX_SIDE,
    with_embeddings: bool = True,
    enforce_min_resolution: bool = True,
) -> List[DetectedFace]:
    """
    Detect every face in an RGB image with InsightFace and return them sorted by area (largest first).
    Strictly discards any detected face smaller than 80x80 pixels (`MIN_FACE_RESOLUTION`) when
    `enforce_min_resolution=True` to prevent background crowds from polluting People Clusters.
    Returns [] when InsightFace is unavailable or inference fails (never raises on bad pixels).
    """
    app = get_insightface_app()
    if app is None or rgb is None or rgb.size == 0:
        return []

    import cv2

    img_h, img_w = rgb.shape[:2]
    effective_min_side = max(int(min_side), MIN_FACE_RESOLUTION) if enforce_min_resolution else max(1, int(min_side))
    if img_w < effective_min_side or img_h < effective_min_side:
        return []

    scale = 1.0
    work = rgb
    longest = max(img_h, img_w)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        work = cv2.resize(rgb, (max(1, int(img_w * scale)), max(1, int(img_h * scale))), interpolation=cv2.INTER_AREA)

    try:
        bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
        with _INFERENCE_LOCK:
            faces = app.get(bgr) or []
    except Exception as exc:
        logger.warning("InsightFace inference failed: %s", exc)
        return []

    results: List[DetectedFace] = []
    for f in faces:
        score = float(getattr(f, "det_score", 1.0))
        if score < min_score:
            continue
        x1, y1, x2, y2 = [float(v) / scale for v in f.bbox[:4]]
        fx, fy = max(0, int(round(x1))), max(0, int(round(y1)))
        fw = min(img_w - fx, int(round(x2 - x1)))
        fh = min(img_h - fy, int(round(y2 - y1)))
        # Strict Junk Face Pollution Filter: discard any face bounding box < 80x80 pixels before embedding extraction
        if fw < effective_min_side or fh < effective_min_side:
            continue
        emb: Optional[List[float]] = None
        if with_embeddings:
            # Ensure proper L2 normalization of 512-d feature embedding before usage
            ne = getattr(f, "normed_embedding", None)
            if ne is None and hasattr(f, "embedding") and f.embedding is not None:
                raw_emb = np.asarray(f.embedding, dtype=np.float32).ravel()
                raw_norm = float(np.linalg.norm(raw_emb))
                if raw_norm > 1e-8:
                    ne = raw_emb / raw_norm
            if ne is not None:
                arr = np.asarray(ne, dtype=np.float32).ravel()
                if arr.size == EMBEDDING_DIM:
                    arr_norm = float(np.linalg.norm(arr))
                    if arr_norm > 1e-8:
                        arr = arr / arr_norm
                    emb = [round(float(v), 6) for v in arr.tolist()]
        results.append(DetectedFace(box=(fx, fy, fw, fh), score=score, embedding=emb))

    results.sort(key=lambda d: d.area, reverse=True)
    return results


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    union = float(a[2] * a[3] + b[2] * b[3]) - inter
    return inter / union if union > 0 else 0.0


def detect_faces_thorough(
    rgb: np.ndarray,
    min_score: float = 0.5,
    min_side: int = MIN_FACE_RESOLUTION,
) -> List[DetectedFace]:
    """
    Find EVERY qualifying face (>= 80x80 px) in a (possibly large group) photo. A normal pass runs on
    the downscaled full image; large images additionally run on 4 overlapping tiles at native resolution
    so medium faces in high-res group photos are detected while background crowd faces (< 80x80 px) are
    strictly discarded. Duplicate detections (IoU > 0.4) are merged keeping the best score.
    """
    if rgb is None or rgb.size == 0:
        return []
    strict_min_side = max(int(min_side), MIN_FACE_RESOLUTION)
    img_h, img_w = rgb.shape[:2]
    found = detect_faces(rgb, min_score=min_score, min_side=strict_min_side, max_side=DEFAULT_MAX_SIDE)

    if max(img_h, img_w) >= 900:
        tw, th = int(img_w * 0.6), int(img_h * 0.6)
        for ty in (0, img_h - th):
            for tx in (0, img_w - tw):
                tile = np.ascontiguousarray(rgb[ty : ty + th, tx : tx + tw])
                # Cap tile max_side to 2048 to prevent VRAM spikes on ultra-high-resolution images
                for f in detect_faces(tile, min_score=min_score, min_side=strict_min_side, max_side=2048):
                    x, y, w, h = f.box
                    if w < strict_min_side or h < strict_min_side:
                        continue
                    found.append(DetectedFace(box=(x + tx, y + ty, w, h), score=f.score, embedding=f.embedding))

    found.sort(key=lambda d: d.score, reverse=True)
    merged: List[DetectedFace] = []
    for f in found:
        if all(_iou(f.box, m.box) <= 0.4 for m in merged):
            merged.append(f)
    merged.sort(key=lambda d: d.area, reverse=True)
    return merged


def extract_primary_face(
    rgb: np.ndarray,
    is_reference_crop: bool = False,
) -> Optional[Tuple[DetectedFace, np.ndarray]]:
    """
    Largest face + its embedding in `rgb`.
    - For database indexing (`is_reference_crop=False`), strictly enforces the 80x80 minimum face
      bounding-box resolution on the original image so tiny background/watermark faces are ignored.
    - For user-cropped query samples from Square Cropper (`is_reference_crop=True`, never stored in DB),
      retries on a padded canvas so tight 512x512 crops always extract cleanly.
    Returns (face, image_used) or None.
    """
    if not is_available() or rgb is None or rgb.size == 0:
        return None
    import cv2

    if not is_reference_crop:
        # Strict 80x80 px check on original document/photo before saving to database
        direct_faces = [
            f
            for f in detect_faces(
                rgb,
                min_score=0.45,
                min_side=MIN_FACE_RESOLUTION,
                max_side=DEFAULT_MAX_SIDE,
                enforce_min_resolution=True,
            )
            if f.embedding is not None and f.box[2] >= MIN_FACE_RESOLUTION and f.box[3] >= MIN_FACE_RESOLUTION
        ]
        if direct_faces:
            return direct_faces[0], rgb
        return None

    # Reference query image (e.g., 512x512 Square Cropper output)
    work_rgb = rgb
    if min(work_rgb.shape[:2]) < 160:
        scale_up = 256.0 / float(max(1, min(work_rgb.shape[:2])))
        work_rgb = cv2.resize(
            work_rgb,
            (int(round(work_rgb.shape[1] * scale_up)), int(round(work_rgb.shape[0] * scale_up))),
            interpolation=cv2.INTER_CUBIC,
        )
    pad = int(round(max(work_rgb.shape[:2]) * 0.5))
    padded = cv2.copyMakeBorder(work_rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    for candidate in (work_rgb, padded):
        faces = detect_faces(
            candidate,
            min_score=0.35,
            min_side=32,
            max_side=0,
            enforce_min_resolution=False,
        )
        faces = [f for f in faces if f.embedding is not None]
        if faces:
            return faces[0], candidate
    return None


def best_match_similarity(
    query_embedding: Sequence[float], faces: Sequence[DetectedFace]
) -> Tuple[float, Optional[DetectedFace]]:
    """Highest cosine similarity between `query_embedding` and any face in `faces`."""
    best = -1.0
    best_face: Optional[DetectedFace] = None
    for f in faces:
        sim = compute_cosine_similarity(query_embedding, f.embedding)
        if sim > best:
            best, best_face = sim, f
    return best, best_face


# -----------------------------------------------------------------------------
# Agglomerative Fallback / Post-Scan Cluster Refinement
# -----------------------------------------------------------------------------
def agglomerative_cluster_cleanup(
    db_manager: Any,
    merge_threshold: float = INITIAL_CLUSTERING_THRESHOLD,
) -> int:
    """
    Run an agglomerative clustering pass on existing people records in SQLite.
    Automatically merges clusters whose centroid embeddings have a cosine similarity
    >= `merge_threshold` (e.g. 0.38-0.40). This recovers identical people split into
    different clusters due to accessories (sunglasses), varying lighting, or camera angles.

    Returns the number of merged clusters.
    """
    people_rows = db_manager.get_all_people(include_embeddings=True)
    active_people = [
        p for p in people_rows
        if int(p.get("document_count") or 0) > 0 and p.get("face_embedding")
    ]
    if len(active_people) < 2:
        return 0

    # Sort so larger clusters (primary identity) are prioritized as target
    active_people.sort(key=lambda x: int(x.get("document_count") or 0), reverse=True)

    parent: Dict[int, int] = {int(p["id"]): int(p["id"]) for p in active_people}
    rank: Dict[int, int] = {int(p["id"]): idx for idx, p in enumerate(active_people)}

    def _find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            # Deterministically union into root with higher document count (lowest rank index)
            if rank[ri] <= rank[rj]:
                parent[rj] = ri
            else:
                parent[ri] = rj

    n = len(active_people)
    for i in range(n):
        emb_i = active_people[i].get("face_embedding")
        for j in range(i + 1, n):
            emb_j = active_people[j].get("face_embedding")
            sim = compute_cosine_similarity(emb_i, emb_j)
            if sim >= merge_threshold:
                id_i, id_j = int(active_people[i]["id"]), int(active_people[j]["id"])
                logger.info(
                    "Agglomerative cluster pass: merging Person %d (%s, %d docs) <-> Person %d (%s, %d docs) (sim=%.4f >= %.2f)",
                    id_i, active_people[i].get("name"), int(active_people[i].get("document_count") or 0),
                    id_j, active_people[j].get("name"), int(active_people[j].get("document_count") or 0),
                    sim, merge_threshold,
                )
                _union(id_i, id_j)

    merged_count = 0
    # Group items by root
    groups: Dict[int, List[int]] = {}
    for p in active_people:
        pid = int(p["id"])
        root = _find(pid)
        groups.setdefault(root, []).append(pid)

    for target_id, member_ids in groups.items():
        for source_id in member_ids:
            if source_id != target_id:
                try:
                    success = db_manager.merge_people(source_person_id=source_id, target_person_id=target_id)
                    if success:
                        merged_count += 1
                        logger.info("Successfully merged person cluster %d into %d", source_id, target_id)
                except Exception as merge_err:
                    logger.warning("Error merging person %d into %d: %s", source_id, target_id, merge_err)

    if merged_count > 0:
        logger.info("Agglomerative clustering completed: merged %d duplicate face cluster(s)", merged_count)
    return merged_count
