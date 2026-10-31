"""
DocuVault AI - FastAPI Local Offline Backend Server (server.py)

Manages the non-blocking background document scanning & visual OCR/Face inspection thread,
exposes live scan progress, search/filtering, statistics, face-cluster management,
Windows Explorer integration, and safe local image/thumbnail streaming, and serves `ui/index.html`.
"""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from backend import database, processor, scanner, studio, watcher
except ImportError:
    import database, processor, scanner, studio, watcher
import setup_assets

logger = logging.getLogger("docuvault.server")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "data").resolve()
EXPORTS_DIR = (BASE_DIR / "exports").resolve()
UI_DIR = (BASE_DIR / "ui").resolve()
ASSETS_DIR = (UI_DIR / "assets").resolve()
CSS_DIR = (UI_DIR / "css").resolve()
JS_DIR = (UI_DIR / "js").resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CSS_DIR.mkdir(parents=True, exist_ok=True)
JS_DIR.mkdir(parents=True, exist_ok=True)
UI_INDEX_PATH = UI_DIR / "index.html"

ALLOWED_STREAM_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}


def _resolve_desktop_directory() -> Path:
    """Resolve the user's Desktop directory safely across Windows configurations."""
    desktop = Path.home() / "Desktop"
    if not desktop.is_dir():
        try:
            for d in watcher.get_default_watch_directories():
                if d.name.lower() == "desktop" and d.is_dir():
                    return d
        except Exception:
            pass
    try:
        desktop.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return desktop


# -----------------------------------------------------------------------------
# Pydantic Request Models
# -----------------------------------------------------------------------------


class ScanStartRequest(BaseModel):
    folder_path: Optional[str] = Field(
        default=None,
        description="Optional custom folder path to scan recursively. Empty/null scans full PC.",
    )
    scan_mode: str = Field(
        default="id_vault",
        description="Scanning mode: 'id_vault' (default ID OCR) or 'face_search' (Deep Face Hunter mode).",
    )
    target_category: Optional[str] = Field(
        default=None,
        description="Optional targeted document category (e.g. 'PAN Card', 'Aadhaar Card'). If specified, only documents matching this category are indexed.",
    )


class RenamePersonRequest(BaseModel):
    person_id: int = Field(..., description="Primary key ID of the person in SQLite.")
    name: Optional[str] = Field(default=None, description="New display name for the person.")
    new_name: Optional[str] = Field(default=None, description="Alias for `name`.")

    def resolved_name(self) -> str:
        chosen = self.name if (self.name and self.name.strip()) else self.new_name
        if not chosen or not chosen.strip():
            raise ValueError("Person name cannot be empty.")
        return chosen.strip()


class OpenFolderRequest(BaseModel):
    file_path: str = Field(..., description="Full local filesystem path of the file to highlight.")


class UnlockPdfRequest(BaseModel):
    file_path: str = Field(..., description="Full path to the password-protected PDF.")
    password: str = Field(..., description="User-provided password to decrypt the PDF.")


class MergePeopleRequest(BaseModel):
    source_person_id: int = Field(..., description="ID of person cluster to merge from (will be deleted).")
    target_person_id: int = Field(..., description="ID of primary person cluster to merge into.")


class ExportPersonZipRequest(BaseModel):
    person_id: int = Field(..., description="ID of the person whose documents to export into a ZIP bundle.")


class EnhanceDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Path to the document to clean and perspective-flatten.")


class ResizePortalRequest(BaseModel):
    file_path: str = Field(..., description="Path to the document to resize for portal upload.")
    max_kb: int = Field(default=100, ge=20, le=2048, description="Target maximum size in KB (e.g. 50, 100, 200).")


class GenerateA4PrintRequest(BaseModel):
    file_paths: List[str] = Field(..., min_length=1, max_length=2, description="1 or 2 document paths (e.g. Front and Back).")
    save_to_desktop: bool = Field(default=True, description="Whether to also save the generated PDF directly to the Desktop.")


class VerifyPinRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=4, description="4-digit PIN code.")


class SetPinRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=4, description="New 4-digit PIN code.")
    current_pin: Optional[str] = Field(default=None, description="Existing 4-digit PIN (required if PIN is already enabled).")


class DisablePinRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=4, description="Current 4-digit PIN code.")


class RemoveDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Full path to the document to remove from the app database.")
    add_to_ignored: bool = Field(default=True, description="Whether to add to ignored_files to prevent re-scanning.")


class DeleteDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Full path to the document to physically delete from disk and SQLite.")
    password: Optional[str] = Field(default=None, description="Master password required for physical deletion.")


class VerifyPasswordRequest(BaseModel):
    password: str = Field(..., description="Alphanumeric Master Password or PIN to verify.")


class SetPasswordRequest(BaseModel):
    password: str = Field(..., min_length=4, description="New Alphanumeric Master Password.")
    current_password: Optional[str] = Field(default=None, description="Current Master Password or PIN.")


class DisablePasswordRequest(BaseModel):
    password: str = Field(..., description="Current Master Password or PIN.")


class ImportSessionRequest(BaseModel):
    session_data: List[Dict[str, Any]] = Field(..., description="List of document records from an exported session.")


class BulkActionRequest(BaseModel):
    action: str = Field(..., description="Action to perform: 'delete_pc', 'remove_app', or 'move'.")
    file_paths: List[str] = Field(..., min_length=1, description="List of target file paths.")
    destination_folder: Optional[str] = Field(default=None, description="Destination folder if action is 'move'.")
    password: Optional[str] = Field(default=None, description="Master password if action is 'delete_pc'.")


class ClearCategoryRequest(BaseModel):
    category: str = Field(..., description="Name of the category whose records should be cleared from SQLite.")


class TransferCategoryRequest(BaseModel):
    file_path: str = Field(..., description="Full path of the document to reclassify.")
    new_category: str = Field(..., description="New document category to assign.")


class MoveFileRequest(BaseModel):
    file_path: str = Field(..., description="Current full file path.")
    target_folder: Optional[str] = Field(default=None, description="Destination directory to physically move the file into.")
    destination_folder: Optional[str] = Field(default=None, description="Alias for destination directory.")

    def get_destination_folder(self) -> str:
        dest = self.target_folder or self.destination_folder
        if not dest or not dest.strip():
            raise ValueError("Destination folder is required.")
        return dest.strip()


class ReferenceFaceSearchRequest(BaseModel):
    image_path: Optional[str] = Field(
        default=None,
        description="Local filesystem path to the reference face image (JPG/PNG/WebP).",
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Optional Base64 data URI or encoded bytes of the selected reference image.",
    )
    image_name: Optional[str] = Field(
        default=None,
        description="Display filename of the selected reference image.",
    )
    threshold: float = Field(
        default=0.48,
        ge=0.20,
        le=0.95,
        description="Cosine similarity threshold (~0.45-0.50) for matching face embeddings.",
    )


# -----------------------------------------------------------------------------
# Real-Time WebSocket Progress Broadcaster & Background Scan Manager
# -----------------------------------------------------------------------------


class WebSocketProgressHub:
    """
    Thread-safe FastAPI WebSocket broadcaster streaming real-time progress updates
    from background `ThreadPoolExecutor` workers to connected frontend clients.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.get_running_loop()
            self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)

    async def _broadcast_async(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        stale: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

    def broadcast_from_thread(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            loop = self._loop
            has_clients = bool(self._clients)
        if not has_clients or loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), loop)
        except Exception:
            pass


ws_hub = WebSocketProgressHub()


def _build_scan_ws_payload(status: Dict[str, Any]) -> Dict[str, Any]:
    phase = str(status.get("phase") or "idle")
    proc = int(status.get("processed_count") or 0)
    tot = int(status.get("total_candidates") or 0)
    found = int(status.get("documents_found_count") or 0)
    if phase == "discovering":
        toast_msg = "Discovering candidate files on disk..."
    elif phase == "inspecting":
        toast_msg = f"Scanning {proc}/{tot}..." if tot > 0 else f"Scanning {proc}..."
    elif phase == "stopping":
        toast_msg = f"Stopping scan ({proc}/{tot})..."
    elif phase == "completed":
        toast_msg = f"Scan complete — {found} verified item(s) indexed ({proc}/{tot})"
    elif phase == "stopped":
        toast_msg = f"Scan stopped at {proc}/{tot}"
    elif phase == "error":
        toast_msg = f"Scan error: {status.get('last_error') or 'Unknown error'}"
    else:
        toast_msg = "Scanner idle"
    return {
        "type": "scan_progress",
        "toast_message": toast_msg,
        **status,
    }


class BackgroundScanManager:
    """
    Coordinates non-blocking background candidate discovery (`scanner.py`) and
    visual OCR + Face inspection (`processor.py`) while tracking real-time metrics.
    """

    def __init__(self, db_manager: Optional[database.DatabaseManager] = None) -> None:
        self._db = db_manager or database.db
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._is_scanning: bool = False
        self._phase: str = "idle"
        self._folder_path: Optional[str] = None
        self._total_candidates: int = 0
        self._processed_count: int = 0
        self._documents_found_count: int = 0
        self._current_file_path: str = ""
        self._scan_speed_files_per_sec: float = 0.0
        self._started_monotonic: Optional[float] = None
        self._started_at_iso: Optional[str] = None
        self._finished_elapsed: float = 0.0
        self._last_error: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of live scan progress."""
        with self._lock:
            if self._is_scanning and self._started_monotonic is not None:
                elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            else:
                elapsed = self._finished_elapsed

            if self._total_candidates > 0:
                progress_pct = round(
                    min(100.0, (self._processed_count / float(self._total_candidates)) * 100.0),
                    1,
                )
            elif self._phase == "completed":
                progress_pct = 100.0
            else:
                progress_pct = 0.0

            return {
                "is_scanning": self._is_scanning,
                "phase": self._phase,
                "folder_path": self._folder_path,
                "total_candidates": self._total_candidates,
                "processed_count": self._processed_count,
                "documents_found_count": self._documents_found_count,
                "current_file_path": self._current_file_path,
                "scan_speed_files_per_sec": round(self._scan_speed_files_per_sec, 2),
                "progress_percent": progress_pct,
                "elapsed_seconds": round(elapsed, 2),
                "started_at": self._started_at_iso,
                "last_error": self._last_error,
            }

    def start_scan(
        self,
        folder_path: Optional[str] = None,
        scan_mode: str = "id_vault",
        target_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Launch a background scan thread if none is currently running.
        Validates `folder_path` if provided. Supports `target_category` for targeted scans.
        """
        clean_folder: Optional[str] = None
        if folder_path and folder_path.strip():
            expanded = os.path.abspath(os.path.expanduser(folder_path.strip()))
            if not os.path.exists(expanded):
                raise ValueError(f"Selected folder does not exist: {expanded}")
            clean_folder = database.normalize_file_path(expanded)

        mode = "face_search" if str(scan_mode).strip().lower() == "face_search" else "id_vault"
        clean_target = target_category.strip() if (target_category and target_category.strip()) else None

        with self._lock:
            if self._is_scanning:
                return {
                    "started": False,
                    "message": "A scan is already running.",
                    "status": self._unlocked_status_snapshot(),
                }

            self._stop_event.clear()
            self._is_scanning = True
            self._phase = "discovering"
            self._folder_path = clean_folder
            self._total_candidates = 0
            self._processed_count = 0
            self._documents_found_count = 0
            self._current_file_path = ""
            self._scan_speed_files_per_sec = 0.0
            self._started_monotonic = time.monotonic()
            self._started_at_iso = datetime.now(timezone.utc).isoformat()
            self._finished_elapsed = 0.0
            self._last_error = None

            self._worker_thread = threading.Thread(
                target=self._run_scan_worker,
                args=(clean_folder, mode, clean_target),
                name="DocuVaultScannerWorker",
                daemon=True,
            )
            self._worker_thread.start()

            snap = self._unlocked_status_snapshot()
            ws_hub.broadcast_from_thread(_build_scan_ws_payload(snap))

            msg_suffix = f" (Target: {clean_target})" if clean_target else ""
            return {
                "started": True,
                "message": f"Background scan started in {'Deep Face Search' if mode == 'face_search' else 'ID Vault'} mode.{msg_suffix}",
                "target_category": clean_target,
                "status": snap,
            }

    def stop_scan(self, wait_timeout: float = 0.0) -> Dict[str, Any]:
        """Signal the active background scan thread to stop gracefully."""
        worker: Optional[threading.Thread] = None
        with self._lock:
            if not self._is_scanning:
                return {
                    "stopped": False,
                    "message": "No active scan is currently running.",
                    "status": self._unlocked_status_snapshot(),
                }
            self._stop_event.set()
            self._phase = "stopping"
            worker = self._worker_thread
            ws_hub.broadcast_from_thread(_build_scan_ws_payload(self._unlocked_status_snapshot()))

        if wait_timeout > 0.0 and worker is not None and worker.is_alive():
            worker.join(timeout=wait_timeout)

        return {
            "stopped": True,
            "message": "Stop signal sent to background scanner.",
            "status": self.get_status(),
        }

    def _unlocked_status_snapshot(self) -> Dict[str, Any]:
        elapsed = (
            max(0.0, time.monotonic() - self._started_monotonic)
            if (self._is_scanning and self._started_monotonic is not None)
            else self._finished_elapsed
        )
        progress_pct = (
            round(min(100.0, (self._processed_count / float(self._total_candidates)) * 100.0), 1)
            if self._total_candidates > 0
            else (100.0 if self._phase == "completed" else 0.0)
        )
        return {
            "is_scanning": self._is_scanning,
            "phase": self._phase,
            "folder_path": self._folder_path,
            "total_candidates": self._total_candidates,
            "processed_count": self._processed_count,
            "documents_found_count": self._documents_found_count,
            "current_file_path": self._current_file_path,
            "scan_speed_files_per_sec": round(self._scan_speed_files_per_sec, 2),
            "progress_percent": progress_pct,
            "elapsed_seconds": round(elapsed, 2),
            "started_at": self._started_at_iso,
            "last_error": self._last_error,
        }

    def _run_scan_worker(
        self,
        custom_folder: Optional[str],
        scan_mode: str = "id_vault",
        target_category: Optional[str] = None,
    ) -> None:
        """Background thread executing Everything/os.walk discovery + visual OCR & Face pipeline."""
        try:
            self._db.init_db()
            # Prune records whose underlying files were deleted from disk
            try:
                self._db.prune_missing_files()
            except Exception as prune_exc:
                logger.debug("Prune missing files skipped: %s", prune_exc)

            if self._stop_event.is_set():
                self._finalize_worker(phase="stopped")
                return

            candidates = scanner.find_all_candidate_files(
                custom_folder=custom_folder,
                db_manager=self._db,
            )

            if self._stop_event.is_set():
                self._finalize_worker(phase="stopped")
                return

            total = len(candidates)
            inspect_start = time.monotonic()

            with self._lock:
                self._total_candidates = total
                self._phase = "inspecting" if total > 0 else "completed"
                ws_hub.broadcast_from_thread(_build_scan_ws_payload(self._unlocked_status_snapshot()))

            if total == 0:
                self._finalize_worker(phase="completed")
                return

            hw_profile = scanner.detect_hardware_profile()
            max_workers = hw_profile.get("max_workers", 4)
            logger.info(
                "Hardware Profile: RAM=%.1fGB, CPUs=%d, GPU=%s -> ThreadPoolExecutor(max_workers=%d) Mode=%s Target=%s",
                hw_profile.get("total_ram_gb", 8.0),
                hw_profile.get("cpu_count", 4),
                hw_profile.get("gpu_name", "None"),
                max_workers,
                scan_mode,
                target_category or "All",
            )

            # Producer-Consumer Architecture:
            # - Fast I/O workers load images, check dimensions and OpenCV edge-density heuristics,
            #   recording skipped/unreadable files instantly via AsyncDatabaseWriter.
            # - Exactly ONE dedicated AI worker thread pulls from ai_queue to run RapidOCR & InsightFace,
            #   completely eliminating lock contention and RAM accumulation.
            last_ws_emit = 0.0

            def _record_progress(c_path: str, found_doc: bool, err_msg: Optional[str] = None) -> None:
                nonlocal last_ws_emit
                now_mono = time.monotonic()
                snap_to_emit: Optional[Dict[str, Any]] = None

                with self._lock:
                    self._current_file_path = c_path
                    self._processed_count += 1
                    if found_doc:
                        self._documents_found_count += 1
                    if err_msg:
                        self._last_error = f"{os.path.basename(c_path)}: {err_msg}"
                    elapsed_inspect = max(0.001, now_mono - inspect_start)
                    self._scan_speed_files_per_sec = self._processed_count / elapsed_inspect
                    if (
                        found_doc
                        or self._processed_count == 1
                        or self._processed_count >= total
                        or (now_mono - last_ws_emit) >= 0.08
                    ):
                        last_ws_emit = now_mono
                        snap_to_emit = self._unlocked_status_snapshot()

                if snap_to_emit is not None:
                    snap_payload = _build_scan_ws_payload(snap_to_emit)
                    snap_payload["new_document_found"] = bool(found_doc)
                    ws_hub.broadcast_from_thread(snap_payload)

                if self._processed_count % 128 == 0:
                    import gc
                    gc.collect()

            ai_queue: queue.Queue = queue.Queue(maxsize=16)

            def _ai_consumer_loop() -> None:
                while True:
                    item = ai_queue.get()
                    if item is None:
                        ai_queue.task_done()
                        break
                    c_path, preloaded_img = item
                    if self._stop_event.is_set():
                        if preloaded_img is not None:
                            try:
                                preloaded_img.close()
                            except Exception:
                                pass
                        ai_queue.task_done()
                        continue
                    try:
                        res = processor.process_file(
                            c_path,
                            db_manager=self._db,
                            scan_mode=scan_mode,
                            target_category=target_category,
                            preloaded_image=preloaded_img,
                        )
                        _record_progress(c_path, bool(res.get("is_document", False)), None)
                    except Exception as ex:
                        _record_progress(c_path, False, str(ex))
                    finally:
                        if preloaded_img is not None:
                            try:
                                preloaded_img.close()
                            except Exception:
                                pass
                        ai_queue.task_done()

            ai_thread = threading.Thread(
                target=_ai_consumer_loop,
                name="DocuVaultDedicatedAIWorker",
                daemon=True,
            )
            ai_thread.start()

            def _io_producer_task(path_str: str) -> None:
                if self._stop_event.is_set():
                    return
                norm_p = database.normalize_file_path(path_str)
                try:
                    st = os.stat(norm_p)
                    f_size = int(st.st_size)
                    mtime_f = float(st.st_mtime)
                    mtime_s = database.normalize_hash_or_mtime(mtime_f)
                except OSError as exc:
                    self._db.enqueue_upsert_file(
                        file_path=norm_p,
                        file_hash_or_mtime="0.0",
                        file_size=0,
                        is_document=False,
                        doc_category=database.UNREADABLE_OR_CORRUPT,
                        ocr_text=f"File access error: {exc}",
                    )
                    _record_progress(norm_p, False, str(exc))
                    return

                # Check if already scanned in SQLite
                if self._db.is_file_scanned(norm_p, mtime_s):
                    existing = self._db.get_file_by_path(norm_p)
                    is_doc = bool(existing.get("is_document", False)) if existing else False
                    _record_progress(norm_p, is_doc, None)
                    return

                # Attempt fast visual load in I/O worker
                try:
                    pil_img = processor.load_visual_image(norm_p)
                except processor.PasswordProtectedError as pass_err:
                    self._db.enqueue_upsert_file(
                        file_path=norm_p,
                        file_hash_or_mtime=mtime_s,
                        file_size=f_size,
                        is_document=True,
                        doc_category=database.PASSWORD_PROTECTED,
                        ocr_text="Password protected encrypted PDF. Click 'Unlock PDF' to decrypt.",
                    )
                    _record_progress(norm_p, True, None)
                    return
                except (processor.VisualLoadError, Exception) as load_err:
                    self._db.enqueue_upsert_file(
                        file_path=norm_p,
                        file_hash_or_mtime=mtime_s,
                        file_size=f_size,
                        is_document=False,
                        doc_category=database.UNREADABLE_OR_CORRUPT,
                        ocr_text=f"Load error: {load_err}",
                    )
                    _record_progress(norm_p, False, str(load_err))
                    return

                # Dimension check
                w, h = pil_img.size
                if w < processor.MIN_IMAGE_WIDTH or h < processor.MIN_IMAGE_HEIGHT:
                    self._db.enqueue_upsert_file(
                        file_path=norm_p,
                        file_hash_or_mtime=mtime_s,
                        file_size=f_size,
                        is_document=False,
                        doc_category=database.NOT_A_DOCUMENT,
                    )
                    try:
                        pil_img.close()
                    except Exception:
                        pass
                    _record_progress(norm_p, False, None)
                    return

                # OpenCV Edge-density pre-filter (< 2ms) for document scanning
                if scan_mode != "face_search":
                    has_edges, _, _ = processor.check_text_edge_density(pil_img)
                    if not has_edges:
                        self._db.enqueue_upsert_file(
                            file_path=norm_p,
                            file_hash_or_mtime=mtime_s,
                            file_size=f_size,
                            is_document=False,
                            doc_category=database.NOT_A_DOCUMENT,
                            ocr_text="Image lacks high-frequency edge structure (blank/plain image).",
                        )
                        try:
                            pil_img.close()
                        except Exception:
                            pass
                        _record_progress(norm_p, False, None)
                        return

                # Route to dedicated AI worker thread via bounded queue
                while not self._stop_event.is_set():
                    try:
                        ai_queue.put((norm_p, pil_img), timeout=0.1)
                        break
                    except queue.Full:
                        continue

            io_workers = max(1, min(max_workers, 4))
            with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="DocuVaultIOWorker") as pool:
                list(pool.map(_io_producer_task, candidates))

            # Signal AI worker completion and join
            ai_queue.put(None)
            ai_thread.join(timeout=30.0)

            # Flush all asynchronous database writes to SQLite
            try:
                self._db.flush_writes()
            except Exception as fl_err:
                logger.warning("DB flush writes warning: %s", fl_err)

            # Prune thread-local SQLite connections and trigger WAL checkpoint
            try:
                self._db.prune_dead_connections()
                self._db.checkpoint_wal(mode="PASSIVE")
            except Exception:
                pass

            final_phase = "stopped" if self._stop_event.is_set() else "completed"
            # Run agglomerative cleanup on completed scan to merge any borderline cluster splits
            if final_phase == "completed" and scan_mode in ("id_vault", "face_search"):
                try:
                    import face_engine
                    face_engine.agglomerative_cluster_cleanup(
                        db_manager=self._db,
                        merge_threshold=face_engine.INITIAL_CLUSTERING_THRESHOLD,
                    )
                except Exception as post_err:
                    logger.debug("Post-scan agglomerative clustering notice: %s", post_err)
            self._finalize_worker(phase=final_phase)

        except Exception as exc:
            logger.exception("Fatal error in background scan worker: %s", exc)
            with self._lock:
                self._last_error = str(exc)
            self._finalize_worker(phase="error")

    def _finalize_worker(self, phase: str) -> None:
        with self._lock:
            if self._started_monotonic is not None:
                self._finished_elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            self._is_scanning = False
            self._phase = phase
            if phase in ("completed", "stopped"):
                self._current_file_path = ""
            final_snap = self._unlocked_status_snapshot()
        ws_hub.broadcast_from_thread(_build_scan_ws_payload(final_snap))


scan_manager = BackgroundScanManager()
live_watcher = watcher.LiveFolderWatcher(db_manager=scan_manager._db)


# -----------------------------------------------------------------------------
# URL & Response Enrichment Helpers
# -----------------------------------------------------------------------------


def _build_image_api_url(path_value: Optional[str]) -> Optional[str]:
    """Build a safe `/api/image?path=...` URL for a relative `data/...` or absolute file path."""
    if not path_value or not str(path_value).strip():
        return None
    return f"/api/image?path={quote(str(path_value).strip(), safe='')}"


def _enrich_document_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Attach convenient UI display fields (`file_name`, `thumbnail_url`, `formatted_dob`, `formatted_date_taken`, etc.)."""
    enriched = dict(doc)
    enriched.pop("face_embedding", None)
    file_path = enriched.get("file_path") or ""
    enriched["file_name"] = os.path.basename(file_path)
    enriched["file_ext"] = os.path.splitext(file_path)[1].lower()

    # Smart Cache Engine: Prefer in-memory base64 thumbnail if available for zero-latency UI loading
    b64_thumb = enriched.get("thumbnail_b64")
    if b64_thumb and str(b64_thumb).startswith("data:image"):
        enriched["thumbnail_url"] = b64_thumb
    else:
        thumb_path = enriched.get("thumbnail_path")
        enriched["thumbnail_url"] = (
            _build_image_api_url(thumb_path)
            if thumb_path
            else _build_image_api_url(file_path)
        )
    enriched["full_image_url"] = _build_image_api_url(file_path)
    enriched["person_face_url"] = _build_image_api_url(
        enriched.get("person_face_thumbnail_path")
    )

    # EXIF Date Taken formatting
    raw_dt = enriched.get("date_taken")
    if raw_dt and str(raw_dt).strip():
        clean_dt_str = str(raw_dt).strip()[:19].replace("T", " ")
        try:
            parsed_dt = datetime.strptime(clean_dt_str, "%Y-%m-%d %H:%M:%S")
            enriched["formatted_date_taken"] = parsed_dt.strftime("%d %b %Y")
            enriched["formatted_datetime_taken"] = parsed_dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            enriched["formatted_date_taken"] = clean_dt_str[:10]
            enriched["formatted_datetime_taken"] = clean_dt_str
    else:
        enriched["formatted_date_taken"] = None
        enriched["formatted_datetime_taken"] = None

    # Expiry and DOB formatting
    raw_dob = enriched.get("dob")
    raw_expiry = enriched.get("expiry_date")
    enriched["is_expired"] = False
    enriched["is_expiring_soon"] = False
    enriched["days_until_expiry"] = None

    if raw_dob:
        try:
            dp = str(raw_dob).strip().split("-")
            if len(dp) == 3:
                enriched["formatted_dob"] = f"{dp[2]}/{dp[1]}/{dp[0]}"
            else:
                enriched["formatted_dob"] = str(raw_dob)
        except Exception:
            enriched["formatted_dob"] = str(raw_dob)
    else:
        enriched["formatted_dob"] = None

    if raw_expiry:
        try:
            clean_exp_str = str(raw_expiry).strip()[:10]
            exp_date = datetime.strptime(clean_exp_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            days_diff = (exp_date - today).days
            enriched["days_until_expiry"] = days_diff
            if days_diff < 0:
                enriched["is_expired"] = True
            elif days_diff <= 90:
                enriched["is_expiring_soon"] = True
            enriched["formatted_expiry"] = exp_date.strftime("%d/%m/%Y")
        except Exception:
            enriched["formatted_expiry"] = str(raw_expiry)
    else:
        enriched["formatted_expiry"] = None

    return enriched


def _enrich_person_record(person: Dict[str, Any]) -> Dict[str, Any]:
    """Attach `face_thumbnail_url` for frontend rendering."""
    enriched = dict(person)
    enriched["face_thumbnail_url"] = _build_image_api_url(
        enriched.get("face_thumbnail_path")
    )
    return enriched


def _resolve_and_authorize_image_path(
    raw_path: str,
    active_db: database.DatabaseManager,
) -> Path:
    """
    Resolve `raw_path` and verify it is strictly authorized for streaming:
      1. Files inside DocuVault's own `data/` directory (thumbnails & cropped faces), OR
      2. Original image/PDF files registered in `docuvault.db` with an allowed visual extension.

    Blocks arbitrary local file disclosure (`403 Forbidden`).
    """
    cleaned = raw_path.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Image path cannot be empty.")

    candidate = Path(cleaned)
    data_root = DATA_DIR.resolve()

    if not candidate.is_absolute():
        resolved = (BASE_DIR / candidate).resolve()
        target_path = resolved
        is_internal_data_file = resolved.is_relative_to(data_root)
    else:
        norm_abs = Path(database.normalize_file_path(cleaned))
        resolved = candidate.resolve()
        is_internal_data_file = resolved.is_relative_to(data_root)
        target_path = resolved if is_internal_data_file else norm_abs

    ext = target_path.suffix.lower()
    if ext not in ALLOWED_STREAM_EXTENSIONS:
        raise HTTPException(
            status_code=403,
            detail=f"File extension '{ext}' is not permitted for image streaming.",
        )

    if not is_internal_data_file:
        db_record = active_db.get_file_by_path(str(target_path)) or active_db.get_file_by_path(
            str(resolved)
        )
        if db_record is None:
            raise HTTPException(
                status_code=403,
                detail="Access denied: file path is not indexed in DocuVault AI.",
            )

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Requested image file not found on disk.")

    return target_path


# -----------------------------------------------------------------------------
# FastAPI Lifespan & Application Setup
# -----------------------------------------------------------------------------


def _cleanup_cache_and_temp() -> None:
    """
    On-Exit / Shutdown cleanup routine:
    Strictly deletes any temporary images, cached downloads, or scratch junk files
    created in local temp or cache directories.
    """
    logger.info("Executing DocuVault AI on-exit cleanup routine...")
    dirs_to_clean = [
        BASE_DIR / ".cache",
        BASE_DIR / "temp",
        DATA_DIR / "temp",
        DATA_DIR / "studio" / "temp",
    ]
    for d in dirs_to_clean:
        if d.exists() and d.is_dir():
            try:
                for item in d.iterdir():
                    try:
                        if item.is_file() or item.is_symlink():
                            item.unlink(missing_ok=True)
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception as clean_err:
                        logger.debug("Could not delete temp item %s: %s", item, clean_err)
            except Exception as d_err:
                logger.debug("Could not clean dir %s: %s", d, d_err)


import atexit
atexit.register(_cleanup_cache_and_temp)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ws_hub.bind_loop(asyncio.get_running_loop())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "thumbnails").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "faces").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "temp").mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(setup_assets.ensure_local_assets, force_download=False)
    except Exception as asset_err:
        logger.warning("Could not verify local assets during startup: %s", asset_err)
    scan_manager._db.init_db()
    # Run agglomerative cluster pass on vault load to merge any split clusters
    try:
        import face_engine
        merged = await asyncio.to_thread(
            face_engine.agglomerative_cluster_cleanup,
            db_manager=scan_manager._db,
            merge_threshold=face_engine.INITIAL_CLUSTERING_THRESHOLD,
        )
        if merged > 0:
            logger.info("Startup agglomerative clustering merged %d split cluster(s)", merged)
    except Exception as cl_err:
        logger.warning("Startup agglomerative clustering notice: %s", cl_err)
    try:
        live_watcher.start()
    except Exception as exc:
        logger.warning("Could not start live folder watcher: %s", exc)
    yield
    live_watcher.stop(timeout=2.0)
    scan_manager.stop_scan(wait_timeout=2.0)
    _cleanup_cache_and_temp()
    scan_manager._db.close()


# Ensure `ui/assets/` exists prior to mounting StaticFiles
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DocuVault AI Local Backend",
    description="100% Offline Visual OCR & Face-Clustered Document Vault API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "null",
    ],
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount `ui/assets`, `/css`, and `/js` directly via FastAPI StaticFiles
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/ui/assets", StaticFiles(directory=str(ASSETS_DIR)), name="ui_assets")
app.mount("/css", StaticFiles(directory=str(CSS_DIR)), name="css")
app.mount("/js", StaticFiles(directory=str(JS_DIR)), name="js")


# -----------------------------------------------------------------------------
# Frontend UI Routes
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_root_ui():
    """Serve `ui/index.html` at the root `/` route."""
    if UI_INDEX_PATH.is_file():
        return FileResponse(
            path=str(UI_INDEX_PATH),
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )
    return HTMLResponse(
        content=(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>DocuVault AI</title></head><body style='font-family:sans-serif;"
            "background:#0f172a;color:#e2e8f0;padding:2rem;'>"
            "<h2>DocuVault AI Backend Running</h2>"
            "<p><code>ui/index.html</code> has not been created yet.</p>"
            "</body></html>"
        ),
        status_code=200,
    )


@app.get("/ui/{asset_path:path}", include_in_schema=False)
def serve_ui_static_asset(asset_path: str):
    """Serve any static CSS/JS/icon assets placed inside `ui/`."""
    resolved = (UI_DIR / asset_path).resolve()
    if not resolved.is_relative_to(UI_DIR) or not resolved.is_file():
        raise HTTPException(status_code=404, detail="UI asset not found.")
    media_type, _ = mimetypes.guess_type(str(resolved))
    return FileResponse(path=str(resolved), media_type=media_type or "application/octet-stream")


# -----------------------------------------------------------------------------
# Scan Control & Live Progress Endpoints
# -----------------------------------------------------------------------------


@app.post("/api/scan/start")
def api_scan_start(payload: Optional[ScanStartRequest] = Body(default=None)) -> Dict[str, Any]:
    """
    Start a background visual document scan.
    Supports 'id_vault' and 'face_search' modes.
    If `folder_path` is omitted or empty, scans the entire PC via `Everything64.dll` (with `os.walk` fallback).
    """
    folder_path = payload.folder_path if payload is not None else None
    scan_mode = payload.scan_mode if payload is not None else "id_vault"
    target_category = payload.target_category if payload is not None else None
    try:
        return scan_manager.start_scan(
            folder_path=folder_path,
            scan_mode=scan_mode,
            target_category=target_category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scan/stop")
def api_scan_stop() -> Dict[str, Any]:
    """Gracefully stop the currently running background scan."""
    return scan_manager.stop_scan()


@app.get("/api/scan/status")
def api_scan_status() -> Dict[str, Any]:
    """Return live scan progress metrics."""
    return scan_manager.get_status()


@app.websocket("/ws/scan")
async def websocket_scan_progress(websocket: WebSocket) -> None:
    """
    Real-time FastAPI WebSocket streaming live background scan progress
    (e.g., 'Scanning 405/20000...') directly to frontend toast notifications & banner.
    """
    await ws_hub.connect(websocket)
    try:
        await websocket.send_json(_build_scan_ws_payload(scan_manager.get_status()))
        while True:
            msg = await websocket.receive_text()
            clean = (msg or "").strip().lower()
            if clean == "ping":
                await websocket.send_json({"type": "pong"})
            elif clean == "status":
                await websocket.send_json(_build_scan_ws_payload(scan_manager.get_status()))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_hub.disconnect(websocket)


# -----------------------------------------------------------------------------
# Documents, Faces, Stats & People Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/documents")
def api_get_documents(
    search: Optional[str] = Query(
        default=None,
        description="Search query matching OCR text, extracted ID number, person name, or filename.",
    ),
    category: Optional[str] = Query(
        default=None,
        description="Filter by exact document category (e.g. 'Aadhaar Card', 'PAN Card', 'Password Protected', 'Unidentified / Unreadable').",
    ),
    person_id: Optional[int] = Query(
        default=None,
        description="Filter by recognized person cluster ID.",
    ),
    expiring_only: bool = Query(
        default=False,
        description="If True, only return documents that are expired or expiring within 90 days.",
    ),
    scan_mode: Optional[str] = Query(
        default=None,
        description="Active scan mode: 'id_vault' or 'face_search'. In 'id_vault' mode, Face Gallery Photos and Unidentified files are hidden by default.",
    ),
    sort_by: str = Query(
        default="scanned_desc",
        description="Sort order: 'scanned_desc', 'date_taken_desc', 'date_taken_asc', 'name_asc', or 'size_desc'.",
    ),
    limit: int = Query(default=50, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """
    Query verified documents filtered by keyword search, category, expiring status, person_id,
    pagination (`limit` & `offset`), and EXIF chronological `sort_by`.
    """
    # Defensive unwrapping in case endpoint is invoked directly as a Python function
    search_str = search if isinstance(search, str) else None
    cat_str = category if isinstance(category, str) else None
    pid_int = int(person_id) if isinstance(person_id, (int, str)) and str(person_id).isdigit() else None
    is_expiring = bool(expiring_only) if isinstance(expiring_only, bool) else False
    mode_str = scan_mode if isinstance(scan_mode, str) else None
    sort_str = sort_by if isinstance(sort_by, str) else "scanned_desc"
    safe_limit = int(limit) if isinstance(limit, (int, str)) else 50
    safe_offset = int(offset) if isinstance(offset, (int, str)) else 0

    if cat_str and cat_str.strip() == database.NOT_A_DOCUMENT:
        return {
            "documents": [],
            "count": 0,
            "total": 0,
            "limit": safe_limit,
            "offset": safe_offset,
            "has_more": False,
            "sort_by": sort_str,
        }

    is_unreadable_cat = bool(cat_str and cat_str.strip() == database.UNREADABLE_OR_CORRUPT)
    total = scan_manager._db.count_documents(
        keyword=search_str,
        category=cat_str,
        person_id=pid_int,
        expiring_only=is_expiring,
        include_non_documents=is_unreadable_cat,
        scan_mode=mode_str,
    )
    rows = scan_manager._db.search_documents(
        keyword=search_str,
        category=cat_str,
        person_id=pid_int,
        expiring_only=is_expiring,
        include_non_documents=is_unreadable_cat,
        scan_mode=mode_str,
        limit=safe_limit,
        offset=safe_offset,
        sort_by=sort_str,
    )
    documents = [
        _enrich_document_record(row)
        for row in rows
        if row.get("is_document") or is_unreadable_cat
    ]
    return {
        "documents": documents,
        "count": len(documents),
        "total": total,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": (safe_offset + len(rows)) < total,
        "sort_by": sort_str,
    }


@app.post("/api/pdf/unlock")
def api_unlock_pdf(req: UnlockPdfRequest) -> Dict[str, Any]:
    """
    Attempt to decrypt and process a password-protected PDF file with user-provided password.
    If valid, runs visual OCR + Face Detection and moves the file to its verified category.
    """
    clean_path = database.normalize_file_path(req.file_path)
    if not os.path.isfile(clean_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    try:
        res = processor.process_file(
            clean_path,
            db_manager=scan_manager._db,
            pdf_password=req.password,
        )
        if res.get("doc_category") == database.PASSWORD_PROTECTED:
            raise HTTPException(
                status_code=400,
                detail="Incorrect password. Please verify the password and try again.",
            )
        return {
            "success": True,
            "file_path": clean_path,
            "doc_category": res.get("doc_category"),
            "extracted_id_number": res.get("extracted_id_number"),
            "is_document": res.get("is_document"),
            "status": res.get("status"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to unlock PDF '%s': %s", clean_path, exc)
        raise HTTPException(status_code=400, detail=f"Failed to decrypt PDF: {exc}")


@app.get("/api/hardware")
def api_get_hardware() -> Dict[str, Any]:
    """Return hardware profiling telemetry (RAM GB, CPU cores, GPU status, max workers)."""
    return scanner.detect_hardware_profile()


@app.get("/api/stats")
def api_get_stats() -> Dict[str, Any]:
    """Return document counts per category, total documents, duplicate groups, expiring docs, and total people detected."""
    summary = scan_manager._db.get_summary_stats()
    dup_count = scan_manager._db.get_duplicate_groups_count()
    watcher_state = live_watcher.get_status()
    hw_profile = scanner.detect_hardware_profile()
    return {
        "total_documents": summary["total_documents"],
        "total_people": summary["total_people"],
        "total_scanned": summary["total_scanned"],
        "total_non_documents": summary["total_non_documents"],
        "expiring_documents_count": summary.get("expiring_documents_count", 0),
        "duplicate_groups_count": dup_count,
        "watcher_active": watcher_state["is_running"],
        "watcher_auto_documents": watcher_state["auto_documents_found"],
        "categories": summary["categories"],
        "category_counts": summary["categories"],
        "hardware": hw_profile,
    }


# -----------------------------------------------------------------------------
# Security & Master Password Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/password/status")
def api_password_status() -> Dict[str, Any]:
    """Return whether Master Password or PIN security protection is enabled."""
    is_master_enabled = scan_manager._db.is_master_password_enabled()
    is_pin_on = scan_manager._db.is_pin_enabled()
    return {
        "master_password_enabled": bool(is_master_enabled),
        "pin_enabled": bool(is_pin_on),
        "has_protection": bool(is_master_enabled or is_pin_on),
    }


@app.post("/api/password/verify")
def api_password_verify(req: VerifyPasswordRequest) -> Dict[str, Any]:
    """Verify Master Password or PIN."""
    valid = scan_manager._db.verify_master_password(req.password)
    return {
        "valid": bool(valid),
        "message": "Password verified successfully." if valid else "Incorrect password."
    }


@app.post("/api/password/set")
def api_password_set(req: SetPasswordRequest) -> Dict[str, Any]:
    """Configure or update the Alphanumeric Master Password."""
    if scan_manager._db.is_master_password_enabled():
        if not req.current_password or not scan_manager._db.verify_master_password(req.current_password):
            raise HTTPException(status_code=403, detail="Current Master Password incorrect.")
    try:
        scan_manager._db.set_master_password(req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "success": True,
        "message": "Master Password configured successfully."
    }


@app.post("/api/password/disable")
def api_password_disable(req: DisablePasswordRequest) -> Dict[str, Any]:
    """Disable Master Password protection."""
    if not scan_manager._db.verify_master_password(req.password):
        raise HTTPException(status_code=403, detail="Incorrect password.")
    scan_manager._db.disable_master_password(req.password)
    return {
        "success": True,
        "message": "Master Password protection disabled."
    }


# Backward-compatible PIN routes
@app.get("/api/pin/status")
def api_pin_status() -> Dict[str, Any]:
    """Return whether 4-digit PIN lock security is currently enabled."""
    return {
        "pin_enabled": scan_manager._db.is_pin_enabled() or scan_manager._db.is_master_password_enabled(),
        "is_master": scan_manager._db.is_master_password_enabled(),
    }


@app.post("/api/pin/verify")
def api_pin_verify(req: VerifyPinRequest) -> Dict[str, Any]:
    """Verify the entered 4-digit PIN or master password."""
    is_valid = scan_manager._db.verify_master_password(req.pin)
    return {
        "valid": bool(is_valid)
    }


@app.post("/api/pin/set")
def api_pin_set(req: SetPinRequest) -> Dict[str, Any]:
    """Configure or change the 4-digit security PIN."""
    # If PIN or Master Password is already enabled, user must confirm their existing PIN first
    if scan_manager._db.is_pin_enabled() or scan_manager._db.is_master_password_enabled():
        if not req.current_pin:
            raise HTTPException(status_code=400, detail="Current PIN is required to change PIN.")
        if not scan_manager._db.verify_master_password(req.current_pin):
            raise HTTPException(status_code=403, detail="Current PIN is incorrect.")
    try:
        scan_manager._db.set_pin(req.pin)
        return {
            "success": True,
            "message": "PIN configured and enabled successfully."
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/pin/disable")
def api_pin_disable(req: DisablePinRequest) -> Dict[str, Any]:
    """Disable 4-digit PIN security lock after verifying current PIN."""
    if not scan_manager._db.is_pin_enabled() and not scan_manager._db.is_master_password_enabled():
        return {"success": True, "message": "PIN security was already disabled."}
    if not scan_manager._db.verify_master_password(req.pin):
        raise HTTPException(status_code=403, detail="Incorrect PIN.")
    success = scan_manager._db.disable_pin(req.pin)
    return {
        "success": bool(success),
        "message": "PIN security disabled successfully."
    }


# -----------------------------------------------------------------------------
# Document Management Endpoints (Remove, Delete, Transfer, Move, Clear Category)
# -----------------------------------------------------------------------------


@app.post("/api/document/remove")
def api_remove_document(payload: RemoveDocumentRequest) -> Dict[str, Any]:
    """
    Remove a document record from SQLite (keeps the physical file untouched on disk).
    Optionally records in `ignored_files` so future scans will not re-index it.
    """
    try:
        norm_path = database.normalize_file_path(payload.file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ok = scan_manager._db.remove_file_record(norm_path, add_to_ignored=payload.add_to_ignored)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found in database.")

    return {
        "ok": True,
        "message": f"Removed from DocuVault. Physical file remains safe on disk.",
        "file_path": norm_path,
        "ignored": payload.add_to_ignored,
    }


@app.post("/api/document/delete")
def api_delete_document(payload: DeleteDocumentRequest) -> Dict[str, Any]:
    """
    Delete a document record from SQLite AND permanently delete the physical file from disk.
    Requires Master Password verification if Master Password or PIN protection is enabled.
    """
    # Enforce Master Password authorization for physical disk deletion
    if scan_manager._db.is_master_password_enabled() or scan_manager._db.is_pin_enabled():
        if not payload.password:
            raise HTTPException(
                status_code=401,
                detail="Master Password authorization required to permanently delete files from your PC.",
            )
        if not scan_manager._db.verify_master_password(payload.password):
            raise HTTPException(status_code=403, detail="Incorrect Master Password.")

    try:
        norm_path = database.normalize_file_path(payload.file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Remove from DB first
    scan_manager._db.delete_file_by_path(norm_path)

    # Physically delete file from disk if it exists
    file_deleted = False
    if os.path.exists(norm_path):
        try:
            os.remove(norm_path)
            file_deleted = True
        except Exception as exc:
            logger.error("Failed to delete physical file '%s': %s", norm_path, exc)
            raise HTTPException(status_code=500, detail=f"Failed to delete file from PC: {exc}")

    return {
        "ok": True,
        "message": "Document permanently deleted from database and PC disk.",
        "file_path": norm_path,
        "file_deleted": file_deleted,
    }


@app.post("/api/bulk-action")
def api_bulk_action(payload: BulkActionRequest) -> Dict[str, Any]:
    """
    Execute bulk actions across multiple selected files or an entire face cluster:
    - 'delete_pc': Permanently delete from disk (with master password check).
    - 'remove_app': Hide from database and mark as ignored.
    - 'move': Move files to destination folder.
    """
    action = payload.action.strip().lower()
    file_paths = [database.normalize_file_path(p) for p in payload.file_paths if p and p.strip()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided.")

    if action == "delete_pc":
        if scan_manager._db.is_master_password_enabled() or scan_manager._db.is_pin_enabled():
            if not payload.password:
                raise HTTPException(status_code=401, detail="Master Password required for bulk physical deletion.")
            if not scan_manager._db.verify_master_password(payload.password):
                raise HTTPException(status_code=403, detail="Incorrect Master Password.")

        deleted_count = 0
        errors = []
        for p in file_paths:
            scan_manager._db.delete_file_by_path(p)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted_count += 1
                except Exception as ex:
                    errors.append(f"{os.path.basename(p)}: {ex}")
        return {
            "ok": True,
            "action": "delete_pc",
            "count": deleted_count,
            "errors": errors,
            "message": f"Successfully deleted {deleted_count} file(s) permanently from disk.",
        }

    elif action == "remove_app":
        removed_count = 0
        for p in file_paths:
            if scan_manager._db.remove_file_record(p, add_to_ignored=True):
                removed_count += 1
        return {
            "ok": True,
            "action": "remove_app",
            "count": removed_count,
            "message": f"Removed {removed_count} file(s) from vault (safe on PC).",
        }

    elif action == "move":
        if not payload.destination_folder or not os.path.isdir(payload.destination_folder):
            raise HTTPException(status_code=400, detail="Valid destination folder path is required.")
        dest_dir = database.normalize_file_path(payload.destination_folder)
        moved_count = 0
        errors = []
        for p in file_paths:
            if not os.path.isfile(p):
                continue
            filename = os.path.basename(p)
            new_p = os.path.join(dest_dir, filename)
            counter = 1
            stem, ext = os.path.splitext(filename)
            while os.path.exists(new_p):
                new_p = os.path.join(dest_dir, f"{stem}_{counter}{ext}")
                counter += 1
            try:
                shutil.move(p, new_p)
                norm_new = database.normalize_file_path(new_p)
                scan_manager._db.update_file_path(p, norm_new)
                moved_count += 1
            except Exception as ex:
                errors.append(f"{filename}: {ex}")
        return {
            "ok": True,
            "action": "move",
            "count": moved_count,
            "errors": errors,
            "message": f"Moved {moved_count} file(s) to '{dest_dir}'.",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported bulk action '{action}'.")


@app.post("/api/session/import")
def api_import_session(payload: ImportSessionRequest) -> Dict[str, Any]:
    """
    Import and restore an exported search session (.json) into the vault database instantly
    without re-running OCR or re-scanning the disk.
    """
    imported_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for doc in payload.session_data:
        file_path = doc.get("file_path")
        if not file_path:
            continue
        try:
            norm_path = database.normalize_file_path(file_path)
            scan_manager._db.upsert_file(
                file_path=norm_path,
                file_hash_or_mtime=doc.get("file_hash_or_mtime") or "imported_session",
                file_size=int(doc.get("file_size") or 0),
                is_document=bool(doc.get("is_document", True)),
                doc_category=doc.get("doc_category") or "Bill/Receipt",
                extracted_id_number=doc.get("extracted_id_number"),
                dob=doc.get("dob"),
                expiry_date=doc.get("expiry_date"),
                ocr_text=doc.get("ocr_text"),
                person_id=doc.get("person_id"),
                thumbnail_path=doc.get("thumbnail_path"),
                dhash=doc.get("dhash"),
                thumbnail_b64=doc.get("thumbnail_b64"),
                scanned_at=now_iso,
            )
            imported_count += 1
        except Exception as ex:
            logger.debug("Import session item notice: %s", ex)

    return {
        "ok": True,
        "imported_count": imported_count,
        "message": f"Imported {imported_count} document(s) from session.",
    }


@app.post("/api/category/clear")
def api_clear_category(payload: ClearCategoryRequest) -> Dict[str, Any]:
    """
    Drop all database records belonging to a category without touching physical files.
    """
    cat = payload.category.strip()
    if cat not in database.ALL_DOC_CATEGORIES and cat != "All":
        raise HTTPException(status_code=400, detail=f"Invalid category '{cat}'.")

    cleared = scan_manager._db.clear_category_documents(cat)
    return {
        "ok": True,
        "category": cat,
        "cleared_count": cleared,
        "message": f"Cleared {cleared} document(s) from category '{cat}'. Physical files are untouched.",
    }


@app.post("/api/document/transfer")
def api_transfer_category(payload: TransferCategoryRequest) -> Dict[str, Any]:
    """
    Manually override/transfer a document's category (e.g. from 'Aadhaar Card' to 'PAN Card' or 'Not a Document').
    """
    try:
        norm_path = database.normalize_file_path(payload.file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    new_cat = payload.new_category.strip()
    try:
        ok = scan_manager._db.update_file_category(norm_path, new_cat)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not ok:
        raise HTTPException(status_code=404, detail="Document not found in database.")

    return {
        "ok": True,
        "file_path": norm_path,
        "new_category": new_cat,
        "message": f"Category updated to '{new_cat}'.",
    }


@app.post("/api/document/move")
def api_move_document(payload: MoveFileRequest) -> Dict[str, Any]:
    """
    Physically move a document file to a chosen directory using shutil.move and update its path in SQLite.
    """
    try:
        old_path = database.normalize_file_path(payload.file_path)
        dest_folder = database.normalize_file_path(payload.get_destination_folder())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not os.path.isfile(old_path):
        raise HTTPException(status_code=404, detail="Source file does not exist on disk.")

    if not os.path.isdir(dest_folder):
        raise HTTPException(status_code=400, detail="Target destination folder does not exist.")

    filename = os.path.basename(old_path)
    new_path = os.path.join(dest_folder, filename)

    # Avoid overwriting existing file with different content
    if os.path.exists(new_path) and os.path.normcase(old_path) != os.path.normcase(new_path):
        stem, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(new_path):
            new_path = os.path.join(dest_folder, f"{stem}_{counter}{ext}")
            counter += 1

    try:
        shutil.move(old_path, new_path)
    except Exception as exc:
        logger.error("Failed to move file '%s' to '%s': %s", old_path, new_path, exc)
        raise HTTPException(status_code=500, detail=f"Could not move file on disk: {exc}")

    norm_new_path = database.normalize_file_path(new_path)
    scan_manager._db.update_file_path(old_path, norm_new_path)

    return {
        "ok": True,
        "old_path": old_path,
        "new_path": norm_new_path,
        "new_file_name": os.path.basename(norm_new_path),
        "message": f"File successfully moved to '{dest_folder}'.",
    }


@app.get("/api/duplicates")
def api_get_duplicates(
    max_distance: int = Query(
        default=6,
        ge=0,
        le=16,
        description="Maximum 64-bit dHash Hamming distance for grouping near-identical document copies.",
    ),
) -> Dict[str, Any]:
    """
    Group identical or near-identical verified document copies scattered across different folders
    using 64-bit perceptual difference hash (dHash) comparison.
    """
    raw_groups = scan_manager._db.find_duplicate_groups(max_hamming_distance=max_distance)
    enriched_groups: List[Dict[str, Any]] = []
    for grp in raw_groups:
        grp_copy = dict(grp)
        grp_copy["documents"] = [_enrich_document_record(d) for d in grp.get("documents", [])]
        enriched_groups.append(grp_copy)

    return {
        "duplicate_groups": enriched_groups,
        "total_groups": len(enriched_groups),
        "total_duplicate_files": sum(int(g["copy_count"]) for g in enriched_groups),
        "total_wasted_bytes": sum(int(g["wasted_bytes"]) for g in enriched_groups),
    }


@app.get("/api/watcher/status")
def api_watcher_status() -> Dict[str, Any]:
    """Return live status of the real-time `Downloads`, `Desktop`, and `Documents` watcher."""
    return live_watcher.get_status()


@app.get("/api/people")
def api_get_people(
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Return active detected face clusters with face thumbnail URL, name, document count, and pagination."""
    safe_limit = int(limit) if isinstance(limit, (int, str)) else 200
    safe_offset = int(offset) if isinstance(offset, (int, str)) else 0
    total = scan_manager._db.count_people(active_only=True)
    people_rows = scan_manager._db.get_all_people(
        include_embeddings=False,
        limit=safe_limit,
        offset=safe_offset,
        active_only=True,
    )
    people = [_enrich_person_record(p) for p in people_rows]
    return {
        "people": people,
        "count": len(people),
        "total": total,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": (safe_offset + len(people)) < total,
    }


@app.get("/api/faces")
def api_get_faces(
    person_id: Optional[int] = Query(default=None, description="Optional person cluster ID filter."),
    search: Optional[str] = Query(default=None, description="Optional search keyword."),
    category: Optional[str] = Query(
        default=None,
        description="Optional category filter (defaults to all face-linked & gallery photos in face_search mode).",
    ),
    sort_by: str = Query(
        default="date_taken_desc",
        description="Sort order: 'date_taken_desc', 'date_taken_asc', 'scanned_desc', 'name_asc', or 'size_desc'.",
    ),
    limit: int = Query(default=50, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """
    Paginated Face Gallery & Identity Clusters endpoint (`limit` & `offset`),
    preventing DOM overload when browsing thousands of face photos and clusters.
    """
    pid_int = int(person_id) if isinstance(person_id, (int, str)) and str(person_id).isdigit() else None
    search_str = search if isinstance(search, str) else None
    cat_str = category if isinstance(category, str) else ("All" if pid_int is not None else "Face Gallery Photo")
    sort_str = sort_by if isinstance(sort_by, str) else "date_taken_desc"
    safe_limit = int(limit) if isinstance(limit, (int, str)) else 50
    safe_offset = int(offset) if isinstance(offset, (int, str)) else 0

    total_faces = scan_manager._db.count_documents(
        keyword=search_str,
        category=cat_str,
        person_id=pid_int,
        include_non_documents=False,
        scan_mode="face_search",
    )
    rows = scan_manager._db.search_documents(
        keyword=search_str,
        category=cat_str,
        person_id=pid_int,
        include_non_documents=False,
        scan_mode="face_search",
        limit=safe_limit,
        offset=safe_offset,
        sort_by=sort_str,
    )
    faces = [_enrich_document_record(row) for row in rows if row.get("is_document")]

    total_people = scan_manager._db.count_people(active_only=True)
    people_rows = scan_manager._db.get_all_people(
        include_embeddings=False,
        limit=safe_limit,
        offset=safe_offset,
        active_only=True,
    )
    people = [_enrich_person_record(p) for p in people_rows]

    return {
        "faces": faces,
        "documents": faces,
        "people": people,
        "count": len(faces),
        "total": total_faces,
        "total_people": total_people,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": (safe_offset + len(rows)) < total_faces,
        "sort_by": sort_str,
    }


@app.post("/api/people/rename")
def api_rename_person(payload: RenamePersonRequest) -> Dict[str, Any]:
    """Update a detected person's display name in SQLite."""
    try:
        clean_name = payload.resolved_name()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    updated = scan_manager._db.rename_person(person_id=payload.person_id, new_name=clean_name)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Person with id={payload.person_id} was not found.",
        )

    person = scan_manager._db.get_person(payload.person_id, include_embedding=False)
    return {
        "ok": True,
        "person": _enrich_person_record(person) if person else None,
    }


@app.post("/api/people/merge")
def api_merge_people(payload: MergePeopleRequest) -> Dict[str, Any]:
    """
    Merge two separate face clusters for the same person (e.g. young photo vs old photo)
    into a single primary person_id.
    """
    if payload.source_person_id == payload.target_person_id:
        raise HTTPException(status_code=400, detail="Cannot merge a person cluster into itself.")

    src = scan_manager._db.get_person(payload.source_person_id, include_embedding=False)
    dst = scan_manager._db.get_person(payload.target_person_id, include_embedding=False)
    if not src or not dst:
        raise HTTPException(status_code=404, detail="One or both person clusters were not found.")

    merged = scan_manager._db.merge_people(
        source_person_id=payload.source_person_id,
        target_person_id=payload.target_person_id,
    )
    if not merged:
        raise HTTPException(status_code=500, detail="Failed to merge person clusters in SQLite.")

    updated_target = scan_manager._db.get_person(payload.target_person_id, include_embedding=False)
    return {
        "ok": True,
        "message": f"Successfully merged '{src['name']}' into '{dst['name']}'.",
        "person": _enrich_person_record(updated_target) if updated_target else None,
    }


@app.post("/api/people/export-zip")
def api_export_person_zip(payload: ExportPersonZipRequest) -> Dict[str, Any]:
    """
    Bundle all detected documents (PAN, Aadhaar, Driving License, etc.) belonging to a person
    into a clean .zip file saved directly to the user's Desktop.
    """
    import zipfile

    person = scan_manager._db.get_person(payload.person_id, include_embedding=False)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person with id={payload.person_id} not found.")

    # Query all verified documents for this person
    docs = scan_manager._db.search_documents(
        person_id=payload.person_id,
        include_non_documents=False,
        limit=2000,
    )
    if not docs:
        raise HTTPException(status_code=400, detail=f"No documents found for '{person['name']}'.")

    # Resolve Desktop path
    desktop_dir = _resolve_desktop_directory()

    # Sanitize person name for zip filename
    safe_name = "".join(c for c in person["name"] if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_name = safe_name.replace(" ", "_") or f"Person_{payload.person_id}"
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"DocuVault_{safe_name}_{date_str}.zip"
    zip_path = desktop_dir / zip_filename

    added_count = 0
    used_names: set[str] = set()

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for doc in docs:
            real_file = Path(doc["file_path"])
            if not real_file.is_file():
                continue

            category_prefix = "".join(c for c in doc.get("doc_category", "Doc") if c.isalnum() or c in (" ", "_")).replace(" ", "_")
            doc_id_part = f"_{doc['extracted_id_number']}" if doc.get("extracted_id_number") else ""
            clean_id_part = "".join(c for c in doc_id_part if c.isalnum() or c == "_")

            dest_name = f"{category_prefix}{clean_id_part}_{real_file.name}"
            # Deduplicate inside zip if multiple files have the same name
            unique_name = dest_name
            counter = 1
            while unique_name in used_names:
                unique_name = f"{category_prefix}{clean_id_part}_{counter}_{real_file.name}"
                counter += 1
            used_names.add(unique_name)

            try:
                zf.write(str(real_file), arcname=unique_name)
                added_count += 1
            except Exception as write_err:
                logger.warning("Could not add '%s' to ZIP: %s", real_file, write_err)

    if added_count == 0:
        if zip_path.exists():
            zip_path.unlink()
        raise HTTPException(status_code=400, detail="None of the indexed documents could be read from disk.")

    return {
        "ok": True,
        "zip_path": str(zip_path),
        "zip_filename": zip_filename,
        "person_name": person["name"],
        "files_bundled": added_count,
    }


# -----------------------------------------------------------------------------
# Document Studio Endpoints (Auto-Crop, Portal Resizer, A4 Print)
# -----------------------------------------------------------------------------


@app.post("/api/studio/enhance")
def api_studio_enhance(payload: EnhanceDocumentRequest) -> Dict[str, Any]:
    """
    CamScanner Mode: Detect 4 corners, flatten perspective, and apply CLAHE contrast enhancement.
    Saves an enhanced preview copy in data/studio/ and returns its access URL.
    """
    norm_path = database.normalize_file_path(payload.file_path)
    if not os.path.isfile(norm_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk.")

    pil_img = processor.load_visual_image(norm_path)
    if pil_img is None:
        raise HTTPException(status_code=400, detail="Could not visually read document.")

    try:
        enhanced_pil, quad_found = studio.auto_perspective_crop(pil_img, enhance=True)
        filename = f"enhanced_{int(time.time())}_{Path(norm_path).stem}.jpg"
        save_path = studio.STUDIO_DIR / filename
        enhanced_pil.save(save_path, format="JPEG", quality=92, optimize=True)

        return {
            "ok": True,
            "quad_detected": quad_found,
            "enhanced_image_url": _build_image_api_url(f"data/studio/{filename}"),
            "width": enhanced_pil.width,
            "height": enhanced_pil.height,
        }
    except Exception as exc:
        logger.error("Studio enhance failed on '%s': %s", norm_path, exc)
        raise HTTPException(status_code=500, detail=f"Enhance failed: {exc}")


@app.post("/api/studio/resize-portal")
def api_studio_resize_portal(payload: ResizePortalRequest) -> Response:
    """
    Resize and compress image using binary search to ensure output is strictly <= target KB limit.
    Directly streams down the compliant JPEG file.
    """
    norm_path = database.normalize_file_path(payload.file_path)
    if not os.path.isfile(norm_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk.")

    pil_img = processor.load_visual_image(norm_path)
    if pil_img is None:
        raise HTTPException(status_code=400, detail="Could not visually read document.")

    try:
        jpeg_bytes, size_bytes, dims = studio.resize_for_portal(pil_img, max_kb=payload.max_kb)
        filename = f"{Path(norm_path).stem}_{payload.max_kb}KB.jpg"
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Target-KB": str(payload.max_kb),
                "X-Actual-Bytes": str(size_bytes),
                "X-Dimensions": f"{dims[0]}x{dims[1]}",
            },
        )
    except Exception as exc:
        logger.error("Portal resize failed on '%s': %s", norm_path, exc)
        raise HTTPException(status_code=500, detail=f"Resize failed: {exc}")


@app.post("/api/studio/generate-a4")
def api_studio_generate_a4(payload: GenerateA4PrintRequest) -> Dict[str, Any]:
    """
    Arrange 1 or 2 document cards on an A4 300 DPI canvas, save a PDF to Desktop,
    and return the generated file path.
    """
    valid_paths: List[str] = []
    for raw_p in payload.file_paths:
        norm = database.normalize_file_path(raw_p)
        if os.path.isfile(norm):
            valid_paths.append(norm)

    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid document files found to print.")

    desktop_dir = _resolve_desktop_directory()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"A4_Print_{Path(valid_paths[0]).stem}_{timestamp}.pdf"
    desktop_pdf_path = desktop_dir / pdf_filename

    try:
        pdf_bytes, saved_path = studio.generate_a4_printable_sheet(
            image_inputs=valid_paths,
            output_pdf_path=desktop_pdf_path if payload.save_to_desktop else None,
        )

        # Also keep a copy in data/studio/ for immediate in-browser viewing
        studio_filename = f"a4_{timestamp}.pdf"
        (studio.STUDIO_DIR / studio_filename).write_bytes(pdf_bytes)

        return {
            "ok": True,
            "desktop_path": str(desktop_pdf_path) if payload.save_to_desktop else None,
            "filename": pdf_filename,
            "pdf_url": _build_image_api_url(f"data/studio/{studio_filename}"),
        }
    except Exception as exc:
        logger.error("A4 print generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"A4 generation failed: {exc}")


# -----------------------------------------------------------------------------
# OS Integration & Safe Image Streaming Endpoints
# -----------------------------------------------------------------------------


@app.post("/api/open-folder")
def api_open_folder(payload: OpenFolderRequest) -> Dict[str, Any]:
    """Highlight the exact file inside Windows Explorer using `explorer /select,"<path>"`."""
    try:
        norm_path = os.path.normpath(database.normalize_file_path(payload.file_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if '"' in norm_path or "\x00" in norm_path:
        raise HTTPException(status_code=400, detail="Invalid characters in file path.")

    # Authorize that the path is an indexed file in DocuVault AI
    indexed = scan_manager._db.get_file_by_path(norm_path)
    if indexed is None:
        raise HTTPException(
            status_code=403,
            detail="Access denied: file is not indexed in DocuVault AI.",
        )

    if not os.path.exists(norm_path):
        raise HTTPException(
            status_code=404,
            detail=f"File no longer exists on disk: {norm_path}",
        )

    if os.name != "nt":
        raise HTTPException(
            status_code=501,
            detail="Windows Explorer integration is only supported on Windows.",
        )

    subprocess.Popen(["explorer", f"/select,{os.path.normpath(norm_path)}"])
    return {
        "ok": True,
        "file_path": norm_path,
    }


@app.post("/api/pick-folder")
def api_pick_folder() -> Dict[str, Any]:
    """Open a native Windows folder selection dialog and return the chosen directory path."""
    selected_folder: Optional[str] = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Select Folder to Scan - DocuVault AI")
        root.destroy()
        if chosen and chosen.strip():
            selected_folder = os.path.normpath(chosen.strip())
    except Exception as exc:
        logger.warning("Native folder picker unavailable: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not open folder picker dialog: {exc}",
        ) from exc

    return {
        "folder_path": selected_folder,
    }


@app.post("/api/pick-reference-image")
def api_pick_reference_image() -> Dict[str, Any]:
    """Open a native file selection dialog for choosing a single reference face image (JPG/PNG/WebP)."""
    selected_file: Optional[str] = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Select Reference Face Image - DocuVault AI",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.webp"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        if chosen and chosen.strip():
            selected_file = os.path.normpath(chosen.strip())
    except Exception as exc:
        logger.warning("Native reference image picker unavailable: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not open image picker dialog: {exc}",
        ) from exc

    return {
        "file_path": selected_file,
    }


class FolderFaceSearchRequest(BaseModel):
    image_b64: Optional[str] = Field(default=None, description="Cropped target face (data URI / base64).")
    image_path: Optional[str] = Field(default=None, description="Local path of the target face image.")
    image_name: Optional[str] = Field(default=None)
    folder_path: str = Field(..., description="The ONLY folder to scan (recursively).")
    threshold: float = Field(default=0.45, ge=0.2, le=0.9)
    recursive: bool = Field(default=True)


class _FolderFaceJob:
    def __init__(self, folder: str, ref_name: str) -> None:
        self.id = os.urandom(6).hex()
        self.folder = folder
        self.ref_name = ref_name
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = "running"  # running | done | cancelled | error
        self.phase = "extracting"
        self.error: Optional[str] = None
        self.scanned = 0
        self.total = 0
        self.matches: List[Dict[str, Any]] = []
        self.started = time.time()


_FOLDER_FACE_JOBS: Dict[str, _FolderFaceJob] = {}


def _decode_reference_input(image_b64: Optional[str], image_path: Optional[str]) -> Any:
    import base64
    from PIL import Image, ImageOps

    if image_path and image_path.strip():
        norm = database.normalize_file_path(image_path.strip())
        if not os.path.isfile(norm):
            raise HTTPException(status_code=404, detail=f"Reference image not found: {norm}")
        return norm
    if image_b64 and image_b64.strip():
        raw = image_b64.strip()
        if "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            with Image.open(io.BytesIO(base64.b64decode(raw))) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode reference image: {exc}") from exc
    raise HTTPException(status_code=400, detail="Either image_b64 or image_path must be provided.")


def _run_folder_face_job(job: _FolderFaceJob, embedding: List[float], threshold: float, recursive: bool) -> None:
    last_ws_emit = 0.0

    def _progress(done: int, total: int, new_match: Optional[Dict[str, Any]]) -> None:
        nonlocal last_ws_emit
        now_mono = time.monotonic()
        emit = False
        match_count = 0
        with job.lock:
            job.scanned, job.total = done, total
            if new_match is not None and all(m.get("id") != new_match.get("id") for m in job.matches):
                job.matches.append(new_match)
                job.matches.sort(key=lambda d: float(d.get("face_similarity", 0.0)), reverse=True)
                emit = True
            match_count = len(job.matches)
            if done == 1 or done >= total or (now_mono - last_ws_emit) >= 0.08:
                emit = True
                last_ws_emit = now_mono
        if emit:
            ws_hub.broadcast_from_thread(
                {
                    "type": "folder_face_progress",
                    "job_id": job.id,
                    "state": job.state,
                    "phase": job.phase,
                    "scanned": done,
                    "total": total,
                    "total_matches": match_count,
                    "toast_message": f"Scanning {done}/{total}..." if total > 0 else f"Scanning {done}...",
                }
            )

    try:
        result = processor.search_folder_for_face(
            job.folder,
            embedding,
            similarity_threshold=threshold,
            db_manager=scan_manager._db,
            recursive=recursive,
            progress_callback=_progress,
            stop_event=job.stop,
            phase_callback=lambda ph: setattr(job, "phase", ph),
        )
        with job.lock:
            job.matches = result["matches"]
            job.scanned = result["scanned"]
            job.total = result["total_images"]
            job.state = "cancelled" if result["cancelled"] else "done"
            done_scanned, done_total, done_state, done_matches = (
                job.scanned,
                job.total,
                job.state,
                len(job.matches),
            )
        ws_hub.broadcast_from_thread(
            {
                "type": "folder_face_progress",
                "job_id": job.id,
                "state": done_state,
                "phase": "done",
                "scanned": done_scanned,
                "total": done_total,
                "total_matches": done_matches,
                "toast_message": f"Face scan {done_state}: {done_matches} match(es) in {done_scanned}/{done_total} photos",
            }
        )
    except Exception as exc:
        logger.error("Folder face search failed: %s", exc)
        with job.lock:
            job.state = "error"
            job.error = str(exc)
        ws_hub.broadcast_from_thread(
            {
                "type": "folder_face_progress",
                "job_id": job.id,
                "state": "error",
                "error": str(exc),
                "toast_message": f"Folder face search error: {exc}",
            }
        )


def _folder_job_payload(job: _FolderFaceJob) -> Dict[str, Any]:
    with job.lock:
        docs = []
        for m in job.matches:
            enriched = _enrich_document_record(m)
            enriched["face_similarity"] = m.get("face_similarity")
            enriched["face_match_percent"] = m.get("face_match_percent")
            docs.append(enriched)
        return {
            "ok": True,
            "job_id": job.id,
            "state": job.state,
            "phase": job.phase,
            "error": job.error,
            "folder": job.folder,
            "reference_image_name": f"{job.ref_name} in {os.path.basename(job.folder.rstrip(os.sep)) or job.folder}",
            "scanned": job.scanned,
            "total": job.total,
            "total_matches": len(docs),
            "documents": docs,
        }


@app.post("/api/search-folder-face")
def api_search_folder_face(payload: FolderFaceSearchRequest) -> Dict[str, Any]:
    """Targeted Folder Face Search: scan ONLY `folder_path` for the cropped target face (offline, GPU)."""
    folder = database.normalize_file_path(payload.folder_path.strip())
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    ref_input = _decode_reference_input(payload.image_b64, payload.image_path)
    try:
        ref = processor.extract_reference_face_embedding(ref_input)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the target face: {exc}") from exc

    embedding = ref.get("insight_embedding")
    if embedding is None:
        if not processor.face_engine.is_available():
            raise HTTPException(
                status_code=503,
                detail="InsightFace is not available, so folder face search cannot run.",
            )
        raise HTTPException(
            status_code=422,
            detail="No clear face found in the cropped area. Re-crop tighter around the face.",
        )

    job = _FolderFaceJob(folder, payload.image_name or "Reference Face")
    _FOLDER_FACE_JOBS[job.id] = job
    for old_id in [k for k, j in _FOLDER_FACE_JOBS.items() if j.state != "running" and time.time() - j.started > 3600]:
        _FOLDER_FACE_JOBS.pop(old_id, None)
    threading.Thread(
        target=_run_folder_face_job,
        args=(job, embedding, float(payload.threshold), bool(payload.recursive)),
        name=f"folder-face-{job.id}",
        daemon=True,
    ).start()

    out = _folder_job_payload(job)
    out["reference_face_b64"] = ref.get("reference_face_b64")
    return out


@app.get("/api/search-folder-face/{job_id}")
def api_search_folder_face_status(job_id: str) -> Dict[str, Any]:
    job = _FOLDER_FACE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown folder face search job.")
    return _folder_job_payload(job)


@app.post("/api/search-folder-face/{job_id}/cancel")
def api_search_folder_face_cancel(job_id: str) -> Dict[str, Any]:
    job = _FOLDER_FACE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown folder face search job.")
    job.stop.set()
    return {"ok": True}


class ReferenceImageDataRequest(BaseModel):
    image_path: str = Field(..., description="Local path of a user-selected reference image.")


@app.post("/api/reference-image-data")
def api_reference_image_data(payload: ReferenceImageDataRequest) -> Dict[str, Any]:
    """Return a (downscaled) JPEG data URL of a user-selected local image for the square face cropper."""
    import base64
    from PIL import Image, ImageOps

    norm_path = database.normalize_file_path(payload.image_path.strip())
    if os.path.splitext(norm_path)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise HTTPException(status_code=400, detail="Reference file must be an image (JPG/PNG/WebP/BMP).")
    if not os.path.isfile(norm_path):
        raise HTTPException(status_code=404, detail=f"Reference image not found: {norm_path}")
    try:
        with Image.open(norm_path) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        img.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc
    return {
        "ok": True,
        "name": os.path.basename(norm_path),
        "data_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    }


@app.post("/api/search-by-face")
@app.post("/api/search_face")
def api_search_by_face(payload: ReferenceFaceSearchRequest) -> Dict[str, Any]:
    """
    Reverse Face Search:
    1. Load the user-selected reference image (from `image_path` or `image_b64`).
    2. Extract its 512-d face embedding via InsightFace / processor.py.
    3. Run Cosine Similarity comparison (threshold ~0.45-0.50) against all stored face embeddings in SQLite.
    4. Return all matching records (both ID Vault documents and Gallery Photos).
    """
    import base64
    from PIL import Image, ImageOps

    image_input: Any = None
    ref_name = payload.image_name or "Reference Face"

    if payload.image_path and payload.image_path.strip():
        norm_path = database.normalize_file_path(payload.image_path.strip())
        if not os.path.isfile(norm_path):
            raise HTTPException(status_code=404, detail=f"Reference image not found: {norm_path}")
        image_input = norm_path
        ref_name = os.path.basename(norm_path)
    elif payload.image_b64 and payload.image_b64.strip():
        raw_b64 = payload.image_b64.strip()
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
            with Image.open(io.BytesIO(img_bytes)) as opened:
                image_input = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode reference image: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="Either image_path or image_b64 must be provided.")

    try:
        search_res = processor.search_by_reference_face(
            image_input=image_input,
            db_manager=scan_manager._db,
            similarity_threshold=float(payload.threshold),
        )
    except Exception as exc:
        logger.error("Reverse face search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Face search failed: {exc}") from exc

    enriched_docs: List[Dict[str, Any]] = []
    for doc in search_res.get("matched_documents", []):
        enriched = _enrich_document_record(doc)
        enriched["face_similarity"] = doc.get("face_similarity")
        enriched["face_match_percent"] = doc.get("face_match_percent")
        enriched_docs.append(enriched)

    enriched_people: List[Dict[str, Any]] = []
    for p in search_res.get("matched_people", []):
        ep = _enrich_person_record(p)
        ep["similarity"] = p.get("similarity")
        ep["similarity_percent"] = p.get("similarity_percent")
        enriched_people.append(ep)

    return {
        "ok": True,
        "reference_image_name": ref_name,
        "reference_face_b64": search_res.get("reference_face_b64"),
        "face_detected": bool(search_res.get("face_detected", False)),
        "threshold": float(search_res.get("threshold", payload.threshold)),
        "total_matches": len(enriched_docs),
        "matched_people": enriched_people,
        "documents": enriched_docs,
    }


@app.get("/api/image")
def api_stream_image(
    path: str = Query(..., description="Relative `data/...` path or indexed full file path."),
    raw: bool = Query(default=False, description="If True and file is a PDF, stream raw PDF bytes."),
):
    """
    Safely stream local document thumbnails, cropped face avatars, or full document images.
    For `.pdf` documents, renders Page 1 as a crisp JPEG preview by default so `<img>` tags work seamlessly.
    """
    resolved = _resolve_and_authorize_image_path(path, active_db=scan_manager._db)
    ext = resolved.suffix.lower()

    if ext == ".pdf" and not raw:
        try:
            pil_img = processor.load_visual_image(resolved)
        except processor.PasswordProtectedError:
            raise HTTPException(status_code=403, detail="PDF is password-protected.")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Unable to render PDF preview: {exc}")

        if pil_img is None:
            raise HTTPException(status_code=422, detail="Unable to render PDF preview.")
        try:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=88)
            return Response(
                content=buf.getvalue(),
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=3600"},
            )
        finally:
            try:
                pil_img.close()
            except Exception:
                pass

    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }
    media_type = mime_map.get(ext, "application/octet-stream")
    return FileResponse(
        path=str(resolved),
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
