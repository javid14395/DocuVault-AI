"""
DocuVault AI - Native Windows Desktop Application Entrypoint (main.py)

Bootstraps required local directories (`data/faces`, `data/thumbnails`), initializes
the SQLite database (`docuvault.db`), launches the local FastAPI backend (`server:app`)
via Uvicorn on `127.0.0.1:8765` in a background daemon thread, and opens the native
desktop window using `pywebview`.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
import webview

import database
import server
import setup_assets

HOST = "127.0.0.1"
PORT = 8765
APP_URL = f"http://{HOST}:{PORT}"
WINDOW_TITLE = "DocuVault AI - Smart Document & ID Finder"


def _file_dialog_type(kind: str):
    """Return the modern `webview.FileDialog.<KIND>` constant (pywebview >= 5.x), falling back to the
    legacy `webview.<KIND>_DIALOG` constant on older versions. Avoids deprecation warnings."""
    enum = getattr(webview, "FileDialog", None)
    if enum is not None and hasattr(enum, kind):
        return getattr(enum, kind)
    return getattr(webview, f"{kind}_DIALOG")


class PyWebViewApi:
    """Native Python API bridge exposed directly to the frontend via `window.pywebview.api`."""

    @staticmethod
    def _safe_call(fn, *args, **kwargs) -> dict:
        """Guard pywebview JS-to-Python RPC calls so FastAPI HTTPExceptions or runtime errors return structured dicts."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            status_code = getattr(exc, "status_code", 500)
            logging.getLogger("docuvault.bridge").warning("PyWebViewApi error (%s): %s", status_code, detail)
            return {"ok": False, "error": str(detail), "detail": str(detail), "status_code": status_code}

    def scan_folder(
        self,
        folder_path: Optional[str] = None,
        scan_mode: str = "id_vault",
        target_category: Optional[str] = None,
    ) -> dict:
        req = server.ScanStartRequest(
            folder_path=folder_path,
            scan_mode=scan_mode,
            target_category=target_category,
        )
        return self._safe_call(server.api_scan_start, req)

    def full_pc_scan(
        self,
        scan_mode: str = "id_vault",
        target_category: Optional[str] = None,
    ) -> dict:
        req = server.ScanStartRequest(
            folder_path=None,
            scan_mode=scan_mode,
            target_category=target_category,
        )
        return self._safe_call(server.api_scan_start, req)

    def stop_scan(self) -> dict:
        return self._safe_call(server.api_scan_stop)

    def get_scan_status(self) -> dict:
        return self._safe_call(server.api_scan_status)

    def pick_folder(self) -> dict:
        try:
            if webview.windows:
                result = webview.windows[0].create_file_dialog(_file_dialog_type("FOLDER"))
                if result:
                    chosen = result[0] if isinstance(result, (list, tuple)) else str(result)
                    if chosen and chosen.strip():
                        return {"folder_path": os.path.normpath(chosen.strip())}
                return {"folder_path": None}  # user cancelled
        except Exception:
            pass
        return self._safe_call(server.api_pick_folder)

    def pick_reference_image(self) -> dict:
        """Open a native pywebview file dialog (FileDialog.OPEN) allowing the user to select a single reference face image (JPG/PNG)."""
        try:
            if webview.windows:
                file_types = ("Image Files (*.jpg;*.jpeg;*.png;*.webp)", "All files (*.*)")
                result = webview.windows[0].create_file_dialog(
                    _file_dialog_type("OPEN"),
                    allow_multiple=False,
                    file_types=file_types,
                )
                if result:
                    chosen = result[0] if isinstance(result, (list, tuple)) else str(result)
                    if chosen and chosen.strip():
                        return {"file_path": os.path.normpath(chosen.strip())}
                return {"file_path": None}
        except Exception:
            pass
        return self._safe_call(server.api_pick_reference_image)

    def search_by_face(self, image_path: Optional[str] = None, threshold: float = 0.48) -> dict:
        """Run Reverse Face Search against all stored face embeddings in docuvault.db."""
        req = server.ReferenceFaceSearchRequest(image_path=image_path, threshold=threshold)
        return self._safe_call(server.api_search_by_face, req)

    def get_stats(self) -> dict:
        return self._safe_call(server.api_get_stats)

    def get_documents(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        person_id: Optional[int] = None,
        scan_mode: Optional[str] = None,
        sort_by: str = "scanned_desc",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        return self._safe_call(
            server.api_get_documents,
            category=category,
            search=search,
            person_id=person_id,
            expiring_only=False,
            scan_mode=scan_mode,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )


def ensure_app_directories() -> None:
    """Create `data/faces` and `data/thumbnails` directories on startup."""
    faces_dir = BASE_DIR / "data" / "faces"
    thumbnails_dir = BASE_DIR / "data" / "thumbnails"
    faces_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)


def wait_for_server_ready(host: str, port: int, timeout_sec: float = 12.0) -> bool:
    """Block briefly until the background Uvicorn server begins accepting TCP connections."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def start_backend_thread() -> tuple[uvicorn.Server, threading.Thread]:
    """Configure and start `uvicorn` serving `server:app` on `127.0.0.1:8765`."""
    config = uvicorn.Config(
        server.app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    )
    uv_server = uvicorn.Server(config=config)
    thread = threading.Thread(
        target=uv_server.run,
        name="DocuVaultUvicornThread",
        daemon=True,
    )
    thread.start()
    return uv_server, thread


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("docuvault.main")

    os.chdir(BASE_DIR)
    ensure_app_directories()
    setup_assets.ensure_local_assets(force_download=False)
    database.init_db()

    logger.info("Starting local DocuVault AI backend at %s ...", APP_URL)
    uv_server, uv_thread = start_backend_thread()

    if not wait_for_server_ready(HOST, PORT, timeout_sec=12.0):
        logger.error("Backend server failed to bind %s:%d within timeout.", HOST, PORT)
        sys.exit(1)

    logger.info("Backend ready. Launching native desktop window...")

    api_bridge = PyWebViewApi()

    try:
        webview.create_window(
            WINDOW_TITLE,
            APP_URL,
            js_api=api_bridge,
            width=1360,
            height=860,
            min_size=(1000, 650),
            background_color="#070b14",
        )
        webview.start()
    except Exception as exc:
        logger.warning(
            "Native pywebview window unavailable (%s). Falling back to default browser at %s",
            exc,
            APP_URL,
        )
        webbrowser.open(APP_URL)
        try:
            while uv_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    finally:
        logger.info("Shutting down DocuVault AI cleanly...")
        try:
            server.scan_manager.stop_scan(wait_timeout=2.0)
        except Exception:
            pass
        uv_server.should_exit = True
        uv_thread.join(timeout=3.0)
        database.db.close()


if __name__ == "__main__":
    main()
