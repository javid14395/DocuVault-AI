"""
DocuVault AI - Live Real-Time Folder Watcher (watcher.py)

Monitors the user's `Downloads`, `Desktop`, and `Documents` folders in the background
using `watchdog`. Whenever a new or downloaded `.jpg`, `.jpeg`, `.png`, `.webp`, or `.pdf`
file lands in those folders (including browser `.crdownload` / `.part` rename completions),
waits for write-lock stabilization and automatically runs `processor.process_file` in real
time without requiring a manual scan.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Union

from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

try:
    from backend import database, processor, scanner
except ImportError:
    import database, processor, scanner

logger = logging.getLogger("docuvault.watcher")

WATCHED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
MIN_WATCH_FILE_SIZE_BYTES: int = 8 * 1024              # 8 KB minimum (processor Step A enforces >= 300x300 px)
MAX_WATCH_FILE_SIZE_BYTES: int = 25 * 1024 * 1024      # 25 MB maximum
DEBOUNCE_DELAY_SECONDS: float = 0.65
FILE_READY_TIMEOUT_SECONDS: float = 20.0


def get_default_watch_directories() -> List[Path]:
    """
    Resolve the user's real `Downloads`, `Desktop`, and `Documents` folders on Windows,
    including OneDrive-redirected Shell folders if configured.
    """
    candidates: List[Path] = []
    home = Path.home()

    # 1. Query Windows User Shell Folders registry (handles OneDrive Desktop/Documents redirection)
    if os.name == "nt":
        try:
            import winreg

            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
            shell_keys = (
                "{374DE290-123F-4565-9164-39C4925E467B}",  # Downloads
                "Desktop",                                  # Desktop
                "Personal",                                 # Documents
            )
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                for val_name in shell_keys:
                    try:
                        raw_val, _ = winreg.QueryValueEx(key, val_name)
                        if raw_val:
                            expanded = os.path.expandvars(str(raw_val)).strip()
                            candidates.append(Path(os.path.abspath(expanded)))
                    except OSError:
                        continue
        except Exception as exc:
            logger.debug("Registry shell folder lookup skipped: %s", exc)

    # 2. Standard home folders
    for folder_name in ("Downloads", "Desktop", "Documents"):
        candidates.append(Path(os.path.abspath(str(home / folder_name))))

    # Deduplicate existing directories case-insensitively
    unique_dirs: List[Path] = []
    seen_lower: Set[str] = set()
    for folder in candidates:
        norm = database.normalize_file_path(folder)
        key = norm.lower()
        if key not in seen_lower and os.path.isdir(norm):
            seen_lower.add(key)
            unique_dirs.append(Path(norm))

    return unique_dirs


class _VaultEventHandler(FileSystemEventHandler):
    """Watchdog event handler forwarding candidate file creations, moves, and writes."""

    def __init__(self, enqueue_callback: Callable[[str], None]) -> None:
        super().__init__()
        self._enqueue = enqueue_callback

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(os.fsdecode(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        # Crucial for Chrome/Edge/Firefox downloads: `file.crdownload` -> `aadhaar.pdf`
        if not event.is_directory:
            self._enqueue(os.fsdecode(event.dest_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(os.fsdecode(event.src_path))


class LiveFolderWatcher:
    """
    Background real-time folder watcher monitoring `Downloads`, `Desktop`, and `Documents`.
    Automatically visually inspects newly dropped/downloaded images & PDFs via `processor.py`.
    """

    def __init__(
        self,
        db_manager: Optional[database.DatabaseManager] = None,
        watch_dirs: Optional[Sequence[Union[str, Path]]] = None,
        ocr_engine: Any = None,
        debounce_seconds: float = DEBOUNCE_DELAY_SECONDS,
    ) -> None:
        self._db = db_manager or database.db
        self._custom_watch_dirs = (
            [Path(os.path.abspath(str(p))) for p in watch_dirs]
            if watch_dirs is not None
            else None
        )
        self._ocr_engine = ocr_engine
        self._debounce_seconds = max(0.1, float(debounce_seconds))

        self._lock = threading.Lock()
        self._pending: Dict[str, float] = {}
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        self._observer: Optional[Any] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._is_running: bool = False
        self._watched_folders: List[str] = []
        self._auto_processed_count: int = 0
        self._auto_documents_found: int = 0
        self._last_event_file: Optional[str] = None
        self._last_event_category: Optional[str] = None
        self._last_event_at: Optional[str] = None
        self._recent_events: Deque[Dict[str, Any]] = collections.deque(maxlen=15)

    def start(self) -> bool:
        """Start background `watchdog` observer and debounced visual inspection thread."""
        with self._lock:
            if self._is_running:
                return True

            dirs = (
                [d for d in self._custom_watch_dirs if d.is_dir()]
                if self._custom_watch_dirs is not None
                else get_default_watch_directories()
            )
            if not dirs:
                logger.warning("LiveFolderWatcher: No valid directories found to monitor.")
                return False

            self._stop_event.clear()
            self._wake_event.clear()

            observer = Observer()
            observer.daemon = True
            handler = _VaultEventHandler(self._enqueue_candidate_path)

            scheduled_paths: List[str] = []
            for folder in dirs:
                norm_dir = database.normalize_file_path(folder)
                try:
                    observer.schedule(handler, norm_dir, recursive=True)
                    scheduled_paths.append(norm_dir)
                except Exception as exc:
                    logger.warning("Could not watch folder '%s': %s", norm_dir, exc)

            if not scheduled_paths:
                return False

            observer.start()
            self._observer = observer
            self._watched_folders = scheduled_paths
            self._is_running = True

            self._worker_thread = threading.Thread(
                target=self._consume_loop,
                name="DocuVaultLiveWatcherWorker",
                daemon=True,
            )
            self._worker_thread.start()

            logger.info("Live real-time watcher active on: %s", ", ".join(scheduled_paths))
            return True

    def stop(self, timeout: float = 2.0) -> None:
        """Gracefully stop the `watchdog` observer and background processing loop."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()
            self._wake_event.set()
            observer = self._observer
            worker = self._worker_thread
            self._observer = None
            self._worker_thread = None

        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=timeout)
            except Exception:
                pass

        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def get_status(self) -> Dict[str, Any]:
        """Return live telemetry for the real-time folder watcher."""
        with self._lock:
            return {
                "is_running": self._is_running,
                "watched_folders": list(self._watched_folders),
                "pending_queue_size": len(self._pending),
                "auto_processed_count": self._auto_processed_count,
                "auto_documents_found": self._auto_documents_found,
                "last_event_file": self._last_event_file,
                "last_event_category": self._last_event_category,
                "last_event_at": self._last_event_at,
                "recent_events": list(self._recent_events),
            }

    def _enqueue_candidate_path(self, raw_path: str) -> None:
        """Filter incoming filesystem events and schedule matching files for debounced inspection."""
        if not raw_path:
            return
        try:
            norm_path = database.normalize_file_path(raw_path)
        except ValueError:
            return

        lower_path = norm_path.lower()
        ext = os.path.splitext(lower_path)[1]
        if ext not in WATCHED_EXTENSIONS:
            return

        # Ignore internal DocuVault thumbnails/faces and system/dev directories
        internal_prefix = database.normalize_file_path(processor.DATA_DIR).lower().rstrip("\\") + "\\"
        if lower_path.startswith(internal_prefix):
            return
        for excluded_sub in ("\\node_modules\\", "\\.git\\", "\\$recycle.bin\\", "\\__pycache__\\"):
            if excluded_sub in lower_path:
                return

        due_time = time.monotonic() + self._debounce_seconds
        with self._lock:
            self._pending[norm_path] = due_time
        self._wake_event.set()

    def _wait_until_file_ready(self, file_path: str) -> bool:
        """
        Wait until a newly downloaded or copied file has finished writing on disk
        and its exclusive write lock is released.
        """
        deadline = time.monotonic() + FILE_READY_TIMEOUT_SECONDS
        prev_size = -1

        while time.monotonic() < deadline and not self._stop_event.is_set():
            try:
                st = os.stat(file_path)
                curr_size = int(st.st_size)
            except OSError:
                return False

            if curr_size > MAX_WATCH_FILE_SIZE_BYTES:
                return False

            if curr_size >= MIN_WATCH_FILE_SIZE_BYTES and curr_size == prev_size:
                try:
                    with open(file_path, "rb") as fh:
                        fh.read(16)
                    return True
                except (PermissionError, OSError):
                    pass

            prev_size = curr_size
            time.sleep(0.25)

        return False

    def _consume_loop(self) -> None:
        """Background thread draining debounced file events and running `processor.process_file`."""
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                break

            now = time.monotonic()
            ready_paths: List[str] = []

            with self._lock:
                for path, due in list(self._pending.items()):
                    if now >= due:
                        ready_paths.append(path)
                        del self._pending[path]
                if not self._pending:
                    self._wake_event.clear()

            for file_path in ready_paths:
                if self._stop_event.is_set():
                    break
                if not os.path.isfile(file_path):
                    continue
                if not self._wait_until_file_ready(file_path):
                    continue

                try:
                    self._db.init_db()
                    res = processor.process_file(
                        file_path,
                        db_manager=self._db,
                        ocr_engine=self._ocr_engine,
                    )
                    if res.get("status") == "skipped_already_scanned":
                        continue

                    is_doc = bool(res.get("is_document", False))
                    cat = str(res.get("doc_category", database.NOT_A_DOCUMENT))
                    ts = datetime.now(timezone.utc).isoformat()

                    with self._lock:
                        self._auto_processed_count += 1
                        if is_doc:
                            self._auto_documents_found += 1
                        self._last_event_file = file_path
                        self._last_event_category = cat
                        self._last_event_at = ts
                        self._recent_events.appendleft(
                            {
                                "file_path": file_path,
                                "file_name": os.path.basename(file_path),
                                "is_document": is_doc,
                                "doc_category": cat,
                                "extracted_id_number": res.get("extracted_id_number"),
                                "timestamp": ts,
                            }
                        )

                    logger.info(
                        "LiveWatcher auto-indexed '%s' -> %s (is_document=%s)",
                        file_path,
                        cat,
                        is_doc,
                    )
                except Exception as exc:
                    logger.warning("LiveWatcher error processing '%s': %s", file_path, exc)
