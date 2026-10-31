"""
DocuVault AI - Ultra-Fast Candidate File Scanner (scanner.py)

Discovers candidate images (.jpg, .jpeg, .png, .webp) and .pdf files across drives
in milliseconds via Voidtools Everything SDK (`Everything64.dll` IPC), with an
automatic, pruned `os.walk` fallback when Everything is not running or when scanning
a custom target folder.

Never classifies by filename — all candidate files passing size (25 KB - 25 MB)
and modification-time checks are handed off to `processor.py` for visual inspection.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import threading
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import database

logger = logging.getLogger("docuvault.scanner")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CANDIDATE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}

MIN_FILE_SIZE_BYTES: int = 25 * 1024           # 25 KB
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024    # 25 MB

# Standard EXIF Date Taken tags (Pillow / ExifTags)
EXIF_TAG_DATETIME_ORIGINAL = 36867   # DateTimeOriginal ("Date Taken")
EXIF_TAG_DATETIME_DIGITIZED = 36868  # DateTimeDigitized
EXIF_TAG_DATETIME = 306              # DateTime (ModifyDate)
EXIF_IFD_POINTER = 0x8769            # ExifOffset sub-IFD pointer

_RE_EXIF_DT = re.compile(
    r"(\d{4})[:\-/](\d{2})[:\-/](\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2}))?"
)


def _normalize_exif_datetime(raw_value: Any) -> Optional[str]:
    """
    Parse an EXIF date string (typically 'YYYY:MM:DD HH:MM:SS') into canonical
    ISO-sortable 'YYYY-MM-DD HH:MM:SS'. Returns None if invalid or zeroed.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        try:
            text = raw_value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    else:
        text = str(raw_value)
    text = text.replace("\x00", "").strip()
    if not text or text.startswith("0000"):
        return None

    m = _RE_EXIF_DT.search(text)
    if not m:
        return None
    try:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hour = int(m.group(4)) if m.group(4) is not None else 0
        minute = int(m.group(5)) if m.group(5) is not None else 0
        second = int(m.group(6)) if m.group(6) is not None else 0
        if not (1900 <= year <= 2100):
            return None
        dt = datetime(year, month, day, hour, minute, second)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def extract_exif_date_taken(
    source: Union[str, Path, Any],
    fallback_mtime: Optional[float] = None,
    use_mtime_fallback: bool = True,
) -> Optional[str]:
    """
    Extract the original 'Date Taken' (EXIF DateTimeOriginal, tag 36867) from an image
    path or open `PIL.Image.Image` using Pillow (`PIL.Image` / `PIL.ExifTags`).

    Priority:
      1. EXIF Sub-IFD / IFD0 `DateTimeOriginal` (36867)
      2. EXIF Sub-IFD / IFD0 `DateTimeDigitized` (36868)
      3. EXIF IFD0 `DateTime` (306)
      4. File modification timestamp (`fallback_mtime` or `os.stat().st_mtime`) when
         `use_mtime_fallback=True`, formatted as 'YYYY-MM-DD HH:MM:SS' for chronological sorting.
    """
    from PIL import ExifTags, Image

    # Resolve tag ID dynamically via PIL.ExifTags.TAGS as well for completeness
    dto_tag = next(
        (tid for tid, name in ExifTags.TAGS.items() if name == "DateTimeOriginal"),
        EXIF_TAG_DATETIME_ORIGINAL,
    )

    def _extract_from_pil(pil_obj: Image.Image) -> Optional[str]:
        candidates: List[Any] = []
        try:
            exif_data = pil_obj.getexif()
            if exif_data:
                # In Pillow >= 10, DateTimeOriginal lives inside the Exif sub-IFD (0x8769)
                exif_ifd: Dict[Any, Any] = {}
                try:
                    exif_ifd = exif_data.get_ifd(EXIF_IFD_POINTER) or {}
                except Exception:
                    exif_ifd = {}

                for tag_id in (dto_tag, EXIF_TAG_DATETIME_DIGITIZED, EXIF_TAG_DATETIME):
                    if tag_id in exif_ifd:
                        candidates.append(exif_ifd.get(tag_id))
                    if tag_id in exif_data:
                        candidates.append(exif_data.get(tag_id))
        except Exception:
            pass

        # Legacy Pillow _getexif() fallback for older JPEG/TIFF containers
        if not candidates and hasattr(pil_obj, "_getexif"):
            try:
                raw_exif = pil_obj._getexif()  # type: ignore[attr-defined]
                if isinstance(raw_exif, dict):
                    for tag_id in (dto_tag, EXIF_TAG_DATETIME_DIGITIZED, EXIF_TAG_DATETIME):
                        if tag_id in raw_exif:
                            candidates.append(raw_exif[tag_id])
            except Exception:
                pass

        for cand in candidates:
            parsed = _normalize_exif_datetime(cand)
            if parsed:
                return parsed
        return None

    if isinstance(source, Image.Image):
        found_dt = _extract_from_pil(source)
        if found_dt:
            return found_dt
    elif isinstance(source, (str, Path)):
        path_obj = Path(source)
        ext = path_obj.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".jfif"} and path_obj.is_file():
            try:
                if path_obj.stat().st_size > 0:
                    with Image.open(path_obj) as opened_img:
                        found_dt = _extract_from_pil(opened_img)
                        if found_dt:
                            return found_dt
            except Exception:
                pass
        if fallback_mtime is None and use_mtime_fallback:
            try:
                fallback_mtime = float(path_obj.stat().st_mtime)
            except OSError:
                fallback_mtime = None

    if use_mtime_fallback and fallback_mtime is not None and fallback_mtime > 0:
        try:
            return datetime.fromtimestamp(float(fallback_mtime)).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            return None

    return None

_CACHED_HARDWARE_PROFILE: Optional[Dict[str, Any]] = None


def detect_hardware_profile(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Intelligently profile host hardware at startup (cached in memory):
    1. Detect total system RAM (via Win32 GlobalMemoryStatusEx or psutil fallback)
    2. Check GPU acceleration (NVIDIA CUDA / TensorRT / DirectML in ONNX Runtime or OpenCV CUDA)
    3. Dynamically scale background ThreadPoolExecutor worker count:
       - Low RAM (<8GB) / CPU only: max_workers = 2
       - Mid RAM (8GB-16GB) / Standard GPU: max_workers = 4 to 6
       - High RAM (32GB+) / Dual GPU / Server configs: max_workers = 8 to 12
    """
    global _CACHED_HARDWARE_PROFILE
    if _CACHED_HARDWARE_PROFILE is not None and not force_refresh:
        return dict(_CACHED_HARDWARE_PROFILE)

    total_ram_gb = 8.0
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        try:
            import psutil
            total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            pass

    cpu_count = os.cpu_count() or 4
    gpu_available = False
    gpu_name = "CPU Only"

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        accel = [p for p in providers if any(k in p for k in ("CUDA", "Tensorrt", "Dml", "OpenVINO"))]
        if accel:
            gpu_available = True
            gpu_name = ", ".join(accel)
    except Exception:
        pass

    if not gpu_available:
        try:
            import cv2
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_available = True
                gpu_name = f"OpenCV CUDA ({cv2.cuda.getCudaEnabledDeviceCount()} dev)"
        except Exception:
            pass

    # Hardware-adaptive worker scaling:
    # 2 workers for <8GB, 4 for 8-16GB, max 8 for >16GB
    if total_ram_gb < 8.0:
        max_workers = 2
    elif total_ram_gb <= 16.0:
        max_workers = 4
    else:
        max_workers = min(8, max(4, cpu_count))

    profile = {
        "total_ram_gb": total_ram_gb,
        "cpu_count": cpu_count,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "max_workers": max_workers,
    }
    _CACHED_HARDWARE_PROFILE = profile
    return dict(profile)


# Everything SDK Constants
EVERYTHING_OK = 0
EVERYTHING_ERROR_MEMORY = 1
EVERYTHING_ERROR_IPC = 2
EVERYTHING_ERROR_REGISTERCLASSEX = 3
EVERYTHING_ERROR_CREATEWINDOW = 4
EVERYTHING_ERROR_CREATETHREAD = 5
EVERYTHING_ERROR_INVALIDINDEX = 6
EVERYTHING_ERROR_INVALIDCALL = 7

EVERYTHING_REQUEST_FILE_NAME = 0x00000001
EVERYTHING_REQUEST_PATH = 0x00000002
EVERYTHING_REQUEST_FULL_PATH_AND_FILE_NAME = 0x00000004
EVERYTHING_REQUEST_SIZE = 0x00000010
EVERYTHING_REQUEST_DATE_MODIFIED = 0x00000040

# Windows FILETIME (100-ns intervals since 1601-01-01) -> Unix Epoch (1970-01-01)
FILETIME_EPOCH_DIFF_TICKS = 116444736000000000
TICKS_PER_SECOND = 10_000_000.0
INVALID_FILETIME_TICKS = 0xFFFFFFFFFFFFFFFF

# Canonical Everything IPC search query.
# Note: Paths with spaces must be double-quoted in Everything search syntax so
# spaces are not parsed as an AND term.
EVERYTHING_SEARCH_QUERY = (
    'file: ext:jpg;jpeg;png;webp;pdf '
    '!C:\\Windows\\ '
    '!\\AppData\\ '
    '!"\\Program Files\\" '
    '!"\\Program Files (x86)\\" '
    '!\\node_modules\\ '
    '!\\.git\\'
)

# Directory names (lowercase) skipped during os.walk and post-filtering
EXCLUDED_DIR_NAMES: Set[str] = {
    "appdata",
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "$recycle.bin",
    "system volume information",
    "program files",
    "program files (x86)",
    "programdata",
    "recovery",
    "perflogs",
    "msocache",
    "config.msi",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
}

# Path substrings (lowercase, backslash-delimited) excluded from all scans
EXCLUDED_PATH_SUBSTRINGS: Tuple[str, ...] = (
    "c:\\windows\\",
    "\\appdata\\",
    "\\program files\\",
    "\\program files (x86)\\",
    "\\node_modules\\",
    "\\.git\\",
    "\\$recycle.bin\\",
    "\\system volume information\\",
)

_EVERYTHING_LOCK = threading.Lock()


class EverythingError(RuntimeError):
    """Raised when Everything64.dll cannot be loaded or returns an SDK/IPC error."""

    def __init__(self, message: str, error_code: int = -1) -> None:
        super().__init__(message)
        self.error_code = error_code


class EverythingSDK:
    """Thread-safe ctypes wrapper around `Everything64.dll`."""

    def __init__(self, dll_path: Optional[Union[str, Path]] = None) -> None:
        self.dll_path = self._resolve_dll_path(dll_path)
        self._dll: Optional[ctypes.WinDLL] = None

    @staticmethod
    def _resolve_dll_path(dll_path: Optional[Union[str, Path]] = None) -> Path:
        if dll_path is not None:
            return Path(dll_path).resolve()

        # Primary target per specification: ./Everything64.dll inside workspace
        primary = BASE_DIR / "Everything64.dll"
        cwd_candidate = Path("./Everything64.dll").resolve()

        machine = platform.machine().lower()
        is_64bit = ctypes.sizeof(ctypes.c_void_p) == 8

        if "arm" in machine:
            arch_name = "EverythingARM64.dll" if is_64bit else "EverythingARM.dll"
            arch_candidate = BASE_DIR / arch_name
            if arch_candidate.is_file():
                return arch_candidate
        elif not is_64bit:
            x86_candidate = BASE_DIR / "Everything32.dll"
            if x86_candidate.is_file():
                return x86_candidate

        if primary.is_file():
            return primary
        if cwd_candidate.is_file():
            return cwd_candidate
        return primary

    def _load(self) -> ctypes.WinDLL:
        if self._dll is not None:
            return self._dll

        if os.name != "nt":
            raise EverythingError("Everything SDK is only supported on Windows.")

        if not self.dll_path.is_file():
            raise EverythingError(f"Everything DLL not found at: {self.dll_path}")

        try:
            dll = ctypes.WinDLL(str(self.dll_path))
        except OSError as exc:
            raise EverythingError(f"Failed to load {self.dll_path}: {exc}") from exc

        # Configure ctypes function prototypes
        dll.Everything_SetSearchW.argtypes = [wintypes.LPCWSTR]
        dll.Everything_SetSearchW.restype = None

        dll.Everything_SetRequestFlags.argtypes = [wintypes.DWORD]
        dll.Everything_SetRequestFlags.restype = None

        dll.Everything_QueryW.argtypes = [wintypes.BOOL]
        dll.Everything_QueryW.restype = wintypes.BOOL

        dll.Everything_GetLastError.argtypes = []
        dll.Everything_GetLastError.restype = wintypes.DWORD

        dll.Everything_GetNumResults.argtypes = []
        dll.Everything_GetNumResults.restype = wintypes.DWORD

        dll.Everything_GetResultFullPathNameW.argtypes = [
            wintypes.DWORD,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        dll.Everything_GetResultFullPathNameW.restype = wintypes.DWORD

        dll.Everything_GetResultSize.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        dll.Everything_GetResultSize.restype = wintypes.BOOL

        dll.Everything_GetResultDateModified.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(wintypes.FILETIME),
        ]
        dll.Everything_GetResultDateModified.restype = wintypes.BOOL

        if hasattr(dll, "Everything_IsFileResult"):
            dll.Everything_IsFileResult.argtypes = [wintypes.DWORD]
            dll.Everything_IsFileResult.restype = wintypes.BOOL

        dll.Everything_CleanUp.argtypes = []
        dll.Everything_CleanUp.restype = None

        self._dll = dll
        return dll

    def query_files(self, search_query: str = EVERYTHING_SEARCH_QUERY) -> List[Tuple[str, int, float]]:
        """
        Execute an Everything IPC search query and return a list of
        `(full_path, file_size_bytes, mtime_unix_seconds)` tuples.

        Raises `EverythingError` if Everything IPC is unavailable.
        """
        dll = self._load()

        with _EVERYTHING_LOCK:
            dll.Everything_SetSearchW(search_query)
            dll.Everything_SetRequestFlags(
                EVERYTHING_REQUEST_FILE_NAME
                | EVERYTHING_REQUEST_PATH
                | EVERYTHING_REQUEST_FULL_PATH_AND_FILE_NAME
                | EVERYTHING_REQUEST_SIZE
                | EVERYTHING_REQUEST_DATE_MODIFIED
            )

            ok = dll.Everything_QueryW(True)
            if not ok:
                err_code = int(dll.Everything_GetLastError())
                dll.Everything_CleanUp()
                if err_code == EVERYTHING_ERROR_IPC:
                    raise EverythingError(
                        "Everything IPC service is not running in the system tray.",
                        error_code=err_code,
                    )
                raise EverythingError(
                    f"Everything_QueryW failed with SDK error code {err_code}.",
                    error_code=err_code,
                )

            try:
                num_results = int(dll.Everything_GetNumResults())
                results: List[Tuple[str, int, float]] = []

                buf_capacity = 1024
                path_buf = ctypes.create_unicode_buffer(buf_capacity)
                size_val = ctypes.c_longlong(0)
                ft_val = wintypes.FILETIME()
                has_is_file = hasattr(dll, "Everything_IsFileResult")

                for idx in range(num_results):
                    if has_is_file and not dll.Everything_IsFileResult(idx):
                        continue

                    copied_len = int(
                        dll.Everything_GetResultFullPathNameW(idx, path_buf, buf_capacity)
                    )
                    if copied_len >= buf_capacity - 1:
                        # Grow buffer for deep Windows paths (> 1024 chars)
                        buf_capacity = copied_len + 256
                        path_buf = ctypes.create_unicode_buffer(buf_capacity)
                        dll.Everything_GetResultFullPathNameW(idx, path_buf, buf_capacity)

                    full_path = path_buf.value
                    if not full_path:
                        continue

                    file_size = -1
                    if dll.Everything_GetResultSize(idx, ctypes.byref(size_val)):
                        file_size = int(size_val.value)

                    mtime_sec = -1.0
                    if dll.Everything_GetResultDateModified(idx, ctypes.byref(ft_val)):
                        ticks = (int(ft_val.dwHighDateTime) << 32) | int(ft_val.dwLowDateTime)
                        if 0 < ticks < INVALID_FILETIME_TICKS:
                            mtime_sec = (ticks - FILETIME_EPOCH_DIFF_TICKS) / TICKS_PER_SECOND

                    results.append((full_path, file_size, mtime_sec))

                return results
            finally:
                dll.Everything_CleanUp()


_everything_sdk = EverythingSDK()


def _is_excluded_path(
    norm_path: str,
    internal_data_prefix: str,
    custom_root_prefix: Optional[str] = None,
) -> bool:
    """
    Return True if a normalized file path resides inside excluded system/dev/app folders.
    When `custom_root_prefix` is provided, only check exclusions on the path relative
    to the user-selected folder so explicitly chosen folders (e.g. inside Temp/AppData) work.
    """
    lowered = norm_path.lower()
    if lowered.startswith(internal_data_prefix):
        return True

    check_target = lowered
    if custom_root_prefix and lowered.startswith(custom_root_prefix):
        check_target = "\\" + lowered[len(custom_root_prefix) :]

    for marker in EXCLUDED_PATH_SUBSTRINGS:
        if marker in check_target:
            return True
    return False


def _get_windows_fallback_roots() -> List[Path]:
    """
    Determine root directories to walk when `custom_folder` is None and Everything IPC is offline.
    Walks non-system folders on the system drive plus all other local fixed/removable drives.
    """
    roots: List[Path] = []
    if os.name != "nt":
        return [Path.home()]

    system_drive = os.environ.get("SystemDrive", "C:").rstrip("\\/").upper() + "\\"

    # Enumerate available logical drives via Win32 API
    drives: List[str] = []
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        bitmask = int(kernel32.GetLogicalDrives())
        for i in range(26):
            if bitmask & (1 << i):
                drive_root = f"{chr(65 + i)}:\\"
                # DRIVE_REMOVABLE = 2, DRIVE_FIXED = 3
                drive_type = int(kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive_root)))
                if drive_type in (2, 3):
                    drives.append(drive_root)
    except Exception:
        drives = [system_drive]

    for drive in drives:
        if drive.upper() == system_drive:
            # Always prioritize user's home folder first, then non-system folders on C:\
            home_dir = Path.home()
            if home_dir.exists():
                roots.append(home_dir)
            try:
                for entry in os.scandir(drive):
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name_low = entry.name.lower()
                    if name_low in EXCLUDED_DIR_NAMES or name_low in {"windows", "users"}:
                        continue
                    roots.append(Path(entry.path))
            except OSError:
                pass
        else:
            roots.append(Path(drive))

    return roots


def _walk_roots_for_candidates(
    roots: List[Path],
    internal_data_prefix: str,
) -> List[Tuple[str, int, float]]:
    """
    Recursively walk `roots` using `os.walk` with in-place directory pruning.
    Wraps directory and file access in strict try/except blocks to skip locked,
    inaccessible, git, or corrupted directories immediately without hanging.
    Returns `(full_path, file_size_bytes, mtime_unix_seconds)` tuples for matching extensions.
    """
    discovered: List[Tuple[str, int, float]] = []

    for root_path in roots:
        try:
            if not root_path.exists():
                continue
        except (OSError, PermissionError):
            continue

        try:
            if root_path.is_file():
                if root_path.suffix.lower() in CANDIDATE_EXTENSIONS:
                    try:
                        st = root_path.stat()
                        discovered.append((str(root_path), int(st.st_size), float(st.st_mtime)))
                    except (OSError, PermissionError):
                        pass
                continue
        except (OSError, PermissionError):
            continue

        try:
            walk_gen = os.walk(str(root_path), topdown=True, onerror=lambda err: None)
            for current_root, dirs, files in walk_gen:
                try:
                    curr_norm_low = os.path.normpath(current_root).lower()
                except Exception:
                    dirs[:] = []
                    continue

                if curr_norm_low.startswith(internal_data_prefix.rstrip("\\")):
                    dirs[:] = []
                    continue

                # Prune excluded and git/temp subdirectories in-place before os.walk descends into them
                pruned_dirs: List[str] = []
                for d in dirs:
                    d_low = d.lower()
                    if d_low in EXCLUDED_DIR_NAMES or d_low.startswith(".git"):
                        continue
                    if curr_norm_low == "c:\\" and d_low == "windows":
                        continue
                    try:
                        full_sub_low = os.path.join(curr_norm_low, d_low) + "\\"
                        if full_sub_low.startswith(internal_data_prefix):
                            continue
                        pruned_dirs.append(d)
                    except Exception:
                        continue
                dirs[:] = pruned_dirs

                for fname in files:
                    try:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in CANDIDATE_EXTENSIONS:
                            continue
                        full_path = os.path.join(current_root, fname)
                        st = os.stat(full_path)
                        discovered.append((full_path, int(st.st_size), float(st.st_mtime)))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as walk_exc:
            logger.debug("Skipping inaccessible root directory %s: %s", root_path, walk_exc)
            continue

    return discovered


def find_all_candidate_files(
    custom_folder: Optional[Union[str, Path]] = None,
    db_manager: Optional[database.DatabaseManager] = None,
) -> List[str]:
    """
    Discover all new or modified candidate image/PDF files ready for visual OCR & Face inspection.

    - If `custom_folder` is provided: recursively walks `custom_folder`.
    - Otherwise: queries `./Everything64.dll` via IPC across all drives in milliseconds,
      and falls back cleanly to a pruned `os.walk` if Everything is not running.
    - Filters out tiny files (< 25 KB) and huge files (> 25 MB).
    - Cross-checks against `docuvault.db` (`file_path` + `mtime`) so already-scanned
      unchanged files are skipped immediately.

    Returns a list of normalized absolute file path strings.
    """
    active_db = db_manager or database.db
    internal_data_prefix = database.normalize_file_path(DATA_DIR).lower() + "\\"

    raw_entries: List[Tuple[str, int, float]] = []
    custom_root_prefix: Optional[str] = None

    if custom_folder is not None and str(custom_folder).strip():
        target = Path(os.path.abspath(os.path.expanduser(str(custom_folder).strip())))
        if not target.exists():
            logger.warning("Custom scan folder does not exist: %s", target)
            return []
        if target.is_dir():
            custom_root_prefix = database.normalize_file_path(target).lower().rstrip("\\") + "\\"
        else:
            custom_root_prefix = database.normalize_file_path(target.parent).lower().rstrip("\\") + "\\"
        logger.info("Scanning custom folder recursively via os.walk: %s", target)
        raw_entries = _walk_roots_for_candidates([target], internal_data_prefix)
    else:
        try:
            raw_entries = _everything_sdk.query_files(EVERYTHING_SEARCH_QUERY)
            logger.info(
                "Everything64.dll IPC returned %d raw candidate matches.", len(raw_entries)
            )
        except EverythingError as exc:
            logger.warning(
                "Everything64.dll unavailable (%s). Falling back to os.walk.", exc
            )
            fallback_roots = _get_windows_fallback_roots()
            raw_entries = _walk_roots_for_candidates(fallback_roots, internal_data_prefix)
            logger.info("os.walk fallback discovered %d raw matches.", len(raw_entries))

    # Bulk-load existing {lowercase_file_path: file_hash_or_mtime} map from SQLite once
    scanned_map: Dict[str, str] = active_db.get_scanned_files_map()
    ignored_paths: Set[str] = active_db.get_ignored_files_set()
    ghost_cache_map: Dict[str, Tuple[int, float]] = active_db.get_ghost_cache_map()

    candidates: List[str] = []
    seen_paths: Set[str] = set()

    for raw_path, raw_size, raw_mtime in raw_entries:
        try:
            norm_path = database.normalize_file_path(raw_path)
        except ValueError:
            continue

        path_key = norm_path.lower()
        if path_key in seen_paths or path_key in ignored_paths:
            continue

        ext = os.path.splitext(path_key)[1]
        if ext not in CANDIDATE_EXTENSIONS:
            continue

        if _is_excluded_path(norm_path, internal_data_prefix, custom_root_prefix=custom_root_prefix):
            continue

        file_size = raw_size
        mtime_sec = raw_mtime

        # Fallback to os.stat if Everything didn't have size or mtime indexed for this entry
        if file_size < 0 or mtime_sec <= 0.0:
            try:
                st = os.stat(norm_path)
                file_size = int(st.st_size)
                mtime_sec = float(st.st_mtime)
            except OSError:
                continue

        # Enforce size bounds: 25 KB <= size <= 25 MB
        if file_size < MIN_FILE_SIZE_BYTES or file_size > MAX_FILE_SIZE_BYTES:
            continue

        # Incremental Scanning: Check Ghost Cache (scan_history)
        # If size and mtime match an already inspected file, skip immediately
        if path_key in ghost_cache_map:
            cached_size, cached_mtime = ghost_cache_map[path_key]
            if cached_size == file_size and abs(cached_mtime - mtime_sec) < 0.001:
                continue

        # Cross-check against active SQLite files database (path + normalized mtime)
        norm_mtime = database.normalize_hash_or_mtime(mtime_sec)
        if scanned_map.get(path_key) == norm_mtime:
            continue

        # Verify file actually exists and is a regular file when obtained via Everything cache
        if not os.path.isfile(norm_path):
            continue

        seen_paths.add(path_key)
        candidates.append(norm_path)

    logger.info("Filtered down to %d new/modified candidate files to inspect.", len(candidates))
    return candidates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    found = find_all_candidate_files()
    print(f"Discovered {len(found)} new/modified candidate files.")
    for sample in found[:10]:
        print(f"  - {sample}")
