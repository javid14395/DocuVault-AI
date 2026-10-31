"""
DocuVault AI - SQLite Database Manager (docuvault.db)

Manages persistent metadata for visually inspected files (OCR + Face Detection),
document classifications, extracted ID numbers, thumbnails, and recognized people.
"""

from __future__ import annotations

import base64
from collections import Counter
import ctypes
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("docuvault.database")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "docuvault.db"

ENC_PREFIX = "ENC1:"
_ENC_AAD = b"docuvault-sqlite-v1"
_PIN_KDF_ITERATIONS = 200_000

NOT_A_DOCUMENT = "Not a Document"
PASSWORD_PROTECTED = "Password Protected"
UNREADABLE_OR_CORRUPT = "Unidentified / Unreadable"

# Core Indian document categories
CORE_DOCUMENT_CATEGORIES: Tuple[str, ...] = (
    "Aadhaar Card",
    "PAN Card",
    "Driving License",
    "Voter ID",
    "Passport",
    "Certificate/Marksheet",
    "Admit Card / Hall Ticket",
    "Bill/Receipt",
    "Face Gallery Photo",
)

DOCUMENT_CATEGORIES: Tuple[str, ...] = CORE_DOCUMENT_CATEGORIES + (
    PASSWORD_PROTECTED,
    UNREADABLE_OR_CORRUPT,
)

ALL_DOC_CATEGORIES: Tuple[str, ...] = DOCUMENT_CATEGORIES + (NOT_A_DOCUMENT,)


def normalize_file_path(file_path: Union[str, Path]) -> str:
    """Normalize a filesystem path consistently on Windows while preserving readable casing."""
    raw = str(file_path).strip()
    if not raw:
        raise ValueError("file_path cannot be empty.")
    normalized = os.path.normpath(raw)
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def normalize_hash_or_mtime(value: Union[str, int, float]) -> str:
    """Convert a file hash or modification timestamp into a deterministic string representation."""
    if value is None:
        raise ValueError("file_hash_or_mtime cannot be None.")
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value).strip()
    if not text:
        raise ValueError("file_hash_or_mtime cannot be empty.")
    return text


def serialize_embedding(embedding: Any) -> Optional[Union[str, bytes]]:
    """
    Serialize a face embedding vector for SQLite storage (JSON text or raw BLOB).
    Supports numpy arrays (via .tolist()), sequences of floats, JSON strings, or raw bytes.
    """
    if embedding is None:
        return None
    if isinstance(embedding, (bytes, bytearray, memoryview)):
        return bytes(embedding)
    if isinstance(embedding, str):
        parsed = json.loads(embedding)
        if not isinstance(parsed, list):
            raise ValueError("JSON face_embedding must encode a list of numbers.")
        return json.dumps([float(x) for x in parsed])
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if isinstance(embedding, (list, tuple)):
        return json.dumps([float(x) for x in embedding])
    raise TypeError(f"Unsupported face_embedding type: {type(embedding).__name__}")


def deserialize_embedding(raw: Any) -> Optional[List[float]]:
    """
    Deserialize a stored face embedding (from JSON string or float32 BLOB) into a list of floats.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(raw)
        if not raw_bytes:
            return None
        try:
            decoded = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if len(raw_bytes) % 4 != 0:
            raise ValueError(f"Binary face_embedding byte length ({len(raw_bytes)}) is not a multiple of 4.")
        count = len(raw_bytes) // 4
        return list(struct.unpack(f"<{count}f", raw_bytes))
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Stored JSON face_embedding is not a list.")
        return [float(x) for x in parsed]
    raise TypeError(f"Cannot deserialize face_embedding of type: {type(raw).__name__}")


def _escape_like(term: str) -> str:
    """Escape SQLite LIKE wildcards so literal %, _, and \\ characters are matched accurately."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def dhash_hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute the 64-bit Hamming distance between two 16-char hexadecimal dHash strings."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


# -----------------------------------------------------------------------------
# Encrypted SQLite Wrapper (AES-256-GCM + Master PIN PBKDF2 + Win32 DPAPI Seal)
# -----------------------------------------------------------------------------


def _dpapi_protect(raw_bytes: bytes) -> Optional[bytes]:
    """Seal key bytes with Windows OS DPAPI (`CryptProtectData`, current-user scope)."""
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return None
    try:
        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            wintypes.LPCWSTR,
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB),
        ]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p

        buf = ctypes.create_string_buffer(raw_bytes, len(raw_bytes))
        in_blob = DATA_BLOB(len(raw_bytes), buf)
        out_blob = DATA_BLOB()
        # CRYPTPROTECT_UI_FORBIDDEN = 0x1
        if not crypt32.CryptProtectData(
            ctypes.byref(in_blob), "DocuVaultMasterDEK", None, None, None, 0x1, ctypes.byref(out_blob)
        ):
            return None
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))
    except Exception:
        return None


def _dpapi_unprotect(sealed_bytes: bytes) -> Optional[bytes]:
    """Unseal key bytes previously protected with Windows OS DPAPI (`CryptUnprotectData`)."""
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return None
    try:
        class DATA_BLOB(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_char)),
            ]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.POINTER(DATA_BLOB),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DATA_BLOB),
        ]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p

        buf = ctypes.create_string_buffer(sealed_bytes, len(sealed_bytes))
        in_blob = DATA_BLOB(len(sealed_bytes), buf)
        out_blob = DATA_BLOB()
        if not crypt32.CryptUnprotectData(
            ctypes.byref(in_blob), None, None, None, None, 0x1, ctypes.byref(out_blob)
        ):
            return None
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))
    except Exception:
        return None


class SQLiteEncryptionWrapper:
    """
    Enterprise AES-256-GCM encryption wrapper for Python `sqlite3`.

    - Encrypts sensitive columns (`ocr_text`, `extracted_id_number`, `dob`, `expiry_date`,
      `face_embedding`) so opening `docuvault.db` in any external DB browser reveals only
      authenticated ciphertext (`ENC1:<urlsafe_base64>`).
    - Wraps the 256-bit Data Encryption Key (DEK) with the user's Master PIN / Master Password
      via PBKDF2-HMAC-SHA256 (200,000 iterations), while maintaining a local OS-user DPAPI
      keystore envelope outside `docuvault.db` for seamless background worker access.
    """

    def __init__(self, keystore_path: Path, master_pin: Optional[str] = None) -> None:
        self.keystore_path = keystore_path
        self._lock = threading.RLock()
        self._dek: bytes = self._load_or_create_dek(master_pin=master_pin)
        self._aesgcm: AESGCM = AESGCM(self._dek)

    @staticmethod
    def _derive_kek(secret: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PIN_KDF_ITERATIONS,
        )
        return kdf.derive(secret.strip().encode("utf-8"))

    def _seal_dek_to_keystore(self, dek: bytes) -> None:
        try:
            self.keystore_path.parent.mkdir(parents=True, exist_ok=True)
            sealed = _dpapi_protect(dek)
            if sealed is not None:
                payload = b"DPAPI1:" + sealed
            else:
                # Non-Windows fallback: local machine-scoped key file outside .db
                payload = b"RAW1:" + base64.urlsafe_b64encode(dek)
            self.keystore_path.write_bytes(payload)
        except Exception as exc:
            logger.debug("Keystore write notice (%s): %s", self.keystore_path, exc)

    def _load_or_create_dek(self, master_pin: Optional[str] = None) -> bytes:
        if self.keystore_path.is_file():
            try:
                blob = self.keystore_path.read_bytes()
                if blob.startswith(b"DPAPI1:"):
                    unsealed = _dpapi_unprotect(blob[7:])
                    if unsealed and len(unsealed) == 32:
                        return unsealed
                elif blob.startswith(b"RAW1:"):
                    raw = base64.urlsafe_b64decode(blob[5:])
                    if len(raw) == 32:
                        return raw
            except Exception as exc:
                logger.warning("Could not read local vault keystore: %s", exc)

        if master_pin and master_pin.strip():
            # Deterministic derivation when explicitly initialized with a Master PIN on a clean keystore
            det_salt = hashlib.sha256(f"docuvault-pin-seed:{self.keystore_path.name}".encode("utf-8")).digest()[:16]
            dek = self._derive_kek(master_pin, det_salt)
        else:
            dek = secrets.token_bytes(32)

        self._seal_dek_to_keystore(dek)
        return dek

    def wrap_dek_with_pin(self, pin_or_password: str) -> str:
        """
        Wrap the active 256-bit DEK using a KEK derived from the user's Master PIN / Password.
        Returns `<salt_b64>$<nonce_and_ciphertext_b64>` suitable for storing in `settings`.
        """
        with self._lock:
            salt = secrets.token_bytes(16)
            kek = self._derive_kek(pin_or_password, salt)
            nonce = secrets.token_bytes(12)
            wrapped = AESGCM(kek).encrypt(nonce, self._dek, _ENC_AAD)
            salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
            blob_b64 = base64.urlsafe_b64encode(nonce + wrapped).decode("ascii")
            return f"{salt_b64}${blob_b64}"

    def unlock_dek_with_pin(self, pin_or_password: str, wrapped_token: str) -> bool:
        """
        Unwrap and activate the 256-bit DEK using the user's Master PIN / Password.
        Returns True if the PIN successfully decrypted the DEK.
        """
        if not wrapped_token or "$" not in wrapped_token:
            return False
        try:
            salt_b64, blob_b64 = wrapped_token.split("$", 1)
            salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
            blob = base64.urlsafe_b64decode(blob_b64.encode("ascii"))
            if len(blob) < 12 + 32 + 16:
                return False
            nonce, ct = blob[:12], blob[12:]
            kek = self._derive_kek(pin_or_password, salt)
            dek = AESGCM(kek).decrypt(nonce, ct, _ENC_AAD)
            if len(dek) != 32:
                return False
            with self._lock:
                self._dek = dek
                self._aesgcm = AESGCM(dek)
                self._seal_dek_to_keystore(dek)
            return True
        except Exception:
            return False

    @staticmethod
    def is_encrypted_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.startswith(ENC_PREFIX)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value[:5]) == ENC_PREFIX.encode("ascii")
        return False

    def encrypt_bytes_to_token(self, plain_bytes: bytes) -> str:
        nonce = os.urandom(12)
        with self._lock:
            ct = self._aesgcm.encrypt(nonce, plain_bytes, _ENC_AAD)
        return ENC_PREFIX + base64.urlsafe_b64encode(nonce + ct).decode("ascii")

    def decrypt_token_to_bytes(self, token: Union[str, bytes]) -> bytes:
        if isinstance(token, (bytes, bytearray, memoryview)):
            text = bytes(token).decode("ascii")
        else:
            text = token
        if not text.startswith(ENC_PREFIX):
            raise ValueError("Value does not carry ENC1: encryption header.")
        raw = base64.urlsafe_b64decode(text[len(ENC_PREFIX):].encode("ascii"))
        nonce, ct = raw[:12], raw[12:]
        with self._lock:
            return self._aesgcm.decrypt(nonce, ct, _ENC_AAD)

    def encrypt_text(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        if text == "":
            return ""
        if text.startswith(ENC_PREFIX):
            return text
        return self.encrypt_bytes_to_token(text.encode("utf-8"))

    def decrypt_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw_b = bytes(value)
            if raw_b.startswith(ENC_PREFIX.encode("ascii")):
                try:
                    return self.decrypt_token_to_bytes(raw_b).decode("utf-8")
                except Exception:
                    return None
            return raw_b.decode("utf-8", errors="replace")
        text = str(value)
        if not text.startswith(ENC_PREFIX):
            return text
        try:
            return self.decrypt_token_to_bytes(text).decode("utf-8")
        except Exception:
            return None

    def decrypt_sql_text(self, value: Any) -> str:
        """Deterministic SQLite UDF callback (`dv_decrypt(col)`) for encrypted LIKE/date queries."""
        res = self.decrypt_text(value)
        return res if res is not None else ""

    def encrypt_embedding(self, embedding: Any) -> Optional[str]:
        """Serialize a 512-d face embedding into compact float32 bytes and encrypt with AES-256-GCM."""
        if embedding is None:
            return None
        if self.is_encrypted_value(embedding):
            if isinstance(embedding, (bytes, bytearray, memoryview)):
                return bytes(embedding).decode("ascii")
            return str(embedding)
        floats = deserialize_embedding(embedding)
        if not floats:
            return None
        packed = struct.pack(f"<{len(floats)}f", *[float(v) for v in floats])
        return self.encrypt_bytes_to_token(packed)

    def decrypt_embedding(self, raw: Any) -> Optional[List[float]]:
        """Decrypt an AES-256-GCM `ENC1:` face embedding (or fallback to legacy plaintext/BLOB)."""
        if raw is None:
            return None
        if self.is_encrypted_value(raw):
            try:
                plain_bytes = self.decrypt_token_to_bytes(raw)
                return deserialize_embedding(plain_bytes)
            except Exception:
                return None
        return deserialize_embedding(raw)


class DatabaseManager:
    """
    Thread-safe SQLite database manager for DocuVault AI (`docuvault.db`).

    Uses WAL journal mode, per-thread connections, and `SQLiteEncryptionWrapper`
    (AES-256-GCM + Master PIN PBKDF2 + Win32 DPAPI keystore) so sensitive columns
    cannot be read as plain text on disk.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        auto_init: bool = False,
        master_pin: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connections: List[sqlite3.Connection] = []
        self._thread_connections: Dict[int, sqlite3.Connection] = {}
        self._generation: int = 0
        self._mutation_seq: int = 0
        self._savepoint_seq: int = 0
        self._dup_count_cache: Tuple[int, int, int] = (-1, -1, 0)
        self._dup_groups_cache: Optional[Tuple[int, int, List[Dict[str, Any]]]] = None
        self._initialized: bool = False
        self._initial_master_pin: Optional[str] = (
            master_pin.strip() if (master_pin and str(master_pin).strip()) else None
        )
        if self.db_path == DEFAULT_DB_PATH.resolve():
            keystore_path = BASE_DIR / "data" / ".vault_dek.dpapi"
        else:
            keystore_path = self.db_path.parent / f".{self.db_path.stem}_dek.dpapi"
        self._crypto = SQLiteEncryptionWrapper(
            keystore_path=keystore_path,
            master_pin=self._initial_master_pin,
        )
        if auto_init:
            self.init_db()

    def _prune_dead_connections_unlocked(self, exclude_tid: Optional[int] = None) -> int:
        """Close and release SQLite connections belonging to terminated worker threads. Caller must hold `self._lock`."""
        alive_tids = {t.ident for t in threading.enumerate() if t.ident is not None}
        if exclude_tid is not None:
            alive_tids.add(exclude_tid)
        dead_tids = [tid for tid in list(self._thread_connections.keys()) if tid not in alive_tids]
        pruned = 0
        for dead_tid in dead_tids:
            dead_conn = self._thread_connections.pop(dead_tid, None)
            if dead_conn is not None:
                try:
                    self._connections.remove(dead_conn)
                except ValueError:
                    pass
                try:
                    dead_conn.close()
                except Exception:
                    pass
                pruned += 1
        return pruned

    def prune_dead_connections(self) -> int:
        """Public thread-safe cleanup for SQLite connections left behind by finished ThreadPoolExecutor threads."""
        with self._lock:
            return self._prune_dead_connections_unlocked()

    def checkpoint_wal(self, mode: str = "PASSIVE") -> None:
        """Run a non-blocking WAL checkpoint (`PASSIVE` or `TRUNCATE`) to prevent unbounded `.db-wal` growth."""
        safe_mode = mode.upper() if mode.upper() in ("PASSIVE", "FULL", "RESTART", "TRUNCATE") else "PASSIVE"
        with self._lock:
            if not self._initialized:
                return
            try:
                conn = self._ensure_thread_connection()
                if not conn.in_transaction:
                    conn.execute(f"PRAGMA wal_checkpoint({safe_mode});")
            except Exception as exc:
                logger.debug("WAL checkpoint (%s) skipped: %s", safe_mode, exc)

    def _ensure_thread_connection(self) -> sqlite3.Connection:
        """Return or create the current thread's autocommit SQLite connection without triggering schema init."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        gen: int = getattr(self._local, "generation", -1)

        if conn is None or gen != self._generation:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.create_function(
                "dv_decrypt",
                1,
                self._crypto.decrypt_sql_text,
                deterministic=True,
            )

            tid = threading.get_ident()
            with self._lock:
                # If a dead thread previously held this recycled OS thread ident, close its stale handle
                stale_conn = self._thread_connections.pop(tid, None)
                if stale_conn is not None and stale_conn is not conn:
                    try:
                        self._connections.remove(stale_conn)
                    except ValueError:
                        pass
                    try:
                        stale_conn.close()
                    except Exception:
                        pass
                if len(self._thread_connections) >= 8:
                    self._prune_dead_connections_unlocked(exclude_tid=tid)
                self._thread_connections[tid] = conn
                self._connections.append(conn)
                self._local.conn = conn
                self._local.generation = self._generation

        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection and guarantee the schema is initialized."""
        conn = self._ensure_thread_connection()
        if not self._initialized:
            self.init_db()
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Execute write queries atomically under a re-entrant thread lock using `BEGIN IMMEDIATE`
        (preventing SQLite DEFERRED lock-upgrade deadlocks) with transient busy/locked retry backoff.
        """
        with self._lock:
            conn = self._get_connection()
            if conn.in_transaction:
                self._savepoint_seq += 1
                sp_name = f"dv_sp_{self._savepoint_seq}"
                conn.execute(f"SAVEPOINT {sp_name};")
                try:
                    yield conn
                    conn.execute(f"RELEASE SAVEPOINT {sp_name};")
                    self._mutation_seq += 1
                except Exception:
                    try:
                        conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name};")
                        conn.execute(f"RELEASE SAVEPOINT {sp_name};")
                    except Exception:
                        pass
                    raise
                return

            last_err: Optional[sqlite3.OperationalError] = None
            for attempt in range(6):
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    last_err = None
                    break
                except sqlite3.OperationalError as op_err:
                    msg = str(op_err).lower()
                    if ("locked" in msg or "busy" in msg) and attempt < 5:
                        last_err = op_err
                        time.sleep(0.02 * (2 ** attempt))
                        continue
                    raise
            if last_err is not None:
                raise last_err

            try:
                yield conn
                if conn.in_transaction:
                    conn.execute("COMMIT;")
                self._mutation_seq += 1
            except Exception:
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                raise

    def init_db(self) -> None:
        """Create the `people` and `files` tables, migrate columns/encryption, and ensure indexes exist."""
        with self._lock:
            if self._initialized:
                return
            conn = self._ensure_thread_connection()
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL DEFAULT 'Person',
                    face_thumbnail_path TEXT,
                    face_embedding BLOB
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    file_hash_or_mtime TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    is_document INTEGER NOT NULL DEFAULT 0,
                    doc_category TEXT NOT NULL DEFAULT 'Not a Document',
                    extracted_id_number TEXT,
                    dob TEXT,
                    expiry_date TEXT,
                    ocr_text TEXT,
                    person_id INTEGER,
                    thumbnail_path TEXT,
                    dhash TEXT,
                    date_taken TEXT,
                    scanned_at TEXT NOT NULL,
                    thumbnail_b64 TEXT,
                    face_embedding BLOB,
                    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ignored_files (
                    file_path TEXT PRIMARY KEY COLLATE NOCASE,
                    ignored_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scan_history (
                    file_path TEXT PRIMARY KEY COLLATE NOCASE,
                    file_size INTEGER NOT NULL,
                    mtime_sec REAL NOT NULL,
                    processed_at TEXT NOT NULL
                );
                """
            )
            # Seamless backward-compatible schema migration for existing docuvault.db files
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(files);").fetchall()
            }
            if "dhash" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN dhash TEXT;")
            if "dob" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN dob TEXT;")
            if "expiry_date" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN expiry_date TEXT;")
            if "thumbnail_b64" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN thumbnail_b64 TEXT;")
            if "face_embedding" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN face_embedding BLOB;")
            if "date_taken" not in existing_cols:
                conn.execute("ALTER TABLE files ADD COLUMN date_taken TEXT;")

            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_files_is_document ON files(is_document);
                CREATE INDEX IF NOT EXISTS idx_files_doc_category ON files(doc_category);
                CREATE INDEX IF NOT EXISTS idx_files_person_id ON files(person_id);
                CREATE INDEX IF NOT EXISTS idx_files_scanned_at ON files(scanned_at DESC);
                CREATE INDEX IF NOT EXISTS idx_files_date_taken ON files(date_taken DESC);
                CREATE INDEX IF NOT EXISTS idx_files_dhash ON files(dhash);
                CREATE INDEX IF NOT EXISTS idx_files_expiry ON files(expiry_date);
                CREATE INDEX IF NOT EXISTS idx_scan_history_mtime ON scan_history(mtime_sec);
                """
            )

            # Unlock wrapped DEK if an explicit Master PIN/Password was supplied on startup
            if self._initial_master_pin:
                wrap_row = conn.execute(
                    "SELECT value FROM settings WHERE key = 'vault_pin_wrapped_dek';"
                ).fetchone()
                if wrap_row and wrap_row["value"]:
                    self._crypto.unlock_dek_with_pin(self._initial_master_pin, str(wrap_row["value"]))

            # Migrate any legacy plaintext rows to AES-256-GCM ciphertext and purge SQLite freelist
            migrated_rows = 0
            try:
                conn.execute("BEGIN IMMEDIATE;")
                migrated_rows = self._migrate_plaintext_records(conn)
                self._backfill_missing_date_taken(conn)
                if conn.in_transaction:
                    conn.execute("COMMIT;")
                self._mutation_seq += 1
            except Exception as mig_exc:
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                logger.warning("Startup migration/backfill notice: %s", mig_exc)

            if migrated_rows > 0:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.execute("VACUUM;")
                except Exception as vac_exc:
                    logger.debug("Post-migration VACUUM skipped: %s", vac_exc)

            self._initialized = True

            # Clean up existing false-positive Aadhaar entries on startup
            try:
                self.cleanup_false_positive_aadhaar()
            except Exception:
                pass

            # Clean up existing false-positive Certificate/Marksheet entries on startup
            try:
                self.cleanup_false_positive_certificates()
            except Exception:
                pass

    def _migrate_plaintext_records(self, conn: sqlite3.Connection) -> int:
        """
        Encrypt any existing plaintext `extracted_id_number`, `dob`, `expiry_date`,
        `ocr_text`, or `face_embedding` columns in `files` and `people` in-place.
        """
        file_rows = conn.execute(
            """
            SELECT id, extracted_id_number, dob, expiry_date, ocr_text, face_embedding
            FROM files;
            """
        ).fetchall()

        file_updates: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]] = []
        for row in file_rows:
            needs_enc = False
            raw_id = row["extracted_id_number"]
            raw_dob = row["dob"]
            raw_exp = row["expiry_date"]
            raw_ocr = row["ocr_text"]
            raw_emb = row["face_embedding"]

            enc_id = raw_id
            if raw_id is not None and str(raw_id) != "" and not self._crypto.is_encrypted_value(raw_id):
                enc_id = self._crypto.encrypt_text(str(raw_id))
                needs_enc = True

            enc_dob = raw_dob
            if raw_dob is not None and str(raw_dob) != "" and not self._crypto.is_encrypted_value(raw_dob):
                enc_dob = self._crypto.encrypt_text(str(raw_dob))
                needs_enc = True

            enc_exp = raw_exp
            if raw_exp is not None and str(raw_exp) != "" and not self._crypto.is_encrypted_value(raw_exp):
                enc_exp = self._crypto.encrypt_text(str(raw_exp))
                needs_enc = True

            enc_ocr = raw_ocr
            if raw_ocr is not None and str(raw_ocr) != "" and not self._crypto.is_encrypted_value(raw_ocr):
                enc_ocr = self._crypto.encrypt_text(str(raw_ocr))
                needs_enc = True

            enc_emb: Any = raw_emb
            if raw_emb is not None and not self._crypto.is_encrypted_value(raw_emb):
                enc_emb = self._crypto.encrypt_embedding(raw_emb)
                needs_enc = True

            if needs_enc:
                file_updates.append((enc_id, enc_dob, enc_exp, enc_ocr, enc_emb, int(row["id"])))

        if file_updates:
            conn.executemany(
                """
                UPDATE files
                SET extracted_id_number = ?,
                    dob = ?,
                    expiry_date = ?,
                    ocr_text = ?,
                    face_embedding = ?
                WHERE id = ?;
                """,
                file_updates,
            )

        people_rows = conn.execute(
            "SELECT id, face_embedding FROM people WHERE face_embedding IS NOT NULL;"
        ).fetchall()
        people_updates: List[Tuple[Optional[str], int]] = []
        for row in people_rows:
            raw_emb = row["face_embedding"]
            if raw_emb is not None and not self._crypto.is_encrypted_value(raw_emb):
                people_updates.append((self._crypto.encrypt_embedding(raw_emb), int(row["id"])))

        if people_updates:
            conn.executemany(
                "UPDATE people SET face_embedding = ? WHERE id = ?;",
                people_updates,
            )

        return len(file_updates) + len(people_updates)

    def _backfill_missing_date_taken(self, conn: sqlite3.Connection) -> int:
        """Populate `date_taken` on legacy rows where `date_taken IS NULL`."""
        rows = conn.execute(
            """
            SELECT id, file_path, file_hash_or_mtime, is_document, scanned_at
            FROM files
            WHERE date_taken IS NULL OR TRIM(date_taken) = '';
            """
        ).fetchall()
        if not rows:
            return 0

        try:
            from scanner import extract_exif_date_taken
        except Exception:
            extract_exif_date_taken = None  # type: ignore[assignment]

        updates: List[Tuple[str, int]] = []
        for row in rows:
            dt_str: Optional[str] = None
            fpath = str(row["file_path"] or "")
            mtime_tok = str(row["file_hash_or_mtime"] or "")
            mtime_sec: Optional[float] = None
            if mtime_tok.startswith("mtime:"):
                parts = mtime_tok.split(":")
                if len(parts) >= 2:
                    try:
                        mtime_sec = int(parts[1]) / 1_000_000_000.0
                    except ValueError:
                        mtime_sec = None
            else:
                try:
                    mtime_sec = float(mtime_tok)
                except ValueError:
                    mtime_sec = None

            if int(row["is_document"] or 0) == 1 and extract_exif_date_taken is not None and os.path.isfile(fpath):
                dt_str = extract_exif_date_taken(fpath, fallback_mtime=mtime_sec, use_mtime_fallback=True)

            if not dt_str and mtime_sec and mtime_sec > 0:
                try:
                    dt_str = datetime.fromtimestamp(mtime_sec).strftime("%Y-%m-%d %H:%M:%S")
                except (OSError, OverflowError, ValueError):
                    dt_str = None

            if not dt_str:
                raw_scan = str(row["scanned_at"] or "")
                dt_str = raw_scan[:19].replace("T", " ") if len(raw_scan) >= 19 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            updates.append((dt_str, int(row["id"])))

        if updates:
            conn.executemany("UPDATE files SET date_taken = ? WHERE id = ?;", updates)
        return len(updates)

    def cleanup_false_positive_certificates(self) -> int:
        """
        Scan and reclassify any false-positive Certificate/Marksheet records from docuvault.db.
        - Documents containing admit card/hall ticket keywords are moved to 'Admit Card / Hall Ticket'.
        - Documents containing textbook/chapter/syllabus negative keywords are moved to 'Not a Document'.
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT id, file_path, ocr_text
            FROM files
            WHERE doc_category = 'Certificate/Marksheet';
            """
        ).fetchall()

        to_admit_card: List[int] = []
        to_not_document: List[int] = []

        admit_keywords = ("hall ticket", "call letter", "confirmation number", "admit card")
        negative_keywords = (
            "chapter",
            "lesson",
            "syllabus",
            "hall ticket",
            "call letter",
            "admit card",
            "mathematics",
            "maths",
            "chap1",
            "chap2",
            "chap3",
            "chap4",
        )
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

        for row in rows:
            rec_id = row["id"]
            file_name = os.path.basename(str(row["file_path"] or "")).lower()
            decrypted_ocr = self._crypto.decrypt_text(row["ocr_text"]) or ""
            ocr = (decrypted_ocr + " " + file_name).lower()

            if any(kw in ocr for kw in admit_keywords):
                to_admit_card.append(rec_id)
            elif any(kw in ocr for kw in negative_keywords) or not any(a in ocr for a in cert_anchors):
                to_not_document.append(rec_id)

        count = 0
        with self.transaction() as write_conn:
            if to_admit_card:
                placeholders = ",".join("?" for _ in to_admit_card)
                write_conn.execute(
                    f"UPDATE files SET doc_category = 'Admit Card / Hall Ticket', is_document = 1 WHERE id IN ({placeholders});",
                    to_admit_card,
                )
                count += len(to_admit_card)

            if to_not_document:
                placeholders = ",".join("?" for _ in to_not_document)
                write_conn.execute(
                    f"UPDATE files SET doc_category = '{NOT_A_DOCUMENT}', is_document = 0, extracted_id_number = NULL, person_id = NULL WHERE id IN ({placeholders});",
                    to_not_document,
                )
                count += len(to_not_document)

        return count

    def cleanup_false_positive_aadhaar(self) -> int:
        """
        Scan and purge/reclassify any false-positive Aadhaar records from docuvault.db.
        Specifically targets:
        1. 12-digit numbers starting with '91' followed by [6-9] (mobile numbers with country code)
        2. Numbers starting with '0' or '1' (invalid Aadhaar)
        3. Records classified as Aadhaar Card without mandatory anchor keywords in OCR text
        """
        aadhaar_anchors = (
            "government of india",
            "govt of india",
            "govt. of india",
            "unique identification",
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
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT id, file_path, extracted_id_number, ocr_text
            FROM files
            WHERE doc_category = 'Aadhaar Card';
            """
        ).fetchall()

        reclassified_ids: List[int] = []
        for row in rows:
            rec_id = row["id"]
            raw_id = (self._crypto.decrypt_text(row["extracted_id_number"]) or "").strip()
            ocr = (self._crypto.decrypt_text(row["ocr_text"]) or "").lower()
            clean_digits = "".join(c for c in raw_id if c.isdigit())

            is_invalid_number = False
            if len(clean_digits) == 12:
                if clean_digits.startswith(("0", "1")):
                    is_invalid_number = True
                elif clean_digits.startswith("91") and clean_digits[2] in "6789":
                    is_invalid_number = True

            has_anchor = any(anchor in ocr for anchor in aadhaar_anchors)

            # If number is invalid OR it lacks any Aadhaar anchor keywords in OCR text
            if is_invalid_number or not has_anchor:
                reclassified_ids.append(rec_id)

        if reclassified_ids:
            with self.transaction() as write_conn:
                for chunk_start in range(0, len(reclassified_ids), 400):
                    chunk = reclassified_ids[chunk_start : chunk_start + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    write_conn.execute(
                        f"""
                        UPDATE files
                        SET is_document = 0,
                            doc_category = '{NOT_A_DOCUMENT}',
                            extracted_id_number = NULL,
                            person_id = NULL
                        WHERE id IN ({placeholders});
                        """,
                        chunk,
                    )
        return len(reclassified_ids)

    def close(self) -> None:
        """Close all open thread-local SQLite connections."""
        with self._lock:
            self._generation += 1
            self._mutation_seq += 1
            self._dup_count_cache = (-1, -1, 0)
            self._dup_groups_cache = None
            seen_ids: Set[int] = set()
            for conn in list(self._connections) + list(self._thread_connections.values()):
                if id(conn) in seen_ids:
                    continue
                seen_ids.add(id(conn))
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            self._thread_connections.clear()
            self._local.conn = None
            self._initialized = False

    # -------------------------------------------------------------------------
    # Row Formatters (Transparent AES-256-GCM Decryption)
    # -------------------------------------------------------------------------

    def _row_to_file_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["is_document"] = bool(data.get("is_document", 0))
        for sensitive_col in ("extracted_id_number", "dob", "expiry_date", "ocr_text"):
            if sensitive_col in data and data[sensitive_col] is not None:
                data[sensitive_col] = self._crypto.decrypt_text(data[sensitive_col])
        if "face_embedding" in data:
            try:
                data["face_embedding"] = self._crypto.decrypt_embedding(data.get("face_embedding"))
            except Exception:
                data["face_embedding"] = None
        return data

    def _row_to_person_dict(self, row: sqlite3.Row, include_embedding: bool = True) -> Dict[str, Any]:
        data = dict(row)
        if include_embedding:
            data["face_embedding"] = self._crypto.decrypt_embedding(data.get("face_embedding"))
        else:
            data.pop("face_embedding", None)
        return data

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    def upsert_file(
        self,
        file_path: Union[str, Path],
        file_hash_or_mtime: Union[str, int, float],
        file_size: int = 0,
        is_document: Optional[bool] = None,
        doc_category: Optional[str] = None,
        extracted_id_number: Optional[str] = None,
        dob: Optional[str] = None,
        expiry_date: Optional[str] = None,
        ocr_text: Optional[str] = None,
        person_id: Optional[int] = None,
        thumbnail_path: Optional[str] = None,
        dhash: Optional[str] = None,
        scanned_at: Optional[str] = None,
        thumbnail_b64: Optional[str] = None,
        face_embedding: Any = None,
        date_taken: Optional[str] = None,
    ) -> int:
        """
        Insert a newly scanned file or update its record if `file_path` already exists.
        Sensitive fields (`extracted_id_number`, `dob`, `expiry_date`, `ocr_text`, `face_embedding`)
        are transparently encrypted with AES-256-GCM before reaching SQLite disk storage.
        Returns the `files.id` primary key.
        """
        norm_path = normalize_file_path(file_path)
        norm_mtime = normalize_hash_or_mtime(file_hash_or_mtime)
        norm_size = max(0, int(file_size or 0))

        # Resolve is_document and doc_category consistently
        if doc_category is None:
            resolved_category = NOT_A_DOCUMENT if not is_document else "Bill/Receipt"
        else:
            resolved_category = doc_category.strip()

        if resolved_category not in ALL_DOC_CATEGORIES:
            raise ValueError(
                f"Invalid doc_category '{resolved_category}'. Expected one of: {ALL_DOC_CATEGORIES}"
            )

        if is_document is None:
            resolved_is_doc = resolved_category not in (NOT_A_DOCUMENT, UNREADABLE_OR_CORRUPT)
        else:
            resolved_is_doc = bool(is_document)
            if not resolved_is_doc and resolved_category != UNREADABLE_OR_CORRUPT:
                resolved_category = NOT_A_DOCUMENT
            elif resolved_category == NOT_A_DOCUMENT:
                resolved_is_doc = False

        clean_id_number = extracted_id_number.strip() if extracted_id_number else None
        clean_dob = dob.strip() if dob else None
        clean_expiry = expiry_date.strip() if expiry_date else None
        clean_ocr_text = ocr_text.strip() if ocr_text else None
        clean_thumb = str(thumbnail_path).strip() if thumbnail_path else None
        clean_dhash = str(dhash).strip().lower() if (dhash and str(dhash).strip()) else None
        clean_b64 = str(thumbnail_b64).strip() if thumbnail_b64 else None
        clean_date_taken = str(date_taken).strip() if (date_taken and str(date_taken).strip()) else None
        if clean_date_taken is None and resolved_is_doc and os.path.isfile(norm_path):
            try:
                from scanner import extract_exif_date_taken
                clean_date_taken = extract_exif_date_taken(norm_path, use_mtime_fallback=True)
            except Exception:
                clean_date_taken = None

        enc_id_number = self._crypto.encrypt_text(clean_id_number) if clean_id_number is not None else None
        enc_dob = self._crypto.encrypt_text(clean_dob) if clean_dob is not None else None
        enc_expiry = self._crypto.encrypt_text(clean_expiry) if clean_expiry is not None else None
        enc_ocr_text = self._crypto.encrypt_text(clean_ocr_text) if clean_ocr_text is not None else None
        enc_emb = self._crypto.encrypt_embedding(face_embedding) if face_embedding is not None else None
        timestamp = scanned_at or datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO files (
                    file_path,
                    file_hash_or_mtime,
                    file_size,
                    is_document,
                    doc_category,
                    extracted_id_number,
                    dob,
                    expiry_date,
                    ocr_text,
                    person_id,
                    thumbnail_path,
                    dhash,
                    date_taken,
                    scanned_at,
                    thumbnail_b64,
                    face_embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash_or_mtime = excluded.file_hash_or_mtime,
                    file_size = excluded.file_size,
                    is_document = excluded.is_document,
                    doc_category = excluded.doc_category,
                    extracted_id_number = excluded.extracted_id_number,
                    dob = excluded.dob,
                    expiry_date = excluded.expiry_date,
                    ocr_text = excluded.ocr_text,
                    person_id = excluded.person_id,
                    thumbnail_path = excluded.thumbnail_path,
                    dhash = excluded.dhash,
                    date_taken = COALESCE(excluded.date_taken, files.date_taken),
                    scanned_at = excluded.scanned_at,
                    thumbnail_b64 = COALESCE(excluded.thumbnail_b64, files.thumbnail_b64),
                    face_embedding = COALESCE(excluded.face_embedding, files.face_embedding)
                RETURNING id;
                """,
                (
                    norm_path,
                    norm_mtime,
                    norm_size,
                    int(resolved_is_doc),
                    resolved_category,
                    enc_id_number,
                    enc_dob,
                    enc_expiry,
                    enc_ocr_text,
                    person_id,
                    clean_thumb,
                    clean_dhash,
                    clean_date_taken,
                    timestamp,
                    clean_b64,
                    enc_emb,
                ),
            )
            row = cursor.fetchone()
            cursor.close()
            if row is not None:
                return int(row["id"])

            fallback = conn.execute(
                "SELECT id FROM files WHERE file_path = ?;",
                (norm_path,),
            ).fetchone()
            return int(fallback["id"])

    # Alias for readability across modules
    insert_or_update_file = upsert_file

    # -------------------------------------------------------------------------
    # Ghost Cache (Incremental Scanning Engine)
    # -------------------------------------------------------------------------

    def record_scan_history(
        self,
        file_path: Union[str, Path],
        file_size: int,
        mtime_sec: float,
    ) -> None:
        """
        Record processed file metadata in the scan_history table (Ghost Cache).
        Allows skipping files in subsequent scans even if cleared from the UI.
        """
        norm_path = normalize_file_path(file_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO scan_history (file_path, file_size, mtime_sec, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_size = excluded.file_size,
                    mtime_sec = excluded.mtime_sec,
                    processed_at = excluded.processed_at;
                """,
                (norm_path, int(file_size), float(mtime_sec), now_iso),
            )

    def is_in_ghost_cache(
        self,
        file_path: Union[str, Path],
        file_size: int,
        mtime_sec: float,
        tolerance_sec: float = 0.001,
    ) -> bool:
        """
        Check if a file's size and modified time exactly match the Ghost Cache.
        """
        norm_path = normalize_file_path(file_path)
        conn = self._get_connection()
        row = conn.execute(
            "SELECT file_size, mtime_sec FROM scan_history WHERE file_path = ?;",
            (norm_path,),
        ).fetchone()
        if not row:
            return False
        cached_size = int(row["file_size"])
        cached_mtime = float(row["mtime_sec"])
        return cached_size == int(file_size) and abs(cached_mtime - float(mtime_sec)) < tolerance_sec

    def get_ghost_cache_map(self) -> Dict[str, Tuple[int, float]]:
        """
        Return a high-speed memory lookup of `{lowercase_file_path: (file_size, mtime_sec)}`
        for instant pre-filtering in scanner.py.
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT file_path, file_size, mtime_sec FROM scan_history;").fetchall()
        return {r["file_path"].lower(): (int(r["file_size"]), float(r["mtime_sec"])) for r in rows}

    def clear_scan_history(self) -> int:
        """Clear all Ghost Cache records."""
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM scan_history;")
            return cur.rowcount

    def is_file_scanned(
        self,
        file_path: Union[str, Path],
        file_hash_or_mtime: Union[str, int, float],
    ) -> bool:
        """
        Return True if `file_path` is already present in `files` with an identical `file_hash_or_mtime`.
        Returns False if the file has never been scanned or has been modified since the last scan.
        """
        norm_path = normalize_file_path(file_path)
        norm_mtime = normalize_hash_or_mtime(file_hash_or_mtime)

        conn = self._get_connection()
        row = conn.execute(
            "SELECT file_hash_or_mtime FROM files WHERE file_path = ?;",
            (norm_path,),
        ).fetchone()
        if row is None:
            return False
        return row["file_hash_or_mtime"] == norm_mtime

    def get_scanned_files_map(self) -> Dict[str, str]:
        """
        Return a mapping of `{lowercase_normalized_file_path: file_hash_or_mtime}` for fast
        in-memory pre-filtering during full-drive Everything SDK discovery scans.
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT file_path, file_hash_or_mtime FROM files;").fetchall()
        return {row["file_path"].lower(): row["file_hash_or_mtime"] for row in rows}

    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single file record (with linked person details if any) by its primary key."""
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT
                f.*,
                p.name AS person_name,
                p.face_thumbnail_path AS person_face_thumbnail_path
            FROM files f
            LEFT JOIN people p ON f.person_id = p.id
            WHERE f.id = ?;
            """,
            (int(file_id),),
        ).fetchone()
        return self._row_to_file_dict(row) if row else None

    def get_file_by_path(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Fetch a single file record (with linked person details if any) by its filesystem path."""
        norm_path = normalize_file_path(file_path)
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT
                f.*,
                p.name AS person_name,
                p.face_thumbnail_path AS person_face_thumbnail_path
            FROM files f
            LEFT JOIN people p ON f.person_id = p.id
            WHERE f.file_path = ?;
            """,
            (norm_path,),
        ).fetchone()
        return self._row_to_file_dict(row) if row else None

    def get_files_by_ids(self, file_ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Batch-fetch full decrypted file records (with linked person details) for specific primary keys."""
        clean_ids = [int(fid) for fid in file_ids if fid is not None]
        if not clean_ids:
            return []
        conn = self._get_connection()
        results_by_id: Dict[int, Dict[str, Any]] = {}
        for chunk_start in range(0, len(clean_ids), 400):
            chunk = clean_ids[chunk_start : chunk_start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT
                    f.*,
                    p.name AS person_name,
                    p.face_thumbnail_path AS person_face_thumbnail_path
                FROM files f
                LEFT JOIN people p ON f.person_id = p.id
                WHERE f.id IN ({placeholders});
                """,
                chunk,
            ).fetchall()
            for r in rows:
                d = self._row_to_file_dict(r)
                results_by_id[int(d["id"])] = d
        return [results_by_id[fid] for fid in clean_ids if fid in results_by_id]

    def get_document_face_candidates(self) -> List[Dict[str, Any]]:
        """
        Fetch lightweight face-matching metadata (`id`, `file_path`, `person_id`, decrypted `face_embedding`)
        for all verified documents/photos without loading or decrypting heavy `ocr_text` or `thumbnail_b64` blobs.
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT id, file_path, person_id, face_embedding
            FROM files
            WHERE is_document = 1;
            """
        ).fetchall()
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            raw_emb = row["face_embedding"]
            dec_emb = None
            if raw_emb is not None:
                try:
                    dec_emb = self._crypto.decrypt_embedding(raw_emb)
                except Exception:
                    dec_emb = None
            candidates.append(
                {
                    "id": int(row["id"]),
                    "file_path": row["file_path"],
                    "person_id": row["person_id"],
                    "face_embedding": dec_emb,
                }
            )
        return candidates

    def update_file_person(self, file_id: int, person_id: Optional[int]) -> bool:
        """Assign or clear `person_id` on an existing scanned file."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE files SET person_id = ? WHERE id = ?;",
                (person_id, int(file_id)),
            )
            return cursor.rowcount > 0

    def update_file_face_embedding(self, file_id: int, face_embedding: Any) -> bool:
        """Store or update the encrypted 512-d `face_embedding` vector on an existing file record."""
        enc_emb = self._crypto.encrypt_embedding(face_embedding)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE files SET face_embedding = ? WHERE id = ?;",
                (enc_emb, int(file_id)),
            )
            return cursor.rowcount > 0

    def delete_file_by_path(self, file_path: Union[str, Path]) -> bool:
        """Delete a file record by path (e.g., when a file has been removed from disk)."""
        norm_path = normalize_file_path(file_path)
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM files WHERE file_path = ?;", (norm_path,))
            return cursor.rowcount > 0

    def update_file_category(self, file_path: Union[str, Path], new_category: str) -> bool:
        """
        Manually override/transfer a document's category.
        If set to 'Not a Document', sets is_document = 0, otherwise sets is_document = 1.
        """
        norm_path = normalize_file_path(file_path)
        cat = new_category.strip()
        if cat not in ALL_DOC_CATEGORIES:
            raise ValueError(f"Invalid category '{cat}'. Expected one of: {ALL_DOC_CATEGORIES}")

        is_doc = 0 if cat in (NOT_A_DOCUMENT, UNREADABLE_OR_CORRUPT) else 1
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE files SET doc_category = ?, is_document = ? WHERE file_path = ?;",
                (cat, is_doc, norm_path),
            )
            return cursor.rowcount > 0

    def update_file_path(self, old_file_path: Union[str, Path], new_file_path: Union[str, Path]) -> bool:
        """
        Update a document's filesystem path in SQLite after moving the file on disk.
        """
        old_norm = normalize_file_path(old_file_path)
        new_norm = normalize_file_path(new_file_path)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE files SET file_path = ? WHERE file_path = ?;",
                (new_norm, old_norm),
            )
            return cursor.rowcount > 0

    def ignore_file(self, file_path: Union[str, Path]) -> None:
        """
        Add a file path to `ignored_files` so future scans will permanently skip it.
        """
        norm_path = normalize_file_path(file_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO ignored_files (file_path, ignored_at) VALUES (?, ?) ON CONFLICT(file_path) DO NOTHING;",
                (norm_path, now_iso),
            )

    def is_file_ignored(self, file_path: Union[str, Path]) -> bool:
        """Check if a file path is in `ignored_files`."""
        norm_path = normalize_file_path(file_path)
        conn = self._get_connection()
        row = conn.execute("SELECT 1 FROM ignored_files WHERE file_path = ?;", (norm_path,)).fetchone()
        return row is not None

    def get_ignored_files_set(self) -> Set[str]:
        """Return a set of lowercase normalized paths present in `ignored_files`."""
        conn = self._get_connection()
        rows = conn.execute("SELECT file_path FROM ignored_files;").fetchall()
        return {r["file_path"].lower() for r in rows}

    def remove_file_record(self, file_path: Union[str, Path], add_to_ignored: bool = True) -> bool:
        """
        Remove a document record from SQLite (keeps the physical file untouched).
        If `add_to_ignored` is True, registers the path in `ignored_files` so it won't be scanned again.
        """
        norm_path = normalize_file_path(file_path)
        if add_to_ignored:
            self.ignore_file(norm_path)
        return self.delete_file_by_path(norm_path)

    def clear_category_documents(self, category: str) -> int:
        """
        Delete all DB records belonging to `category` (without touching the physical files on disk).
        Returns the number of deleted records.
        """
        clean_cat = category.strip()
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM files WHERE doc_category = ?;", (clean_cat,))
            return cursor.rowcount

    def prune_missing_files(self) -> int:
        """
        Remove entries from `files` whose underlying disk path no longer exists.
        Returns the count of pruned database records.
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT id, file_path FROM files;").fetchall()
        missing_ids = [int(r["id"]) for r in rows if not os.path.exists(r["file_path"])]
        if not missing_ids:
            return 0

        with self.transaction() as write_conn:
            removed = 0
            # Batch in chunks of 500 to stay well within SQLite variable limits
            for idx in range(0, len(missing_ids), 500):
                chunk = missing_ids[idx : idx + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor = write_conn.execute(
                    f"DELETE FROM files WHERE id IN ({placeholders});",
                    chunk,
                )
                removed += cursor.rowcount
            return removed

    # -------------------------------------------------------------------------
    # Search & Aggregations
    # -------------------------------------------------------------------------

    def _build_search_where(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        person_id: Optional[int] = None,
        expiring_only: bool = False,
        include_non_documents: bool = False,
        scan_mode: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        conditions: List[str] = []
        params: List[Any] = []

        clean_category = category.strip() if category else None
        if clean_category and clean_category.lower() != "all":
            conditions.append("f.doc_category = ?")
            params.append(clean_category)
            if clean_category not in (NOT_A_DOCUMENT, UNREADABLE_OR_CORRUPT) and not include_non_documents:
                conditions.append("f.is_document = 1")
        elif not include_non_documents:
            conditions.append("f.is_document = 1")
            # If in ID Vault mode, strictly exclude Face Gallery Photos unless specifically requested
            clean_mode = (scan_mode or "").strip().lower()
            if clean_mode in ("id_vault", "id vault"):
                conditions.append(
                    "f.doc_category NOT IN ('Face Gallery Photo', 'Unidentified / Unreadable', 'Not a Document')"
                )

        if person_id is not None:
            conditions.append("f.person_id = ?")
            params.append(int(person_id))

        if expiring_only:
            conditions.append("f.is_document = 1")
            conditions.append("f.expiry_date IS NOT NULL")
            conditions.append("TRIM(dv_decrypt(f.expiry_date)) != ''")
            conditions.append("date(dv_decrypt(f.expiry_date)) <= date('now', '+90 days')")

        if keyword and keyword.strip():
            tokens = [tok for tok in keyword.strip().split() if tok]
            for token in tokens:
                escaped = f"%{_escape_like(token)}%"
                compact_token = token.replace(" ", "").replace("-", "")
                compact_escaped = f"%{_escape_like(compact_token)}%"
                conditions.append(
                    """(
                        dv_decrypt(f.ocr_text) LIKE ? ESCAPE '\\'
                        OR dv_decrypt(f.extracted_id_number) LIKE ? ESCAPE '\\'
                        OR REPLACE(REPLACE(dv_decrypt(f.extracted_id_number), ' ', ''), '-', '') LIKE ? ESCAPE '\\'
                        OR dv_decrypt(f.dob) LIKE ? ESCAPE '\\'
                        OR dv_decrypt(f.expiry_date) LIKE ? ESCAPE '\\'
                        OR COALESCE(f.doc_category, '') LIKE ? ESCAPE '\\'
                        OR COALESCE(p.name, '') LIKE ? ESCAPE '\\'
                        OR f.file_path LIKE ? ESCAPE '\\'
                    )"""
                )
                params.extend([escaped, escaped, compact_escaped, escaped, escaped, escaped, escaped, escaped])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where_clause, params

    @staticmethod
    def _resolve_order_by(sort_by: Optional[str]) -> str:
        clean_sort = (sort_by or "scanned_desc").strip().lower()
        if clean_sort in ("date_taken_desc", "date_desc", "newest"):
            return "COALESCE(NULLIF(TRIM(f.date_taken), ''), f.scanned_at) DESC, f.id DESC"
        if clean_sort in ("date_taken_asc", "date_asc", "oldest"):
            return "COALESCE(NULLIF(TRIM(f.date_taken), ''), f.scanned_at) ASC, f.id ASC"
        if clean_sort == "name_asc":
            return "f.file_path COLLATE NOCASE ASC, f.id ASC"
        if clean_sort == "size_desc":
            return "f.file_size DESC, f.id DESC"
        return "f.scanned_at DESC, f.id DESC"

    def count_documents(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        person_id: Optional[int] = None,
        expiring_only: bool = False,
        include_non_documents: bool = False,
        scan_mode: Optional[str] = None,
    ) -> int:
        """Return total matching record count for paginated `/api/documents` and `/api/faces` queries."""
        where_clause, params = self._build_search_where(
            keyword=keyword,
            category=category,
            person_id=person_id,
            expiring_only=expiring_only,
            include_non_documents=include_non_documents,
            scan_mode=scan_mode,
        )
        query = f"""
            SELECT COUNT(*) AS total
            FROM files f
            LEFT JOIN people p ON f.person_id = p.id
            {where_clause};
        """
        conn = self._get_connection()
        row = conn.execute(query, params).fetchone()
        return int(row["total"]) if row else 0

    def search_documents(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        person_id: Optional[int] = None,
        expiring_only: bool = False,
        include_non_documents: bool = False,
        scan_mode: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
        sort_by: str = "scanned_desc",
    ) -> List[Dict[str, Any]]:
        """
        Search scanned documents by keyword, document category, expiring status, and/or linked person_id,
        supporting SQL `LIMIT`/`OFFSET` pagination and EXIF `date_taken` chronological sorting.
        """
        where_clause, params = self._build_search_where(
            keyword=keyword,
            category=category,
            person_id=person_id,
            expiring_only=expiring_only,
            include_non_documents=include_non_documents,
            scan_mode=scan_mode,
        )
        order_clause = self._resolve_order_by(sort_by)
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))

        query = f"""
            SELECT
                f.*,
                p.name AS person_name,
                p.face_thumbnail_path AS person_face_thumbnail_path
            FROM files f
            LEFT JOIN people p ON f.person_id = p.id
            {where_clause}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?;
        """
        params.extend([safe_limit, safe_offset])

        conn = self._get_connection()
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_file_dict(row) for row in rows]

    def get_category_counts(self, include_non_documents: bool = False) -> Dict[str, int]:
        """
        Return a dictionary mapping each document category to its total file count.
        All standard document categories are pre-populated with 0 for predictable UI rendering.
        """
        categories_to_seed = ALL_DOC_CATEGORIES if include_non_documents else DOCUMENT_CATEGORIES
        counts: Dict[str, int] = {cat: 0 for cat in categories_to_seed}

        conn = self._get_connection()
        if include_non_documents:
            rows = conn.execute(
                """
                SELECT doc_category, COUNT(*) AS total
                FROM files
                GROUP BY doc_category;
                """
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT doc_category, COUNT(*) AS total
                FROM files
                WHERE is_document = 1 OR doc_category = '{UNREADABLE_OR_CORRUPT}'
                GROUP BY doc_category;
                """
            ).fetchall()

        for row in rows:
            cat = row["doc_category"]
            if not include_non_documents and cat == NOT_A_DOCUMENT:
                continue
            counts[cat] = int(row["total"])

        return counts

    def get_expiring_documents_count(self, days: int = 90) -> int:
        """
        Count verified documents where decrypted expiry_date is expired or expiring within `days` days.
        """
        conn = self._get_connection()
        query = f"""
            SELECT COUNT(*) AS total
            FROM files
            WHERE is_document = 1
              AND expiry_date IS NOT NULL
              AND TRIM(dv_decrypt(expiry_date)) != ''
              AND date(dv_decrypt(expiry_date)) <= date('now', '+{int(days)} days');
        """
        row = conn.execute(query).fetchone()
        return int(row["total"]) if row else 0

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return overall scan statistics (total scanned, total documents, expiring, people count, categories)."""
        conn = self._get_connection()
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total_scanned,
                COALESCE(SUM(CASE WHEN is_document = 1 THEN 1 ELSE 0 END), 0) AS total_documents,
                COALESCE(SUM(CASE WHEN is_document = 0 THEN 1 ELSE 0 END), 0) AS total_non_documents
            FROM files;
            """
        ).fetchone()
        people_row = conn.execute("SELECT COUNT(*) AS total_people FROM people;").fetchone()
        expiring_count = self.get_expiring_documents_count(days=90)

        return {
            "total_scanned": int(totals["total_scanned"]) if totals else 0,
            "total_documents": int(totals["total_documents"]) if totals else 0,
            "total_non_documents": int(totals["total_non_documents"]) if totals else 0,
            "total_people": int(people_row["total_people"]) if people_row else 0,
            "expiring_documents_count": expiring_count,
            "categories": self.get_category_counts(include_non_documents=False),
        }

    def get_duplicate_groups_count(self, max_hamming_distance: int = 6) -> int:
        """
        Fast O(1)-cached count of duplicate document groups using ONLY unencrypted 64-bit `dhash` integers.
        Avoids loading or AES-decrypting `ocr_text`, `face_embedding`, or `thumbnail_b64` during `/api/stats` polls.
        """
        threshold = max(0, min(64, int(max_hamming_distance)))
        seq = self._mutation_seq
        cached_seq, cached_thresh, cached_cnt = self._dup_count_cache
        if cached_seq == seq and cached_thresh == threshold:
            return cached_cnt

        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT dhash
            FROM files
            WHERE is_document = 1
              AND dhash IS NOT NULL
              AND LENGTH(dhash) = 16;
            """
        ).fetchall()

        freq: Counter[int] = Counter()
        for row in rows:
            try:
                freq[int(str(row["dhash"]), 16)] += 1
            except (ValueError, TypeError):
                continue

        total_items = sum(freq.values())
        if total_items < 2:
            self._dup_count_cache = (seq, threshold, 0)
            return 0

        if threshold == 0:
            count = sum(1 for c in freq.values() if c >= 2)
            self._dup_count_cache = (seq, threshold, count)
            return count

        unique_hashes = list(freq.keys())
        u_len = len(unique_hashes)
        parent = list(range(u_len))

        def find_root(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        for i in range(u_len):
            hi = unique_hashes[i]
            for j in range(i + 1, u_len):
                if (hi ^ unique_hashes[j]).bit_count() <= threshold:
                    ra, rb = find_root(i), find_root(j)
                    if ra != rb:
                        if ra < rb:
                            parent[rb] = ra
                        else:
                            parent[ra] = rb

        comp_counts: Dict[int, int] = {}
        for idx, h_val in enumerate(unique_hashes):
            root = find_root(idx)
            comp_counts[root] = comp_counts.get(root, 0) + freq[h_val]

        count = sum(1 for total_in_comp in comp_counts.values() if total_in_comp >= 2)
        self._dup_count_cache = (seq, threshold, count)
        return count

    def find_duplicate_groups(self, max_hamming_distance: int = 6) -> List[Dict[str, Any]]:
        """
        Group identical or near-identical verified documents across different folders
        using 64-bit perceptual difference hash (dHash) Hamming distance clustering.
        Pre-clusters lightweight `(id, dhash)` rows first so non-duplicate documents are never decrypted.
        """
        threshold = max(0, min(64, int(max_hamming_distance)))
        seq = self._mutation_seq
        if self._dup_groups_cache is not None:
            c_seq, c_thresh, c_groups = self._dup_groups_cache
            if c_seq == seq and c_thresh == threshold:
                return [
                    dict(g, documents=[dict(d) for d in g["documents"]])
                    for g in c_groups
                ]

        conn = self._get_connection()
        light_rows = conn.execute(
            """
            SELECT id, file_size, dhash
            FROM files
            WHERE is_document = 1
              AND dhash IS NOT NULL
              AND LENGTH(dhash) = 16
            ORDER BY file_size DESC, id ASC;
            """
        ).fetchall()

        doc_ids: List[int] = []
        hash_ints: List[int] = []
        for row in light_rows:
            try:
                h_int = int(str(row["dhash"]), 16)
            except (ValueError, TypeError):
                continue
            doc_ids.append(int(row["id"]))
            hash_ints.append(h_int)

        n = len(doc_ids)
        if n < 2:
            self._dup_count_cache = (seq, threshold, 0)
            self._dup_groups_cache = (seq, threshold, [])
            return []

        # Union-Find (Disjoint Set) clustering over perceptual Hamming distance <= threshold
        parent = list(range(n))

        def find_root(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union_nodes(a: int, b: int) -> None:
            ra, rb = find_root(a), find_root(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        for i in range(n):
            hi = hash_ints[i]
            for j in range(i + 1, n):
                if (hi ^ hash_ints[j]).bit_count() <= threshold:
                    union_nodes(i, j)

        clusters: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find_root(idx)
            clusters.setdefault(root, []).append(idx)

        multi_clusters = [indices for indices in clusters.values() if len(indices) >= 2]
        if not multi_clusters:
            self._dup_count_cache = (seq, threshold, 0)
            self._dup_groups_cache = (seq, threshold, [])
            return []

        # Decrypt ONLY the documents that actually belong to a duplicate cluster
        needed_ids = [doc_ids[i] for indices in multi_clusters for i in indices]
        fetched_docs = {int(d["id"]): d for d in self.get_files_by_ids(needed_ids)}

        duplicate_groups: List[Dict[str, Any]] = []
        for member_indices in multi_clusters:
            member_docs = [fetched_docs[doc_ids[i]] for i in member_indices if doc_ids[i] in fetched_docs]
            if len(member_docs) < 2:
                continue

            # Compute max pairwise Hamming distance inside this group
            max_dist = 0
            for idx_pos, a_idx in enumerate(member_indices):
                for b_idx in member_indices[idx_pos + 1 :]:
                    dist = (hash_ints[a_idx] ^ hash_ints[b_idx]).bit_count()
                    if dist > max_dist:
                        max_dist = dist

            similarity_pct = round(max(0.0, (1.0 - (max_dist / 64.0)) * 100.0), 1)
            sizes = [int(d.get("file_size") or 0) for d in member_docs]
            wasted_bytes = max(0, sum(sizes) - max(sizes))

            primary = member_docs[0]
            extracted_id = next(
                (d["extracted_id_number"] for d in member_docs if d.get("extracted_id_number")),
                None,
            )

            duplicate_groups.append(
                {
                    "doc_category": primary.get("doc_category"),
                    "extracted_id_number": extracted_id,
                    "copy_count": len(member_docs),
                    "max_hamming_distance": max_dist,
                    "similarity_percent": similarity_pct,
                    "match_type": "exact" if max_dist == 0 else "near_identical",
                    "dhash_anchor": primary.get("dhash"),
                    "wasted_bytes": wasted_bytes,
                    "documents": member_docs,
                }
            )

        duplicate_groups.sort(
            key=lambda g: (int(g["copy_count"]), int(g["wasted_bytes"])),
            reverse=True,
        )
        for group_num, grp in enumerate(duplicate_groups, start=1):
            grp["group_id"] = group_num

        self._dup_count_cache = (seq, threshold, len(duplicate_groups))
        self._dup_groups_cache = (
            seq,
            threshold,
            [dict(g, documents=[dict(d) for d in g["documents"]]) for g in duplicate_groups],
        )
        return duplicate_groups

    # -------------------------------------------------------------------------
    # People / Face Identity Operations
    # -------------------------------------------------------------------------

    def add_person(
        self,
        face_embedding: Any = None,
        face_thumbnail_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> int:
        """
        Insert a new person record into `people` (with AES-256-GCM encrypted `face_embedding`).
        Defaults `name` automatically to `"Person 1"`, `"Person 2"`, etc. based on the new row ID.
        Returns the `people.id` primary key.
        """
        enc_emb = self._crypto.encrypt_embedding(face_embedding)
        clean_thumb = str(face_thumbnail_path).strip() if face_thumbnail_path else None
        clean_name = name.strip() if (name and name.strip()) else None

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO people (name, face_thumbnail_path, face_embedding)
                VALUES (?, ?, ?);
                """,
                (clean_name or "Person", clean_thumb, enc_emb),
            )
            person_id = int(cursor.lastrowid)
            if clean_name is None:
                conn.execute(
                    "UPDATE people SET name = ? WHERE id = ?;",
                    (f"Person {person_id}", person_id),
                )
            return person_id

    def get_person(self, person_id: int, include_embedding: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch a single person record by ID, including their linked document count."""
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT
                p.id,
                p.name,
                p.face_thumbnail_path,
                p.face_embedding,
                COUNT(CASE WHEN f.is_document = 1 THEN 1 END) AS document_count
            FROM people p
            LEFT JOIN files f ON f.person_id = p.id
            WHERE p.id = ?
            GROUP BY p.id;
            """,
            (int(person_id),),
        ).fetchone()
        return self._row_to_person_dict(row, include_embedding=include_embedding) if row else None

    def count_people(self, active_only: bool = True) -> int:
        """Return total person clusters (optionally restricted to clusters with >= 1 active document)."""
        conn = self._get_connection()
        if active_only:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total FROM (
                    SELECT p.id
                    FROM people p
                    INNER JOIN files f ON f.person_id = p.id AND f.is_document = 1
                    GROUP BY p.id
                );
                """
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS total FROM people;").fetchone()
        return int(row["total"]) if row else 0

    def get_all_people(
        self,
        include_embeddings: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return people ordered by document count descending (then ID ascending),
        with decrypted `face_embedding` when `include_embeddings=True` and optional SQL pagination.
        """
        conn = self._get_connection()
        having_sql = "HAVING COUNT(CASE WHEN f.is_document = 1 THEN 1 END) > 0" if active_only else ""
        pagination_sql = ""
        params: List[Any] = []
        if limit is not None:
            pagination_sql = "LIMIT ? OFFSET ?"
            params.extend([max(1, int(limit)), max(0, int(offset))])

        rows = conn.execute(
            f"""
            SELECT
                p.id,
                p.name,
                p.face_thumbnail_path,
                p.face_embedding,
                COUNT(CASE WHEN f.is_document = 1 THEN 1 END) AS document_count
            FROM people p
            LEFT JOIN files f ON f.person_id = p.id
            GROUP BY p.id
            {having_sql}
            ORDER BY document_count DESC, p.id ASC
            {pagination_sql};
            """,
            params,
        ).fetchall()
        return [self._row_to_person_dict(row, include_embedding=include_embeddings) for row in rows]

    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        face_thumbnail_path: Optional[str] = None,
        face_embedding: Any = None,
    ) -> bool:
        """Update non-None fields of an existing person record."""
        updates: List[str] = []
        params: List[Any] = []

        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Person name cannot be empty.")
            updates.append("name = ?")
            params.append(clean_name)

        if face_thumbnail_path is not None:
            updates.append("face_thumbnail_path = ?")
            params.append(str(face_thumbnail_path).strip() or None)

        if face_embedding is not None:
            updates.append("face_embedding = ?")
            params.append(self._crypto.encrypt_embedding(face_embedding))

        if not updates:
            return False

        params.append(int(person_id))
        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE people SET {', '.join(updates)} WHERE id = ?;",
                params,
            )
            return cursor.rowcount > 0

    def rename_person(self, person_id: int, new_name: str) -> bool:
        """Rename a detected person (e.g., 'Person 1' -> 'Aarav Patel')."""
        return self.update_person(person_id=person_id, name=new_name)

    def merge_people(self, source_person_id: int, target_person_id: int) -> bool:
        """Reassign all files from `source_person_id` to `target_person_id` and delete `source_person_id`."""
        src_id = int(source_person_id)
        dst_id = int(target_person_id)
        if src_id == dst_id:
            return False

        with self.transaction() as conn:
            target_exists = conn.execute(
                "SELECT 1 FROM people WHERE id = ?;",
                (dst_id,),
            ).fetchone()
            if not target_exists:
                return False

            conn.execute(
                "UPDATE files SET person_id = ? WHERE person_id = ?;",
                (dst_id, src_id),
            )
            cursor = conn.execute("DELETE FROM people WHERE id = ?;", (src_id,))
            return cursor.rowcount > 0

    def delete_person(self, person_id: int) -> bool:
        """Delete a person record; linked files automatically have `person_id` set to NULL."""
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM people WHERE id = ?;", (int(person_id),))
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # App Settings, Master PIN Security & DEK Key Wrapping
    # -------------------------------------------------------------------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch a value from the settings key-value store."""
        conn = self._get_connection()
        row = conn.execute("SELECT value FROM settings WHERE key = ?;", (str(key),)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        """Store or update a key-value setting in SQLite."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value;
                """,
                (str(key), str(value)),
            )

    def delete_setting(self, key: str) -> bool:
        """Remove a key from the settings table."""
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM settings WHERE key = ?;", (str(key),))
            return cur.rowcount > 0

    def unlock_with_pin(self, pin_or_password: str) -> bool:
        """
        Unlock/re-verify the SQLite AES-256-GCM DEK envelope using the Master PIN or Master Password.
        """
        wrapped = self.get_setting("vault_pin_wrapped_dek")
        if not wrapped:
            return False
        return self._crypto.unlock_dek_with_pin(str(pin_or_password).strip(), wrapped)

    def is_master_password_enabled(self) -> bool:
        """Return True if an alphanumeric Master Password has been configured."""
        enabled = self.get_setting("master_password_enabled", "0")
        has_hash = bool(self.get_setting("master_password_hash"))
        return enabled == "1" and has_hash

    def set_master_password(self, password: str) -> None:
        """
        Store a salted PBKDF2-HMAC-SHA256 hash of an alphanumeric Master Password
        and wrap the SQLite AES-256-GCM DEK with a KEK derived from this Master Password.
        Requires at least 4 characters.
        """
        clean_pwd = str(password).strip()
        if len(clean_pwd) < 4:
            raise ValueError("Master Password must be at least 4 characters.")
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", clean_pwd.encode("utf-8"), salt.encode("utf-8"), 100_000)
        stored_val = f"{salt}${dk.hex()}"
        wrapped_dek = self._crypto.wrap_dek_with_pin(clean_pwd)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('master_password_hash', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (stored_val,),
            )
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('master_password_enabled', '1') ON CONFLICT(key) DO UPDATE SET value = excluded.value;"
            )
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('vault_pin_wrapped_dek', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (wrapped_dek,),
            )

    def verify_master_password(self, password: str) -> bool:
        """
        Verify the provided password against the Master Password hash and unlock the wrapped DEK.
        If Master Password is not enabled, checks fallback 4-digit PIN (if set).
        If neither is configured, returns True (unrestricted mode).
        """
        if self.is_master_password_enabled():
            stored = self.get_setting("master_password_hash")
            if not stored or "$" not in stored:
                return False
            clean_pwd = str(password).strip()
            salt, expected_hash = stored.split("$", 1)
            dk = hashlib.pbkdf2_hmac("sha256", clean_pwd.encode("utf-8"), salt.encode("utf-8"), 100_000)
            ok = secrets.compare_digest(dk.hex(), expected_hash)
            if ok:
                wrapped = self.get_setting("vault_pin_wrapped_dek")
                if wrapped:
                    self._crypto.unlock_dek_with_pin(clean_pwd, wrapped)
                else:
                    self.set_setting("vault_pin_wrapped_dek", self._crypto.wrap_dek_with_pin(clean_pwd))
            return ok

        # Fallback to PIN if PIN is configured
        if self.is_pin_enabled():
            return self.verify_pin(password)

        return True

    def disable_master_password(self, current_password: str) -> bool:
        """Disable Master Password after verifying current password."""
        if not self.is_master_password_enabled():
            return True
        if not self.verify_master_password(current_password):
            return False
        with self.transaction() as conn:
            conn.execute("UPDATE settings SET value = '0' WHERE key = 'master_password_enabled';")
        return True

    def is_pin_enabled(self) -> bool:
        """Return True if 4-digit PIN security is enabled and a valid hash is stored."""
        enabled = self.get_setting("pin_enabled", "0")
        has_hash = bool(self.get_setting("app_pin_hash"))
        return enabled == "1" and has_hash

    def set_pin(self, pin_code: str) -> None:
        """
        Store a salted PBKDF2-HMAC-SHA256 hash of the 4-digit Master PIN, enable PIN lock,
        and wrap the SQLite AES-256-GCM DEK with a KEK derived from the Master PIN.
        """
        clean_pin = str(pin_code).strip()
        if len(clean_pin) != 4 or not clean_pin.isdigit():
            raise ValueError("PIN must be exactly 4 numeric digits.")
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", clean_pin.encode("utf-8"), salt.encode("utf-8"), 100_000)
        stored_val = f"{salt}${dk.hex()}"
        wrapped_dek = self._crypto.wrap_dek_with_pin(clean_pin)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('app_pin_hash', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (stored_val,),
            )
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('pin_enabled', '1') ON CONFLICT(key) DO UPDATE SET value = excluded.value;"
            )
            if not self.is_master_password_enabled():
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES ('vault_pin_wrapped_dek', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                    (wrapped_dek,),
                )

    def verify_pin(self, pin_code: str) -> bool:
        """
        Verify the provided 4-digit PIN against the stored hash and unlock the wrapped DEK.
        If PIN protection is disabled, always returns True.
        """
        if not self.is_pin_enabled():
            return True
        stored = self.get_setting("app_pin_hash")
        if not stored or "$" not in stored:
            return False
        clean_pin = str(pin_code).strip()
        salt, expected_hash = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", clean_pin.encode("utf-8"), salt.encode("utf-8"), 100_000)
        ok = secrets.compare_digest(dk.hex(), expected_hash)
        if ok and not self.is_master_password_enabled():
            wrapped = self.get_setting("vault_pin_wrapped_dek")
            if wrapped:
                self._crypto.unlock_dek_with_pin(clean_pin, wrapped)
            else:
                self.set_setting("vault_pin_wrapped_dek", self._crypto.wrap_dek_with_pin(clean_pin))
        return ok

    def disable_pin(self, pin_code: str) -> bool:
        """
        Disable PIN lock after verifying current PIN.
        Returns True if successful, False if PIN is incorrect.
        """
        if not self.is_pin_enabled():
            return True
        if not self.verify_pin(pin_code):
            return False
        with self.transaction() as conn:
            conn.execute("UPDATE settings SET value = '0' WHERE key = 'pin_enabled';")
        return True


# -----------------------------------------------------------------------------
# Default Singleton & Module-Level Helper Functions
# -----------------------------------------------------------------------------

db = DatabaseManager(DEFAULT_DB_PATH, auto_init=False)


def init_db() -> None:
    """Initialize the default `docuvault.db` SQLite database."""
    db.init_db()


def upsert_file(
    file_path: Union[str, Path],
    file_hash_or_mtime: Union[str, int, float],
    file_size: int = 0,
    is_document: Optional[bool] = None,
    doc_category: Optional[str] = None,
    extracted_id_number: Optional[str] = None,
    dob: Optional[str] = None,
    expiry_date: Optional[str] = None,
    ocr_text: Optional[str] = None,
    person_id: Optional[int] = None,
    thumbnail_path: Optional[str] = None,
    dhash: Optional[str] = None,
    scanned_at: Optional[str] = None,
    thumbnail_b64: Optional[str] = None,
    face_embedding: Any = None,
    date_taken: Optional[str] = None,
) -> int:
    """Module-level helper to insert or update a scanned file in `docuvault.db`."""
    return db.upsert_file(
        file_path=file_path,
        file_hash_or_mtime=file_hash_or_mtime,
        file_size=file_size,
        is_document=is_document,
        doc_category=doc_category,
        extracted_id_number=extracted_id_number,
        dob=dob,
        expiry_date=expiry_date,
        ocr_text=ocr_text,
        person_id=person_id,
        thumbnail_path=thumbnail_path,
        dhash=dhash,
        scanned_at=scanned_at,
        thumbnail_b64=thumbnail_b64,
        face_embedding=face_embedding,
        date_taken=date_taken,
    )


insert_or_update_file = upsert_file


def is_file_scanned(
    file_path: Union[str, Path],
    file_hash_or_mtime: Union[str, int, float],
) -> bool:
    """Module-level helper to check whether a file (by path + hash/mtime) is already scanned."""
    return db.is_file_scanned(file_path=file_path, file_hash_or_mtime=file_hash_or_mtime)


def get_scanned_files_map() -> Dict[str, str]:
    """Module-level helper returning `{lowercase_file_path: file_hash_or_mtime}`."""
    return db.get_scanned_files_map()


def count_documents(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    person_id: Optional[int] = None,
    expiring_only: bool = False,
    include_non_documents: bool = False,
    scan_mode: Optional[str] = None,
) -> int:
    """Module-level helper returning total matching document count for pagination."""
    return db.count_documents(
        keyword=keyword,
        category=category,
        person_id=person_id,
        expiring_only=expiring_only,
        include_non_documents=include_non_documents,
        scan_mode=scan_mode,
    )


def search_documents(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    person_id: Optional[int] = None,
    expiring_only: bool = False,
    include_non_documents: bool = False,
    scan_mode: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
    sort_by: str = "scanned_desc",
) -> List[Dict[str, Any]]:
    """Module-level helper to search documents by keyword, category, expiring status, person_id, and sort order."""
    return db.search_documents(
        keyword=keyword,
        category=category,
        person_id=person_id,
        expiring_only=expiring_only,
        include_non_documents=include_non_documents,
        scan_mode=scan_mode,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
    )


def get_expiring_documents_count(days: int = 90) -> int:
    """Module-level helper to count documents expired or expiring within `days` days."""
    return db.get_expiring_documents_count(days=days)


def get_category_counts(include_non_documents: bool = False) -> Dict[str, int]:
    """Module-level helper to retrieve document counts grouped by `doc_category`."""
    return db.get_category_counts(include_non_documents=include_non_documents)


def get_summary_stats() -> Dict[str, Any]:
    """Module-level helper to retrieve overall scan and document counts."""
    return db.get_summary_stats()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Module-level helper to get a setting value."""
    return db.get_setting(key=key, default=default)


def set_setting(key: str, value: str) -> None:
    """Module-level helper to set a setting value."""
    db.set_setting(key=key, value=value)


def unlock_with_pin(pin_or_password: str) -> bool:
    """Module-level helper to unlock the SQLite AES-256-GCM DEK using Master PIN / Password."""
    return db.unlock_with_pin(pin_or_password=pin_or_password)


def is_master_password_enabled() -> bool:
    """Module-level helper to check if Master Password protection is active."""
    return db.is_master_password_enabled()


def set_master_password(password: str) -> None:
    """Module-level helper to configure Master Password."""
    db.set_master_password(password=password)


def verify_master_password(password: str) -> bool:
    """Module-level helper to verify Master Password."""
    return db.verify_master_password(password=password)


def disable_master_password(current_password: str) -> bool:
    """Module-level helper to disable Master Password."""
    return db.disable_master_password(current_password=current_password)


def record_scan_history(file_path: Union[str, Path], file_size: int, mtime_sec: float) -> None:
    """Module-level helper to record Ghost Cache entry."""
    db.record_scan_history(file_path=file_path, file_size=file_size, mtime_sec=mtime_sec)


def is_in_ghost_cache(file_path: Union[str, Path], file_size: int, mtime_sec: float) -> bool:
    """Module-level helper to check Ghost Cache."""
    return db.is_in_ghost_cache(file_path=file_path, file_size=file_size, mtime_sec=mtime_sec)


def get_ghost_cache_map() -> Dict[str, Tuple[int, float]]:
    """Module-level helper to fetch Ghost Cache map."""
    return db.get_ghost_cache_map()


def is_pin_enabled() -> bool:
    """Module-level helper to check if 4-digit PIN protection is active."""
    return db.is_pin_enabled()


def set_pin(pin_code: str) -> None:
    """Module-level helper to configure and enable 4-digit PIN."""
    db.set_pin(pin_code=pin_code)


def verify_pin(pin_code: str) -> bool:
    """Module-level helper to verify 4-digit PIN."""
    return db.verify_pin(pin_code=pin_code)


def disable_pin(pin_code: str) -> bool:
    """Module-level helper to disable 4-digit PIN."""
    return db.disable_pin(pin_code=pin_code)


def get_duplicate_groups_count(max_hamming_distance: int = 6) -> int:
    """Module-level helper to fast-count duplicate document groups by 64-bit dHash."""
    return db.get_duplicate_groups_count(max_hamming_distance=max_hamming_distance)


def find_duplicate_groups(max_hamming_distance: int = 6) -> List[Dict[str, Any]]:
    """Module-level helper to group identical or near-identical documents by 64-bit dHash."""
    return db.find_duplicate_groups(max_hamming_distance=max_hamming_distance)


def get_file_by_id(file_id: int) -> Optional[Dict[str, Any]]:
    """Module-level helper to fetch a file record by ID."""
    return db.get_file_by_id(file_id=file_id)


def get_file_by_path(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Module-level helper to fetch a file record by filesystem path."""
    return db.get_file_by_path(file_path=file_path)


def add_person(
    face_embedding: Any = None,
    face_thumbnail_path: Optional[str] = None,
    name: Optional[str] = None,
) -> int:
    """Module-level helper to create a person entry (defaults to 'Person 1', 'Person 2', etc.)."""
    return db.add_person(
        face_embedding=face_embedding,
        face_thumbnail_path=face_thumbnail_path,
        name=name,
    )


def get_person(person_id: int, include_embedding: bool = True) -> Optional[Dict[str, Any]]:
    """Module-level helper to fetch a person entry by ID."""
    return db.get_person(person_id=person_id, include_embedding=include_embedding)


def count_people(active_only: bool = True) -> int:
    """Module-level helper returning total person clusters."""
    return db.count_people(active_only=active_only)


def get_all_people(
    include_embeddings: bool = True,
    limit: Optional[int] = None,
    offset: int = 0,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    """Module-level helper to fetch all people with their document counts."""
    return db.get_all_people(
        include_embeddings=include_embeddings,
        limit=limit,
        offset=offset,
        active_only=active_only,
    )


def update_person(
    person_id: int,
    name: Optional[str] = None,
    face_thumbnail_path: Optional[str] = None,
    face_embedding: Any = None,
) -> bool:
    """Module-level helper to update a person's name, thumbnail, or embedding."""
    return db.update_person(
        person_id=person_id,
        name=name,
        face_thumbnail_path=face_thumbnail_path,
        face_embedding=face_embedding,
    )


def rename_person(person_id: int, new_name: str) -> bool:
    """Module-level helper to rename a person."""
    return db.rename_person(person_id=person_id, new_name=new_name)


def merge_people(source_person_id: int, target_person_id: int) -> bool:
    """Module-level helper to merge two people entries."""
    return db.merge_people(source_person_id=source_person_id, target_person_id=target_person_id)


def update_file_category(file_path: Union[str, Path], new_category: str) -> bool:
    """Module-level helper to update/transfer a file's category in SQLite."""
    return db.update_file_category(file_path=file_path, new_category=new_category)


def update_file_path(old_file_path: Union[str, Path], new_file_path: Union[str, Path]) -> bool:
    """Module-level helper to update a file's filesystem path in SQLite."""
    return db.update_file_path(old_file_path=old_file_path, new_file_path=new_file_path)


def ignore_file(file_path: Union[str, Path]) -> None:
    """Module-level helper to add a file path to `ignored_files`."""
    db.ignore_file(file_path=file_path)


def is_file_ignored(file_path: Union[str, Path]) -> bool:
    """Module-level helper to check if a file path is ignored."""
    return db.is_file_ignored(file_path=file_path)


def get_ignored_files_set() -> Set[str]:
    """Module-level helper to get all ignored paths."""
    return db.get_ignored_files_set()


def remove_file_record(file_path: Union[str, Path], add_to_ignored: bool = True) -> bool:
    """Module-level helper to remove a file from DB and optionally ignore it."""
    return db.remove_file_record(file_path=file_path, add_to_ignored=add_to_ignored)


def clear_category_documents(category: str) -> int:
    """Module-level helper to delete all DB records belonging to a category."""
    return db.clear_category_documents(category=category)


def delete_person(person_id: int) -> bool:
    """Module-level helper to delete a person entry."""
    return db.delete_person(person_id=person_id)


if __name__ == "__main__":
    init_db()
    print(f"Initialized DocuVault AI database at: {DEFAULT_DB_PATH}")
    print(f"Summary stats: {get_summary_stats()}")
