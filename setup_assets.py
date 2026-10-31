"""
DocuVault AI - Local Offline Asset Downloader & Setup Utility (setup_assets.py)

Downloads and verifies all frontend UI dependencies (Tailwind CSS runtime,
Lucide Icons JS bundle, Cropper.js, and FontAwesome 6 CSS + Webfonts) into `ui/assets/`
so that the application runs 100% offline with zero internet dependency.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("docuvault.assets")

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
ASSETS_DIR = UI_DIR / "assets"
WEBFONTS_DIR = ASSETS_DIR / "webfonts"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

ASSET_MANIFEST: Dict[str, Tuple[Path, str]] = {
    "tailwind.js": (
        ASSETS_DIR / "tailwind.js",
        "https://cdn.tailwindcss.com",
    ),
    "lucide.js": (
        ASSETS_DIR / "lucide.js",
        "https://unpkg.com/lucide@latest/dist/umd/lucide.min.js",
    ),
    "fontawesome.min.css": (
        ASSETS_DIR / "fontawesome.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
    ),
    "cropper.min.js": (
        ASSETS_DIR / "cropper.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js",
    ),
    "cropper.min.css": (
        ASSETS_DIR / "cropper.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css",
    ),
    "fa-solid-900.woff2": (
        WEBFONTS_DIR / "fa-solid-900.woff2",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.woff2",
    ),
    "fa-solid-900.ttf": (
        WEBFONTS_DIR / "fa-solid-900.ttf",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-solid-900.ttf",
    ),
    "fa-regular-400.woff2": (
        WEBFONTS_DIR / "fa-regular-400.woff2",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-regular-400.woff2",
    ),
    "fa-regular-400.ttf": (
        WEBFONTS_DIR / "fa-regular-400.ttf",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-regular-400.ttf",
    ),
    "fa-brands-400.woff2": (
        WEBFONTS_DIR / "fa-brands-400.woff2",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-brands-400.woff2",
    ),
    "fa-brands-400.ttf": (
        WEBFONTS_DIR / "fa-brands-400.ttf",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-brands-400.ttf",
    ),
    "fa-v4compatibility.woff2": (
        WEBFONTS_DIR / "fa-v4compatibility.woff2",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-v4compatibility.woff2",
    ),
    "fa-v4compatibility.ttf": (
        WEBFONTS_DIR / "fa-v4compatibility.ttf",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/webfonts/fa-v4compatibility.ttf",
    ),
}

ASSET_ALIASES: List[Tuple[Path, Path]] = [
    (ASSETS_DIR / "tailwind.js", ASSETS_DIR / "tailwind.min.js"),
    (ASSETS_DIR / "lucide.js", ASSETS_DIR / "lucide.min.js"),
    (ASSETS_DIR / "fontawesome.min.css", ASSETS_DIR / "fa.all.min.css"),
]

_ASSETS_VERIFIED_IN_PROCESS = False


def _is_valid_asset_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 100
    except OSError:
        return False


def _patch_fontawesome_css(css_path: Path) -> None:
    """Ensure FontAwesome CSS references `webfonts/` relative to `ui/assets/`."""
    if not css_path.is_file():
        return
    try:
        content = css_path.read_text(encoding="utf-8", errors="ignore")
        patched = content.replace("../webfonts/", "webfonts/")
        if patched != content:
            css_path.write_text(patched, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not patch FontAwesome CSS font paths: %s", exc)


def ensure_local_assets(force_download: bool = False) -> bool:
    """
    Ensure all frontend assets exist in `ui/assets/` (and `ui/assets/webfonts/`).
    Fast-returns in O(1) once verified locally; downloads missing files with a short
    timeout and aborts remaining network attempts immediately if the host is offline.
    """
    global _ASSETS_VERIFIED_IN_PROCESS

    if not force_download and _ASSETS_VERIFIED_IN_PROCESS:
        return True

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    WEBFONTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fast O(1) check when all manifest assets + aliases already exist and no legacy files remain
    legacy_candidates = [
        UI_DIR / "tailwind.min.js",
        UI_DIR / "lucide.min.js",
        UI_DIR / "fa.all.min.css",
        UI_DIR / "svg_icons.js",
        UI_DIR / "webfonts",
    ]
    if (
        not force_download
        and not any(p.exists() for p in legacy_candidates)
        and all(_is_valid_asset_file(target) for target, _ in ASSET_MANIFEST.values())
        and all(_is_valid_asset_file(alias) for _, alias in ASSET_ALIASES)
    ):
        _ASSETS_VERIFIED_IN_PROCESS = True
        return True

    # Migrate any previously downloaded root `ui/` files into `ui/assets/` first
    legacy_migrations = [
        (UI_DIR / "tailwind.min.js", ASSETS_DIR / "tailwind.js"),
        (UI_DIR / "lucide.min.js", ASSETS_DIR / "lucide.js"),
        (UI_DIR / "fa.all.min.css", ASSETS_DIR / "fontawesome.min.css"),
        (UI_DIR / "svg_icons.js", ASSETS_DIR / "svg_icons.js"),
    ]
    for src, dst in legacy_migrations:
        if _is_valid_asset_file(src):
            if not _is_valid_asset_file(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass
            try:
                src.unlink()
            except OSError:
                pass

    legacy_webfonts_dir = UI_DIR / "webfonts"
    if legacy_webfonts_dir.is_dir():
        for wf in legacy_webfonts_dir.iterdir():
            if wf.is_file():
                target_wf = WEBFONTS_DIR / wf.name
                if not _is_valid_asset_file(target_wf):
                    try:
                        shutil.copy2(wf, target_wf)
                    except OSError:
                        pass
        try:
            shutil.rmtree(legacy_webfonts_dir, ignore_errors=True)
        except OSError:
            pass

    all_ready = True
    network_unreachable = False
    for name, (target_path, url) in ASSET_MANIFEST.items():
        if not force_download and _is_valid_asset_file(target_path):
            continue

        if network_unreachable:
            if not _is_valid_asset_file(target_path):
                all_ready = False
            continue

        logger.info("Downloading local UI asset: %s ...", name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = resp.read()
            if data and len(data) > 100:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(data)
                logger.info("Saved %s (%d bytes)", target_path.relative_to(BASE_DIR), len(data))
            else:
                logger.warning("Downloaded empty/small payload for %s", name)
                all_ready = False
        except urllib.error.HTTPError as exc:
            logger.warning("HTTP error downloading %s (%s): %s", name, url, exc)
            if not _is_valid_asset_file(target_path):
                all_ready = False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Offline or network unreachable while fetching %s: %s", name, exc)
            network_unreachable = True
            if not _is_valid_asset_file(target_path):
                all_ready = False
        except Exception as exc:
            logger.warning("Failed to download %s (%s): %s", name, url, exc)
            if not _is_valid_asset_file(target_path):
                all_ready = False

    # Fix relative webfont paths in fontawesome.min.css so `url(webfonts/...)` resolves inside `/assets/`
    _patch_fontawesome_css(ASSETS_DIR / "fontawesome.min.css")

    # Maintain convenient filename aliases inside `ui/assets/`
    for primary, alias in ASSET_ALIASES:
        if _is_valid_asset_file(primary) and (not _is_valid_asset_file(alias) or force_download):
            try:
                shutil.copy2(primary, alias)
            except OSError:
                pass

    # Synchronize modular UI folders (`ui/css` and `ui/js`)
    css_dir = UI_DIR / "css"
    js_dir = UI_DIR / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    for css_file in ASSETS_DIR.glob("*.css"):
        try:
            shutil.copy2(css_file, css_dir / css_file.name)
        except OSError:
            pass
    for js_file in ASSETS_DIR.glob("*.js"):
        try:
            shutil.copy2(js_file, js_dir / js_file.name)
        except OSError:
            pass

    if all_ready:
        _ASSETS_VERIFIED_IN_PROCESS = True
    return all_ready


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ok = ensure_local_assets(force_download=False)
    if ok:
        print("All local offline UI assets are verified and ready in ui/assets/!")
    else:
        print("Warning: Some assets could not be downloaded. Check network connection.")
