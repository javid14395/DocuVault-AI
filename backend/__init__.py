"""
DocuVault AI - Backend Core Package
"""

from . import database
from . import face_engine
from . import scanner
from . import processor
from . import studio
from . import watcher

__all__ = [
    "database",
    "face_engine",
    "scanner",
    "processor",
    "studio",
    "watcher",
]
