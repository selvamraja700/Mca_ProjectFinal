"""
cookie_section — Session, Cookie, and Cache utilities
======================================================
For the ML-Based Image Forgery Detection Flask application.

Framework: Flask (backend/app.py)
Authentication: None (single-user local tool — no login required)

Provides:
  - CookieManager   : Flask session cookie configuration helpers
  - SessionManager  : Server-side session data store (file-backed)
  - CacheManager    : In-process TTL cache for predictions & static data
  - Config          : Central configuration for all three modules

Usage
-----
Import the public API::

    from cookie_section import CookieManager, SessionManager, CacheManager

Integration note
----------------
These utilities are ready but intentionally NOT wired into backend/app.py.
See README.md for the exact diff to apply once you approve integration.
"""

from .cookie_manager import CookieManager
from .session_manager import SessionManager
from .cache_manager import CacheManager

__all__ = ["CookieManager", "SessionManager", "CacheManager"]
__version__ = "1.0.0"
