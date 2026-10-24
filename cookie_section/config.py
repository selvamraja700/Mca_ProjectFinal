"""
config.py — Central configuration for cookie_section
=====================================================

All tuneable values live here. Environment variables override defaults
so you can adjust behaviour without editing source files:

    set FLASK_SECRET_KEY=my-secret          (Windows cmd)
    $env:FLASK_SECRET_KEY="my-secret"       (PowerShell)

Important
---------
SECRET_KEY must be set to a stable, random value in production (or even
for a local project if you care about cookie security across restarts).
When it changes all existing sessions become invalid — which is fine for
a local college project, but it means the browser will lose its stored
UI state after each server restart unless you persist the key.

Recommendation: run  python -c "import secrets; print(secrets.token_hex(32))"
once, copy the output, and put it in a .env file or set it as an env-var.
"""

import os
import secrets

# ---------------------------------------------------------------------------
# Secret key
# ---------------------------------------------------------------------------
# The key signs Flask session cookies; keep it constant across restarts
# so that cookies issued before a restart are still valid.
# If the env var is absent we generate a random key for this process
# lifetime (cookies will be invalidated on next restart — acceptable for
# development, but noted as a limitation).
_DEFAULT_KEY_WARNING = (
    "WARNING [cookie_section]: FLASK_SECRET_KEY is not set. "
    "A random key will be generated each startup; "
    "browser session cookies will be invalidated on every server restart. "
    "Set the env var to a fixed value to avoid this."
)

SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "")
if not SECRET_KEY:
    import sys
    print(_DEFAULT_KEY_WARNING, file=sys.stderr)
    SECRET_KEY = secrets.token_hex(32)

# ---------------------------------------------------------------------------
# Cookie / session behaviour
# ---------------------------------------------------------------------------

# Session lifetime (seconds).  Default: 8 hours — covers a full working day.
SESSION_LIFETIME_SECONDS: int = int(
    os.environ.get("SESSION_LIFETIME_SECONDS", str(8 * 60 * 60))
)

# Whether the browser keeps the cookie after closing the window.
# True  → cookie survives browser restart (persistent cookie).
# False → cookie is deleted when the browser window closes (session cookie).
SESSION_PERMANENT: bool = os.environ.get("SESSION_PERMANENT", "true").lower() in (
    "1", "true", "yes"
)

# Cookie name that Flask uses for the session.
SESSION_COOKIE_NAME: str = os.environ.get(
    "SESSION_COOKIE_NAME", "forensics_session"
)

# SameSite attribute for the session cookie.
# "Lax" is appropriate for a local app accessed via 127.0.0.1.
SESSION_COOKIE_SAMESITE: str = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")

# Secure attribute — set to True only when serving over HTTPS.
# For 127.0.0.1 plain HTTP this must be False.
SESSION_COOKIE_SECURE: bool = os.environ.get(
    "SESSION_COOKIE_SECURE", "false"
).lower() in ("1", "true", "yes")

# HttpOnly — prevents JavaScript from reading the session cookie.
SESSION_COOKIE_HTTPONLY: bool = True

# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

# Default TTL (seconds) for cached prediction results.
# 30 minutes: long enough to be useful within a session; short enough that
# users re-uploading a modified image get fresh results.
CACHE_PREDICTION_TTL_SECONDS: int = int(
    os.environ.get("CACHE_PREDICTION_TTL_SECONDS", str(30 * 60))
)

# Default TTL for static/benchmark data (effectively permanent per-process).
CACHE_STATIC_TTL_SECONDS: int = int(
    os.environ.get("CACHE_STATIC_TTL_SECONDS", str(24 * 60 * 60))
)

# Maximum number of prediction results to keep in-memory at once.
# Old entries are evicted (LRU-style) when this limit is reached.
CACHE_MAX_PREDICTION_ENTRIES: int = int(
    os.environ.get("CACHE_MAX_PREDICTION_ENTRIES", "50")
)

# ---------------------------------------------------------------------------
# Server-side session store
# ---------------------------------------------------------------------------

# Directory where server-side session files are written.
# Defaults to <project_root>/.session_store — kept out of the backend dir.
import pathlib as _pathlib

_HERE = _pathlib.Path(__file__).parent          # cookie_section/
_PROJECT_ROOT = _HERE.parent                    # project root

SESSION_STORE_DIR: str = os.environ.get(
    "SESSION_STORE_DIR",
    str(_PROJECT_ROOT / ".session_store"),
)

# Server-side session TTL (seconds).  Must be >= SESSION_LIFETIME_SECONDS.
SERVER_SESSION_TTL_SECONDS: int = int(
    os.environ.get("SERVER_SESSION_TTL_SECONDS", str(SESSION_LIFETIME_SECONDS + 3600))
)
