import os
import secrets

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

SESSION_LIFETIME_SECONDS: int = int(
    os.environ.get("SESSION_LIFETIME_SECONDS", str(8 * 60 * 60))
)

SESSION_PERMANENT: bool = os.environ.get("SESSION_PERMANENT", "true").lower() in (
    "1", "true", "yes"
)

SESSION_COOKIE_NAME: str = os.environ.get(
    "SESSION_COOKIE_NAME", "forensics_session"
)

SESSION_COOKIE_SAMESITE: str = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")

SESSION_COOKIE_SECURE: bool = os.environ.get(
    "SESSION_COOKIE_SECURE", "false"
).lower() in ("1", "true", "yes")

SESSION_COOKIE_HTTPONLY: bool = True

CACHE_PREDICTION_TTL_SECONDS: int = int(
    os.environ.get("CACHE_PREDICTION_TTL_SECONDS", str(30 * 60))
)

CACHE_STATIC_TTL_SECONDS: int = int(
    os.environ.get("CACHE_STATIC_TTL_SECONDS", str(24 * 60 * 60))
)

CACHE_MAX_PREDICTION_ENTRIES: int = int(
    os.environ.get("CACHE_MAX_PREDICTION_ENTRIES", "50")
)

import pathlib as _pathlib

_HERE = _pathlib.Path(__file__).parent
_PROJECT_ROOT = _HERE.parent

SESSION_STORE_DIR: str = os.environ.get(
    "SESSION_STORE_DIR",
    str(_PROJECT_ROOT / ".session_store"),
)

SERVER_SESSION_TTL_SECONDS: int = int(
    os.environ.get("SERVER_SESSION_TTL_SECONDS", str(SESSION_LIFETIME_SECONDS + 3600))
)
