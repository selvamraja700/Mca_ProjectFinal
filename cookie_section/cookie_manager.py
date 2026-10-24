from __future__ import annotations

import datetime
import logging
from typing import Any

from flask import Flask, session

from . import config as cfg

logger = logging.getLogger(__name__)


class CookieManager:
    """Configure Flask signed session cookies and provide helpers."""

    ALLOWED_KEYS: frozenset[str] = frozenset(
        {
            "theme",
            "last_filename",
            "last_prediction",
            "last_confidence",
            "scan_count",
            "session_id",
        }
    )

    def __init__(self) -> None:
        self._configured_app: Flask | None = None

    def configure(self, app: Flask) -> None:
        """Apply all session-cookie settings to *app*."""
        app.secret_key = cfg.SECRET_KEY
        app.config.update(
            SESSION_COOKIE_NAME=cfg.SESSION_COOKIE_NAME,
            SESSION_COOKIE_HTTPONLY=cfg.SESSION_COOKIE_HTTPONLY,
            SESSION_COOKIE_SECURE=cfg.SESSION_COOKIE_SECURE,
            SESSION_COOKIE_SAMESITE=cfg.SESSION_COOKIE_SAMESITE,
            PERMANENT_SESSION_LIFETIME=datetime.timedelta(
                seconds=cfg.SESSION_LIFETIME_SECONDS
            ),
            SESSION_PERMANENT=cfg.SESSION_PERMANENT,
        )
        self._configured_app = app
        logger.info(
            "CookieManager configured: cookie=%s, permanent=%s, lifetime=%ds",
            cfg.SESSION_COOKIE_NAME,
            cfg.SESSION_PERMANENT,
            cfg.SESSION_LIFETIME_SECONDS,
        )

    def set(self, key: str, value: Any) -> None:
        """Write *key* → *value* into the signed session cookie."""
        if key not in self.ALLOWED_KEYS:
            raise KeyError(
                f"'{key}' is not an allowed session key. "
                f"Allowed: {sorted(self.ALLOWED_KEYS)}"
            )
        session[key] = value
        if cfg.SESSION_PERMANENT:
            session.permanent = True

    def get(self, key: str, default: Any = None) -> Any:
        """Read *key* from the session cookie, returning *default* if absent."""
        return session.get(key, default)

    def delete(self, key: str) -> None:
        """Remove *key* from the session cookie (no-op if absent)."""
        session.pop(key, None)

    def clear(self) -> None:
        """Remove all keys written by this app from the session cookie."""
        for key in list(self.ALLOWED_KEYS):
            session.pop(key, None)

    def record_scan(self, filename: str, prediction: str, confidence: float) -> None:
        """Update scan-related session fields after a prediction."""
        self.set("last_filename", filename)
        self.set("last_prediction", prediction)
        self.set("last_confidence", round(float(confidence), 2))
        count = int(self.get("scan_count", 0)) + 1
        self.set("scan_count", count)
        logger.debug(
            "Session updated: scan #%d  file=%s  result=%s  confidence=%.2f%%",
            count,
            filename,
            prediction,
            confidence,
        )

    def get_session_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all stored session values."""
        return {key: session.get(key) for key in self.ALLOWED_KEYS}

    def is_session_intact(self) -> bool:
        """Return True if the session cookie is present and non-empty."""
        return any(key in session for key in self.ALLOWED_KEYS)

    @staticmethod
    def cookie_attributes_for(app: Flask) -> dict[str, Any]:
        """Return the current cookie attribute settings from *app*.config."""
        return {
            "name": app.config.get("SESSION_COOKIE_NAME"),
            "httponly": app.config.get("SESSION_COOKIE_HTTPONLY"),
            "secure": app.config.get("SESSION_COOKIE_SECURE"),
            "samesite": app.config.get("SESSION_COOKIE_SAMESITE"),
            "permanent": app.config.get("SESSION_PERMANENT"),
            "lifetime_seconds": int(
                app.config.get(
                    "PERMANENT_SESSION_LIFETIME",
                    datetime.timedelta(seconds=cfg.SESSION_LIFETIME_SECONDS),
                ).total_seconds()
            ),
        }
