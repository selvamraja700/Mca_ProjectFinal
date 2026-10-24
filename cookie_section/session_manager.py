from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import secrets
import threading
import time
from typing import Any

from . import config as cfg

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class SessionManager:
    """File-backed server-side session store."""

    def __init__(self, store_dir: str | None = None, ttl_seconds: int | None = None) -> None:
        self._store_dir = pathlib.Path(store_dir or cfg.SESSION_STORE_DIR)
        self._ttl = ttl_seconds if ttl_seconds is not None else cfg.SERVER_SESSION_TTL_SECONDS
        self._store_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "SessionManager initialised: store=%s, ttl=%ds",
            self._store_dir,
            self._ttl,
        )

    def create_session(self, data: dict[str, Any]) -> str:
        """Persist *data* as a new session file and return the session token."""
        token = secrets.token_hex(32)
        self._write(token, data)
        logger.debug("Session created: token=%.8s…", token)
        return token

    def load_session(self, token: str) -> dict[str, Any] | None:
        """Return the session payload for *token*, or ``None`` if expired/absent/tampered."""
        if not self._is_valid_token(token):
            logger.warning("load_session: invalid token format rejected.")
            return None

        path = self._path_for(token)
        if not path.exists():
            return None

        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("load_session: could not read %s: %s", path, exc)
            return None

        if time.time() > envelope.get("expires_at", 0):
            logger.debug("Session expired, deleting: token=%.8s…", token)
            self._delete_file(path)
            return None

        return envelope.get("data", {})

    def update_session(self, token: str, data: dict[str, Any]) -> bool:
        """Overwrite the payload for an existing *token*. Returns ``True`` on success."""
        if self.load_session(token) is None:
            return False
        self._write(token, data)
        logger.debug("Session updated: token=%.8s…", token)
        return True

    def delete_session(self, token: str) -> bool:
        """Remove the session file for *token*. Returns ``True`` if the file existed."""
        if not self._is_valid_token(token):
            return False
        path = self._path_for(token)
        if not path.exists():
            return False
        return self._delete_file(path)

    def extend_session(self, token: str) -> bool:
        """Reset the TTL on an existing session. Returns ``True`` on success."""
        data = self.load_session(token)
        if data is None:
            return False
        self._write(token, data)
        logger.debug("Session TTL extended: token=%.8s…", token)
        return True

    def session_exists(self, token: str) -> bool:
        """Return ``True`` if the session exists and is not yet expired."""
        return self.load_session(token) is not None

    def cleanup_expired(self) -> int:
        """Delete all expired session files. Returns the number of files deleted."""
        deleted = 0
        now = time.time()
        for path in self._store_dir.glob("sess_*.json"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                if now > envelope.get("expires_at", 0):
                    self._delete_file(path)
                    deleted += 1
            except (json.JSONDecodeError, OSError):
                self._delete_file(path)
                deleted += 1
        logger.info("cleanup_expired: removed %d session file(s).", deleted)
        return deleted

    def active_session_count(self) -> int:
        """Return the number of non-expired session files on disk."""
        now = time.time()
        count = 0
        for path in self._store_dir.glob("sess_*.json"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                if now <= envelope.get("expires_at", 0):
                    count += 1
            except (json.JSONDecodeError, OSError):
                pass
        return count

    def _path_for(self, token: str) -> pathlib.Path:
        return self._store_dir / f"sess_{token}.json"

    def _write(self, token: str, data: dict[str, Any]) -> None:
        """Atomically write session data to disk."""
        envelope = {
            "token_hash": hashlib.sha256(token.encode()).hexdigest(),
            "created_at": time.time(),
            "expires_at": time.time() + self._ttl,
            "data": data,
        }
        target = self._path_for(token)
        tmp = target.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
            tmp.replace(target)

    @staticmethod
    def _delete_file(path: pathlib.Path) -> bool:
        """Delete *path* and return True. Caller must confirm existence first."""
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("Could not delete session file %s: %s", path, exc)
            return False

    @staticmethod
    def _is_valid_token(token: str) -> bool:
        """Reject obviously invalid tokens before touching the filesystem."""
        if not isinstance(token, str):
            return False
        if len(token) != 64:
            return False
        try:
            int(token, 16)
            return True
        except ValueError:
            return False
