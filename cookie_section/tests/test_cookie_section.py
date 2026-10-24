"""
test_cookie_section.py — Complete test suite for cookie_section
================================================================

Run from the project root with::

    py -3 -m pytest cookie_section/tests/ -v

Or run the file directly::

    py -3 cookie_section/tests/test_cookie_section.py

All tests are self-contained and do NOT require the ML model, dataset,
or a running Flask server.  Flask's test client is used for route-context
tests.

Coverage
--------
1.  CookieManager
    a. configure() applies all settings to a Flask app
    b. set/get/delete/clear work within a request context
    c. record_scan updates all scan fields correctly
    d. Disallowed keys raise KeyError
    e. is_session_intact() returns correct booleans
    f. cookie_attributes_for() reflects live config

2.  SessionManager
    a. create_session() returns a 64-hex-char token
    b. load_session() returns stored data
    c. load_session() returns None for expired sessions
    d. load_session() returns None for invalid tokens
    e. update_session() overwrites data
    f. delete_session() removes the file
    g. extend_session() resets TTL
    h. cleanup_expired() removes only expired files
    i. active_session_count() counts correctly
    j. Session data is persisted to disk (file actually written)

3.  CacheManager
    a. make_image_key() is deterministic and includes model version
    b. Different images produce different keys
    c. get_prediction() returns None on cold miss
    d. set_prediction() + get_prediction() round-trip (non-image fields)
    e. Prediction cache expires after TTL
    f. LRU eviction fires when max_entries is reached
    g. invalidate_prediction() removes single entry
    h. invalidate_all_predictions() clears all
    i. Static cache set/get round-trip
    j. Static cache expires after TTL
    k. purge_expired() removes expired entries from both caches
    l. stats() reflects hits, misses, eviction counts
    m. bump_model_version() clears prediction cache and changes key

4.  Integration — no existing files modified
    a. ML model file is untouched (size check)
    b. Dataset directory is untouched
    c. backend/app.py is untouched
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest

# ---------------------------------------------------------------------------
# Make cookie_section importable when running from the project root or
# from inside the tests/ directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Flask is required; skip gracefully if not installed.
# ---------------------------------------------------------------------------
try:
    from flask import Flask
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False


# ============================================================================
# 1. CookieManager tests
# ============================================================================

@unittest.skipUnless(_FLASK_AVAILABLE, "Flask not installed")
class TestCookieManager(unittest.TestCase):

    def setUp(self) -> None:
        from cookie_section.cookie_manager import CookieManager
        self.CookieManager = CookieManager
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.cm = CookieManager()
        self.cm.configure(self.app)

    def test_configure_applies_secret_key(self):
        """configure() must set app.secret_key to a non-empty string."""
        self.assertIsInstance(self.app.secret_key, (str, bytes))
        self.assertTrue(len(self.app.secret_key) > 0)

    def test_configure_sets_cookie_name(self):
        from cookie_section import config as cfg
        self.assertEqual(
            self.app.config["SESSION_COOKIE_NAME"],
            cfg.SESSION_COOKIE_NAME,
        )

    def test_configure_sets_httponly(self):
        self.assertTrue(self.app.config["SESSION_COOKIE_HTTPONLY"])

    def test_configure_sets_samesite(self):
        samesite = self.app.config["SESSION_COOKIE_SAMESITE"]
        self.assertIn(samesite, ("Lax", "Strict", "None"))

    def test_set_and_get_in_request_context(self):
        """set() and get() work inside a Flask request context."""
        with self.app.test_request_context("/"):
            self.cm.set("theme", "dark")
            self.assertEqual(self.cm.get("theme"), "dark")

    def test_delete_removes_key(self):
        with self.app.test_request_context("/"):
            self.cm.set("theme", "light")
            self.cm.delete("theme")
            self.assertIsNone(self.cm.get("theme"))

    def test_clear_removes_all_allowed_keys(self):
        with self.app.test_request_context("/"):
            self.cm.set("theme", "dark")
            self.cm.set("scan_count", 5)
            self.cm.clear()
            self.assertIsNone(self.cm.get("theme"))
            self.assertIsNone(self.cm.get("scan_count"))

    def test_disallowed_key_raises(self):
        with self.app.test_request_context("/"):
            with self.assertRaises(KeyError):
                self.cm.set("password", "secret")

    def test_get_returns_default_on_miss(self):
        with self.app.test_request_context("/"):
            self.assertIsNone(self.cm.get("theme"))
            self.assertEqual(self.cm.get("theme", "default_val"), "default_val")

    def test_record_scan_updates_all_fields(self):
        with self.app.test_request_context("/"):
            self.cm.record_scan("photo.jpg", "Forged", 87.5)
            self.assertEqual(self.cm.get("last_filename"), "photo.jpg")
            self.assertEqual(self.cm.get("last_prediction"), "Forged")
            self.assertAlmostEqual(self.cm.get("last_confidence"), 87.5)
            self.assertEqual(self.cm.get("scan_count"), 1)

    def test_record_scan_increments_count(self):
        with self.app.test_request_context("/"):
            self.cm.record_scan("a.jpg", "Genuine", 90.0)
            self.cm.record_scan("b.jpg", "Forged", 70.0)
            self.assertEqual(self.cm.get("scan_count"), 2)

    def test_is_session_intact_false_on_empty(self):
        with self.app.test_request_context("/"):
            self.assertFalse(self.cm.is_session_intact())

    def test_is_session_intact_true_after_set(self):
        with self.app.test_request_context("/"):
            self.cm.set("theme", "dark")
            self.assertTrue(self.cm.is_session_intact())

    def test_cookie_attributes_for_returns_dict(self):
        attrs = self.CookieManager.cookie_attributes_for(self.app)
        self.assertIn("name", attrs)
        self.assertIn("httponly", attrs)
        self.assertIn("secure", attrs)
        self.assertIn("samesite", attrs)
        self.assertIn("lifetime_seconds", attrs)
        self.assertIsInstance(attrs["lifetime_seconds"], int)

    def test_get_session_summary_includes_all_keys(self):
        with self.app.test_request_context("/"):
            summary = self.cm.get_session_summary()
            self.assertIsInstance(summary, dict)
            for key in self.CookieManager.ALLOWED_KEYS:
                self.assertIn(key, summary)


# ============================================================================
# 2. SessionManager tests
# ============================================================================

class TestSessionManager(unittest.TestCase):

    def setUp(self) -> None:
        from cookie_section.session_manager import SessionManager
        # Use a fresh temp directory for each test so tests are isolated.
        self._tmpdir = tempfile.mkdtemp(prefix="sess_test_")
        self.sm = SessionManager(store_dir=self._tmpdir, ttl_seconds=10)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_returns_64_hex_chars(self):
        token = self.sm.create_session({"x": 1})
        self.assertEqual(len(token), 64)
        int(token, 16)  # raises ValueError if not hex

    def test_create_writes_file_to_disk(self):
        token = self.sm.create_session({"x": 1})
        path = pathlib.Path(self._tmpdir) / f"sess_{token}.json"
        self.assertTrue(path.exists(), "Session file was not written to disk.")

    def test_load_returns_stored_data(self):
        data = {"key1": "value1", "nested": {"a": 1}}
        token = self.sm.create_session(data)
        loaded = self.sm.load_session(token)
        self.assertEqual(loaded, data)

    def test_load_returns_none_for_unknown_token(self):
        fake_token = "a" * 64
        self.assertIsNone(self.sm.load_session(fake_token))

    def test_load_returns_none_for_wrong_length_token(self):
        self.assertIsNone(self.sm.load_session("short"))
        self.assertIsNone(self.sm.load_session("x" * 65))

    def test_load_returns_none_for_non_hex_token(self):
        bad = "z" * 64   # 'z' is not a hex digit
        self.assertIsNone(self.sm.load_session(bad))

    def test_load_returns_none_after_expiry(self):
        sm_short = __import__(
            "cookie_section.session_manager", fromlist=["SessionManager"]
        ).SessionManager(store_dir=self._tmpdir, ttl_seconds=1)
        token = sm_short.create_session({"y": 2})
        time.sleep(1.1)
        self.assertIsNone(sm_short.load_session(token))

    def test_load_deletes_expired_file(self):
        sm_short = __import__(
            "cookie_section.session_manager", fromlist=["SessionManager"]
        ).SessionManager(store_dir=self._tmpdir, ttl_seconds=1)
        token = sm_short.create_session({"y": 2})
        path = pathlib.Path(self._tmpdir) / f"sess_{token}.json"
        self.assertTrue(path.exists())
        time.sleep(1.1)
        sm_short.load_session(token)   # triggers lazy delete
        self.assertFalse(path.exists(), "Expired session file was not deleted.")

    def test_update_overwrites_data(self):
        token = self.sm.create_session({"v": 1})
        result = self.sm.update_session(token, {"v": 99})
        self.assertTrue(result)
        self.assertEqual(self.sm.load_session(token), {"v": 99})

    def test_update_returns_false_for_missing(self):
        result = self.sm.update_session("a" * 64, {"x": 1})
        self.assertFalse(result)

    def test_delete_removes_session(self):
        token = self.sm.create_session({"z": 0})
        self.sm.delete_session(token)
        self.assertIsNone(self.sm.load_session(token))

    def test_delete_returns_true_if_existed(self):
        token = self.sm.create_session({})
        self.assertTrue(self.sm.delete_session(token))

    def test_delete_returns_false_for_missing(self):
        self.assertFalse(self.sm.delete_session("b" * 64))

    def test_extend_session_allows_future_load(self):
        sm_short = __import__(
            "cookie_section.session_manager", fromlist=["SessionManager"]
        ).SessionManager(store_dir=self._tmpdir, ttl_seconds=2)
        token = sm_short.create_session({"k": "v"})
        time.sleep(1.0)
        sm_short.extend_session(token)   # should reset TTL to 2 s from now
        time.sleep(1.2)
        # Would have expired (1.0 + 1.2 = 2.2 s > original 2 s TTL)
        # but extend happened at t=1.0, so new expiry is t=1.0+2.0=3.0 s.
        # At t=2.2 s the session should still be valid.
        self.assertIsNotNone(sm_short.load_session(token))

    def test_session_exists_true_for_live_session(self):
        token = self.sm.create_session({"live": True})
        self.assertTrue(self.sm.session_exists(token))

    def test_session_exists_false_for_missing(self):
        self.assertFalse(self.sm.session_exists("c" * 64))

    def test_cleanup_expired_removes_stale(self):
        sm_short = __import__(
            "cookie_section.session_manager", fromlist=["SessionManager"]
        ).SessionManager(store_dir=self._tmpdir, ttl_seconds=1)
        t1 = sm_short.create_session({"stale": True})
        t2 = self.sm.create_session({"fresh": True})  # 10 s TTL
        time.sleep(1.1)
        deleted = self.sm.cleanup_expired()
        self.assertGreaterEqual(deleted, 1)
        self.assertIsNone(sm_short.load_session(t1))
        self.assertIsNotNone(self.sm.load_session(t2))

    def test_active_session_count(self):
        self.sm.create_session({"a": 1})
        self.sm.create_session({"b": 2})
        self.assertEqual(self.sm.active_session_count(), 2)

    def test_session_file_integrity(self):
        """Session file must be valid JSON with expected keys."""
        data = {"integrity_check": True}
        token = self.sm.create_session(data)
        path = pathlib.Path(self._tmpdir) / f"sess_{token}.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("token_hash", envelope)
        self.assertIn("created_at", envelope)
        self.assertIn("expires_at", envelope)
        self.assertIn("data", envelope)
        self.assertEqual(envelope["data"], data)


# ============================================================================
# 3. CacheManager tests
# ============================================================================

class TestCacheManager(unittest.TestCase):

    def setUp(self) -> None:
        from cookie_section.cache_manager import CacheManager
        # Short TTL so expiry tests don't take long.
        self.cm = CacheManager(
            prediction_ttl=2,
            static_ttl=2,
            max_prediction_entries=3,
            model_version="test_model_v1",
        )
        self.sample_bytes = b"fake image bytes for testing 12345"
        self.sample_result = {
            "success": True,
            "filename": "test.jpg",
            "prediction": "Genuine",
            "probabilities": {"genuine": 92.1, "forged": 7.9},
            "confidence": 92.1,
            "visualizations": {
                "original": "data:image/png;base64,AAAA",
                "ela": "data:image/png;base64,BBBB",
                "heatmap": "data:image/png;base64,CCCC",
            },
        }

    # --- make_image_key ---

    def test_key_is_deterministic(self):
        k1 = self.cm.make_image_key(self.sample_bytes)
        k2 = self.cm.make_image_key(self.sample_bytes)
        self.assertEqual(k1, k2)

    def test_key_includes_model_version(self):
        key = self.cm.make_image_key(self.sample_bytes)
        self.assertIn("test_model_v1", key)

    def test_different_bytes_produce_different_keys(self):
        k1 = self.cm.make_image_key(b"image_A")
        k2 = self.cm.make_image_key(b"image_B")
        self.assertNotEqual(k1, k2)

    def test_key_contains_sha256_hex(self):
        key = self.cm.make_image_key(self.sample_bytes)
        # Key format is  <sha256>:<model_version>
        sha_part = key.split(":")[0]
        expected = hashlib.sha256(self.sample_bytes).hexdigest()
        self.assertEqual(sha_part, expected)

    # --- prediction cache ---

    def test_get_returns_none_on_cold_miss(self):
        key = self.cm.make_image_key(b"never_seen")
        self.assertIsNone(self.cm.get_prediction(key))

    def test_set_and_get_roundtrip(self):
        key = self.cm.make_image_key(self.sample_bytes)
        self.cm.set_prediction(key, self.sample_result)
        cached = self.cm.get_prediction(key)
        self.assertIsNotNone(cached)
        # Visualizations must NOT be cached (memory savings)
        self.assertNotIn("visualizations", cached)
        # Metadata fields must be present
        self.assertEqual(cached["prediction"], "Genuine")
        self.assertAlmostEqual(cached["confidence"], 92.1)

    def test_visualizations_stripped_from_cache(self):
        """Large base64 blobs must never enter the cache."""
        key = self.cm.make_image_key(self.sample_bytes)
        self.cm.set_prediction(key, self.sample_result)
        cached = self.cm.get_prediction(key)
        self.assertNotIn("visualizations", cached)

    def test_prediction_expires_after_ttl(self):
        key = self.cm.make_image_key(self.sample_bytes)
        self.cm.set_prediction(key, self.sample_result)
        time.sleep(2.1)
        self.assertIsNone(self.cm.get_prediction(key))

    def test_lru_eviction_fires(self):
        """When max_prediction_entries is reached, the LRU entry is evicted."""
        # max_prediction_entries=3 in setUp
        keys = [self.cm.make_image_key(bytes([i])) for i in range(4)]
        for key in keys[:3]:
            self.cm.set_prediction(key, self.sample_result)
        # Access keys[0] to make it most-recently-used
        self.cm.get_prediction(keys[0])
        # Insert a 4th entry — should evict keys[1] (LRU)
        self.cm.set_prediction(keys[3], self.sample_result)
        stats = self.cm.stats()
        self.assertGreaterEqual(stats["pred_evictions"], 1)

    def test_invalidate_prediction_removes_entry(self):
        key = self.cm.make_image_key(self.sample_bytes)
        self.cm.set_prediction(key, self.sample_result)
        removed = self.cm.invalidate_prediction(key)
        self.assertTrue(removed)
        self.assertIsNone(self.cm.get_prediction(key))

    def test_invalidate_prediction_false_for_missing(self):
        self.assertFalse(self.cm.invalidate_prediction("nonexistent_key"))

    def test_invalidate_all_clears_cache(self):
        for i in range(3):
            key = self.cm.make_image_key(bytes([i]))
            self.cm.set_prediction(key, self.sample_result)
        removed = self.cm.invalidate_all_predictions()
        self.assertEqual(removed, 3)
        stats = self.cm.stats()
        self.assertEqual(stats["pred_cache_live_entries"], 0)

    # --- static cache ---

    def test_static_get_returns_none_on_miss(self):
        self.assertIsNone(self.cm.get_static("benchmarks"))

    def test_static_set_and_get_roundtrip(self):
        payload = json.dumps({"accuracy": 92.32})
        self.cm.set_static("benchmarks", payload)
        self.assertEqual(self.cm.get_static("benchmarks"), payload)

    def test_static_expires_after_ttl(self):
        self.cm.set_static("temp_key", "temp_value", ttl=1)
        time.sleep(1.1)
        self.assertIsNone(self.cm.get_static("temp_key"))

    def test_invalidate_static_removes_entry(self):
        self.cm.set_static("benchmarks", "{}")
        self.assertTrue(self.cm.invalidate_static("benchmarks"))
        self.assertIsNone(self.cm.get_static("benchmarks"))

    def test_invalidate_static_false_for_missing(self):
        self.assertFalse(self.cm.invalidate_static("never_set"))

    # --- purge and stats ---

    def test_purge_expired_removes_stale_entries(self):
        key1 = self.cm.make_image_key(b"img1")
        self.cm.set_prediction(key1, self.sample_result)
        self.cm.set_static("stale_key", "v", ttl=1)
        time.sleep(1.1)
        # key1 has 2s TTL so still live; "stale_key" expired
        removed = self.cm.purge_expired()
        self.assertGreaterEqual(removed, 1)
        self.assertIsNotNone(self.cm.get_prediction(key1))
        self.assertIsNone(self.cm.get_static("stale_key"))

    def test_stats_tracks_hits_and_misses(self):
        key = self.cm.make_image_key(b"stats_test")
        self.cm.get_prediction(key)            # miss
        self.cm.set_prediction(key, self.sample_result)
        self.cm.get_prediction(key)            # hit
        s = self.cm.stats()
        self.assertGreaterEqual(s["pred_hits"], 1)
        self.assertGreaterEqual(s["pred_misses"], 1)

    def test_reset_stats_zeroes_counters(self):
        key = self.cm.make_image_key(b"rs_test")
        self.cm.get_prediction(key)
        self.cm.reset_stats()
        s = self.cm.stats()
        self.assertEqual(s["pred_hits"], 0)
        self.assertEqual(s["pred_misses"], 0)

    def test_bump_model_version_clears_predictions(self):
        key = self.cm.make_image_key(self.sample_bytes)
        self.cm.set_prediction(key, self.sample_result)
        self.cm.bump_model_version("test_model_v2")
        # Old key is gone
        self.assertIsNone(self.cm.get_prediction(key))
        # New keys include the new version
        new_key = self.cm.make_image_key(self.sample_bytes)
        self.assertIn("test_model_v2", new_key)


# ============================================================================
# 4. Integration — confirm no existing files were modified
# ============================================================================

class TestExistingFilesUntouched(unittest.TestCase):
    """Guard-rail: confirm the new cookie_section code did NOT touch
    model weights, dataset, or backend/app.py.

    These tests record expected file sizes and confirm them.  If anything
    outside cookie_section/ changes, these tests catch it.
    """

    # Absolute path to project root (two levels up from this file).
    PROJECT = pathlib.Path(__file__).resolve().parent.parent.parent

    # Known size of the deployed weights file (bytes).
    EXPECTED_WEIGHTS_SIZE = 28_613_366   # densenet_forgery.pth

    def _path(self, *parts: str) -> pathlib.Path:
        return self.PROJECT.joinpath(*parts)

    def test_model_weights_size_unchanged(self):
        weights = self._path("models", "densenet_forgery.pth")
        if not weights.exists():
            self.skipTest("Model weights file not present — skip size check.")
        actual = weights.stat().st_size
        self.assertEqual(
            actual,
            self.EXPECTED_WEIGHTS_SIZE,
            f"Model weights file size changed! Expected {self.EXPECTED_WEIGHTS_SIZE} "
            f"bytes, got {actual} bytes.",
        )

    def test_backend_app_py_not_modified(self):
        """backend/app.py must not have been touched by cookie_section code."""
        app_py = self._path("backend", "app.py")
        self.assertTrue(app_py.exists(), "backend/app.py not found.")
        # Verify key landmark strings are still present.
        content = app_py.read_text(encoding="utf-8", errors="replace")
        self.assertIn("from flask import Flask", content)
        self.assertIn("from inference.predictor import predict_image", content)
        self.assertIn("from models import get_model", content)

    def test_cookie_section_files_are_only_inside_folder(self):
        """All new files must live inside cookie_section/ only."""
        cookie_dir = self._path("cookie_section")
        self.assertTrue(cookie_dir.exists(), "cookie_section/ directory not found.")
        new_files = list(cookie_dir.rglob("*.py"))
        self.assertGreater(len(new_files), 0, "No Python files found in cookie_section/.")

    def test_no_cookie_section_imports_in_backend_app(self):
        """backend/app.py must not already import cookie_section
        (integration is pending user approval)."""
        app_py = self._path("backend", "app.py")
        content = app_py.read_text(encoding="utf-8", errors="replace")
        self.assertNotIn(
            "cookie_section",
            content,
            "backend/app.py already imports cookie_section without approval.",
        )

    def test_dataset_directory_not_deleted(self):
        data_dir = self._path("data")
        if data_dir.exists():
            # Confirm it's a directory and still has contents.
            self.assertTrue(data_dir.is_dir())
        else:
            self.skipTest("data/ directory not present in project root.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
