"""
cache_manager.py — In-process TTL cache for Flask ML application
=================================================================

What is cached
--------------
1. **Prediction results** (keyed by image content hash + model version):
   The most expensive operation in each request is:
     a. ELA computation          (~50–200 ms on CPU)
     b. DenseNet forward pass    (~100–400 ms on CPU)
     c. Heatmap generation       (~20–80 ms on CPU)
   Total: 170–680 ms per image on CPU.

   If the same image file is uploaded twice in the same session (common
   during development/testing), we return the cached result immediately.
   The cache key includes a model version string so stale results from an
   older model checkpoint are automatically invalidated.

2. **Benchmark / static data** (keyed by constant string):
   ``/api/benchmarks`` returns a large static JSON blob from
   ``config.settings.PAPER_METRICS``.  While Python dict access is fast,
   JSON serialisation is not free.  Caching the serialised form eliminates
   redundant work on every benchmarks request.

What is NOT cached
------------------
* The uploaded image bytes themselves.
* User-identifiable information.
* Model weights (PyTorch handles that in memory).

Design
------
* Pure in-process dict — no Redis, no external service, no extra deps.
* Thread-safe via ``threading.Lock`` on every read and write.
* LRU eviction when the prediction cache reaches ``CACHE_MAX_PREDICTION_ENTRIES``.
* TTL: each entry stores ``expires_at = time.monotonic() + ttl``.
* Cache key for images: SHA-256 of the raw image bytes + model version string.

Usage
-----
    from cookie_section.cache_manager import CacheManager

    cache = CacheManager()

    # In predict():
    key = cache.make_image_key(image_bytes)
    result = cache.get_prediction(key)
    if result is None:
        result = run_expensive_prediction(...)
        cache.set_prediction(key, result)

    # In get_benchmarks():
    data = cache.get_static("benchmarks")
    if data is None:
        data = json.dumps(PAPER_METRICS)
        cache.set_static("benchmarks", data)
"""

from __future__ import annotations

import collections
import hashlib
import logging
import threading
import time
from typing import Any

from . import config as cfg

logger = logging.getLogger(__name__)

# Model version string — bumped manually when weights are updated.
# A mismatch between stored key and current version invalidates the cache.
_MODEL_VERSION = "densenet_forgery_v1"


class _TTLEntry:
    """A single cache entry with its expiry timestamp."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at: float = time.monotonic() + ttl

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class CacheManager:
    """
    Thread-safe in-process TTL cache with LRU eviction.

    Maintains two separate namespaces:
      * ``predictions`` — image-content-keyed prediction payloads.
      * ``static``      — long-lived data (benchmarks, health strings).
    """

    def __init__(
        self,
        prediction_ttl: int | None = None,
        static_ttl: int | None = None,
        max_prediction_entries: int | None = None,
        model_version: str | None = None,
    ) -> None:
        self._pred_ttl: int = prediction_ttl or cfg.CACHE_PREDICTION_TTL_SECONDS
        self._static_ttl: int = static_ttl or cfg.CACHE_STATIC_TTL_SECONDS
        self._max_pred: int = max_prediction_entries or cfg.CACHE_MAX_PREDICTION_ENTRIES
        self._model_version: str = model_version or _MODEL_VERSION

        # OrderedDict used as an LRU container (most-recently-used at right).
        self._pred_cache: collections.OrderedDict[str, _TTLEntry] = (
            collections.OrderedDict()
        )
        self._static_cache: dict[str, _TTLEntry] = {}

        self._lock = threading.Lock()
        self._stats = {
            "pred_hits": 0,
            "pred_misses": 0,
            "pred_evictions": 0,
            "static_hits": 0,
            "static_misses": 0,
        }
        logger.info(
            "CacheManager initialised: pred_ttl=%ds, static_ttl=%ds, "
            "max_pred=%d, model=%s",
            self._pred_ttl,
            self._static_ttl,
            self._max_pred,
            self._model_version,
        )

    # ------------------------------------------------------------------
    # Prediction cache
    # ------------------------------------------------------------------

    def make_image_key(self, image_bytes: bytes) -> str:
        """Return a stable cache key for *image_bytes* tied to the current model.

        SHA-256 of the raw bytes is collision-resistant for practical file
        sizes.  Appending the model version string ensures that the same
        image uploaded after a model update gets re-evaluated.
        """
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        return f"{content_hash}:{self._model_version}"

    def get_prediction(self, key: str) -> dict[str, Any] | None:
        """Return cached prediction for *key*, or ``None`` on miss / expiry.

        Moves the entry to the right (most-recently-used) on a hit.
        """
        with self._lock:
            entry = self._pred_cache.get(key)
            if entry is None:
                self._stats["pred_misses"] += 1
                return None
            if entry.is_expired():
                del self._pred_cache[key]
                self._stats["pred_misses"] += 1
                logger.debug("Prediction cache expired: key=%.16s…", key)
                return None
            # Move to end (LRU refresh)
            self._pred_cache.move_to_end(key)
            self._stats["pred_hits"] += 1
            logger.debug("Prediction cache HIT: key=%.16s…", key)
            return entry.value

    def set_prediction(self, key: str, result: dict[str, Any]) -> None:
        """Store a prediction *result* under *key*.

        Evicts the least-recently-used entry if the cache is full.
        Prediction results contain base64-encoded images; we strip the
        ``visualizations`` sub-dict before caching to keep memory usage
        manageable, since those can be hundreds of KB each.

        Only the non-image metadata is cached; the caller reconstructs
        visualisations only when a cache miss occurs.
        """
        # Deep-copy only the non-image fields to avoid caching large blobs.
        safe_result = {
            k: v for k, v in result.items() if k != "visualizations"
        }
        with self._lock:
            if key in self._pred_cache:
                self._pred_cache.move_to_end(key)
            self._pred_cache[key] = _TTLEntry(safe_result, self._pred_ttl)
            # Evict oldest entries until within limit
            while len(self._pred_cache) > self._max_pred:
                evicted_key, _ = self._pred_cache.popitem(last=False)
                self._stats["pred_evictions"] += 1
                logger.debug("LRU eviction: key=%.16s…", evicted_key)
        logger.debug("Prediction cached: key=%.16s…", key)

    def invalidate_prediction(self, key: str) -> bool:
        """Manually remove *key* from the prediction cache.

        Returns ``True`` if the key existed.
        """
        with self._lock:
            if key in self._pred_cache:
                del self._pred_cache[key]
                return True
            return False

    def invalidate_all_predictions(self) -> int:
        """Clear the entire prediction cache.

        Returns the number of entries removed.
        """
        with self._lock:
            count = len(self._pred_cache)
            self._pred_cache.clear()
        logger.info("All %d prediction cache entries cleared.", count)
        return count

    # ------------------------------------------------------------------
    # Static / benchmark cache
    # ------------------------------------------------------------------

    def get_static(self, key: str) -> Any | None:
        """Return a cached static value, or ``None`` on miss / expiry."""
        with self._lock:
            entry = self._static_cache.get(key)
            if entry is None:
                self._stats["static_misses"] += 1
                return None
            if entry.is_expired():
                del self._static_cache[key]
                self._stats["static_misses"] += 1
                return None
            self._stats["static_hits"] += 1
            return entry.value

    def set_static(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key* in the static cache."""
        effective_ttl = ttl if ttl is not None else self._static_ttl
        with self._lock:
            self._static_cache[key] = _TTLEntry(value, effective_ttl)
        logger.debug("Static cached: key=%s, ttl=%ds", key, effective_ttl)

    def invalidate_static(self, key: str) -> bool:
        """Manually remove a static cache entry.

        Returns ``True`` if it existed.
        """
        with self._lock:
            if key in self._static_cache:
                del self._static_cache[key]
                return True
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a copy of internal hit/miss/eviction counters."""
        with self._lock:
            pred_live = sum(
                1 for e in self._pred_cache.values() if not e.is_expired()
            )
            static_live = sum(
                1 for e in self._static_cache.values() if not e.is_expired()
            )
        return {
            **self._stats,
            "pred_cache_live_entries": pred_live,
            "static_cache_live_entries": static_live,
            "pred_ttl_seconds": self._pred_ttl,
            "static_ttl_seconds": self._static_ttl,
            "model_version": self._model_version,
        }

    def reset_stats(self) -> None:
        """Zero all hit/miss counters."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0

    def purge_expired(self) -> int:
        """Remove all expired entries from both caches.

        Returns the total number of entries removed.
        """
        now = time.monotonic()
        removed = 0
        with self._lock:
            for key in [k for k, v in self._pred_cache.items() if now > v.expires_at]:
                del self._pred_cache[key]
                removed += 1
            for key in [k for k, v in self._static_cache.items() if now > v.expires_at]:
                del self._static_cache[key]
                removed += 1
        if removed:
            logger.debug("purge_expired: removed %d expired entries.", removed)
        return removed

    def bump_model_version(self, new_version: str) -> None:
        """Update the model version string and clear the prediction cache.

        Call this if model weights are hot-swapped while the server is running
        (unusual but possible for development).
        """
        with self._lock:
            self._model_version = new_version
            count = len(self._pred_cache)
            self._pred_cache.clear()
        logger.info(
            "Model version updated to '%s'; %d stale prediction(s) evicted.",
            new_version,
            count,
        )
