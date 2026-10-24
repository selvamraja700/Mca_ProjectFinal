# cookie_section — Session, Cookie & Cache Architecture

Lightweight session persistence and prediction caching for the
**ML-Based Image Forgery Detection System** (Flask + PyTorch DenseNet-121).

---

## 1. Architectural Overview

| Component | Class | Responsibility | Storage Mechanism |
|-----------|-------|----------------|-------------------|
| **Cookie Manager** | `CookieManager` | Browser session persistence for UI state (theme, last scan, scan count) | Client-side cryptographic HMAC-signed Flask cookie |
| **Session Manager** | `SessionManager` | Server-side persistence for larger session payloads | File-backed JSON store with atomic writes & TTL |
| **Cache Manager** | `CacheManager` | In-memory prediction and static benchmark caching | In-process thread-safe LRU + TTL cache |
| **Configuration** | `config.py` | Centralized parameter repository | Typed constants with environment variable overrides |

---

## 2. Deep Code Explanation

### 2.1 `cookie_manager.py` — `CookieManager`

#### Purpose & Design
The application is a single-user local image forensics scanner with no user authentication system. `CookieManager` provides state persistence across browser restarts without inventing an artificial login system.

#### How Signed Cookies Work
Flask handles client-side sessions by serializing session data into a JSON string, compressing it, and signing it with an HMAC (Hash-based Message Authentication Code) using `SECRET_KEY`.
- **Integrity**: The browser cannot tamper with the stored values without invalidating the cryptographic signature.
- **No Sensitive Data**: Passwords, raw model weights, and heavy image buffers are never placed in the cookie.

#### Security & Cookie Attributes
`CookieManager.configure(app: Flask)` applies the following security attributes:
- `SESSION_COOKIE_HTTPONLY = True`: Blocks client-side JavaScript (`document.cookie`) from accessing the session cookie, neutralizing cross-site scripting (XSS) cookie theft.
- `SESSION_COOKIE_SAMESITE = 'Lax'`: Enforces CSRF protection while permitting top-level navigations.
- `SESSION_COOKIE_SECURE = False`: Tailored for local HTTP (`127.0.0.1`); toggled to `True` when served over TLS/HTTPS.
- `PERMANENT_SESSION_LIFETIME = 8 hours`: Configured via `datetime.timedelta`.
- `SESSION_PERMANENT = True`: Writes an explicit `Max-Age` / `Expires` attribute to the HTTP response header, ensuring the cookie survives browser window closures.

#### Key Whitelist (`ALLOWED_KEYS`)
To prevent cookie bloat and enforce security boundaries, `set()` restricts storage to an immutable frozenset:
```python
ALLOWED_KEYS = frozenset({
    "theme",            # UI theme preference ('dark' | 'light')
    "last_filename",    # Filename of the most recently scanned image
    "last_prediction",  # Classification outcome ('Genuine' | 'Forged')
    "last_confidence",  # Model confidence score (float percentage)
    "scan_count",       # Number of scans performed during the session
    "session_id",       # 64-character token referencing server-side state
})
```
Attempting to write an unauthorized key raises a `KeyError`.

#### Key Methods
- `configure(app)`: Sets application secret key and session configuration parameters. Idempotent.
- `set(key, value)`: Validates against `ALLOWED_KEYS`, writes to `flask.session`, and flags `session.permanent = True`.
- `get(key, default=None)`: Reads from `flask.session`.
- `record_scan(filename, prediction, confidence)`: High-level helper updating `last_filename`, `last_prediction`, `last_confidence`, and incrementing `scan_count` atomically.
- `clear()`: Pops all allowed keys from the active session.
- `is_session_intact()`: Verifies if any valid application keys are currently stored.
- `cookie_attributes_for(app)`: Inspection utility returning active cookie configuration values.

---

### 2.2 `session_manager.py` — `SessionManager`

#### Purpose & Design
Browser cookies have a hard size limit of 4 KB. Prediction payloads in this forensics tool contain base64-encoded visualizations (ELA and heatmaps) reaching 100–300 KB. `SessionManager` provides server-side persistence for these larger payloads while keeping the client cookie minimal (holding only a 32-byte / 64-character hex token).

#### Storage Scheme & Envelope Format
Sessions are stored as structured JSON documents inside `.session_store/`:
```json
{
  "token_hash": "<sha256_of_token>",
  "created_at": 1729780000.0,
  "expires_at": 1729808800.0,
  "data": { ... }
}
```

#### Atomic Write Pattern
To prevent corrupt files from partial writes or concurrent access:
1. Session data is serialized to JSON.
2. Written to a temporary file: `sess_<token>.tmp`.
3. Renamed to `sess_<token>.json` using `pathlib.Path.replace()`.
Operating systems guarantee this replace operation is atomic on POSIX and NTFS filesystems.

#### Concurrency & Expiration
- **Locking**: A process-wide `threading.Lock()` guards all file write operations.
- **Token Validation**: Tokens must be exactly 64 hexadecimal characters, verified with `int(token, 16)` prior to any disk I/O to prevent directory traversal attacks.
- **Lazy Eviction**: `load_session(token)` checks `time.time() > expires_at`. If expired, the file is unlinked immediately and returns `None`.
- **Bulk Cleanup**: `cleanup_expired()` scans the directory and removes all stale or malformed session files.
- **Sliding Expiry**: `extend_session(token)` resets the `expires_at` timestamp relative to the current time.

---

### 2.3 `cache_manager.py` — `CacheManager`

#### Purpose & Bottleneck Addressed
In the existing image forensics pipeline, every prediction execution involves:
1. Error Level Analysis (ELA) generation via PIL recompression and difference computation: ~50–200 ms.
2. DenseNet-121 forward pass inference on CPU: ~100–400 ms.
3. Heatmap blending and normalization: ~20–80 ms.
Total execution latency: 170–680 ms per image.

When testing or re-evaluating the same image within a session, re-running inference is redundant. `CacheManager` provides an in-process, thread-safe cache with automatic eviction.

#### Cache Architecture & Namespaces
`CacheManager` maintains two distinct, isolated memory structures:
1. **Prediction Cache (`_pred_cache`)**: An `OrderedDict` of image results.
2. **Static Cache (`_static_cache`)**: A standard dictionary for immutable benchmark data (`/api/benchmarks`).

#### Cache Entry Structure (`_TTLEntry`)
```python
class _TTLEntry:
    __slots__ = ("value", "expires_at")
    def __init__(self, value: Any, ttl: float):
        self.value = value
        self.expires_at = time.monotonic() + ttl
```
Using `__slots__` reduces per-entry memory overhead, and `time.monotonic()` ensures immune behavior against system clock adjustments.

#### Image Content Hashing & Model Versioning
Cache keys are constructed from the raw byte stream of the image plus the model version string:
```python
def make_image_key(self, image_bytes: bytes) -> str:
    content_hash = hashlib.sha256(image_bytes).hexdigest()
    return f"{content_hash}:{self._model_version}"
```
- **Content-Addressable**: Two identical images with different filenames produce the exact same key.
- **Automatic Invalidation on Model Updates**: If model weights are updated and `_model_version` changes, all existing keys immediately miss without requiring manual cache wipes.

#### Memory-Conscious Caching
Large visualization payloads (`data:image/png;base64,...`) consume significant RAM when held in memory. `set_prediction()` automatically strips the `"visualizations"` key before inserting into the cache:
```python
safe_result = {k: v for k, v in result.items() if k != "visualizations"}
```
Only classification outcomes, confidence scores, probabilities, and ELA noise metrics are cached.

#### LRU Eviction & Concurrency
- `_pred_cache` utilizes Python's `collections.OrderedDict`.
- On every cache hit, the entry is relocated to the right via `move_to_end(key)`.
- When the size exceeds `_max_pred` (default: 50), the oldest item on the left is purged via `popitem(last=False)`.
- All operations are synchronized with `threading.Lock()`.

---

### 2.4 `config.py` — Central Configuration

Provides type-annotated configuration parameters initialized from `os.environ` with fallback defaults:
- `SECRET_KEY`: Uses environment variable if present; otherwise generates a cryptographically secure 256-bit key via `secrets.token_hex(32)`.
- `SESSION_LIFETIME_SECONDS`: Integer duration for session validity (default: 28,800 seconds / 8 hours).
- `SESSION_PERMANENT`: Boolean indicating whether cookie expiration is persistent across browser restarts.
- `SESSION_COOKIE_NAME`: String name for the HTTP cookie (default: `"forensics_session"`).
- `CACHE_PREDICTION_TTL_SECONDS`: Time-to-live for cached model inferences (default: 1,800 seconds / 30 minutes).
- `CACHE_STATIC_TTL_SECONDS`: Time-to-live for static application datasets (default: 86,400 seconds / 24 hours).
- `CACHE_MAX_PREDICTION_ENTRIES`: Upper limit for cached prediction entries before LRU eviction triggers (default: 50).
- `SESSION_STORE_DIR`: Filesystem directory for server-side session files (default: `<project_root>/.session_store`).

---

## 3. Integration Code Reference (`backend/app.py`)

The utilities inside `cookie_section/` are completely decoupled and non-destructive. To integrate them into the Flask application, apply the following minimal code additions:

### Step 1: Expose Root Directory & Import Modules
```python
import sys, os
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cookie_section import CookieManager, CacheManager
```

### Step 2: Initialize Managers on the Flask App Instance
```python
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# Configure Cookie and Cache subsystems
cookie_mgr = CookieManager()
cookie_mgr.configure(app)

cache_mgr = CacheManager()
```

### Step 3: Implement Caching in `/api/predict`
```python
@app.route('/api/predict', methods=['POST'])
def predict():
    # ... [Existing image decoding and validation logic] ...

    cache_key = cache_mgr.make_image_key(image_bytes)
    cached_result = cache_mgr.get_prediction(cache_key)

    if cached_result is not None:
        # Return cached classification metrics immediately
        return jsonify({**cached_result, "cached": True})

    try:
        result = predict_image(model, image_bytes, filename=filename)
        
        # Cache prediction and record scan metadata in session cookie
        cache_mgr.set_prediction(cache_key, result)
        cookie_mgr.record_scan(filename, result["prediction"], result["confidence"])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 400
```

### Step 4: Implement Caching in `/api/benchmarks`
```python
@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    cached_payload = cache_mgr.get_static("benchmarks")
    if cached_payload is None:
        import json
        cached_payload = json.dumps(PAPER_METRICS)
        cache_mgr.set_static("benchmarks", cached_payload)
    
    return Response(cached_payload, mimetype="application/json")
```

---

## 4. Test Suite Coverage Analysis

The verification suite in `cookie_section/tests/test_cookie_section.py` provides 60 automated unit and integration tests across 4 categories:

1. **`TestCookieManager`**:
   - `test_configure_applies_secret_key`: Validates non-empty secret key application.
   - `test_configure_sets_cookie_name`, `test_configure_sets_httponly`, `test_configure_sets_samesite`: Verifies HTTP cookie attribute headers.
   - `test_set_and_get_in_request_context`: Tests request-bound session dictionary reading and writing.
   - `test_disallowed_key_raises`: Confirms strict key whitelist enforcement.
   - `test_record_scan_updates_all_fields`, `test_record_scan_increments_count`: Tests scan history accounting.

2. **`TestSessionManager`**:
   - `test_create_returns_64_hex_chars`: Asserts 256-bit token entropy.
   - `test_create_writes_file_to_disk`: Confirms filesystem persistence.
   - `test_load_returns_stored_data`: Verifies JSON deserialization.
   - `test_load_returns_none_after_expiry`: Verifies lazy time-based invalidation.
   - `test_delete_returns_false_for_missing`: Validates existence check on file removal.
   - `test_extend_session_allows_future_load`: Validates sliding TTL refresh.
   - `test_cleanup_expired_removes_stale`: Tests bulk cleanup of stale sessions.

3. **`TestCacheManager`**:
   - `test_key_is_deterministic`, `test_key_includes_model_version`: Validates SHA-256 key stability and model version binding.
   - `test_visualizations_stripped_from_cache`: Verifies base64 payload stripping to protect RAM.
   - `test_lru_eviction_fires`: Tests eviction of oldest entries when capacity is exceeded.
   - `test_prediction_expires_after_ttl`: Verifies time-based entry invalidation.
   - `test_bump_model_version_clears_predictions`: Confirms model invalidation clears stale inference data.

4. **`TestExistingFilesUntouched` (Integrity Guardrails)**:
   - `test_model_weights_size_unchanged`: Asserts `densenet_forgery.pth` remains exactly 28,613,366 bytes.
   - `test_backend_app_py_not_modified`: Confirms `backend/app.py` has not been altered.
   - `test_no_cookie_section_imports_in_backend_app`: Confirms unapproved imports are absent.
   - `test_dataset_directory_not_deleted`: Confirms dataset integrity.

---

## 5. Technical Limitations & Reality Checks

- **Python Startup**: Cookie and session mechanisms cannot reduce Python interpreter startup latency or PyTorch CUDA/CPU model loading time (~1–3 seconds).
- **First Inference**: The initial scan of any image will always execute the full ELA and DenseNet pipeline. Speedups apply strictly to repeat analyses.
- **In-Memory Cache Scope**: `CacheManager` is process-bound; restarting the Flask server clears the cache by design. Server-side session files in `.session_store/` survive server restarts.
