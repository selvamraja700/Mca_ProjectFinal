# cookie_section — Session, Cookie & Cache Utilities

Lightweight session persistence and prediction caching for the
**ML-Based Image Forgery Detection System** (Flask + PyTorch DenseNet-121).

---

## What This Solves

| Problem | What cookie_section does |
|---------|--------------------------|
| Every browser restart loses UI state (theme, last scan result) | Flask signed cookie stores `theme`, `last_filename`, `scan_count`, etc. — survives browser restart for up to 8 hours by default |
| ELA + DenseNet inference takes 170–680 ms on CPU | In-process LRU+TTL cache skips re-computation when the same image is uploaded again within 30 minutes |
| `/api/benchmarks` re-serialises the same large JSON on every call | Static cache holds the serialised payload for 24 hours |
| Session data grows too large for a browser cookie | Optional file-backed server-side session store holds large payloads; browser only holds a 64-char token |

---

## Structure

```
cookie_section/
├── __init__.py          Public API: CookieManager, SessionManager, CacheManager
├── config.py            All tuneable values (override with env vars)
├── cookie_manager.py    Flask session-cookie configuration + helpers
├── session_manager.py   File-backed server-side session store
├── cache_manager.py     In-process LRU+TTL cache (predictions + static data)
├── tests/
│   ├── __init__.py
│   └── test_cookie_section.py   53 tests covering all three modules
└── README.md            This file
```

---

## Dependencies

No new packages required.  All three modules use only:

| Module | Dependency | Already in requirements.txt? |
|--------|-----------|------------------------------|
| `cookie_manager.py` | `flask` | ✅ Yes (`flask>=2.0.0`) |
| `session_manager.py` | stdlib (`json`, `pathlib`, `threading`, `secrets`, `hashlib`) | ✅ stdlib |
| `cache_manager.py` | stdlib (`collections`, `hashlib`, `threading`, `time`) | ✅ stdlib |

---

## Running the Tests

From the **project root** (`d:\MiniProject_MCA_final\`):

```powershell
# Using pytest (recommended):
py -3 -m pytest cookie_section/tests/test_cookie_section.py -v

# Or directly:
py -3 cookie_section/tests/test_cookie_section.py
```

### Expected output (excerpt)
```
test_configure_applies_secret_key ... ok
test_configure_sets_cookie_name ... ok
test_set_and_get_in_request_context ... ok
test_record_scan_updates_all_fields ... ok
test_disallowed_key_raises ... ok
test_create_returns_64_hex_chars ... ok
test_load_returns_stored_data ... ok
test_load_returns_none_after_expiry ... ok
test_lru_eviction_fires ... ok
test_prediction_expires_after_ttl ... ok
test_model_weights_size_unchanged ... ok
test_backend_app_py_not_modified ... ok
...
----------------------------------------------------------------------
Ran 53 tests in X.Xs

OK
```

---

## Configuration (Environment Variables)

Set these before running `python app.py` to change behaviour.  Defaults
shown are appropriate for a local single-user project.

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_SECRET_KEY` | *(random per process)* | HMAC signing key for session cookies.  **Set to a fixed string** to survive server restarts. |
| `SESSION_LIFETIME_SECONDS` | `28800` (8 h) | How long the browser cookie is valid. |
| `SESSION_PERMANENT` | `true` | `true` → cookie survives browser close; `false` → session cookie. |
| `SESSION_COOKIE_NAME` | `forensics_session` | Name of the browser cookie. |
| `SESSION_COOKIE_SAMESITE` | `Lax` | SameSite attribute. |
| `SESSION_COOKIE_SECURE` | `false` | Set `true` only when serving over HTTPS. |
| `CACHE_PREDICTION_TTL_SECONDS` | `1800` (30 min) | How long a cached prediction result is valid. |
| `CACHE_STATIC_TTL_SECONDS` | `86400` (24 h) | How long static/benchmark data is cached. |
| `CACHE_MAX_PREDICTION_ENTRIES` | `50` | Maximum in-memory prediction entries (LRU eviction). |
| `SESSION_STORE_DIR` | `<project_root>/.session_store` | Where server-side session files are stored. |

PowerShell example:
```powershell
$env:FLASK_SECRET_KEY = "my-stable-random-key-here"
$env:SESSION_PERMANENT = "true"
py -3 app.py
```

To generate a good secret key once:
```powershell
py -3 -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output and use it as `FLASK_SECRET_KEY`.

---

## Integration into backend/app.py

> ⚠️ **No changes to `backend/app.py` have been made.**
> The snippets below are the **minimum changes required** — apply them
> only after your approval.

### Step 1 — Add the project root to `sys.path` (already done in `app.py` entry-point)

This step is handled by the root `app.py` which adds `backend/` to `sys.path`.
For `cookie_section` to be importable from `backend/app.py`, the **project root**
must also be on `sys.path`.  The root `app.py` already sets `os.chdir(BASE_DIR)`,
so adding:

```python
# In backend/app.py, near top
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

…makes `cookie_section` importable from within the backend.

### Step 2 — Initialise CookieManager once

```python
# backend/app.py — after  app = Flask(...)  and  CORS(app)
from cookie_section import CookieManager, CacheManager
cookie_mgr = CookieManager()
cookie_mgr.configure(app)         # sets secret_key, SameSite, HttpOnly, lifetime

cache_mgr  = CacheManager()       # in-process prediction + static cache
```

### Step 3 — Use cache in `/api/predict`

```python
@app.route('/api/predict', methods=['POST'])
def predict():
    # ... existing file / base64 reading code unchanged ...

    # Cache lookup (add BEFORE the existing try/except block):
    cache_key = cache_mgr.make_image_key(image_bytes)
    cached = cache_mgr.get_prediction(cache_key)
    if cached is not None:
        # Re-run visualizations only (fast) — skip DenseNet + ELA
        # Or return cached without visualizations for an instant response:
        return jsonify({**cached, "cached": True})

    # Existing code unchanged:
    try:
        result = predict_image(model, image_bytes, filename=filename)
        cache_mgr.set_prediction(cache_key, result)   # store for next time
        cookie_mgr.record_scan(filename, result["prediction"], result["confidence"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "..."}), 400
```

### Step 4 — Cache `/api/benchmarks`

```python
@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    cached = cache_mgr.get_static("benchmarks")
    if cached is None:
        import json
        cached = json.dumps(PAPER_METRICS)
        cache_mgr.set_static("benchmarks", cached)
    from flask import Response
    return Response(cached, mimetype="application/json")
```

### Step 5 — Expose session summary (optional, for UI)

```python
@app.route('/api/session', methods=['GET'])
def get_session():
    return jsonify(cookie_mgr.get_session_summary())
```

---

## Honest Limitations

| Claim | Reality |
|-------|---------|
| "Cookies make Python startup faster" | **False.** Python startup and PyTorch model loading time (~1–3 s) are unaffected by cookies. |
| "Cache makes first prediction faster" | **False.** First prediction always runs the full pipeline. Only *repeat uploads of the same file* within 30 min benefit from the cache. |
| "Server-side sessions scale to many users" | **Not needed.** This is a single-user local project. File-backed sessions are appropriate. |
| "Browser cookie persists without approval" | **Not yet.** `backend/app.py` has not been modified. The utilities are ready but not wired in. |
| "Cache is shared across server restarts" | **No.** The in-process cache resets when the Flask server is restarted. Server-side session *files* do survive restarts. |

---

## Files Changed Outside `cookie_section/`

**None.** Zero changes were made to any file outside `cookie_section/`.

Verified: `backend/app.py`, `models/densenet_forgery.pth`, `frontend/`,
`train.py`, `evaluate.py`, `data/` — all untouched.
