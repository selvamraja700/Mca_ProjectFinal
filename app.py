"""
Digital Image Forensics - Quick Launcher
=========================================
Run this file to start the project:
    python app.py

The Flask server will start and Chrome will open automatically
with the Image Forensics Scanner.
"""

import sys
import os
import time
import webbrowser
import subprocess
import threading

# ── Setup paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')

# Change to project root so relative paths in backend work correctly
os.chdir(BASE_DIR)

# Add backend directory to Python path so imports work
sys.path.insert(0, BACKEND_DIR)


def print_banner():
    """Print a startup banner."""
    print()
    print("=" * 65)
    print("  🔍  Digital Image Forensics Scanner")
    print("  📄  Improved DenseNet-121 Architecture")
    print("  📝  Paper: Ahmed Alzahrani (ETASR 2024)")
    print("=" * 65)
    print()


def open_chrome(url, delay=2.5):
    """Open the scanner URL in Chrome after a short delay."""
    time.sleep(delay)
    print(f"[✓] Opening scanner in Chrome → {url}")
    
    # Try to open specifically in Chrome
    try:
        # Windows: try Chrome first
        chrome_path = None
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                chrome_path = p
                break
        
        if chrome_path:
            webbrowser.register('chrome', None,
                                webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get('chrome').open(url)
        else:
            # Fallback: open in default browser
            webbrowser.open(url)
    except Exception:
        # Final fallback
        webbrowser.open(url)


def main():
    """Start the Flask backend and open Chrome with the scanner."""
    print_banner()

    url = "http://127.0.0.1:5000"

    print(f"[→] Starting Flask server...")
    print(f"[→] Backend : {os.path.join(BACKEND_DIR, 'app.py')}")
    print(f"[→] URL     : {url}")
    print(f"[→] Press Ctrl+C to stop the server.\n")

    # Launch Chrome in a background thread
    browser_thread = threading.Thread(target=open_chrome, args=(url,), daemon=True)
    browser_thread.start()

    # Import and run the Flask app from the backend
    try:
        from app import app  # imports backend/app.py since BACKEND_DIR is in sys.path
        app.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n[✗] Server stopped by user.")
    except ImportError as e:
        print(f"\n[✗] Import error: {e}")
        print("[!] Make sure all dependencies are installed:")
        print("    pip install -r backend/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
