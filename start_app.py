import sys
import os
import time
import webbrowser
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("=" * 70)
print("  Digital Image Forensics: Improved DenseNet-121 Architecture")
print("  Paper Implementation by Ahmed Alzahrani (ETASR 2024)")
print("=" * 70)

def main():
    backend_path = os.path.join(BASE_DIR, 'backend', 'app.py')
    print(f"\n[+] Starting Flask API Backend server...")
    print(f"[+] Script: {backend_path}")
    print(f"[+] Server URL: http://127.0.0.1:5000")
    print(f"[+] Press Ctrl+C in terminal to stop server.")

    def open_browser():
        time.sleep(2)
        print("[+] Opening web application in browser...")
        webbrowser.open("http://127.0.0.1:5000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
    from app import app
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    main()
