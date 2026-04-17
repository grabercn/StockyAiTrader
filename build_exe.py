"""
EXE Builder for StockyAiTrader.

Builds a standalone .exe that:
- Launches StockyHub on startup
- Includes all source code and core modules
- Does NOT bundle ML models (FinBERT etc.) — user downloads them via the Model Manager
- Bundles the banner image and default settings

Uses PyInstaller under the hood. Install it first: pip install pyinstaller

Usage:
    python build_exe.py
"""

import os
import sys
import subprocess
import shutil

# ─── Configuration ───────────────────────────────────────────────────────────

APP_NAME = "StockyAiTrader"
MAIN_SCRIPT = "StockyHub.py"
ICON_FILE = None  # Set to "icon.ico" if you have one

# Files/folders to bundle WITH the exe (data files, not Python)
DATA_FILES = [
    ("banner.jpg", "."),                    # Banner image -> root
    ("StockyApps", "StockyApps"),           # All app scripts + core/
]

# Folders to EXCLUDE from the bundle (models are downloaded at runtime)
EXCLUDE_DIRS = [
    "models",           # Trained LightGBM models (created at runtime)
    ".git",
    "__pycache__",
    "build",
    "dist",
    "results",
    "logs",
]

# Large packages to exclude from the exe to keep size manageable.
# These must be installed on the user's system (or bundled separately).
# We keep torch/transformers since they're needed, but exclude training-only stuff.
HIDDEN_IMPORTS = [
    "PyQt5",
    "PyQt5.QtWidgets",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "lightgbm",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.metrics",
    "ta",
    "ta.volatility",
    "ta.trend",
    "ta.momentum",
    "ta.volume",
    "yfinance",
    "feedparser",
    "vaderSentiment",
    "vaderSentiment.vaderSentiment",
    "transformers",
    "numpy",
    "pandas",
    "matplotlib",
    "matplotlib.backends.backend_qt5agg",
    "requests",
    "joblib",
    "pytz",
    "torch",
]


def build():
    """Build the exe using PyInstaller."""

    # Check PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Clean previous builds
    for d in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
            print(f"Cleaned: {d}")

    # Build the PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--windowed",                # No console window (GUI app)
        "--noconfirm",               # Overwrite without asking
        "--clean",                   # Clean cache before building
    ]

    # Add icon if available
    if ICON_FILE and os.path.exists(ICON_FILE):
        cmd.extend(["--icon", ICON_FILE])

    # Add data files
    sep = ";" if sys.platform == "win32" else ":"
    for src, dst in DATA_FILES:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src}{sep}{dst}"])
            print(f"Bundling: {src} -> {dst}")
        else:
            print(f"Warning: {src} not found, skipping")

    # Add hidden imports (modules PyInstaller might miss)
    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])

    # Exclude large unused packages
    for exclude in ["tkinter", "unittest", "test", "PIL"]:
        cmd.extend(["--exclude-module", exclude])

    # The main script to compile
    cmd.append(MAIN_SCRIPT)

    print()
    print("=" * 60)
    print(f"Building {APP_NAME}.exe ...")
    print(f"Entry point: {MAIN_SCRIPT}")
    print(f"This will take a few minutes...")
    print("=" * 60)
    print()

    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\nBuild FAILED: {e}")
        print("Check the output above for errors.")
        sys.exit(1)

    # Copy default settings template to dist folder
    dist_dir = os.path.join("dist", APP_NAME)
    if os.path.isdir(dist_dir):
        # Create empty models directory so the app doesn't crash
        os.makedirs(os.path.join(dist_dir, "models"), exist_ok=True)

        # Create a default settings.json if one doesn't exist there
        settings_path = os.path.join(dist_dir, "settings.json")
        if not os.path.exists(settings_path):
            import json
            default = {
                "alpaca_api_key": "",
                "alpaca_secret_key": "",
                "default_ticker": "AAPL",
                "refresh_rate": 5,
            }
            with open(settings_path, "w") as f:
                json.dump(default, f, indent=4)
            print(f"Created default settings.json in {dist_dir}")

    # Summary
    exe_path = os.path.join("dist", APP_NAME, f"{APP_NAME}.exe")
    if not os.path.exists(exe_path):
        # --onefile mode puts it directly in dist/
        exe_path = os.path.join("dist", f"{APP_NAME}.exe")

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print("=" * 60)
        print(f"BUILD SUCCESSFUL!")
        print(f"Executable: {os.path.abspath(exe_path)}")
        print(f"Size: {size_mb:.1f} MB")
        print()
        print("NOTE: ML models are NOT included in the exe.")
        print("Users should click 'Download' in the Model Manager")
        print("on first launch to download FinBERT (~300MB).")
        print("=" * 60)
    else:
        print("\nBuild finished but exe not found. Check dist/ folder.")


if __name__ == "__main__":
    build()
