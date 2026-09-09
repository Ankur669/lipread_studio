"""
LipRead Studio v5.0 — Application Entrypoint
Initializes crash diagnostics, verifies environment dependencies,
and boots the Tkinter user interface.
"""

import sys
import os
import time
import logging
import traceback
import platform
from pathlib import Path
import tkinter as tk

from config import LOG_DIR
from lipread_app import LipReadingRecorder

# ============================================================
# CRASH & DIAGNOSTIC LOGGING
# ============================================================

LOG_FILE = LOG_DIR / "lipread_studio.log"
CRASH_LOG_FILE = LOG_DIR / "crash.log"

try:
    from logging.handlers import RotatingFileHandler
    _handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
except Exception:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    handlers=[_handler, logging.StreamHandler()],
    force=True
)

def _uncaught_exception_handler(exc_type, exc_val, exc_tb):
    try:
        with open(CRASH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write("CRASH TIME: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(f"EXCEPTION: {exc_val}\n")
            traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
            f.write("=" * 90 + "\n")
    except Exception:
        pass

    logging.critical("UNCAUGHT EXCEPTION: %s", exc_val, exc_info=(exc_type, exc_val, exc_tb))
    sys.__excepthook__(exc_type, exc_val, exc_tb)

sys.excepthook = _uncaught_exception_handler

# ============================================================
# APPLICATION LAUNCHER
# ============================================================

def main():
    logging.info("=" * 80)
    logging.info("Starting LipRead Studio v5.0")
    logging.info("Python %s on %s", sys.version.replace('\n', ' '), platform.platform())
    logging.info("=" * 80)

    root = tk.Tk()
    app = LipReadingRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
