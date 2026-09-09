"""
Cloud Sync and Automated Backup Engine for LipRead Studio.
Provides background asynchronous replication to Google Drive, OneDrive,
NAS, or external drives without stalling real-time video/audio recording.
"""

import shutil
import time
import queue
import threading
from pathlib import Path
import logging


class CloudSyncEngine:
    """Asynchronously syncs dataset artifacts to a designated backup/cloud folder."""

    def __init__(self, target_dir: Path = None):
        self.target_dir = None
        self.queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()

        self.synced_count = 0
        self.failed_count = 0
        self.last_error = ""

        if target_dir:
            self.set_target_dir(target_dir)

    def set_target_dir(self, target_path):
        """Configure and validate new backup target directory."""
        if not target_path:
            self.stop()
            self.target_dir = None
            return True, "Cloud Sync disabled"

        path = Path(target_path).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".sync_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()

            with self.lock:
                self.target_dir = path

            self.start()
            return True, f"Backup target set: {path}"
        except Exception as e:
            return False, f"Failed to set backup directory: {e}"

    def is_enabled(self) -> bool:
        return self.target_dir is not None and self.target_dir.exists()

    def enqueue_file(self, source_path: Path, rel_destination: str):
        """Add a file to the backup queue."""
        if not self.is_enabled():
            return
        self.queue.put(("file", Path(source_path), rel_destination))

    def enqueue_dir(self, source_dir: Path, rel_destination: str):
        """Add an entire directory (e.g. extracted lips folder) to backup queue."""
        if not self.is_enabled():
            return
        self.queue.put(("dir", Path(source_dir), rel_destination))

    def start(self):
        """Start background queue processor."""
        if self.running or not self.target_dir:
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="CloudSyncWorker"
        )
        self.worker_thread.start()

    def _worker_loop(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            item_type, src, rel_dest = item
            if not self.target_dir or not src.exists():
                self.queue.task_done()
                continue

            dest_path = self.target_dir / rel_dest
            try:
                if item_type == "file":
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest_path)
                elif item_type == "dir":
                    dest_path.mkdir(parents=True, exist_ok=True)
                    for sub in src.glob("*.jpg"):
                        shutil.copy2(sub, dest_path / sub.name)

                with self.lock:
                    self.synced_count += 1
            except Exception as e:
                with self.lock:
                    self.failed_count += 1
                    self.last_error = str(e)
                logging.warning("CloudSync error copying %s: %s", src, e)
            finally:
                self.queue.task_done()

    def get_status_string(self) -> str:
        """Human-readable sync status for UI badges."""
        if not self.is_enabled():
            return "☁ Backup: OFF"
        pending = self.queue.qsize()
        if pending > 0:
            return f"☁ Syncing ({pending} queued)"
        return f"☁ Synced ({self.synced_count} files)"

    def stop(self):
        """Cleanly stop worker thread."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None
