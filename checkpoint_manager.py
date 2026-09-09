"""
Checkpoint and Crash Recovery Manager for LipRead Studio.
Provides atomic session persistence, sudden shutdown resilience,
and 1-click resumption from the exact last recorded word.
"""

import json
import time
import os
from pathlib import Path
from config import CHECKPOINTS_ROOT


class CheckpointManager:
    """Manages recording checkpoints for seamless crash recovery."""

    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_ROOT):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, speaker_id: str) -> Path:
        safe_id = "".join(c for c in speaker_id if c.isalnum() or c in "-_")
        return self.checkpoints_dir / f"{safe_id}_checkpoint.json"

    def save_checkpoint(
        self,
        speaker_id: str,
        word_list_name: str,
        current_index: int,
        total_words: int,
        rows: list,
        extra_meta: dict = None
    ) -> bool:
        """Atomically persist session state to disk."""
        target_file = self._get_path(speaker_id)
        temp_file = target_file.with_suffix(".tmp")

        data = {
            "speaker_id": speaker_id,
            "word_list_name": word_list_name,
            "current_index": current_index,
            "total_words": total_words,
            "rows": rows,
            "timestamp": time.time(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extra_meta": extra_meta or {}
        }

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(target_file)
            return True
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return False

    def has_unfinished_session(self, speaker_id: str) -> bool:
        """Check if an incomplete session checkpoint exists for the given speaker."""
        target_file = self._get_path(speaker_id)
        if not target_file.exists():
            return False

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = int(data.get("current_index", 0))
            tot = int(data.get("total_words", 0))
            return tot > 0 and idx < tot - 1
        except Exception:
            return False

    def load_checkpoint(self, speaker_id: str) -> dict:
        """Load and return session checkpoint data."""
        target_file = self._get_path(speaker_id)
        if not target_file.exists():
            return {}

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def clear_checkpoint(self, speaker_id: str):
        """Remove checkpoint after complete session finish."""
        target_file = self._get_path(speaker_id)
        if target_file.exists():
            try:
                target_file.unlink()
            except Exception:
                pass

    def list_all_unfinished(self) -> list:
        """List all pending unfinished sessions across speakers."""
        results = []
        for file in self.checkpoints_dir.glob("*_checkpoint.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                idx = int(data.get("current_index", 0))
                tot = int(data.get("total_words", 0))
                if tot > 0 and idx < tot - 1:
                    results.append({
                        "speaker_id": data.get("speaker_id"),
                        "current_index": idx,
                        "total_words": tot,
                        "updated_at": data.get("updated_at")
                    })
            except Exception:
                pass
        return results
