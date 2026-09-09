"""
Machine Learning Export Engine for LipRead Studio.
Converts dataset collections into research-ready manifests, vocabulary tokens,
TensorFlow/PyTorch CTC formats, and compressed NPZ frame sequence archives.
"""

import json
import csv
from pathlib import Path
import numpy as np
from config import EXPORTS_ROOT, DATASET_DIR


class MLExportEngine:
    """Exports recorded visual speech datasets to standard ML frameworks."""

    def __init__(self, exports_dir: Path = EXPORTS_ROOT):
        self.exports_dir = exports_dir
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def export_pytorch_manifest(self, rows: list, speaker_id: str, val_split: float = 0.2):
        """Generate PyTorch/HuggingFace-compatible dataset manifest and vocabulary."""
        out_file = self.exports_dir / f"{speaker_id}_pytorch_manifest.json"
        vocab_file = self.exports_dir / f"{speaker_id}_vocab.json"

        # Unique words and characters for vocabulary
        all_words = sorted(list({r.get("word", "").strip() for r in rows if r.get("word")}))
        chars = sorted(list({c for w in all_words for c in w.lower()}))

        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx["<pad>"] = 0
        char2idx["<blank>"] = 1

        word2idx = {w: i for i, w in enumerate(all_words)}

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump({
                "char2idx": char2idx,
                "word2idx": word2idx,
                "words": all_words,
                "total_words": len(all_words)
            }, f, indent=2, ensure_ascii=False)

        # Build items
        items = []
        for r in rows:
            w = r.get("word", "").strip()
            item = {
                "speaker_id": r.get("speaker_id"),
                "word": w,
                "word_id": word2idx.get(w, -1),
                "token_ids": [char2idx.get(c, 0) for c in w.lower()],
                "video_relpath": r.get("video_file"),
                "audio_relpath": r.get("audio_file"),
                "lip_dir_relpath": r.get("lip_directory"),
                "frames": int(r.get("video_frames", 0)),
                "quality_score": float(r.get("quality_score", 0.0)),
                "emotion": r.get("emotion_detected", "Neutral")
            }
            items.append(item)

        # Train / Validation partition
        np.random.seed(42)
        indices = np.random.permutation(len(items))
        split_idx = int(len(items) * (1.0 - val_split))

        train_items = [items[i] for i in indices[:split_idx]]
        val_items = [items[i] for i in indices[split_idx:]]

        manifest = {
            "dataset": "LipReadStudio",
            "speaker_id": speaker_id,
            "vocab_file": vocab_file.name,
            "train_count": len(train_items),
            "val_count": len(val_items),
            "train": train_items,
            "val": val_items
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return out_file, vocab_file

    def export_ctc_csv(self, rows: list, speaker_id: str):
        """Export simplified CSV for LipNet / CTC Loss pipelines."""
        out_file = self.exports_dir / f"{speaker_id}_ctc_manifest.csv"

        fields = ["video_path", "audio_path", "lip_dir", "text", "frames", "duration_sec", "quality"]
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "video_path": r.get("video_file"),
                    "audio_path": r.get("audio_file"),
                    "lip_dir": r.get("lip_directory"),
                    "text": r.get("word"),
                    "frames": r.get("video_frames"),
                    "duration_sec": r.get("duration_sec"),
                    "quality": r.get("quality_score")
                })

        return out_file
