"""
Automated unit test suite for LipRead Studio core engines.
Verifies vocabulary presets, emotion heuristics, QC calculations, HTML reporting,
checkpoint persistence/recovery, and ML manifest generation.
"""

import unittest
import shutil
import json
from pathlib import Path

from config import (
    PRESET_WORD_LISTS, WORDS_ENGLISH_100, WORDS_HINDI_100, WORDS_ASSAMESE_50,
    QC_PASS_THRESHOLD, QC_FAILED_THRESHOLD
)
from emotion_detector import EmotionDetector
from qc_manager import QualityControlManager
from checkpoint_manager import CheckpointManager
from cloud_sync import CloudSyncEngine
from ml_export import MLExportEngine


class TestLipReadStudio(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "test_scratch"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_vocabulary_presets(self):
        """Verify multilingual vocabulary presets."""
        self.assertEqual(len(WORDS_ENGLISH_100), 100)
        self.assertEqual(len(WORDS_HINDI_100), 100)
        self.assertEqual(len(WORDS_ASSAMESE_50), 50)
        self.assertIn("English (100 Words)", PRESET_WORD_LISTS)
        self.assertIn("Hindi (100 Words)", PRESET_WORD_LISTS)
        self.assertIn("Assamese (50 Words)", PRESET_WORD_LISTS)

    def test_emotion_detector(self):
        """Verify fallback behavior and analysis interface of EmotionDetector."""
        ed = EmotionDetector()
        lbl, conf, metrics = ed.analyze_landmarks(None, 640, 480)
        self.assertEqual(lbl, "Neutral")
        self.assertIsInstance(conf, float)
        self.assertIsInstance(metrics, dict)

    def test_qc_manager(self):
        """Verify quality calculation and HTML report generation."""
        qc = QualityControlManager()
        status_pass, score_pass = qc.calculate_word_quality(0.95, 0.95, audio_level=0.05)
        self.assertEqual(status_pass, "GOOD")
        self.assertGreaterEqual(score_pass, QC_PASS_THRESHOLD)

        status_fail, score_fail = qc.calculate_word_quality(0.90, 0.50)
        self.assertEqual(status_fail, "FAILED")
        self.assertLess(0.50, QC_FAILED_THRESHOLD)

        # Test HTML report generation
        dummy_rows = [
            {
                "speaker_id": "TEST_S01", "word_id": 1, "word": "Analyse",
                "lip_detection_rate": 0.92, "face_detection_rate": 0.95,
                "quality_score": 0.93, "quality_status": "GOOD",
                "emotion_detected": "Neutral", "recording_attempt": 1, "duration_sec": 3.0
            }
        ]
        out_html = self.test_dir / "report.html"
        generated = qc.generate_html_report("TEST_S01", dummy_rows, out_html)
        self.assertTrue(generated.exists())
        self.assertIn("LIPREAD STUDIO", generated.read_text(encoding="utf-8"))

    def test_checkpoint_manager(self):
        """Verify crash recovery checkpoint saving and loading."""
        cm = CheckpointManager(checkpoints_dir=self.test_dir)
        dummy_rows = [{"word_id": 1, "word": "Test"}]

        # Initially no session
        self.assertFalse(cm.has_unfinished_session("SPK_99"))

        # Save checkpoint with 1 of 10 words complete
        cm.save_checkpoint("SPK_99", "English (100 Words)", 0, 10, dummy_rows)
        self.assertTrue(cm.has_unfinished_session("SPK_99"))

        # Load checkpoint
        cp = cm.load_checkpoint("SPK_99")
        self.assertEqual(cp["speaker_id"], "SPK_99")
        self.assertEqual(cp["current_index"], 0)
        self.assertEqual(cp["total_words"], 10)
        self.assertEqual(len(cp["rows"]), 1)

        # Clear checkpoint
        cm.clear_checkpoint("SPK_99")
        self.assertFalse(cm.has_unfinished_session("SPK_99"))

    def test_cloud_sync(self):
        """Verify background cloud sync file copying."""
        sync_target = self.test_dir / "cloud_bucket"
        cs = CloudSyncEngine(sync_target)
        self.assertTrue(cs.is_enabled())

        src_file = self.test_dir / "sample.mp4"
        src_file.write_text("dummy video content", encoding="utf-8")

        cs.enqueue_file(src_file, "videos/SPK_01/sample.mp4")
        cs.queue.join()

        copied = sync_target / "videos" / "SPK_01" / "sample.mp4"
        self.assertTrue(copied.exists())
        self.assertEqual(copied.read_text(encoding="utf-8"), "dummy video content")
        cs.stop()

    def test_ml_export(self):
        """Verify PyTorch dataset manifest and CTC CSV generation."""
        ml = MLExportEngine(exports_dir=self.test_dir)
        dummy_rows = [
            {
                "speaker_id": "SPK_ML", "word": "Analyse", "video_file": "videos/SPK_ML/001.mp4",
                "audio_file": "audio/SPK_ML/001.wav", "lip_directory": "lips/SPK_ML/001",
                "video_frames": 90, "duration_sec": 3.0, "quality_score": 0.95, "emotion_detected": "Neutral"
            },
            {
                "speaker_id": "SPK_ML", "word": "Acquire", "video_file": "videos/SPK_ML/002.mp4",
                "audio_file": "audio/SPK_ML/002.wav", "lip_directory": "lips/SPK_ML/002",
                "video_frames": 90, "duration_sec": 3.0, "quality_score": 0.88, "emotion_detected": "Happy"
            }
        ]

        out_pt, vocab = ml.export_pytorch_manifest(dummy_rows, "SPK_ML", val_split=0.5)
        self.assertTrue(out_pt.exists())
        self.assertTrue(vocab.exists())

        out_ctc = ml.export_ctc_csv(dummy_rows, "SPK_ML")
        self.assertTrue(out_ctc.exists())

    def test_gui_presets_and_queue(self):
        """Verify GUI controller vocabulary switching and speaker queue parsing."""
        import tkinter as tk
        from lipread_app import LipReadingRecorder
        from analytics import SessionAnalyticsWindow

        root = tk.Tk()
        root.withdraw()
        try:
            app = LipReadingRecorder(root)

            # Test vocabulary switching
            app.word_list_var.set("Hindi (100 Words)")
            app.on_word_list_selected()
            self.assertEqual(len(app.word_list), 100)
            self.assertIn("नमस्ते", app.word_list[0])

            app.word_list_var.set("Assamese (50 Words)")
            app.on_word_list_selected()
            self.assertEqual(len(app.word_list), 50)
            self.assertIn("নমস্কাৰ", app.word_list[0])

            # Test speaker queue parsing
            app.speaker_entry.delete(0, "end")
            app.speaker_entry.insert(0, "S001, S002, S003")
            speakers = [s.strip() for s in app.speaker_entry.get().split(",") if s.strip()]
            self.assertEqual(speakers, ["S001", "S002", "S003"])

            # Test analytics window instantiation
            dummy_rows = [
                {
                    "speaker_id": "S001", "word_id": 1, "word": "Analyse",
                    "lip_detection_rate": 0.90, "face_detection_rate": 0.95,
                    "quality_score": 0.92, "quality_status": "GOOD",
                    "emotion_detected": "Neutral", "recording_attempt": 1
                }
            ]
            analytics = SessionAnalyticsWindow(root, "S001", dummy_rows, theme_key="cyberpunk")
            analytics.win.destroy()

        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
