"""
LipRead Studio v5.0 — Core Application
Modular, thread-safe, research-ready Visual Speech Dataset Recording System.
Features: Multi-speaker queue, multilingual word lists, real-time posture warnings,
audio oscilloscope visualizer, crash recovery checkpoints, cloud backup, and analytics.
"""

import math
import csv
import time
import json
import logging
import threading
import subprocess
import sys
import platform
import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image, ImageTk

# Safe native package imports
try:
    import cv2
except Exception as e:
    cv2 = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import mediapipe as mp
except Exception:
    mp = None

from config import (
    APP_DIR, DATASET_DIR, VIDEO_ROOT, AUDIO_ROOT, LIPS_ROOT,
    METADATA_ROOT, REPORTS_ROOT, CHECKPOINTS_ROOT, EXPORTS_ROOT,
    LOG_DIR, WORDS_ENGLISH_100, PRESET_WORD_LISTS, load_word_list_from_file,
    DEFAULT_CAMERA_INDEX, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS,
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, LIP_SIZE, LIP_MARGIN_X, LIP_MARGIN_Y,
    MOUTH_LANDMARKS, BEEP_FREQUENCY, BEEP_DURATION, START_FREQUENCY, START_DURATION,
    STOP_FREQUENCY, STOP_DURATION, QC_PASS_THRESHOLD, QC_REVIEW_THRESHOLD,
    QC_FAILED_THRESHOLD, MAX_RERECORD_ATTEMPTS, THEME_PALETTES
)
from audio_visualizer import AudioVisualizer
from emotion_detector import EmotionDetector
from qc_manager import QualityControlManager
from checkpoint_manager import CheckpointManager
from cloud_sync import CloudSyncEngine
from ml_export import MLExportEngine
from analytics import SessionAnalyticsWindow


class LipReadingRecorder:
    """Enterprise Lip Reading Dataset Recording Studio controller."""

    def __init__(self, root):
        self.root = root
        self.current_theme = "cyberpunk"
        self.palette = THEME_PALETTES[self.current_theme]

        # Operational Sub-engines
        self.audio_viz = AudioVisualizer(buffer_size=1024, sample_rate=AUDIO_SAMPLE_RATE)
        self.emotion_detector = EmotionDetector()
        self.qc_manager = QualityControlManager()
        self.checkpoint_manager = CheckpointManager(CHECKPOINTS_ROOT)
        self.cloud_sync = CloudSyncEngine()
        self.ml_exporter = MLExportEngine(EXPORTS_ROOT)

        # Word Lists
        self.word_list_name = "English (100 Words)"
        self.word_list = list(WORDS_ENGLISH_100)
        self.current_index = 0

        # Multi-speaker queue
        self.speaker_queue = []
        self.speaker_id = "S001"

        # Hardware & Thread Synchronization
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        self.latest_frame = None

        self.camera_lock = threading.RLock()
        self.face_mesh_lock = threading.RLock()
        self.audio_lock = threading.RLock()
        self.resource_lock = threading.RLock()
        self.detection_state_lock = threading.RLock()
        self.close_requested = False

        # Session State
        self.running = False
        self.session_paused = False
        self.recording_word = False
        self.qc_running = False
        self.session_thread = None
        self.stop_requested = threading.Event()
        self.pause_event = threading.Event()
        self.session_start = None

        # Telemetry & Detection State
        self.face_detected = False
        self.mouth_detected = False
        self.current_face_box = None
        self.current_lip_box = None
        self.current_landmarks = None
        self.current_emotion = "Neutral"
        self.active_warnings = []
        self.dashboard_fps = 0.0
        self.dashboard_frame_counter = 0
        self.dashboard_fps_time = time.time()
        self.dashboard_quality = 0
        self.dashboard_scan_phase = 0.0

        # Device selections
        self.camera_devices = {}
        self.audio_devices = {}
        self.selected_camera_index = DEFAULT_CAMERA_INDEX
        self.selected_audio_index = None

        # GPU acceleration flag
        self.gpu_accel_enabled = True
        if cv2 is not None:
            try:
                cv2.ocl.setUseOpenCL(True)
            except Exception:
                pass

        # Data collection buffers
        self.rows = []
        self.failed_words = []
        self.review_words = []

        # MediaPipe FaceMesh initialization
        self.mp_face_mesh = None
        self.face_mesh = None
        self._init_mediapipe()

        # Tkinter StringVars
        self._init_tk_vars()

        # Build GUI
        self.build_ui()

        # Protocol
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _init_mediapipe(self):
        self.face_landmarker = None
        self.face_mesh = None

        if mp is not None:
            # 1. Try modern mediapipe.tasks.python.vision (FaceLandmarker)
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
                model_path = APP_DIR / "face_landmarker.task"
                if model_path.exists():
                    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=False,
                        num_faces=1
                    )
                    self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                    logging.info("MediaPipe Tasks FaceLandmarker successfully initialized.")
                    return
            except Exception as e:
                logging.debug("MediaPipe Tasks initialization note: %s", e)

            # 2. Fallback to legacy solutions API if present
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                try:
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    logging.info("MediaPipe solutions FaceMesh initialized.")
                except Exception as e:
                    logging.warning("MediaPipe solutions error: %s", e)
                    self.face_mesh = None

    def _init_tk_vars(self):
        self.camera_var = tk.StringVar(value="Detecting cameras...")
        self.audio_var = tk.StringVar(value="Detecting microphones...")
        self.word_list_var = tk.StringVar(value=self.word_list_name)
        self.device_status_var = tk.StringVar(value="Ready")

        self.ready_var = tk.IntVar(value=3)
        self.record_var = tk.IntVar(value=3)
        self.pause_var = tk.IntVar(value=1)
        self.voice_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Enter Speaker ID and press START SESSION.")
        self.phase_var = tk.StringVar(value="READY")
        self.word_var = tk.StringVar(value="READY")
        self.countdown_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0)

        self.face_status_var = tk.StringVar(value="FACE: --")
        self.mouth_status_var = tk.StringVar(value="MOUTH: --")
        self.emotion_status_var = tk.StringVar(value="EMOTION: NEUTRAL")

    # ========================================================
    # GUI BUILDER
    # ========================================================

    def build_ui(self):
        p = self.palette
        self.root.title("LipRead Studio v5.0 • Enterprise Visual Speech Recorder")
        self.root.configure(bg=p["bg"])
        self.root.geometry("1480x920")
        self.root.minsize(1200, 780)

        # ----------------------------------------------------
        # TOP FIXED HEADER BAR
        # ----------------------------------------------------
        header = tk.Frame(self.root, bg=p["bg"])
        header.pack(fill="x", padx=18, pady=(10, 6))

        self.menu_btn = tk.Button(
            header, text="☰", command=self.show_navigation_menu,
            font=("Segoe UI Symbol", 18, "bold"), relief="flat", bd=0,
            bg=p["panel"], fg=p["accent_secondary"], activebackground=p["panel_alt"],
            cursor="hand2", width=3, pady=2
        )
        self.menu_btn.pack(side="left", padx=(0, 14))

        brand_box = tk.Frame(header, bg=p["bg"])
        brand_box.pack(side="left", fill="x", expand=True)

        tk.Label(
            brand_box, text="◉  LIPREAD STUDIO v5.0",
            font=("Segoe UI", 21, "bold"), fg=p["accent_secondary"], bg=p["bg"]
        ).pack(anchor="w")

        self.sub_brand_label = tk.Label(
            brand_box, text="MULTI-MODAL DATASET ENGINE  •  VIDEO + AUDIO + LIP CROPS + ML MANIFESTS",
            font=("Segoe UI", 8, "bold"), fg=p["text_muted"], bg=p["bg"]
        )
        self.sub_brand_label.pack(anchor="w")

        # Top Right Controls
        control_bar = tk.Frame(
            header, bg=p["panel"], highlightthickness=1,
            highlightbackground=p["border"], padx=8, pady=6
        )
        control_bar.pack(side="right")

        self.pause_btn = tk.Button(
            control_bar, text="Ⅱ  PAUSE", command=self.toggle_session_pause,
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            fg=p["warning"], bg=p["warning_bg"], cursor="hand2", padx=14, pady=6, state="disabled"
        )
        self.pause_btn.pack(side="left", padx=4)

        self.stop_btn = tk.Button(
            control_bar, text="■  STOP", command=self.stop_session, state="disabled",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            fg="#FFFFFF", bg=p["danger_bg"], cursor="hand2", padx=14, pady=6
        )
        self.stop_btn.pack(side="left", padx=4)

        self.start_btn = tk.Button(
            control_bar, text="▶  START SESSION", command=self.start_session,
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            fg="#070B14", bg=p["accent"], activebackground="#7BFFE5",
            cursor="hand2", padx=16, pady=6
        )
        self.start_btn.pack(side="left", padx=4)

        # ----------------------------------------------------
        # WORKSPACE AREA
        # ----------------------------------------------------
        workspace = tk.Frame(self.root, bg=p["bg"])
        workspace.pack(fill="both", expand=True, padx=18, pady=(0, 12))

        # LEFT PANEL: Control Center
        self.left_panel = tk.Frame(
            workspace, bg=p["panel"], width=285,
            highlightthickness=1, highlightbackground=p["border"]
        )
        self.left_panel.pack(side="left", fill="y", padx=(0, 12))
        self.left_panel.pack_propagate(False)

        self._build_left_panel()

        # CENTER / RIGHT: Camera Preview & Speaker Telemetry
        center = tk.Frame(workspace, bg=p["bg"])
        center.pack(side="left", fill="both", expand=True)

        self._build_center_area(center)

        # Start device scanning
        self.root.after(150, self.refresh_cameras)
        self.root.after(250, self.refresh_audio_devices)
        self.root.after(100, self._telemetry_refresh_loop)

    def _build_left_panel(self):
        p = self.palette
        lp = self.left_panel

        tk.Label(
            lp, text="CONTROL CENTER", bg=p["panel"], fg=p["text"],
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(
            lp, text="DEVICES • SESSIONS • VOCABULARY", bg=p["panel"], fg=p["text_muted"],
            font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", padx=14, pady=(0, 10))

        # 1. Camera Device
        self._section_hdr(lp, "CAMERA DEVICE", p["accent_secondary"])
        self.camera_combo = ttk.Combobox(lp, textvariable=self.camera_var, state="readonly", width=28)
        self.camera_combo.pack(fill="x", padx=14, pady=(4, 4))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        cam_btns = tk.Frame(lp, bg=p["panel"])
        cam_btns.pack(fill="x", padx=14, pady=(0, 10))
        self._mini_btn(cam_btns, "↻ Refresh", self.refresh_cameras).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._mini_btn(cam_btns, "Test", self.test_camera).pack(side="left", fill="x", expand=True, padx=(3, 0))

        # 2. Audio Device
        self._section_hdr(lp, "MICROPHONE / BLUETOOTH", p["accent"])
        self.audio_combo = ttk.Combobox(lp, textvariable=self.audio_var, state="readonly", width=28)
        self.audio_combo.pack(fill="x", padx=14, pady=(4, 4))
        self.audio_combo.bind("<<ComboboxSelected>>", self.on_audio_selected)

        aud_btns = tk.Frame(lp, bg=p["panel"])
        aud_btns.pack(fill="x", padx=14, pady=(0, 10))
        self._mini_btn(aud_btns, "↻ Refresh", self.refresh_audio_devices).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._mini_btn(aud_btns, "Test", self.test_microphone).pack(side="left", fill="x", expand=True, padx=(3, 0))

        # 3. Multi-Speaker Queue
        self._section_hdr(lp, "SPEAKER QUEUE", "#C4B5FD")
        tk.Label(
            lp, text="Speaker IDs (comma-separated for multi-speaker)",
            bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7)
        ).pack(anchor="w", padx=14, pady=(3, 1))
        self.speaker_entry = ttk.Entry(lp)
        self.speaker_entry.pack(fill="x", padx=14, pady=(0, 10))
        self.speaker_entry.insert(0, "S001")

        # 4. Word List Selection
        self._section_hdr(lp, "VOCABULARY LIST", "#FCD34D")
        word_list_choices = list(PRESET_WORD_LISTS.keys()) + ["Load Custom File..."]
        self.word_list_combo = ttk.Combobox(
            lp, textvariable=self.word_list_var, values=word_list_choices, state="readonly", width=28
        )
        self.word_list_combo.pack(fill="x", padx=14, pady=(4, 4))
        self.word_list_combo.bind("<<ComboboxSelected>>", self.on_word_list_selected)

        self.word_count_lbl = tk.Label(
            lp, text=f"Active: {len(self.word_list)} words", bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7)
        )
        self.word_count_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        # 5. Timing Parameters
        self._section_hdr(lp, "TIMING PARAMETERS", p["warning"])
        timing = tk.Frame(lp, bg=p["panel"])
        timing.pack(fill="x", padx=14, pady=(3, 6))
        for lbl, var, frm, to in [
            ("READY (s)", self.ready_var, 1, 10),
            ("RECORD (s)", self.record_var, 1, 10),
            ("PAUSE (s)", self.pause_var, 0, 5)
        ]:
            cell = tk.Frame(timing, bg=p["panel"])
            cell.pack(fill="x", pady=2)
            tk.Label(cell, text=lbl, bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7, "bold"), width=11, anchor="w").pack(side="left")
            ttk.Spinbox(cell, from_=frm, to=to, width=5, textvariable=var).pack(side="right")

        ttk.Checkbutton(lp, text="Spoken voice prompts", variable=self.voice_var).pack(anchor="w", padx=14, pady=(4, 8))

        # Status & Cloud Sync badge
        self.device_status_lbl = tk.Label(
            lp, textvariable=self.device_status_var, font=("Segoe UI", 8, "bold"),
            fg=p["accent"], bg=p["panel"], wraplength=250, justify="left"
        )
        self.device_status_lbl.pack(anchor="w", padx=14, pady=(4, 2))

        self.sync_status_lbl = tk.Label(
            lp, text="☁ Backup: Ready", font=("Segoe UI", 7),
            fg=p["text_muted"], bg=p["panel"]
        )
        self.sync_status_lbl.pack(anchor="w", padx=14, pady=(0, 8))

    def _build_center_area(self, center):
        p = self.palette

        # 1. SPEAKER WORD PROMPT (Prominently displayed above camera)
        prompt_box = tk.Frame(
            center, bg=p["panel"], highlightthickness=1,
            highlightbackground=p["border"], padx=18, pady=8
        )
        prompt_box.pack(fill="x", pady=(0, 8))

        self.word_prompt_header = tk.Label(
            prompt_box, text="GET READY!", bg=p["panel"], fg=p["accent"],
            font=("Segoe UI", 12, "bold")
        )
        self.word_prompt_header.pack()

        self.word_prompt_word = tk.Label(
            prompt_box, text="READY", bg=p["panel"], fg=p["text"],
            font=("Segoe UI", 28, "bold")
        )
        self.word_prompt_word.pack()

        self.word_prompt_meta = tk.Label(
            prompt_box, text=f"WORD 1 / {len(self.word_list)}", bg=p["panel"], fg=p["text_muted"],
            font=("Segoe UI", 8, "bold")
        )
        self.word_prompt_meta.pack()

        # 2. CAMERA PREVIEW
        preview_container = tk.Frame(
            center, bg=p["preview_bg"], highlightthickness=1,
            highlightbackground=p["border"]
        )
        preview_container.pack(fill="both", expand=True)

        self.preview_label = tk.Label(preview_container, bg=p["preview_bg"])
        self.preview_label.pack(fill="both", expand=True)

        # 3. DETECTION & ACTIVE WARNINGS BAR
        sub_bar = tk.Frame(center, bg=p["bg"])
        sub_bar.pack(fill="x", pady=(6, 4))

        self.phase_label = tk.Label(
            sub_bar, textvariable=self.phase_var, bg=p["bg"], fg=p["accent_secondary"],
            font=("Segoe UI", 10, "bold")
        )
        self.phase_label.pack(side="left")

        tk.Label(sub_bar, text="  •  ", bg=p["bg"], fg=p["border"]).pack(side="left")
        tk.Label(sub_bar, textvariable=self.face_status_var, bg=p["bg"], fg=p["success"], font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Label(sub_bar, text="  ", bg=p["bg"]).pack(side="left")
        tk.Label(sub_bar, textvariable=self.mouth_status_var, bg=p["bg"], fg=p["accent"], font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Label(sub_bar, text="  ", bg=p["bg"]).pack(side="left")
        tk.Label(sub_bar, textvariable=self.emotion_status_var, bg=p["bg"], fg="#C4B5FD", font=("Segoe UI", 9, "bold")).pack(side="left")

        self.status_label = tk.Label(
            sub_bar, textvariable=self.status_var, bg=p["bg"], fg=p["text_muted"],
            font=("Segoe UI", 8), anchor="e"
        )
        self.status_label.pack(side="right")

        # 4. TELEMETRY STRIP & AUDIO OSCILLOSCOPE
        telemetry = tk.Frame(
            center, bg=p["panel"], highlightthickness=1,
            highlightbackground=p["border"], padx=10, pady=6
        )
        telemetry.pack(fill="x")

        self._telemetry_box(telemetry, "QUALITY", "quality_value", "0%", 0)
        self._telemetry_box(telemetry, "FPS", "fps_value", "0.0", 1)
        self._telemetry_box(telemetry, "PROGRESS", "words_value", f"0 / {len(self.word_list)}", 2)

        # Live Oscilloscope Waveform Canvas
        scope_box = tk.Frame(telemetry, bg=p["panel"])
        scope_box.grid(row=0, column=3, sticky="nsew", padx=8)
        tk.Label(scope_box, text="LIVE MICROPHONE WAVEFORM", bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7, "bold")).pack(anchor="w")

        self.waveform_canvas = tk.Canvas(
            scope_box, width=220, height=36, bg=p["panel_alt"], highlightthickness=0
        )
        self.waveform_canvas.pack(fill="x", pady=(2, 0))

        telemetry.columnconfigure(4, weight=1)

        # Session Progress Bar
        prog_box = tk.Frame(telemetry, bg=p["panel"])
        prog_box.grid(row=0, column=4, sticky="ew", padx=10)
        tk.Label(prog_box, text="SESSION COMPLETION", bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(prog_box, maximum=100, variable=self.progress_var)
        self.progress_bar.pack(fill="x", pady=(5, 0))

    def _section_hdr(self, parent, text, fg):
        tk.Label(
            parent, text="▸  " + text, bg=self.palette["panel"], fg=fg,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=14, pady=(4, 0))

    def _mini_btn(self, parent, text, command):
        p = self.palette
        return tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 8, "bold"), relief="flat", bd=0,
            bg=p["btn_normal"], fg=p["text"], activebackground=p["btn_active"],
            cursor="hand2", padx=4, pady=4
        )

    def _telemetry_box(self, parent, title, attr, default, col):
        p = self.palette
        box = tk.Frame(parent, bg=p["panel"], width=90)
        box.grid(row=0, column=col, sticky="ns", padx=(6 if col == 0 else 4, 4))
        tk.Label(box, text=title, bg=p["panel"], fg=p["text_muted"], font=("Segoe UI", 7, "bold")).pack(anchor="w")
        lbl = tk.Label(box, text=default, bg=p["panel"], fg=p["text"], font=("Segoe UI", 12, "bold"))
        lbl.pack(anchor="w", pady=(2, 0))
        setattr(self, attr, lbl)

    # ========================================================
    # TELEMETRY REFRESH LOOP
    # ========================================================

    def _telemetry_refresh_loop(self):
        try:
            if hasattr(self, "quality_value"):
                self.quality_value.configure(text=f"{int(self.dashboard_quality)}%")
            if hasattr(self, "fps_value"):
                self.fps_value.configure(text=f"{float(self.dashboard_fps):.1f}")
            if hasattr(self, "words_value"):
                total = len(self.word_list)
                curr = min(self.current_index, total)
                self.words_value.configure(text=f"{curr} / {total}")
            if hasattr(self, "sync_status_lbl"):
                self.sync_status_lbl.configure(text=self.cloud_sync.get_status_string())

            # Draw waveform
            if hasattr(self, "waveform_canvas"):
                self.audio_viz.render_to_canvas(
                    self.waveform_canvas, 0, 0, 220, 36, self.palette, tag="wave"
                )
        except Exception:
            pass

        try:
            self.root.after(100, self._telemetry_refresh_loop)
        except Exception:
            pass

    # ========================================================
    # HARDWARE DETECTION
    # ========================================================

    def refresh_cameras(self):
        if cv2 is None:
            self.camera_var.set("OpenCV not installed")
            self.device_status_var.set("⚠ OpenCV is not available")
            return

        devices = {}
        values = []
        for index in range(8):
            cap = None
            try:
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    ok, _ = cap.read()
                    if ok:
                        label = f"Camera {index}  (USB/Built-in)"
                        devices[label] = index
                        values.append(label)
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()

        self.camera_devices = devices
        self.camera_combo["values"] = values

        if values:
            current = next((v for v, i in devices.items() if i == self.selected_camera_index), values[0])
            self.camera_var.set(current)
            self.device_status_var.set(f"● {len(values)} camera(s) detected")
        else:
            self.camera_var.set("No camera detected")
            self.device_status_var.set("⚠ No camera detected")

    def on_camera_selected(self, event=None):
        label = self.camera_var.get()
        if label in self.camera_devices:
            self.selected_camera_index = self.camera_devices[label]
            self.device_status_var.set(f"● Selected: Camera {self.selected_camera_index}")

    def refresh_audio_devices(self):
        if sd is None:
            self.audio_var.set("sounddevice not installed")
            self.device_status_var.set("⚠ Audio system unavailable")
            return

        devices = {}
        values = []
        try:
            with self.audio_lock:
                audio_device_list = list(sd.query_devices())
            for index, device in enumerate(audio_device_list):
                if int(device.get("max_input_channels", 0)) > 0:
                    name = str(device.get("name", f"Input {index}")).strip()
                    label = f"{name} (Ch {index})"
                    devices[label] = index
                    values.append(label)
        except Exception as e:
            self.audio_var.set("Audio error")
            self.device_status_var.set(f"⚠ Audio scan failed: {e}")
            return

        self.audio_devices = devices
        self.audio_combo["values"] = values

        if values:
            try:
                default_input = sd.default.device[0]
            except Exception:
                default_input = None
            current = next((v for v, i in devices.items() if i == default_input), values[0])
            self.selected_audio_index = devices[current]
            self.audio_var.set(current)
            self.device_status_var.set(f"● {len(values)} microphone(s) detected")
            self.audio_viz.start_passive_monitor(self.selected_audio_index)
        else:
            self.audio_var.set("No microphone detected")

    def on_audio_selected(self, event=None):
        label = self.audio_var.get()
        if label in self.audio_devices:
            self.selected_audio_index = self.audio_devices[label]
            self.audio_viz.start_passive_monitor(self.selected_audio_index)
            self.device_status_var.set("● Audio input updated")

    def test_camera(self):
        if cv2 is None:
            messagebox.showerror("Error", "OpenCV is not installed.")
            return
        if self.camera_running:
            messagebox.showinfo("Camera Test", "Camera is already actively streaming.")
            return
        try:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.selected_camera_index, backend)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                messagebox.showinfo("Camera Test", f"Camera {self.selected_camera_index} is operational!\nResolution: {w}×{h}")
            else:
                raise RuntimeError("Could not capture test frame.")
        except Exception as e:
            messagebox.showerror("Camera Test Failed", str(e))

    def test_microphone(self):
        if sd is None or self.selected_audio_index is None:
            messagebox.showwarning("Microphone", "Select a valid microphone first.")
            return
        try:
            info = sd.query_devices(self.selected_audio_index)
            with self.audio_lock:
                rec = sd.rec(int(0.5 * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32", device=self.selected_audio_index)
                sd.wait()
            rms = float(np.sqrt(np.mean(rec ** 2)))
            messagebox.showinfo("Microphone Test", f"Microphone {info['name']} is working!\nSignal RMS: {rms:.4f}")
        except Exception as e:
            messagebox.showerror("Microphone Test Failed", str(e))

    # ========================================================
    # VOCABULARY SELECTION
    # ========================================================

    def on_word_list_selected(self, event=None):
        choice = self.word_list_var.get()
        if choice == "Load Custom File...":
            path = filedialog.askopenfilename(
                title="Select Custom Word List",
                filetypes=[("Text/CSV/JSON", "*.csv *.txt *.json"), ("All Files", "*.*")]
            )
            if path:
                try:
                    words = load_word_list_from_file(Path(path))
                    self.word_list = words
                    self.word_list_name = Path(path).stem
                    self.word_list_var.set(f"Custom ({len(words)} words)")
                    self.word_count_lbl.configure(text=f"Active: {len(words)} words")
                    self.status_var.set(f"Loaded custom vocabulary: {len(words)} words")
                except Exception as e:
                    messagebox.showerror("File Error", f"Failed to load word list: {e}")
                    self.word_list_var.set(self.word_list_name)
            else:
                self.word_list_var.set(self.word_list_name)
        elif choice in PRESET_WORD_LISTS:
            self.word_list = list(PRESET_WORD_LISTS[choice])
            self.word_list_name = choice
            self.word_count_lbl.configure(text=f"Active: {len(self.word_list)} words")
            self.status_var.set(f"Switched vocabulary: {choice}")

    # ========================================================
    # VOICE ENGINE
    # ========================================================

    def speak(self, text: str):
        """Asynchronously speak prompt cues via Windows SpeechSynthesizer."""
        if not self.voice_var.get() or sys.platform != "win32":
            return

        def _worker():
            try:
                escaped = text.replace("'", "''")
                cmd = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Rate = 1; "
                    f"$s.Speak('{escaped}');"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7
                )
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="LipReadVoiceSynthesizer").start()

    def play_tone(self, freq: int, dur: float):
        """Play synthetic tone safely without overlapping recording stream."""
        if sd is None or dur <= 0:
            return
        samples = max(1, int(AUDIO_SAMPLE_RATE * dur))
        t = np.linspace(0, dur, samples, endpoint=False, dtype=np.float32)
        audio = (0.22 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        fade = min(250, samples // 4)
        if fade > 0:
            audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        try:
            with self.audio_lock:
                sd.play(audio, AUDIO_SAMPLE_RATE)
                sd.wait()
        except Exception:
            pass

    # ========================================================
    # CAMERA STREAMING & DETECTION
    # ========================================================

    def start_camera(self):
        if cv2 is None:
            return
        with self.resource_lock:
            if self.camera_running:
                return

            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.selected_camera_index, backend)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera device.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

            with self.camera_lock:
                self.cap = cap
                self.latest_frame = None
                self.camera_running = True

            self.camera_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
            self.camera_thread.start()

            self.root.after(30, self.update_preview)

    def _capture_loop(self):
        while self.camera_running:
            with self.camera_lock:
                cap = self.cap
                if cap is None:
                    break
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.latest_frame = frame

            if not ok:
                time.sleep(0.02)
                continue
            time.sleep(0.001)

    def get_latest_frame(self):
        with self.camera_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop_camera(self):
        with self.resource_lock:
            self.camera_running = False
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2.0)

            with self.camera_lock:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                self.cap = None
                self.latest_frame = None
                self.camera_thread = None

    def detect_face_and_mouth(self, frame):
        """Thread-safe FaceMesh landmark detection and ROI computation."""
        if frame is None or (self.face_landmarker is None and self.face_mesh is None):
            return None, None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = None

        try:
            with self.face_mesh_lock:
                if self.face_landmarker is not None:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = self.face_landmarker.detect(mp_image)
                    if res and res.face_landmarks:
                        landmarks = res.face_landmarks[0]
                elif self.face_mesh is not None:
                    results = self.face_mesh.process(rgb)
                    if results and results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
        except Exception:
            return None, None, None

        if landmarks is None:
            with self.detection_state_lock:
                self.face_detected = False
                self.mouth_detected = False
                self.current_face_box = None
                self.current_lip_box = None
                self.current_landmarks = None
            return None, None, None

        pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        h, w = frame.shape[:2]

        fx = [int(lm.x * w) for lm in pts]
        fy = [int(lm.y * h) for lm in pts]
        face_box = (max(0, min(fx)), max(0, min(fy)), min(w - 1, max(fx)), min(h - 1, max(fy)))

        xs = [int(pts[idx].x * w) for idx in MOUTH_LANDMARKS if 0 <= idx < len(pts)]
        ys = [int(pts[idx].y * h) for idx in MOUTH_LANDMARKS if 0 <= idx < len(pts)]

        lip_box = None
        if xs and ys:
            mw = max(xs) - min(xs)
            mh = max(ys) - min(ys)
            if mw >= 5 and mh >= 3:
                mx = int(mw * LIP_MARGIN_X)
                my = int(mh * LIP_MARGIN_Y)
                lip_box = (max(0, min(xs) - mx), max(0, min(ys) - my), min(w - 1, max(xs) + mx), min(h - 1, max(ys) + my))

        with self.detection_state_lock:
            self.face_detected = True
            self.mouth_detected = (lip_box is not None)
            self.current_face_box = face_box
            self.current_lip_box = lip_box
            self.current_landmarks = landmarks

            # Real-time emotion classification
            emo, conf, _ = self.emotion_detector.analyze_landmarks(landmarks, w, h)
            self.current_emotion = emo

            # Real-time warnings (posture, lighting, occlusion)
            self.active_warnings = self.qc_manager.evaluate_realtime_warnings(frame, landmarks, lip_box)

        return face_box, lip_box, landmarks

    # ========================================================
    # PREVIEW PROCESSING & HUD
    # ========================================================

    def process_preview_frame(self, frame):
        display = frame.copy()
        face_box, lip_box, landmarks = self.detect_face_and_mouth(frame)
        p = self.palette

        # FPS calculation
        now = time.time()
        self.dashboard_frame_counter += 1
        el = now - self.dashboard_fps_time
        if el >= 1.0:
            self.dashboard_fps = self.dashboard_frame_counter / el
            self.dashboard_frame_counter = 0
            self.dashboard_fps_time = now

        # Quality score
        self.dashboard_quality = int(
            (35 if self.face_detected else 0) +
            (45 if self.mouth_detected else 0) +
            (20 if self.audio_viz.current_rms > 0.02 else 0)
        )

        h, w = display.shape[:2]

        # 1. Face Box & Brackets
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (255, 230, 70), 1)

            corner = min(30, (fx2 - fx1) // 6)
            for x, y, sx, sy in [(fx1, fy1, 1, 1), (fx2, fy1, -1, 1), (fx1, fy2, 1, -1), (fx2, fy2, -1, -1)]:
                cv2.line(display, (x, y), (x + sx * corner, y), (255, 230, 70), 3)
                cv2.line(display, (x, y), (x, y + sy * corner), (255, 230, 70), 3)

        # 2. Glowing Lip ROI Box
        if lip_box is not None:
            lx1, ly1, lx2, ly2 = lip_box
            glow = display.copy()
            cv2.rectangle(glow, (lx1, ly1), (lx2, ly2), (80, 255, 210), 6)
            display = cv2.addWeighted(glow, 0.20, display, 0.80, 0)
            cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (80, 255, 210), 2)
            cv2.putText(display, "LIP ROI", (lx1, max(22, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 210), 2)

        # 3. Warning Badges Banner (Lighting, Tilt, Occlusion)
        if self.active_warnings:
            warn_txt = "  •  ".join(self.active_warnings)
            tw = cv2.getTextSize(warn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            wx = (w - tw) // 2
            cv2.rectangle(display, (wx - 16, 12), (wx + tw + 16, 46), (15, 20, 160), -1)
            cv2.putText(display, warn_txt, (wx, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display

    def update_preview(self):
        if not self.camera_running:
            return

        frame = self.get_latest_frame()
        if frame is not None:
            if self.recording_word:
                # Lightweight rendering during recording
                display = frame.copy()
                with self.detection_state_lock:
                    if self.current_face_box:
                        fx1, fy1, fx2, fy2 = self.current_face_box
                        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (255, 230, 70), 1)
                    if self.current_lip_box:
                        lx1, ly1, lx2, ly2 = self.current_lip_box
                        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (80, 255, 210), 2)
            else:
                display = self.process_preview_frame(frame)

            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display)
            img.thumbnail((1020, 620), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.preview_photo)

            with self.detection_state_lock:
                self.face_status_var.set("FACE: OK" if self.face_detected else "FACE: SEARCH")
                self.mouth_status_var.set("MOUTH: OK" if self.mouth_detected else "MOUTH: SEARCH")
                self.emotion_status_var.set(f"EMOTION: {self.current_emotion.upper()}")

        if self.camera_running:
            self.root.after(30, self.update_preview)

    # ========================================================
    # RECORDING ENGINE
    # ========================================================

    def start_session(self):
        if self.running:
            return

        raw_id = self.speaker_entry.get().strip()
        if not raw_id:
            messagebox.showwarning("Speaker ID", "Enter a Speaker ID (e.g. S001).")
            return

        # Parse speaker queue
        speakers = [s.strip() for s in raw_id.split(",") if s.strip()]
        self.speaker_queue = speakers
        self.speaker_id = self.speaker_queue.pop(0)

        # Crash recovery check
        if self.checkpoint_manager.has_unfinished_session(self.speaker_id):
            cp = self.checkpoint_manager.load_checkpoint(self.speaker_id)
            idx = cp.get("current_index", 0)
            tot = cp.get("total_words", len(self.word_list))
            ans = messagebox.askyesno(
                "Resume Unfinished Session",
                f"Found unfinished recording session for {self.speaker_id}!\n"
                f"Completed: {idx + 1} of {tot} words.\n\n"
                f"Would you like to resume from Word #{idx + 2}?"
            )
            if ans:
                self.rows = cp.get("rows", [])
                self.current_index = idx + 1
            else:
                self.checkpoint_manager.clear_checkpoint(self.speaker_id)
                self.rows = []
                self.current_index = 0
        else:
            self.rows = []
            self.current_index = 0

        # Destination directories
        self.video_dir = VIDEO_ROOT / self.speaker_id
        self.audio_dir = AUDIO_ROOT / self.speaker_id
        self.lips_dir = LIPS_ROOT / self.speaker_id
        self.csv_path = METADATA_ROOT / f"{self.speaker_id}.csv"

        for d in [self.video_dir, self.audio_dir, self.lips_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Start Camera
        try:
            self.start_camera()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            return

        self.running = True
        self.session_paused = False
        self.stop_requested.clear()
        self.pause_event.clear()
        self.session_start = time.perf_counter()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.pause_btn.configure(state="normal", text="Ⅱ  PAUSE")
        self.speaker_entry.configure(state="disabled")

        # Launch recording thread
        def _worker():
            try:
                self._record_session_loop()
            except Exception as e:
                logging.exception("Session worker failure")
                self.root.after(0, self.handle_error, str(e))
            finally:
                self.root.after(0, self._on_session_finished)

        self.session_thread = threading.Thread(target=_worker, daemon=False, name="LipReadSessionWorker")
        self.session_thread.start()

    def _record_session_loop(self):
        total = len(self.word_list)
        ready_sec = max(1, self.ready_var.get())
        record_sec = max(1, self.record_var.get())
        pause_sec = max(0, self.pause_var.get())

        for i in range(self.current_index, total):
            if not self._check_pause_and_stop():
                break

            self.current_index = i
            word = self.word_list[i]

            # 1. Prepare word UI
            self.root.after(0, self._update_word_display, i, total, word, "GET READY")

            # 2. Get Ready Countdown
            self.speak(f"Get ready.")
            for c in range(ready_sec, 0, -1):
                if not self.running or self.stop_requested.is_set():
                    break
                self.root.after(0, self.countdown_var.set, str(c))
                self.play_tone(BEEP_FREQUENCY, BEEP_DURATION)
                time.sleep(1.0)

            if not self._check_pause_and_stop():
                break

            # 3. Speak Now Prompt
            self.root.after(0, self._update_word_display, i, total, word, "SPEAK NOW!")
            self.play_tone(START_FREQUENCY, START_DURATION)
            self.speak(f"Say {word.split('(')[0].strip()}")

            time.sleep(0.2)
            if not self._check_pause_and_stop():
                break

            # 4. Record Word (Video + Audio + Lip crops)
            result = self._record_single_word(i, word, record_sec)

            # 5. Stop Sound & Prompt
            self.play_tone(STOP_FREQUENCY, STOP_DURATION)
            self.root.after(0, self.phase_var.set, "STOP")
            self.root.after(0, self.countdown_var.set, "")

            # 6. Quality & Metadata
            q_status, q_score = self.qc_manager.calculate_word_quality(
                result["face_rate"], result["lip_rate"], result.get("audio_rms", 0.0)
            )

            row = {
                "speaker_id": self.speaker_id,
                "word_id": i + 1,
                "word": word,
                "video_file": result["video_path"],
                "audio_file": result["audio_path"],
                "lip_directory": result["lip_dir"],
                "video_frames": result["frames"],
                "face_detected_frames": result["face_frames"],
                "lip_detected_frames": result["lip_frames"],
                "face_detection_rate": result["face_rate"],
                "lip_detection_rate": result["lip_rate"],
                "quality_status": q_status,
                "quality_score": q_score,
                "emotion_detected": result.get("emotion", "Neutral"),
                "recording_attempt": 1,
                "video_fps": result["fps"],
                "duration_sec": result["duration"]
            }
            self.rows.append(row)

            # 7. Incremental Checkpoint & Cloud Sync
            self.root.after(0, self._save_csv)
            self.checkpoint_manager.save_checkpoint(
                self.speaker_id, self.word_list_name, i, total, self.rows
            )
            self.cloud_sync.enqueue_file(Path(DATASET_DIR) / result["video_path"], result["video_path"])
            self.cloud_sync.enqueue_file(Path(DATASET_DIR) / result["audio_path"], result["audio_path"])

            # 8. Inter-word pause
            if pause_sec > 0:
                self.root.after(0, self.phase_var.set, "PAUSE")
                for pc in range(pause_sec, 0, -1):
                    if not self.running or self.stop_requested.is_set():
                        break
                    self.root.after(0, self.countdown_var.set, str(pc))
                    time.sleep(1.0)

        # End of session loop
        if self.running and not self.stop_requested.is_set():
            self.speak("Session completed. Great job!")

    def _record_single_word(self, word_idx: int, word: str, duration_sec: int):
        clean_word = "".join(c for c in word if c.isalnum() or c in "_-")
        video_name = f"{word_idx + 1:03d}_{clean_word}.mp4"
        audio_name = f"{word_idx + 1:03d}_{clean_word}.wav"
        lip_folder = f"{word_idx + 1:03d}_{clean_word}"

        v_path = self.video_dir / video_name
        a_path = self.audio_dir / audio_name
        l_path = self.lips_dir / lip_folder
        l_path.mkdir(parents=True, exist_ok=True)

        # Clear old lip crops if re-recording
        for f in l_path.glob("*.jpg"):
            try:
                f.unlink()
            except Exception:
                pass

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(v_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

        self.recording_word = True
        audio_samples = []

        def _rec_audio():
            if sd is not None and self.selected_audio_index is not None:
                try:
                    s = int(duration_sec * AUDIO_SAMPLE_RATE)
                    with self.audio_lock:
                        data = sd.rec(s, samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32", device=self.selected_audio_index)
                        sd.wait()
                    if sf is not None:
                        sf.write(str(a_path), data, AUDIO_SAMPLE_RATE)
                    audio_samples.append(data)
                except Exception:
                    pass

        a_thread = threading.Thread(target=_rec_audio, daemon=False, name="LipReadAudioThread")
        a_thread.start()

        start = time.perf_counter()
        end_time = start + duration_sec
        frame_idx = 0
        face_count = 0
        lip_count = 0
        recorded_emotions = []

        try:
            while self.running and time.perf_counter() < end_time:
                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                writer.write(frame)

                # Process face and lip ROI
                _, lip_box, _ = self.detect_face_and_mouth(frame)
                with self.detection_state_lock:
                    if self.face_detected:
                        face_count += 1
                    recorded_emotions.append(self.current_emotion)

                if lip_box is not None:
                    x1, y1, x2, y2 = lip_box
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_res = cv2.resize(crop, (LIP_SIZE, LIP_SIZE), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(str(l_path / f"{frame_idx:04d}.jpg"), crop_res)
                            lip_count += 1

                frame_idx += 1
        finally:
            writer.release()
            self.recording_word = False
            if a_thread.is_alive():
                a_thread.join(timeout=3.0)

        tot_dur = round(time.perf_counter() - start, 2)
        dom_emotion = max(set(recorded_emotions), key=recorded_emotions.count) if recorded_emotions else "Neutral"
        rms = float(np.sqrt(np.mean(audio_samples[0] ** 2))) if audio_samples and len(audio_samples[0]) > 0 else 0.0

        return {
            "video_path": str(v_path.relative_to(DATASET_DIR)),
            "audio_path": str(a_path.relative_to(DATASET_DIR)),
            "lip_dir": str(l_path.relative_to(DATASET_DIR)),
            "frames": frame_idx,
            "face_frames": face_count,
            "lip_frames": lip_count,
            "face_rate": round(face_count / max(1, frame_idx), 3),
            "lip_rate": round(lip_count / max(1, frame_idx), 3),
            "fps": VIDEO_FPS,
            "duration": tot_dur,
            "emotion": dom_emotion,
            "audio_rms": rms
        }

    def _check_pause_and_stop(self):
        while self.running and self.pause_event.is_set():
            time.sleep(0.1)
        return self.running and not self.stop_requested.is_set()

    def _update_word_display(self, idx, total, word, phase):
        self.phase_var.set(phase)
        self.word_var.set(word.upper())
        self.progress_var.set(((idx + 1) / total) * 100.0)

        # Update word prompt card above camera
        self.word_prompt_header.configure(
            text=phase, fg=self.palette["danger"] if "SPEAK" in phase else self.palette["accent"]
        )
        self.word_prompt_word.configure(text=word.upper())
        self.word_prompt_meta.configure(text=f"WORD {idx + 1} / {total}")
        self.status_var.set(f"Word {idx + 1} of {total}: {word}")

    def _save_csv(self):
        fields = [
            "speaker_id", "word_id", "word", "video_file", "audio_file", "lip_directory",
            "video_frames", "face_detected_frames", "lip_detected_frames", "face_detection_rate",
            "lip_detection_rate", "quality_status", "quality_score", "emotion_detected",
            "recording_attempt", "video_fps", "duration_sec"
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in fields})

    def _on_session_finished(self):
        self.running = False
        self.recording_word = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.pause_btn.configure(state="disabled", text="Ⅱ  PAUSE")
        self.speaker_entry.configure(state="normal")

        # Finalize CSV & Generate HTML Report
        if self.rows:
            self._save_csv()
            self.qc_manager.generate_html_report(self.speaker_id, self.rows)
            self.checkpoint_manager.clear_checkpoint(self.speaker_id)

            # Auto open Analytics Dashboard
            self.open_analytics_dashboard()

        # Multi-speaker queue check
        if self.speaker_queue:
            next_spk = self.speaker_queue[0]
            ans = messagebox.askyesno(
                "Next Speaker in Queue",
                f"Session for {self.speaker_id} complete!\n\n"
                f"Ready to begin next queued speaker: {next_spk}?"
            )
            if ans:
                self.speaker_entry.delete(0, "end")
                self.speaker_entry.insert(0, ", ".join(self.speaker_queue))
                self.start_session()

    # ========================================================
    # CONTROLS: PAUSE, STOP, RETRY
    # ========================================================

    def toggle_session_pause(self):
        if not self.running:
            return
        if self.session_paused:
            self.session_paused = False
            self.pause_event.clear()
            self.pause_btn.configure(text="Ⅱ  PAUSE", bg=self.palette["warning_bg"])
            self.speak("Session resumed.")
            self.status_var.set("Session resumed.")
        else:
            self.session_paused = True
            self.pause_event.set()
            self.pause_btn.configure(text="▶  RESUME", bg=self.palette["btn_normal"])
            self.speak("Recording paused.")
            self.status_var.set("Session paused. Press RESUME to continue.")

    def stop_session(self):
        if not self.running:
            return
        self.running = False
        self.stop_requested.set()
        self.session_paused = False
        self.pause_event.clear()
        self.status_var.set("Stopping recording session safely...")

    def rerecord_single_word(self, word_id: int, word_name: str):
        """Immediately re-record a specific word from the analytics table."""
        if self.running:
            messagebox.showwarning("Active Session", "Stop the current session before re-recording.")
            return

        idx = word_id - 1
        record_sec = max(1, self.record_var.get())

        def _retry_worker():
            try:
                self.running = True
                self.speak(f"Re-recording {word_name.split('(')[0]}")
                self.root.after(0, self._update_word_display, idx, len(self.word_list), word_name, "SPEAK NOW!")
                self.play_tone(START_FREQUENCY, START_DURATION)

                result = self._record_single_word(idx, word_name, record_sec)
                self.play_tone(STOP_FREQUENCY, STOP_DURATION)

                q_status, q_score = self.qc_manager.calculate_word_quality(
                    result["face_rate"], result["lip_rate"], result.get("audio_rms", 0.0)
                )

                # Update row
                for r in self.rows:
                    if int(r.get("word_id", -1)) == word_id:
                        r["quality_status"] = q_status
                        r["quality_score"] = q_score
                        r["face_detection_rate"] = result["face_rate"]
                        r["lip_detection_rate"] = result["lip_rate"]
                        r["recording_attempt"] = int(r.get("recording_attempt", 1)) + 1
                        r["emotion_detected"] = result.get("emotion", "Neutral")
                        break

                self.root.after(0, self._save_csv)
                self.qc_manager.generate_html_report(self.speaker_id, self.rows)
                self.root.after(0, lambda: messagebox.showinfo("Retry Complete", f"Successfully re-recorded '{word_name}'!\nNew Quality: {q_status} ({q_score * 100:.1f}%)"))
            finally:
                self.running = False

        threading.Thread(target=_retry_worker, daemon=False).start()

    # ========================================================
    # DIALOGS & NAVIGATION
    # ========================================================

    def show_navigation_menu(self):
        p = self.palette
        menu = tk.Toplevel(self.root)
        menu.title("LipRead Studio Menu")
        menu.geometry("340x360")
        menu.configure(bg=p["bg"])
        menu.transient(self.root)
        menu.resizable(False, False)

        tk.Label(menu, text="LIPREAD STUDIO", font=("Segoe UI", 16, "bold"), fg=p["accent"], bg=p["bg"]).pack(anchor="w", padx=20, pady=(18, 2))
        tk.Label(menu, text="ENTERPRISE TOOLS & NAVIGATION", font=("Segoe UI", 7, "bold"), fg=p["text_muted"], bg=p["bg"]).pack(anchor="w", padx=20, pady=(0, 14))

        def item(text, cmd):
            tk.Button(
                menu, text=text, command=lambda: (menu.destroy(), cmd()),
                anchor="w", bg=p["panel"], fg=p["text"], activebackground=p["panel_alt"],
                relief="flat", bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2", padx=14, pady=8
            ).pack(fill="x", padx=16, pady=3)

        item("📊  Session Analytics Dashboard", self.open_analytics_dashboard)
        item("📦  Export to ML (PyTorch / CTC)", self.open_ml_export_dialog)
        item("⚙  Studio Settings & Theme", self.open_settings_dialog)
        item("ℹ  About LipRead Studio", self.show_about_dialog)

        tk.Button(
            menu, text="CLOSE", command=menu.destroy,
            bg=p["panel"], fg=p["text_muted"], relief="flat", bd=0, font=("Segoe UI", 8, "bold"), padx=12, pady=6
        ).pack(anchor="e", padx=16, pady=12)

    def open_analytics_dashboard(self):
        if not self.rows:
            messagebox.showinfo("Analytics", "No session recording data available yet. Record at least one word first.")
            return
        SessionAnalyticsWindow(
            self.root, self.speaker_id, self.rows, self.current_theme,
            on_retry_word_callback=self.rerecord_single_word
        )

    def open_ml_export_dialog(self):
        if not self.rows:
            messagebox.showinfo("ML Export", "Record session data before exporting to ML formats.")
            return

        out_pt, vocab = self.ml_exporter.export_pytorch_manifest(self.rows, self.speaker_id)
        out_ctc = self.ml_exporter.export_ctc_csv(self.rows, self.speaker_id)

        messagebox.showinfo(
            "ML Export Complete",
            f"Successfully exported ML manifests to:\n\n"
            f"• PyTorch Manifest: {out_pt.name}\n"
            f"• Vocabulary JSON: {vocab.name}\n"
            f"• CTC CSV: {out_ctc.name}\n\n"
            f"Saved in: {EXPORTS_ROOT}"
        )

    def open_settings_dialog(self):
        p = self.palette
        win = tk.Toplevel(self.root)
        win.title("Studio Settings")
        win.geometry("560x520")
        win.configure(bg=p["bg"])
        win.transient(self.root)

        tk.Label(win, text="STUDIO SETTINGS", font=("Segoe UI", 18, "bold"), fg=p["text"], bg=p["bg"]).pack(anchor="w", padx=20, pady=(18, 2))
        tk.Label(win, text="APPEARANCE • HARDWARE ACCELERATION • BACKUP", font=("Segoe UI", 7, "bold"), fg=p["text_muted"], bg=p["bg"]).pack(anchor="w", padx=20, pady=(0, 14))

        panel = tk.Frame(win, bg=p["panel"], highlightthickness=1, highlightbackground=p["border"], padx=18, pady=14)
        panel.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # 1. Theme Selection
        tk.Label(panel, text="THEME PRESET", font=("Segoe UI", 8, "bold"), fg=p["accent"], bg=p["panel"]).pack(anchor="w", pady=(0, 4))
        theme_var = tk.StringVar(value=self.current_theme)
        theme_combo = ttk.Combobox(
            panel, textvariable=theme_var, values=list(THEME_PALETTES.keys()), state="readonly", width=25
        )
        theme_combo.pack(anchor="w", pady=(0, 12))

        # 2. GPU OpenCL Toggle
        gpu_var = tk.BooleanVar(value=self.gpu_accel_enabled)
        ttk.Checkbutton(panel, text="Enable OpenCV OpenCL Hardware Acceleration", variable=gpu_var).pack(anchor="w", pady=(0, 12))

        # 3. Cloud / Local Backup Directory
        tk.Label(panel, text="CLOUD / BACKUP DIRECTORY (OneDrive, Google Drive, NAS)", font=("Segoe UI", 8, "bold"), fg=p["accent_secondary"], bg=p["panel"]).pack(anchor="w", pady=(0, 4))
        backup_path_var = tk.StringVar(value=str(self.cloud_sync.target_dir) if self.cloud_sync.target_dir else "")
        b_frame = tk.Frame(panel, bg=p["panel"])
        b_frame.pack(fill="x", pady=(0, 16))
        ttk.Entry(b_frame, textvariable=backup_path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))

        def _browse_backup():
            d = filedialog.askdirectory(title="Select Cloud/Backup Folder")
            if d:
                backup_path_var.set(d)

        tk.Button(b_frame, text="Browse...", command=_browse_backup, bg=p["btn_normal"], fg=p["text"], relief="flat", bd=0, padx=8, pady=4).pack(side="right")

        def _apply():
            new_theme = theme_var.get()
            if new_theme != self.current_theme:
                self.current_theme = new_theme
                self.palette = THEME_PALETTES[new_theme]
                messagebox.showinfo("Theme Changed", "Theme settings saved. Restart or rebuild UI to apply fully.")

            self.gpu_accel_enabled = gpu_var.get()
            if cv2 is not None:
                try:
                    cv2.ocl.setUseOpenCL(self.gpu_accel_enabled)
                except Exception:
                    pass

            b_dir = backup_path_var.get().strip()
            ok, msg = self.cloud_sync.set_target_dir(b_dir if b_dir else None)
            win.destroy()
            messagebox.showinfo("Settings Saved", f"Settings applied!\n{msg}")

        tk.Button(
            panel, text="SAVE & APPLY", command=_apply,
            bg=p["accent"], fg="#070B14", font=("Segoe UI", 9, "bold"), relief="flat", bd=0, padx=16, pady=8
        ).pack(anchor="e")

    def show_about_dialog(self):
        messagebox.showinfo(
            "About LipRead Studio",
            "LipRead Studio v5.0\n"
            "Enterprise Multi-modal Visual Speech Recording System\n\n"
            "• Video + Audio + Lip Crop extraction\n"
            "• Multi-speaker sequence automation\n"
            "• Multilingual vocabulary support (English, Hindi, Assamese)\n"
            "• Real-time posture & lighting intelligence\n"
            "• Automated HTML/CSV audits and ML manifests\n\n"
            "Designed for AI Research and High-Fidelity Dataset Collection."
        )

    def handle_error(self, err_msg):
        self.running = False
        self.recording_word = False
        messagebox.showerror("Recording Error", str(err_msg))

    def close(self):
        """Safe application teardown."""
        if self.running:
            if not messagebox.askyesno("Exit Studio", "A recording session is active. Stop and exit safely?"):
                return
            self.stop_session()

        self.audio_viz.stop_passive_monitor()
        self.cloud_sync.stop()
        self.stop_camera()

        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass

        self.root.destroy()
