"""
Real-time audio waveform buffer, RMS energy calculator, and visualizer.
Provides live oscilloscope and VU meter rendering for Tkinter canvas and HUD.
"""

import math
import time
import threading
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


class AudioVisualizer:
    """Manages audio sample buffers and renders real-time waveforms."""

    def __init__(self, buffer_size=1024, sample_rate=44100):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.lock = threading.Lock()
        self.waveform_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.current_rms = 0.0
        self.current_peak = 0.0

        self.monitor_running = False
        self.monitor_stream = None
        self.monitor_thread = None
        self.device_index = None

    def update_samples(self, new_samples: np.ndarray):
        """Update the internal circular waveform buffer with fresh audio samples."""
        if new_samples is None or len(new_samples) == 0:
            return

        flat = new_samples.flatten().astype(np.float32)
        n = len(flat)

        with self.lock:
            if n >= self.buffer_size:
                self.waveform_buffer[:] = flat[-self.buffer_size:]
            else:
                self.waveform_buffer[:-n] = self.waveform_buffer[n:]
                self.waveform_buffer[-n:] = flat

            peak = float(np.max(np.abs(self.waveform_buffer)))
            rms = float(np.sqrt(np.mean(self.waveform_buffer ** 2)))

            # Exponential decay smoothing for meters
            self.current_peak = max(peak, self.current_peak * 0.85)
            self.current_rms = self.current_rms * 0.7 + rms * 0.3

    def get_metrics(self):
        """Return (rms, peak, db) metrics."""
        with self.lock:
            rms = max(0.0, min(1.0, self.current_rms * 4.0))  # scaled for sensitivity
            peak = max(0.0, min(1.0, self.current_peak * 2.0))
            db = 20.0 * math.log10(max(1e-5, self.current_rms))
            return rms, peak, db

    def get_waveform_copy(self, num_points=128):
        """Return downsampled waveform points for smooth canvas rendering."""
        with self.lock:
            buf = self.waveform_buffer.copy()

        if len(buf) == 0:
            return np.zeros(num_points)

        # Downsample cleanly to num_points
        step = max(1, len(buf) // num_points)
        downsampled = buf[::step][:num_points]
        if len(downsampled) < num_points:
            downsampled = np.pad(downsampled, (0, num_points - len(downsampled)))
        return downsampled

    def start_passive_monitor(self, device_index=None):
        """Start non-blocking passive sounddevice stream to show live audio during idle/preview."""
        if sd is None:
            return

        self.stop_passive_monitor()
        self.device_index = device_index
        self.monitor_running = True

        def _audio_callback(indata, frames, time_info, status):
            if not self.monitor_running:
                return
            self.update_samples(indata)

        def _monitor_worker():
            try:
                with sd.InputStream(
                    device=self.device_index,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=512,
                    callback=_audio_callback
                ):
                    while self.monitor_running:
                        time.sleep(0.05)
            except Exception:
                pass
            finally:
                self.monitor_running = False

        self.monitor_thread = threading.Thread(
            target=_monitor_worker,
            daemon=True,
            name="AudioPassiveMonitor"
        )
        self.monitor_thread.start()

    def stop_passive_monitor(self):
        """Stop background audio stream."""
        self.monitor_running = False
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=0.3)
        self.monitor_thread = None

    def render_to_canvas(self, canvas, x, y, width, height, theme_palette, tag="waveform"):
        """Draw real-time oscilloscope line and segmented VU meter on a Tkinter canvas."""
        canvas.delete(tag)

        # Background box
        bg = theme_palette.get("panel_alt", "#111A28")
        border = theme_palette.get("border", "#26354A")
        accent = theme_palette.get("accent", "#55F6D2")
        danger = theme_palette.get("danger", "#FF647C")

        canvas.create_rectangle(
            x, y, x + width, y + height,
            fill=bg, outline=border, tags=tag
        )

        mid_y = y + height / 2.0
        wave = self.get_waveform_copy(num_points=max(32, int(width / 3)))

        # Oscilloscope line
        points = []
        dx = width / max(1, len(wave) - 1)
        scale_y = (height / 2.0) * 0.85

        for i, val in enumerate(wave):
            px = x + i * dx
            py = mid_y - float(val) * scale_y
            py = max(y + 2, min(y + height - 2, py))
            points.extend([px, py])

        if len(points) >= 4:
            canvas.create_line(
                points,
                fill=accent,
                width=2,
                smooth=True,
                tags=tag
            )

        # Mini VU meter bar at the bottom
        rms, peak, _ = self.get_metrics()
        meter_y = y + height - 5
        meter_w = width - 8
        fill_w = max(0.0, min(meter_w, meter_w * rms))

        bar_color = danger if rms > 0.85 else accent
        canvas.create_rectangle(
            x + 4, meter_y - 2, x + 4 + fill_w, meter_y + 2,
            fill=bar_color, outline="", tags=tag
        )
