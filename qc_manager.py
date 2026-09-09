"""
Quality Control Manager for LipRead Studio.
Provides real-time frame visibility & orientation checks, quality scoring,
automated HTML/CSV report generation, and single-word re-recording tracking.
"""

import math
import time
import csv
from pathlib import Path
import numpy as np

from config import (
    QC_PASS_THRESHOLD,
    QC_REVIEW_THRESHOLD,
    QC_FAILED_THRESHOLD,
    FACE_WEIGHT,
    LIP_WEIGHT,
    REPORTS_ROOT,
    METADATA_ROOT
)


class QualityControlManager:
    """Evaluates recording quality, flags real-time warnings, and compiles reports."""

    def __init__(self):
        pass

    def evaluate_realtime_warnings(self, frame: np.ndarray, landmarks, lip_box):
        """
        Check for lighting, head tilt, and lip occlusion in real time.
        Returns: list of active warning strings (e.g., ['LOW LIGHTING', 'FACE TILTED'])
        """
        warnings = []
        if frame is None:
            return ["NO VIDEO SIGNAL"]

        h, w = frame.shape[:2]

        # 1. Lighting check (average intensity in center region)
        ch1, ch2 = int(h * 0.25), int(h * 0.75)
        cw1, cw2 = int(w * 0.25), int(w * 0.75)
        center_crop = frame[ch1:ch2, cw1:cw2]
        if center_crop.size > 0:
            mean_brightness = float(np.mean(center_crop))
            if mean_brightness < 42.0:
                warnings.append("LOW LIGHTING")
            elif mean_brightness > 232.0:
                warnings.append("OVEREXPOSED")

        # 2. Orientation & tilt check via landmarks
        pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        if pts and len(pts) >= 468:
            # Left eye outer (33) and right eye outer (263)
            left_eye = (pts[33].x * w, pts[33].y * h)
            right_eye = (pts[263].x * w, pts[263].y * h)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))

            # Roll tilt
            if abs(angle) > 13.0:
                warnings.append("FACE TILTED")

            # Yaw check (nose tip vs eye midpoint)
            nose = pts[1].x * w
            eye_mid = (left_eye[0] + right_eye[0]) / 2.0
            eye_dist = max(1.0, math.hypot(dx, dy))
            yaw_offset = abs(nose - eye_mid) / eye_dist
            if yaw_offset > 0.30:
                warnings.append("TURN TOWARDS CAMERA")

        # 3. Lip bounds & occlusion check
        if lip_box is not None:
            x1, y1, x2, y2 = lip_box
            if (x2 - x1) < 28 or (y2 - y1) < 16:
                warnings.append("MOVE CLOSER")
            elif x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3:
                warnings.append("LIP OCCLUDED / EDGE")
        else:
            if landmarks:
                warnings.append("LIPS NOT DETECTED")
            else:
                warnings.append("NO FACE DETECTED")

        return warnings

    def calculate_word_quality(self, face_rate: float, lip_rate: float, audio_level: float = 0.0):
        """Calculate weighted score and categorize into GOOD, REVIEW, or FAILED."""
        score = (FACE_WEIGHT * float(face_rate)) + (LIP_WEIGHT * float(lip_rate))

        # Audio clarity factor
        if audio_level > 0.02:
            score = min(1.0, score + 0.02)

        if float(lip_rate) < QC_FAILED_THRESHOLD:
            status = "FAILED"
        elif score >= QC_PASS_THRESHOLD:
            status = "GOOD"
        else:
            status = "REVIEW"

        return status, round(score, 3)

    def generate_html_report(self, speaker_id: str, rows: list, output_path: Path = None):
        """Generate a sleek, responsive HTML QC report for researchers and participants."""
        if output_path is None:
            output_path = REPORTS_ROOT / f"{speaker_id}_qc_report.html"

        total_words = len(rows)
        good_count = sum(1 for r in rows if r.get("quality_status") == "GOOD")
        review_count = sum(1 for r in rows if r.get("quality_status") == "REVIEW")
        failed_count = sum(1 for r in rows if r.get("quality_status") == "FAILED")

        avg_lip = np.mean([float(r.get("lip_detection_rate", 0)) for r in rows]) * 100 if rows else 0
        avg_face = np.mean([float(r.get("face_detection_rate", 0)) for r in rows]) * 100 if rows else 0
        avg_score = np.mean([float(r.get("quality_score", 0)) for r in rows]) * 100 if rows else 0

        pass_rate = (good_count / max(1, total_words)) * 100

        # Build table rows
        table_html = []
        for r in rows:
            status = r.get("quality_status", "UNKNOWN")
            status_color = "#34D399" if status == "GOOD" else ("#FBBF24" if status == "REVIEW" else "#F87171")
            status_bg = "rgba(52, 211, 153, 0.15)" if status == "GOOD" else ("rgba(251, 191, 36, 0.15)" if status == "REVIEW" else "rgba(248, 113, 113, 0.15)")

            lip_pct = float(r.get("lip_detection_rate", 0)) * 100
            face_pct = float(r.get("face_detection_rate", 0)) * 100
            score_pct = float(r.get("quality_score", 0)) * 100
            emotion = r.get("emotion_detected", "Neutral")
            attempt = r.get("recording_attempt", 1)

            row_markup = f"""
            <tr>
                <td style="font-weight:600;">{r.get('word_id', '')}</td>
                <td style="font-size:1.05rem; font-weight:700; color:#F4F8FF;">{r.get('word', '')}</td>
                <td>{lip_pct:.1f}%</td>
                <td>{face_pct:.1f}%</td>
                <td>{score_pct:.1f}%</td>
                <td><span style="display:inline-block; padding:3px 8px; border-radius:4px; font-weight:700; color:{status_color}; background:{status_bg};">{status}</span></td>
                <td>{emotion}</td>
                <td>Attempt {attempt}</td>
                <td>{r.get('duration_sec', 0)}s</td>
            </tr>
            """
            table_html.append(row_markup)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QC Report — Speaker {speaker_id}</title>
    <style>
        :root {{
            --bg: #070B14;
            --panel: #0E1625;
            --border: #233450;
            --text: #F4F8FF;
            --muted: #71839A;
            --accent: #55F6D2;
            --success: #34D399;
            --warning: #FBBF24;
            --danger: #F87171;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            color: var(--accent);
            margin: 0 0 6px 0;
            font-size: 1.8rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            margin-bottom: 6px;
            font-weight: 700;
        }}
        .kpi-val {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #111A28;
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
        }}
        tr:hover {{
            background: rgba(85, 246, 210, 0.03);
        }}
        .footer {{
            margin-top: 30px;
            font-size: 0.8rem;
            color: var(--muted);
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LIPREAD STUDIO • QUALITY CONTROL AUDIT</h1>
            <div style="color:var(--muted); font-size:0.9rem;">
                Speaker ID: <strong style="color:var(--text);">{speaker_id}</strong> &nbsp;•&nbsp; Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Pass Rate</div>
                <div class="kpi-val" style="color:var(--success);">{pass_rate:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Good / Review / Failed</div>
                <div class="kpi-val"><span style="color:var(--success);">{good_count}</span> / <span style="color:var(--warning);">{review_count}</span> / <span style="color:var(--danger);">{failed_count}</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Avg Lip Detection</div>
                <div class="kpi-val">{avg_lip:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Avg Quality Score</div>
                <div class="kpi-val">{avg_score:.1f}%</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Word</th>
                    <th>Lip %</th>
                    <th>Face %</th>
                    <th>Quality</th>
                    <th>Status</th>
                    <th>Emotion</th>
                    <th>Attempt</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_html)}
            </tbody>
        </table>

        <div class="footer">
            LipRead Studio AI Dataset Collection Pipeline • Enterprise Research Suite
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
