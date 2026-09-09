"""
Facial sentiment and emotion detection heuristic engine using MediaPipe FaceMesh.
Provides real-time expression categorization (Neutral, Happy/Smiling, Focused/Serious, Expressive)
to enrich metadata for multi-modal visual speech recognition.
"""

import math


class EmotionDetector:
    """Classifies facial expressions into heuristic emotion categories."""

    def __init__(self):
        # Key landmark indices in MediaPipe FaceMesh
        self.LIP_LEFT = 61
        self.LIP_RIGHT = 291
        self.LIP_TOP = 13
        self.LIP_BOTTOM = 14
        self.NOSE_TIP = 1
        self.LEFT_EYE_TOP = 159
        self.RIGHT_EYE_TOP = 386
        self.LEFT_BROW = 70
        self.RIGHT_BROW = 300

    def analyze_landmarks(self, landmarks, frame_w: int, frame_h: int):
        """
        Analyze facial landmarks and return classified emotion, confidence score, and metrics.
        Returns: (emotion_label: str, confidence: float, metrics: dict)
        """
        pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        if not pts or len(pts) < 468:
            return "Neutral", 0.50, {}

        try:

            def _coords(idx):
                return (pts[idx].x * frame_w, pts[idx].y * frame_h)

            lip_left = _coords(self.LIP_LEFT)
            lip_right = _coords(self.LIP_RIGHT)
            lip_top = _coords(self.LIP_TOP)
            lip_bottom = _coords(self.LIP_BOTTOM)
            nose = _coords(self.NOSE_TIP)

            # Distances
            mouth_width = math.hypot(lip_right[0] - lip_left[0], lip_right[1] - lip_left[1])
            mouth_height = math.hypot(lip_bottom[0] - lip_top[0], lip_bottom[1] - lip_top[1])

            if mouth_width < 1.0:
                return "Neutral", 0.50, {}

            # Mouth Aspect Ratio (MAR)
            mar = mouth_height / mouth_width

            # Smile index: corners higher than mouth center (in image coords, smaller Y is higher)
            lip_mid_y = (lip_top[1] + lip_bottom[1]) / 2.0
            corner_mid_y = (lip_left[1] + lip_right[1]) / 2.0
            smile_lift = (lip_mid_y - corner_mid_y) / mouth_width

            # Eyebrow elevation relative to eyes
            left_eye = _coords(self.LEFT_EYE_TOP)
            left_brow = _coords(self.LEFT_BROW)
            brow_dist = (left_eye[1] - left_brow[1]) / max(1.0, mouth_width)

            metrics = {
                "mar": round(mar, 3),
                "smile_lift": round(smile_lift, 3),
                "brow_dist": round(brow_dist, 3)
            }

            # Heuristic decision tree
            if smile_lift > 0.045 and mar > 0.15:
                label = "Happy / Smiling"
                conf = min(0.95, 0.65 + smile_lift * 3.0)
            elif mar > 0.50:
                label = "Expressive / Open"
                conf = min(0.95, 0.60 + mar * 0.5)
            elif brow_dist < 0.20 or mar < 0.08:
                label = "Focused / Serious"
                conf = 0.75
            else:
                label = "Neutral"
                conf = 0.85

            return label, round(conf, 2), metrics

        except Exception:
            return "Neutral", 0.50, {}
