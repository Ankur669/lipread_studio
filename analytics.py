"""
Session Analytics Dashboard for LipRead Studio.
Provides rich interactive charts (accuracy distribution, lip/face detection trends,
emotion breakdown), KPI summary cards, and single-word retry controls.
"""

import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from config import THEME_PALETTES


class SessionAnalyticsWindow:
    """Renders the post-session visual analytics dashboard."""

    def __init__(self, parent, speaker_id: str, rows: list, theme_key: str = "cyberpunk", on_retry_word_callback=None):
        self.parent = parent
        self.speaker_id = speaker_id
        self.rows = rows or []
        self.palette = THEME_PALETTES.get(theme_key, THEME_PALETTES["cyberpunk"])
        self.on_retry_word_callback = on_retry_word_callback

        self.win = tk.Toplevel(parent)
        self.win.title(f"Session Analytics — Speaker {speaker_id}")
        self.win.geometry("1180x820")
        self.win.minsize(980, 680)
        self.win.configure(bg=self.palette["bg"])
        self.win.transient(parent)

        self._build_ui()

    def _build_ui(self):
        p = self.palette

        # Header
        hdr = tk.Frame(self.win, bg=p["bg"])
        hdr.pack(fill="x", padx=24, pady=(20, 10))

        tk.Label(
            hdr, text="SESSION ANALYTICS & DIAGNOSTICS",
            font=("Segoe UI", 18, "bold"), fg=p["accent"], bg=p["bg"]
        ).pack(anchor="w")

        tk.Label(
            hdr, text=f"Speaker: {self.speaker_id}  •  Total Words: {len(self.rows)}  •  Quality & Signal Integrity Audit",
            font=("Segoe UI", 9), fg=p["text_muted"], bg=p["bg"]
        ).pack(anchor="w", pady=(2, 0))

        # KPI Summary Cards
        kpi_frame = tk.Frame(self.win, bg=p["bg"])
        kpi_frame.pack(fill="x", padx=24, pady=(5, 15))

        total = len(self.rows)
        good = sum(1 for r in self.rows if r.get("quality_status") == "GOOD")
        review = sum(1 for r in self.rows if r.get("quality_status") == "REVIEW")
        failed = sum(1 for r in self.rows if r.get("quality_status") == "FAILED")
        pass_rate = (good / max(1, total)) * 100.0

        avg_lip = sum(float(r.get("lip_detection_rate", 0)) for r in self.rows) / max(1, total) * 100.0
        avg_face = sum(float(r.get("face_detection_rate", 0)) for r in self.rows) / max(1, total) * 100.0

        self._add_kpi_card(kpi_frame, "PASS RATE", f"{pass_rate:.1f}%", p["success"], 0)
        self._add_kpi_card(kpi_frame, "GOOD / REVIEW / FAIL", f"{good} / {review} / {failed}", p["text"], 1)
        self._add_kpi_card(kpi_frame, "AVG LIP TRACKING", f"{avg_lip:.1f}%", p["accent"], 2)
        self._add_kpi_card(kpi_frame, "AVG FACE TRACKING", f"{avg_face:.1f}%", p["accent_secondary"], 3)

        # Notebook Tabs: Charts / Data Table
        nb_frame = tk.Frame(self.win, bg=p["bg"])
        nb_frame.pack(fill="both", expand=True, padx=24, pady=(0, 15))

        notebook = ttk.Notebook(nb_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Charts
        charts_tab = tk.Frame(notebook, bg=p["panel"])
        notebook.add(charts_tab, text="  📊 Visual Charts  ")
        self._render_charts(charts_tab)

        # Tab 2: Table
        table_tab = tk.Frame(notebook, bg=p["panel"])
        notebook.add(table_tab, text="  📋 Word Breakdown & Retry  ")
        self._render_table(table_tab)

        # Footer Action Bar
        footer = tk.Frame(self.win, bg=p["bg"])
        footer.pack(fill="x", padx=24, pady=(0, 18))

        tk.Button(
            footer, text="Open HTML QC Report", command=self._open_html_report,
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            bg=p["btn_normal"], fg=p["text"], activebackground=p["btn_active"],
            cursor="hand2", padx=14, pady=8
        ).pack(side="left")

        tk.Button(
            footer, text="Close", command=self.win.destroy,
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            bg=p["border"], fg=p["text"], cursor="hand2", padx=18, pady=8
        ).pack(side="right")

    def _add_kpi_card(self, parent, label, value, value_color, col):
        p = self.palette
        card = tk.Frame(parent, bg=p["panel"], highlightthickness=1, highlightbackground=p["border"], padx=16, pady=12)
        card.grid(row=0, column=col, sticky="nsew", padx=6 if col > 0 else 0)
        parent.columnconfigure(col, weight=1)

        tk.Label(card, text=label, font=("Segoe UI", 7, "bold"), fg=p["text_muted"], bg=p["panel"]).pack(anchor="w")
        tk.Label(card, text=value, font=("Segoe UI", 16, "bold"), fg=value_color, bg=p["panel"]).pack(anchor="w", pady=(4, 0))

    def _render_charts(self, parent):
        p = self.palette
        if not MATPLOTLIB_AVAILABLE or not self.rows:
            lbl = tk.Label(
                parent, text="Chart visualization requires matplotlib and recorded rows.",
                font=("Segoe UI", 11), fg=p["text_muted"], bg=p["panel"]
            )
            lbl.pack(expand=True)
            return

        fig = Figure(figsize=(9, 4.8), dpi=100, facecolor=p["panel"])

        # Subplot 1: Quality scores bar chart
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_facecolor(p["panel_alt"])

        indices = list(range(1, len(self.rows) + 1))
        scores = [float(r.get("quality_score", 0)) * 100 for r in self.rows]
        statuses = [r.get("quality_status", "GOOD") for r in self.rows]

        colors = []
        for s in statuses:
            if s == "GOOD":
                colors.append(p["success"])
            elif s == "REVIEW":
                colors.append(p["warning"])
            else:
                colors.append(p["danger"])

        ax1.bar(indices, scores, color=colors, width=0.7, alpha=0.9)
        ax1.axhline(85, color=p["success"], linestyle="--", linewidth=1, alpha=0.7, label="Good (85%)")
        ax1.axhline(70, color=p["danger"], linestyle="--", linewidth=1, alpha=0.7, label="Fail (<70%)")

        ax1.set_title("Quality Score Distribution Across Word Sequence", color=p["text"], fontsize=9, fontweight="bold")
        ax1.set_ylabel("Quality %", color=p["text_muted"], fontsize=8)
        ax1.tick_params(colors=p["text_muted"], labelsize=7)
        ax1.set_ylim(0, 105)
        for spine in ax1.spines.values():
            spine.set_color(p["border"])

        # Subplot 2: Lip vs Face detection rates
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_facecolor(p["panel_alt"])

        lip_rates = [float(r.get("lip_detection_rate", 0)) * 100 for r in self.rows]
        face_rates = [float(r.get("face_detection_rate", 0)) * 100 for r in self.rows]

        ax2.plot(indices, lip_rates, color=p["accent"], linewidth=1.8, label="Lip %")
        ax2.plot(indices, face_rates, color=p["accent_secondary"], linewidth=1.5, linestyle=":", label="Face %")

        ax2.set_title("Lip vs Face Detection Rates", color=p["text"], fontsize=9, fontweight="bold")
        ax2.set_xlabel("Word ID", color=p["text_muted"], fontsize=8)
        ax2.set_ylabel("Detection %", color=p["text_muted"], fontsize=8)
        ax2.tick_params(colors=p["text_muted"], labelsize=7)
        ax2.set_ylim(0, 105)
        ax2.legend(loc="lower right", facecolor=p["panel"], edgecolor=p["border"], fontsize=7, labelcolor=p["text"])
        for spine in ax2.spines.values():
            spine.set_color(p["border"])

        fig.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _render_table(self, parent):
        p = self.palette

        frame = tk.Frame(parent, bg=p["panel"], padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        cols = ("id", "word", "lip", "face", "score", "status", "emotion", "attempt")
        tree = ttk.Treeview(frame, columns=cols, show="headings")

        tree.heading("id", text="#")
        tree.heading("word", text="Word")
        tree.heading("lip", text="Lip %")
        tree.heading("face", text="Face %")
        tree.heading("score", text="Score")
        tree.heading("status", text="Status")
        tree.heading("emotion", text="Emotion")
        tree.heading("attempt", text="Attempt")

        tree.column("id", width=40, anchor="center")
        tree.column("word", width=180)
        tree.column("lip", width=80, anchor="center")
        tree.column("face", width=80, anchor="center")
        tree.column("score", width=80, anchor="center")
        tree.column("status", width=100, anchor="center")
        tree.column("emotion", width=120, anchor="center")
        tree.column("attempt", width=70, anchor="center")

        for r in self.rows:
            lip = float(r.get("lip_detection_rate", 0)) * 100
            face = float(r.get("face_detection_rate", 0)) * 100
            sc = float(r.get("quality_score", 0)) * 100
            tree.insert("", "end", values=(
                r.get("word_id"),
                r.get("word"),
                f"{lip:.1f}%",
                f"{face:.1f}%",
                f"{sc:.1f}%",
                r.get("quality_status", "GOOD"),
                r.get("emotion_detected", "Neutral"),
                r.get("recording_attempt", 1)
            ))

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Action bar inside tab
        bar = tk.Frame(parent, bg=p["panel"], pady=8)
        bar.pack(fill="x", padx=10)

        def _retry_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Select Word", "Please select a word row to re-record.")
                return
            vals = tree.item(sel[0], "values")
            word_id = int(vals[0])
            word_name = vals[1]

            if self.on_retry_word_callback:
                self.win.destroy()
                self.on_retry_word_callback(word_id, word_name)
            else:
                messagebox.showinfo("Retry Word", f"Retry requested for Word #{word_id}: {word_name}")

        tk.Button(
            bar, text="⚡ Re-record Selected Word", command=_retry_selected,
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            bg=p["warning_bg"], fg=p["warning"], cursor="hand2", padx=12, pady=6
        ).pack(side="left")

    def _open_html_report(self):
        report_path = Path(__file__).resolve().parent / "dataset" / "reports" / f"{self.speaker_id}_qc_report.html"
        if report_path.exists():
            webbrowser.open(report_path.as_uri())
        else:
            messagebox.showwarning("Report Not Found", f"HTML report does not exist yet at:\n{report_path}")
