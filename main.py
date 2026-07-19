"""
Reti-AI — Explainable Eye Disease Diagnosis (PySide6 desktop dashboard).

Layout follows the UX redesign: a sticky patient banner, a 3-zone body
(Input · Analysis · Verdict) with the diagnosis verdict as the most prominent
element, a structured 3-tier narrative with a collapsible full explanation, and
an ordinal (sequential + hatched) grade-distribution chart.

Run:  python main.py
"""

import os
import sys
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QFont, QImage, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QScrollArea, QTextEdit,
    QMessageBox, QDoubleSpinBox, QCheckBox, QLineEdit, QSizePolicy,
)
from PIL import Image

import inference
import xai
import segmentation
import llm_gemini
import report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_CHECKPOINT = os.path.join(ROOT, "best_strict_cbm.pt")
DEFAULT_SEG_CKPT = os.path.join(ROOT, "best_unet_multilesion.pt")
DEFAULT_KB = os.path.join(ROOT, "fgadr_concept_kb.yaml")

DIAGNOSIS_NAMES = {
    0: "No Apparent Diabetic Retinopathy",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}
SEVERITY_LABELS = {0: "NONE", 1: "MILD", 2: "MODERATE", 3: "SEVERE", 4: "PROLIFERATIVE"}

# Vivid grade colours for badges / verdict text / predicted-bar outline.
SEV_COLORS = {0: "#22c55e", 1: "#84cc16", 2: "#f59e0b", 3: "#f97316", 4: "#ef4444"}
# Sequential (ordinal) ramp for the grade-distribution chart bars.
GRADE_RAMP = ["#FEF9C3", "#FDE68A", "#FDBA74", "#F97316", "#DC2626"]
# Descriptive x-axis labels for the grade-distribution chart (2 lines each).
GRADE_CHART_LABELS = [
    "No\nRetinopathy",
    "Mild\nNPDR",
    "Moderate\nNPDR",
    "Severe\nNPDR",
    "Proliferative\nDR",
]

# ---- dark palette ----
BG = "#0d1420"
PANEL = "#131c2b"
PANEL2 = "#0f1826"
BORDER = "#233247"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
FAINT = "#64748b"
ACCENT = "#3b82f6"

STYLESHEET = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-size:13px; }}
QLabel {{ background:transparent; }}
#card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; }}
#verdictCard {{ background:{PANEL}; border:1px solid {BORDER};
               border-top:3px solid {ACCENT}; border-radius:12px; }}
#cardTitle {{ font-size:11px; font-weight:bold; color:{MUTED};
             letter-spacing:1px; text-transform:uppercase; }}
#banner {{ background:{PANEL2}; border-bottom:1px solid {BORDER}; }}
#bannerName {{ font-size:20px; font-weight:bold; color:{TEXT}; }}
#bannerSub {{ font-size:12px; color:{MUTED}; }}
#appName {{ font-size:12px; color:{MUTED}; letter-spacing:1px; }}
#sidebar {{ background:{PANEL2}; border-right:1px solid {BORDER}; }}
#thumb {{ background:#0a0f18; border:1px solid {BORDER}; border-radius:8px; }}
#caption {{ color:{MUTED}; font-size:11px; }}
#tierLabel {{ font-size:10px; font-weight:bold; color:{FAINT};
             letter-spacing:1px; text-transform:uppercase; }}
#metricKey {{ color:{MUTED}; font-size:12px; }}
#metricVal {{ color:{TEXT}; font-size:13px; font-weight:bold; }}
#verdictDiag {{ font-size:20px; font-weight:bold; }}
QLineEdit {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;
            padding:3px 6px; color:{TEXT}; }}
QPushButton {{ background:{ACCENT}; color:white; border:none; border-radius:8px;
              padding:9px 16px; font-weight:bold; }}
QPushButton:hover {{ background:#2563eb; }}
QPushButton:disabled {{ background:#334155; color:#94a3b8; }}
QPushButton#ghost {{ background:{PANEL}; border:1px solid {BORDER}; color:{TEXT};
                    font-weight:normal; padding:6px 10px; }}
QPushButton#ghost:hover {{ background:#1c2942; }}
QDoubleSpinBox {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;
                 padding:2px 4px; color:{TEXT}; }}
QTextEdit {{ background:{PANEL2}; border:1px solid {BORDER}; border-radius:8px; }}
QScrollArea {{ border:none; }}
"""


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgba(h, a):
    r, g, b = hex_to_rgb(h)
    return f"rgba({r},{g},{b},{a})"


def pil_to_qpixmap(pil_img):
    img = pil_img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def card(title, object_name="card"):
    frame = QFrame()
    frame.setObjectName(object_name)
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(16, 14, 16, 16)
    outer.setSpacing(10)
    lbl = QLabel(title)
    lbl.setObjectName("cardTitle")
    outer.addWidget(lbl)
    return frame, outer


def badge(text, base_hex):
    """A tinted severity/status badge (tint bg + coloured border/text)."""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background:{rgba(base_hex, 0.16)};color:{base_hex};"
        f"border:1px solid {rgba(base_hex, 0.55)};border-radius:6px;"
        f"padding:3px 10px;font-size:12px;font-weight:bold;letter-spacing:0.5px;"
    )
    return lbl


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------
class ModelLoader(QObject):
    finished = Signal(object, object, object, int)
    failed = Signal(str)

    def __init__(self, cbm_path, seg_path):
        super().__init__()
        self.cbm_path = cbm_path
        self.seg_path = seg_path

    def run(self):
        try:
            cbm = inference.load_model(self.cbm_path)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        seg_model, lesions, size = None, [], 512
        try:
            if os.path.exists(self.seg_path):
                seg_model, lesions, size = segmentation.load_segmenter(self.seg_path)
        except Exception:
            seg_model, lesions, size = None, [], 512
        self.finished.emit(cbm, seg_model, lesions, size)


class InferenceWorker(QObject):
    finished = Signal(dict, dict, object)
    failed = Signal(str)

    def __init__(self, cbm, seg_model, lesions, seg_size, image_path, kb, threshold):
        super().__init__()
        self.cbm = cbm
        self.seg_model = seg_model
        self.lesions = lesions
        self.seg_size = seg_size
        self.image_path = image_path
        self.kb = kb
        self.threshold = threshold

    def run(self):
        try:
            result = inference.predict(self.cbm, self.image_path, threshold=self.threshold)
            explanation = xai.narrate(result, self.kb)
            # LLM (Gemini) narrative — deterministic RAG + faithful prompt,
            # falls back to the template narrative on any failure.
            llm_text, source = llm_gemini.generate_narrative(result, self.kb)
            explanation["llm_narrative"] = llm_text
            explanation["narrative_source"] = source
            prob_maps = None
            if self.seg_model is not None:
                prob_maps = segmentation.predict_masks(
                    self.seg_model, self.image_path, self.lesions, self.seg_size
                )
            self.finished.emit(result, explanation, prob_maps)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------
class GradeChart(QWidget):
    """Ordinal grade-probability chart: sequential ramp, labels, %, hatching."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(188)
        self.dist = [0, 0, 0, 0, 0]
        self.predicted = -1

    def set_data(self, dist, predicted):
        self.dist = list(dist)
        self.predicted = predicted
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad_b, pad_t, pad_x = 46, 18, 8
        plot_h = h - pad_b - pad_t
        n = 5
        slot = (w - 2 * pad_x) / n
        bw = slot * 0.62
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(pad_x, h - pad_b, w - pad_x, h - pad_b)
        maxv = max(max(self.dist), 1e-6)
        f = p.font(); f.setPointSize(8)
        for i in range(n):
            val = self.dist[i]
            bh = max(plot_h * (val / maxv), 2)
            x = pad_x + slot * i + (slot - bw) / 2
            y = pad_t + (plot_h - bh)
            col = QColor(GRADE_RAMP[i])
            p.fillRect(int(x), int(y), int(bw), int(bh), col)
            # secondary encoding: hatching for severe grades (colour-blind safe)
            if i >= 3:
                p.setPen(QPen(QColor(0, 0, 0, 90), 1))
                step = 6
                for hx in range(int(x), int(x + bw), step):
                    p.drawLine(hx, int(y + bh), int(min(x + bw, hx + bh)),
                               int(y + bh - min(bh, x + bw - hx)))
            # predicted highlight
            if i == self.predicted:
                p.setPen(QPen(QColor(TEXT), 2))
                p.setBrush(QBrush(Qt.NoBrush))
                p.drawRect(int(x) - 1, int(y) - 1, int(bw) + 2, int(bh) + 2)
            # value label
            p.setFont(f)
            p.setPen(QColor(TEXT if i == self.predicted else MUTED))
            p.drawText(int(x - 8), int(y - 16), int(bw + 16), 14,
                       Qt.AlignCenter, f"{val*100:.0f}%")
            # x-axis descriptive label (full slot width, 2 lines)
            fl = p.font(); fl.setPointSize(7); p.setFont(fl)
            p.setPen(QColor(TEXT if i == self.predicted else MUTED))
            slot_x = pad_x + slot * i
            p.drawText(int(slot_x), h - pad_b + 4, int(slot), pad_b - 6,
                       Qt.AlignHCenter | Qt.AlignTop, GRADE_CHART_LABELS[i])
        p.end()


class LesionSizeBar(QWidget):
    """Proportional horizontal bar of per-lesion total area."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)
        self.parts = []

    def set_data(self, per_lesion):
        total = sum(d["area_px"] for d in per_lesion.values()) or 1
        self.parts = [(k, d["area_px"] / total) for k, d in per_lesion.items()]
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        x = 0
        if not self.parts:
            p.fillRect(0, 0, w, h, QColor(PANEL2))
        for lesion, frac in self.parts:
            seg_w = w * frac
            r, g, b = segmentation.LESION_COLORS.get(lesion, (150, 150, 150))
            p.fillRect(int(x), 0, int(seg_w) + 1, h, QColor(r, g, b))
            x += seg_w
        p.end()


class ConceptChips(QWidget):
    """Color-coded concept chips (KEY DRIVERS). Grid, 3 columns."""

    def __init__(self, columns=3):
        super().__init__()
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)

    def _clear(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            wdt = item.widget()
            if wdt:
                wdt.deleteLater()

    def set_data(self, concepts, active, drivers_only=False):
        self._clear()
        items = [c for c in inference.CONCEPT_COLS
                 if (c in active) or (not drivers_only)]
        if drivers_only and not items:
            lbl = QLabel("No lesion concept above threshold")
            lbl.setObjectName("caption")
            self.grid.addWidget(lbl, 0, 0, 1, self.columns)
            return
        for idx, cid in enumerate(items):
            r, g, b = segmentation.LESION_COLORS.get(cid, (150, 150, 150))
            is_active = cid in active
            chip = QLabel(f"{cid}  {concepts[cid]*100:.0f}%")
            chip.setAlignment(Qt.AlignCenter)
            if is_active:
                chip.setStyleSheet(
                    f"background:rgba({r},{g},{b},0.20);color:{TEXT};"
                    f"border:1px solid rgba({r},{g},{b},0.85);border-radius:6px;"
                    f"padding:4px 6px;font-weight:bold;font-size:12px;"
                )
            else:
                chip.setStyleSheet(
                    f"background:transparent;color:{FAINT};"
                    f"border:1px solid {BORDER};border-radius:6px;"
                    f"padding:4px 6px;font-size:12px;"
                )
            self.grid.addWidget(chip, idx // self.columns, idx % self.columns)


class Thumb(QWidget):
    """Image thumbnail with a caption underneath."""

    def __init__(self, caption, min_h=140):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.img = QLabel()
        self.img.setObjectName("thumb")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumSize(96, min_h)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cap = QLabel(caption)
        self.cap.setObjectName("caption")
        self.cap.setAlignment(Qt.AlignCenter)
        self.cap.setWordWrap(True)
        lay.addWidget(self.img, 1)
        lay.addWidget(self.cap)
        self._pil = None

    def set_pil(self, pil):
        self._pil = pil
        self._rescale()

    def _rescale(self):
        if self._pil is None:
            return
        pix = pil_to_qpixmap(self._pil)
        self.img.setPixmap(
            pix.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reti-AI — Explainable Eye Disease Diagnosis")
        self.resize(1500, 900)

        self.cbm = None
        self.seg_model = None
        self.seg_lesions = []
        self.seg_size = 512
        self.kb = None
        self.image_path = None
        self.prob_maps = None
        self._current_pil = None
        self._last_result = None
        self._last_explanation = None
        self._last_quant = None
        self._narrative_expanded = False

        self._build_ui()
        self._load_kb()
        self._start_model_load()

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_banner())    # sticky, full-width
        root.addWidget(self._build_toolbar())   # sticky controls

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.vbody = QVBoxLayout(content)
        self.vbody.setContentsMargins(18, 18, 18, 18)
        self.vbody.setSpacing(16)
        scroll.setWidget(content)
        body.addWidget(scroll, 1)
        root.addLayout(body, 1)

        self._build_zones()

    def _build_banner(self):
        banner = QFrame()
        banner.setObjectName("banner")
        banner.setFixedHeight(70)
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(20, 10, 20, 10)

        avatar = QLabel("👤")
        avatar.setStyleSheet("font-size:30px;")
        lay.addWidget(avatar)
        lay.addSpacing(8)

        idbox = QVBoxLayout()
        idbox.setSpacing(1)
        self.pt_name = QLineEdit("Rudi Santoso")
        self.pt_name.setObjectName("bannerName")
        self.pt_name.setStyleSheet(
            f"background:transparent;border:none;font-size:20px;font-weight:bold;color:{TEXT};"
        )
        self.pt_sub = QLabel("RD-240901  ·  58 y/o Male  ·  DOB 15/03/1966")
        self.pt_sub.setObjectName("bannerSub")
        idbox.addWidget(self.pt_name)
        idbox.addWidget(self.pt_sub)
        lay.addLayout(idbox)
        lay.addStretch(1)

        appn = QLabel("RETI-AI · Explainable Eye Disease Diagnosis")
        appn.setObjectName("appName")
        lay.addWidget(appn)
        lay.addSpacing(14)
        self.session_badge = badge("● Active session", "#22c55e")
        lay.addWidget(self.session_badge)
        return banner

    def _build_toolbar(self):
        tb = QFrame()
        tb.setObjectName("banner")
        tb.setFixedHeight(48)
        lay = QHBoxLayout(tb)
        lay.setContentsMargins(20, 6, 20, 6)
        self.btn_open = QPushButton("📂  Load fundus image")
        self.btn_open.clicked.connect(self.on_open)
        self.btn_analyze = QPushButton("🔬  Analyze")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.on_analyze)
        thr_lbl = QLabel("Threshold:")
        thr_lbl.setObjectName("metricKey")
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.05, 0.95)
        self.thr_spin.setSingleStep(0.05)
        self.thr_spin.setValue(0.50)
        self.thr_spin.valueChanged.connect(self._on_threshold_changed)
        self.status = QLabel("Loading model…")
        self.status.setObjectName("metricKey")
        lay.addWidget(self.btn_open)
        lay.addWidget(self.btn_analyze)
        lay.addSpacing(14)
        lay.addWidget(thr_lbl)
        lay.addWidget(self.thr_spin)
        lay.addStretch(1)
        lay.addWidget(self.status)
        return tb

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(52)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(8, 14, 8, 14)
        lay.setSpacing(14)
        for icon in ["🏠", "📋", "👤", "🫀", "⚙"]:
            b = QLabel(icon)
            b.setAlignment(Qt.AlignCenter)
            b.setStyleSheet("font-size:18px;")
            lay.addWidget(b)
        lay.addStretch(1)
        return side

    def _build_zones(self):
        zones = QHBoxLayout()
        zones.setSpacing(14)

        # ---- INPUT ZONE ----
        input_col = QVBoxLayout()
        input_col.setSpacing(16)
        fcard, flay = card("Input · Fundus Image Analysis")
        self.thumb_orig = Thumb("(1) Original Scan", min_h=210)
        flay.addWidget(self.thumb_orig, 3)
        sub = QHBoxLayout()
        sub.setSpacing(10)
        self.thumb_det = Thumb("(2) Lesion Detection", min_h=120)
        self.thumb_map = Thumb("(3) Segmentation Mask", min_h=120)
        sub.addWidget(self.thumb_det)
        sub.addWidget(self.thumb_map)
        flay.addLayout(sub, 2)

        toggles = QHBoxLayout()
        toggles.setSpacing(8)
        self.chk_overlay = QCheckBox("Overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self._refresh_derived)
        toggles.addWidget(self.chk_overlay)
        self.lesion_checks = {}
        for cid in ["MA", "HE", "EX", "SE", "IRMA", "NV"]:
            r, g, b = segmentation.LESION_COLORS.get(cid, (200, 200, 200))
            chk = QCheckBox(cid)
            chk.setChecked(True)
            chk.stateChanged.connect(self._refresh_derived)
            chk.setStyleSheet(
                "QCheckBox{color:%s;font-weight:bold;}"
                "QCheckBox::indicator{width:13px;height:13px;border-radius:3px;"
                "background:rgb(%d,%d,%d);border:1px solid %s;}" % (TEXT, r, g, b, BORDER)
            )
            self.lesion_checks[cid] = chk
            toggles.addWidget(chk)
        toggles.addStretch(1)
        flay.addLayout(toggles)
        input_col.addWidget(fcard)
        zones.addLayout(input_col, 4)

        # ---- ANALYSIS ZONE ----
        analysis_col = QVBoxLayout()
        analysis_col.setSpacing(16)

        ccard, clay = card("Analysis · CBM Concepts")
        drv = QLabel("KEY DRIVERS")
        drv.setObjectName("tierLabel")
        clay.addWidget(drv)
        self.concept_chips = ConceptChips(columns=3)
        clay.addWidget(self.concept_chips)
        analysis_col.addWidget(ccard)

        gcard, glay = card("Analysis · Grade Distribution")
        self.chart = GradeChart()
        glay.addWidget(self.chart, 1)
        analysis_col.addWidget(gcard)

        qcard, qlay = card("Analysis · Quantitative")
        self.q_summary = QLabel("Lesions: —   ·   Retina area: —")
        self.q_summary.setObjectName("metricVal")
        qlay.addWidget(self.q_summary)
        self.chk_details = QCheckBox("Show quantitative details")
        self.chk_details.stateChanged.connect(self._toggle_details)
        qlay.addWidget(self.chk_details)

        self.detail_widget = QWidget()
        dv = QVBoxLayout(self.detail_widget)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(6)
        self.q_vals = {}
        metrics = [
            ("Lesions", "n_lesions"), ("Total area", "total_area"),
            ("Avg size", "avg_size"), ("Retina coverage", "area_pct"),
            ("Location", "location"), ("Confidence", "confidence"),
        ]
        for label, key in metrics:
            row = QHBoxLayout()
            k = QLabel(label)
            k.setObjectName("metricKey")
            v = QLabel("—")
            v.setObjectName("metricVal")
            self.q_vals[key] = v
            row.addWidget(k)
            row.addStretch(1)
            row.addWidget(v)
            dv.addLayout(row)
        size_lbl = QLabel("Lesion Size Distribution")
        size_lbl.setObjectName("caption")
        self.size_bar = LesionSizeBar()
        dv.addWidget(size_lbl)
        dv.addWidget(self.size_bar)
        self.detail_widget.setVisible(False)
        qlay.addWidget(self.detail_widget)
        analysis_col.addWidget(qcard)
        analysis_col.addStretch(1)
        zones.addLayout(analysis_col, 4)

        # ---- VERDICT ZONE (most prominent) ----
        vcard, vlay = card("Verdict · Final Classification", object_name="verdictCard")
        vcard.setMinimumWidth(300)
        vlay.setSpacing(12)

        row_sev = QHBoxLayout()
        self.sev_badge = badge("—", MUTED)
        row_sev.addWidget(self.sev_badge)
        row_sev.addStretch(1)
        vlay.addLayout(row_sev)

        dlbl = QLabel("DIAGNOSIS")
        dlbl.setObjectName("tierLabel")
        vlay.addWidget(dlbl)
        self.diag_name = QLabel("Awaiting analysis")
        self.diag_name.setObjectName("verdictDiag")
        self.diag_name.setWordWrap(True)
        vlay.addWidget(self.diag_name)

        conf_row = QHBoxLayout()
        ck = QLabel("Confidence")
        ck.setObjectName("metricKey")
        conf_row.addWidget(ck)
        conf_row.addStretch(1)
        self.conf_badge = badge("—", MUTED)
        conf_row.addWidget(self.conf_badge)
        vlay.addLayout(conf_row)

        rlbl = QLabel("RECOMMENDED ACTION")
        rlbl.setObjectName("tierLabel")
        vlay.addWidget(rlbl)
        self.action_box = QLabel("—")
        self.action_box.setWordWrap(True)
        self.action_box.setStyleSheet(
            f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px;"
            f"padding:10px;color:{MUTED};font-size:12px;"
        )
        vlay.addWidget(self.action_box)

        # ---- Narrative Explanation (LLM + RAG) — inside the verdict column ----
        nsep = QFrame()
        nsep.setFrameShape(QFrame.HLine)
        nsep.setStyleSheet(f"color:{BORDER};background:{BORDER};max-height:1px;")
        vlay.addWidget(nsep)
        ntitle = QLabel("NARRATIVE EXPLANATION (LLM + RAG)")
        ntitle.setObjectName("tierLabel")
        vlay.addWidget(ntitle)
        self.narr_implication = QLabel("Awaiting analysis.")
        self.narr_implication.setWordWrap(True)
        self.narr_implication.setStyleSheet(f"color:{TEXT};font-size:13px;")
        vlay.addWidget(self.narr_implication)

        self.btn_narr = QPushButton("Read full explanation  ▾")
        self.btn_narr.setObjectName("ghost")
        self.btn_narr.clicked.connect(self._toggle_narrative)
        nrow = QHBoxLayout()
        nrow.addWidget(self.btn_narr)
        nrow.addStretch(1)
        self.narr_source = QLabel("")
        self.narr_source.setObjectName("caption")
        nrow.addWidget(self.narr_source)
        vlay.addLayout(nrow)

        self.narr_full = QTextEdit()
        self.narr_full.setReadOnly(True)
        self.narr_full.setMinimumHeight(180)
        self.narr_full.setVisible(False)
        vlay.addWidget(self.narr_full)

        vlay.addStretch(1)
        self.btn_report = QPushButton("Generate Report")
        self.btn_report.setEnabled(False)
        self.btn_report.setToolTip("Export PDF/JSON hasil analisis")
        self.btn_report.clicked.connect(self.on_report)
        vlay.addWidget(self.btn_report)
        zones.addWidget(vcard, 5)

        self.vbody.addLayout(zones, 1)

    # ---- toggles --------------------------------------------------------
    def _toggle_details(self, _s):
        self.detail_widget.setVisible(self.chk_details.isChecked())

    def _toggle_narrative(self):
        self._narrative_expanded = not self._narrative_expanded
        self.narr_full.setVisible(self._narrative_expanded)
        self.btn_narr.setText(
            "Hide full explanation  ▴" if self._narrative_expanded
            else "Read full explanation  ▾"
        )

    # ---- KB + model loading --------------------------------------------
    def _load_kb(self):
        try:
            self.kb = xai.load_kb(DEFAULT_KB)
        except Exception as e:
            QMessageBox.warning(self, "Knowledge base", f"Could not load KB:\n{e}")
            self.kb = {"icdr_scale": {}, "concepts": {}}

    def _start_model_load(self):
        if not os.path.exists(DEFAULT_CHECKPOINT):
            self.status.setText("Checkpoint not found")
            QMessageBox.critical(self, "Model", f"Not found:\n{DEFAULT_CHECKPOINT}")
            return
        self.btn_open.setEnabled(False)
        self._thread = QThread()
        self._loader = ModelLoader(DEFAULT_CHECKPOINT, DEFAULT_SEG_CKPT)
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.failed.connect(self._on_model_failed)
        self._loader.finished.connect(self._thread.quit)
        self._loader.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_model_loaded(self, cbm, seg_model, lesions, seg_size):
        self.cbm = cbm
        self.seg_model = seg_model
        self.seg_lesions = list(lesions)
        self.seg_size = seg_size
        dev = str(inference.DEVICE).upper()
        seg_note = "seg on" if seg_model is not None else "seg off"
        self._set_status(f"● Model ready · {dev} · {seg_note}", "#22c55e")
        self.btn_open.setEnabled(True)
        if self.image_path:
            self.btn_analyze.setEnabled(True)

    def _on_model_failed(self, tb):
        self._set_status("● Model failed", "#ef4444")
        self.btn_open.setEnabled(True)
        QMessageBox.critical(self, "Model load error", tb)

    def _set_status(self, text, color):
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{color};")

    # ---- actions --------------------------------------------------------
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open fundus image", ROOT,
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            self._current_pil = Image.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.warning(self, "Image", f"Could not load:\n{e}")
            return
        self.image_path = path
        self.prob_maps = None
        self.thumb_orig.set_pil(self._current_pil)
        self.thumb_det.set_pil(self._current_pil)
        self.thumb_map.set_pil(Image.new("RGB", (256, 256), (10, 15, 24)))
        if self.cbm is not None:
            self.btn_analyze.setEnabled(True)

    def on_analyze(self):
        if self.cbm is None or not self.image_path:
            return
        self.btn_analyze.setEnabled(False)
        self._set_status("● Analyzing…", ACCENT)
        self._inf_thread = QThread()
        self._worker = InferenceWorker(
            self.cbm, self.seg_model, self.seg_lesions, self.seg_size,
            self.image_path, self.kb, self.thr_spin.value(),
        )
        self._worker.moveToThread(self._inf_thread)
        self._inf_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.failed.connect(self._on_infer_failed)
        self._worker.finished.connect(self._inf_thread.quit)
        self._worker.failed.connect(self._inf_thread.quit)
        self._inf_thread.start()

    def _on_infer_failed(self, tb):
        self._set_status("● Analysis failed", "#ef4444")
        self.btn_analyze.setEnabled(True)
        QMessageBox.critical(self, "Inference error", tb)

    def on_report(self):
        if self._last_result is None:
            return
        default_name = f"reti-ai_report_{datetime.now():%Y%m%d_%H%M}.pdf"
        path, selected = QFileDialog.getSaveFileName(
            self, "Save report", os.path.join(ROOT, default_name),
            "PDF report (*.pdf);;JSON data (*.json)",
        )
        if not path:
            return
        patient = {"name": self.pt_name.text(), "sub": self.pt_sub.text()}
        try:
            if path.lower().endswith(".json") or "json" in selected.lower():
                if not path.lower().endswith(".json"):
                    path += ".json"
                report.export_json(path, patient, self._last_result,
                                   self._last_explanation, self._last_quant)
            else:
                if not path.lower().endswith(".pdf"):
                    path += ".pdf"
                images = {
                    "original": getattr(self.thumb_orig, "_pil", None),
                    "detection": getattr(self.thumb_det, "_pil", None),
                    "segmentation": getattr(self.thumb_map, "_pil", None),
                }
                report.export_pdf(path, patient, self._last_result,
                                  self._last_explanation, self._last_quant, images)
            self._set_status(f"● Report saved · {os.path.basename(path)}", "#22c55e")
        except Exception:
            QMessageBox.critical(self, "Report error", traceback.format_exc())

    def _on_result(self, result, explanation, prob_maps):
        self.btn_analyze.setEnabled(True)
        dev = str(inference.DEVICE).upper()
        seg_note = "seg on" if self.seg_model is not None else "seg off"
        self._set_status(f"● Model ready · {dev} · {seg_note}", "#22c55e")

        self._last_result = result
        self._last_explanation = explanation
        self.prob_maps = prob_maps
        grade = result["grade"]
        base = SEV_COLORS.get(grade, MUTED)

        # verdict
        self._restyle_badge(self.sev_badge, SEVERITY_LABELS.get(grade, "—"), base)
        self.diag_name.setText(DIAGNOSIS_NAMES.get(grade, result["grade_name"]))
        self.diag_name.setStyleSheet(f"color:{TEXT};font-size:20px;font-weight:bold;")
        self._restyle_badge(
            self.conf_badge, f"{result['grade_probability']*100:.0f}%", base
        )

        # recommended action — red styling reserved for urgent (grade >= 3)
        followup = xai.followup_for_grade(grade)
        if grade >= 3:
            self.action_box.setText("⚠  " + followup)
            self.action_box.setStyleSheet(
                f"background:{rgba('#ef4444', 0.14)};border:1px solid {rgba('#ef4444', 0.6)};"
                f"border-radius:8px;padding:10px;color:#fca5a5;font-size:12px;font-weight:bold;"
            )
        else:
            self.action_box.setText(followup)
            self.action_box.setStyleSheet(
                f"background:{PANEL2};border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px;color:{MUTED};font-size:12px;"
            )

        # chart + concept chips
        dist = [result["grade_distribution"][inference.GRADE_NAMES[i]] for i in range(5)]
        self.chart.set_data(dist, grade)
        self.concept_chips.set_data(result["concepts"], result["active_concepts"])

        # narrative tiers
        self.narr_implication.setText(explanation["clinical_interpretation"])
        raw_source = explanation.get("narrative_source", "")
        is_llm = raw_source.startswith("Gemini")
        if is_llm:
            self.narr_source.setText("✓ AI-assisted · grounded in clinical KB (RAG)")
            self.narr_source.setStyleSheet("color:#22c55e;font-size:11px;")
        else:
            self.narr_source.setText("Rule-based · clinical KB")
            self.narr_source.setStyleSheet(f"color:{FAINT};font-size:11px;")
        self.narr_source.setToolTip(
            "Explanation is faithful to the model's detected concepts — no lesions "
            f"added or removed.\nEngine: {raw_source}"
        )
        self.narr_full.setHtml(self._format_full(result, explanation))

        # enable report export
        self.btn_report.setEnabled(True)

        # lesion toggles
        if prob_maps:
            summary = segmentation.lesion_pixel_summary(prob_maps, self.thr_spin.value())
            for cid, chk in self.lesion_checks.items():
                has = summary.get(cid, 0.0) > 0
                chk.setEnabled(has)
                if not has:
                    chk.setChecked(False)
        self._refresh_derived()

    @staticmethod
    def _restyle_badge(lbl, text, base_hex):
        lbl.setText(text)
        lbl.setStyleSheet(
            f"background:{rgba(base_hex, 0.16)};color:{base_hex};"
            f"border:1px solid {rgba(base_hex, 0.55)};border-radius:6px;"
            f"padding:3px 10px;font-size:12px;font-weight:bold;letter-spacing:0.5px;"
        )

    def _on_threshold_changed(self, _v):
        if self.prob_maps:
            self._refresh_derived()

    # ---- derived views --------------------------------------------------
    def _refresh_derived(self):
        if self._current_pil is None:
            return
        self.thumb_orig.set_pil(self._current_pil)
        if not self.prob_maps:
            return
        thr = self.thr_spin.value()
        active = [c for c, chk in self.lesion_checks.items()
                  if chk.isChecked() and chk.isEnabled()]
        overlay_on = self.chk_overlay.isChecked()

        det = segmentation.render_detection(
            self._current_pil, self.prob_maps,
            active_lesions=active if overlay_on else [], threshold=thr,
        )
        self.thumb_det.set_pil(det)
        smap = segmentation.render_segmentation_map(
            self.prob_maps, self.seg_size,
            active_lesions=active if overlay_on else [], threshold=thr,
        )
        self.thumb_map.set_pil(smap)

        q = segmentation.quantitative_summary(
            self._current_pil, self.prob_maps, active_lesions=active, threshold=thr
        )
        self.q_summary.setText(
            f"Lesions: {q['n_lesions']}   ·   Retina coverage: {q['total_area_pct']:.2f}%"
        )
        self.q_vals["n_lesions"].setText(str(q["n_lesions"]))
        self.q_vals["total_area"].setText(f"{q['total_area_px']:,} px")
        self.q_vals["avg_size"].setText(f"{q['avg_area_px']:,.0f} px")
        self.q_vals["area_pct"].setText(f"{q['total_area_pct']:.2f}%")
        self.q_vals["location"].setText(q["location"])
        if self._last_result:
            self.q_vals["confidence"].setText(
                f"{self._last_result['grade_probability']*100:.0f}%"
            )
        self.size_bar.set_data(q["per_lesion"])
        self._last_quant = q

    def _format_full(self, result, expl):
        import html as _html
        narrative = expl.get("llm_narrative") or ""
        body = _html.escape(narrative).replace("\n", "<br>")
        return f"""
        <div style='font-size:12px;line-height:1.55;color:{TEXT};'>
          <div>{body}</div>
          <hr style='border:none;border-top:1px solid {BORDER};margin:10px 0;'>
          <p style='color:{FAINT};'><b>Limitations.</b> {expl['limitations']}</p>
          <p style='color:{FAINT};'><b>Caveats.</b> {expl['caveats']}</p>
        </div>
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
