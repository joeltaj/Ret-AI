"""
Report export for Reti-AI.

Builds a self-contained clinical decision-support report (PDF or JSON) from the
real analysis results. PDF is produced with Qt's built-in QTextDocument +
QPrinter — no extra dependencies.

The report is explicitly labelled as AI decision support, not a diagnosis.
"""

import json
import html as _html
from datetime import datetime

from PySide6.QtCore import QUrl, QMarginsF
from PySide6.QtGui import QTextDocument, QImage, QPageLayout, QPageSize
from PySide6.QtPrintSupport import QPrinter

import src.inference


def _pil_to_qimage(pil, max_w=560):
    pil = pil.convert("RGB")
    if pil.width > max_w:
        h = int(pil.height * max_w / pil.width)
        pil = pil.resize((max_w, h))
    data = pil.tobytes("raw", "RGB")
    return QImage(data, pil.width, pil.height, 3 * pil.width,
                  QImage.Format_RGB888).copy()


def _esc(text):
    return _html.escape(str(text))


def _build_html(patient, result, explanation, quant, images):
    grade = result["grade"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # fundus image cells (only those available)
    img_cells = ""
    captions = {
        "original": "Original fundus",
        "detection": "Lesion detection",
        "segmentation": "Segmentation mask",
    }
    for key, cap in captions.items():
        if images.get(key) is not None:
            img_cells += (
                f"<td style='text-align:center;padding:4px;'>"
                f"<img src='{key}' width='230'><br>"
                f"<span style='font-size:9pt;color:#555;'>{cap}</span></td>"
            )
    img_row = f"<table width='100%'><tr>{img_cells}</tr></table>" if img_cells else ""

    # concepts
    concept_rows = ""
    for c in inference.CONCEPT_COLS:
        p = result["concepts"][c]
        fired = c in result["active_concepts"]
        mark = "●" if fired else "○"
        weight = "bold" if fired else "normal"
        concept_rows += (
            f"<tr><td>{mark} {c} — {inference.CONCEPT_NAMES.get(c, c)}</td>"
            f"<td align='right' style='font-weight:{weight};'>{p*100:.0f}%</td></tr>"
        )

    # grade distribution
    dist_rows = ""
    for i in range(5):
        p = result["grade_distribution"][inference.GRADE_NAMES[i]]
        weight = "bold" if i == grade else "normal"
        dist_rows += (
            f"<tr><td style='font-weight:{weight};'>Grade {i} — {inference.GRADE_NAMES[i]}</td>"
            f"<td align='right' style='font-weight:{weight};'>{p*100:.1f}%</td></tr>"
        )

    # quantitative
    q_rows = ""
    if quant:
        q_pairs = [
            ("Number of lesions", quant.get("n_lesions", "—")),
            ("Total lesion area", f"{quant.get('total_area_px', 0):,} px"),
            ("Avg lesion size", f"{quant.get('avg_area_px', 0):,.0f} px"),
            ("Retina coverage", f"{quant.get('total_area_pct', 0):.2f}%"),
            ("Dominant location", quant.get("location", "—")),
        ]
        for k, v in q_pairs:
            q_rows += f"<tr><td>{_esc(k)}</td><td align='right'>{_esc(v)}</td></tr>"

    narrative = _esc(explanation.get("llm_narrative", "")).replace("\n", "<br>")
    source = explanation.get("narrative_source", "")
    engine = "AI-assisted (RAG-grounded)" if source.startswith("Gemini") else "Rule-based (clinical KB)"

    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#1a1a1a;">
      <div style="border-bottom:3px solid #1565c0;padding-bottom:6px;">
        <span style="font-size:20pt;font-weight:bold;color:#1565c0;">Reti-AI</span>
        <span style="font-size:11pt;color:#555;"> · Diabetic Retinopathy Decision Support Report</span>
      </div>
      <p style="font-size:9pt;color:#666;">Generated: {now}</p>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Patient</h3>
      <p><b>{_esc(patient.get('name', '—'))}</b><br>
         <span style="color:#555;">{_esc(patient.get('sub', ''))}</span></p>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Final Classification</h3>
      <p style="font-size:14pt;"><b>{_esc(inference.GRADE_NAMES[grade])}</b>
         &nbsp; (ICDR grade {grade})</p>
      <p>Confidence: <b>{result['grade_probability']*100:.0f}%</b></p>

      {img_row}

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Detected Concepts (CBM)</h3>
      <table width="60%">{concept_rows}</table>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Grade Distribution</h3>
      <table width="60%">{dist_rows}</table>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Quantitative Analysis</h3>
      <table width="60%">{q_rows}</table>
      <p style="font-size:8pt;color:#888;">Measurements from segmentation masks
         (pixels / % of retina). No physical (mm) calibration assumed.</p>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Explainable Narrative</h3>
      <p style="font-size:8pt;color:#888;">Engine: {engine}. Faithful to the model's
         detected concepts — no lesions added or removed.</p>
      <div style="font-size:10pt;line-height:1.4;">{narrative}</div>

      <h3 style="color:#333;border-bottom:1px solid #ccc;">Limitations &amp; Caveats</h3>
      <p style="font-size:9pt;color:#555;">{_esc(explanation.get('limitations', ''))}</p>
      <p style="font-size:9pt;color:#555;">{_esc(explanation.get('caveats', ''))}</p>

      <div style="margin-top:16px;padding:8px;background:#fff3f3;border:1px solid #f0c0c0;">
        <b style="color:#a11;">Disclaimer.</b>
        <span style="font-size:9pt;">This is an AI decision-support report, not a
        diagnosis. Clinical confirmation requires ophthalmological assessment.</span>
      </div>
    </body></html>
    """


def export_pdf(path, patient, result, explanation, quant, images):
    doc = QTextDocument()
    for name, pil in images.items():
        if pil is not None:
            doc.addResource(QTextDocument.ImageResource, QUrl(name), _pil_to_qimage(pil))
    doc.setHtml(_build_html(patient, result, explanation, quant, images))

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(path)
    printer.setPageSize(QPageSize(QPageSize.A4))
    printer.setPageMargins(QMarginsF(14, 14, 14, 14), QPageLayout.Millimeter)
    doc.print_(printer)
    return path


def export_json(path, patient, result, explanation, quant):
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "patient": patient,
        "classification": {
            "grade": result["grade"],
            "grade_name": result["grade_name"],
            "confidence": result["grade_probability"],
            "grade_distribution": result["grade_distribution"],
        },
        "concepts": result["concepts"],
        "active_concepts": result["active_concepts"],
        "quantitative": quant,
        "narrative": explanation.get("llm_narrative", ""),
        "narrative_source": explanation.get("narrative_source", ""),
        "limitations": explanation.get("limitations", ""),
        "caveats": explanation.get("caveats", ""),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
