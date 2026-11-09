#!/usr/bin/env python3
"""
Generate a small but realistic, deliberately messy file collection for trying out / testing the finder.

    python scripts/make_sample_corpus.py ./sample_corpus

Dev-only dependency: reportlab (for the text PDF). Everything else is already in requirements.txt.
Includes: a scanned (image-only) PDF to exercise OCR, files with different modified dates (for date filters),
exact part numbers / codes (for BM25), and unhelpful file names.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DAY = 86400


def set_age(path: Path, days_ago: float) -> None:
    t = time.time() - days_ago * DAY
    os.utime(path, (t, t))


def _font(size: int):
    from PIL import ImageFont
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_image(lines: list[str], path: Path, size: int = 30, width: int = 1400) -> None:
    from PIL import Image, ImageDraw
    font = _font(size)
    h = 60 + len(lines) * (size + 18)
    img = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for ln in lines:
        d.text((40, y), ln, fill="black", font=font)
        y += size + 18
    img.save(path)


def make(out: str) -> dict[str, Path]:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "projects").mkdir(exist_ok=True)
    (root / "personal").mkdir(exist_ok=True)
    (root / "archive").mkdir(exist_ok=True)
    made: dict[str, Path] = {}

    # ---- 1. XLSX: pump efficiency calculations (the "Q3 pump efficiency" example)
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Q3 Calc"
    ws.append(["Pump ID", "Flow (m3/h)", "Head (m)", "Shaft power (kW)", "Hydraulic power (kW)", "Efficiency (%)"])
    for i, (q, h, p) in enumerate([(120, 32, 14.1), (150, 30, 16.2), (95, 35, 11.0), (180, 28, 17.9)], start=1):
        hyd = q / 3600 * 1000 * 9.81 * h / 1000
        ws.append([f"P-10{i}", q, h, p, round(hyd, 2), round(hyd / p * 100, 1)])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Q3 pump efficiency calculation uses hydraulic power / shaft power. Impeller trim applied to P-103."])
    p = root / "projects" / "Book1_final2.xlsx"
    wb.save(p); set_age(p, 25); made["pump_xlsx"] = p

    # ---- 2. DOCX: stress tensor for gear shaft (recent: "last month")
    from docx import Document
    d = Document()
    d.add_heading("Shaft check - draft", level=1)
    d.add_paragraph("Material: 42CrMo4, yield strength 900 MPa. Torque 1250 Nm at 1500 rpm.")
    d.add_heading("Stress tensor at the keyway", level=2)
    d.add_paragraph("We computed the stress tensor for the gear shaft at section B. The principal stresses are "
                    "sigma_1 = 212 MPa and sigma_3 = -48 MPa. Von Mises equivalent stress = 238 MPa, safety factor 3.8.")
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate([["Section", "sigma_x (MPa)", "tau_xy (MPa)"], ["A", "120", "35"], ["B", "212", "61"]]):
        for c, v in enumerate(row):
            t.rows[r].cells[c].text = v
    p = root / "projects" / "shaft_calc_v2.docx"
    d.save(p); set_age(p, 18); made["shaft_docx"] = p

    # ---- 3. TXT: older gear shaft notes, no stress tensor (distractor, 8 months old)
    p = root / "archive" / "gear_notes.txt"
    p.write_text("Gear box ideas\n\nThe gear shaft supplier recommended induction hardening. "
                 "Lead time for the gear shaft forging is six weeks. Ask about surface finish Ra 0.8.\n"
                 "Remember to call the foundry about pricing.\n")
    set_age(p, 240); made["gear_notes"] = p

    # ---- 4. PDF with text: supplier quote containing an exact part number
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    p = root / "projects" / "scan_0091.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    c.setFont("Helvetica-Bold", 14); c.drawString(72, 780, "QUOTATION Q-2291")
    c.setFont("Helvetica", 11)
    for i, ln in enumerate([
        "Supplier: Northfield Bearings Ltd.",
        "Item 1: Deep groove ball bearing, part number XJ-4471-B, qty 40, unit price EUR 18.40",
        "Item 2: Seal kit SK-220, qty 40, unit price EUR 3.15",
        "Delivery: 4 weeks ex works. Payment: 30 days net.",
    ]):
        c.drawString(72, 740 - i * 18, ln)
    c.showPage()
    c.setFont("Helvetica", 11)
    c.drawString(72, 780, "Terms and conditions. Prices are valid for 60 days. Warranty 12 months on all bearings.")
    c.save(); set_age(p, 60); made["quote_pdf"] = p

    # ---- 5. scanned PDF (image only -> needs OCR): insurance policy
    import pymupdf
    img_path = root / "_tmp_scan.png"
    text_image([
        "HOMESAFE INSURANCE - POLICY SCHEDULE",
        "Policy number: HX-99231",
        "Insured: Maria Chen, 48 Birchwood Lane",
        "Coverage: Building 450,000 EUR; Contents 60,000 EUR",
        "Annual premium: 612 EUR. Renewal date: 1 March.",
    ], img_path)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(20, 40, 575, 330), filename=str(img_path))
    p = root / "personal" / "scanned_0007.pdf"
    doc.save(str(p)); doc.close(); img_path.unlink(); set_age(p, 100); made["scanned_pdf"] = p

    # ---- 6. image: receipt photo
    p = root / "personal" / "IMG_20260314_221007.png"
    text_image(["GREENLEAF HARDWARE", "Receipt 48213", "Cordless drill kit  89.99", "Box of screws  6.49",
                "Paint roller set  14.25", "Total  119.88"], p, size=34, width=1000)
    set_age(p, 40); made["receipt_img"] = p

    # ---- 7. PPTX: investor deck
    from pptx import Presentation
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1]); s.shapes.title.text = "Series A pitch - Loomly"
    s.placeholders[1].text = "Home inventory tracking with computer vision"
    s = prs.slides.add_slide(prs.slide_layouts[1]); s.shapes.title.text = "The ask"
    s.placeholders[1].text = "Raising 2.5M USD seed extension at a 14M cap, closing end of Q3."
    s.notes_slide.notes_text_frame.text = "Mention that runway is 22 months after the raise."
    p = root / "deck2.pptx"
    prs.save(p); set_age(p, 12); made["deck"] = p

    # ---- 8. CSV: contacts
    p = root / "export.csv"
    p.write_text("Name,Department,Email,Extension\nDana Whitfield,Finance,dana.w@corp.com,4521\n"
                 "Marcus Yu,Engineering,marcus.y@corp.com,4109\nIsabel Reyes,Human Resources,isabel.r@corp.com,4287\n")
    set_age(p, 5); made["contacts_csv"] = p

    # ---- 9. Python source
    p = root / "projects" / "utils_old.py"
    p.write_text('"""Hydraulic helpers."""\n\n\ndef pump_head(pressure_pa, density=998.0, g=9.81):\n'
                 '    """Convert discharge pressure to pump head in metres."""\n    return pressure_pa / (density * g)\n\n\n'
                 'def hydraulic_power_kw(flow_m3h, head_m):\n    return flow_m3h / 3600 * 998 * 9.81 * head_m / 1000\n')
    set_age(p, 90); made["py"] = p

    # ---- 10. Markdown: personal recipe + meeting notes
    p = root / "personal" / "untitled.md"
    p.write_text("# Sourdough starter\n\nFeed the starter every 12 hours with equal parts flour and water by weight. "
                 "Keep it around 24C. If it smells like nail polish remover it is hungry.\n")
    set_age(p, 200); made["sourdough"] = p

    # ---- distractors with plausible vocabulary overlap
    distractors = {
        "projects/pump_maintenance.md": "# Pump maintenance\n\nQuarterly pump maintenance: check seals, bearings and vibration. Log findings in the CMMS.\n",
        "projects/hvac_efficiency_report.txt": "HVAC efficiency report\n\nChiller efficiency improved 4% after the retrofit. Energy savings estimated at 38 MWh per year.\n",
        "personal/travel_plan.txt": "Trip to Lisbon\n\nFlights booked for the 14th. Hotel near Alfama. Museum tickets for Thursday.\n",
        "archive/hr_policy.md": "# Leave policy\n\nEmployees accrue 2 days of paid leave per month. Requests go through the HR portal.\n",
        "projects/stress_test_plan.md": "# Load test plan\n\nWe will stress test the API with 5k requests per second and measure p99 latency.\n",
        "personal/budget_notes.txt": "Household budget\n\nRent 1200, groceries 400, transport 90. Savings target 300 per month.\n",
    }
    for rel, txt in distractors.items():
        p = root / rel
        p.write_text(txt)
        set_age(p, 30 + len(rel))
    return made


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_corpus"
    files = make(out)
    print(f"Created {len(files)} key files (+ distractors) in {out}")
    for k, v in files.items():
        print(f"  {k:14s} {v}")
