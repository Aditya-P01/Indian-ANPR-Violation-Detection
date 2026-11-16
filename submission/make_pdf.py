"""Build the single-file submission PDF:  python submission/make_pdf.py"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
OUT = os.path.join(HERE, "Manoj_Chandrappa_Lamani_ANPR_Traffic_Violation_Report.pdf")
REPO = "https://github.com/ManojLamani/Indian-ANPR-Violation-Detection"

ACCENT = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#EAF1F8")
GRID = colors.HexColor("#B8C7D9")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=ACCENT, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=6, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
CAP = ParagraphStyle("C", parent=SMALL, alignment=TA_CENTER, spaceAfter=8)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=21, leading=26, textColor=ACCENT, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=11.5, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths, header=True, bold_first_col=False):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            st = CELLB if (header and r == 0) or (bold_first_col and c == 0) else CELL
            cells.append(Paragraph(str(v), st))
        data.append(cells)
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    t.setStyle(TableStyle(style))
    return KeepTogether([t])   # never split a small table across pages


def fig(name, width_mm, caption):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    width = width_mm * mm
    return KeepTogether([Image(path, width=width, height=width * h / w), P(caption, CAP)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 10 * mm, "Manoj Chandrappa Lamani - Computer Vision Project (5th Semester)")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def pipeline():
    steps = ["Video frame<br/>(4K)", "YOLOv8n<br/>vehicle detector", "ByteTrack<br/>vehicle IDs",
             "YOLOv8n<br/>plate detector", "EasyOCR +<br/>format fix", "Multi-frame<br/>voting"]
    row = []
    for i, s in enumerate(steps):
        row.append(Paragraph(f"<b>{s}</b>", ParagraphStyle("pp", parent=CELL, alignment=TA_CENTER, fontSize=8.5)))
        if i < len(steps) - 1:
            row.append(Paragraph("→", ParagraphStyle("ar", parent=CELL, alignment=TA_CENTER, fontSize=14)))
    w = [24 * mm if i % 2 == 0 else 6 * mm for i in range(len(row))]
    t = Table([row], colWidths=w)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")] +
                          [("BOX", (i, 0), (i, 0), 0.8, ACCENT) for i in range(0, len(row), 2)] +
                          [("BACKGROUND", (i, 0), (i, 0), LIGHT) for i in range(0, len(row), 2)]))
    rule = Table([[Paragraph("<b>Rule engine</b> (from vehicle tracks): stop line + signal state → red-light "
                             "jumping; two lines a known distance apart → speed; crossing order → wrong way. "
                             "Each violation → CSV row + snapshot image + plate.", CELL)]], colWidths=[174 * mm])
    rule.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, ACCENT), ("BACKGROUND", (0, 0), (-1, -1), LIGHT)]))
    return [t, Spacer(1, 4), rule]


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="Traffic Violation Detection with Indian License Plate Recognition",
                            author="Manoj Chandrappa Lamani")
    s = []

    # ---------------- title ----------------
    s += [Spacer(1, 4),
          P("Traffic Violation Detection with<br/>Indian License Plate Recognition (ANPR)", TITLE),
          P("Computer Vision Project - 5th Semester &nbsp;|&nbsp; "
            "Assignment 2: Recognise license plates using OCR", SUB),
          Spacer(1, 8)]
    s.append(table([["Student", "Manoj Chandrappa Lamani"],
                    ["Project", "Detect vehicles violating traffic rules at a junction and identify them by "
                                "reading their number plates with OCR"],
                    ["Code (GitHub)", f'<link href="{REPO}" color="blue">{REPO}</link>'],
                    ["Tech stack", "Python, OpenCV, YOLOv8 (Ultralytics), ByteTrack, EasyOCR, Pandas"]],
                   [32 * mm, 142 * mm], header=False, bold_first_col=True))
    s.append(Spacer(1, 8))

    s.append(P("Summary", H1))
    s.append(P("The system processes traffic-junction video, detects and tracks every vehicle, and flags "
               "<b>red-light jumping, over-speeding and wrong-way driving</b> using virtual lines and the signal "
               "state. For each vehicle a <b>YOLOv8 plate detector trained in this project</b> finds the number "
               "plate in the full-resolution frame, <b>EasyOCR</b> reads it, the text is corrected using the "
               "Indian plate format, and reads from many frames are combined by voting. The output is an "
               "annotated demo video, a violations log with snapshots, and an accuracy report."))
    s.append(table([["Key result", "Value"],
                    ["Plate detector on 882 unseen test images",
                     "<b>96.6 % precision, 85.7 % recall, 93.2 % mAP@0.5</b>"],
                    ["Red-light violations in the Mumbai demo (checked by eye)",
                     "<b>3 / 3 detected, 0 false alarms</b> (precision = recall = 100 %)"],
                    ["Plates read exactly right", "Mumbai: 3 / 12 (25 %) &nbsp;|&nbsp; Bhopal: 2 / 3 (67 %)"],
                    ["Processing speed", "1.8 frames/s on 4K video, laptop CPU only (no GPU)"]],
                   [84 * mm, 90 * mm]))
    s.append(Spacer(1, 6))
    s.append(fig("frame_red.jpg", 174, "Figure 1 - Demo output at 11.4 s (Mumbai). The signal is RED; taxis that "
                                      "cross the red STOP LINE are boxed in red with their plate and the "
                                      "violation. Yellow lines A/B are used for speed."))

    # ---------------- method ----------------
    s.append(PageBreak())
    s.append(P("1. System design", H1))
    s += pipeline()
    s.append(Spacer(1, 6))
    s.append(P("1.1 Vehicle detection and tracking", H2))
    s.append(P("Each frame is resized to 1280 px and passed to a COCO-pretrained YOLOv8n model limited to car, "
               "motorcycle, bus and truck. ByteTrack links detections over time so each vehicle keeps one ID. "
               "The bottom-centre of the box (where the wheels touch the road) is the vehicle's position for "
               "all line-crossing rules."))
    s.append(P("1.2 Plate detection", H2))
    s.append(P("A YOLOv8n plate detector (Section 2) searches only inside each vehicle, so shop signs and road "
               "signs are not mistaken for plates. The vehicle is cropped from the <b>original 4K frame</b>, not "
               "the resized one, giving OCR about three times more pixels per character."))
    s.append(P("1.3 OCR and Indian plate-format correction", H2))
    s += bullets([
        "Pre-processing: grayscale, upscale to ~96 px height, bilateral filter, CLAHE contrast.",
        "EasyOCR with an A-Z/0-9 allow-list; text boxes are merged row by row so two-line plates read correctly.",
        "If no valid plate is read, retry on a deskewed (Hough-line angle) and an Otsu-binarised version.",
        "Indian plates follow <b>SS DD [A-AAA] NNNN</b> (state, district, series, number). Each character is "
        "fixed by its position: e.g. <i>0</i> → <i>O</i> in the state part, <i>S</i> → <i>5</i> and "
        "<i>B</i> → <i>8</i> in the number part. A read is accepted only if it starts with a real state code.",
    ])
    s.append(P("1.4 Multi-frame voting", H2))
    s.append(P("A vehicle is read up to every 3rd frame. Reads are weighted by OCR confidence, format validity "
               "and plate size, then combined <b>character by character</b>, so several partly wrong reads can "
               "still produce the correct plate. OCR stops for a vehicle after 4 identical valid reads."))
    s.append(P("1.5 Violation rules", H2))
    s.append(table([["Violation", "Rule"],
                    ["Red-light jumping", "Wheel point crosses the stop line while the signal is RED "
                                          "(signal from a schedule, a timed cycle or lamp-colour detection)."],
                    ["Over-speeding", "Distance between two lines ÷ time taken between them → km/h, "
                                      "compared with the limit."],
                    ["Wrong way", "The two speed lines are crossed in the reverse order."]],
                   [38 * mm, 136 * mm]))

    # ---------------- training ----------------
    s.append(P("2. Plate detector training", H1))
    s.append(table([["Item", "Value"],
                    ["Dataset", "keremberke/license-plate-object-detection (Roboflow, CC BY 4.0): 6,176 train / "
                                "1,765 validation / 882 test images, one class"],
                    ["Model", "YOLOv8n starting from COCO weights; horizontal flip disabled (mirrored text is "
                              "not a real plate)"],
                    ["Training", "Laptop CPU only. 3 epochs on 50 % of the train set at 416 px, then 4 "
                                 "fine-tuning epochs (lr 0.002). Best validation mAP@0.5 = 0.922"]],
                   [30 * mm, 144 * mm]))
    s.append(Spacer(1, 4))
    s.append(table([["Test split (882 images, 902 plates, never used in training)", "Value"],
                    ["Precision", "<b>96.6 %</b>"], ["Recall", "<b>85.7 %</b>"],
                    ["mAP@0.5", "<b>93.2 %</b>"], ["mAP@0.5:0.95", "57.7 %"]],
                   [110 * mm, 64 * mm]))

    # ---------------- demo + results ----------------
    s.append(P("3. Demo setup and ground truth", H1))
    s.append(table([["", "Mumbai junction (main demo)", "Bhopal intersection (second test)"],
                    ["Footage", "Pexels 33604950, 4K, 30 fps, 19 s, handheld", "Pexels 34430025, 4K, 60 fps, 16 s, panning"],
                    ["Rules", "Red light + speed (30 km/h, lines ~8 m apart)", "Plate reading only"],
                    ["Signal", "RED from 9.7 s (read frame by frame from the visible light)", "-"],
                    ["Demo video", "demo/mumbai_violations_demo.mp4", "demo/bhopal_plates_demo.mp4"]],
                   [24 * mm, 75 * mm, 75 * mm]))
    s.append(Spacer(1, 4))
    s.append(P("<b>Ground truth</b> was made by hand: every plate crop in the clip (sampled every 0.5 s at full "
               "4K resolution) was read by eye, and every vehicle was checked against the stop line and signal in "
               "the annotated video. Plates that even a person cannot read for certain were left out "
               "(3 in Mumbai)."))

    s.append(P("4. Results", H1))
    s.append(P("4.1 Violation detection (Mumbai)", H2))
    s.append(table([["Metric", "Value"],
                    ["True red-light violations / detected", "3 / 3"],
                    ["True positives / false positives / false negatives", "3 / 0 / 0"],
                    ["Precision / recall / F1", "<b>100 % / 100 % / 100 %</b>"],
                    ["Over-speeding", "0 flagged; the measured speeds were 10 and 7 km/h (busy junction)"]],
                   [100 * mm, 74 * mm]))
    s.append(Spacer(1, 4))
    s.append(fig("violations.jpg", 174, "Figure 2 - The three red-light violations saved by the system "
                                        "(10.2 s, 11.4 s, 14.5 s). Each taxi crosses the stop line after the light "
                                        "turned red."))
    s.append(P("4.2 Plate recognition", H2))
    s.append(table([["Metric", "Mumbai (final)", "Mumbai (OCR-only baseline)", "Bhopal"],
                    ["Readable plates (ground truth)", "12", "12", "3"],
                    ["Read exactly right", "3 (25 %)", "3 (25 %)", "2 (67 %)"],
                    ["Plates reported (valid format)", "6", "7", "6"],
                    ["Plate precision", "<b>50 %</b>", "43 %", "33 %"],
                    ["Character accuracy", "64 %", "69 %", "93 %"]],
                   [62 * mm, 36 * mm, 42 * mm, 34 * mm]))
    s.append(Spacer(1, 4))
    s.append(fig("plates.jpg", 174, "Figure 3 - Examples. Top row and bottom-left: read correctly (green). "
                                    "Others: wrong reads (red) - two characters off, motion blur, and the word "
                                    "'BHOPAL' on a bus."))

    # ---------------- analysis ----------------
    s.append(P("5. Error analysis", H1))
    s.append(P("The <b>plate detector is strong</b> (93 % mAP@0.5): almost every visible plate in the clips gets a "
               "box. The <b>OCR step is the weak link</b>. The baseline (EasyOCR's own text detector on the whole "
               "vehicle) also read road-sign text such as \"KARVE ROAD\" as plates; the trained detector removes "
               "that, which is why precision rises from 43 % to 50 %."))
    s.append(table([["Cause of error", "Example"],
                    ["Decorative dotted fonts on Mumbai taxis", "MH.01.BT.6915 written with dots and curly serifs"],
                    ["Motion blur from the handheld camera", "MH01AT1825 read as MH014745"],
                    ["Small, far plates (< 60 px wide)", "Plates behind the first row of cars"],
                    ["Tracker ID switches when pedestrians block a car", "One car split into two IDs; the second ID "
                                                                          "collects fewer reads (AP04ZN2562)"],
                    ["Text on vehicles that fits the plate pattern", "'BHOPAL' on a bus read as BH0P4141 - fixed "
                                                                     "after the run (BH is not a state code)"]],
                   [70 * mm, 104 * mm]))
    s.append(Spacer(1, 4))
    s.append(P("Plain modern fonts (Bhopal) give 93 % character accuracy, against 64 % on Mumbai's decorative "
               "taxi plates - the font, not the pipeline design, is the main factor."))

    s.append(P("6. Limitations and future work", H1))
    s += bullets([
        "<b>OCR:</b> EasyOCR is a general text reader. A plate-specific recogniser (CRNN / PARSeq) trained on "
        "Indian plates is the biggest expected improvement.",
        "<b>Camera:</b> footage is handheld, so the virtual lines move with the camera. A fixed CCTV view would "
        "make line crossing exact.",
        "<b>Signal:</b> the signal phase was given as a schedule read from the video; a real system would take it "
        "from the signal controller or detect the lamp automatically.",
        "<b>Speed:</b> line distance was estimated from the zebra crossing, so speeds are approximate.",
        "<b>Training:</b> only 7 CPU epochs on half the data; 50+ GPU epochs at 640 px on the full set "
        "(about 20 min on free Colab) should improve recall.",
        "<b>Speed of processing:</b> 1.8 FPS on CPU is fine for recorded video; live use needs a GPU.",
    ])

    s.append(P("7. How to run", H1))
    code = ParagraphStyle("code", parent=CELL, fontName="Courier", fontSize=8.5, leading=11)
    t = Table([[Paragraph("pip install -r requirements.txt<br/>"
                          "python -m unittest discover tests &nbsp;&nbsp;# 14 unit tests<br/>"
                          "python main.py --config configs/mumbai.yaml &nbsp;&nbsp;# demo video + CSV logs<br/>"
                          "python evaluate.py --gt data/ground_truth_mumbai.csv --outputs outputs/mumbai<br/>"
                          "python train_plate_detector.py --data datasets/plates/data.yaml", code)]],
              colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F4F4")),
                           ("BOX", (0, 0), (-1, -1), 0.5, GRID)]))
    s.append(t)
    s.append(Spacer(1, 4))
    s.append(P(f'Full code, trained weights, demo videos, ground truth and detailed report: '
               f'<link href="{REPO}" color="blue">{REPO}</link>'))

    s.append(P("8. Conclusion", H1))
    s.append(P("The project delivers a complete, working pipeline from junction video to a violations log with "
               "plate numbers. Vehicle tracking and rule checking are reliable (all red-light violations caught "
               "with no false alarms), and the custom plate detector reaches 93 % mAP@0.5 on unseen images. Plate "
               "reading works well on standard fonts (67 % exact, 93 % of characters on the Bhopal clip) and is "
               "the clear next area to improve for decorative fonts and blurred plates."))

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
