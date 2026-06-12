"""
reports/pdf_report.py
=====================
Generates a professional PDF credit assessment report.
"""

import io, base64, json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)

W, _ = A4
CW   = (W - 30*mm)          # content width

C_NAVY  = colors.HexColor("#0f172a")
C_BLUE  = colors.HexColor("#3b82f6")
C_GREEN = colors.HexColor("#16a34a")
C_AMBER = colors.HexColor("#d97706")
C_RED   = colors.HexColor("#dc2626")
C_GRAY  = colors.HexColor("#6b7280")
C_LIGHT = colors.HexColor("#f8fafc")

DEC_COLOR  = {"APPROVE": C_GREEN, "REVIEW": C_AMBER, "REJECT": C_RED}
STATUS_CLR = {"Approved": C_GREEN, "Pending": C_AMBER, "Rejected": C_RED}
RISK_COLOR = {"LOW": C_GREEN, "MEDIUM": C_AMBER, "HIGH": C_RED}

PRIORITY_CLR = {"high": C_RED, "medium": C_AMBER, "low": C_GREEN, "info": C_BLUE}


def _tbl(data, widths, hdr_color=None):
    t = Table(data, colWidths=widths)
    style = [
        ("FONTSIZE",        (0,0),(-1,-1), 9),
        ("PADDING",         (0,0),(-1,-1), 6),
        ("GRID",            (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS",  (0,1),(-1,-1), [colors.white, C_LIGHT]),
        ("ALIGN",           (0,0),(-1,-1), "LEFT"),
        ("VALIGN",          (0,0),(-1,-1), "MIDDLE"),
    ]
    if hdr_color:
        style += [
            ("BACKGROUND", (0,0),(-1,0), hdr_color),
            ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0),(-1,0), 9),
        ]
    t.setStyle(TableStyle(style))
    return t


def generate_pdf(result: dict, inputs: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=12*mm, bottomMargin=12*mm)

    S = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=S["Normal"], **kw)

    H1   = sty("h1", fontSize=18, fontName="Helvetica-Bold", textColor=C_NAVY,
               alignment=TA_CENTER, spaceAfter=3)
    H2   = sty("h2", fontSize=11, fontName="Helvetica-Bold", textColor=C_NAVY,
               spaceBefore=8, spaceAfter=4)
    BODY = sty("b",  fontSize=9,  leading=13, spaceAfter=3)
    SM   = sty("sm", fontSize=7.5, textColor=C_GRAY)
    CTR  = sty("c",  fontSize=9,  alignment=TA_CENTER)

    story = []

    # ─── Header ────────────────────────────────────────────────────
    story.append(Paragraph("CREDIT RISK ASSESSMENT REPORT", H1))
    story.append(Paragraph(
        f"Application #{result.get('application_id','—')}  •  "
        f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        sty("sub", fontSize=8, alignment=TA_CENTER, textColor=C_GRAY),
    ))
    story.append(HRFlowable(width="100%", thickness=2.5, color=C_NAVY, spaceAfter=6))

    # ─── Decision banner ────────────────────────────────────────────
    decision = result.get("ml_recommendation", "N/A")
    status   = result.get("status", "Pending")
    risk     = result.get("risk_category", "—")
    prob     = result.get("probability", 0)
    sc_color = STATUS_CLR.get(status, C_GRAY)
    rk_color = RISK_COLOR.get(risk, C_GRAY)

    banner = Table([[
        Paragraph(f"<b>AI Recommendation</b><br/><font size='13'><b>{decision}</b></font>", CTR),
        Paragraph(f"<b>Admin Status</b><br/>"
                  f"<font size='14' color='{sc_color.hexval()}'><b>{status}</b></font>", CTR),
        Paragraph(f"<b>Risk Level</b><br/>"
                  f"<font size='14' color='{rk_color.hexval()}'><b>{risk}</b></font>", CTR),
        Paragraph(f"<b>Risk Probability</b><br/><font size='16'><b>{prob*100:.1f}%</b></font>", CTR),
    ]], colWidths=[CW/4]*4)
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#f0f4ff")),
        ("BOX",        (0,0),(-1,-1), 1.5, C_BLUE),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",    (0,0),(-1,-1), 10),
        ("ALIGN",      (0,0),(-1,-1), "CENTER"),
    ]))
    story += [banner, Spacer(1, 5*mm)]

    # ─── Credit score ───────────────────────────────────────────────
    story.append(Paragraph("Credit Score", H2))
    score = result.get("credit_score", 0)
    slabel = result.get("score_label", "—")
    bd    = result.get("score_breakdown", {})
    score_row = [[
        Paragraph(
            f"<font size='26' color='{C_BLUE.hexval()}'><b>{score}</b></font><br/>"
            f"<font size='10'>{slabel}</font><br/>"
            f"<font size='7.5' color='grey'>300 (Poor) → 900 (Excellent)</font>",
            CTR),
        _tbl([["Component","Score/100"]] + [[k.replace("_"," ").title(), str(v)] for k,v in bd.items()],
             [80*mm, 30*mm], hdr_color=C_NAVY),
    ]]
    story.append(Table(score_row, colWidths=[40*mm, 115*mm]))
    story.append(Spacer(1, 4*mm))

    # ─── Model predictions ──────────────────────────────────────────
    mp = result.get("model_predictions", {})
    if mp:
        story.append(Paragraph("Model Predictions (Ensemble)", H2))
        rows = [["Model","Probability","Risk Level"]]
        labels = {"cnn":"CNN","lstm":"BiLSTM","tab":"TabTransformer","ensemble":"Ensemble ★"}
        for k,v in mp.items():
            rl = "Low" if v<0.4 else "Medium" if v<0.7 else "High"
            rows.append([labels.get(k,k), f"{v*100:.1f}%", rl])
        story += [_tbl(rows, [60*mm, 45*mm, 45*mm], hdr_color=C_NAVY), Spacer(1,4*mm)]

    # ─── Applicant inputs ───────────────────────────────────────────
    story.append(Paragraph("Applicant Information", H2))
    in_rows = [["Field","Value"]] + [[k.replace("_"," ").title(), str(v)] for k,v in inputs.items()]
    story += [_tbl(in_rows, [85*mm, 70*mm], hdr_color=C_NAVY), Spacer(1,4*mm)]

    # ─── AI explanation ─────────────────────────────────────────────
    story.append(Paragraph("AI Explanation (SHAP)", H2))
    story.append(Paragraph(result.get("shap_explanation","N/A"), BODY))
    story.append(Spacer(1, 3*mm))

    # ─── Reason codes ───────────────────────────────────────────────
    story.append(Paragraph("Risk Reason Codes", H2))
    rc = result.get("reason_codes", [])
    if isinstance(rc, str):
        rc = [r.strip() for r in rc.split(";") if r.strip()]
    rc_rows = [["#","Risk Factor"]] + [[str(i+1), r] for i,r in enumerate(rc)]
    story += [_tbl(rc_rows, [12*mm, CW-12*mm], hdr_color=colors.HexColor("#7c3aed")), Spacer(1,4*mm)]

    # ─── Recommendations ────────────────────────────────────────────
    story.append(Paragraph("Personalised Recommendations", H2))
    recs = result.get("recommendations", [])
    if isinstance(recs, str):
        try: recs = json.loads(recs)
        except: recs = []
    for r in recs:
        if not isinstance(r, dict): continue
        p  = r.get("priority","medium")
        pc = PRIORITY_CLR.get(p, C_GRAY)
        story.append(KeepTogether([
            Paragraph(f"<font color='{pc.hexval()}'><b>[{p.upper()}]</b></font> "
                      f"<b>{r.get('title','')}</b>",
                      sty("rh", fontSize=9, fontName="Helvetica-Bold", spaceAfter=1)),
            Paragraph(r.get("detail",""), BODY),
            Paragraph(f"<i>Impact: {r.get('impact','')}</i>",
                      sty("ri", fontSize=8, textColor=C_GRAY, spaceAfter=5)),
        ]))

    # ─── SHAP chart ─────────────────────────────────────────────────
    b64 = result.get("shap_chart_b64")
    if b64:
        try:
            story.append(Paragraph("Feature Impact Chart (SHAP)", H2))
            story.append(RLImage(io.BytesIO(base64.b64decode(b64)), width=165*mm, height=88*mm))
            story.append(Spacer(1, 4*mm))
        except Exception:
            pass

    # ─── Disclaimer ─────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0"), spaceBefore=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted system for academic demonstration. "
        "It does not constitute financial or legal advice. Final decisions are made by authorised personnel.",
        sty("disc", fontSize=7, textColor=C_GRAY, alignment=TA_CENTER),
    ))

    doc.build(story)
    return buf.getvalue()
