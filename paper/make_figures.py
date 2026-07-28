"""Generate vector PDF figures from the reported aggregate metrics.

The script uses ReportLab so that paper figures can be rebuilt without a GUI
plotting stack. Only aggregate values documented in the public research summary
are encoded here; no private rows, recordings, or model artifacts are required.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = colors.HexColor("#2563EB")
ORANGE = colors.HexColor("#EA580C")
GREEN = colors.HexColor("#16A34A")
GRAY = colors.HexColor("#64748B")
RED = colors.HexColor("#DC2626")
GRID = colors.HexColor("#D8DEE9")
INK = colors.HexColor("#172033")


def label(c: canvas.Canvas, text: str, x: float, y: float, size: float = 8, *, bold: bool = False, align: str = "left", color=INK) -> None:
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    c.setFillColor(color)
    width = stringWidth(text, font, size)
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    c.drawString(x, y, text)


def axes(c: canvas.Canvas, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float, ticks: list[float]) -> None:
    c.setStrokeColor(GRID)
    c.setLineWidth(0.5)
    for value in ticks:
        y = y0 + (value - ymin) / (ymax - ymin) * height
        c.line(x0, y, x0 + width, y)
        label(c, f"{value:.1f}", x0 - 6, y - 2.5, 6.5, align="right", color=GRAY)
    c.setStrokeColor(INK)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)


def objective_alignment() -> None:
    path = OUT / "objective_alignment.pdf"
    c = canvas.Canvas(str(path), pagesize=(490, 190))
    title = "A useful temporal representation still requires an aligned output objective"
    label(c, title, 245, 171, 9, bold=True, align="center")
    x0, y0, w, h = 47, 43, 422, 107
    axes(c, x0, y0, w, h, 0.0, 1.05, [0.0, 0.5, 1.0])
    names = ["Proxy head", "Base LM head", "Direct action", "Control + response"]
    sub = ["residual only", "residual only", "token LoRA", "token LoRA"]
    vals = [0.989, 0.428, 0.998, 0.998]
    fills = [BLUE, RED, GREEN, GREEN]
    slot = w / len(vals)
    for i, (name, subtitle, value, fill) in enumerate(zip(names, sub, vals, fills)):
        bw = 48
        x = x0 + slot * (i + 0.5) - bw / 2
        bh = value / 1.05 * h
        c.setFillColor(fill)
        c.rect(x, y0, bw, bh, stroke=0, fill=1)
        label(c, f"{value:.3f}", x + bw / 2, y0 + bh + 5, 7.5, bold=True, align="center")
        label(c, name, x + bw / 2, 25, 6.8, align="center")
        label(c, subtitle, x + bw / 2, 15, 6.2, align="center", color=GRAY)
    c.save()


def adapter_ablation() -> None:
    path = OUT / "adapter_ablation.pdf"
    c = canvas.Canvas(str(path), pagesize=(490, 190))
    label(c, "Sequential proxy-head ablation (500 held-out audio timepoints)", 245, 171, 9, bold=True, align="center")
    x0, y0, w, h = 47, 43, 422, 107
    axes(c, x0, y0, w, h, 0.5, 1.02, [0.5, 0.75, 1.0])
    names = ["No time", "Zero", "Correct", "Shuffled", "Random", "Non-time", "Oracle"]
    vals = [0.951, 0.690, 0.989, 0.760, 0.662, 0.765, 0.982]
    fills = [GRAY, GRAY, BLUE, ORANGE, ORANGE, ORANGE, GREEN]
    slot = w / len(vals)
    for i, (name, value, fill) in enumerate(zip(names, vals, fills)):
        bw = 35
        x = x0 + slot * (i + 0.5) - bw / 2
        bh = (value - 0.5) / 0.52 * h
        c.setFillColor(fill)
        c.rect(x, y0, bw, bh, stroke=0, fill=1)
        label(c, f"{value:.3f}", x + bw / 2, y0 + bh + 4, 6.5, align="center")
        label(c, name, x + bw / 2, 22, 6.2, align="center")
    label(c, "Test macro F1", 10, 96, 7, color=GRAY)
    c.save()


def latency_summary() -> None:
    path = OUT / "latency_summary.pdf"
    c = canvas.Canvas(str(path), pagesize=(490, 185))
    label(c, "Pseudo-realtime action scoring on a local RTX 4090", 245, 168, 9, bold=True, align="center")
    x0, y0, w, h = 62, 49, 395, 101
    c.setStrokeColor(GRID)
    for tick in [0, 100, 200, 300, 400, 500]:
        x = x0 + tick / 550 * w
        c.line(x, y0, x, y0 + h)
        label(c, str(tick), x, 36, 6.5, align="center", color=GRAY)
    budget_x = x0 + 500 / 550 * w
    c.setStrokeColor(ORANGE)
    c.setDash(4, 3)
    c.setLineWidth(1.2)
    c.line(budget_x, y0, budget_x, y0 + h)
    c.setDash()
    label(c, "500 ms budget", budget_x - 3, 153, 6.8, align="right", color=ORANGE)
    names = ["p50", "mean", "p90", "p95", "p99"]
    vals = [270.8, 295.8, 300.4, 309.1, 354.0]
    row = h / len(vals)
    for i, (name, value) in enumerate(zip(names, vals)):
        y = y0 + h - (i + 0.78) * row
        label(c, name, x0 - 9, y + 3, 7, align="right")
        c.setFillColor(BLUE)
        c.rect(x0, y, value / 550 * w, 11, stroke=0, fill=1)
        label(c, f"{value:.1f}", x0 + value / 550 * w + 5, y + 2.5, 6.8)
    label(c, "Action-token scoring latency (ms)", 245, 22, 7.2, align="center", color=GRAY)
    label(c, "137 ticks; 99.3% below 500 ms; 3.50 s cold-start maximum omitted", 469, 9, 6.2, align="right", color=GRAY)
    c.save()


if __name__ == "__main__":
    objective_alignment()
    adapter_ablation()
    latency_summary()
    print(f"Wrote vector PDF figures to {OUT}")
