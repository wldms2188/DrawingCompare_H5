from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

OUT = Path(__file__).resolve().parent / "synthetic_data"
W, H = A4


def draw(c, after=False):
    x, y, w, h = 35*mm, 105*mm, 120*mm, 85*mm
    c.rect(x, y, w, h)
    c.rect(x+25*mm, y+25*mm, 55*mm, 30*mm)
    c.line(x+80*mm, y+40*mm, x+105*mm, y+40*mm)

    # Dimension: 25 -> 30
    c.line(x+25*mm, y+18*mm, x+80*mm, y+18*mm)
    c.line(x+25*mm, y+16*mm, x+25*mm, y+20*mm)
    c.line(x+80*mm, y+16*mm, x+80*mm, y+20*mm)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x+52.5*mm, y+12*mm, "30" if after else "25")

    # GD&T-like frame: 0.05 -> 0.10
    gx, gy = x+18*mm, y+68*mm
    c.rect(gx, gy, 38*mm, 9*mm)
    c.line(gx+12*mm, gy, gx+12*mm, gy+9*mm)
    c.line(gx+25*mm, gy, gx+25*mm, gy+9*mm)
    c.setFont("Helvetica", 7)
    c.drawCentredString(gx+6*mm, gy+2.8*mm, "POS")
    c.drawCentredString(gx+18.5*mm, gy+2.8*mm, "0.10" if after else "0.05")
    c.drawCentredString(gx+31.5*mm, gy+2.8*mm, "A")

    # NOTE block: AL6061 -> AL7075
    nx, ny, nw, nh = 170*mm, 125*mm, 28*mm, 55*mm
    c.rect(nx, ny, nw, nh)
    c.setFont("Helvetica", 7)
    lines = ["NOTE", "1. DEBURR", "2. MATERIAL:", "   AL7075" if after else "   AL6061", "3. FINISH: NONE"]
    ty = ny + nh - 8*mm
    for line in lines:
        c.drawString(nx+2*mm, ty, line)
        ty -= 7*mm

    # Another dimension: Ø8 -> Ø10
    c.drawString(x+88*mm, y+67*mm, "Ø10" if after else "Ø8")
    c.line(x+90*mm, y+64*mm, x+90*mm, y+55*mm)

    # Unchanged decoy box
    dx, dy = 35*mm, 45*mm
    c.rect(dx, dy, 55*mm, 35*mm)
    c.drawString(dx+5*mm, dy+22*mm, "FIXED")
    c.drawString(dx+5*mm, dy+12*mm, "AREA")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(35*mm, 215*mm, "SYNTHETIC DRAWING COMPARE TEST")
    c.setFont("Helvetica", 8)
    c.drawString(35*mm, 209*mm, "AFTER" if after else "BEFORE")


def make(path, after=False, scale=1.0):
    c = canvas.Canvas(str(path), pagesize=A4)
    if scale != 1.0:
        c.translate(W*(1-scale)/2, H*(1-scale)/2)
        c.scale(scale, scale)
    draw(c, after)
    c.showPage()
    c.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make(OUT / "synthetic_before.pdf", after=False, scale=1.0)
    make(OUT / "synthetic_after.pdf", after=True, scale=0.94)
    print(OUT)
