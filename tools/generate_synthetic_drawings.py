from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "synthetic_dataset"
BEFORE = OUT / "Before"
AFTER = OUT / "After"


def draw_common(c, page_id, page_size):
    pw, ph = page_size
    sx, sy = pw / A4[0], ph / A4[1]
    X = lambda v: v * sx
    Y = lambda v: v * sy

    # Main drawing boundary and internal feature box.
    x, y, w, h = X(30 * mm), Y(100 * mm), X(125 * mm), Y(90 * mm)
    c.rect(x, y, w, h)
    c.rect(x + X(28 * mm), y + Y(25 * mm), X(58 * mm), Y(32 * mm))
    c.line(x + X(86 * mm), y + Y(45 * mm), x + X(112 * mm), y + Y(45 * mm))
    c.circle(x + X(58 * mm), y + Y(41 * mm), X(7 * mm))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(X(25 * mm), Y(255 * mm), f"DRAWING PAGE {page_id}")

    c.setFont("Helvetica", 8)
    c.drawString(X(20 * mm), Y(18 * mm), "COMMON TITLE BLOCK")
    return X, Y, x, y, w, h


def draw_alpha(c, changed, page_size):
    X, Y, x, y, w, h = draw_common(c, "ALPHA", page_size)
    c.setFont("Helvetica", 9)
    c.drawCentredString(x + X(57 * mm), y + Y(12 * mm), "30" if changed else "25")
    c.drawString(x + X(91 * mm), y + Y(69 * mm), "Ø10" if changed else "Ø8")
    if changed:
        c.drawString(x + X(66 * mm), y + Y(7 * mm), "ADDED")
    else:
        c.drawString(x + X(66 * mm), y + Y(7 * mm), "REMOVE_ME")


def draw_beta(c, changed, page_size):
    X, Y, x, y, w, h = draw_common(c, "BETA", page_size)
    gx, gy = x + X(18 * mm), y + Y(68 * mm)
    c.rect(gx, gy, X(40 * mm), Y(9 * mm))
    c.line(gx + X(13 * mm), gy, gx + X(13 * mm), gy + Y(9 * mm))
    c.line(gx + X(27 * mm), gy, gx + X(27 * mm), gy + Y(9 * mm))
    c.setFont("Helvetica", 7)
    c.drawCentredString(gx + X(6.5 * mm), gy + Y(2.8 * mm), "POS")
    c.drawCentredString(gx + X(20 * mm), gy + Y(2.8 * mm), "0.10" if changed else "0.05")
    c.drawCentredString(gx + X(33.5 * mm), gy + Y(2.8 * mm), "A")

    nx, ny, nw, nh = X(168 * mm), Y(122 * mm), X(30 * mm), Y(58 * mm)
    c.rect(nx, ny, nw, nh)
    c.setFont("Helvetica", 7)
    lines = ["NOTE", "1. DEBURR", "2. MATERIAL:", "   AL7075" if changed else "   AL6061", "3. FINISH: NONE"]
    if changed:
        lines.append("4. UPDATED")
    ty = ny + nh - Y(8 * mm)
    for line in lines:
        c.drawString(nx + X(2 * mm), ty, line)
        ty -= Y(7 * mm)


def draw_gamma(c, page_size):
    X, Y, *_ = draw_common(c, "GAMMA", page_size)
    c.setFont("Helvetica", 9)
    c.drawString(X(55 * mm), Y(155 * mm), "GAMMA BASE")
    c.rect(X(50 * mm), Y(135 * mm), X(70 * mm), Y(35 * mm))
    c.line(X(55 * mm), Y(140 * mm), X(115 * mm), Y(165 * mm))
    c.drawString(X(55 * mm), Y(120 * mm), "UNCHANGED")


def draw_delta(c, changed, page_size):
    X, Y, x, y, w, h = draw_common(c, "DELTA", page_size)
    c.setFont("Helvetica", 9)
    c.drawString(x + X(8 * mm), y + Y(70 * mm), "R12" if changed else "R10")
    c.drawString(x + X(8 * mm), y + Y(60 * mm), "SURFACE A")
    # A small geometry-only change: line endpoint moves in After.
    if changed:
        c.line(x + X(12 * mm), y + Y(35 * mm), x + X(92 * mm), y + Y(52 * mm))
    else:
        c.line(x + X(12 * mm), y + Y(35 * mm), x + X(92 * mm), y + Y(35 * mm))


def make_pdf(path, order, changed, sizes):
    c = canvas.Canvas(str(path), pagesize=sizes[0])
    for i, page_id in enumerate(order):
        if i:
            c.setPageSize(sizes[i])
        fn = {"ALPHA": draw_alpha, "BETA": draw_beta, "GAMMA": lambda c, ch, sz: draw_gamma(c, sz), "DELTA": draw_delta}[page_id]
        fn(c, page_id in changed, sizes[i])
        c.showPage()
    c.save()


def main():
    BEFORE.mkdir(parents=True, exist_ok=True)
    AFTER.mkdir(parents=True, exist_ok=True)
    make_pdf(BEFORE / "synthetic_before.pdf", ["ALPHA", "BETA", "GAMMA", "DELTA"], set(), [A4] * 4)
    make_pdf(AFTER / "synthetic_after.pdf", ["BETA", "DELTA", "GAMMA", "ALPHA"], {"ALPHA", "BETA", "DELTA"}, [LETTER, A4, LETTER, A4])
    print(f"Created: {BEFORE}")
    print(f"Created: {AFTER}")


if __name__ == "__main__":
    main()
