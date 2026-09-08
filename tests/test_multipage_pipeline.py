from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm

from core.image_loader import ImageLoader
from core.page_matcher import PageMatcher
from core.change_detector_v2 import ChangeDetector

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tests" / "_multipage_runtime"


def _page(c, page_id, changed=False, page_size=A4):
    pw, ph = page_size
    # Keep the drawing layout proportional to the page so that page-size changes
    # do not change its semantic coordinates.
    sx, sy = pw / A4[0], ph / A4[1]
    def X(v): return v * sx
    def Y(v): return v * sy

    x, y, w, h = X(35 * mm), Y(105 * mm), X(120 * mm), Y(85 * mm)
    c.rect(x, y, w, h)
    c.rect(x + X(25 * mm), y + Y(25 * mm), X(55 * mm), Y(30 * mm))
    c.line(x + X(80 * mm), y + Y(40 * mm), x + X(105 * mm), y + Y(40 * mm))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(X(25 * mm), Y(255 * mm), f"DRAWING PAGE {page_id}")

    c.setFont("Helvetica", 9)
    if page_id == "ALPHA":
        c.drawCentredString(x + X(52.5 * mm), y + Y(12 * mm), "30" if changed else "25")
        c.drawString(x + X(88 * mm), y + Y(67 * mm), "Ø10" if changed else "Ø8")
        if not changed:
            c.drawString(x + X(65 * mm), y + Y(8 * mm), "REMOVE_ME")
        else:
            c.drawString(x + X(65 * mm), y + Y(8 * mm), "ADDED")

    if page_id == "BETA":
        gx, gy = x + X(18 * mm), y + Y(68 * mm)
        c.rect(gx, gy, X(38 * mm), Y(9 * mm))
        c.line(gx + X(12 * mm), gy, gx + X(12 * mm), gy + Y(9 * mm))
        c.line(gx + X(25 * mm), gy, gx + X(25 * mm), gy + Y(9 * mm))
        c.setFont("Helvetica", 7)
        c.drawCentredString(gx + X(6 * mm), gy + Y(2.8 * mm), "POS")
        c.drawCentredString(gx + X(18.5 * mm), gy + Y(2.8 * mm), "0.10" if changed else "0.05")
        c.drawCentredString(gx + X(31.5 * mm), gy + Y(2.8 * mm), "A")

        nx, ny, nw, nh = X(170 * mm), Y(125 * mm), X(28 * mm), Y(55 * mm)
        c.rect(nx, ny, nw, nh)
        ty = ny + nh - Y(8 * mm)
        lines = ["NOTE", "1. DEBURR", "2. MATERIAL:", "   AL7075" if changed else "   AL6061", "3. FINISH: NONE"]
        if changed:
            lines.append("4. UPDATED")
        for line in lines:
            c.drawString(nx + X(2 * mm), ty, line)
            ty -= Y(7 * mm)

    if page_id == "GAMMA":
        c.setFont("Helvetica", 9)
        c.drawString(X(55 * mm), Y(155 * mm), "GAMMA BASE")
        c.rect(X(50 * mm), Y(135 * mm), X(70 * mm), Y(35 * mm))
        c.line(X(55 * mm), Y(140 * mm), X(115 * mm), Y(165 * mm))
        c.drawString(X(55 * mm), Y(120 * mm), "UNCHANGED")

    c.setFont("Helvetica", 8)
    c.drawString(X(20 * mm), Y(20 * mm), "COMMON TITLE BLOCK")


def make_pdf(path, order, changed_ids, sizes):
    c = canvas.Canvas(str(path), pagesize=sizes[0])
    for idx, page_id in enumerate(order):
        if idx > 0:
            c.setPageSize(sizes[idx])
        _page(c, page_id, page_id in changed_ids, sizes[idx])
        c.showPage()
    c.save()


def test_multipage_shuffled_scaled_pipeline():
    TMP.mkdir(parents=True, exist_ok=True)
    before_path = TMP / "multipage_before.pdf"
    after_path = TMP / "multipage_after.pdf"

    # Before order: ALPHA, BETA, GAMMA.
    make_pdf(before_path, ["ALPHA", "BETA", "GAMMA"], set(), [A4, A4, A4])
    # After order is deliberately shuffled. Each page also uses a different
    # physical size while preserving the drawing's normalized layout.
    make_pdf(after_path, ["BETA", "GAMMA", "ALPHA"], {"ALPHA", "BETA"}, [LETTER, A4, LETTER])

    loader = ImageLoader()
    before = loader.load_pdf(before_path)
    after = loader.load_pdf(after_path)
    assert before.page_count == 3
    assert after.page_count == 3

    matches = PageMatcher().match_pages(before, after)
    assert len(matches) == 3, [(m.before_page.page_index, m.after_page.page_index, m.score, m.status) for m in matches]

    mapping = {m.before_page.page_index: m.after_page.page_index for m in matches}
    assert mapping == {0: 2, 1: 0, 2: 1}, mapping
    assert all(m.status == "MATCH" for m in matches), [(m.score, m.status, m.reason) for m in matches]

    detector = ChangeDetector()
    observed = []
    for match in matches:
        result = detector.detect(
            match.before_page,
            match.after_page,
            aligned_after=match.after_page.image,
            alignment_matrix=None,
        )
        assert result.success, result.reason
        observed.extend(result.regions)

    values = {(r.old_text.strip(), r.new_text.strip()) for r in observed if r.old_text or r.new_text}
    normalized = {(a.replace("⌀", "Ø"), b.replace("⌀", "Ø")) for a, b in values}

    assert ("25", "30") in values, values
    assert ("Ø8", "Ø10") in values or ("Ø8", "Ø10") in normalized, values
    assert ("0.05", "0.10") in values, values
    assert ("AL6061", "AL7075") in values, values
    assert any(r.change_kind == "text_deleted" and r.old_text.strip() == "REMOVE_ME" for r in observed), values
    assert any(r.change_kind == "text_added" and r.new_text.strip() == "ADDED" for r in observed), values
    assert not any("UNCHANGED" in (r.old_text + r.new_text).upper() for r in observed)
    assert not any("COMMON TITLE BLOCK" in (r.old_text + r.new_text).upper() for r in observed)
