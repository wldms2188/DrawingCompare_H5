import fitz
from reportlab.lib.pagesizes import A4, LETTER

from core.change_detector_v2 import ChangeDetector
from core.image_loader import ImageLoader
from core.page_matcher import PageMatcher
from tools.generate_synthetic_drawings import make_pdf


def _word_center(pdf_path, page_index, text):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pw, ph = page.rect.width, page.rect.height
    target = text.strip().upper()
    hits = []
    for z in page.get_text("words"):
        if len(z) < 5:
            continue
        value = str(z[4]).strip()
        if value.upper() == target:
            x0, y0, x1, y1 = z[:4]
            hits.append(((x0 + x1) / 2 / pw, (y0 + y1) / 2 / ph))
    doc.close()
    assert hits, f"text not found: {text!r} page={page_index}"
    return hits[0]


def _contains_norm_point(region, point, reference_shape, margin=0.035):
    H, W = reference_shape[:2]
    x, y = point[0] * W, point[1] * H
    return (
        region.x - margin * W <= x <= region.x + region.width + margin * W
        and region.y - margin * H <= y <= region.y + region.height + margin * H
    )


def test_synthetic_crops_are_corresponding_semantic_locations(tmp_path):
    before_pdf = tmp_path / "before.pdf"
    after_pdf = tmp_path / "after.pdf"
    make_pdf(before_pdf, ["ALPHA", "BETA", "GAMMA", "DELTA"], set(), [A4] * 4)
    make_pdf(after_pdf, ["BETA", "DELTA", "GAMMA", "ALPHA"], {"ALPHA", "BETA", "DELTA"}, [LETTER, A4, LETTER, A4])

    loader = ImageLoader()
    before = loader.load_pdf(before_pdf)
    after = loader.load_pdf(after_pdf)
    matches = PageMatcher().match_pages(before, after)
    mapping = {m.before_page.page_index: m.after_page.page_index for m in matches}
    assert mapping == {0: 3, 1: 0, 2: 2, 3: 1}

    detector = ChangeDetector()
    cases = [
        (0, 3, "25", "30"),
        (0, 3, "Ø8", "Ø10"),
        (1, 0, "0.05", "0.10"),
        (1, 0, "AL6061", "AL7075"),
        (3, 1, "R10", "R12"),
    ]

    for before_idx, after_idx, old_text, new_text in cases:
        result = detector.detect(
            before.pages[before_idx],
            after.pages[after_idx],
            aligned_after=after.pages[after_idx].image,
            alignment_matrix=None,
        )
        assert result.success, result.reason
        regions = [
            r for r in result.regions
            if r.old_text.strip() == old_text and r.new_text.strip() == new_text
        ]
        assert regions, f"missing exact semantic region {old_text!r}->{new_text!r}: {result.reason}"
        region = regions[0]

        old_point = _word_center(before_pdf, before_idx, old_text)
        new_point = _word_center(after_pdf, after_idx, new_text)
        # The detector expresses both sides in BEFORE_RASTER coordinates when
        # no explicit alignment matrix is supplied. Compare normalized PDF
        # locations against the BEFORE raster, never raw after-pixel coords.
        assert _contains_norm_point(region, old_point, before.pages[before_idx].image.shape), old_text
        assert _contains_norm_point(region, new_point, before.pages[before_idx].image.shape), new_text

        H, W = before.pages[before_idx].image.shape[:2]
        assert 0 < region.width < W * 0.35
        assert 0 < region.height < H * 0.35
        assert region.old_crop is not None and region.new_crop is not None
        assert region.old_crop.size > 0 and region.new_crop.size > 0


def test_synthetic_note_and_add_delete_crops_stay_local(tmp_path):
    before_pdf = tmp_path / "before.pdf"
    after_pdf = tmp_path / "after.pdf"
    make_pdf(before_pdf, ["ALPHA", "BETA", "GAMMA", "DELTA"], set(), [A4] * 4)
    make_pdf(after_pdf, ["BETA", "DELTA", "GAMMA", "ALPHA"], {"ALPHA", "BETA", "DELTA"}, [LETTER, A4, LETTER, A4])

    loader = ImageLoader()
    before = loader.load_pdf(before_pdf)
    after = loader.load_pdf(after_pdf)
    detector = ChangeDetector()

    alpha = detector.detect(before.pages[0], after.pages[3], aligned_after=after.pages[3].image, alignment_matrix=None)
    beta = detector.detect(before.pages[1], after.pages[0], aligned_after=after.pages[0].image, alignment_matrix=None)

    assert alpha.success and beta.success
    deleted = [r for r in alpha.regions if r.change_kind == "text_deleted" and r.old_text.strip() == "REMOVE_ME"]
    added = [r for r in alpha.regions if r.change_kind == "text_added" and r.new_text.strip() == "ADDED"]
    assert deleted and added

    note = [r for r in beta.regions if r.old_text.strip() == "AL6061" and r.new_text.strip() == "AL7075"]
    assert note

    H, W = before.pages[1].image.shape[:2]
    for r in deleted + added + note:
        assert r.width < W * 0.35 and r.height < H * 0.35
        assert r.x >= 0 and r.y >= 0
        assert r.x + r.width <= W and r.y + r.height <= H
        assert r.old_crop is not None and r.new_crop is not None
        assert r.old_crop.size and r.new_crop.size
