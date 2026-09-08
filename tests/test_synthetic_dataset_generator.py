from pathlib import Path

from core.change_detector_v2 import ChangeDetector
from core.image_loader import ImageLoader
from core.page_matcher import PageMatcher
from tools.generate_synthetic_drawings import BEFORE, AFTER, make_pdf
from reportlab.lib.pagesizes import A4, LETTER


def test_generated_dataset_pipeline(tmp_path):
    before_path = tmp_path / "before.pdf"
    after_path = tmp_path / "after.pdf"
    make_pdf(before_path, ["ALPHA", "BETA", "GAMMA", "DELTA"], set(), [A4] * 4)
    make_pdf(after_path, ["BETA", "DELTA", "GAMMA", "ALPHA"], {"ALPHA", "BETA", "DELTA"}, [LETTER, A4, LETTER, A4])

    loader = ImageLoader()
    before = loader.load_pdf(before_path)
    after = loader.load_pdf(after_path)
    matches = PageMatcher().match_pages(before, after)
    mapping = {m.before_page.page_index: m.after_page.page_index for m in matches}

    assert mapping == {0: 3, 1: 0, 2: 2, 3: 1}, mapping
    assert all(m.status == "MATCH" for m in matches)

    detector = ChangeDetector()
    observed = []
    for m in matches:
        result = detector.detect(m.before_page, m.after_page, aligned_after=m.after_page.image, alignment_matrix=None)
        assert result.success, result.reason
        observed.extend(result.regions)

    values = {(r.old_text.strip(), r.new_text.strip()) for r in observed if r.old_text or r.new_text}
    assert ("25", "30") in values
    assert ("Ø8", "Ø10") in values or ("⌀8", "⌀10") in values
    assert ("0.05", "0.10") in values
    assert ("AL6061", "AL7075") in values
    assert ("R10", "R12") in values
    assert any(r.change_kind == "text_deleted" and r.old_text.strip() == "REMOVE_ME" for r in observed)
    assert any(r.change_kind == "text_added" and r.new_text.strip() == "ADDED" for r in observed)
    assert not any("COMMON TITLE BLOCK" in (r.old_text + r.new_text).upper() for r in observed)
    assert not any("UNCHANGED" in (r.old_text + r.new_text).upper() for r in observed)
