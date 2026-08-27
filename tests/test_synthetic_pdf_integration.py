from pathlib import Path
from tests.generate_synthetic_pdfs import make
from core.image_loader import ImageLoader
from core.change_detector import ChangeDetector


def test_synthetic_pdf_pipeline_detects_expected_semantic_changes(tmp_path):
    before_pdf = tmp_path / 'synthetic_before.pdf'
    after_pdf = tmp_path / 'synthetic_after.pdf'
    make(before_pdf, False)
    make(after_pdf, True)

    loader = ImageLoader()
    before_doc = loader.load_pdf(before_pdf)
    after_doc = loader.load_pdf(after_pdf)
    assert before_doc.page_count == 1
    assert after_doc.page_count == 1

    detector = ChangeDetector()
    result = detector.detect(before_doc.pages[0], after_doc.pages[0], aligned_after=after_doc.pages[0].image)
    assert result.success, result.reason

    changes = {(r.old_text, r.new_text, r.change_kind) for r in result.regions if r.old_text or r.new_text}
    expected = {
        ('Ø20', 'Ø22', 'dimension_change'),
        ('35', '40', 'dimension_change'),
        ('R5', 'R6', 'dimension_change'),
        ('10±0.2', '12±0.2', 'dimension_change'),
        ('AL6061', 'AL7075', 'note_change'),
        ('0.05', '0.08', 'gdt_change'),
    }
    assert expected.issubset(changes), f'missing semantic changes: {expected - changes}; diag={result.reason}'
    assert not any(r.old_text == 'REMOVE' and r.new_text == 'BURR' for r in result.regions)
    assert not any(r.old_text == 'ALL' and r.new_text == 'DIMENSIONS' for r in result.regions)
