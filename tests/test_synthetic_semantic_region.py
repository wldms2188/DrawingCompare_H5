from core.change_detector_v2 import ChangeDetector
from core.image_loader import ImageLoader
from tools.generate_synthetic_drawings import AFTER, BEFORE, main as generate_dataset


def test_note_change_uses_note_block_and_ignores_title_block():
    generate_dataset()
    loader = ImageLoader()
    before = loader.load_pdf(BEFORE / 'synthetic_before.pdf')
    after = loader.load_pdf(AFTER / 'synthetic_after.pdf')

    # BETA is Before page 2 -> After page 1.
    result = ChangeDetector().detect(
        before.pages[1],
        after.pages[0],
        aligned_after=after.pages[0].image,
        alignment_matrix=None,
    )
    assert result.success, result.reason

    note = [
        r for r in result.regions
        if r.old_text == 'AL6061' and r.new_text == 'AL7075'
    ]
    assert note, [(r.change_kind, r.old_text, r.new_text) for r in result.regions]
    region = note[0]
    H, W = before.pages[1].image.shape[:2]

    # The NOTE change must remain on the right-side NOTE block, not become a page-wide region.
    assert region.x > W * 0.65
    assert region.y > H * 0.25
    assert region.x + region.width < W * 0.99
    assert region.height < H * 0.30

    # COMMON TITLE BLOCK must never be reported as a change.
    assert not any('COMMON TITLE BLOCK' in (r.old_text + r.new_text) for r in result.regions)
    assert not any(r.change_kind == 'geometry_change' and r.y > H * 0.88 for r in result.regions)
