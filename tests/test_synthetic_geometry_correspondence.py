from pathlib import Path

from core.change_detector_v2 import ChangeDetector
from core.image_loader import ImageLoader
from tools.generate_synthetic_drawings import AFTER, BEFORE, main as generate_dataset


def _find_region(result, kind):
    return [r for r in result.regions if r.change_kind == kind]


def test_delta_geometry_change_is_local_and_corresponding(tmp_path):
    generate_dataset()

    before_pdf = BEFORE / "synthetic_before.pdf"
    after_pdf = AFTER / "synthetic_after.pdf"
    loader = ImageLoader()
    before = loader.load_pdf(before_pdf)
    after = loader.load_pdf(after_pdf)

    # DELTA is Before page 4 -> After page 2.
    detector = ChangeDetector()
    result = detector.detect(
        before.pages[3],
        after.pages[1],
        aligned_after=after.pages[1].image,
        alignment_matrix=None,
    )

    assert result.success, result.reason
    geometry = _find_region(result, "geometry_change")
    assert geometry, [(r.change_kind, r.old_text, r.new_text) for r in result.regions]

    # The changed line is around the center of the DELTA drawing area.
    H, W = before.pages[3].image.shape[:2]
    expected_x = (30 + 52) / 210
    expected_y = (100 + 43.5) / 297
    px, py = expected_x * W, expected_y * H

    def contains(r):
        return (
            r.x - 0.04 * W <= px <= r.x + r.width + 0.04 * W
            and r.y - 0.04 * H <= py <= r.y + r.height + 0.04 * H
        )

    local = [r for r in geometry if contains(r)]
    assert local, [(r.x, r.y, r.width, r.height) for r in geometry]

    # A geometry-only change must not become a page-sized or unrelated region.
    region = local[0]
    assert 0 < region.width < W * 0.35
    assert 0 < region.height < H * 0.20
    assert region.old_crop is not None and region.new_crop is not None
    assert region.old_crop.size > 0 and region.new_crop.size > 0

    # The radius label is a separate semantic change and should remain local.
    assert any(r.old_text == "R10" and r.new_text == "R12" for r in result.regions)
