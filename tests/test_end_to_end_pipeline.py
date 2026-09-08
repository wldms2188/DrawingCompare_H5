from __future__ import annotations

from pathlib import Path

import numpy as np

from core.auto_align import AutoAlign, AlignmentResult
from core.change_detector_v2 import ChangeDetector
from core.image_loader import ImageLoader
from core.page_matcher import PageMatcher
from tools.generate_synthetic_drawings import AFTER, BEFORE, main as generate_dataset


def _match_documents(before_docs, after_docs):
    remaining = list(after_docs)
    pairs = []
    for bd in before_docs:
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda ad: (
                0
                if Path(bd.filename).stem.lower() == Path(ad.filename).stem.lower()
                else 10
            )
            + abs(bd.page_count - ad.page_count),
        )
        pairs.append((bd, best))
        remaining.remove(best)
    return pairs


def test_end_to_end_matching_alignment_detection_and_crops():
    """Exercise the same core flow used by the GUI without requiring confidential PDFs."""
    generate_dataset()
    loader = ImageLoader()
    before_docs = loader.load_folder(BEFORE)
    after_docs = loader.load_folder(AFTER)
    assert len(before_docs) == 1 and len(after_docs) == 1

    pairs = _match_documents(before_docs, after_docs)
    assert len(pairs) == 1

    bd, ad = pairs[0]
    page_matches = PageMatcher().match_pages(bd, ad)
    matches = [m for m in page_matches if m.status != "NO_MATCH"]
    assert len(matches) == 4

    expected_after_index = {0: 3, 1: 0, 2: 2, 3: 1}
    aligner = AutoAlign(max_rotation_deg=2.0)
    detector = ChangeDetector()
    all_regions = []
    alignment_results = []

    for pm in matches:
        bp, ap = pm.before_page, pm.after_page
        alignment = aligner.align(bp.image, ap.image)
        assert isinstance(alignment, AlignmentResult)
        assert alignment.image is not None and alignment.image.size > 0
        alignment_results.append(alignment)

        result = detector.detect(
            bp,
            ap,
            aligned_after=alignment.image,
            alignment_matrix=alignment.matrix,
        )
        assert result.success, result.reason
        all_regions.extend((bp.page_index, ap.page_index, r) for r in result.regions)

    assert {bp: ap for bp, ap, _ in all_regions} == expected_after_index

    pairs_text = {(r.old_text, r.new_text) for _, _, r in all_regions}
    for expected in {
        ("25", "30"),
        ("Ø8", "Ø10"),
        ("0.05", "0.10"),
        ("AL6061", "AL7075"),
        ("R10", "R12"),
    }:
        assert expected in pairs_text, expected

    assert not any("COMMON TITLE BLOCK" in (r.old_text + r.new_text) for _, _, r in all_regions)
    assert not any("UNCHANGED" in (r.old_text + r.new_text) for _, _, r in all_regions)

    # Every reported semantic change must have a real local crop. Geometry changes
    # are also expected to carry both sides of the corresponding drawing fragment.
    for bp_index, _, region in all_regions:
        assert region.old_crop is not None and region.new_crop is not None
        assert region.old_crop.size > 0 and region.new_crop.size > 0
        assert region.width > 0 and region.height > 0
        before_img = before_docs[0].pages[bp_index].image
        H, W = before_img.shape[:2]
        assert 0 <= region.x < W and 0 <= region.y < H
        assert region.x + region.width <= W
        assert region.y + region.height <= H
        assert region.width < W * 0.60
        assert region.height < H * 0.60

    # The geometry-only DELTA edit must remain a local structural region.
    delta_geometry = [
        r
        for bp, _, r in all_regions
        if bp == 3 and r.change_kind == "geometry_change"
    ]
    assert delta_geometry
    assert all(r.width < before_docs[0].pages[3].image.shape[1] * 0.35 for r in delta_geometry)
    assert all(r.height < before_docs[0].pages[3].image.shape[0] * 0.20 for r in delta_geometry)

    # At least one page must have produced a real alignment transform; if all pages
    # are rejected, the detector is only being exercised in fallback-free NONE mode.
    assert any(a.success and a.matrix is not None for a in alignment_results)
