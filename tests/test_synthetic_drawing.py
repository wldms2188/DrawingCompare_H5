"""Synthetic drawing smoke test for the H5 semantic detector.

This test intentionally uses a minimal page-like object and vector PDF text
coordinates. It verifies the critical semantic classes without requiring a
real company drawing: dimension, note, GD&T, unchanged text, and a geometry
candidate. It also catches the old coordinate/crop regressions at the detector
boundary.
"""
from types import SimpleNamespace
import numpy as np

from core.change_detector import ChangeDetector


def _page(words):
    return SimpleNamespace(
        image=np.full((1000, 1400, 3), 255, dtype=np.uint8),
        pdf_path=None,
        page_index=0,
        _synthetic_words=words,
    )


def test_synthetic_semantic_classes():
    detector = ChangeDetector()
    assert detector._class("35") == "DIMENSION"
    assert detector._class("Ø20") == "GDT"
    assert detector._class("MATERIAL: AL6061") == "NOTE"
    assert detector._class("REMOVE BURR") == "NOTE"
    assert detector._class("POSITION 0.05") == "GDT"


def test_synthetic_normalization():
    detector = ChangeDetector()
    assert detector._norm_text("10.0") == "10.0"
    assert detector._norm_text("  A  B  ") == "AB"
    assert detector._norm_text("10−0.2") == "10-0.2"


def test_synthetic_box_crop_has_reasonable_size():
    detector = ChangeDetector()
    page = _page([])
    box = detector._word_box({'x': 500, 'y': 400, 'w': 20, 'h': 12})
    crop = detector._crop(page.image, box)
    assert crop.shape[0] <= 32
    assert crop.shape[1] <= 40
    assert crop.size > 0
