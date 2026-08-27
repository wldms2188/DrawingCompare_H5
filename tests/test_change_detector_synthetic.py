import numpy as np
from core.change_detector import ChangeDetector, Box


def test_box_padding_stays_inside_image():
    b = Box(2, 3, 20, 10).pad(20, 100, 100)
    assert b.x >= 0 and b.y >= 0
    assert b.right <= 100 and b.bottom <= 100


def test_text_normalization_and_classification():
    d = ChangeDetector()
    assert d._norm_text('  Ø 20.0 ') == 'Ø20.0'
    assert d._class('Ø20') in ('GDT', 'DIMENSION')
    assert d._class('R5') == 'DIMENSION'
    assert d._class('MATERIAL: AL6061') == 'NOTE'
    assert d._kind('DIMENSION') == 'dimension_change'
    assert d._kind('NOTE') == 'note_change'
    assert d._kind('GDT') == 'gdt_change'


def test_semantic_context_wins_for_note_and_gdt_values():
    d = ChangeDetector()
    assert d._class_token_context('AL6061', 'MATERIAL: AL6061') == 'NOTE'
    assert d._class_token_context('0.05', 'POSITION 0.05 | A') == 'GDT'


def test_pixel_region_detector_catches_small_local_geometry_change():
    d = ChangeDetector()
    before = np.full((400, 400), 255, np.uint8)
    after = before.copy()
    after[180:205, 180:215] = 0
    regions = d._pixel_regions(before, after)
    assert regions
    assert any(r.x < 200 and r.right > 180 for r in regions)


def test_no_change_returns_no_text_regions():
    d = ChangeDetector()
    before = np.full((300, 300), 255, np.uint8)
    after = before.copy()
    assert d._pixel_regions(before, after) == []
