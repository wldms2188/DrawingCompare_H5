from pathlib import Path
import tempfile
import fitz
from h6.engine import H6Engine


def make_pdf(path, page_specs):
    doc = fitz.open()
    for spec in page_specs:
        p = doc.new_page(width=800, height=600)
        # Drawing frame and stable geometry
        p.draw_rect(fitz.Rect(40, 40, 760, 560), color=(0, 0, 0), width=1)
        p.draw_rect(fitz.Rect(120, 120, 360, 320), color=(0, 0, 0), width=1.5)
        p.draw_circle((520, 260), 80, color=(0, 0, 0), width=1.5)
        p.draw_line((120, 400), (700, 400), color=(0, 0, 0), width=1)
        p.insert_text((70, 85), spec['title'], fontsize=16, color=(0, 0, 0))
        p.insert_text((480, 520), spec['value'], fontsize=14, color=(0, 0, 0))
        if spec.get('extra_line'):
            p.draw_line((180, 180), (330, 280), color=(0, 0, 0), width=3)
    doc.save(path)
    doc.close()


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        before = root / 'before.pdf'
        after = root / 'after.pdf'
        make_pdf(before, [
            {'title': 'DRAWING-A', 'value': 'DIM 10.00'},
            {'title': 'DRAWING-B', 'value': 'DIM 20.00'},
        ])
        # Reverse page order and change both a text value and geometry.
        make_pdf(after, [
            {'title': 'DRAWING-B', 'value': 'DIM 25.00'},
            {'title': 'DRAWING-A', 'value': 'DIM 10.00', 'extra_line': True},
        ])

        e = H6Engine()
        bp = e.load_pdf(before)
        ap = e.load_pdf(after)
        matches = e.match_pages(bp, ap)
        assert len(matches) == 2, f'page matching failed: {len(matches)}'
        mapping = {(m.before.index, m.after.index) for m in matches}
        assert mapping == {(0, 1), (1, 0)}, f'wrong page mapping: {mapping}'

        all_changes = []
        for m in matches:
            all_changes.extend(e.compare_match(m))
        assert all_changes, 'no changes detected'
        kinds = {c.kind for c in all_changes}
        assert 'GEOMETRY' in kinds, f'geometry change not detected: {kinds}'
        print(f'SYNTHETIC TEST PASS: {len(matches)} page pairs, {len(all_changes)} changes, kinds={sorted(kinds)}')


if __name__ == '__main__':
    main()
