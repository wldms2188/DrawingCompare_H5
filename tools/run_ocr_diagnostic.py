"""Command-line runner for the independent H5 OCR diagnostic.

Examples (Windows PowerShell):
    py tools\run_ocr_diagnostic.py
    py tools\run_ocr_diagnostic.py --pdf "C:\\path\\drawing.pdf"
    py tools\run_ocr_diagnostic.py --pdf "C:\\path\\drawing.pdf" --page 2 --dpi 400
    py tools\run_ocr_diagnostic.py --pdf "C:\\path\\drawing.pdf" --out "output\\ocr_diagnostic"

If --pdf is omitted, the first PDF in input/before is used.

Memory note:
    1200 DPI can create extremely large raster images for engineering drawings.
    The diagnostic therefore uses 400 DPI by default. Higher DPI can still be
    requested explicitly after the basic OCR path is confirmed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as: py tools\run_ocr_diagnostic.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ocr_diagnostic import OCRDiagnostic, find_first_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DrawingCompare H5 independent OCR diagnostic")
    parser.add_argument(
        "--pdf",
        type=str,
        default="",
        help="검사할 PDF 경로. 생략하면 input/before의 첫 PDF를 사용합니다.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="검사할 페이지 번호(1부터 시작). 기본값: 1",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PDF 렌더링 DPI. 150~2400. 기본값: 400",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "output" / "ocr_diagnostic"),
        help="결과 저장 폴더",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    pdf_path = Path(args.pdf) if args.pdf else find_first_pdf(ROOT / "input" / "before")
    if pdf_path is None:
        print("OCR TEST")
        print("ERROR: 검사할 PDF를 찾지 못했습니다.")
        print(r"예: py tools\run_ocr_diagnostic.py --pdf \"C:\path\drawing.pdf\"")
        return 2

    output_dir = Path(args.out)
    diagnostic = OCRDiagnostic(dpi=args.dpi, language="eng")

    print("=" * 72)
    print("OCR TEST")
    print("=" * 72)
    print(f"PDF: {pdf_path.resolve()}")
    print(f"Page: {args.page}")
    print(f"DPI: {args.dpi}")
    print("Note: 기본 400 DPI로 메모리 사용량을 제한합니다.")

    status = diagnostic.engine_status()
    print(f"Engine: {status['engine']}")
    print(f"Engine available: {status['available']}")
    print(f"Engine path: {status['path']}")
    print(f"Engine version: {status['version']}")
    if status["reason"]:
        print(f"Engine reason: {status['reason']}")

    try:
        result = diagnostic.run(pdf_path, page_number=args.page)
        image_paths = diagnostic.write_annotated_images(
            pdf_path,
            page_number=args.page,
            output_dir=output_dir,
        )
        txt_path, xlsx_path = diagnostic.write_report(result, output_dir)
    except MemoryError:
        print("ERROR: OCR 진단 실행 실패: MemoryError")
        print("원인: PDF 페이지를 OCR용 고해상도 이미지로 펼치는 과정에서 메모리가 부족했습니다.")
        print("해결: 기본값은 400 DPI입니다. 그래도 실패하면 --dpi 300으로 실행하세요.")
        return 1
    except Exception as exc:
        print(f"ERROR: OCR 진단 실행 실패: {type(exc).__name__}: {exc}")
        return 1

    print("")
    print(f"Image size: {result.image_width} x {result.image_height}")
    print("")
    print("[VARIANT SUMMARY]")
    for item in result.variants:
        print(
            f"{item.variant}: tokens={item.token_count}, "
            f"high_conf={item.high_confidence_count}, "
            f"NOTE_candidates={item.note_candidate_count}, "
            f"avg_conf={item.average_confidence:.1f}, "
            f"status={'OK' if item.success else 'FAIL'}"
        )
        if item.reason:
            print(f"  reason: {item.reason}")

    print("")
    print("[HIGH-CONFIDENCE SAMPLE]")
    high_tokens = [t for t in result.tokens if t.confidence >= 60.0]
    if high_tokens:
        for token in high_tokens[:30]:
            print(
                f"[{token.confidence / 100:.2f}] {token.text} | "
                f"{token.variant} | x={token.x}, y={token.y}, "
                f"w={token.width}, h={token.height}"
            )
    else:
        print("(none)")

    print("")
    print("[NOTE CANDIDATES]")
    note_tokens = [t for t in result.tokens if t.is_note_candidate and t.confidence >= 30.0]
    if note_tokens:
        for token in note_tokens[:30]:
            print(
                f"[{token.confidence / 100:.2f}] {token.text} | "
                f"{token.variant} | x={token.x}, y={token.y}, "
                f"w={token.width}, h={token.height}"
            )
    else:
        print("(none)")

    print("")
    print("[FILES]")
    print(f"Summary TXT: {txt_path.resolve()}")
    if xlsx_path:
        print(f"Excel: {xlsx_path.resolve()}")
    for path in image_paths:
        print(f"Annotated: {path.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
