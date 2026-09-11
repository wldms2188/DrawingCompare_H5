"""DrawingCompare H5 - independent OCR diagnostic pipeline.

This module is intentionally separate from ChangeDetector.
It answers one question only:
    "Can the local PDF/image actually be read as text?"

It does NOT perform Before/After comparison, alignment, geometry detection,
or change-region detection.

Outputs:
- raw OCR tokens with confidence and bounding boxes
- per-variant text summary
- annotated page images
- Excel report when openpyxl is available
- explicit OCR engine status/errors instead of silently returning zero text
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import os
import re
import shutil

import cv2
import fitz
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - environment dependent
    pytesseract = None
    Output = None

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover - environment dependent
    Workbook = None


@dataclass
class OCRToken:
    variant: str
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    page: int

    @property
    def is_note_candidate(self) -> bool:
        return bool(
            re.search(
                r"\b(?:NOTE|NOTES|MATERIAL|FINISH|BURR|DEBURR|UNLESS|REMARK|COMMENT|INSPECT|SEE)\b",
                self.text,
                re.IGNORECASE,
            )
        )


@dataclass
class OCRVariantResult:
    variant: str
    success: bool
    token_count: int
    high_confidence_count: int
    note_candidate_count: int
    average_confidence: float
    text: str
    reason: str = ""


@dataclass
class OCRDiagnosticResult:
    pdf_path: str
    page: int
    dpi: int
    image_width: int
    image_height: int
    engine: str
    engine_version: str
    engine_path: str
    variants: list[OCRVariantResult]
    tokens: list[OCRToken]


class OCRDiagnostic:
    """Run OCR without touching the H5 change-detection pipeline."""

    DEFAULT_VARIANTS = ("gray", "otsu", "adaptive")

    def __init__(self, dpi: int = 1200, language: str = "eng") -> None:
        self.dpi = int(dpi)
        self.language = language
        self.tesseract_path = self._find_tesseract()
        self.engine_available = pytesseract is not None and self.tesseract_path is not None
        if self.engine_available:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    # ------------------------------------------------------------------
    # Engine discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _find_tesseract() -> Optional[str]:
        if pytesseract is None:
            return None

        env_path = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
        candidates: list[str] = []
        if env_path:
            candidates.append(env_path)

        discovered = shutil.which("tesseract")
        if discovered:
            candidates.append(discovered)

        candidates.extend(
            [
                r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            ]
        )

        for candidate in candidates:
            if Path(candidate).is_file():
                return str(Path(candidate))
        return None

    def engine_status(self) -> dict[str, str]:
        if pytesseract is None:
            return {
                "engine": "Tesseract",
                "available": "NO",
                "reason": "pytesseract가 설치되어 있지 않습니다.",
                "path": "",
                "version": "",
            }
        if self.tesseract_path is None:
            return {
                "engine": "Tesseract",
                "available": "NO",
                "reason": "Tesseract 실행 파일을 찾지 못했습니다.",
                "path": "",
                "version": "",
            }
        try:
            version = str(pytesseract.get_tesseract_version()).strip().splitlines()[0]
        except Exception as exc:  # pragma: no cover - machine dependent
            version = f"version 확인 실패: {exc}"
        return {
            "engine": "Tesseract",
            "available": "YES",
            "reason": "",
            "path": self.tesseract_path,
            "version": version,
        }

    # ------------------------------------------------------------------
    # PDF rendering
    # ------------------------------------------------------------------
    def render_page(self, pdf_path: str | Path, page_number: int = 1) -> np.ndarray:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {path}")
        if self.dpi < 150 or self.dpi > 2400:
            raise ValueError("DPI는 150~2400 범위로 지정하세요.")

        doc = fitz.open(path)
        try:
            if page_number < 1 or page_number > doc.page_count:
                raise IndexError(f"페이지 범위 오류: 1~{doc.page_count}")
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0),
                alpha=False,
                colorspace=fitz.csRGB,
            )
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Preprocessing variants
    # ------------------------------------------------------------------
    @staticmethod
    def make_variants(image: np.ndarray) -> dict[str, np.ndarray]:
        if image is None or image.size == 0:
            raise ValueError("OCR 입력 이미지가 비어 있습니다.")
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Do not resize here. The PDF was rendered at the requested DPI.
        # This keeps the diagnostic honest: DPI and preprocessing are separate variables.
        variants: dict[str, np.ndarray] = {"gray": gray}
        variants["otsu"] = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        variants["adaptive"] = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            11,
        )
        return variants

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _read_variant(self, image: np.ndarray, variant: str, page: int) -> tuple[OCRVariantResult, list[OCRToken]]:
        if not self.engine_available:
            status = self.engine_status()
            return (
                OCRVariantResult(
                    variant=variant,
                    success=False,
                    token_count=0,
                    high_confidence_count=0,
                    note_candidate_count=0,
                    average_confidence=0.0,
                    text="",
                    reason=status["reason"],
                ),
                [],
            )

        config = "--psm 11 -c preserve_interword_spaces=1"
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
                output_type=Output.DICT,
            )
        except Exception as exc:  # pragma: no cover - machine dependent
            return (
                OCRVariantResult(
                    variant=variant,
                    success=False,
                    token_count=0,
                    high_confidence_count=0,
                    note_candidate_count=0,
                    average_confidence=0.0,
                    text="",
                    reason=f"Tesseract 실행 실패: {exc}",
                ),
                [],
            )

        tokens: list[OCRToken] = []
        raw_texts: list[str] = []
        confidences: list[float] = []
        count = len(data.get("text", []))

        for i in range(count):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue

            token = OCRToken(
                variant=variant,
                text=text,
                confidence=conf,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                page=page,
            )
            tokens.append(token)
            raw_texts.append(text)
            confidences.append(conf)

        high = [t for t in tokens if t.confidence >= 60.0]
        notes = [t for t in tokens if t.is_note_candidate and t.confidence >= 30.0]
        avg = float(np.mean(confidences)) if confidences else 0.0
        reason = "문자 검출 없음" if not tokens else "OCR 완료"

        result = OCRVariantResult(
            variant=variant,
            success=True,
            token_count=len(tokens),
            high_confidence_count=len(high),
            note_candidate_count=len(notes),
            average_confidence=avg,
            text=" ".join(raw_texts),
            reason=reason,
        )
        return result, tokens

    def run(self, pdf_path: str | Path, page_number: int = 1) -> OCRDiagnosticResult:
        image = self.render_page(pdf_path, page_number)
        status = self.engine_status()
        variants = self.make_variants(image)

        results: list[OCRVariantResult] = []
        tokens: list[OCRToken] = []
        for name, variant_image in variants.items():
            result, variant_tokens = self._read_variant(variant_image, name, page_number)
            results.append(result)
            tokens.extend(variant_tokens)

        return OCRDiagnosticResult(
            pdf_path=str(Path(pdf_path).resolve()),
            page=page_number,
            dpi=self.dpi,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            engine=status["engine"],
            engine_version=status["version"],
            engine_path=status["path"],
            variants=results,
            tokens=tokens,
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_stem(path: str | Path) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", Path(path).stem)

    def write_annotated_images(
        self,
        pdf_path: str | Path,
        page_number: int,
        output_dir: str | Path,
    ) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        image = self.render_page(pdf_path, page_number)
        variants = self.make_variants(image)
        written: list[Path] = []

        for name, variant_image in variants.items():
            _, tokens = self._read_variant(variant_image, name, page_number)
            canvas = image.copy()
            for token in tokens:
                color = (0, 180, 0) if token.confidence >= 60 else (0, 165, 255)
                p1 = (token.x, token.y)
                p2 = (token.x + token.width, token.y + token.height)
                cv2.rectangle(canvas, p1, p2, color, 2)
                label = f"{token.text} [{token.confidence:.0f}]"
                label_y = max(18, token.y - 4)
                cv2.putText(
                    canvas,
                    label[:80],
                    (token.x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            filename = f"{self._safe_stem(pdf_path)}_p{page_number}_{name}.png"
            target = out / filename
            if not cv2.imwrite(str(target), canvas):
                raise IOError(f"이미지 저장 실패: {target}")
            written.append(target)
        return written

    @staticmethod
    def write_report(result: OCRDiagnosticResult, output_dir: str | Path) -> tuple[Path, Optional[Path]]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = OCRDiagnostic._safe_stem(result.pdf_path)
        txt_path = out / f"{stem}_p{result.page}_ocr_summary.txt"
        xlsx_path: Optional[Path] = None

        lines = [
            "OCR TEST",
            f"PDF: {result.pdf_path}",
            f"Page: {result.page}",
            f"DPI: {result.dpi}",
            f"Image: {result.image_width} x {result.image_height}",
            f"Engine: {result.engine}",
            f"Engine version: {result.engine_version}",
            f"Engine path: {result.engine_path}",
            "",
            "[VARIANT SUMMARY]",
        ]
        for r in result.variants:
            lines.append(
                f"{r.variant}: tokens={r.token_count}, high_conf={r.high_confidence_count}, "
                f"NOTE_candidates={r.note_candidate_count}, avg_conf={r.average_confidence:.1f}, "
                f"status={'OK' if r.success else 'FAIL'}, reason={r.reason}"
            )
        lines.extend(["", "[SAMPLE TOKENS - confidence >= 60]"])
        high_tokens = [t for t in result.tokens if t.confidence >= 60]
        for t in high_tokens[:80]:
            lines.append(
                f"[{t.confidence/100:.2f}] {t.text} | {t.variant} | "
                f"x={t.x}, y={t.y}, w={t.width}, h={t.height}"
            )
        if not high_tokens:
            lines.append("(none)")
        lines.extend(["", "[NOTE CANDIDATES]"])
        note_tokens = [t for t in result.tokens if t.is_note_candidate and t.confidence >= 30]
        for t in note_tokens[:80]:
            lines.append(
                f"[{t.confidence/100:.2f}] {t.text} | {t.variant} | "
                f"x={t.x}, y={t.y}, w={t.width}, h={t.height}"
            )
        if not note_tokens:
            lines.append("(none)")
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        if Workbook is not None:
            xlsx_path = out / f"{stem}_p{result.page}_ocr_result.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Summary"
            ws.append(["PDF", result.pdf_path])
            ws.append(["Page", result.page])
            ws.append(["DPI", result.dpi])
            ws.append(["Image Width", result.image_width])
            ws.append(["Image Height", result.image_height])
            ws.append(["Engine", result.engine])
            ws.append(["Engine Version", result.engine_version])
            ws.append(["Engine Path", result.engine_path])
            ws.append([])
            ws.append(["Variant", "Tokens", "High Confidence", "NOTE Candidates", "Average Confidence", "Status", "Reason"])
            for r in result.variants:
                ws.append([
                    r.variant, r.token_count, r.high_confidence_count,
                    r.note_candidate_count, r.average_confidence,
                    "OK" if r.success else "FAIL", r.reason,
                ])

            wt = wb.create_sheet("Tokens")
            wt.append(["Variant", "Text", "Confidence", "Page", "X", "Y", "Width", "Height", "NOTE Candidate"])
            for t in result.tokens:
                wt.append([
                    t.variant, t.text, t.confidence / 100.0, t.page,
                    t.x, t.y, t.width, t.height, "YES" if t.is_note_candidate else "NO",
                ])
            wb.save(xlsx_path)

        return txt_path, xlsx_path


def find_first_pdf(folder: str | Path) -> Optional[Path]:
    path = Path(folder)
    if not path.is_dir():
        return None
    pdfs = sorted(path.glob("*.pdf"))
    return pdfs[0] if pdfs else None
