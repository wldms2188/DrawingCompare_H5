"""Independent, memory-safe OCR diagnostic for DrawingCompare H5.

This module deliberately does not perform alignment, change detection, or geometry detection.
It answers one question: can the rendered drawing page be read as text by the local OCR engine?

Large engineering PDFs can become hundreds of millions of pixels even at 400 DPI. Therefore the
pipeline processes one preprocessing variant at a time, never stores all variants simultaneously,
and uses a bounded OCR image when the rendered page is too large.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gc
import os
import re
import shutil

import cv2
import fitz
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None
    Output = None

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
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
        return bool(re.search(r"\b(?:NOTE|NOTES|MATERIAL|FINISH|BURR|DEBURR|UNLESS|REMARK|COMMENT|INSPECT|SEE)\b", self.text, re.IGNORECASE))


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
    """Run OCR independently from the H5 comparison pipeline."""

    DEFAULT_VARIANTS = ("gray", "otsu", "adaptive")
    MAX_OCR_PIXELS = 80_000_000

    def __init__(self, dpi: int = 1200, language: str = "eng") -> None:
        self.dpi = int(dpi)
        self.language = language
        self.tesseract_path = self._find_tesseract()
        self.engine_available = pytesseract is not None and self.tesseract_path is not None
        if self.engine_available:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    @staticmethod
    def _find_tesseract() -> Optional[str]:
        if pytesseract is None:
            return None
        candidates: list[str] = []
        env_path = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
        if env_path:
            candidates.append(env_path)
        found = shutil.which("tesseract")
        if found:
            candidates.append(found)
        candidates.extend([r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe", r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"])
        for candidate in candidates:
            if Path(candidate).is_file():
                return str(Path(candidate))
        return None

    def engine_status(self) -> dict[str, str]:
        if pytesseract is None:
            return {"engine": "Tesseract", "available": "NO", "reason": "pytesseract가 설치되어 있지 않습니다.", "path": "", "version": ""}
        if self.tesseract_path is None:
            return {"engine": "Tesseract", "available": "NO", "reason": "Tesseract 실행 파일을 찾지 못했습니다.", "path": "", "version": ""}
        try:
            version = str(pytesseract.get_tesseract_version()).strip().splitlines()[0]
        except Exception as exc:
            version = f"version 확인 실패: {exc}"
        return {"engine": "Tesseract", "available": "YES", "reason": "", "path": self.tesseract_path, "version": version}

    def render_page(self, pdf_path: str | Path, page_number: int = 1) -> np.ndarray:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {path}")
        if not 150 <= self.dpi <= 2400:
            raise ValueError("DPI는 150~2400 범위로 지정하세요.")
        doc = fitz.open(path)
        try:
            if not 1 <= page_number <= doc.page_count:
                raise IndexError(f"페이지 범위 오류: 1~{doc.page_count}")
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0), alpha=False, colorspace=fitz.csRGB)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            del rgb, pix
            return image
        finally:
            doc.close()
            gc.collect()

    @classmethod
    def _ocr_image(cls, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        pixels = h * w
        if pixels <= cls.MAX_OCR_PIXELS:
            return image, 1.0
        scale = (cls.MAX_OCR_PIXELS / float(pixels)) ** 0.5
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        bounded = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        return bounded, scale

    @staticmethod
    def _make_variant(gray: np.ndarray, name: str) -> np.ndarray:
        if name == "gray":
            return gray
        if name == "otsu":
            return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if name == "adaptive":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 11)
        raise ValueError(f"알 수 없는 OCR variant: {name}")

    def _read_variant(self, image: np.ndarray, variant: str, page: int, coordinate_scale: float = 1.0) -> tuple[OCRVariantResult, list[OCRToken]]:
        if not self.engine_available:
            status = self.engine_status()
            return OCRVariantResult(variant, False, 0, 0, 0, 0.0, "", status["reason"]), []
        try:
            data = pytesseract.image_to_data(image, lang=self.language, config="--psm 11 -c preserve_interword_spaces=1", output_type=Output.DICT)
        except Exception as exc:
            return OCRVariantResult(variant, False, 0, 0, 0, 0.0, "", f"Tesseract 실행 실패: {exc}"), []
        tokens: list[OCRToken] = []
        raw: list[str] = []
        confidences: list[float] = []
        inv = 1.0 / coordinate_scale if coordinate_scale > 0 else 1.0
        for i in range(len(data.get("text", []))):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            token = OCRToken(variant, text, conf, int(round(int(data["left"][i]) * inv)), int(round(int(data["top"][i]) * inv)), int(round(int(data["width"][i]) * inv)), int(round(int(data["height"][i]) * inv)), page)
            tokens.append(token)
            raw.append(text)
            confidences.append(conf)
        high = sum(t.confidence >= 60.0 for t in tokens)
        notes = sum(t.is_note_candidate and t.confidence >= 30.0 for t in tokens)
        avg = float(np.mean(confidences)) if confidences else 0.0
        return OCRVariantResult(variant, True, len(tokens), high, notes, avg, " ".join(raw), "문자 검출 없음" if not tokens else "OCR 완료"), tokens

    def run(self, pdf_path: str | Path, page_number: int = 1) -> OCRDiagnosticResult:
        image = self.render_page(pdf_path, page_number)
        original_h, original_w = image.shape[:2]
        ocr_image, scale = self._ocr_image(image)
        gray = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY) if ocr_image.ndim == 3 else ocr_image
        results: list[OCRVariantResult] = []
        tokens: list[OCRToken] = []
        for name in self.DEFAULT_VARIANTS:
            variant_image = self._make_variant(gray, name)
            result, variant_tokens = self._read_variant(variant_image, name, page_number, scale)
            results.append(result)
            tokens.extend(variant_tokens)
            if name != "gray":
                del variant_image
            gc.collect()
        del gray, ocr_image, image
        gc.collect()
        status = self.engine_status()
        return OCRDiagnosticResult(str(Path(pdf_path).resolve()), page_number, self.dpi, int(original_w), int(original_h), status["engine"], status["version"], status["path"], results, tokens)

    @staticmethod
    def _safe_stem(path: str | Path) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", Path(path).stem)

    def write_annotated_images(self, pdf_path: str | Path, page_number: int, output_dir: str | Path) -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        image = self.render_page(pdf_path, page_number)
        ocr_image, scale = self._ocr_image(image)
        gray = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY) if ocr_image.ndim == 3 else ocr_image
        written: list[Path] = []
        for name in self.DEFAULT_VARIANTS:
            variant_image = self._make_variant(gray, name)
            _, tokens = self._read_variant(variant_image, name, page_number, scale)
            canvas = image.copy()
            for token in tokens:
                color = (0, 180, 0) if token.confidence >= 60 else (0, 165, 255)
                cv2.rectangle(canvas, (token.x, token.y), (token.x + token.width, token.y + token.height), color, 2)
                cv2.putText(canvas, f"{token.text} [{token.confidence:.0f}]"[:80], (token.x, max(18, token.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            target = out / f"{self._safe_stem(pdf_path)}_p{page_number}_{name}.png"
            if not cv2.imwrite(str(target), canvas):
                raise IOError(f"이미지 저장 실패: {target}")
            written.append(target)
            del variant_image, canvas, tokens
            gc.collect()
        del gray, ocr_image, image
        gc.collect()
        return written

    @staticmethod
    def write_report(result: OCRDiagnosticResult, output_dir: str | Path) -> tuple[Path, Optional[Path]]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = OCRDiagnostic._safe_stem(result.pdf_path)
        txt_path = out / f"{stem}_p{result.page}_ocr_summary.txt"
        lines = ["OCR TEST", f"PDF: {result.pdf_path}", f"Page: {result.page}", f"DPI: {result.dpi}", f"Image: {result.image_width} x {result.image_height}", f"Engine: {result.engine}", f"Engine version: {result.engine_version}", f"Engine path: {result.engine_path}", "", "[VARIANT SUMMARY]"]
        for r in result.variants:
            lines.append(f"{r.variant}: tokens={r.token_count}, high_conf={r.high_confidence_count}, NOTE_candidates={r.note_candidate_count}, avg_conf={r.average_confidence:.1f}, status={'OK' if r.success else 'FAIL'}, reason={r.reason}")
        lines += ["", "[SAMPLE TOKENS - confidence >= 60]"]
        high = [t for t in result.tokens if t.confidence >= 60]
        lines += [f"[{t.confidence/100:.2f}] {t.text} | {t.variant} | x={t.x}, y={t.y}, w={t.width}, h={t.height}" for t in high[:80]] or ["(none)"]
        lines += ["", "[NOTE CANDIDATES]"]
        notes = [t for t in result.tokens if t.is_note_candidate and t.confidence >= 30]
        lines += [f"[{t.confidence/100:.2f}] {t.text} | {t.variant} | x={t.x}, y={t.y}, w={t.width}, h={t.height}" for t in notes[:80]] or ["(none)"]
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        xlsx_path: Optional[Path] = None
        if Workbook is not None:
            xlsx_path = out / f"{stem}_p{result.page}_ocr_result.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Summary"
            for key, value in [("PDF", result.pdf_path), ("Page", result.page), ("DPI", result.dpi), ("Image Width", result.image_width), ("Image Height", result.image_height), ("Engine", result.engine), ("Engine Version", result.engine_version), ("Engine Path", result.engine_path)]:
                ws.append([key, value])
            ws.append([])
            ws.append(["Variant", "Tokens", "High Confidence", "NOTE Candidates", "Average Confidence", "Status", "Reason"])
            for r in result.variants:
                ws.append([r.variant, r.token_count, r.high_confidence_count, r.note_candidate_count, r.average_confidence, "OK" if r.success else "FAIL", r.reason])
            wt = wb.create_sheet("Tokens")
            wt.append(["Variant", "Text", "Confidence", "X", "Y", "Width", "Height", "Page", "NOTE Candidate"])
            for t in result.tokens:
                wt.append([t.variant, t.text, t.confidence, t.x, t.y, t.width, t.height, t.page, "YES" if t.is_note_candidate else "NO"])
            wb.save(xlsx_path)
        return txt_path, xlsx_path


def find_first_pdf(folder: str | Path) -> Optional[Path]:
    folder = Path(folder)
    if not folder.exists():
        return None
    pdfs = sorted(folder.glob("*.pdf"))
    return pdfs[0] if pdfs else None
