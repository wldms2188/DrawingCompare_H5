from __future__ import annotations
 
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol
 
import cv2
import numpy as np
 
 
# ============================================================
# OCR Result
# ============================================================
 
@dataclass
class OCRItem:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
 
 
# ============================================================
# Text Region
# ============================================================
 
@dataclass
class TextRegion:
    text: str
    x: int
    y: int
    width: int
    height: int
 
    confidence: float = 0.0
    region_type: str = "UNKNOWN"
 
    page_index: int = 0
    source: str = "OCR"
 
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0
 
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0
 
    @property
    def area(self) -> int:
        return self.width * self.height
 
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "region_type": self.region_type,
            "page_index": self.page_index,
            "source": self.source,
        }
 
 
# ============================================================
# OCR Engine Interface
# ============================================================
 
class OCREngine(Protocol):
 
    def extract(
        self,
        image: np.ndarray,
    ) -> List[OCRItem]:
        ...
 
 
# ============================================================
# Tesseract OCR Engine
# ============================================================
 
class TesseractOCREngine:
 
    def __init__(
        self,
        psm: int = 11,
    ):
        self.psm = psm
 
        self.available = False
        self.error = ""
 
        try:
 
            import pytesseract
 
            self.pytesseract = pytesseract
 
            # 실제 Tesseract 실행파일 확인
            self.pytesseract.get_tesseract_version()
 
            self.available = True
 
        except Exception as exc:
 
            self.pytesseract = None
 
            self.error = str(exc)
 
    def extract(
        self,
        image: np.ndarray,
    ) -> List[OCRItem]:
 
        if not self.available:
 
            return []
 
        if image is None:
 
            return []
 
        if image.size == 0:
 
            return []
 
        try:
 
            data = self.pytesseract.image_to_data(
                image,
                output_type=(
                    self.pytesseract.Output.DICT
                ),
                config=f"--psm {self.psm}",
            )
 
        except Exception as exc:
 
            self.error = str(exc)
 
            return []
 
        results = []
 
        texts = data.get(
            "text",
            [],
        )
 
        for i, raw_text in enumerate(texts):
 
            text = str(
                raw_text
            ).strip()
 
            if not text:
 
                continue
 
            try:
 
                confidence = float(
                    data["conf"][i]
                )
 
            except Exception:
 
                confidence = 0.0
 
            try:
 
                x = int(
                    data["left"][i]
                )
 
                y = int(
                    data["top"][i]
                )
 
                width = int(
                    data["width"][i]
                )
 
                height = int(
                    data["height"][i]
                )
 
            except Exception:
 
                continue
 
            if width <= 0:
 
                continue
 
            if height <= 0:
 
                continue
 
            results.append(
                OCRItem(
                    text=text,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=(
                        confidence / 100.0
                    ),
                )
            )
 
        return results
 
 
# ============================================================
# Text Extractor
# ============================================================
 
class TextExtractor:
 
    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        min_confidence: float = 0.20,
        min_text_length: int = 1,
    ):
 
        self.min_confidence = (
            min_confidence
        )
 
        self.min_text_length = (
            min_text_length
        )
 
        if ocr_engine is not None:
 
            self.ocr_engine = ocr_engine
 
        else:
 
            self.ocr_engine = (
                TesseractOCREngine()
            )
 
    # ========================================================
    # Main
    # ========================================================
 
    def extract(
        self,
        image: np.ndarray,
        page_index: int = 0,
    ) -> List[TextRegion]:
 
        if image is None:
 
            return []
 
        if image.size == 0:
 
            return []
 
        gray = self._to_gray(
            image
        )
 
        processed = self._preprocess(
            gray
        )
 
        ocr_results = (
            self.ocr_engine.extract(
                processed
            )
        )
 
        results = []
 
        for item in ocr_results:
 
            text = self._normalize_text(
                item.text
            )
 
            if not text:
 
                continue
 
            if len(text) < (
                self.min_text_length
            ):
 
                continue
 
            if item.confidence < (
                self.min_confidence
            ):
 
                continue
 
            region_type = (
                self.classify_text(
                    text
                )
            )
 
            results.append(
                TextRegion(
                    text=text,
                    x=item.x,
                    y=item.y,
                    width=item.width,
                    height=item.height,
                    confidence=item.confidence,
                    region_type=region_type,
                    page_index=page_index,
                    source="OCR",
                )
            )
 
        return self._merge_nearby(
            results
        )
 
    # ========================================================
    # Gray conversion
    # ========================================================
 
    def _to_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
 
        if len(image.shape) == 2:
 
            return image
 
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
 
    # ========================================================
    # Preprocess
    # ========================================================
 
    def _preprocess(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
 
        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )
 
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
            # ========================================================
    # Text classification
    # ========================================================
 
    DIMENSION_PATTERNS = [
        r"(?i)Ø\s*\d+(?:\.\d+)?",
        r"(?i)R\s*\d+(?:\.\d+)?",
        r"(?i)\d+(?:\.\d+)?\s*[±]\s*\d+(?:\.\d+)?",
        r"(?i)\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?",
        r"(?i)M\d+(?:\.\d+)?",
    ]
 
    GDT_KEYWORDS = [
        "POSITION",
        "FLATNESS",
        "PARALLEL",
        "PERPENDICULAR",
        "ANGULAR",
        "PROFILE",
        "CONCENTRIC",
        "SYMMETRY",
        "CIRCULAR",
        "RUNOUT",
        "DATUM",
        "직각도",
        "평행도",
        "평면도",
        "위치도",
        "동심도",
        "대칭도",
        "프로파일",
        "데이텀",
    ]
 
    COMMENT_KEYWORDS = [
        "NOTE",
        "NOTES",
        "REMARK",
        "REMARKS",
        "COMMENT",
        "비고",
        "주석",
        "주의",
        "참고",
        "검사",
    ]
 
    TITLE_BLOCK_KEYWORDS = [
        "TITLE",
        "DRAWING",
        "DRAWING NO",
        "DWG NO",
        "REV",
        "REVISION",
        "DATE",
        "MATERIAL",
        "SCALE",
        "DRAWN",
        "CHECKED",
        "APPROVED",
        "도면번호",
        "도면명",
        "개정",
        "재질",
        "축척",
        "작성",
        "검토",
        "승인",
    ]
 
    def classify_text(
        self,
        text: str,
    ) -> str:
 
        upper = text.upper()
 
        for keyword in self.GDT_KEYWORDS:
            if keyword.upper() in upper:
                return "GDT"
 
        for keyword in self.COMMENT_KEYWORDS:
            if keyword.upper() in upper:
                return "COMMENT"
 
        for keyword in self.TITLE_BLOCK_KEYWORDS:
            if keyword.upper() in upper:
                return "TITLE_BLOCK"
 
        for pattern in self.DIMENSION_PATTERNS:
            if re.search(pattern, text):
                return "DIMENSION"
 
        if re.search(
            r"(?i)\bITEM\s*[NO.]?\s*\d+",
            text,
        ):
            return "ITEM"
 
        if re.search(
            r"(?i)\bPART\s*[NO.]?\s*[:\-]?\s*[A-Z0-9]",
            text,
        ):
            return "ITEM"
 
        if re.search(
            r"(?i)\bP\/N\s*[:\-]?\s*[A-Z0-9]",
            text,
        ):
            return "ITEM"
 
        return "TEXT"
 
    # ========================================================
    # Normalize OCR text
    # ========================================================
 
    def _normalize_text(
        self,
        text: str,
    ) -> str:
 
        text = str(text).strip()
 
        text = text.replace(
            "\n",
            " ",
        )
 
        text = re.sub(
            r"\s+",
            " ",
            text,
        )
 
        return text.strip()
 
    # ========================================================
    # Merge nearby text
    # ========================================================
 
    def _merge_nearby(
        self,
        regions: List[TextRegion],
    ) -> List[TextRegion]:
 
        if len(regions) < 2:
            return regions
 
        regions = sorted(
            regions,
            key=lambda item: (
                item.y,
                item.x,
            ),
        )
 
        merged = []
 
        for current in regions:
 
            merged_flag = False
 
            for previous in merged:
 
                vertical_distance = abs(
                    current.center_y
                    -
                    previous.center_y
                )
 
                horizontal_gap = (
                    current.x
                    -
                    (
                        previous.x
                        +
                        previous.width
                    )
                )
 
                same_line = (
                    vertical_distance
                    <=
                    max(
                        current.height,
                        previous.height,
                    ) * 0.6
                )
 
                close = (
                    -20
                    <=
                    horizontal_gap
                    <=
                    30
                )
 
                if not (
                    same_line
                    and close
                ):
                    continue
 
                right = max(
                    previous.x
                    + previous.width,
                    current.x
                    + current.width,
                )
 
                bottom = max(
                    previous.y
                    + previous.height,
                    current.y
                    + current.height,
                )
 
                previous.text = (
                    previous.text
                    + " "
                    + current.text
                )
 
                previous.x = min(
                    previous.x,
                    current.x,
                )
 
                previous.y = min(
                    previous.y,
                    current.y,
                )
 
                previous.width = (
                    right
                    - previous.x
                )
 
                previous.height = (
                    bottom
                    - previous.y
                )
 
                previous.confidence = (
                    previous.confidence
                    +
                    current.confidence
                ) / 2.0
 
                previous.region_type = (
                    self._merge_type(
                        previous.region_type,
                        current.region_type,
                    )
                )
 
                merged_flag = True
 
                break
 
            if not merged_flag:
                merged.append(current)
 
        return merged
 
    # ========================================================
    # Merge classification
    # ========================================================
 
    def _merge_type(
        self,
        first: str,
        second: str,
    ) -> str:
 
        priority = {
            "TITLE_BLOCK": 5,
            "GDT": 4,
            "DIMENSION": 3,
            "ITEM": 2,
            "COMMENT": 2,
            "TEXT": 1,
            "UNKNOWN": 0,
        }
 
        if priority.get(second, 0) > priority.get(first, 0):
            return second
 
        return first
 
    # ========================================================
    # OCR engine status
    # ========================================================
 
    def status(self) -> dict:
 
        engine = self.ocr_engine
 
        return {
            "available": bool(
                getattr(
                    engine,
                    "available",
                    False,
                )
            ),
            "engine": engine.__class__.__name__,
            "error": getattr(
                engine,
                "error",
                "",
            ),
        }
 
 
# ============================================================
# Convenience Function
# ============================================================
 
def extract_text_regions(
    image: np.ndarray,
    page_index: int = 0,
) -> List[TextRegion]:
 
    extractor = TextExtractor()
 
    return extractor.extract(
        image,
        page_index,
    )
 