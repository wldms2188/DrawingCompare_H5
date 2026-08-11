"""
DrawingCompare H5
core/ocr_detector.py
 
역할
------------------------------------------------------------
변경 영역의 Before / After 이미지를 OCR하여
문자 및 치수 변경 여부를 확인한다.
 
원칙
------------------------------------------------------------
1. AI 사용하지 않음
2. Tesseract OCR을 선택적으로 사용
3. Tesseract가 없어도 전체 프로그램은 계속 동작
4. 변경 영역에서만 OCR 수행
5. OCR 결과는 Before / After를 직접 비교
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import List, Optional
 
import re
 
import cv2
import numpy as np
 
from core.image_loader import PageImage
from core.change_detector import (
    ChangeRegion,
    ChangeDetectionResult,
)
 
 
# ============================================================
# OPTIONAL TESSERACT
# ============================================================
 
try:
 
    import pytesseract
 
    TESSERACT_AVAILABLE = True
 
except ImportError:
 
    pytesseract = None
 
    TESSERACT_AVAILABLE = False
 
 
# ============================================================
# OCR RESULT
# ============================================================
 
@dataclass
class OCRResult:
 
    success: bool
 
    text: str
 
    normalized_text: str
 
    confidence: float
 
    status: str
 
    reason: str
 
 
@dataclass
class TextChange:
 
    region_id: int
 
    before_text: str
 
    after_text: str
 
    change_type: str
 
    confidence: float
 
    reason: str
 
 
# ============================================================
# OCR DETECTOR
# ============================================================
 
class OCRDetector:
 
    def __init__(
        self,
        language: str = "eng",
    ):
 
        self.language = language
 
        self.available = (
            TESSERACT_AVAILABLE
        )
 
 
    # ========================================================
    # STATUS
    # ========================================================
 
    def status(self) -> str:
        """
        OCR 사용 가능 여부를 반환한다.
        """
 
        if self.available:
 
            return "AVAILABLE"
 
        return "UNAVAILABLE"
 
 
    # ========================================================
    # OCR IMAGE
    # ========================================================
 
    def preprocess(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        도면 OCR에 적합하도록 이미지를 전처리한다.
        """
 
        if image is None:
 
            return None
 
        if image.size == 0:
 
            return None
 
        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------
 
        if image.ndim == 3:
 
            try:
 
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )
 
            except cv2.error:
 
                return None
 
        else:
 
            gray = image.copy()
 
        # ----------------------------------------------------
        # 확대
        #
        # 작은 도면 문자를 OCR하기 위해
        # 2배 확대
        # ----------------------------------------------------
 
        height, width = (
            gray.shape[:2]
        )
 
        if height <= 0 or width <= 0:
 
            return None
 
        scale = 2
 
        enlarged = cv2.resize(
            gray,
            (
                width * scale,
                height * scale
            ),
            interpolation=cv2.INTER_CUBIC
        )
 
        # ----------------------------------------------------
        # 약한 노이즈 제거
        # ----------------------------------------------------
 
        blurred = cv2.GaussianBlur(
            enlarged,
            (3, 3),
            0
        )
 
        # ----------------------------------------------------
        # Adaptive Threshold
        # ----------------------------------------------------
 
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
 
        return binary
 
 
    # ========================================================
    # OCR
    # ========================================================
 
    def read(
        self,
        image: np.ndarray
    ) -> OCRResult:
        """
        이미지에서 문자를 읽는다.
        """
 
        if not self.available:
 
            return OCRResult(
                success=False,
                text="",
                normalized_text="",
                confidence=0.0,
                status="UNAVAILABLE",
                reason=(
                    "pytesseract가 설치되어 있지 않음"
                ),
            )
 
        processed = self.preprocess(
            image
        )
 
        if processed is None:
 
            return OCRResult(
                success=False,
                text="",
                normalized_text="",
                confidence=0.0,
                status="FAILED",
                reason=(
                    "OCR 이미지 전처리 실패"
                ),
            )
 
        try:
 
            data = (
                pytesseract.image_to_data(
                    processed,
                    lang=self.language,
                    config=(
                        "--psm 6 "
                        "-c "
                        "preserve_interword_spaces=1"
                    ),
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )
 
        except Exception as exc:
 
            return OCRResult(
                success=False,
                text="",
                normalized_text="",
                confidence=0.0,
                status="FAILED",
                reason=(
                    f"Tesseract 실행 실패: {exc}"
                ),
            )
 
        texts = []
 
        confidences = []
 
        count = len(
            data.get(
                "text",
                []
            )
        )
 
        for index in range(count):
 
            text = str(
                data["text"][index]
            ).strip()
 
            if not text:
 
                continue
 
            try:
 
                confidence = float(
                    data["conf"][index]
                )
 
            except (
                ValueError,
                TypeError
            ):
 
                confidence = 0.0
 
            if confidence < 20:
 
                continue
 
            texts.append(
                text
            )
 
            confidences.append(
                confidence
            )
 
        if not texts:
 
            return OCRResult(
                success=True,
                text="",
                normalized_text="",
                confidence=0.0,
                status="NO_TEXT",
                reason=(
                    "인식된 문자가 없음"
                ),
            )
 
        text = " ".join(
            texts
        )
 
        average_confidence = (
            sum(confidences)
            /
            len(confidences)
        )
 
        normalized = (
            self.normalize_text(
                text
            )
        )
 
        return OCRResult(
            success=True,
            text=text,
            normalized_text=normalized,
            confidence=(
                average_confidence / 100.0
            ),
            status="SUCCESS",
            reason="OCR 완료",
        )
 
 
    # ========================================================
    # NORMALIZE TEXT
    # ========================================================
 
    @staticmethod
    def normalize_text(
        text: str
    ) -> str:
        """
        OCR 결과를 비교하기 쉽게 정리한다.
 
        예:
            '  Ø20.0 '
            ->
            'Ø20.0'
        """
 
        if text is None:
 
            return ""
 
        value = str(text)
 
        # 줄바꿈 제거
        value = value.replace(
            "\n",
            " "
        )
 
        # 여러 공백 제거
        value = re.sub(
            r"\s+",
            " ",
            value
        )
 
        # 양쪽 공백 제거
        value = value.strip()
 
        return value

    # ========================================================
    # COMPARE REGION
    # ========================================================
 
    def compare_region(
        self,
        region: ChangeRegion
    ) -> Optional[TextChange]:
        """
        하나의 변경 영역에 대해
        Before / After OCR 결과를 비교한다.
        """
 
        if region.before_crop is None:
            return None
 
        if region.after_crop is None:
            return None
 
        before_result = self.read(
            region.before_crop
        )
 
        after_result = self.read(
            region.after_crop
        )
 
        # ----------------------------------------------------
        # OCR 자체를 사용할 수 없는 경우
        # ----------------------------------------------------
 
        if (
            before_result.status
            == "UNAVAILABLE"
        ):
 
            return TextChange(
                region_id=region.region_id,
                before_text="",
                after_text="",
                change_type="OCR_UNAVAILABLE",
                confidence=0.0,
                reason=(
                    "Tesseract OCR을 "
                    "사용할 수 없음"
                ),
            )
 
        # ----------------------------------------------------
        # 둘 다 문자가 없는 경우
        # ----------------------------------------------------
 
        if (
            not before_result.normalized_text
            and
            not after_result.normalized_text
        ):
 
            return None
 
        before_text = (
            before_result.normalized_text
        )
 
        after_text = (
            after_result.normalized_text
        )
 
        # ----------------------------------------------------
        # 완전히 동일한 경우
        # ----------------------------------------------------
 
        if before_text == after_text:
 
            return None
 
        # ----------------------------------------------------
        # 한쪽만 존재
        # ----------------------------------------------------
 
        if (
            before_text
            and
            not after_text
        ):
 
            confidence = (
                before_result.confidence
            )
 
            return TextChange(
                region_id=region.region_id,
                before_text=before_text,
                after_text="",
                change_type="TEXT_DELETED",
                confidence=confidence,
                reason=(
                    "Before에는 문자가 있으나 "
                    "After에서는 검출되지 않음"
                ),
            )
 
        if (
            not before_text
            and
            after_text
        ):
 
            confidence = (
                after_result.confidence
            )
 
            return TextChange(
                region_id=region.region_id,
                before_text="",
                after_text=after_text,
                change_type="TEXT_ADDED",
                confidence=confidence,
                reason=(
                    "After에서 새로운 문자가 검출됨"
                ),
            )
 
        # ----------------------------------------------------
        # 둘 다 존재하지만 값이 다름
        # ----------------------------------------------------
 
        confidence = (
            before_result.confidence
            +
            after_result.confidence
        ) / 2.0
 
        # ----------------------------------------------------
        # 치수 변경인지 확인
        # ----------------------------------------------------
 
        if self.is_dimension_change(
            before_text,
            after_text
        ):
 
            return TextChange(
                region_id=region.region_id,
                before_text=before_text,
                after_text=after_text,
                change_type="DIMENSION_CHANGED",
                confidence=confidence,
                reason=(
                    "Before / After의 "
                    "치수 관련 값이 변경됨"
                ),
            )
 
        # ----------------------------------------------------
        # 일반 문자 변경
        # ----------------------------------------------------
 
        return TextChange(
            region_id=region.region_id,
            before_text=before_text,
            after_text=after_text,
            change_type="TEXT_CHANGED",
            confidence=confidence,
            reason=(
                "Before / After 문자 내용이 변경됨"
            ),
        )
 
 
    # ========================================================
    # COMPARE RESULT
    # ========================================================
 
    def compare_result(
        self,
        result: ChangeDetectionResult
    ) -> List[TextChange]:
        """
        ChangeDetector 결과의 모든 변경 영역을
        OCR 비교한다.
        """
 
        changes = []
 
        if not result.success:
 
            return changes
 
        for region in result.regions:
 
            # ------------------------------------------------
            # OCR이 의미 있는 영역에만 수행
            # ------------------------------------------------
 
            if region.change_type not in (
                "TEXT",
                "DIMENSION",
                "UNKNOWN",
            ):
 
                continue
 
            change = (
                self.compare_region(
                    region
                )
            )
 
            if change is None:
 
                continue
 
            changes.append(
                change
            )
 
        return changes
 
 
    # ========================================================
    # DIMENSION DETECTION
    # ========================================================
 
    @staticmethod
    def extract_dimension_values(
        text: str
    ) -> List[str]:
        """
        OCR 문자에서 치수처럼 보이는 값을 추출한다.
 
        예:
            Ø20
            Ø20.5
            10
            10.0
            R5
            R10.5
            M6
            M8x1.25
            25±0.1
        """
 
        if not text:
 
            return []
 
        # ----------------------------------------------------
        # OCR에서 자주 생기는 문자 보정
        # ----------------------------------------------------
 
        value = text.upper()
 
        value = value.replace(
            "O",
            "0"
        )
 
        value = value.replace(
            "Φ",
            "Ø"
        )
 
        # ----------------------------------------------------
        # 치수 패턴
        # ----------------------------------------------------
 
        patterns = [
 
            # Ø20 / Ø20.5
            r"Ø\s*\d+(?:\.\d+)?",
 
            # R5 / R10.5
            r"R\s*\d+(?:\.\d+)?",
 
            # M6 / M8
            r"M\s*\d+(?:\.\d+)?",
 
            # M8X1.25
            r"M\s*\d+(?:\.\d+)?\s*[X×]\s*\d+(?:\.\d+)?",
 
            # 25±0.1
            r"\d+(?:\.\d+)?\s*[±\+\-]\s*\d+(?:\.\d+)?",
 
            # 일반 숫자
            r"\d+(?:\.\d+)?",
        ]
 
        values = []
 
        for pattern in patterns:
 
            matches = re.findall(
                pattern,
                value
            )
 
            for match in matches:
 
                cleaned = (
                    re.sub(
                        r"\s+",
                        "",
                        match
                    )
                )
 
                if cleaned not in values:
 
                    values.append(
                        cleaned
                    )
 
        return values
 
 
    # ========================================================
    # IS DIMENSION CHANGE
    # ========================================================
 
    @staticmethod
    def is_dimension_change(
        before_text: str,
        after_text: str
    ) -> bool:
        """
        Before / After 문자열에
        치수 값이 포함되어 있는지 확인한다.
        """
 
        before_values = (
            OCRDetector
            .extract_dimension_values(
                before_text
            )
        )
 
        after_values = (
            OCRDetector
            .extract_dimension_values(
                after_text
            )
        )
 
        if not before_values:
            return False
 
        if not after_values:
            return False
 
        # ----------------------------------------------------
        # 숫자/치수 값이 실제로 달라졌는지 확인
        # ----------------------------------------------------
 
        if before_values == after_values:
 
            return False
 
        return True
 
 
    # ========================================================
    # APPLY OCR TO RESULT
    # ========================================================
 
    def apply_to_result(
        self,
        result: ChangeDetectionResult
    ) -> List[TextChange]:
        """
        ChangeDetectionResult에 OCR 분석을 적용한다.
        """
 
        changes = (
            self.compare_result(
                result
            )
        )
 
        return changes
  
    # ========================================================
    # FILTER OCR CHANGES
    # ========================================================
 
    def filter_changes(
        self,
        changes: List[TextChange],
        minimum_confidence: float = 0.45,
    ) -> List[TextChange]:
        """
        OCR 신뢰도가 너무 낮은 결과를 제거한다.
 
        OCR 오류 때문에
        존재하지 않는 문자 변경이 발생하는 것을
        최대한 방지한다.
        """
 
        filtered = []
 
        for change in changes:
 
            # ------------------------------------------------
            # OCR 자체를 사용할 수 없는 경우
            # ------------------------------------------------
 
            if (
                change.change_type
                == "OCR_UNAVAILABLE"
            ):
 
                continue
 
            # ------------------------------------------------
            # 신뢰도 검사
            # ------------------------------------------------
 
            if (
                change.confidence
                <
                minimum_confidence
            ):
 
                continue
 
            # ------------------------------------------------
            # 실제 텍스트가 둘 다 존재하는 변경
            # ------------------------------------------------
 
            if (
                change.before_text
                and
                change.after_text
            ):
 
                if (
                    change.before_text
                    ==
                    change.after_text
                ):
 
                    continue
 
            filtered.append(
                change
            )
 
        return filtered
 
 
    # ========================================================
    # ANALYZE
    # ========================================================
 
    def analyze(
        self,
        result: ChangeDetectionResult
    ) -> List[TextChange]:
        """
        변경점 결과에 OCR을 적용하고
        신뢰도가 낮은 결과를 제거한다.
        """
 
        if not result.success:
 
            return []
 
        changes = (
            self.compare_result(
                result
            )
        )
 
        changes = (
            self.filter_changes(
                changes
            )
        )
 
        return changes
 
 
    # ========================================================
    # CHANGE TO DICT
    # ========================================================
 
    @staticmethod
    def change_to_dict(
        change: TextChange
    ) -> dict:
        """
        TextChange를 Excel / JSON용 dict로 변환한다.
        """
 
        return {
            "region_id": (
                change.region_id
            ),
 
            "before_text": (
                change.before_text
            ),
 
            "after_text": (
                change.after_text
            ),
 
            "change_type": (
                change.change_type
            ),
 
            "confidence": round(
                change.confidence,
                4
            ),
 
            "reason": change.reason,
        }
 
 
    # ========================================================
    # CHANGES TO DICT
    # ========================================================
 
    @staticmethod
    def changes_to_dict(
        changes: List[TextChange]
    ) -> List[dict]:
        """
        여러 OCR 변경 결과를
        Excel / JSON에서 사용할 수 있는 형태로 변환한다.
        """
 
        return [
            OCRDetector.change_to_dict(
                change
            )
            for change in changes
        ]
 
 
    # ========================================================
    # SUMMARY
    # ========================================================
 
    @staticmethod
    def summary(
        changes: List[TextChange]
    ) -> dict:
        """
        OCR 변경 결과를 요약한다.
        """
 
        text_changed = 0
        dimension_changed = 0
        text_added = 0
        text_deleted = 0
 
        for change in changes:
 
            if (
                change.change_type
                == "TEXT_CHANGED"
            ):
 
                text_changed += 1
 
            elif (
                change.change_type
                == "DIMENSION_CHANGED"
            ):
 
                dimension_changed += 1
 
            elif (
                change.change_type
                == "TEXT_ADDED"
            ):
 
                text_added += 1
 
            elif (
                change.change_type
                == "TEXT_DELETED"
            ):
 
                text_deleted += 1
 
        return {
            "total": len(changes),
 
            "text_changed": (
                text_changed
            ),
 
            "dimension_changed": (
                dimension_changed
            ),
 
            "text_added": (
                text_added
            ),
 
            "text_deleted": (
                text_deleted
            ),
        }
 
 
    # ========================================================
    # TEST TESSERACT
    # ========================================================
 
    def test_tesseract(self) -> bool:
        """
        Tesseract가 실제로 실행 가능한지 확인한다.
 
        import는 되었지만
        Tesseract 실행 파일이 없는 경우도
        있기 때문에 별도로 확인한다.
        """
 
        if not self.available:
 
            return False
 
        try:
 
            test_image = np.ones(
                (100, 300),
                dtype=np.uint8
            ) * 255
 
            pytesseract.image_to_string(
                test_image,
                lang=self.language,
                config="--psm 7"
            )
 
            return True
 
        except Exception:
 
            return False
 
 
    # ========================================================
    # STATUS DETAIL
    # ========================================================
 
    def status_detail(self) -> dict:
        """
        OCR 환경 상태를 반환한다.
        """
 
        if not self.available:
 
            return {
                "available": False,
                "tesseract": False,
                "status": "UNAVAILABLE",
                "reason": (
                    "pytesseract 미설치"
                ),
            }
 
        executable_available = (
            self.test_tesseract()
        )
 
        if not executable_available:
 
            return {
                "available": False,
                "tesseract": False,
                "status": "UNAVAILABLE",
                "reason": (
                    "pytesseract는 있으나 "
                    "Tesseract 실행이 불가능함"
                ),
            }
 
        return {
            "available": True,
            "tesseract": True,
            "status": "AVAILABLE",
            "reason": (
                "Tesseract OCR 사용 가능"
            ),
        }
 
 
# ============================================================
# DEFAULT OCR DETECTOR
# ============================================================
 
_default_ocr = OCRDetector()
 
 
def analyze_text_changes(
    result: ChangeDetectionResult
) -> List[TextChange]:
    """
    외부 모듈에서 OCR 변경 분석을 실행한다.
    """
 
    return _default_ocr.analyze(
        result
    )
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print(
        "DrawingCompare H5 - OCR Detector Test"
    )
    print("=" * 60)
 
    detector = OCRDetector()
 
    status = (
        detector.status_detail()
    )
 
    print(
        f"OCR Status : "
        f"{status['status']}"
    )
 
    print(
        f"Reason : "
        f"{status['reason']}"
    )
 
    print("=" * 60)
 