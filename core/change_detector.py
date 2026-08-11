"""
DrawingCompare H5
core/change_detector.py
 
역할
------------------------------------------------------------
Auto Align이 완료된 Before / After 도면을 비교하여
실제 변경 가능성이 있는 영역을 검출한다.
 
기본 원칙
------------------------------------------------------------
1. 정렬 오차를 변경점으로 최대한 오인하지 않는다.
2. 작은 노이즈는 제거한다.
3. 아주 작은 변경은 하나의 큰 변경으로 묶지 않는다.
4. 변경 영역을 Bounding Box로 저장한다.
5. 이후 OCR / 문자 / 치수 분석에서 사용할 수 있도록
   변경 영역 이미지를 만들 수 있게 한다.
 
AI 사용 없음
------------------------------------------------------------
OpenCV 기반으로 동작한다.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import List, Optional, Tuple
 
import cv2
import numpy as np
 
from config import CONFIG
from core.image_loader import PageImage
 
 
# ============================================================
# DATA CLASS
# ============================================================
 
@dataclass
class ChangeRegion:
    """
    하나의 변경 영역.
    """
 
    region_id: int
 
    x: int
    y: int
 
    width: int
    height: int
 
    area: int
 
    change_score: float
 
    before_crop: Optional[np.ndarray]
 
    after_crop: Optional[np.ndarray]
 
    change_type: str = "UNKNOWN"
 
    confidence: float = 0.0
 
    reason: str = ""
 
 
@dataclass
class ChangeDetectionResult:
    """
    한 페이지의 변경점 분석 결과.
    """
 
    success: bool
 
    regions: List[ChangeRegion]
 
    difference_image: Optional[np.ndarray]
 
    threshold_image: Optional[np.ndarray]
 
    change_pixel_ratio: float
 
    reason: str
 
 
# ============================================================
# CHANGE DETECTOR
# ============================================================
 
class ChangeDetector:
 
    def __init__(self, config=None):
 
        self.config = config or CONFIG
 
        # ----------------------------------------------------
        # 최소 변경 영역
        # ----------------------------------------------------
 
        self.minimum_area = getattr(
            self.config.change,
            "minimum_area",
            100
        )
 
        # ----------------------------------------------------
        # 작은 노이즈 제거 크기
        # ----------------------------------------------------
 
        self.morph_kernel_size = getattr(
            self.config.change,
            "morph_kernel_size",
            3
        )
 
        # ----------------------------------------------------
        # 변경점으로 판단하는 픽셀 차이
        # ----------------------------------------------------
 
        self.pixel_threshold = getattr(
            self.config.change,
            "pixel_threshold",
            30
        )
 
        # ----------------------------------------------------
        # 너무 큰 영역은 별도 처리
        # ----------------------------------------------------
 
        self.max_region_ratio = getattr(
            self.config.change,
            "max_region_ratio",
            0.60
        )
 
        # ----------------------------------------------------
        # 변경 영역 간 가까운 거리
        # ----------------------------------------------------
 
        self.merge_distance = getattr(
            self.config.change,
            "merge_distance",
            15
        )
 
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def detect(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> ChangeDetectionResult:
        """
        Before / After 페이지의 변경 영역을 검출한다.
        """
 
        before = before_page.image
        after = after_page.image
 
        if before is None:
 
            return self._failed(
                "Before 이미지가 없음"
            )
 
        if after is None:
 
            return self._failed(
                "After 이미지가 없음"
            )
 
        # ----------------------------------------------------
        # 크기 확인
        # ----------------------------------------------------
 
        if before.shape[:2] != after.shape[:2]:
 
            return self._failed(
                "Before / After 이미지 크기가 다름"
            )
 
        # ----------------------------------------------------
        # 전처리
        # ----------------------------------------------------
 
        before_gray = self._prepare(
            before
        )
 
        after_gray = self._prepare(
            after
        )
 
        # ----------------------------------------------------
        # 차이 이미지
        # ----------------------------------------------------
 
        difference = self._calculate_difference(
            before_gray,
            after_gray
        )
 
        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------
 
        threshold = self._threshold_difference(
            difference
        )
 
        # ----------------------------------------------------
        # 노이즈 제거
        # ----------------------------------------------------
 
        cleaned = self._clean_mask(
            threshold
        )
 
        # ----------------------------------------------------
        # 변경 영역 추출
        # ----------------------------------------------------
 
        regions = self._extract_regions(
            before,
            after,
            difference,
            cleaned
        )
 
        # ----------------------------------------------------
        # 변경 픽셀 비율
        # ----------------------------------------------------
 
        total_pixels = cleaned.size
 
        if total_pixels == 0:
 
            change_pixel_ratio = 0.0
 
        else:
 
            change_pixel_ratio = float(
                np.count_nonzero(cleaned)
                /
                total_pixels
            )
 
        # ----------------------------------------------------
        # 결과
        # ----------------------------------------------------
 
        return ChangeDetectionResult(
            success=True,
            regions=regions,
            difference_image=difference,
            threshold_image=cleaned,
            change_pixel_ratio=change_pixel_ratio,
            reason="변경 영역 검출 완료",
        )
 
 
    # ========================================================
    # PREPARE
    # ========================================================
 
    @staticmethod
    def _prepare(
        image: np.ndarray
    ) -> np.ndarray:
        """
        변경 검출용 grayscale 이미지 생성.
        """
 
        if image is None:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        if image.ndim == 3:
 
            try:
 
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )
 
            except cv2.error:
 
                return np.empty(
                    (0, 0),
                    dtype=np.uint8
                )
 
        else:
 
            gray = image.copy()
 
        # ----------------------------------------------------
        # 약한 노이즈 제거
        # ----------------------------------------------------
 
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )
 
        return gray
 
 
    # ========================================================
    # DIFFERENCE
    # ========================================================
 
    @staticmethod
    def _calculate_difference(
        before: np.ndarray,
        after: np.ndarray
    ) -> np.ndarray:
        """
        Before / After의 절대 차이를 계산한다.
        """
 
        if before.size == 0:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        if after.size == 0:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        difference = cv2.absdiff(
            before,
            after
        )
 
        return difference
 
 
    # ========================================================
    # THRESHOLD
    # ========================================================
 
    def _threshold_difference(
        self,
        difference: np.ndarray
    ) -> np.ndarray:
        """
        단순 픽셀 차이를 그대로 사용하지 않고
        일정 수준 이상의 차이만 변경 후보로 만든다.
        """
 
        if difference.size == 0:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        _, threshold = cv2.threshold(
            difference,
            self.pixel_threshold,
            255,
            cv2.THRESH_BINARY
        )
 
        return threshold
 
 
    # ========================================================
    # CLEAN MASK
    # ========================================================
 
    def _clean_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        작은 노이즈를 제거하고
        실제 변경 영역을 연결한다.
        """
 
        if mask.size == 0:
 
            return mask
 
        kernel_size = max(
            1,
            int(
                self.morph_kernel_size
            )
        )
 
        # 홀수 크기로 보정
        if kernel_size % 2 == 0:
 
            kernel_size += 1
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_size,
                kernel_size
            )
        )
 
        # ----------------------------------------------------
        # 작은 점 제거
        # ----------------------------------------------------
 
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )
 
        # ----------------------------------------------------
        # 가까운 변경 영역 연결
        # ----------------------------------------------------
 
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )
 
        return closed

    # ========================================================
    # EXTRACT REGIONS
    # ========================================================
 
    def _extract_regions(
        self,
        before: np.ndarray,
        after: np.ndarray,
        difference: np.ndarray,
        mask: np.ndarray,
    ) -> List[ChangeRegion]:
        """
        Threshold mask에서 실제 변경 영역을 추출한다.
        """
 
        if mask.size == 0:
 
            return []
 
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        if not contours:
 
            return []
 
        image_height, image_width = (
            mask.shape[:2]
        )
 
        image_area = (
            image_width *
            image_height
        )
 
        regions = []
 
        region_id = 1
 
        for contour in contours:
 
            area = cv2.contourArea(
                contour
            )
 
            # ------------------------------------------------
            # 너무 작은 영역 제거
            # ------------------------------------------------
 
            if area < self.minimum_area:
 
                continue
 
            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )
 
            bbox_area = (
                width *
                height
            )
 
            if bbox_area <= 0:
 
                continue
 
            # ------------------------------------------------
            # 페이지 전체가 변경된 것처럼 보이는 경우
            # ------------------------------------------------
 
            region_ratio = (
                bbox_area /
                max(1, image_area)
            )
 
            if (
                region_ratio
                >
                self.max_region_ratio
            ):
 
                continue
 
            # ------------------------------------------------
            # 변경점 밀도
            # ------------------------------------------------
 
            roi_mask = mask[
                y:y + height,
                x:x + width
            ]
 
            if roi_mask.size == 0:
 
                continue
 
            changed_pixels = (
                np.count_nonzero(
                    roi_mask
                )
            )
 
            density = (
                changed_pixels /
                roi_mask.size
            )
 
            # 변경 영역이라고 하기에는
            # 너무 희박한 영역 제거
            if density < 0.02:
 
                continue
 
            # ------------------------------------------------
            # 실제 Before / After 이미지 잘라내기
            # ------------------------------------------------
 
            before_crop = self._safe_crop(
                before,
                x,
                y,
                width,
                height
            )
 
            after_crop = self._safe_crop(
                after,
                x,
                y,
                width,
                height
            )
 
            # ------------------------------------------------
            # 변경 점수
            # ------------------------------------------------
 
            change_score = (
                self._calculate_region_score(
                    difference,
                    x,
                    y,
                    width,
                    height
                )
            )
 
            confidence = (
                self._calculate_confidence(
                    density,
                    change_score,
                    width,
                    height,
                    image_width,
                    image_height
                )
            )
 
            regions.append(
                ChangeRegion(
                    region_id=region_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    area=int(area),
                    change_score=change_score,
                    before_crop=before_crop,
                    after_crop=after_crop,
                    change_type="UNKNOWN",
                    confidence=confidence,
                    reason="픽셀 차이 기반 변경 후보",
                )
            )
 
            region_id += 1
 
        # ----------------------------------------------------
        # 작은 영역끼리 가까운 경우 통합
        # ----------------------------------------------------
 
        regions = self._merge_nearby_regions(
            regions
        )
 
        # ----------------------------------------------------
        # 큰 변경 영역부터 정렬
        # ----------------------------------------------------
 
        regions.sort(
            key=lambda region: (
                region.area
            ),
            reverse=True
        )
 
        # ----------------------------------------------------
        # ID 재부여
        # ----------------------------------------------------
 
        for index, region in enumerate(
            regions,
            start=1
        ):
 
            region.region_id = index
 
        return regions
 
 
    # ========================================================
    # SAFE CROP
    # ========================================================
 
    @staticmethod
    def _safe_crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        padding: int = 10,
    ) -> Optional[np.ndarray]:
        """
        이미지 범위를 벗어나지 않도록 안전하게 Crop한다.
        """
 
        if image is None:
 
            return None
 
        if image.size == 0:
 
            return None
 
        image_height, image_width = (
            image.shape[:2]
        )
 
        x1 = max(
            0,
            x - padding
        )
 
        y1 = max(
            0,
            y - padding
        )
 
        x2 = min(
            image_width,
            x + width + padding
        )
 
        y2 = min(
            image_height,
            y + height + padding
        )
 
        if x1 >= x2:
 
            return None
 
        if y1 >= y2:
 
            return None
 
        crop = image[
            y1:y2,
            x1:x2
        ]
 
        if crop.size == 0:
 
            return None
 
        return crop.copy()
 
 
    # ========================================================
    # REGION SCORE
    # ========================================================
 
    @staticmethod
    def _calculate_region_score(
        difference: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        """
        해당 영역의 평균 픽셀 차이를
        0~1 점수로 변환한다.
        """
 
        if difference.size == 0:
 
            return 0.0
 
        roi = difference[
            y:y + height,
            x:x + width
        ]
 
        if roi.size == 0:
 
            return 0.0
 
        mean_difference = float(
            np.mean(roi)
        )
 
        score = (
            mean_difference /
            255.0
        )
 
        return float(
            max(
                0.0,
                min(
                    1.0,
                    score
                )
            )
        )
 
 
    # ========================================================
    # CONFIDENCE
    # ========================================================
 
    @staticmethod
    def _calculate_confidence(
        density: float,
        change_score: float,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        변경 영역의 신뢰도를 계산한다.
 
        단순히 픽셀 차이가 많다고
        높은 신뢰도를 주지 않는다.
        """
 
        # ----------------------------------------------------
        # Density 점수
        # ----------------------------------------------------
 
        density_score = min(
            1.0,
            density * 3.0
        )
 
        # ----------------------------------------------------
        # Pixel difference 점수
        # ----------------------------------------------------
 
        difference_score = min(
            1.0,
            change_score * 2.0
        )
 
        # ----------------------------------------------------
        # 너무 긴 얇은 선은
        # 정렬 오차 가능성이 있으므로 감점
        # ----------------------------------------------------
 
        aspect_ratio = (
            max(width, height)
            /
            max(
                1,
                min(width, height)
            )
        )
 
        shape_score = 1.0
 
        if aspect_ratio > 30:
 
            shape_score = 0.50
 
        elif aspect_ratio > 15:
 
            shape_score = 0.70
 
        confidence = (
            density_score * 0.35
            +
            difference_score * 0.45
            +
            shape_score * 0.20
        )
 
        return float(
            max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )
        )
 
 
    # ========================================================
    # MERGE NEARBY REGIONS
    # ========================================================
 
    def _merge_nearby_regions(
        self,
        regions: List[ChangeRegion]
    ) -> List[ChangeRegion]:
        """
        서로 가까운 변경 영역을 하나로 합친다.
 
        예:
            문자 + 치수가 매우 가까이 변경된 경우
            하나의 변경 그룹으로 묶는다.
        """
 
        if len(regions) <= 1:
 
            return regions
 
        merged = []
 
        used = set()
 
        for i, region in enumerate(
            regions
        ):
 
            if i in used:
 
                continue
 
            current = region
 
            used.add(i)
 
            changed = True
 
            while changed:
 
                changed = False
 
                for j, other in enumerate(
                    regions
                ):
 
                    if j in used:
 
                        continue
 
                    if self._regions_are_close(
                        current,
                        other
                    ):
 
                        current = (
                            self._merge_two_regions(
                                current,
                                other
                            )
                        )
 
                        used.add(j)
 
                        changed = True
 
            merged.append(
                current
            )
 
        return merged
 
 
    # ========================================================
    # REGION DISTANCE
    # ========================================================
 
    def _regions_are_close(
        self,
        first: ChangeRegion,
        second: ChangeRegion
    ) -> bool:
        """
        두 Bounding Box가 가까운지 검사한다.
        """
 
        first_left = first.x
        first_right = (
            first.x +
            first.width
        )
 
        first_top = first.y
        first_bottom = (
            first.y +
            first.height
        )
 
        second_left = second.x
        second_right = (
            second.x +
            second.width
        )
 
        second_top = second.y
        second_bottom = (
            second.y +
            second.height
        )
 
        horizontal_gap = max(
            second_left - first_right,
            first_left - second_right,
            0
        )
        
    # ========================================================
    # CLASSIFY REGIONS
    # ========================================================
 
    def classify_regions(
        self,
        result: ChangeDetectionResult
    ) -> ChangeDetectionResult:
        """
        검출된 변경 영역의 특성을 분석한다.
 
        현재 단계에서는 OpenCV 기반의 형상 분석을 사용한다.
        OCR은 다음 단계에서 별도로 연결한다.
        """
 
        if not result.success:
 
            return result
 
        for region in result.regions:
 
            if (
                region.before_crop is None
                or
                region.after_crop is None
            ):
 
                region.change_type = "UNKNOWN"
 
                region.reason = (
                    "변경 영역 Crop 실패"
                )
 
                continue
 
            change_type, reason = (
                self._classify_single_region(
                    region
                )
            )
 
            region.change_type = change_type
 
            region.reason = reason
 
        return result
 
 
    # ========================================================
    # CLASSIFY SINGLE REGION
    # ========================================================
 
    def _classify_single_region(
        self,
        region: ChangeRegion
    ) -> Tuple[str, str]:
        """
        하나의 변경 영역을 분석한다.
        """
 
        before = region.before_crop
        after = region.after_crop
 
        if before is None:
 
            return (
                "UNKNOWN",
                "Before Crop 없음"
            )
 
        if after is None:
 
            return (
                "UNKNOWN",
                "After Crop 없음"
            )
 
        before_gray = self._prepare(
            before
        )
 
        after_gray = self._prepare(
            after
        )
 
        if before_gray.size == 0:
 
            return (
                "UNKNOWN",
                "Before 이미지 분석 실패"
            )
 
        if after_gray.size == 0:
 
            return (
                "UNKNOWN",
                "After 이미지 분석 실패"
            )
 
        # ----------------------------------------------------
        # 특징 계산
        # ----------------------------------------------------
 
        text_score = (
            self._text_likelihood(
                before_gray,
                after_gray
            )
        )
 
        geometry_score = (
            self._geometry_likelihood(
                before_gray,
                after_gray
            )
        )
 
        dimension_score = (
            self._dimension_likelihood(
                before_gray,
                after_gray
            )
        )
 
        # ----------------------------------------------------
        # 가장 높은 유형 선택
        # ----------------------------------------------------
 
        scores = {
            "TEXT": text_score,
            "GEOMETRY": geometry_score,
            "DIMENSION": dimension_score,
        }
 
        best_type = max(
            scores,
            key=scores.get
        )
 
        best_score = scores[
            best_type
        ]
 
        # ----------------------------------------------------
        # 확신이 낮으면 UNKNOWN
        # ----------------------------------------------------
 
        if best_score < 0.35:
 
            return (
                "UNKNOWN",
                "변경 유형을 확실하게 분류하기 어려움"
            )
 
        return (
            best_type,
            (
                f"{best_type} 가능성 "
                f"{best_score:.2f}"
            )
        )
 
 
    # ========================================================
    # TEXT LIKELIHOOD
    # ========================================================
 
    @staticmethod
    def _text_likelihood(
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        변경 영역이 문자/기호일 가능성을 계산한다.
 
        도면의 문자는 일반적으로
        작은 connected component가 여러 개 존재한다.
        """
 
        try:
 
            before_binary = (
                ChangeDetector
                ._binary_image(
                    before
                )
            )
 
            after_binary = (
                ChangeDetector
                ._binary_image(
                    after
                )
            )
 
            before_components = (
                ChangeDetector
                ._component_statistics(
                    before_binary
                )
            )
 
            after_components = (
                ChangeDetector
                ._component_statistics(
                    after_binary
                )
            )
 
            component_count = (
                len(before_components)
                +
                len(after_components)
            )
 
            if component_count == 0:
 
                return 0.0
 
            # ------------------------------------------------
            # 작은 component가 여러 개이면
            # 문자일 가능성을 높인다.
            # ------------------------------------------------
 
            small_components = 0
 
            for component in (
                before_components
                +
                after_components
            ):
 
                width = component[2]
 
                height = component[3]
 
                area = component[4]
 
                if (
                    width <= 100
                    and
                    height <= 100
                    and
                    area <= 3000
                ):
 
                    small_components += 1
 
            small_ratio = (
                small_components /
                component_count
            )
 
            score = min(
                1.0,
                small_ratio * 1.5
            )
 
            return float(score)
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # GEOMETRY LIKELIHOOD
    # ========================================================
 
    @staticmethod
    def _geometry_likelihood(
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        선/형상 변경 가능성을 계산한다.
        """
 
        try:
 
            before_edges = cv2.Canny(
                before,
                50,
                150
            )
 
            after_edges = cv2.Canny(
                after,
                50,
                150
            )
 
            before_count = (
                np.count_nonzero(
                    before_edges
                )
            )
 
            after_count = (
                np.count_nonzero(
                    after_edges
                )
            )
 
            if (
                before_count == 0
                and
                after_count == 0
            ):
 
                return 0.0
 
            total = (
                before_count
                +
                after_count
            )
 
            density = (
                total /
                max(
                    1,
                    before_edges.size * 2
                )
            )
 
            # 선이 충분히 존재하면
            # Geometry 가능성을 높인다.
 
            score = min(
                1.0,
                density * 5.0
            )
 
            return float(score)
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # DIMENSION LIKELIHOOD
    # ========================================================
 
    @staticmethod
    def _dimension_likelihood(
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        치수 변경 가능성을 계산한다.
 
        현재는 치수선/문자 주변의
        길쭉한 구조와 작은 문자 component를 함께 본다.
 
        실제 숫자값 비교는 OCR 단계에서 수행한다.
        """
 
        try:
 
            before_edges = cv2.Canny(
                before,
                50,
                150
            )
 
            after_edges = cv2.Canny(
                after,
                50,
                150
            )
 
            before_lines = (
                ChangeDetector
                ._count_lines(
                    before_edges
                )
            )
 
            after_lines = (
                ChangeDetector
                ._count_lines(
                    after_edges
                )
            )
 
            line_count = (
                before_lines
                +
                after_lines
            )
 
            if line_count == 0:
 
                return 0.0
 
            # 치수 영역은
            # 직선 + 작은 문자 구조가
            # 함께 존재할 가능성이 높다.
 
            text_score = (
                ChangeDetector
                ._text_likelihood(
                    before,
                    after
                )
            )
 
            line_score = min(
                1.0,
                line_count / 20.0
            )
 
            score = (
                line_score * 0.60
                +
                text_score * 0.40
            )
 
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        score
                    )
                )
            )
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # BINARY IMAGE
    # ========================================================
 
    @staticmethod
    def _binary_image(
        gray: np.ndarray
    ) -> np.ndarray:
        """
        문자/선 분석용 이진화.
        """
 
        if gray.size == 0:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        _, binary = cv2.threshold(
            gray,
            180,
            255,
            cv2.THRESH_BINARY_INV
        )
 
        return binary
 
 
    # ========================================================
    # COMPONENT STATISTICS
    # ========================================================
 
    @staticmethod
    def _component_statistics(
        binary: np.ndarray
    ) -> List[Tuple]:
        """
        Connected Component 통계를 반환한다.
        """
 
        if binary.size == 0:
 
            return []
 
        try:
 
            count, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    binary,
                    connectivity=8
                )
            )
 
        except cv2.error:
 
            return []
 
        components = []
 
        for index in range(
            1,
            count
        ):
 
            x = stats[
                index,
                cv2.CC_STAT_LEFT
            ]
 
            y = stats[
                index,
                cv2.CC_STAT_TOP
            ]
 
            width = stats[
                index,
                cv2.CC_STAT_WIDTH
            ]
 
            height = stats[
                index,
                cv2.CC_STAT_HEIGHT
            ]
 
            area = stats[
                index,
                cv2.CC_STAT_AREA
            ]
 
            components.append(
                (
                    x,
                    y,
                    width,
                    height,
                    area
                )
            )
 
        return components
 
 
    # ========================================================
    # COUNT LINES
    # ========================================================
 
    @staticmethod
    def _count_lines(
        edges: np.ndarray
    ) -> int:
        """
        Hough Line을 이용하여
        직선 구조 개수를 대략적으로 계산한다.
        """
 
        if edges.size == 0:
 
            return 0
 
        try:
 
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=15,
                maxLineGap=5
            )
 
        except cv2.error:
 
            return 0
 
        if lines is None:
 
            return 0
 
        return len(lines)
 
 
    # ========================================================
    # RESULT CLASSIFICATION
    # ========================================================
 
    def analyze(
        self,
        before_page: PageImage,
        after_page: PageImage
    ) -> ChangeDetectionResult:
        """
        변경점 검출 + 변경 유형 분류를
        한 번에 실행한다.
        """
 
        result = self.detect(
            before_page,
            after_page
        )
 
        if not result.success:
 
            return result
 
        result = self.classify_regions(
            result
        )
 
        result.reason = (
            "변경 영역 검출 및 "
            "유형 분류 완료"
        )
 
        return result
 
    # ========================================================
    # FINALIZE RESULT
    # ========================================================
 
    def finalize(
        self,
        result: ChangeDetectionResult
    ) -> ChangeDetectionResult:
        """
        변경 검출 결과를 최종 정리한다.
 
        너무 작은 영역이나 신뢰도가 낮은 영역은
        최종 결과에서 제외한다.
        """
 
        if not result.success:
            return result
 
        final_regions = []
 
        for region in result.regions:
 
            if region.area < self.minimum_area:
                continue
 
            if region.confidence < 0.20:
                continue
 
            final_regions.append(region)
 
        # ID 재정렬
        for index, region in enumerate(
            final_regions,
            start=1
        ):
            region.region_id = index
 
        result.regions = final_regions
 
        result.reason = (
            "변경점 최종 정리 완료"
        )
 
        return result
 
 
    # ========================================================
    # ANALYZE FINAL
    # ========================================================
 
    def analyze_final(
        self,
        before_page: PageImage,
        after_page: PageImage
    ) -> ChangeDetectionResult:
        """
        변경점 검출 → 분류 → 최종 정리를
        한 번에 수행한다.
        """
 
        result = self.analyze(
            before_page,
            after_page
        )
 
        if not result.success:
            return result
 
        return self.finalize(
            result
        )
 
 
    # ========================================================
    # RESULT TO DICT
    # ========================================================
 
    @staticmethod
    def result_to_dict(
        result: ChangeDetectionResult
    ) -> dict:
        """
        전체 결과를 JSON / Excel에서 사용할 수 있는
        dictionary 형태로 변환한다.
        """
 
        return {
            "success": result.success,
 
            "change_pixel_ratio": round(
                result.change_pixel_ratio,
                6
            ),
 
            "region_count": len(
                result.regions
            ),
 
            "reason": result.reason,
 
            "regions": [
                ChangeDetector.region_to_dict(
                    region
                )
                for region
                in result.regions
            ],
        }
 
 
    # ========================================================
    # REGION TO DICT
    # ========================================================
 
    @staticmethod
    def region_to_dict(
        region: ChangeRegion
    ) -> dict:
        """
        하나의 변경 영역을
        저장 가능한 dictionary로 변환한다.
        """
 
        return {
            "region_id": (
                region.region_id
            ),
 
            "x": region.x,
 
            "y": region.y,
 
            "width": region.width,
 
            "height": region.height,
 
            "area": region.area,
 
            "change_score": round(
                region.change_score,
                6
            ),
 
            "change_type": (
                region.change_type
            ),
 
            "confidence": round(
                region.confidence,
                6
            ),
 
            "reason": region.reason,
        }
 
 
    # ========================================================
    # DRAW REGIONS
    # ========================================================
 
    @staticmethod
    def draw_regions(
        image: np.ndarray,
        regions: List[ChangeRegion],
        show_labels: bool = True,
    ) -> Optional[np.ndarray]:
        """
        변경 영역을 이미지 위에 표시한다.
 
        원본 이미지를 수정하지 않고
        복사본을 반환한다.
        """
 
        if image is None:
            return None
 
        if image.size == 0:
            return None
 
        output = image.copy()
 
        for region in regions:
 
            x = region.x
            y = region.y
            width = region.width
            height = region.height
 
            # ------------------------------------------------
            # 변경 유형에 따라 표시
            # ------------------------------------------------
 
            if region.change_type == "TEXT":
 
                color = (
                    255,
                    0,
                    0
                )
 
            elif region.change_type == "DIMENSION":
 
                color = (
                    0,
                    165,
                    255
                )
 
            elif region.change_type == "GEOMETRY":
 
                color = (
                    0,
                    0,
                    255
                )
 
            else:
 
                color = (
                    255,
                    0,
                    255
                )
 
            cv2.rectangle(
                output,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                color,
                2
            )
 
            if show_labels:
 
                label = (
                    f"{region.region_id}:"
                    f"{region.change_type}"
                )
 
                cv2.putText(
                    output,
                    label,
                    (
                        x,
                        max(
                            20,
                            y - 5
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA
                )
 
        return output
 
 
    # ========================================================
    # SAVE RESULT IMAGE
    # ========================================================
 
    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path
    ) -> bool:
        """
        결과 이미지를 저장한다.
        """
 
        if image is None:
            return False
 
        if image.size == 0:
            return False
 
        try:
 
            return bool(
                cv2.imwrite(
                    str(output_path),
                    image
                )
            )
 
        except cv2.error:
 
            return False
 
 
    # ========================================================
    # SAVE REGION CROPS
    # ========================================================
 
    @staticmethod
    def save_region_crops(
        result: ChangeDetectionResult,
        output_dir
    ) -> List[str]:
        """
        각 변경 영역의 Before / After Crop을 저장한다.
 
        나중에 Excel 보고서에서
        Before / After 이미지를 삽입할 때 사용한다.
        """
 
        from pathlib import Path
 
        if not result.success:
            return []
 
        output_dir = Path(
            output_dir
        )
 
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )
 
        saved_files = []
 
        for region in result.regions:
 
            # ------------------------------------------------
            # Before
            # ------------------------------------------------
 
            if region.before_crop is not None:
 
                before_path = (
                    output_dir
                    /
                    f"region_{region.region_id:03d}"
                    f"_before.png"
                )
 
                if ChangeDetector.save_image(
                    region.before_crop,
                    before_path
                ):
 
                    saved_files.append(
                        str(before_path)
                    )
 
            # ------------------------------------------------
            # After
            # ------------------------------------------------
 
            if region.after_crop is not None:
 
                after_path = (
                    output_dir
                    /
                    f"region_{region.region_id:03d}"
                    f"_after.png"
                )
 
                if ChangeDetector.save_image(
                    region.after_crop,
                    after_path
                ):
 
                    saved_files.append(
                        str(after_path)
                    )
 
        return saved_files
 
 
    # ========================================================
    # SUMMARY
    # ========================================================
 
    @staticmethod
    def summary(
        result: ChangeDetectionResult
    ) -> dict:
        """
        결과를 간단하게 요약한다.
        """
 
        text_count = 0
        geometry_count = 0
        dimension_count = 0
        unknown_count = 0
 
        for region in result.regions:
 
            if region.change_type == "TEXT":
 
                text_count += 1
 
            elif region.change_type == "GEOMETRY":
 
                geometry_count += 1
 
            elif region.change_type == "DIMENSION":
 
                dimension_count += 1
 
            else:
 
                unknown_count += 1
 
        return {
            "total": len(
                result.regions
            ),
 
            "text": text_count,
 
            "geometry": geometry_count,
 
            "dimension": dimension_count,
 
            "unknown": unknown_count,
 
            "change_pixel_ratio": (
                round(
                    result.change_pixel_ratio,
                    6
                )
            ),
        }
 
 
# ============================================================
# DEFAULT DETECTOR
# ============================================================
 
_default_detector = ChangeDetector()
 
 
def detect_changes(
    before_page: PageImage,
    after_page: PageImage
) -> ChangeDetectionResult:
    """
    외부 모듈에서 변경점을 검출한다.
    """
 
    return _default_detector.analyze_final(
        before_page,
        after_page
    )
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print(
        "DrawingCompare H5 - Change Detector Test"
    )
    print("=" * 60)
 
    print(
        "change_detector.py 로드 성공"
    )
 
    print(
        f"Minimum Area : "
        f"{CONFIG.change.minimum_area}"
    )
 
    print(
        f"Pixel Threshold : "
        f"{CONFIG.change.pixel_threshold}"
    )
 
    print(
        f"Merge Distance : "
        f"{CONFIG.change.merge_distance}"
    )
 
    print("=" * 60)
 
  