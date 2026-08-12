from __future__ import annotations
 
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import cv2
import numpy as np
 
 
# ============================================================
# Data Classes
# ============================================================
 
@dataclass
class RegionCandidate:
    x: int
    y: int
    width: int
    height: int
 
    area: int
    page_area_ratio: float
 
    content_ratio: float
    border_score: float
    density_score: float
    position_score: float
 
    total_score: float
 
    region_type: str = "drawing"
    confidence: float = 0.0
    review_required: bool = False
 
    @property
    def left(self) -> int:
        return self.x
 
    @property
    def top(self) -> int:
        return self.y
 
    @property
    def right(self) -> int:
        return self.x + self.width
 
    @property
    def bottom(self) -> int:
        return self.y + self.height
 
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
 
 
@dataclass
class DrawingRegionResult:
    success: bool
 
    page_width: int
    page_height: int
 
    candidates: List[RegionCandidate]
 
    selected: Optional[RegionCandidate]
 
    confidence: float
 
    status: str
 
    reason: str
 
    def to_dict(self) -> Dict[str, Any]:
 
        return {
            "success": self.success,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "selected": (
                self.selected.to_dict()
                if self.selected is not None
                else None
            ),
            "confidence": self.confidence,
            "status": self.status,
            "reason": self.reason,
        }
 
 
# ============================================================
# Drawing Region Detector
# ============================================================
 
class DrawingRegionDetector:
 
    def __init__(
        self,
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.98,
        border_margin_ratio: float = 0.02,
        min_content_ratio: float = 0.002,
        review_threshold: float = 0.55,
        accept_threshold: float = 0.70,
    ):
 
        self.min_area_ratio = (
            float(min_area_ratio)
        )
 
        self.max_area_ratio = (
            float(max_area_ratio)
        )
 
        self.border_margin_ratio = (
            float(border_margin_ratio)
        )
 
        self.min_content_ratio = (
            float(min_content_ratio)
        )
 
        self.review_threshold = (
            float(review_threshold)
        )
 
        self.accept_threshold = (
            float(accept_threshold)
        )
 
    # ========================================================
    # Public API
    # ========================================================
 
    def detect(
        self,
        image: np.ndarray
    ) -> DrawingRegionResult:
 
        if image is None:
 
            return DrawingRegionResult(
                success=False,
                page_width=0,
                page_height=0,
                candidates=[],
                selected=None,
                confidence=0.0,
                status="ERROR",
                reason="image is None",
            )
 
        if not isinstance(
            image,
            np.ndarray
        ):
 
            return DrawingRegionResult(
                success=False,
                page_width=0,
                page_height=0,
                candidates=[],
                selected=None,
                confidence=0.0,
                status="ERROR",
                reason="image must be numpy.ndarray",
            )
 
        if image.size == 0:
 
            return DrawingRegionResult(
                success=False,
                page_width=0,
                page_height=0,
                candidates=[],
                selected=None,
                confidence=0.0,
                status="ERROR",
                reason="image is empty",
            )
 
        height, width = image.shape[:2]
 
        try:
 
            gray = self._to_gray(
                image
            )
 
            binary = self._make_binary(
                gray
            )
 
            binary = self._remove_page_border_noise(
                binary
            )
 
            candidates = (
                self._generate_candidates(
                    binary,
                    width,
                    height
                )
            )
 
            candidates = (
                self._score_candidates(
                    candidates,
                    binary,
                    width,
                    height
                )
            )
 
            candidates.sort(
                key=lambda item: item.total_score,
                reverse=True
            )
 
            selected = self._select_best(
                candidates
            )
 
            if selected is None:
 
                return DrawingRegionResult(
                    success=True,
                    page_width=width,
                    page_height=height,
                    candidates=candidates,
                    selected=None,
                    confidence=0.0,
                    status="REVIEW",
                    reason=(
                        "no reliable drawing region found"
                    ),
                )
 
            confidence = (
                selected.confidence
            )
 
            if confidence >= self.accept_threshold:
 
                status = "ACCEPT"
 
            elif confidence >= self.review_threshold:
 
                status = "REVIEW"
 
            else:
 
                status = "REVIEW"
 
            selected.review_required = (
                status != "ACCEPT"
            )
 
            return DrawingRegionResult(
                success=True,
                page_width=width,
                page_height=height,
                candidates=candidates,
                selected=selected,
                confidence=confidence,
                status=status,
                reason=(
                    "drawing region detected"
                ),
            )
 
        except Exception as exc:
 
            return DrawingRegionResult(
                success=False,
                page_width=width,
                page_height=height,
                candidates=[],
                selected=None,
                confidence=0.0,
                status="ERROR",
                reason=str(exc),
            )
 
    # ========================================================
    # Image Conversion
    # ========================================================
 
    def _to_gray(
        self,
        image: np.ndarray
    ) -> np.ndarray:
 
        if len(image.shape) == 2:
 
            return image.copy()
 
        if image.shape[2] == 4:
 
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY
            )
 
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
 
    # ========================================================
    # Binary Image
    # ========================================================
 
    def _make_binary(
        self,
        gray: np.ndarray
    ) -> np.ndarray:
 
        # 도면은 대부분 밝은 배경 위의
        # 어두운 선/문자로 구성되므로
        # adaptive threshold를 사용한다.
 
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
 
        return binary
 
    # ========================================================
    # Remove Page Border Noise
    # ========================================================
 
    def _remove_page_border_noise(
        self,
        binary: np.ndarray
    ) -> np.ndarray:
 
        result = binary.copy()
 
        height, width = result.shape[:2]
 
        margin_x = max(
            1,
            int(
                width *
                self.border_margin_ratio
            )
        )
 
        margin_y = max(
            1,
            int(
                height *
                self.border_margin_ratio
            )
        )
 
        # 페이지 가장자리의 아주 얇은
        # 스캔/렌더링 노이즈 제거
 
        result[
            :margin_y,
            :
        ] = 0
 
        result[
            height - margin_y:,
            :
        ] = 0
 
        result[
            :,
            :margin_x
        ] = 0
 
        result[
            :,
            width - margin_x:
        ] = 0
 
        return result
    # ========================================================
    # Candidate Generation
    # ========================================================
 
    def _generate_candidates(
        self,
        binary: np.ndarray,
        page_width: int,
        page_height: int
    ) -> List[RegionCandidate]:
 
        page_area = (
            page_width *
            page_height
        )
 
        candidates = []
 
        # ----------------------------------------------------
        # Strategy 1
        # Connected components
        # ----------------------------------------------------
 
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )
 
        for index in range(
            1,
            num_labels
        ):
 
            x = int(
                stats[index,
                       cv2.CC_STAT_LEFT]
            )
 
            y = int(
                stats[index,
                       cv2.CC_STAT_TOP]
            )
 
            width = int(
                stats[index,
                       cv2.CC_STAT_WIDTH]
            )
 
            height = int(
                stats[index,
                       cv2.CC_STAT_HEIGHT]
            )
 
            area = int(
                stats[index,
                       cv2.CC_STAT_AREA]
            )
 
            if width <= 0 or height <= 0:
                continue
 
            bbox_area = (
                width *
                height
            )
 
            area_ratio = (
                bbox_area /
                max(1, page_area)
            )
 
            if (
                area_ratio <
                self.min_area_ratio
            ):
                continue
 
            if (
                area_ratio >
                self.max_area_ratio
            ):
                continue
 
            content_ratio = (
                area /
                max(1, bbox_area)
            )
 
            if (
                content_ratio <
                self.min_content_ratio
            ):
                continue
 
            candidate = RegionCandidate(
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                page_area_ratio=area_ratio,
                content_ratio=content_ratio,
                border_score=0.0,
                density_score=0.0,
                position_score=0.0,
                total_score=0.0,
            )
 
            candidates.append(
                candidate
            )
 
        # ----------------------------------------------------
        # Strategy 2
        # Morphological drawing body
        # ----------------------------------------------------
 
        kernel_width = max(
            5,
            page_width // 120
        )
 
        kernel_height = max(
            5,
            page_height // 120
        )
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_width,
                kernel_height
            )
        )
 
        connected = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )
 
        connected = cv2.dilate(
            connected,
            kernel,
            iterations=1
        )
 
        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        for contour in contours:
 
            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )
 
            if width <= 0 or height <= 0:
                continue
 
            bbox_area = (
                width *
                height
            )
 
            area_ratio = (
                bbox_area /
                max(1, page_area)
            )
 
            if (
                area_ratio <
                self.min_area_ratio
            ):
                continue
 
            if (
                area_ratio >
                self.max_area_ratio
            ):
                continue
 
            mask = np.zeros_like(
                binary
            )
 
            cv2.rectangle(
                mask,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                255,
                -1
            )
 
            content_pixels = (
                np.count_nonzero(
                    binary[
                        y:y + height,
                        x:x + width
                    ]
                )
            )
 
            content_ratio = (
                content_pixels /
                max(1, bbox_area)
            )
 
            if (
                content_ratio <
                self.min_content_ratio
            ):
                continue
 
            candidate = RegionCandidate(
                x=x,
                y=y,
                width=width,
                height=height,
                area=int(
                    cv2.contourArea(
                        contour
                    )
                ),
                page_area_ratio=area_ratio,
                content_ratio=content_ratio,
                border_score=0.0,
                density_score=0.0,
                position_score=0.0,
                total_score=0.0,
            )
 
            candidates.append(
                candidate
            )
 
        return self._deduplicate_candidates(
            candidates
        )
 
    # ========================================================
    # Candidate Deduplication
    # ========================================================
 
    def _deduplicate_candidates(
        self,
        candidates: List[RegionCandidate]
    ) -> List[RegionCandidate]:
 
        if len(candidates) <= 1:
            return candidates
 
        result = []
 
        candidates = sorted(
            candidates,
            key=lambda item: (
                item.width *
                item.height
            ),
            reverse=True
        )
 
        for candidate in candidates:
 
            duplicate = False
 
            for existing in result:
 
                iou = self._intersection_over_union(
                    candidate,
                    existing
                )
 
                if iou >= 0.85:
 
                    duplicate = True
                    break
 
            if not duplicate:
 
                result.append(
                    candidate
                )
 
        return result
 
    # ========================================================
    # IoU
    # ========================================================
 
    def _intersection_over_union(
        self,
        a: RegionCandidate,
        b: RegionCandidate
    ) -> float:
 
        left = max(
            a.left,
            b.left
        )
 
        top = max(
            a.top,
            b.top
        )
 
        right = min(
            a.right,
            b.right
        )
 
        bottom = min(
            a.bottom,
            b.bottom
        )
 
        if (
            right <= left
            or bottom <= top
        ):
            return 0.0
 
        intersection = (
            right - left
        ) * (
            bottom - top
        )
 
        area_a = (
            a.width *
            a.height
        )
 
        area_b = (
            b.width *
            b.height
        )
 
        union = (
            area_a +
            area_b -
            intersection
        )
 
        if union <= 0:
            return 0.0
 
        return (
            intersection /
            union
        )
 
    # ========================================================
    # Candidate Scoring
    # ========================================================
 
    def _score_candidates(
        self,
        candidates: List[RegionCandidate],
        binary: np.ndarray,
        page_width: int,
        page_height: int
    ) -> List[RegionCandidate]:
 
        for candidate in candidates:
 
            candidate.border_score = (
                self._calculate_border_score(
                    candidate,
                    binary
                )
            )
 
            candidate.density_score = (
                self._calculate_density_score(
                    candidate,
                    binary
                )
            )
 
            candidate.position_score = (
                self._calculate_position_score(
                    candidate,
                    page_width,
                    page_height
                )
            )
 
            area_score = min(
                1.0,
                candidate.page_area_ratio /
                0.70
            )
 
            # 도면 영역은 일반적으로
            # 페이지의 상당 부분을 차지하지만
            # 페이지 전체 자체는 피한다.
 
            total = (
                area_score * 0.35
                +
                candidate.content_ratio *
                0.20
                +
                candidate.border_score *
                0.25
                +
                candidate.density_score *
                0.15
                +
                candidate.position_score *
                0.05
            )
 
            candidate.total_score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        total
                    )
                )
            )
 
            candidate.confidence = (
                candidate.total_score
            )
 
        return candidates
 
    # ========================================================
    # Border Score
    # ========================================================
 
    def _calculate_border_score(
        self,
        candidate: RegionCandidate,
        binary: np.ndarray
    ) -> float:
 
        x = candidate.x
        y = candidate.y
        w = candidate.width
        h = candidate.height
 
        if w < 10 or h < 10:
            return 0.0
 
        top = binary[
            y,
            x:x + w
        ]
 
        bottom = binary[
            y + h - 1,
            x:x + w
        ]
 
        left = binary[
            y:y + h,
            x
        ]
 
        right = binary[
            y:y + h,
            x + w - 1
        ]
 
        horizontal = (
            np.count_nonzero(top)
            +
            np.count_nonzero(bottom)
        ) / max(
            1,
            2 * w
        )
 
        vertical = (
            np.count_nonzero(left)
            +
            np.count_nonzero(right)
        ) / max(
            1,
            2 * h
        )
 
        score = (
            horizontal +
            vertical
        ) / 2.0
 
        return float(
            min(
                1.0,
                score
            )
        )
     # ========================================================
    # Density Score
    # ========================================================
 
    def _calculate_density_score(
        self,
        candidate: RegionCandidate,
        binary: np.ndarray
    ) -> float:
 
        x = candidate.x
        y = candidate.y
        w = candidate.width
        h = candidate.height
 
        roi = binary[
            y:y + h,
            x:x + w
        ]
 
        if roi.size == 0:
            return 0.0
 
        # 전체 영역을 여러 구역으로 나누어
        # 내용이 특정 한 곳에만 몰려 있는지 확인한다.
 
        grid_rows = 4
        grid_cols = 4
 
        densities = []
 
        for row in range(
            grid_rows
        ):
 
            y1 = (
                row * h //
                grid_rows
            )
 
            y2 = (
                (row + 1) * h //
                grid_rows
            )
 
            for col in range(
                grid_cols
            ):
 
                x1 = (
                    col * w //
                    grid_cols
                )
 
                x2 = (
                    (col + 1) * w //
                    grid_cols
                )
 
                cell = roi[
                    y1:y2,
                    x1:x2
                ]
 
                if cell.size == 0:
                    continue
 
                density = (
                    np.count_nonzero(
                        cell
                    ) /
                    cell.size
                )
 
                densities.append(
                    float(density)
                )
 
        if not densities:
            return 0.0
 
        mean_density = float(
            np.mean(
                densities
            )
        )
 
        non_empty = sum(
            d > 0.001
            for d in densities
        )
 
        distribution = (
            non_empty /
            max(
                1,
                len(densities)
            )
        )
 
        # 도면은 내용이 한 작은 영역에만
        # 몰리는 것보다 넓게 분포하는 경우가 많다.
 
        score = (
            min(
                1.0,
                mean_density * 8.0
            ) * 0.45
            +
            distribution * 0.55
        )
 
        return float(
            min(
                1.0,
                max(
                    0.0,
                    score
                )
            )
        )
 
    # ========================================================
    # Position Score
    # ========================================================
 
    def _calculate_position_score(
        self,
        candidate: RegionCandidate,
        page_width: int,
        page_height: int
    ) -> float:
 
        center_x = (
            candidate.x +
            candidate.width / 2
        )
 
        center_y = (
            candidate.y +
            candidate.height / 2
        )
 
        page_center_x = (
            page_width / 2
        )
 
        page_center_y = (
            page_height / 2
        )
 
        dx = abs(
            center_x -
            page_center_x
        ) / max(
            1,
            page_center_x
        )
 
        dy = abs(
            center_y -
            page_center_y
        ) / max(
            1,
            page_center_y
        )
 
        distance = min(
            1.0,
            (dx + dy) / 2.0
        )
 
        return float(
            1.0 - distance
        )
 
    # ========================================================
    # Select Best Candidate
    # ========================================================
 
    def _select_best(
        self,
        candidates: List[RegionCandidate]
    ) -> Optional[RegionCandidate]:
 
        if not candidates:
            return None
 
        candidates = sorted(
            candidates,
            key=lambda item: item.total_score,
            reverse=True
        )
 
        best = candidates[0]
 
        # ----------------------------------------------------
        # 두 번째 후보와 점수 차이가 너무 작으면
        # 자동 확정하지 않는다.
        # ----------------------------------------------------
 
        if len(candidates) >= 2:
 
            second = candidates[1]
 
            score_gap = (
                best.total_score -
                second.total_score
            )
 
            if score_gap < 0.08:
 
                best.review_required = True
 
                best.confidence = (
                    best.total_score * 0.85
                )
 
                return best
 
        best.confidence = (
            best.total_score
        )
 
        return best
 
    # ========================================================
    # Crop
    # ========================================================
 
    def crop(
        self,
        image: np.ndarray,
        region: RegionCandidate,
        padding_ratio: float = 0.01
    ) -> np.ndarray:
 
        if image is None:
            raise ValueError(
                "image is None"
            )
 
        height, width = image.shape[:2]
 
        padding_x = int(
            region.width *
            padding_ratio
        )
 
        padding_y = int(
            region.height *
            padding_ratio
        )
 
        x1 = max(
            0,
            region.x -
            padding_x
        )
 
        y1 = max(
            0,
            region.y -
            padding_y
        )
 
        x2 = min(
            width,
            region.right +
            padding_x
        )
 
        y2 = min(
            height,
            region.bottom +
            padding_y
        )
 
        return image[
            y1:y2,
            x1:x2
        ].copy()
 
    # ========================================================
    # Draw Candidates
    # ========================================================
 
    def draw_candidates(
        self,
        image: np.ndarray,
        result: DrawingRegionResult
    ) -> np.ndarray:
 
        output = image.copy()
 
        if len(
            output.shape
        ) == 2:
 
            output = cv2.cvtColor(
                output,
                cv2.COLOR_GRAY2BGR
            )
 
        for index, candidate in enumerate(
            result.candidates,
            start=1
        ):
 
            if (
                result.selected is candidate
            ):
 
                thickness = 4
                color = (
                    0,
                    255,
                    0
                )
 
            else:
 
                thickness = 2
                color = (
                    255,
                    180,
                    0
                )
 
            cv2.rectangle(
                output,
                (
                    candidate.x,
                    candidate.y
                ),
                (
                    candidate.right,
                    candidate.bottom
                ),
                color,
                thickness
            )
 
            label = (
                f"{index}: "
                f"{candidate.total_score:.2f}"
            )
 
            cv2.putText(
                output,
                label,
                (
                    candidate.x,
                    max(
                        20,
                        candidate.y - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )
 
        if result.selected is not None:
 
            selected = result.selected
 
            text = (
                f"{result.status} "
                f"confidence="
                f"{result.confidence:.2f}"
            )
 
            cv2.putText(
                output,
                text,
                (
                    20,
                    35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    0
                )
                if result.status == "ACCEPT"
                else (
                    0,
                    165,
                    255
                ),
                2,
                cv2.LINE_AA
            )
 
        return output
 
    # ========================================================
    # Save Visualization
    # ========================================================
 
    def save_visualization(
        self,
        image: np.ndarray,
        result: DrawingRegionResult,
        output_path: str | Path
    ) -> Path:
 
        output_path = Path(
            output_path
        )
 
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
 
        visualization = (
            self.draw_candidates(
                image,
                result
            )
        )
 
        success = cv2.imwrite(
            str(output_path),
            visualization
        )
 
        if not success:
 
            raise IOError(
                f"failed to save image: "
                f"{output_path}"
            )
 
        return output_path
 
    # ========================================================
    # Result Summary
    # ========================================================
 
    def summary(
        self,
        result: DrawingRegionResult
    ) -> Dict[str, Any]:
 
        selected = result.selected
 
        return {
            "success": result.success,
            "status": result.status,
            "confidence": round(
                result.confidence,
                4
            ),
            "candidate_count": len(
                result.candidates
            ),
            "selected_region": (
                {
                    "x": selected.x,
                    "y": selected.y,
                    "width": selected.width,
                    "height": selected.height,
                }
                if selected is not None
                else None
            ),
            "review_required": (
                selected.review_required
                if selected is not None
                else True
            ),
            "reason": result.reason,
        }
 # ============================================================
# Standalone Helper Functions
# ============================================================
 
def detect_drawing_region(
    image: np.ndarray,
    **kwargs
) -> DrawingRegionResult:
 
    detector = DrawingRegionDetector(
        **kwargs
    )
 
    return detector.detect(
        image
    )
 
 
def crop_selected_region(
    image: np.ndarray,
    result: DrawingRegionResult,
    padding_ratio: float = 0.01
) -> np.ndarray:
 
    if result.selected is None:
 
        raise ValueError(
            "No drawing region selected"
        )
 
    detector = DrawingRegionDetector()
 
    return detector.crop(
        image,
        result.selected,
        padding_ratio
    )
 
 
# ============================================================
# PDF / PageImage Compatibility
# ============================================================
 
def extract_image_from_page(
    page: Any
) -> np.ndarray:
 
    # PageImage가 ndarray를 직접 가지고 있는 경우
    if isinstance(
        page,
        np.ndarray
    ):
 
        return page
 
    # 일반적으로 사용되는 속성 순서
    possible_attributes = [
        "image",
        "array",
        "img",
        "data",
    ]
 
    for attribute in (
        possible_attributes
    ):
 
        if hasattr(
            page,
            attribute
        ):
 
            value = getattr(
                page,
                attribute
            )
 
            if isinstance(
                value,
                np.ndarray
            ):
 
                return value
 
    raise TypeError(
        "Page object does not contain "
        "a supported numpy image attribute"
    )
 
 
def detect_page_region(
    page: Any,
    **kwargs
) -> DrawingRegionResult:
 
    image = extract_image_from_page(
        page
    )
 
    return detect_drawing_region(
        image,
        **kwargs
    )
 
 
# ============================================================
# Diagnostic Information
# ============================================================
 
def result_to_text(
    result: DrawingRegionResult
) -> str:
 
    lines = []
 
    lines.append(
        "DRAWING REGION RESULT"
    )
 
    lines.append(
        f"SUCCESS: {result.success}"
    )
 
    lines.append(
        f"STATUS: {result.status}"
    )
 
    lines.append(
        f"CONFIDENCE: "
        f"{result.confidence:.4f}"
    )
 
    lines.append(
        f"CANDIDATES: "
        f"{len(result.candidates)}"
    )
 
    if result.selected is not None:
 
        selected = result.selected
 
        lines.append(
            "SELECTED:"
        )
 
        lines.append(
            f"  X={selected.x}"
        )
 
        lines.append(
            f"  Y={selected.y}"
        )
 
        lines.append(
            f"  WIDTH={selected.width}"
        )
 
        lines.append(
            f"  HEIGHT={selected.height}"
        )
 
        lines.append(
            f"  SCORE="
            f"{selected.total_score:.4f}"
        )
 
        lines.append(
            f"  REVIEW="
            f"{selected.review_required}"
        )
 
    else:
 
        lines.append(
            "SELECTED: NONE"
        )
 
    lines.append(
        f"REASON: {result.reason}"
    )
 
    return "\n".join(
        lines
    )
 
 
# ============================================================
# Module Self Test
# ============================================================
 
def _self_test() -> bool:
 
    # 실제 PDF를 읽지 않고
    # 기본적인 객체 생성과 빈 이미지 처리를 확인한다.
 
    detector = DrawingRegionDetector()
 
    test_image = np.full(
        (
            400,
            600,
            3
        ),
        255,
        dtype=np.uint8
    )
 
    # 테스트용 사각형 도면
    cv2.rectangle(
        test_image,
        (60, 50),
        (540, 350),
        (0, 0, 0),
        3
    )
 
    result = detector.detect(
        test_image
    )
 
    if not result.success:
        return False
 
    if result.page_width != 600:
        return False
 
    if result.page_height != 400:
        return False
 
    return True
 
 
# ============================================================
# Main
# ============================================================
 
if __name__ == "__main__":
 
    try:
 
        ok = _self_test()
 
        if ok:
 
            print(
                "DRAWING_REGION SELF TEST: OK"
            )
 
        else:
 
            print(
                "DRAWING_REGION SELF TEST: FAIL"
            )
 
    except Exception as exc:
 
        print(
            "DRAWING_REGION SELF TEST: ERROR"
        )
 
        print(
            type(exc).__name__,
            ":",
            exc
        )
# ============================================================
# Part 5
# Safe Drawing Region Boundary Correction
# ============================================================
 
def _safe_expand_region(
    self,
    image: np.ndarray,
    region: RegionCandidate,
) -> RegionCandidate:
 
    """
    최종 ROI 경계 보정.
 
    원칙
    ----
    1. 기존 ROI를 기본적으로 신뢰한다.
    2. 좌우 방향은 자동 확장하지 않는다.
    3. 위쪽은 확장하지 않는다.
    4. 아래쪽에 실제 표제란 내용이 잘린 경우에만 제한적으로 확장한다.
    5. 페이지 전체를 ROI로 선택하지 않는다.
    """
 
    if image is None:
        return region
 
    height, width = image.shape[:2]
 
    x1 = int(region.x)
    y1 = int(region.y)
    x2 = int(region.right)
    y2 = int(region.bottom)
 
    original_width = x2 - x1
    original_height = y2 - y1
 
    if (
        original_width <= 0
        or original_height <= 0
    ):
        return region
 
    # --------------------------------------------------------
    # 좌우는 절대 자동 확장하지 않는다.
    # --------------------------------------------------------
 
    new_x1 = x1
    new_x2 = x2
 
    # --------------------------------------------------------
    # 위쪽도 자동 확장하지 않는다.
    # --------------------------------------------------------
 
    new_y1 = y1
    new_y2 = y2
 
    # --------------------------------------------------------
    # 아래쪽 표제란 확인
    # --------------------------------------------------------
 
    gray = self._to_gray(image)
 
    binary = self._make_binary(
        gray
    )
 
    # 기존 ROI 바로 아래쪽만 검사한다.
    #
    # 전체 페이지를 검색하지 않는다.
    # 따라서 페이지 하단의 다른 정보 때문에
    # ROI가 페이지 전체로 확장되지 않는다.
    # --------------------------------------------------------
 
    max_bottom_expand = max(
        10,
        int(
            original_height * 0.08
        )
    )
 
    bottom_start = y2
 
    bottom_end = min(
        height,
        y2 + max_bottom_expand
    )
 
    if bottom_end > bottom_start:
 
        bottom_band = binary[
            bottom_start:bottom_end,
            x1:x2
        ]
 
        if bottom_band.size > 0:
 
            # 각 행에 존재하는 실제 픽셀 수
            row_pixels = np.count_nonzero(
                bottom_band,
                axis=1
            )
 
            # 기존 ROI 폭 대비 최소 픽셀 비율
            minimum_pixels = max(
                5,
                int(
                    original_width *
                    0.001
                )
            )
 
            valid_rows = np.where(
                row_pixels >= minimum_pixels
            )[0]
 
            if len(valid_rows) > 0:
 
                # 바로 아래쪽에 실제 내용이
                # 연속적으로 존재하는지 확인
                first = int(
                    valid_rows[0]
                )
 
                last = int(
                    valid_rows[-1]
                )
 
                continuous_length = (
                    last - first + 1
                )
 
                # 한두 줄의 노이즈는 무시한다.
                minimum_continuous = max(
                    3,
                    int(
                        max_bottom_expand *
                        0.20
                    )
                )
 
                if (
                    continuous_length
                    >=
                    minimum_continuous
                ):
 
                    # 최대 8%까지만 아래로 확장
                    extension = min(
                        max_bottom_expand,
                        last + 1
                    )
 
                    new_y2 = min(
                        height,
                        y2 + extension
                    )
 
    # --------------------------------------------------------
    # 페이지 외곽 보호
    # --------------------------------------------------------
 
    # 페이지 폭의 97% 이상을 차지하는 ROI는
    # 자동 확장을 허용하지 않는다.
    page_width_ratio = (
        (new_x2 - new_x1)
        /
        max(
            1,
            width
        )
    )
 
    if page_width_ratio >= 0.97:
 
        new_x1 = x1
        new_x2 = x2
 
    # 페이지 높이의 97% 이상도 방지
    page_height_ratio = (
        (new_y2 - new_y1)
        /
        max(
            1,
            height
        )
    )
 
    if page_height_ratio >= 0.97:
 
        new_y1 = y1
        new_y2 = y2
 
    # --------------------------------------------------------
    # 최종 크기
    # --------------------------------------------------------
 
    new_width = (
        new_x2 - new_x1
    )
 
    new_height = (
        new_y2 - new_y1
    )
 
    if (
        new_width <= 0
        or
        new_height <= 0
    ):
        return region
 
    # --------------------------------------------------------
    # 실제로 변경되지 않았다면 기존 객체 반환
    # --------------------------------------------------------
 
    if (
        new_x1 == x1
        and
        new_y1 == y1
        and
        new_x2 == x2
        and
        new_y2 == y2
    ):
 
        return region
 
    # --------------------------------------------------------
    # 새로운 RegionCandidate
    # --------------------------------------------------------
 
    expanded = RegionCandidate(
        x=int(new_x1),
        y=int(new_y1),
        width=int(new_width),
        height=int(new_height),
        area=int(
            new_width *
            new_height
        ),
        page_area_ratio=(
            new_width *
            new_height
        )
        /
        max(
            1,
            width * height
        ),
        content_ratio=region.content_ratio,
        border_score=region.border_score,
        density_score=region.density_score,
        position_score=region.position_score,
        total_score=region.total_score,
        region_type="drawing_with_title_block",
        confidence=region.confidence,
        review_required=True,
    )
 
    return expanded
 
 
# ============================================================
# Improved Detect
# ============================================================
 
_original_detect = DrawingRegionDetector.detect
 
 
def _safe_detect(
    self,
    image: np.ndarray
) -> DrawingRegionResult:
 
    result = _original_detect(
        self,
        image
    )
 
    if (
        not result.success
        or
        result.selected is None
    ):
        return result
 
    original = result.selected
 
    corrected = _safe_expand_region(
        self,
        image,
        original
    )
 
    original_area = (
        original.width *
        original.height
    )
 
    corrected_area = (
        corrected.width *
        corrected.height
    )
 
    # --------------------------------------------------------
    # 실제로 확장된 경우만 적용
    # --------------------------------------------------------
 
    if corrected_area > original_area:
 
        result.selected = corrected
 
        result.status = "REVIEW"
 
        result.confidence = max(
            0.0,
            min(
                1.0,
                result.confidence * 0.98
            )
        )
 
        result.selected.confidence = (
            result.confidence
        )
 
        result.selected.review_required = True
 
        result.reason = (
            "title block boundary "
            "correction applied"
        )
 
    return result
 
 
DrawingRegionDetector.detect = _safe_detect
 