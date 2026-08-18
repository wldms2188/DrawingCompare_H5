from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import List, Optional
 
import cv2
import numpy as np
 
 
# ============================================================
# Drawing Region
# ============================================================
 
@dataclass
class DrawingRegion:
 
    region_id: int
 
    x: int
    y: int
    width: int
    height: int
 
    region_type: str = "SECTION"
 
    confidence: float = 0.0
 
    text_regions: List = field(
        default_factory=list
    )
 
    @property
    def right(self) -> int:
        return self.x + self.width
 
    @property
    def bottom(self) -> int:
        return self.y + self.height
 
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0
 
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0
 
    @property
    def area(self) -> int:
        return self.width * self.height
 
    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
 
        return (
            self.x <= x <= self.right
            and
            self.y <= y <= self.bottom
        )
 
 
# ============================================================
# Region Match
# ============================================================
 
@dataclass
class RegionMatch:
 
    before: DrawingRegion
 
    after: DrawingRegion
 
    position_score: float
 
    size_score: float
 
    visual_score: float
 
    text_score: float
 
    type_score: float
 
    score: float
 
    status: str = "REVIEW"
 
    reason: str = ""
 
 
# ============================================================
# Region Matcher
# ============================================================
 
class RegionMatcher:
 
    def __init__(
        self,
        min_score: float = 0.45,
        review_score: float = 0.70,
        minimum_visual_score: float = 0.60,
        minimum_position_score: float = 0.55,
    ):
 
        self.min_score = min_score
 
        self.review_score = review_score
 
        self.minimum_visual_score = (
            minimum_visual_score
        )
 
        self.minimum_position_score = (
            minimum_position_score
        )
 
    # ========================================================
    # Main matching
    # ========================================================
 
    def match(
        self,
        before_regions: List[DrawingRegion],
        after_regions: List[DrawingRegion],
        before_image: Optional[np.ndarray] = None,
        after_image: Optional[np.ndarray] = None,
    ) -> List[RegionMatch]:
 
        if not before_regions:
            return []
 
        if not after_regions:
            return []
 
        results = []
 
        used_after = set()
 
        for before in before_regions:
 
            candidates = []
 
            for index, after in enumerate(
                after_regions
            ):
 
                if index in used_after:
                    continue
 
                position_score = (
                    self._position_score(
                        before,
                        after,
                    )
                )
 
                size_score = (
                    self._size_score(
                        before,
                        after,
                    )
                )
 
                visual_score = (
                    self._visual_score(
                        before,
                        after,
                        before_image,
                        after_image,
                    )
                )
 
                text_score = (
                    self._text_score(
                        before,
                        after,
                    )
                )
 
                type_score = (
                    self._type_score(
                        before,
                        after,
                    )
                )
 
                score = (
                    position_score * 0.25
                    +
                    size_score * 0.15
                    +
                    visual_score * 0.35
                    +
                    text_score * 0.20
                    +
                    type_score * 0.05
                )
 
                if score < self.min_score:
                    continue
 
                candidates.append(
                    (
                        score,
                        index,
                        position_score,
                        size_score,
                        visual_score,
                        text_score,
                        type_score,
                    )
                )
 
            if not candidates:
                continue
 
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )
 
            (
                score,
                best_index,
                position_score,
                size_score,
                visual_score,
                text_score,
                type_score,
            ) = candidates[0]
 
            status = self._determine_status(
                score,
                position_score,
                visual_score,
                text_score,
            )
 
            reason = self._make_reason(
                score,
                position_score,
                size_score,
                visual_score,
                text_score,
                type_score,
                status,
            )
 
            results.append(
                RegionMatch(
                    before=before,
                    after=after_regions[
                        best_index
                    ],
                    position_score=position_score,
                    size_score=size_score,
                    visual_score=visual_score,
                    text_score=text_score,
                    type_score=type_score,
                    score=score,
                    status=status,
                    reason=reason,
                )
            )
 
            used_after.add(
                best_index
            )
 
        return results
 
    # ========================================================
    # Determine status
    # ========================================================
 
    def _determine_status(
        self,
        score: float,
        position_score: float,
        visual_score: float,
        text_score: float,
    ) -> str:
 
        strong_position = (
            position_score
            >=
            self.minimum_position_score
        )
 
        strong_visual = (
            visual_score
            >=
            self.minimum_visual_score
        )
 
        strong_text = (
            text_score >= 0.50
        )
 
        if (
            score >= self.review_score
            and
            strong_position
            and
            (
                strong_visual
                or
                strong_text
            )
        ):
 
            return "MATCH"
 
        return "REVIEW"
 
    # ========================================================
    # Position similarity
    # ========================================================
 
    def _position_score(
        self,
        before: DrawingRegion,
        after: DrawingRegion,
    ) -> float:
 
        dx = abs(
            before.center_x
            -
            after.center_x
        )
 
        dy = abs(
            before.center_y
            -
            after.center_y
        )
 
        reference_width = max(
            before.width,
            after.width,
            1,
        )
 
        reference_height = max(
            before.height,
            after.height,
            1,
        )
 
        normalized_x = (
            dx
            /
            reference_width
        )
 
        normalized_y = (
            dy
            /
            reference_height
        )
 
        distance = (
            normalized_x
            ** 2
            +
            normalized_y
            ** 2
        ) ** 0.5
 
        return max(
            0.0,
            1.0
            -
            min(
                distance,
                1.0,
            ),
        )
 
    # ========================================================
    # Size similarity
    # ========================================================
 
    def _size_score(
        self,
        before: DrawingRegion,
        after: DrawingRegion,
    ) -> float:
 
        width_score = (
            min(
                before.width,
                after.width,
            )
            /
            max(
                before.width,
                after.width,
                1,
            )
        )
 
        height_score = (
            min(
                before.height,
                after.height,
            )
            /
            max(
                before.height,
                after.height,
                1,
            )
        )
 
        return (
            width_score
            +
            height_score
        ) / 2.0
            # ========================================================
    # Visual similarity
    # ========================================================
 
    def _visual_score(
        self,
        before: DrawingRegion,
        after: DrawingRegion,
        before_image: Optional[np.ndarray],
        after_image: Optional[np.ndarray],
    ) -> float:
 
        # 이미지가 없는 상태에서는
        # 시각적 근거가 없다고 판단한다.
        if (
            before_image is None
            or
            after_image is None
        ):
 
            return 0.0
 
        before_crop = self._crop(
            before_image,
            before,
        )
 
        after_crop = self._crop(
            after_image,
            after,
        )
 
        if (
            before_crop is None
            or
            after_crop is None
        ):
 
            return 0.0
 
        before_gray = self._prepare(
            before_crop
        )
 
        after_gray = self._prepare(
            after_crop
        )
 
        if (
            before_gray.size == 0
            or
            after_gray.size == 0
        ):
 
            return 0.0
 
        target_width = max(
            before_gray.shape[1],
            1,
        )
 
        target_height = max(
            before_gray.shape[0],
            1,
        )
 
        after_gray = cv2.resize(
            after_gray,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )
 
        difference = cv2.absdiff(
            before_gray,
            after_gray,
        )
 
        mean_difference = (
            float(
                np.mean(
                    difference
                )
            )
            /
            255.0
        )
 
        return max(
            0.0,
            1.0 - mean_difference,
        )
 
    # ========================================================
    # Text similarity
    # ========================================================
 
    def _text_score(
        self,
        before: DrawingRegion,
        after: DrawingRegion,
    ) -> float:
 
        before_regions = (
            before.text_regions
        )
 
        after_regions = (
            after.text_regions
        )
 
        if (
            not before_regions
            or
            not after_regions
        ):
 
            return 0.0
 
        before_texts = [
            str(
                getattr(
                    item,
                    "text",
                    "",
                )
            ).strip().upper()
            for item in before_regions
        ]
 
        after_texts = [
            str(
                getattr(
                    item,
                    "text",
                    "",
                )
            ).strip().upper()
            for item in after_regions
        ]
 
        before_texts = [
            text
            for text in before_texts
            if text
        ]
 
        after_texts = [
            text
            for text in after_texts
            if text
        ]
 
        if (
            not before_texts
            or
            not after_texts
        ):
 
            return 0.0
 
        # 같은 텍스트가 존재하는 비율
        matched = 0
 
        used = set()
 
        for before_text in before_texts:
 
            best_index = None
            best_score = 0.0
 
            for index, after_text in enumerate(
                after_texts
            ):
 
                if index in used:
                    continue
 
                score = self._string_similarity(
                    before_text,
                    after_text,
                )
 
                if score > best_score:
 
                    best_score = score
                    best_index = index
 
            if (
                best_index is not None
                and
                best_score >= 0.55
            ):
 
                matched += 1
                used.add(
                    best_index
                )
 
        return (
            matched
            /
            max(
                len(before_texts),
                len(after_texts),
            )
        )
 
    # ========================================================
    # String similarity
    # ========================================================
 
    def _string_similarity(
        self,
        first: str,
        second: str,
    ) -> float:
 
        if not first or not second:
            return 0.0
 
        if first == second:
            return 1.0
 
        from difflib import SequenceMatcher
 
        return SequenceMatcher(
            None,
            first,
            second,
        ).ratio()
 
    # ========================================================
    # Region type similarity
    # ========================================================
 
    def _type_score(
        self,
        before: DrawingRegion,
        after: DrawingRegion,
    ) -> float:
 
        if (
            before.region_type
            ==
            after.region_type
        ):
 
            return 1.0
 
        # 표제란은 표제란끼리 우선 대응
        if (
            before.region_type
            == "TITLE_BLOCK"
            and
            after.region_type
            == "TITLE_BLOCK"
        ):
 
            return 1.0
 
        # 섹션끼리 대응
        if (
            before.region_type
            == "SECTION"
            and
            after.region_type
            == "SECTION"
        ):
 
            return 1.0
 
        return 0.0
 
    # ========================================================
    # Crop
    # ========================================================
 
    def _crop(
        self,
        image: np.ndarray,
        region: DrawingRegion,
    ) -> Optional[np.ndarray]:
 
        if image is None:
            return None
 
        image_height, image_width = (
            image.shape[:2]
        )
 
        x1 = max(
            0,
            region.x,
        )
 
        y1 = max(
            0,
            region.y,
        )
 
        x2 = min(
            image_width,
            region.right,
        )
 
        y2 = min(
            image_height,
            region.bottom,
        )
 
        if x2 <= x1:
            return None
 
        if y2 <= y1:
            return None
 
        return image[
            y1:y2,
            x1:x2,
        ]
 
    # ========================================================
    # Image preparation
    # ========================================================
 
    def _prepare(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
 
        if len(image.shape) == 3:
 
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
 
        else:
 
            gray = image
 
        return cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )
 
    # ========================================================
    # Reason
    # ========================================================
 
    def _make_reason(
        self,
        score: float,
        position_score: float,
        size_score: float,
        visual_score: float,
        text_score: float,
        type_score: float,
        status: str,
    ) -> str:
 
        reasons = []
 
        if position_score < 0.55:
 
            reasons.append(
                "position differs"
            )
 
        if size_score < 0.50:
 
            reasons.append(
                "region size differs"
            )
 
        if visual_score < 0.60:
 
            reasons.append(
                "visual evidence weak"
            )
 
        if text_score < 0.50:
 
            reasons.append(
                "text evidence weak"
            )
 
        if type_score < 1.0:
 
            reasons.append(
                "region type differs"
            )
 
        if status == "MATCH":
 
            if not reasons:
 
                return (
                    "high confidence region match"
                )
 
            return (
                "region matched; "
                +
                "; ".join(reasons)
            )
 
        if not reasons:
 
            return (
                "region match requires review"
            )
 
        return (
            "review required; "
            +
            "; ".join(reasons)
        )
 
 
# ============================================================
# Convenience Function
# ============================================================
 
def match_regions(
    before_regions: List[DrawingRegion],
    after_regions: List[DrawingRegion],
    before_image: Optional[np.ndarray] = None,
    after_image: Optional[np.ndarray] = None,
) -> List[RegionMatch]:
 
    matcher = RegionMatcher()
 
    return matcher.match(
        before_regions,
        after_regions,
        before_image,
        after_image,
    )