from __future__ import annotations
 
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
 
from .text_extractor import TextRegion
 
 
# ============================================================
# Text Match Result
# ============================================================
 
@dataclass
class TextMatch:
    before: TextRegion
    after: TextRegion
 
    similarity: float
    position_similarity: float
    size_similarity: float
    type_similarity: float
 
    score: float
 
    status: str = "MATCH"
 
 
# ============================================================
# Text Matcher
# ============================================================
 
class TextMatcher:
 
    def __init__(
        self,
        min_score: float = 0.45,
        review_score: float = 0.65,
        max_position_distance: float = 0.35,
    ):
 
        self.min_score = min_score
 
        self.review_score = review_score
 
        self.max_position_distance = (
            max_position_distance
        )
 
    # ========================================================
    # Main matching function
    # ========================================================
 
    def match(
        self,
        before_regions: List[TextRegion],
        after_regions: List[TextRegion],
        page_width: Optional[int] = None,
        page_height: Optional[int] = None,
    ) -> List[TextMatch]:
 
        if not before_regions:
            return []
 
        if not after_regions:
            return []
 
        if page_width is None:
            page_width = self._estimate_page_width(
                before_regions,
                after_regions,
            )
 
        if page_height is None:
            page_height = self._estimate_page_height(
                before_regions,
                after_regions,
            )
 
        matches = []
 
        used_after = set()
 
        for before in before_regions:
 
            candidates = []
 
            for index, after in enumerate(
                after_regions
            ):
 
                if index in used_after:
                    continue
 
                score_data = self._calculate_score(
                    before,
                    after,
                    page_width,
                    page_height,
                )
 
                score = score_data[0]
 
                if score < self.min_score:
                    continue
 
                candidates.append(
                    (
                        score,
                        index,
                        score_data,
                    )
                )
 
            if not candidates:
                continue
 
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )
 
            best_score, best_index, data = (
                candidates[0]
            )
 
            status = (
                "MATCH"
                if best_score >= self.review_score
                else "REVIEW"
            )
 
            matches.append(
                TextMatch(
                    before=before,
                    after=after_regions[
                        best_index
                    ],
                    similarity=data[1],
                    position_similarity=data[2],
                    size_similarity=data[3],
                    type_similarity=data[4],
                    score=best_score,
                    status=status,
                )
            )
 
            used_after.add(
                best_index
            )
 
        return matches
 
    # ========================================================
    # Calculate matching score
    # ========================================================
 
    def _calculate_score(
        self,
        before: TextRegion,
        after: TextRegion,
        page_width: int,
        page_height: int,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
    ]:
 
        text_similarity = (
            self._text_similarity(
                before.text,
                after.text,
            )
        )
 
        position_similarity = (
            self._position_similarity(
                before,
                after,
                page_width,
                page_height,
            )
        )
 
        size_similarity = (
            self._size_similarity(
                before,
                after,
            )
        )
 
        type_similarity = (
            1.0
            if before.region_type
            == after.region_type
            else 0.0
        )
 
        score = (
            text_similarity * 0.45
            +
            position_similarity * 0.30
            +
            size_similarity * 0.10
            +
            type_similarity * 0.15
        )
 
        return (
            score,
            text_similarity,
            position_similarity,
            size_similarity,
            type_similarity,
        )
 
    # ========================================================
    # Text similarity
    # ========================================================
 
    def _text_similarity(
        self,
        before_text: str,
        after_text: str,
    ) -> float:
 
        before_text = (
            before_text
            .strip()
            .upper()
        )
 
        after_text = (
            after_text
            .strip()
            .upper()
        )
 
        if not before_text:
            return 0.0
 
        if not after_text:
            return 0.0
 
        if before_text == after_text:
            return 1.0
 
        return SequenceMatcher(
            None,
            before_text,
            after_text,
        ).ratio()
 
    # ========================================================
    # Position similarity
    # ========================================================
 
    def _position_similarity(
        self,
        before: TextRegion,
        after: TextRegion,
        page_width: int,
        page_height: int,
    ) -> float:
 
        width = max(
            page_width,
            1,
        )
 
        height = max(
            page_height,
            1,
        )
 
        dx = abs(
            before.center_x
            -
            after.center_x
        ) / width
 
        dy = abs(
            before.center_y
            -
            after.center_y
        ) / height
 
        distance = (
            dx * dx
            +
            dy * dy
        ) ** 0.5
 
        if distance >= (
            self.max_position_distance
        ):
 
            return 0.0
 
        return max(
            0.0,
            1.0
            -
            (
                distance
                /
                self.max_position_distance
            ),
        )
 
    # ========================================================
    # Size similarity
    # ========================================================
 
    def _size_similarity(
        self,
        before: TextRegion,
        after: TextRegion,
    ) -> float:
 
        before_width = max(
            before.width,
            1,
        )
 
        after_width = max(
            after.width,
            1,
        )
 
        before_height = max(
            before.height,
            1,
        )
 
        after_height = max(
            after.height,
            1,
        )
 
        width_ratio = min(
            before_width,
            after_width,
        ) / max(
            before_width,
            after_width,
        )
 
        height_ratio = min(
            before_height,
            after_height,
        ) / max(
            before_height,
            after_height,
        )
 
        return (
            width_ratio
            +
            height_ratio
        ) / 2.0
     # ========================================================
    # Estimate page width
    # ========================================================
 
    def _estimate_page_width(
        self,
        before_regions: List[TextRegion],
        after_regions: List[TextRegion],
    ) -> int:
 
        all_regions = (
            before_regions
            +
            after_regions
        )
 
        if not all_regions:
            return 1
 
        right_edges = [
            region.x + region.width
            for region in all_regions
        ]
 
        return max(
            max(right_edges),
            1,
        )
 
    # ========================================================
    # Estimate page height
    # ========================================================
 
    def _estimate_page_height(
        self,
        before_regions: List[TextRegion],
        after_regions: List[TextRegion],
    ) -> int:
 
        all_regions = (
            before_regions
            +
            after_regions
        )
 
        if not all_regions:
            return 1
 
        bottom_edges = [
            region.y + region.height
            for region in all_regions
        ]
 
        return max(
            max(bottom_edges),
            1,
        )
 
 
# ============================================================
# Convenience Function
# ============================================================
 
def match_text_regions(
    before_regions: List[TextRegion],
    after_regions: List[TextRegion],
    page_width: Optional[int] = None,
    page_height: Optional[int] = None,
) -> List[TextMatch]:
 
    matcher = TextMatcher()
 
    return matcher.match(
        before_regions,
        after_regions,
        page_width,
        page_height,
    )