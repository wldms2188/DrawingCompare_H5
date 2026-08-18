from __future__ import annotations
 
from dataclasses import dataclass
from typing import List
 
from .text_matcher import TextMatch
 
 
# ============================================================
# Change Result
# ============================================================
 
@dataclass
class TextChange:
    change_type: str
 
    before_text: str
    after_text: str
 
    page_index: int
 
    score: float
 
    status: str
 
    before_x: int
    before_y: int
 
    after_x: int
    after_y: int
 
    reason: str = ""
 
    def to_dict(self) -> dict:
 
        return {
            "change_type": self.change_type,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "page_index": self.page_index,
            "score": self.score,
            "status": self.status,
            "before_x": self.before_x,
            "before_y": self.before_y,
            "after_x": self.after_x,
            "after_y": self.after_y,
            "reason": self.reason,
        }
 
 
# ============================================================
# Change Classifier
# ============================================================
 
class ChangeClassifier:
 
    def __init__(
        self,
        minimum_change_score: float = 0.45,
        review_score: float = 0.65,
    ):
 
        self.minimum_change_score = (
            minimum_change_score
        )
 
        self.review_score = (
            review_score
        )
 
    # ========================================================
    # Main classification
    # ========================================================
 
    def classify(
        self,
        matches: List[TextMatch],
    ) -> List[TextChange]:
 
        results = []
 
        for match in matches:
 
            if not self._is_actual_change(
                match
            ):
                continue
 
            change_type = (
                self._get_change_type(
                    match
                )
            )
 
            status = self._get_status(
                match
            )
 
            reason = self._get_reason(
                match,
                status,
            )
 
            results.append(
                TextChange(
                    change_type=change_type,
                    before_text=(
                        match.before.text
                    ),
                    after_text=(
                        match.after.text
                    ),
                    page_index=(
                        match.before.page_index
                    ),
                    score=match.score,
                    status=status,
                    before_x=(
                        match.before.x
                    ),
                    before_y=(
                        match.before.y
                    ),
                    after_x=(
                        match.after.x
                    ),
                    after_y=(
                        match.after.y
                    ),
                    reason=reason,
                )
            )
 
        return results
 
    # ========================================================
    # Check actual text change
    # ========================================================
 
    def _is_actual_change(
        self,
        match: TextMatch,
    ) -> bool:
 
        before = (
            match.before.text
            .strip()
            .upper()
        )
 
        after = (
            match.after.text
            .strip()
            .upper()
        )
 
        if before == after:
            return False
 
        if match.score < (
            self.minimum_change_score
        ):
            return False
 
        return True
 
    # ========================================================
    # Determine change type
    # ========================================================
 
    def _get_change_type(
        self,
        match: TextMatch,
    ) -> str:
 
        before_type = (
            match.before.region_type
        )
 
        after_type = (
            match.after.region_type
        )
 
        # ----------------------------------------------------
        # Title block has highest priority
        # ----------------------------------------------------
 
        if (
            before_type
            == "TITLE_BLOCK"
            or
            after_type
            == "TITLE_BLOCK"
        ):
            return "TITLE_BLOCK"
 
        # ----------------------------------------------------
        # GD&T
        # ----------------------------------------------------
 
        if (
            before_type
            == "GDT"
            or
            after_type
            == "GDT"
        ):
            return "GDT"
 
        # ----------------------------------------------------
        # Dimension
        # ----------------------------------------------------
 
        if (
            before_type
            == "DIMENSION"
            or
            after_type
            == "DIMENSION"
        ):
            return "DIMENSION"
 
        # ----------------------------------------------------
        # Item
        # ----------------------------------------------------
 
        if (
            before_type
            == "ITEM"
            or
            after_type
            == "ITEM"
        ):
            return "ITEM"
 
        # ----------------------------------------------------
        # Comment
        # ----------------------------------------------------
 
        if (
            before_type
            == "COMMENT"
            or
            after_type
            == "COMMENT"
        ):
            return "COMMENT"
 
        return "TEXT"
        # ========================================================
    # Status
    # ========================================================
 
    def _get_status(
        self,
        match: TextMatch,
    ) -> str:
 
        if match.score >= self.review_score:
            return "CHANGED"
 
        return "REVIEW"
 
    # ========================================================
    # Reason
    # ========================================================
 
    def _get_reason(
        self,
        match: TextMatch,
        status: str,
    ) -> str:
 
        reasons = []
 
        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------
 
        if match.position_similarity < 0.70:
 
            reasons.append(
                "position similarity low"
            )
 
        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------
 
        if match.size_similarity < 0.70:
 
            reasons.append(
                "text size changed"
            )
 
        # ----------------------------------------------------
        # Type
        # ----------------------------------------------------
 
        if match.type_similarity < 1.0:
 
            reasons.append(
                "region type changed"
            )
 
        # ----------------------------------------------------
        # Text similarity
        #
        # 실제 변경된 값은 문자열이 달라지는 것이 정상이다.
        # 따라서 text similarity만 낮다고 REVIEW로 만들지 않는다.
        # ----------------------------------------------------
 
        if match.similarity < 0.40:
 
            reasons.append(
                "text similarity very low"
            )
 
        # ----------------------------------------------------
        # Final reason
        # ----------------------------------------------------
 
        if status == "CHANGED":
 
            if not reasons:
 
                return (
                    "high confidence text change"
                )
 
            return (
                "high confidence change; "
                + "; ".join(reasons)
            )
 
        if not reasons:
 
            return (
                "matched text requires review"
            )
 
        return (
            "review required; "
            + "; ".join(reasons)
        )
 
 
# ============================================================
# Convenience Function
# ============================================================
 
def classify_text_changes(
    matches: List[TextMatch],
) -> List[TextChange]:
 
    classifier = ChangeClassifier()
 
    return classifier.classify(
        matches
    )
 