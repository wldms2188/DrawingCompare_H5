"""
DrawingCompare H5
core/page_matcher.py
 
역할
------------------------------------------------------------
Before PDF와 After PDF 사이에서
실제로 같은 도면을 나타내는 페이지를 자동으로 찾는다.
 
예:
    Before
        1페이지
        2페이지
        3페이지
 
    After
        1페이지
        2페이지
        3페이지
 
페이지 순서가 달라져도
 
    Before 1 -> After 3
    Before 2 -> After 1
    Before 3 -> After 2
 
처럼 자동 매칭한다.
 
중요
------------------------------------------------------------
- 단순히 페이지 번호를 비교하지 않는다.
- 이미지 크기가 달라도 비교한다.
- 파일 매칭과 페이지 매칭을 분리한다.
- 확신이 낮은 매칭은 REVIEW로 보낸다.
- 한 After 페이지가 여러 Before 페이지에 중복 매칭되지 않는다.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import List, Optional
 
import cv2
import numpy as np
 
from config import CONFIG
from core.image_loader import (
    PDFDocument,
    PageImage,
)
 
 
# ============================================================
# DATA CLASS
# ============================================================
 
@dataclass
class PageMatch:
    """
    Before 페이지와 After 페이지의 매칭 결과
    """
 
    before_page: PageImage
 
    after_page: PageImage
 
    score: float
 
    feature_score: float
 
    structure_score: float
 
    size_score: float
 
    position_score: float
 
    status: str
 
    reason: str
 
 
# ============================================================
# PAGE MATCHER
# ============================================================
 
class PageMatcher:
 
    def __init__(self, config=None):
 
        self.config = config or CONFIG
 
        self.minimum_score = (
            self.config.page_match.minimum_score
        )
 
        self.review_score = (
            self.config.page_match.review_score
        )
 
        self.minimum_feature_matches = (
            self.config.page_match.minimum_feature_matches
        )
 
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def match_pages(
        self,
        before: PDFDocument,
        after: PDFDocument,
    ) -> List[PageMatch]:
        """
        두 PDF의 모든 페이지를 비교하여
        대응 페이지를 자동으로 찾는다.
        """
 
        if not before.pages:
 
            return []
 
        if not after.pages:
 
            return []
 
        candidates = []
 
        for before_page in before.pages:
 
            for after_page in after.pages:
 
                result = self._calculate_page_match(
                    before_page,
                    after_page
                )
 
                candidates.append(
                    result
                )
 
        return self._select_page_matches(
            candidates
        )
 
 
    # ========================================================
    # CALCULATE PAGE MATCH
    # ========================================================
 
    def _calculate_page_match(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> PageMatch:
        """
        두 페이지의 매칭 점수를 계산한다.
        """
 
        feature_score = (
            self._feature_similarity(
                before_page,
                after_page
            )
        )
 
        structure_score = (
            self._structure_similarity(
                before_page,
                after_page
            )
        )
 
        size_score = (
            self._size_similarity(
                before_page,
                after_page
            )
        )
 
        position_score = (
            self._position_similarity(
                before_page,
                after_page
            )
        )
 
        # ----------------------------------------------------
        # 점수 종합
        # ----------------------------------------------------
 
        total = (
            feature_score * 0.45
            +
            structure_score * 0.35
            +
            size_score * 0.10
            +
            position_score * 0.10
        )
 
        total = float(
            max(
                0.0,
                min(
                    1.0,
                    total
                )
            )
        )
 
        # ----------------------------------------------------
        # 상태 결정
        # ----------------------------------------------------
 
        if total >= self.minimum_score:
 
            status = "MATCH"
 
            reason = (
                "페이지 자동 매칭 가능"
            )
 
        elif total >= self.review_score:
 
            status = "REVIEW"
 
            reason = (
                "페이지 매칭 점수가 "
                "애매하여 검토 필요"
            )
 
        else:
 
            status = "NO_MATCH"
 
            reason = (
                "동일 페이지로 판단하기 "
                "어려움"
            )
 
        return PageMatch(
            before_page=before_page,
            after_page=after_page,
            score=total,
            feature_score=feature_score,
            structure_score=structure_score,
            size_score=size_score,
            position_score=position_score,
            status=status,
            reason=reason,
        )
 
 
    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================
 
    @staticmethod
    def _prepare_image(
        image: np.ndarray,
        max_size: int = 1200
    ) -> np.ndarray:
        """
        페이지 비교용 이미지로 변환한다.
 
        원본 이미지는 절대 변경하지 않는다.
        """
 
        if image is None:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        if image.ndim == 3:
 
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
 
        else:
 
            gray = image.copy()
 
        height, width = gray.shape[:2]
 
        if height == 0 or width == 0:
 
            return gray
 
        current_max = max(
            width,
            height
        )
 
        if current_max > max_size:
 
            scale = (
                max_size /
                current_max
            )
 
            width = max(
                1,
                int(width * scale)
            )
 
            height = max(
                1,
                int(height * scale)
            )
 
            gray = cv2.resize(
                gray,
                (width, height),
                interpolation=cv2.INTER_AREA
            )
 
        # 약한 노이즈 제거
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )
 
        return gray

    # ========================================================
    # FEATURE SIMILARITY
    # ========================================================
 
    def _feature_similarity(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> float:
        """
        두 페이지의 특징점 기반 유사도를 계산한다.
 
        도면의 크기가 달라도
        같은 형상/문자/선 구조가 존재하면
        높은 점수를 얻을 수 있도록 한다.
        """
 
        before = self._prepare_image(
            before_page.image
        )
 
        after = self._prepare_image(
            after_page.image
        )
 
        if before.size == 0:
            return 0.0
 
        if after.size == 0:
            return 0.0
 
        scores = []
 
        # ----------------------------------------------------
        # ORB
        # ----------------------------------------------------
 
        orb_score = self._orb_score(
            before,
            after
        )
 
        scores.append(
            orb_score
        )
 
        # ----------------------------------------------------
        # AKAZE
        # ----------------------------------------------------
 
        akaze_score = self._akaze_score(
            before,
            after
        )
 
        scores.append(
            akaze_score
        )
 
        if not scores:
 
            return 0.0
 
        return float(
            np.mean(scores)
        )
 
 
    # ========================================================
    # ORB SCORE
    # ========================================================
 
    def _orb_score(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        """
        ORB 특징점 기반 점수.
        """
 
        try:
 
            orb = cv2.ORB_create(
                nfeatures=5000,
                fastThreshold=10
            )
 
            kp1, des1 = (
                orb.detectAndCompute(
                    before,
                    None
                )
            )
 
            kp2, des2 = (
                orb.detectAndCompute(
                    after,
                    None
                )
            )
 
            if des1 is None:
                return 0.0
 
            if des2 is None:
                return 0.0
 
            if len(kp1) < 5:
                return 0.0
 
            if len(kp2) < 5:
                return 0.0
 
            matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False
            )
 
            matches = matcher.knnMatch(
                des1,
                des2,
                k=2
            )
 
            good_matches = []
 
            for pair in matches:
 
                if len(pair) < 2:
                    continue
 
                first = pair[0]
                second = pair[1]
 
                if (
                    first.distance
                    <
                    0.75 * second.distance
                ):
 
                    good_matches.append(
                        first
                    )
 
            match_count = len(
                good_matches
            )
 
            if match_count < (
                self.minimum_feature_matches
            ):
 
                return 0.0
 
            denominator = max(
                1,
                min(
                    len(kp1),
                    len(kp2)
                )
            )
 
            ratio = (
                match_count /
                denominator
            )
 
            # 지나치게 작은 특징점 수에 의해
            # 우연히 높은 점수가 나오지 않도록 제한한다.
 
            score = min(
                1.0,
                ratio * 8.0
            )
 
            return float(score)
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # AKAZE SCORE
    # ========================================================
 
    def _akaze_score(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        """
        AKAZE 특징점 기반 점수.
        """
 
        try:
 
            akaze = cv2.AKAZE_create()
 
            kp1, des1 = (
                akaze.detectAndCompute(
                    before,
                    None
                )
            )
 
            kp2, des2 = (
                akaze.detectAndCompute(
                    after,
                    None
                )
            )
 
            if des1 is None:
                return 0.0
 
            if des2 is None:
                return 0.0
 
            if len(kp1) < 5:
                return 0.0
 
            if len(kp2) < 5:
                return 0.0
 
            matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False
            )
 
            matches = matcher.knnMatch(
                des1,
                des2,
                k=2
            )
 
            good_matches = []
 
            for pair in matches:
 
                if len(pair) < 2:
                    continue
 
                first = pair[0]
                second = pair[1]
 
                if (
                    first.distance
                    <
                    0.75 * second.distance
                ):
 
                    good_matches.append(
                        first
                    )
 
            match_count = len(
                good_matches
            )
 
            if match_count < (
                self.minimum_feature_matches
            ):
 
                return 0.0
 
            denominator = max(
                1,
                min(
                    len(kp1),
                    len(kp2)
                )
            )
 
            ratio = (
                match_count /
                denominator
            )
 
            score = min(
                1.0,
                ratio * 8.0
            )
 
            return float(score)
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # STRUCTURE SIMILARITY
    # ========================================================
 
    def _structure_similarity(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> float:
        """
        도면의 전체적인 선/구조를 비교한다.
 
        단순 픽셀 차이가 아니라
        Edge 구조를 비교한다.
        """
 
        before = self._prepare_image(
            before_page.image,
            max_size=1000
        )
 
        after = self._prepare_image(
            after_page.image,
            max_size=1000
        )
 
        if before.size == 0:
            return 0.0
 
        if after.size == 0:
            return 0.0
 
        # 같은 크기로 정규화
        before = cv2.resize(
            before,
            (512, 512),
            interpolation=cv2.INTER_AREA
        )
 
        after = cv2.resize(
            after,
            (512, 512),
            interpolation=cv2.INTER_AREA
        )
 
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
 
        before_mask = (
            before_edges > 0
        )
 
        after_mask = (
            after_edges > 0
        )
 
        intersection = np.logical_and(
            before_mask,
            after_mask
        ).sum()
 
        union = np.logical_or(
            before_mask,
            after_mask
        ).sum()
 
        if union == 0:
 
            return 0.0
 
        iou = (
            intersection /
            union
        )
 
        return float(
            max(
                0.0,
                min(
                    1.0,
                    iou
                )
            )
        )
 
 
    # ========================================================
    # SIZE SIMILARITY
    # ========================================================
 
    @staticmethod
    def _size_similarity(
        before_page: PageImage,
        after_page: PageImage,
    ) -> float:
        """
        페이지의 종횡비를 비교한다.
 
        실제 크기가 달라도
        비율이 같으면 높은 점수를 준다.
        """
 
        before_ratio = (
            before_page.aspect_ratio
        )
 
        after_ratio = (
            after_page.aspect_ratio
        )
 
        if before_ratio <= 0:
            return 0.0
 
        if after_ratio <= 0:
            return 0.0
 
        difference = abs(
            before_ratio -
            after_ratio
        )
 
        base = max(
            before_ratio,
            after_ratio
        )
 
        score = 1.0 - (
            difference /
            base
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
    # POSITION SIMILARITY
    # ========================================================
 
    @staticmethod
    def _position_similarity(
        before_page: PageImage,
        after_page: PageImage,
    ) -> float:
        """
        페이지 내부의 기본 여백/도면 영역 비율을 비교한다.
 
        정확한 좌표 정렬은 aligner.py에서 수행한다.
        """
 
        before = before_page.image
        after = after_page.image
 
        if before is None:
            return 0.0
 
        if after is None:
            return 0.0
 
        try:
 
            before_gray = cv2.cvtColor(
                before,
                cv2.COLOR_BGR2GRAY
            )
 
            after_gray = cv2.cvtColor(
                after,
                cv2.COLOR_BGR2GRAY
            )
 
        except cv2.error:
 
            return 0.0
 
        before_ratio = (
            PageMatcher
            ._content_ratio(
                before_gray
            )
        )
 
        after_ratio = (
            PageMatcher
            ._content_ratio(
                after_gray
            )
        )
 
        difference = abs(
            before_ratio -
            after_ratio
        )
 
        return float(
            max(
                0.0,
                min(
                    1.0,
                    1.0 - difference
                )
            )
        )
 
 
    # ========================================================
    # CONTENT RATIO
    # ========================================================
 
    @staticmethod
    def _content_ratio(
        gray: np.ndarray
    ) -> float:
        """
        페이지에서 실제 도면/문자 등이 차지하는
        대략적인 비율을 계산한다.
        """
 
        if gray.size == 0:
 
            return 0.0
 
        _, binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )
 
        ratio = (
            np.count_nonzero(binary)
            /
            binary.size
        )
 
        return float(
            max(
                0.0,
                min(
                    1.0,
                    ratio
                )
            )
        )
 
     # ========================================================
    # SELECT PAGE MATCHES
    # ========================================================
 
    def _select_page_matches(
        self,
        candidates: List[PageMatch]
    ) -> List[PageMatch]:
        """
        모든 페이지 후보 중에서 최종 대응 페이지를 선택한다.
 
        핵심 원칙
        ------------------------------------------------------
        1. 하나의 Before 페이지는 하나의 After 페이지만 사용
        2. 하나의 After 페이지도 한 번만 사용
        3. 점수가 낮으면 매칭하지 않음
        4. 1위와 2위가 너무 비슷하면 REVIEW
        """
 
        if not candidates:
 
            return []
 
        # 점수가 높은 후보부터 정렬
        ordered = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )
 
        selected = []
 
        used_before = set()
        used_after = set()
 
        # ----------------------------------------------------
        # 1차: 확실한 MATCH
        # ----------------------------------------------------
 
        for candidate in ordered:
 
            if candidate.status != "MATCH":
                continue
 
            before_key = (
                candidate.before_page.page_index
            )
 
            after_key = (
                candidate.after_page.page_index
            )
 
            if before_key in used_before:
                continue
 
            if after_key in used_after:
                continue
 
            selected.append(
                candidate
            )
 
            used_before.add(
                before_key
            )
 
            used_after.add(
                after_key
            )
 
        # ----------------------------------------------------
        # 2차: REVIEW 후보
        # ----------------------------------------------------
 
        # 이미 MATCH된 페이지는 제외하고
        # 남은 페이지에 대해서만 REVIEW를 선택한다.
 
        for candidate in ordered:
 
            if candidate.status != "REVIEW":
                continue
 
            before_key = (
                candidate.before_page.page_index
            )
 
            after_key = (
                candidate.after_page.page_index
            )
 
            if before_key in used_before:
                continue
 
            if after_key in used_after:
                continue
 
            selected.append(
                candidate
            )
 
            used_before.add(
                before_key
            )
 
            used_after.add(
                after_key
            )
 
        # 페이지 번호 순서대로 반환
        selected.sort(
            key=lambda x: (
                x.before_page.page_index
            )
        )
 
        return selected
 
 
    # ========================================================
    # GET BEST CANDIDATES
    # ========================================================
 
    def get_best_candidates(
        self,
        before_page: PageImage,
        after_pages: List[PageImage],
        top_n: int = 3
    ) -> List[PageMatch]:
        """
        하나의 Before 페이지에 대해
        가장 가능성이 높은 After 페이지 후보를 반환한다.
        """
 
        candidates = []
 
        for after_page in after_pages:
 
            result = (
                self._calculate_page_match(
                    before_page,
                    after_page
                )
            )
 
            candidates.append(
                result
            )
 
        candidates.sort(
            key=lambda x: x.score,
            reverse=True
        )
 
        return candidates[
            :max(1, top_n)
        ]
 
 
    # ========================================================
    # MATCH ONE PAGE
    # ========================================================
 
    def match_one_page(
        self,
        before_page: PageImage,
        after_pages: List[PageImage]
    ) -> Optional[PageMatch]:
        """
        하나의 Before 페이지에 대해
        가장 적합한 After 페이지를 찾는다.
 
        확신이 낮으면 None.
        """
 
        candidates = (
            self.get_best_candidates(
                before_page,
                after_pages,
                top_n=3
            )
        )
 
        if not candidates:
 
            return None
 
        best = candidates[0]
 
        # ----------------------------------------------------
        # 1위와 2위의 차이가 작은 경우
        # ----------------------------------------------------
 
        if len(candidates) >= 2:
 
            gap = (
                candidates[0].score
                -
                candidates[1].score
            )
 
            # 두 후보가 거의 비슷하면
            # 잘못된 자동 매칭을 막는다.
            if gap < 0.10:
 
                best.status = "REVIEW"
 
                best.reason = (
                    "상위 페이지 후보의 "
                    "점수 차이가 작음"
                )
 
                return best
 
        # ----------------------------------------------------
        # 충분히 높은 경우
        # ----------------------------------------------------
 
        if best.score >= (
            self.minimum_score
        ):
 
            best.status = "MATCH"
 
            return best
 
        # ----------------------------------------------------
        # 애매한 경우
        # ----------------------------------------------------
 
        if best.score >= (
            self.review_score
        ):
 
            best.status = "REVIEW"
 
            best.reason = (
                "페이지 유사도는 있으나 "
                "자동 확정하기에는 부족함"
            )
 
            return best
 
        # ----------------------------------------------------
        # 매칭하지 않음
        # ----------------------------------------------------
 
        return None
 
 
    # ========================================================
    # CANDIDATE GAP
    # ========================================================
 
    @staticmethod
    def calculate_candidate_gap(
        candidates: List[PageMatch]
    ) -> float:
        """
        1위와 2위 후보의 점수 차이를 반환한다.
        """
 
        if len(candidates) < 2:
 
            return 1.0
 
        ordered = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )
 
        return float(
            ordered[0].score
            -
            ordered[1].score
        )
 
 
    # ========================================================
    # AMBIGUOUS CHECK
    # ========================================================
 
    @staticmethod
    def is_ambiguous(
        candidates: List[PageMatch],
        minimum_gap: float = 0.10
    ) -> bool:
        """
        상위 후보들이 서로 비슷하면
        애매한 매칭으로 판단한다.
        """
 
        if len(candidates) < 2:
 
            return False
 
        gap = (
            PageMatcher
            .calculate_candidate_gap(
                candidates
            )
        )
 
        return gap < minimum_gap
 
 
    # ========================================================
    # UNMATCHED PAGES
    # ========================================================
 
    @staticmethod
    def get_unmatched_pages(
        before: PDFDocument,
        after: PDFDocument,
        matches: List[PageMatch]
    ) -> dict:
        """
        매칭되지 않은 Before / After 페이지를 찾는다.
        """
 
        matched_before = {
            match.before_page.page_index
            for match in matches
        }
 
        matched_after = {
            match.after_page.page_index
            for match in matches
        }
 
        unmatched_before = [
            page
            for page in before.pages
            if page.page_index
            not in matched_before
        ]
 
        unmatched_after = [
            page
            for page in after.pages
            if page.page_index
            not in matched_after
        ]
 
        return {
            "before": unmatched_before,
            "after": unmatched_after,
        }
 
    # ========================================================
    # BUILD MATCH TABLE
    # ========================================================
 
    def build_match_table(
        self,
        before: PDFDocument,
        after: PDFDocument,
    ) -> dict:
        """
        두 PDF의 전체 페이지 매칭 결과를 생성한다.
 
        반환:
            matches
            reviews
            unmatched_before
            unmatched_after
        """
 
        if not before.pages:
            return {
                "matches": [],
                "reviews": [],
                "unmatched_before": [],
                "unmatched_after": [],
            }
 
        if not after.pages:
            return {
                "matches": [],
                "reviews": [],
                "unmatched_before": list(
                    before.pages
                ),
                "unmatched_after": [],
            }
 
        candidates = []
 
        # ----------------------------------------------------
        # 모든 페이지 조합 생성
        # ----------------------------------------------------
 
        for before_page in before.pages:
 
            for after_page in after.pages:
 
                result = (
                    self._calculate_page_match(
                        before_page,
                        after_page
                    )
                )
 
                candidates.append(
                    result
                )
 
        # ----------------------------------------------------
        # 최종 매칭
        # ----------------------------------------------------
 
        selected = (
            self._select_page_matches(
                candidates
            )
        )
 
        # ----------------------------------------------------
        # MATCH / REVIEW 분리
        # ----------------------------------------------------
 
        matches = []
 
        reviews = []
 
        for result in selected:
 
            if result.status == "MATCH":
 
                matches.append(
                    result
                )
 
            elif result.status == "REVIEW":
 
                reviews.append(
                    result
                )
 
        # ----------------------------------------------------
        # 매칭된 페이지 번호
        # ----------------------------------------------------
 
        matched_before = {
            result.before_page.page_index
            for result in selected
        }
 
        matched_after = {
            result.after_page.page_index
            for result in selected
        }
 
        # ----------------------------------------------------
        # 매칭되지 않은 페이지
        # ----------------------------------------------------
 
        unmatched_before = [
            page
            for page in before.pages
            if page.page_index
            not in matched_before
        ]
 
        unmatched_after = [
            page
            for page in after.pages
            if page.page_index
            not in matched_after
        ]
 
        return {
            "matches": matches,
            "reviews": reviews,
            "unmatched_before": unmatched_before,
            "unmatched_after": unmatched_after,
        }
 
 
    # ========================================================
    # MATCH SUMMARY
    # ========================================================
 
    @staticmethod
    def summarize(
        result: dict
    ) -> dict:
        """
        페이지 매칭 결과를 간단하게 요약한다.
        """
 
        return {
            "matched": len(
                result["matches"]
            ),
 
            "review": len(
                result["reviews"]
            ),
 
            "unmatched_before": len(
                result["unmatched_before"]
            ),
 
            "unmatched_after": len(
                result["unmatched_after"]
            ),
        }
 
 
    # ========================================================
    # MATCH TO DICT
    # ========================================================
 
    @staticmethod
    def match_to_dict(
        match: PageMatch
    ) -> dict:
        """
        PageMatch를 JSON / Excel 등에 사용할 수 있는
        dictionary 형태로 변환한다.
        """
 
        return {
            "before_page": (
                match.before_page.page_index
            ),
 
            "after_page": (
                match.after_page.page_index
            ),
 
            "before_width": (
                match.before_page.width
            ),
 
            "before_height": (
                match.before_page.height
            ),
 
            "after_width": (
                match.after_page.width
            ),
 
            "after_height": (
                match.after_page.height
            ),
 
            "score": round(
                match.score,
                4
            ),
 
            "feature_score": round(
                match.feature_score,
                4
            ),
 
            "structure_score": round(
                match.structure_score,
                4
            ),
 
            "size_score": round(
                match.size_score,
                4
            ),
 
            "position_score": round(
                match.position_score,
                4
            ),
 
            "status": match.status,
 
            "reason": match.reason,
        }
 
 
    # ========================================================
    # RESULT TO DICT
    # ========================================================
 
    def result_to_dict(
        self,
        result: dict
    ) -> dict:
        """
        전체 페이지 매칭 결과를
        JSON 저장 가능한 형태로 변환한다.
        """
 
        return {
            "matches": [
                self.match_to_dict(
                    match
                )
                for match
                in result["matches"]
            ],
 
            "reviews": [
                self.match_to_dict(
                    match
                )
                for match
                in result["reviews"]
            ],
 
            "unmatched_before": [
                page.page_index
                for page
                in result[
                    "unmatched_before"
                ]
            ],
 
            "unmatched_after": [
                page.page_index
                for page
                in result[
                    "unmatched_after"
                ]
            ],
        }
 
 
    # ========================================================
    # PAGE PAIR LIST
    # ========================================================
 
    @staticmethod
    def get_page_pairs(
        result: dict
    ) -> List[Tuple[int, int]]:
        """
        Align / Compare 단계에서 사용할
        페이지 번호 쌍을 반환한다.
 
        예:
            [
                (0, 2),
                (1, 0),
                (2, 1)
            ]
        """
 
        pairs = []
 
        for match in result["matches"]:
 
            pairs.append(
                (
                    match.before_page.page_index,
                    match.after_page.page_index,
                )
            )
 
        return pairs
 
 
# ============================================================
# DEFAULT MATCHER
# ============================================================
 
_default_page_matcher = PageMatcher()
 
 
def match_pages(
    before: PDFDocument,
    after: PDFDocument,
) -> List[PageMatch]:
    """
    외부 모듈에서 간단하게 페이지 매칭을 실행한다.
    """
 
    return _default_page_matcher.match_pages(
        before,
        after
    )
 
 
def build_match_table(
    before: PDFDocument,
    after: PDFDocument,
) -> dict:
    """
    전체 페이지 매칭 결과를 반환한다.
    """
 
    return _default_page_matcher.build_match_table(
        before,
        after
    )
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("DrawingCompare H5 - Page Matcher Test")
    print("=" * 60)
 
    print(
        "page_matcher.py 로드 성공"
    )
 
    print(
        f"Minimum Page Match Score : "
        f"{CONFIG.page_match.minimum_score}"
    )
 
    print(
        f"Review Score : "
        f"{CONFIG.page_match.review_score}"
    )
 
    print(
        f"Minimum Feature Matches : "
        f"{CONFIG.page_match.minimum_feature_matches}"
    )
 
    print("=" * 60)
 