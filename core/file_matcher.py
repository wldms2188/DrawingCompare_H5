"""
DrawingCompare H5
core/file_matcher.py
 
역할
------------------------------------------------------------
여러 개의 Before PDF와 After PDF를 자동으로 매칭한다.
 
매칭에 사용하는 정보
------------------------------------------------------------
1. 파일명 유사도
2. PDF 페이지 수
3. 페이지 크기 / 종횡비
4. 페이지 이미지 특징
5. OCR 기반 도면번호
6. 페이지 내용 유사도
 
중요
------------------------------------------------------------
- 파일명만으로 매칭하지 않는다.
- 하나의 After 파일을 여러 Before 파일에 중복 매칭하지 않는다.
- 점수가 낮으면 억지로 매칭하지 않는다.
- 애매한 경우 REVIEW 상태로 남긴다.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
 
import re
from difflib import SequenceMatcher
 
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
class FileMatch:
    """
    Before PDF와 After PDF의 매칭 결과
    """
 
    before: PDFDocument
 
    after: PDFDocument
 
    score: float
 
    filename_score: float
 
    page_count_score: float
 
    layout_score: float
 
    feature_score: float
 
    drawing_number_score: float
 
    status: str
 
    reason: str
 
 
# ============================================================
# FILE MATCHER
# ============================================================
 
class FileMatcher:
 
    def __init__(self, config=None):
 
        self.config = config or CONFIG
 
        self.minimum_score = (
            self.config.file_match.minimum_score
        )
 
        self.review_score = (
            self.config.file_match.review_score
        )
 
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def match_documents(
        self,
        before_documents: List[PDFDocument],
        after_documents: List[PDFDocument],
    ) -> List[FileMatch]:
        """
        여러 Before / After PDF를 자동 매칭한다.
        """
 
        if not before_documents:
 
            return []
 
        if not after_documents:
 
            return []
 
        candidates = []
 
        for before in before_documents:
 
            for after in after_documents:
 
                result = self._calculate_match(
                    before,
                    after
                )
 
                candidates.append(
                    result
                )
 
        return self._select_matches(
            candidates
        )
 
 
    # ========================================================
    # CALCULATE MATCH
    # ========================================================
 
    def _calculate_match(
        self,
        before: PDFDocument,
        after: PDFDocument
    ) -> FileMatch:
        """
        두 PDF의 매칭 점수를 계산한다.
        """
 
        filename_score = (
            self._filename_similarity(
                before.filename,
                after.filename
            )
        )
 
        page_count_score = (
            self._page_count_similarity(
                before.page_count,
                after.page_count
            )
        )
 
        layout_score = (
            self._layout_similarity(
                before,
                after
            )
        )
 
        feature_score = (
            self._feature_similarity(
                before,
                after
            )
        )
 
        drawing_number_score = (
            self._drawing_number_similarity(
                before,
                after
            )
        )
 
        weights = self.config.file_match
 
        total = (
            filename_score *
            weights.filename_weight
 
            +
 
            page_count_score *
            0.15
 
            +
 
            layout_score *
            weights.layout_weight
 
            +
 
            feature_score *
            weights.feature_weight
 
            +
 
            drawing_number_score *
            weights.drawing_no_weight
        )
 
        total = min(
            max(total, 0.0),
            1.0
        )
 
        if total >= self.minimum_score:
 
            status = "MATCH"
 
            reason = (
                "자동 매칭 가능"
            )
 
        elif total >= self.review_score:
 
            status = "REVIEW"
 
            reason = (
                "매칭 점수가 애매하여 "
                "검토 필요"
            )
 
        else:
 
            status = "NO_MATCH"
 
            reason = (
                "매칭 점수가 낮음"
            )
 
        return FileMatch(
            before=before,
            after=after,
            score=total,
            filename_score=filename_score,
            page_count_score=page_count_score,
            layout_score=layout_score,
            feature_score=feature_score,
            drawing_number_score=drawing_number_score,
            status=status,
            reason=reason,
        )
 
 
    # ========================================================
    # FILENAME SIMILARITY
    # ========================================================
 
    def _filename_similarity(
        self,
        before_name: str,
        after_name: str
    ) -> float:
        """
        파일명 유사도를 계산한다.
 
        단순 문자열 비교가 아니라
        revision/version 문자열을 제거하고 비교한다.
        """
 
        before = self._normalize_filename(
            before_name
        )
 
        after = self._normalize_filename(
            after_name
        )
 
        if not before or not after:
 
            return 0.0
 
        return SequenceMatcher(
            None,
            before,
            after
        ).ratio()
 
 
    # ========================================================
    # NORMALIZE FILENAME
    # ========================================================
 
    @staticmethod
    def _normalize_filename(
        filename: str
    ) -> str:
        """
        Revision / version 표기를 제거한다.
 
        예:
            drawing_A_v1.pdf
            drawing_A_rev2.pdf
 
        → drawing_a
        """
 
        name = Path(filename).stem.lower()
 
        patterns = [
            r"[_\-\s]?rev(?:ision)?[_\-\s]?\d+",
            r"[_\-\s]?v\d+(?:\.\d+)*",
            r"[_\-\s]?ver[_\-\s]?\d+",
            r"[_\-\s]?version[_\-\s]?\d+",
            r"[_\-\s]?final",
            r"[_\-\s]?new",
            r"[_\-\s]?old",
            r"[_\-\s]?copy",
        ]
 
        for pattern in patterns:
 
            name = re.sub(
                pattern,
                "",
                name,
                flags=re.IGNORECASE
            )
 
        name = re.sub(
            r"[^a-z0-9가-힣]+",
            "",
            name
        )
 
        return name
 
 
    # ========================================================
    # PAGE COUNT SIMILARITY
    # ========================================================
 
    @staticmethod
    def _page_count_similarity(
        before_count: int,
        after_count: int
    ) -> float:
        """
        페이지 수가 같으면 1.0.
 
        다르더라도 완전히 배제하지 않는다.
        """
 
        if before_count <= 0:
            return 0.0
 
        if after_count <= 0:
            return 0.0
 
        if before_count == after_count:
 
            return 1.0
 
        difference = abs(
            before_count -
            after_count
        )
 
        maximum = max(
            before_count,
            after_count
        )
 
        score = 1.0 - (
            difference / maximum
        )
 
        return max(
            0.0,
            score
        )
 
 
    # ========================================================
    # LAYOUT SIMILARITY
    # ========================================================
 
    def _layout_similarity(
        self,
        before: PDFDocument,
        after: PDFDocument
    ) -> float:
        """
        PDF 전체의 페이지 레이아웃 유사도를 계산한다.
        """
 
        if not before.pages:
            return 0.0
 
        if not after.pages:
            return 0.0
 
        before_ratios = [
            page.aspect_ratio
            for page in before.pages
        ]
 
        after_ratios = [
            page.aspect_ratio
            for page in after.pages
        ]
 
        # 페이지 수가 다르면
        # 앞쪽 몇 페이지만 비교하지 않고
        # 대표적인 비율 분포를 비교한다.
 
        before_mean = float(
            np.mean(before_ratios)
        )
 
        after_mean = float(
            np.mean(after_ratios)
        )
 
        difference = abs(
            before_mean -
            after_mean
        )
 
        base = max(
            abs(before_mean),
            abs(after_mean),
            1e-6
        )
 
        score = 1.0 - (
            difference / base
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
    # FEATURE SIMILARITY
    # ========================================================
 
    def _feature_similarity(
        self,
        before: PDFDocument,
        after: PDFDocument
    ) -> float:
        """
        PDF의 실제 페이지 이미지 특징을 비교한다.
 
        페이지 순서가 달라도 대응 가능한 페이지를
        찾을 수 있도록 모든 후보 조합을 검사한다.
 
        중요한 점:
        파일명보다 실제 도면 이미지 특징을 더 신뢰한다.
        """
 
        if not before.pages or not after.pages:
            return 0.0
 
        before_pages = before.pages
        after_pages = after.pages
 
        page_scores = []
 
        for before_page in before_pages:
 
            best_score = 0.0
 
            for after_page in after_pages:
 
                score = self._page_feature_similarity(
                    before_page,
                    after_page
                )
 
                if score > best_score:
 
                    best_score = score
 
            page_scores.append(
                best_score
            )
 
        if not page_scores:
 
            return 0.0
 
        # 일부 페이지가 완전히 다른 경우에도
        # 평균이 지나치게 낮아지지 않도록
        # 가장 신뢰할 수 있는 페이지들을 중심으로 계산한다.
 
        sorted_scores = sorted(
            page_scores,
            reverse=True
        )
 
        keep_count = max(
            1,
            int(len(sorted_scores) * 0.70)
        )
 
        selected = sorted_scores[
            :keep_count
        ]
 
        return float(
            np.mean(selected)
        )
 
 
    # ========================================================
    # PAGE FEATURE SIMILARITY
    # ========================================================
 
    def _page_feature_similarity(
        self,
        before_page: PageImage,
        after_page: PageImage
    ) -> float:
        """
        두 페이지의 실제 이미지 특징을 비교한다.
        """
 
        before_image = (
            before_page.image
        )
 
        after_image = (
            after_page.image
        )
 
        if before_image is None:
            return 0.0
 
        if after_image is None:
            return 0.0
 
        try:
 
            before_gray = cv2.cvtColor(
                before_image,
                cv2.COLOR_BGR2GRAY
            )
 
            after_gray = cv2.cvtColor(
                after_image,
                cv2.COLOR_BGR2GRAY
            )
 
        except cv2.error:
 
            return 0.0
 
        # 비교 속도를 위해 축소
        before_gray = self._resize_for_match(
            before_gray
        )
 
        after_gray = self._resize_for_match(
            after_gray
        )
 
        scores = []
 
        # ----------------------------------------------------
        # ORB
        # ----------------------------------------------------
 
        if self.config.align.use_orb:
 
            orb_score = (
                self._orb_similarity(
                    before_gray,
                    after_gray
                )
            )
 
            scores.append(
                orb_score
            )
 
        # ----------------------------------------------------
        # AKAZE
        # ----------------------------------------------------
 
        if self.config.align.use_akaze:
 
            akaze_score = (
                self._akaze_similarity(
                    before_gray,
                    after_gray
                )
            )
 
            scores.append(
                akaze_score
            )
 
        # ----------------------------------------------------
        # 구조적 유사도
        # ----------------------------------------------------
 
        layout_score = (
            self._image_structure_similarity(
                before_gray,
                after_gray
            )
        )
 
        scores.append(
            layout_score
        )
 
        if not scores:
 
            return 0.0
 
        # 가장 높은 특징값만 사용하는 것이 아니라
        # 전체 결과를 종합한다.
 
        return float(
            np.mean(scores)
        )
 
 
    # ========================================================
    # RESIZE FOR MATCH
    # ========================================================
 
    @staticmethod
    def _resize_for_match(
        image: np.ndarray,
        max_size: int = 1200
    ) -> np.ndarray:
        """
        특징점 비교용 이미지 크기를 줄인다.
        """
 
        height, width = image.shape[:2]
 
        current_max = max(
            width,
            height
        )
 
        if current_max <= max_size:
 
            return image
 
        scale = (
            max_size /
            current_max
        )
 
        new_width = max(
            int(width * scale),
            1
        )
 
        new_height = max(
            int(height * scale),
            1
        )
 
        return cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )
 
 
    # ========================================================
    # ORB SIMILARITY
    # ========================================================
 
    def _orb_similarity(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        ORB 특징점을 이용한 유사도.
        """
 
        try:
 
            orb = cv2.ORB_create(
                nfeatures=5000
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
 
            if des1 is None or des2 is None:
 
                return 0.0
 
            if len(kp1) < 5 or len(kp2) < 5:
 
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
 
                m, n = pair
 
                if m.distance < (
                    0.75 * n.distance
                ):
 
                    good_matches.append(m)
 
            denominator = max(
                1,
                min(
                    len(kp1),
                    len(kp2)
                )
            )
 
            score = (
                len(good_matches) /
                denominator
            )
 
            return float(
                min(
                    1.0,
                    score * 5.0
                )
            )
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # AKAZE SIMILARITY
    # ========================================================
 
    def _akaze_similarity(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        AKAZE 특징점을 이용한 유사도.
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
 
            if des1 is None or des2 is None:
 
                return 0.0
 
            if len(kp1) < 5 or len(kp2) < 5:
 
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
 
                m, n = pair
 
                if m.distance < (
                    0.75 * n.distance
                ):
 
                    good_matches.append(m)
 
            denominator = max(
                1,
                min(
                    len(kp1),
                    len(kp2)
                )
            )
 
            score = (
                len(good_matches) /
                denominator
            )
 
            return float(
                min(
                    1.0,
                    score * 5.0
                )
            )
 
        except cv2.error:
 
            return 0.0
 
 
    # ========================================================
    # IMAGE STRUCTURE SIMILARITY
    # ========================================================
 
    def _image_structure_similarity(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """
        전체적인 도면 구조를 비교한다.
 
        픽셀 위치가 완전히 동일해야 하는 방식이 아니라
        축소된 구조를 비교한다.
        """
 
        try:
 
            before_small = cv2.resize(
                before,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )
 
            after_small = cv2.resize(
                after,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )
 
            before_edges = cv2.Canny(
                before_small,
                50,
                150
            )
 
            after_edges = cv2.Canny(
                after_small,
                50,
                150
            )
 
            intersection = np.logical_and(
                before_edges > 0,
                after_edges > 0
            ).sum()
 
            union = np.logical_or(
                before_edges > 0,
                after_edges > 0
            ).sum()
 
            if union == 0:
 
                return 0.0
 
            return float(
                intersection / union
            )
 
        except cv2.error:
 
            return 0.0
  
    # ========================================================
    # DRAWING NUMBER SIMILARITY
    # ========================================================
 
    def _drawing_number_similarity(
        self,
        before: PDFDocument,
        after: PDFDocument
    ) -> float:
        """
        도면번호 기반 유사도.
 
        현재 단계에서는 PDF 파일명과 페이지의
        OCR/텍스트 정보를 활용할 수 있도록 구조를 만든다.
 
        OCR 자체는 OCR 모듈에서 수행하고,
        여기서는 파일명에서 추출 가능한 도면번호를
        우선 사용한다.
        """
 
        before_numbers = (
            self._extract_drawing_numbers(
                before
            )
        )
 
        after_numbers = (
            self._extract_drawing_numbers(
                after
            )
        )
 
        if not before_numbers:
            return 0.0
 
        if not after_numbers:
            return 0.0
 
        best_score = 0.0
 
        for before_number in before_numbers:
 
            for after_number in after_numbers:
 
                score = SequenceMatcher(
                    None,
                    before_number,
                    after_number
                ).ratio()
 
                if (
                    before_number ==
                    after_number
                ):
 
                    score = 1.0
 
                best_score = max(
                    best_score,
                    score
                )
 
        return float(
            best_score
        )
 
 
    # ========================================================
    # EXTRACT DRAWING NUMBERS
    # ========================================================
 
    def _extract_drawing_numbers(
        self,
        document: PDFDocument
    ) -> List[str]:
        """
        파일명에서 도면번호 후보를 추출한다.
 
        예:
            ABC-123_rev2.pdf
            ABC_123_v3.pdf
 
        → ABC123
 
        숫자만으로 구성된 값도 후보로 보존한다.
        """
 
        filename = (
            Path(document.filename)
            .stem
            .upper()
        )
 
        # revision / version 제거
        filename = re.sub(
            r"[_\-\s]?REV(?:ISION)?[_\-\s]?\d+",
            "",
            filename
        )
 
        filename = re.sub(
            r"[_\-\s]?V\d+(?:\.\d+)*",
            "",
            filename
        )
 
        filename = re.sub(
            r"[_\-\s]?VER[_\-\s]?\d+",
            "",
            filename
        )
 
        candidates = []
 
        # ----------------------------------------------------
        # 영문 + 숫자 조합
        # ----------------------------------------------------
 
        alphanumeric = re.findall(
            r"[A-Z]{1,8}[-_]?\d{2,12}",
            filename
        )
 
        for value in alphanumeric:
 
            normalized = re.sub(
                r"[^A-Z0-9]",
                "",
                value
            )
 
            if len(normalized) >= 3:
 
                candidates.append(
                    normalized
                )
 
        # ----------------------------------------------------
        # 숫자 조합
        # ----------------------------------------------------
 
        numbers = re.findall(
            r"\d{4,12}",
            filename
        )
 
        for value in numbers:
 
            if value not in candidates:
 
                candidates.append(
                    value
                )
 
        return list(
            dict.fromkeys(
                candidates
            )
        )
 
 
    # ========================================================
    # SELECT MATCHES
    # ========================================================
 
    def _select_matches(
        self,
        candidates: List[FileMatch]
    ) -> List[FileMatch]:
        """
        모든 후보 중에서 최종 매칭을 선택한다.
 
        핵심:
        ------------------------------------------------------
        하나의 After PDF가 여러 Before PDF에
        중복으로 매칭되지 않도록 한다.
 
        또한 낮은 점수의 후보를 억지로 선택하지 않는다.
        """
 
        if not candidates:
 
            return []
 
        # 점수가 높은 후보부터 정렬
        candidates = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )
 
        selected = []
 
        used_before = set()
        used_after = set()
 
        for candidate in candidates:
 
            before_key = str(
                candidate.before.path.resolve()
            )
 
            after_key = str(
                candidate.after.path.resolve()
            )
 
            # 이미 매칭된 파일은 건너뜀
            if before_key in used_before:
                continue
 
            if after_key in used_after:
                continue
 
            # 낮은 점수는 자동 매칭하지 않음
            if candidate.status == "NO_MATCH":
 
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
        # REVIEW 항목은 MATCH와 함께 반환한다.
        # 호출 측에서 사람이 확인할 수 있도록 한다.
        # ----------------------------------------------------
 
        return selected
 
 
    # ========================================================
    # GET BEST CANDIDATES
    # ========================================================
 
    def get_best_candidates(
        self,
        before: PDFDocument,
        after_documents: List[PDFDocument],
        top_n: int = 3
    ) -> List[FileMatch]:
        """
        하나의 Before PDF에 대해
        가장 가능성이 높은 After PDF 후보를 반환한다.
 
        자동 매칭이 애매할 때 사용한다.
        """
 
        candidates = []
 
        for after in after_documents:
 
            candidate = (
                self._calculate_match(
                    before,
                    after
                )
            )
 
            candidates.append(
                candidate
            )
 
        candidates.sort(
            key=lambda x: x.score,
            reverse=True
        )
 
        return candidates[:max(1, top_n)]
 
 
    # ========================================================
    # MATCH ONE DOCUMENT
    # ========================================================
 
    def match_one(
        self,
        before: PDFDocument,
        after_documents: List[PDFDocument]
    ) -> Optional[FileMatch]:
        """
        하나의 Before PDF에 대해
        가장 적합한 After PDF를 찾는다.
 
        확신이 낮으면 None을 반환한다.
        """
 
        candidates = self.get_best_candidates(
            before,
            after_documents,
            top_n=3
        )
 
        if not candidates:
 
            return None
 
        best = candidates[0]
 
        # 자동 확정
        if best.score >= self.minimum_score:
 
            return best
 
        # REVIEW 영역은 반환하되
        # status를 REVIEW로 유지한다.
        if best.score >= self.review_score:
 
            best.status = "REVIEW"
 
            return best
 
        # 그 이하라면 매칭하지 않는다.
        return None
 
 
    # ========================================================
    # MATCH SCORE GAP
    # ========================================================
 
    def calculate_candidate_gap(
        self,
        candidates: List[FileMatch]
    ) -> float:
        """
        1위와 2위 후보의 점수 차이를 계산한다.
 
        점수 차이가 너무 작으면
        "애매한 매칭"으로 판단할 수 있다.
        """
 
        if len(candidates) < 2:
 
            return 1.0
 
        ordered = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )
 
        return float(
            ordered[0].score -
            ordered[1].score
        )
 
 
    # ========================================================
    # AMBIGUOUS MATCH CHECK
    # ========================================================
 
    def is_ambiguous(
        self,
        candidates: List[FileMatch],
        minimum_gap: float = 0.10
    ) -> bool:
        """
        1위와 2위 후보가 너무 비슷하면
        자동 매칭하지 않도록 한다.
        """
 
        if len(candidates) < 2:
 
            return False
 
        gap = self.calculate_candidate_gap(
            candidates
        )
 
        return gap < minimum_gap
 
    # ========================================================
    # BUILD MATCH TABLE
    # ========================================================
 
    def build_match_table(
        self,
        before_documents: List[PDFDocument],
        after_documents: List[PDFDocument],
    ) -> dict:
        """
        Before / After 전체 파일의 매칭 결과를
        프로그램 전체에서 사용하기 쉬운 형태로 정리한다.
 
        결과:
            MATCH
            REVIEW
            NO_MATCH
        """
 
        all_candidates = []
 
        for before in before_documents:
 
            for after in after_documents:
 
                result = self._calculate_match(
                    before,
                    after
                )
 
                all_candidates.append(
                    result
                )
 
        matches = []
        reviews = []
        no_matches = []
 
        # ----------------------------------------------------
        # 각 Before 파일별 후보를 모은다.
        # ----------------------------------------------------
 
        before_groups = {}
 
        for candidate in all_candidates:
 
            key = str(
                candidate.before.path.resolve()
            )
 
            before_groups.setdefault(
                key,
                []
            ).append(
                candidate
            )
 
        # ----------------------------------------------------
        # 각 Before마다 가장 좋은 후보를 찾는다.
        # ----------------------------------------------------
 
        for before_key, candidates in (
            before_groups.items()
        ):
 
            candidates.sort(
                key=lambda x: x.score,
                reverse=True
            )
 
            best = candidates[0]
 
            # 후보가 여러 개이고
            # 점수 차이가 작으면 REVIEW
            if self.is_ambiguous(
                candidates
            ):
 
                best.status = "REVIEW"
 
                best.reason = (
                    "상위 후보 간 점수 차이가 "
                    "작아 자동 매칭하지 않음"
                )
 
            if best.status == "MATCH":
 
                matches.append(
                    best
                )
 
            elif best.status == "REVIEW":
 
                reviews.append(
                    best
                )
 
            else:
 
                no_matches.append(
                    best
                )
 
        # ----------------------------------------------------
        # 중복 After 매칭 방지
        # ----------------------------------------------------
 
        matches = self._remove_duplicate_matches(
            matches
        )
 
        return {
            "matches": matches,
            "reviews": reviews,
            "no_matches": no_matches,
            "candidate_count": len(
                all_candidates
            ),
        }
 
 
    # ========================================================
    # REMOVE DUPLICATE MATCHES
    # ========================================================
 
    def _remove_duplicate_matches(
        self,
        matches: List[FileMatch]
    ) -> List[FileMatch]:
        """
        하나의 After PDF가 여러 Before PDF에
        매칭되는 것을 방지한다.
 
        점수가 가장 높은 매칭을 남긴다.
        """
 
        matches = sorted(
            matches,
            key=lambda x: x.score,
            reverse=True
        )
 
        result = []
 
        used_before = set()
        used_after = set()
 
        for match in matches:
 
            before_key = str(
                match.before.path.resolve()
            )
 
            after_key = str(
                match.after.path.resolve()
            )
 
            if before_key in used_before:
 
                continue
 
            if after_key in used_after:
 
                continue
 
            result.append(
                match
            )
 
            used_before.add(
                before_key
            )
 
            used_after.add(
                after_key
            )
 
        return result
 
 
    # ========================================================
    # CONVERT MATCH TO DICT
    # ========================================================
 
    @staticmethod
    def match_to_dict(
        match: FileMatch
    ) -> dict:
        """
        FileMatch를 JSON / Excel 등에 사용할 수 있는
        dictionary 형태로 변환한다.
        """
 
        return {
            "before_file": (
                match.before.filename
            ),
            "after_file": (
                match.after.filename
            ),
            "before_path": str(
                match.before.path
            ),
            "after_path": str(
                match.after.path
            ),
            "before_pages": (
                match.before.page_count
            ),
            "after_pages": (
                match.after.page_count
            ),
            "score": round(
                match.score,
                4
            ),
            "filename_score": round(
                match.filename_score,
                4
            ),
            "page_count_score": round(
                match.page_count_score,
                4
            ),
            "layout_score": round(
                match.layout_score,
                4
            ),
            "feature_score": round(
                match.feature_score,
                4
            ),
            "drawing_number_score": round(
                match.drawing_number_score,
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
        전체 매칭 결과를 JSON으로 저장 가능한 형태로 변환한다.
        """
 
        return {
            "candidate_count": (
                result["candidate_count"]
            ),
 
            "matches": [
                self.match_to_dict(match)
                for match in result["matches"]
            ],
 
            "reviews": [
                self.match_to_dict(match)
                for match in result["reviews"]
            ],
 
            "no_matches": [
                self.match_to_dict(match)
                for match in result["no_matches"]
            ],
        }
 
 
    # ========================================================
    # SUMMARY
    # ========================================================
 
    @staticmethod
    def summarize(
        result: dict
    ) -> dict:
        """
        매칭 결과를 간단하게 요약한다.
        """
 
        return {
            "matched": len(
                result["matches"]
            ),
            "review": len(
                result["reviews"]
            ),
            "no_match": len(
                result["no_matches"]
            ),
            "candidate_count": (
                result["candidate_count"]
            ),
        }
 
 
# ============================================================
# DEFAULT MATCHER
# ============================================================
 
_default_matcher = FileMatcher()
 
 
def match_documents(
    before_documents: List[PDFDocument],
    after_documents: List[PDFDocument],
) -> List[FileMatch]:
    """
    외부 모듈에서 바로 사용할 수 있는 함수.
    """
 
    return _default_matcher.match_documents(
        before_documents,
        after_documents
    )
 
 
def build_match_table(
    before_documents: List[PDFDocument],
    after_documents: List[PDFDocument],
) -> dict:
    """
    전체 Before / After 파일을 자동 매칭한다.
    """
 
    return _default_matcher.build_match_table(
        before_documents,
        after_documents
    )
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("DrawingCompare H5 - File Matcher Test")
    print("=" * 60)
 
    print(
        "file_matcher.py 로드 성공"
    )
 
    print(
        f"Minimum Match Score : "
        f"{CONFIG.file_match.minimum_score}"
    )
 
    print(
        f"Review Score : "
        f"{CONFIG.file_match.review_score}"
    )
 
    print(
        f"Duplicate Prevention : "
        f"{CONFIG.file_match.prevent_duplicate_match}"
    )
 
    print("=" * 60)
 