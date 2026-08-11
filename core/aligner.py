"""
DrawingCompare H5
core/aligner.py
 
역할
------------------------------------------------------------
Before / After 도면 페이지의 크기와 위치가 달라도
자동으로 같은 좌표계로 정렬한다.
 
처리 순서
------------------------------------------------------------
1. 이미지 전처리
2. 특징점 탐색
3. 특징점 대응
4. Homography 계산
5. Perspective 변환
6. 정렬 품질 검사
7. 실패 시 안전하게 REVIEW 처리
 
중요
------------------------------------------------------------
- 원본 이미지는 절대 수정하지 않는다.
- 정렬 결과는 별도의 이미지로 만든다.
- 잘못된 정렬을 억지로 적용하지 않는다.
- AI 없이 OpenCV만 사용한다.
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
class AlignmentResult:
    """
    Auto Align 결과
    """
 
    success: bool
 
    aligned_image: Optional[np.ndarray]
 
    homography: Optional[np.ndarray]
 
    feature_matches: int
 
    inlier_matches: int
 
    alignment_score: float
 
    scale_x: float
 
    scale_y: float
 
    rotation: float
 
    reason: str
 
 
# ============================================================
# ALIGNER
# ============================================================
 
class Aligner:
 
    def __init__(self, config=None):
 
        self.config = config or CONFIG
 
        self.max_features = (
            self.config.align.max_features
        )
 
        self.ratio_test = (
            self.config.align.ratio_test
        )
 
        self.minimum_matches = (
            self.config.align.minimum_matches
        )
 
        self.minimum_inliers = (
            self.config.align.minimum_inliers
        )
 
        self.ransac_threshold = (
            self.config.align.ransac_threshold
        )
 
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def align(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> AlignmentResult:
        """
        Before 페이지를 기준 좌표계로 사용하여
        After 페이지를 자동 정렬한다.
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
        # 이미지 전처리
        # ----------------------------------------------------
 
        before_gray = self._prepare(
            before
        )
 
        after_gray = self._prepare(
            after
        )
 
        # ----------------------------------------------------
        # 특징점 검출
        # ----------------------------------------------------
 
        keypoints_before, descriptors_before = (
            self._detect_features(
                before_gray
            )
        )
 
        keypoints_after, descriptors_after = (
            self._detect_features(
                after_gray
            )
        )
 
        if descriptors_before is None:
 
            return self._failed(
                "Before 특징점을 찾지 못함"
            )
 
        if descriptors_after is None:
 
            return self._failed(
                "After 특징점을 찾지 못함"
            )
 
        # ----------------------------------------------------
        # 특징점 매칭
        # ----------------------------------------------------
 
        good_matches = (
            self._match_features(
                descriptors_before,
                descriptors_after
            )
        )
 
        feature_match_count = len(
            good_matches
        )
 
        if feature_match_count < (
            self.minimum_matches
        ):
 
            return self._failed(
                f"특징점 매칭 부족 "
                f"({feature_match_count})",
                feature_matches=(
                    feature_match_count
                )
            )
 
        # ----------------------------------------------------
        # Homography 계산
        # ----------------------------------------------------
 
        homography, mask = (
            self._find_homography(
                keypoints_before,
                keypoints_after,
                good_matches
            )
        )
 
        if homography is None:
 
            return self._failed(
                "Homography 계산 실패",
                feature_matches=(
                    feature_match_count
                )
            )
 
        # ----------------------------------------------------
        # Inlier 계산
        # ----------------------------------------------------
 
        inlier_count = 0
 
        if mask is not None:
 
            inlier_count = int(
                np.sum(mask)
            )
 
        if inlier_count < (
            self.minimum_inliers
        ):
 
            return self._failed(
                f"유효 특징점 부족 "
                f"({inlier_count})",
                feature_matches=(
                    feature_match_count
                ),
                inlier_matches=(
                    inlier_count
                ),
                homography=homography
            )
 
        # ----------------------------------------------------
        # 정렬 품질 확인
        # ----------------------------------------------------
 
        alignment_score = (
            self._alignment_score(
                feature_match_count,
                inlier_count
            )
        )
 
        if alignment_score < 0.50:
 
            return self._failed(
                "정렬 신뢰도가 낮음",
                feature_matches=(
                    feature_match_count
                ),
                inlier_matches=(
                    inlier_count
                ),
                homography=homography
            )
 
        # ----------------------------------------------------
        # After → Before 좌표계 변환
        # ----------------------------------------------------
 
        aligned = self._warp_to_before(
            after,
            homography,
            before.shape
        )
 
        if aligned is None:
 
            return self._failed(
                "이미지 정렬 변환 실패",
                feature_matches=(
                    feature_match_count
                ),
                inlier_matches=(
                    inlier_count
                ),
                homography=homography
            )
 
        # ----------------------------------------------------
        # 변환 정보 추출
        # ----------------------------------------------------
 
        scale_x, scale_y, rotation = (
            self._extract_transform(
                homography
            )
        )
 
        return AlignmentResult(
            success=True,
            aligned_image=aligned,
            homography=homography,
            feature_matches=feature_match_count,
            inlier_matches=inlier_count,
            alignment_score=alignment_score,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation,
            reason="자동 정렬 성공",
        )
 
 
    # ========================================================
    # PREPARE
    # ========================================================
 
    @staticmethod
    def _prepare(
        image: np.ndarray
    ) -> np.ndarray:
        """
        특징점 검출용 grayscale 이미지.
        """
 
        if image.ndim == 3:
 
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
 
        else:
 
            gray = image.copy()
 
        # 너무 큰 이미지는 축소
        height, width = gray.shape[:2]
 
        max_size = 1600
 
        current_max = max(
            width,
            height
        )
 
        if current_max > max_size:
 
            scale = (
                max_size /
                current_max
            )
 
            new_width = max(
                1,
                int(width * scale)
            )
 
            new_height = max(
                1,
                int(height * scale)
            )
 
            gray = cv2.resize(
                gray,
                (
                    new_width,
                    new_height
                ),
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
    # FEATURE DETECTION
    # ========================================================
 
    def _detect_features(
        self,
        image: np.ndarray
    ) -> Tuple[
        List[cv2.KeyPoint],
        Optional[np.ndarray]
    ]:
        """
        도면에서 특징점을 검출한다.
 
        ORB를 기본으로 사용한다.
        AI나 외부 서버는 사용하지 않는다.
        """
 
        try:
 
            orb = cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                fastThreshold=10,
            )
 
            keypoints, descriptors = (
                orb.detectAndCompute(
                    image,
                    None
                )
            )
 
            if keypoints is None:
 
                return [], None
 
            if len(keypoints) < 5:
 
                return keypoints, None
 
            return (
                keypoints,
                descriptors
            )
 
        except cv2.error:
 
            return [], None
 
 
    # ========================================================
    # FEATURE MATCHING
    # ========================================================
 
    def _match_features(
        self,
        descriptors_before: np.ndarray,
        descriptors_after: np.ndarray,
    ) -> List[cv2.DMatch]:
        """
        Before / After 특징점을 매칭한다.
 
        Lowe Ratio Test를 사용해서
        우연히 일치하는 특징점을 제거한다.
        """
 
        if descriptors_before is None:
 
            return []
 
        if descriptors_after is None:
 
            return []
 
        try:
 
            matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False
            )
 
            matches = matcher.knnMatch(
                descriptors_after,
                descriptors_before,
                k=2
            )
 
        except cv2.error:
 
            return []
 
        good_matches = []
 
        for pair in matches:
 
            if len(pair) < 2:
 
                continue
 
            first = pair[0]
            second = pair[1]
 
            if (
                first.distance
                <
                self.ratio_test *
                second.distance
            ):
 
                good_matches.append(
                    first
                )
 
        # ----------------------------------------------------
        # 거리 기준으로 추가 정리
        # ----------------------------------------------------
 
        if len(good_matches) > 0:
 
            distances = np.array(
                [
                    match.distance
                    for match
                    in good_matches
                ],
                dtype=np.float32
            )
 
            median_distance = (
                float(
                    np.median(
                        distances
                    )
                )
            )
 
            # 지나치게 먼 매칭 제거
            max_distance = max(
                40.0,
                median_distance * 2.5
            )
 
            good_matches = [
                match
                for match
                in good_matches
                if match.distance
                <= max_distance
            ]
 
        # ----------------------------------------------------
        # 좋은 매칭을 우선순위로 정렬
        # ----------------------------------------------------
 
        good_matches.sort(
            key=lambda x: x.distance
        )
 
        return good_matches
 
 
    # ========================================================
    # HOMOGRAPHY
    # ========================================================
 
    def _find_homography(
        self,
        keypoints_before: List[cv2.KeyPoint],
        keypoints_after: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray]
    ]:
        """
        특징점 대응관계로 Homography를 계산한다.
 
        After → Before 변환을 구한다.
        """
 
        if len(matches) < 4:
 
            return None, None
 
        points_after = np.float32(
            [
                keypoints_after[
                    match.queryIdx
                ].pt
                for match in matches
            ]
        )
 
        points_before = np.float32(
            [
                keypoints_before[
                    match.trainIdx
                ].pt
                for match in matches
            ]
        )
 
        if len(points_after) < 4:
 
            return None, None
 
        if len(points_before) < 4:
 
            return None, None
 
        try:
 
            homography, mask = (
                cv2.findHomography(
                    points_after,
                    points_before,
                    cv2.RANSAC,
                    self.ransac_threshold
                )
            )
 
        except cv2.error:
 
            return None, None
 
        if homography is None:
 
            return None, None
 
        if mask is None:
 
            return homography, None
 
        # ----------------------------------------------------
        # Homography 자체의 안정성 검사
        # ----------------------------------------------------
 
        if not self._is_valid_homography(
            homography
        ):
 
            return None, None
 
        return homography, mask
 
 
    # ========================================================
    # HOMOGRAPHY VALIDATION
    # ========================================================
 
    @staticmethod
    def _is_valid_homography(
        homography: np.ndarray
    ) -> bool:
        """
        계산된 Homography가 비정상적으로
        큰 확대/축소나 왜곡을 만들지 않는지 검사한다.
        """
 
        if homography is None:
 
            return False
 
        if homography.shape != (3, 3):
 
            return False
 
        if not np.all(
            np.isfinite(
                homography
            )
        ):
 
            return False
 
        determinant = np.linalg.det(
            homography
        )
 
        if not np.isfinite(
            determinant
        ):
 
            return False
 
        if abs(determinant) < 1e-8:
 
            return False
 
        # ----------------------------------------------------
        # 너무 큰 원근 왜곡 방지
        # ----------------------------------------------------
 
        perspective_x = abs(
            homography[2, 0]
        )
 
        perspective_y = abs(
            homography[2, 1]
        )
 
        if perspective_x > 0.01:
 
            return False
 
        if perspective_y > 0.01:
 
            return False
 
        return True
 
 
    # ========================================================
    # ALIGNMENT SCORE
    # ========================================================
 
    @staticmethod
    def _alignment_score(
        feature_matches: int,
        inlier_matches: int
    ) -> float:
        """
        특징점 매칭 대비 실제로 같은 변환을 따르는
        Inlier 비율을 이용해 정렬 신뢰도를 계산한다.
        """
 
        if feature_matches <= 0:
 
            return 0.0
 
        if inlier_matches <= 0:
 
            return 0.0
 
        ratio = (
            inlier_matches /
            feature_matches
        )
 
        # Inlier 비율을 기반으로 점수화
        score = min(
            1.0,
            ratio
        )
 
        return float(score)
 
 
    # ========================================================
    # WARP
    # ========================================================
 
    @staticmethod
    def _warp_to_before(
        after: np.ndarray,
        homography: np.ndarray,
        before_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        After 이미지를 Before 이미지 좌표계로 변환한다.
        """
 
        if after is None:
 
            return None
 
        if homography is None:
 
            return None
 
        if len(before_shape) < 2:
 
            return None
 
        height = before_shape[0]
        width = before_shape[1]
 
        if height <= 0:
 
            return None
 
        if width <= 0:
 
            return None
 
        try:
 
            aligned = cv2.warpPerspective(
                after,
                homography,
                (
                    width,
                    height
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
 
        except cv2.error:
 
            return None
 
        if aligned is None:
 
            return None
 
        if aligned.size == 0:
 
            return None
 
        return aligned
 
 
    # ========================================================
    # EXTRACT TRANSFORM
    # ========================================================
 
    @staticmethod
    def _extract_transform(
        homography: np.ndarray
    ) -> Tuple[
        float,
        float,
        float
    ]:
        """
        Homography에서 대략적인
        X/Y Scale과 회전각을 추출한다.
 
        참고:
        실제 비교에서는 Homography 전체를 사용하고,
        이 값들은 결과 표시용이다.
        """
 
        if homography is None:
 
            return (
                1.0,
                1.0,
                0.0
            )
 
        try:
 
            a = homography[0, 0]
            b = homography[0, 1]
 
            c = homography[1, 0]
            d = homography[1, 1]
 
            scale_x = float(
                np.sqrt(
                    a * a +
                    c * c
                )
            )
 
            scale_y = float(
                np.sqrt(
                    b * b +
                    d * d
                )
            )
 
            rotation = float(
                np.degrees(
                    np.arctan2(
                        c,
                        a
                    )
                )
            )
 
            if not np.isfinite(
                scale_x
            ):
 
                scale_x = 1.0
 
            if not np.isfinite(
                scale_y
            ):
 
                scale_y = 1.0
 
            if not np.isfinite(
                rotation
            ):
 
                rotation = 0.0
 
            return (
                scale_x,
                scale_y,
                rotation
            )
 
        except (
            ValueError,
            TypeError,
            IndexError
        ):
 
            return (
                1.0,
                1.0,
                0.0
            )
 
 
    # ========================================================
    # FAILED RESULT
    # ========================================================
 
    @staticmethod
    def _failed(
        reason: str,
        feature_matches: int = 0,
        inlier_matches: int = 0,
        homography: Optional[
            np.ndarray
        ] = None,
    ) -> AlignmentResult:
        """
        정렬 실패 결과를 만든다.
 
        실패한 경우 절대로 원본을
        정렬된 것처럼 반환하지 않는다.
        """
 
        return AlignmentResult(
            success=False,
            aligned_image=None,
            homography=homography,
            feature_matches=feature_matches,
            inlier_matches=inlier_matches,
            alignment_score=0.0,
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            reason=reason,
        )
 
    # ========================================================
    # ALIGNMENT VALIDATION
    # ========================================================
 
    def validate_alignment(
        self,
        before_page: PageImage,
        alignment: AlignmentResult,
    ) -> AlignmentResult:
        """
        Auto Align 결과를 한 번 더 검증한다.
 
        목적:
            잘못된 Homography가 계산되었더라도
            이후 변경점 분석으로 넘어가지 않도록 한다.
        """
 
        if not alignment.success:
 
            return alignment
 
        if alignment.aligned_image is None:
 
            return self._failed(
                "정렬 결과 이미지가 없음",
                feature_matches=(
                    alignment.feature_matches
                ),
                inlier_matches=(
                    alignment.inlier_matches
                ),
                homography=(
                    alignment.homography
                )
            )
 
        before = before_page.image
 
        after_aligned = (
            alignment.aligned_image
        )
 
        if before is None:
 
            return self._failed(
                "검증용 Before 이미지가 없음"
            )
 
        if after_aligned is None:
 
            return self._failed(
                "검증용 정렬 이미지가 없음"
            )
 
        # ----------------------------------------------------
        # 크기 확인
        # ----------------------------------------------------
 
        if before.shape[:2] != (
            after_aligned.shape[:2]
        ):
 
            return self._failed(
                "정렬 후 이미지 크기가 "
                "일치하지 않음"
            )
 
        # ----------------------------------------------------
        # 구조 기반 검증
        # ----------------------------------------------------
 
        structure_score = (
            self._validation_structure_score(
                before,
                after_aligned
            )
        )
 
        # ----------------------------------------------------
        # 중앙 영역 검증
        # ----------------------------------------------------
 
        center_score = (
            self._center_overlap_score(
                before,
                after_aligned
            )
        )
 
        # ----------------------------------------------------
        # 최종 검증 점수
        # ----------------------------------------------------
 
        validation_score = (
            structure_score * 0.70
            +
            center_score * 0.30
        )
 
        validation_score = float(
            max(
                0.0,
                min(
                    1.0,
                    validation_score
                )
            )
        )
 
        # ----------------------------------------------------
        # 너무 낮으면 정렬 실패
        # ----------------------------------------------------
 
        if validation_score < 0.35:
 
            return self._failed(
                (
                    "Auto Align 결과 검증 실패 "
                    f"(score={validation_score:.3f})"
                ),
                feature_matches=(
                    alignment.feature_matches
                ),
                inlier_matches=(
                    alignment.inlier_matches
                ),
                homography=(
                    alignment.homography
                )
            )
 
        # ----------------------------------------------------
        # 기존 결과 갱신
        # ----------------------------------------------------
 
        alignment.alignment_score = float(
            (
                alignment.alignment_score
                +
                validation_score
            )
            / 2.0
        )
 
        if validation_score < 0.50:
 
            alignment.reason = (
                "정렬은 완료되었으나 "
                "검증 신뢰도가 낮음"
            )
 
        else:
 
            alignment.reason = (
                "자동 정렬 및 검증 성공"
            )
 
        return alignment
 
 
    # ========================================================
    # VALIDATION STRUCTURE SCORE
    # ========================================================
 
    @staticmethod
    def _validation_structure_score(
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        """
        정렬된 두 이미지의 Edge 구조가
        얼마나 겹치는지 검사한다.
        """
 
        try:
 
            before_gray = (
                Aligner._to_gray(
                    before
                )
            )
 
            after_gray = (
                Aligner._to_gray(
                    after
                )
            )
 
            before_gray = cv2.resize(
                before_gray,
                (800, 800),
                interpolation=cv2.INTER_AREA
            )
 
            after_gray = cv2.resize(
                after_gray,
                (800, 800),
                interpolation=cv2.INTER_AREA
            )
 
            before_edges = cv2.Canny(
                before_gray,
                50,
                150
            )
 
            after_edges = cv2.Canny(
                after_gray,
                50,
                150
            )
 
            # ------------------------------------------------
            # Edge를 약간 확장해서
            # 1~2픽셀 정도의 미세한 정렬 오차 허용
            # ------------------------------------------------
 
            kernel = np.ones(
                (3, 3),
                np.uint8
            )
 
            before_dilated = cv2.dilate(
                before_edges,
                kernel,
                iterations=1
            )
 
            after_dilated = cv2.dilate(
                after_edges,
                kernel,
                iterations=1
            )
 
            before_mask = (
                before_dilated > 0
            )
 
            after_mask = (
                after_dilated > 0
            )
 
            intersection = np.logical_and(
                before_mask,
                after_mask
            ).sum()
 
            before_count = (
                before_mask.sum()
            )
 
            after_count = (
                after_mask.sum()
            )
 
            denominator = max(
                1,
                min(
                    before_count,
                    after_count
                )
            )
 
            score = (
                intersection /
                denominator
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
    # CENTER OVERLAP SCORE
    # ========================================================
 
    @staticmethod
    def _center_overlap_score(
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        """
        페이지 중앙 영역의 구조가
        얼마나 비슷한지 확인한다.
 
        도면의 외곽 여백이나 테두리보다
        실제 도면 영역의 정렬 상태를 확인하는 데 사용한다.
        """
 
        try:
 
            before_gray = (
                Aligner._to_gray(
                    before
                )
            )
 
            after_gray = (
                Aligner._to_gray(
                    after
                )
            )
 
            height, width = (
                before_gray.shape[:2]
            )
 
            if height <= 0:
                return 0.0
 
            if width <= 0:
                return 0.0
 
            # 중앙 80%
            margin_x = int(
                width * 0.10
            )
 
            margin_y = int(
                height * 0.10
            )
 
            before_crop = (
                before_gray[
                    margin_y:
                    height - margin_y,
                    margin_x:
                    width - margin_x
                ]
            )
 
            after_crop = (
                after_gray[
                    margin_y:
                    height - margin_y,
                    margin_x:
                    width - margin_x
                ]
            )
 
            if before_crop.size == 0:
                return 0.0
 
            if after_crop.size == 0:
                return 0.0
 
            before_crop = cv2.resize(
                before_crop,
                (500, 500),
                interpolation=cv2.INTER_AREA
            )
 
            after_crop = cv2.resize(
                after_crop,
                (500, 500),
                interpolation=cv2.INTER_AREA
            )
 
            before_edges = cv2.Canny(
                before_crop,
                50,
                150
            )
 
            after_edges = cv2.Canny(
                after_crop,
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
 
            score = (
                intersection /
                union
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
    # GRAYSCALE
    # ========================================================
 
    @staticmethod
    def _to_gray(
        image: np.ndarray
    ) -> np.ndarray:
        """
        이미지를 grayscale로 변환한다.
        """
 
        if image is None:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
        if image.ndim == 2:
 
            return image
 
        try:
 
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
 
        except cv2.error:
 
            return np.empty(
                (0, 0),
                dtype=np.uint8
            )
 
 
    # ========================================================
    # SAFE ALIGN
    # ========================================================
 
    def safe_align(
        self,
        before_page: PageImage,
        after_page: PageImage,
    ) -> AlignmentResult:
        """
        Auto Align + 검증을 한 번에 수행한다.
 
        이후 비교 모듈에서는 align() 대신
        이 함수를 사용하는 것을 권장한다.
        """
 
        result = self.align(
            before_page,
            after_page
        )
 
        if not result.success:
 
            return result
 
        return self.validate_alignment(
            before_page,
            result
        )
 
    # ========================================================
    # ALIGN MANY PAGES
    # ========================================================
 
    def align_many(
        self,
        page_matches
    ) -> List[AlignmentResult]:
        """
        PageMatcher가 만든 페이지 매칭 결과를
        한 번에 Auto Align한다.
 
        MATCH 상태인 페이지만 자동 정렬한다.
        REVIEW는 자동 정렬하지 않는다.
        """
 
        results = []
 
        for page_match in page_matches:
 
            if page_match.status != "MATCH":
                continue
 
            result = self.safe_align(
                page_match.before_page,
                page_match.after_page
            )
 
            results.append(
                result
            )
 
        return results
 
 
    # ========================================================
    # ALIGN RESULT TO DICT
    # ========================================================
 
    @staticmethod
    def result_to_dict(
        result: AlignmentResult
    ) -> dict:
        """
        AlignmentResult를 JSON / Excel 등에 사용할 수 있는
        dictionary 형태로 변환한다.
        """
 
        return {
            "success": result.success,
 
            "feature_matches": (
                result.feature_matches
            ),
 
            "inlier_matches": (
                result.inlier_matches
            ),
 
            "alignment_score": round(
                result.alignment_score,
                4
            ),
 
            "scale_x": round(
                result.scale_x,
                6
            ),
 
            "scale_y": round(
                result.scale_y,
                6
            ),
 
            "rotation": round(
                result.rotation,
                4
            ),
 
            "reason": result.reason,
        }
 
 
    # ========================================================
    # SAVE ALIGNED IMAGE
    # ========================================================
 
    @staticmethod
    def save_aligned_image(
        result: AlignmentResult,
        output_path
    ) -> bool:
        """
        정렬된 이미지를 파일로 저장한다.
 
        원본 이미지는 절대 덮어쓰지 않는다.
        """
 
        if not result.success:
 
            return False
 
        if result.aligned_image is None:
 
            return False
 
        try:
 
            success = cv2.imwrite(
                str(output_path),
                result.aligned_image
            )
 
            return bool(success)
 
        except cv2.error:
 
            return False
 
 
    # ========================================================
    # ALIGNMENT QUALITY
    # ========================================================
 
    @staticmethod
    def quality_label(
        result: AlignmentResult
    ) -> str:
        """
        정렬 품질을 사람이 이해하기 쉬운
        상태로 변환한다.
        """
 
        if not result.success:
 
            return "FAILED"
 
        score = (
            result.alignment_score
        )
 
        if score >= 0.80:
 
            return "GOOD"
 
        if score >= 0.60:
 
            return "ACCEPTABLE"
 
        if score >= 0.50:
 
            return "REVIEW"
 
        return "FAILED"
 
 
    # ========================================================
    # CHECK TRANSFORM
    # ========================================================
 
    @staticmethod
    def is_transform_reasonable(
        result: AlignmentResult
    ) -> bool:
        """
        지나치게 큰 확대/축소/회전이 발생했는지 확인한다.
 
        비정상적인 정렬을 변경점 분석에 넘기지 않는다.
        """
 
        if not result.success:
 
            return False
 
        scale_x = (
            result.scale_x
        )
 
        scale_y = (
            result.scale_y
        )
 
        rotation = abs(
            result.rotation
        )
 
        # ----------------------------------------------------
        # 정상적인 도면 크기 변화 허용 범위
        # ----------------------------------------------------
 
        if scale_x < 0.50:
            return False
 
        if scale_x > 2.00:
            return False
 
        if scale_y < 0.50:
            return False
 
        if scale_y > 2.00:
            return False
 
        # 일반적인 도면은
        # 큰 회전이 발생할 가능성이 낮다.
        if rotation > 15.0:
            return False
 
        return True
 
 
    # ========================================================
    # FINALIZE ALIGNMENT
    # ========================================================
 
    def finalize(
        self,
        before_page: PageImage,
        result: AlignmentResult
    ) -> AlignmentResult:
        """
        최종 안전성 검사를 수행한다.
 
        이후 변경점 검출 단계로 넘길 최종 결과.
        """
 
        if not result.success:
 
            return result
 
        # ----------------------------------------------------
        # 변환값 검사
        # ----------------------------------------------------
 
        if not self.is_transform_reasonable(
            result
        ):
 
            return self._failed(
                "비정상적인 변환이 감지됨",
                feature_matches=(
                    result.feature_matches
                ),
                inlier_matches=(
                    result.inlier_matches
                ),
                homography=(
                    result.homography
                )
            )
 
        # ----------------------------------------------------
        # 정렬 결과 존재 확인
        # ----------------------------------------------------
 
        if result.aligned_image is None:
 
            return self._failed(
                "최종 정렬 이미지가 없음",
                feature_matches=(
                    result.feature_matches
                ),
                inlier_matches=(
                    result.inlier_matches
                ),
                homography=(
                    result.homography
                )
            )
 
        # ----------------------------------------------------
        # Before 크기와 동일한지 확인
        # ----------------------------------------------------
 
        if (
            result.aligned_image.shape[:2]
            !=
            before_page.image.shape[:2]
        ):
 
            return self._failed(
                "최종 정렬 이미지 크기가 "
                "Before와 다름",
                feature_matches=(
                    result.feature_matches
                ),
                inlier_matches=(
                    result.inlier_matches
                ),
                homography=(
                    result.homography
                )
            )
 
        # ----------------------------------------------------
        # 모든 검사 통과
        # ----------------------------------------------------
 
        result.reason = (
            "최종 Auto Align 검사 통과"
        )
 
        return result
 
 
# ============================================================
# DEFAULT ALIGNER
# ============================================================
 
_default_aligner = Aligner()
 
 
def align_page(
    before_page: PageImage,
    after_page: PageImage,
) -> AlignmentResult:
    """
    외부 모듈에서 한 페이지를 자동 정렬한다.
    """
 
    result = _default_aligner.safe_align(
        before_page,
        after_page
    )
 
    return _default_aligner.finalize(
        before_page,
        result
    )
 
 
def align_pages(
    page_matches
) -> List[AlignmentResult]:
    """
    PageMatcher의 MATCH 결과를
    한 번에 자동 정렬한다.
    """
 
    results = []
 
    for page_match in page_matches:
 
        if page_match.status != "MATCH":
            continue
 
        result = align_page(
            page_match.before_page,
            page_match.after_page
        )
 
        results.append(
            result
        )
 
    return results
 
 
# ============================================================
# TEST
# ============================================================
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("DrawingCompare H5 - Aligner Test")
    print("=" * 60)
 
    print(
        "aligner.py 로드 성공"
    )
 
    print(
        f"Max Features : "
        f"{CONFIG.align.max_features}"
    )
 
    print(
        f"Ratio Test : "
        f"{CONFIG.align.ratio_test}"
    )
 
    print(
        f"Minimum Matches : "
        f"{CONFIG.align.minimum_matches}"
    )
 
    print(
        f"Minimum Inliers : "
        f"{CONFIG.align.minimum_inliers}"
    )
 
    print(
        f"RANSAC Threshold : "
        f"{CONFIG.align.ransac_threshold}"
    )
 
    print("=" * 60)
 
 