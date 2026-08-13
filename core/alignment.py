from __future__ import annotations
 
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
 
import cv2
import numpy as np
 
 
# ============================================================
# Alignment Result
# ============================================================
 
@dataclass
class AlignmentResult:
 
    success: bool
 
    status: str
 
    confidence: float
 
    reason: str
 
    scale_x: float
 
    scale_y: float
 
    rotation: float
 
    translation_x: float
 
    translation_y: float
 
    before_shape: Tuple[int, int]
 
    after_shape: Tuple[int, int]
 
    aligned_image: Optional[np.ndarray] = None
 
    transform_matrix: Optional[np.ndarray] = None
 
    match_count: int = 0
 
    inlier_count: int = 0
 
    inlier_ratio: float = 0.0
 
    feature_method: str = ""
 
    def to_dict(
        self
    ) -> Dict[str, Any]:
 
        return {
            "success": self.success,
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "rotation": self.rotation,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "before_shape": self.before_shape,
            "after_shape": self.after_shape,
            "match_count": self.match_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "feature_method": self.feature_method,
        }
 
 
# ============================================================
# Alignment Engine
# ============================================================
 
class AlignmentEngine:
 
    def __init__(
        self,
        min_matches: int = 12,
        min_inlier_ratio: float = 0.35,
        accept_confidence: float = 0.70,
        review_confidence: float = 0.50,
        max_rotation: float = 8.0,
        max_scale_change: float = 0.25,
    ):
 
        self.min_matches = int(
            min_matches
        )
 
        self.min_inlier_ratio = float(
            min_inlier_ratio
        )
 
        self.accept_confidence = float(
            accept_confidence
        )
 
        self.review_confidence = float(
            review_confidence
        )
 
        self.max_rotation = float(
            max_rotation
        )
 
        self.max_scale_change = float(
            max_scale_change
        )
 
    # ========================================================
    # Public API
    # ========================================================
 
    def align(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
    ) -> AlignmentResult:
 
        if before_image is None:
 
            return self._error_result(
                "before image is None"
            )
 
        if after_image is None:
 
            return self._error_result(
                "after image is None"
            )
 
        if not isinstance(
            before_image,
            np.ndarray
        ):
 
            return self._error_result(
                "before image must be numpy.ndarray"
            )
 
        if not isinstance(
            after_image,
            np.ndarray
        ):
 
            return self._error_result(
                "after image must be numpy.ndarray"
            )
 
        if before_image.size == 0:
 
            return self._error_result(
                "before image is empty"
            )
 
        if after_image.size == 0:
 
            return self._error_result(
                "after image is empty"
            )
 
        before_gray = self._to_gray(
            before_image
        )
 
        after_gray = self._to_gray(
            after_image
        )
 
        try:
 
            result = self._align_orb(
                before_gray,
                after_gray
            )
 
            if result.success:
 
                return result
 
            # ORB가 실패하면
            # ECC 방식으로 한 번 더 시도한다.
 
            fallback = self._align_ecc(
                before_gray,
                after_gray
            )
 
            if fallback.success:
 
                return fallback
 
            # 둘 다 실패하면
            # REVIEW 상태로 반환한다.
 
            return AlignmentResult(
                success=False,
                status="REVIEW",
                confidence=0.0,
                reason=(
                    "automatic alignment "
                    "could not be confirmed"
                ),
                scale_x=1.0,
                scale_y=1.0,
                rotation=0.0,
                translation_x=0.0,
                translation_y=0.0,
                before_shape=(
                    before_gray.shape[0],
                    before_gray.shape[1],
                ),
                after_shape=(
                    after_gray.shape[0],
                    after_gray.shape[1],
                ),
                feature_method="ORB+ECC",
            )
 
        except Exception as exc:
 
            return AlignmentResult(
                success=False,
                status="ERROR",
                confidence=0.0,
                reason=str(exc),
                scale_x=1.0,
                scale_y=1.0,
                rotation=0.0,
                translation_x=0.0,
                translation_y=0.0,
                before_shape=(
                    before_gray.shape[0],
                    before_gray.shape[1],
                ),
                after_shape=(
                    after_gray.shape[0],
                    after_gray.shape[1],
                ),
                feature_method="",
            )
 
    # ========================================================
    # Gray Conversion
    # ========================================================
 
    def _to_gray(
        self,
        image: np.ndarray
    ) -> np.ndarray:
 
        if len(
            image.shape
        ) == 2:
 
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
    # Error Result
    # ========================================================
 
    def _error_result(
        self,
        reason: str
    ) -> AlignmentResult:
 
        return AlignmentResult(
            success=False,
            status="ERROR",
            confidence=0.0,
            reason=reason,
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            translation_x=0.0,
            translation_y=0.0,
            before_shape=(0, 0),
            after_shape=(0, 0),
            feature_method="",
        )
     # ========================================================
    # ORB Alignment
    # ========================================================
 
    def _align_orb(
        self,
        before_gray: np.ndarray,
        after_gray: np.ndarray,
    ) -> AlignmentResult:
 
        before_h, before_w = (
            before_gray.shape[:2]
        )
 
        after_h, after_w = (
            after_gray.shape[:2]
        )
 
        # ----------------------------------------------------
        # 너무 큰 원본은 특징점 계산용으로 축소한다.
        # 실제 정렬 결과는 이후 원본 크기에 맞춰 적용한다.
        # ----------------------------------------------------
 
        before_small, before_scale = (
            self._resize_for_alignment(
                before_gray
            )
        )
 
        after_small, after_scale = (
            self._resize_for_alignment(
                after_gray
            )
        )
 
        # ----------------------------------------------------
        # ORB
        # ----------------------------------------------------
 
        orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=12,
        )
 
        keypoints_before, descriptors_before = (
            orb.detectAndCompute(
                before_small,
                None
            )
        )
 
        keypoints_after, descriptors_after = (
            orb.detectAndCompute(
                after_small,
                None
            )
        )
 
        if (
            descriptors_before is None
            or
            descriptors_after is None
        ):
 
            return self._orb_failed(
                before_gray,
                after_gray,
                "ORB descriptors not found"
            )
 
        if (
            len(keypoints_before)
            < self.min_matches
            or
            len(keypoints_after)
            < self.min_matches
        ):
 
            return self._orb_failed(
                before_gray,
                after_gray,
                "not enough ORB keypoints"
            )
 
        # ----------------------------------------------------
        # BF Matcher
        # ----------------------------------------------------
 
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )
 
        raw_matches = matcher.knnMatch(
            descriptors_before,
            descriptors_after,
            k=2
        )
 
        # ----------------------------------------------------
        # Lowe Ratio Test
        # ----------------------------------------------------
 
        good_matches = []
 
        for pair in raw_matches:
 
            if len(pair) < 2:
                continue
 
            first, second = pair
 
            if (
                first.distance
                <
                0.75 * second.distance
            ):
 
                good_matches.append(
                    first
                )
 
        if len(
            good_matches
        ) < self.min_matches:
 
            return self._orb_failed(
                before_gray,
                after_gray,
                (
                    "not enough reliable "
                    "feature matches"
                )
            )
 
        # ----------------------------------------------------
        # 대응점 추출
        # ----------------------------------------------------
 
        before_points = np.float32(
            [
                keypoints_before[
                    match.queryIdx
                ].pt
                for match in good_matches
            ]
        )
 
        after_points = np.float32(
            [
                keypoints_after[
                    match.trainIdx
                ].pt
                for match in good_matches
            ]
        )
 
        # ----------------------------------------------------
        # 원본 이미지 좌표계로 복원
        # ----------------------------------------------------
 
        before_points /= (
            before_scale
        )
 
        after_points /= (
            after_scale
        )
 
        # ----------------------------------------------------
        # Similarity Transform
        #
        # 회전 + 이동 + 균일 스케일
        #
        # 도면 비교에서는 perspective보다
        # 먼저 similarity를 사용한다.
        # ----------------------------------------------------
 
        matrix, inliers = (
            cv2.estimateAffinePartial2D(
                after_points,
                before_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=3000,
                confidence=0.99,
                refineIters=20,
            )
        )
 
        if matrix is None:
 
            return self._orb_failed(
                before_gray,
                after_gray,
                "could not estimate transform"
            )
 
        if inliers is None:
 
            return self._orb_failed(
                before_gray,
                after_gray,
                "no RANSAC inliers"
            )
 
        inlier_mask = (
            inliers.ravel()
            .astype(bool)
        )
 
        inlier_count = int(
            np.count_nonzero(
                inlier_mask
            )
        )
 
        match_count = len(
            good_matches
        )
 
        inlier_ratio = (
            inlier_count /
            max(
                1,
                match_count
            )
        )
 
        # ----------------------------------------------------
        # 변환값 추출
        # ----------------------------------------------------
 
        scale_x, scale_y, rotation, tx, ty = (
            self._extract_transform_parameters(
                matrix
            )
        )
 
        # ----------------------------------------------------
        # 비정상적인 변환 차단
        # ----------------------------------------------------
 
        if (
            abs(rotation)
            >
            self.max_rotation
        ):
 
            return self._orb_failed(
                before_gray,
                after_gray,
                (
                    "rotation exceeds "
                    "allowed range"
                )
            )
 
        scale_deviation = max(
            abs(scale_x - 1.0),
            abs(scale_y - 1.0)
        )
 
        if (
            scale_deviation
            >
            self.max_scale_change
        ):
 
            return self._orb_failed(
                before_gray,
                after_gray,
                (
                    "scale change exceeds "
                    "allowed range"
                )
            )
 
        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------
 
        confidence = (
            self._calculate_alignment_confidence(
                match_count=match_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                rotation=rotation,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        )
 
        # ----------------------------------------------------
        # 실제 After 이미지를 Before 좌표계로 변환
        # ----------------------------------------------------
 
        aligned = cv2.warpAffine(
            after_gray,
            matrix,
            (
                before_w,
                before_h
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
 
        # ----------------------------------------------------
        # 상태 결정
        # ----------------------------------------------------
 
        if (
            confidence
            >=
            self.accept_confidence
            and
            inlier_ratio
            >=
            self.min_inlier_ratio
        ):
 
            status = "ACCEPT"
            success = True
 
        elif (
            confidence
            >=
            self.review_confidence
        ):
 
            status = "REVIEW"
            success = False
 
        else:
 
            status = "REJECT"
            success = False
 
        return AlignmentResult(
            success=success,
            status=status,
            confidence=float(
                confidence
            ),
            reason=(
                "ORB similarity alignment"
            ),
            scale_x=float(
                scale_x
            ),
            scale_y=float(
                scale_y
            ),
            rotation=float(
                rotation
            ),
            translation_x=float(
                tx
            ),
            translation_y=float(
                ty
            ),
            before_shape=(
                before_h,
                before_w
            ),
            after_shape=(
                after_h,
                after_w
            ),
            aligned_image=aligned,
            transform_matrix=matrix,
            match_count=match_count,
            inlier_count=inlier_count,
            inlier_ratio=float(
                inlier_ratio
            ),
            feature_method="ORB",
        )
 
    # ========================================================
    # ORB Failure
    # ========================================================
 
    def _orb_failed(
        self,
        before_gray: np.ndarray,
        after_gray: np.ndarray,
        reason: str
    ) -> AlignmentResult:
 
        return AlignmentResult(
            success=False,
            status="REVIEW",
            confidence=0.0,
            reason=reason,
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            translation_x=0.0,
            translation_y=0.0,
            before_shape=(
                before_gray.shape[0],
                before_gray.shape[1],
            ),
            after_shape=(
                after_gray.shape[0],
                after_gray.shape[1],
            ),
            feature_method="ORB",
        )
 
    # ========================================================
    # Resize For Alignment
    # ========================================================
 
    def _resize_for_alignment(
        self,
        image: np.ndarray,
        max_dimension: int = 1800
    ) -> Tuple[np.ndarray, float]:
 
        height, width = (
            image.shape[:2]
        )
 
        largest = max(
            height,
            width
        )
 
        if largest <= max_dimension:
 
            return image, 1.0
 
        scale = (
            max_dimension /
            largest
        )
 
        new_width = max(
            1,
            int(
                width * scale
            )
        )
 
        new_height = max(
            1,
            int(
                height * scale
            )
        )
 
        resized = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )
 
        return (
            resized,
            scale
        )
     # ========================================================
    # ECC Alignment Fallback
    # ========================================================
 
    def _align_ecc(
        self,
        before_gray: np.ndarray,
        after_gray: np.ndarray,
    ) -> AlignmentResult:
 
        before_h, before_w = (
            before_gray.shape[:2]
        )
 
        after_h, after_w = (
            after_gray.shape[:2]
        )
 
        # ----------------------------------------------------
        # ECC는 동일한 크기의 영상에서 계산하는 것이
        # 안정적이므로 After를 Before 크기에 맞춘다.
        # ----------------------------------------------------
 
        resized_after = cv2.resize(
            after_gray,
            (
                before_w,
                before_h
            ),
            interpolation=cv2.INTER_LINEAR
        )
 
        before_float = (
            before_gray
            .astype(np.float32)
            /
            255.0
        )
 
        after_float = (
            resized_after
            .astype(np.float32)
            /
            255.0
        )
 
        # ----------------------------------------------------
        # 초기 변환
        # ----------------------------------------------------
 
        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )
 
        criteria = (
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6,
        )
 
        try:
 
            correlation, warp_matrix = (
                cv2.findTransformECC(
                    before_float,
                    after_float,
                    warp_matrix,
                    cv2.MOTION_AFFINE,
                    criteria,
                    None,
                    5,
                )
            )
 
        except cv2.error:
 
            return AlignmentResult(
                success=False,
                status="REVIEW",
                confidence=0.0,
                reason=(
                    "ECC alignment failed"
                ),
                scale_x=1.0,
                scale_y=1.0,
                rotation=0.0,
                translation_x=0.0,
                translation_y=0.0,
                before_shape=(
                    before_h,
                    before_w
                ),
                after_shape=(
                    after_h,
                    after_w
                ),
                feature_method="ECC",
            )
 
        # ----------------------------------------------------
        # 변환값
        # ----------------------------------------------------
 
        scale_x, scale_y, rotation, tx, ty = (
            self._extract_transform_parameters(
                warp_matrix
            )
        )
 
        # ----------------------------------------------------
        # 비정상적인 회전 차단
        # ----------------------------------------------------
 
        if (
            abs(rotation)
            >
            self.max_rotation
        ):
 
            return AlignmentResult(
                success=False,
                status="REVIEW",
                confidence=0.0,
                reason=(
                    "ECC rotation exceeds "
                    "allowed range"
                ),
                scale_x=float(
                    scale_x
                ),
                scale_y=float(
                    scale_y
                ),
                rotation=float(
                    rotation
                ),
                translation_x=float(
                    tx
                ),
                translation_y=float(
                    ty
                ),
                before_shape=(
                    before_h,
                    before_w
                ),
                after_shape=(
                    after_h,
                    after_w
                ),
                feature_method="ECC",
            )
 
        # ----------------------------------------------------
        # ECC confidence
        # ----------------------------------------------------
 
        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    correlation
                )
            )
        )
 
        # ----------------------------------------------------
        # 정렬 이미지
        # ----------------------------------------------------
 
        aligned = cv2.warpAffine(
            resized_after,
            warp_matrix,
            (
                before_w,
                before_h
            ),
            flags=(
                cv2.INTER_LINEAR
                |
                cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
 
        if (
            confidence
            >=
            self.accept_confidence
        ):
 
            status = "ACCEPT"
            success = True
 
        elif (
            confidence
            >=
            self.review_confidence
        ):
 
            status = "REVIEW"
            success = False
 
        else:
 
            status = "REJECT"
            success = False
 
        return AlignmentResult(
            success=success,
            status=status,
            confidence=confidence,
            reason=(
                "ECC affine alignment"
            ),
            scale_x=float(
                scale_x
            ),
            scale_y=float(
                scale_y
            ),
            rotation=float(
                rotation
            ),
            translation_x=float(
                tx
            ),
            translation_y=float(
                ty
            ),
            before_shape=(
                before_h,
                before_w
            ),
            after_shape=(
                after_h,
                after_w
            ),
            aligned_image=aligned,
            transform_matrix=warp_matrix,
            match_count=0,
            inlier_count=0,
            inlier_ratio=0.0,
            feature_method="ECC",
        )
 
    # ========================================================
    # Transform Parameter Extraction
    # ========================================================
 
    def _extract_transform_parameters(
        self,
        matrix: np.ndarray
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float
    ]:
 
        a = float(
            matrix[0, 0]
        )
 
        b = float(
            matrix[0, 1]
        )
 
        c = float(
            matrix[1, 0]
        )
 
        d = float(
            matrix[1, 1]
        )
 
        tx = float(
            matrix[0, 2]
        )
 
        ty = float(
            matrix[1, 2]
        )
 
        # ----------------------------------------------------
        # 행렬에서 X/Y scale 계산
        # ----------------------------------------------------
 
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
 
        # ----------------------------------------------------
        # 회전각
        # ----------------------------------------------------
 
        rotation_rad = np.arctan2(
            c,
            a
        )
 
        rotation = float(
            np.degrees(
                rotation_rad
            )
        )
 
        return (
            scale_x,
            scale_y,
            rotation,
            tx,
            ty,
        )
 
    # ========================================================
    # Alignment Confidence
    # ========================================================
 
    def _calculate_alignment_confidence(
        self,
        match_count: int,
        inlier_count: int,
        inlier_ratio: float,
        rotation: float,
        scale_x: float,
        scale_y: float,
    ) -> float:
 
        # ----------------------------------------------------
        # Match score
        # ----------------------------------------------------
 
        match_score = min(
            1.0,
            match_count / 80.0
        )
 
        # ----------------------------------------------------
        # Inlier score
        # ----------------------------------------------------
 
        inlier_score = min(
            1.0,
            inlier_ratio / 0.75
        )
 
        # ----------------------------------------------------
        # 회전 안정성
        # ----------------------------------------------------
 
        rotation_score = max(
            0.0,
            1.0 -
            (
                abs(rotation)
                /
                max(
                    1.0,
                    self.max_rotation
                )
            )
        )
 
        # ----------------------------------------------------
        # Scale 안정성
        # ----------------------------------------------------
 
        scale_deviation = max(
            abs(scale_x - 1.0),
            abs(scale_y - 1.0)
        )
 
        scale_score = max(
            0.0,
            1.0 -
            (
                scale_deviation
                /
                max(
                    0.01,
                    self.max_scale_change
                )
            )
        )
 
        # ----------------------------------------------------
        # 최종 confidence
        # ----------------------------------------------------
 
        confidence = (
            0.20 * match_score
            +
            0.50 * inlier_score
            +
            0.15 * rotation_score
            +
            0.15 * scale_score
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
    # Alignment Validation
    # ========================================================
 
    def validate(
        self,
        before_image: np.ndarray,
        aligned_image: np.ndarray,
        result: AlignmentResult,
    ) -> AlignmentResult:
 
        """
        정렬 결과가 실제 변경점 비교에 사용할 만큼
        안정적인지 최종 검증한다.
        """
 
        if before_image is None:
            result.success = False
            result.status = "ERROR"
            result.reason = "before image is None"
            return result
 
        if aligned_image is None:
            result.success = False
            result.status = "ERROR"
            result.reason = "aligned image is None"
            return result
 
        if before_image.size == 0:
            result.success = False
            result.status = "ERROR"
            result.reason = "before image is empty"
            return result
 
        if aligned_image.size == 0:
            result.success = False
            result.status = "ERROR"
            result.reason = "aligned image is empty"
            return result
 
        before_gray = self._to_gray(
            before_image
        )
 
        aligned_gray = self._to_gray(
            aligned_image
        )
 
        # ----------------------------------------------------
        # 크기 확인
        # ----------------------------------------------------
 
        if (
            before_gray.shape
            !=
            aligned_gray.shape
        ):
 
            result.success = False
            result.status = "REVIEW"
            result.reason = (
                "aligned image size mismatch"
            )
 
            return result
 
        # ----------------------------------------------------
        # 전체 평균 밝기 차이
        # ----------------------------------------------------
 
        before_float = (
            before_gray.astype(
                np.float32
            )
        )
 
        aligned_float = (
            aligned_gray.astype(
                np.float32
            )
        )
 
        diff = cv2.absdiff(
            before_gray,
            aligned_gray
        )
 
        mean_difference = float(
            np.mean(diff)
        )
 
        # ----------------------------------------------------
        # 너무 큰 차이는 정렬 실패 가능성
        # ----------------------------------------------------
 
        if mean_difference > 80:
 
            result.success = False
            result.status = "REVIEW"
            result.confidence *= 0.70
 
            result.reason = (
                "large residual difference "
                "after alignment"
            )
 
            return result
 
        # ----------------------------------------------------
        # 중앙 영역 검증
        #
        # 페이지 외곽의 빈 영역이나
        # 표제란 하나만으로 정렬이 통과하지 않도록 한다.
        # ----------------------------------------------------
 
        h, w = before_gray.shape[:2]
 
        margin_x = int(
            w * 0.05
        )
 
        margin_y = int(
            h * 0.05
        )
 
        if (
            margin_x * 2 >= w
            or
            margin_y * 2 >= h
        ):
 
            result.success = False
            result.status = "REVIEW"
            result.reason = (
                "image too small for validation"
            )
 
            return result
 
        before_center = before_gray[
            margin_y:h - margin_y,
            margin_x:w - margin_x,
        ]
 
        aligned_center = aligned_gray[
            margin_y:h - margin_y,
            margin_x:w - margin_x,
        ]
 
        center_diff = cv2.absdiff(
            before_center,
            aligned_center
        )
 
        center_difference = float(
            np.mean(center_diff)
        )
 
        # ----------------------------------------------------
        # 중앙 영역이 전체보다 지나치게 나쁘면
        # 정렬을 의심한다.
        # ----------------------------------------------------
 
        if (
            center_difference
            >
            mean_difference * 1.8
            and
            center_difference > 35
        ):
 
            result.success = False
            result.status = "REVIEW"
            result.confidence *= 0.80
 
            result.reason = (
                "center alignment "
                "quality is insufficient"
            )
 
            return result
 
        # ----------------------------------------------------
        # 최종 confidence 보정
        # ----------------------------------------------------
 
        quality_factor = 1.0
 
        if mean_difference < 15:
            quality_factor *= 1.05
 
        elif mean_difference < 30:
            quality_factor *= 1.00
 
        elif mean_difference < 50:
            quality_factor *= 0.90
 
        else:
            quality_factor *= 0.80
 
        if center_difference < 20:
            quality_factor *= 1.05
 
        elif center_difference > 40:
            quality_factor *= 0.90
 
        result.confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    result.confidence
                    *
                    quality_factor
                )
            )
        )
 
        # ----------------------------------------------------
        # 최종 상태
        # ----------------------------------------------------
 
        if (
            result.confidence
            >=
            self.accept_confidence
            and
            (
                result.inlier_ratio == 0
                or
                result.inlier_ratio
                >=
                self.min_inlier_ratio
            )
        ):
 
            result.success = True
            result.status = "ACCEPT"
 
            result.reason = (
                "alignment validated"
            )
 
        elif (
            result.confidence
            >=
            self.review_confidence
        ):
 
            result.success = False
            result.status = "REVIEW"
 
            result.reason = (
                "alignment requires review"
            )
 
        else:
 
            result.success = False
            result.status = "REJECT"
 
            result.reason = (
                "alignment confidence "
                "is too low"
            )
 
        return result
 
    # ========================================================
    # Full Alignment Pipeline
    # ========================================================
 
    def align_and_validate(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
    ) -> AlignmentResult:
 
        result = self.align(
            before_image,
            after_image
        )
 
        if result.aligned_image is None:
 
            return result
 
        return self.validate(
            before_image,
            result.aligned_image,
            result
        )
 
    # ========================================================
    # Save Diagnostic Image
    # ========================================================
 
    def save_diagnostic(
        self,
        before_image: np.ndarray,
        aligned_image: np.ndarray,
        output_path: str,
    ) -> bool:
 
        """
        Before와 정렬된 After를 반투명하게 겹쳐
        정렬 상태를 확인할 수 있는 진단 이미지를 저장한다.
        """
 
        if before_image is None:
            return False
 
        if aligned_image is None:
            return False
 
        before_gray = self._to_gray(
            before_image
        )
 
        aligned_gray = self._to_gray(
            aligned_image
        )
 
        if (
            before_gray.shape
            !=
            aligned_gray.shape
        ):
 
            return False
 
        before_bgr = cv2.cvtColor(
            before_gray,
            cv2.COLOR_GRAY2BGR
        )
 
        aligned_bgr = cv2.cvtColor(
            aligned_gray,
            cv2.COLOR_GRAY2BGR
        )
 
        overlay = cv2.addWeighted(
            before_bgr,
            0.5,
            aligned_bgr,
            0.5,
            0
        )
 
        try:
 
            return bool(
                cv2.imwrite(
                    output_path,
                    overlay
                )
            )
 
        except Exception:
 
            return False
 
 
# ============================================================
# Convenience Function
# ============================================================
 
def align_images(
    before_image: np.ndarray,
    after_image: np.ndarray,
) -> AlignmentResult:
 
    engine = AlignmentEngine()
 
    return engine.align_and_validate(
        before_image,
        after_image
    )
 
 
# ============================================================
# Result Helper
# ============================================================
 
def alignment_result_to_dict(
    result: AlignmentResult
) -> Dict[str, Any]:
 
    return result.to_dict()
# ============================================================
# Drawing Structure Alignment - Part 5-1
# ============================================================
 
def _drawing_structure_profile(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
 
    gray = image
 
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )
 
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )
 
    edges = cv2.Canny(
        blurred,
        50,
        150,
    )
 
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 1),
    )
 
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 31),
    )
 
    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
 
    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )
 
    horizontal_profile = np.sum(
        horizontal > 0,
        axis=1,
    ).astype(np.float32)
 
    vertical_profile = np.sum(
        vertical > 0,
        axis=0,
    ).astype(np.float32)
 
    horizontal_max = float(
        np.max(horizontal_profile)
    )
 
    vertical_max = float(
        np.max(vertical_profile)
    )
 
    if horizontal_max > 0:
        horizontal_profile /= horizontal_max
 
    if vertical_max > 0:
        vertical_profile /= vertical_max
 
    return (
        horizontal_profile,
        vertical_profile,
    )
 
 
def _drawing_profile_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
 
    if first.size == 0 or second.size == 0:
        return 0.0
 
    length = min(
        len(first),
        len(second),
    )
 
    if length < 10:
        return 0.0
 
    first = first[:length]
    second = second[:length]
 
    first = first - np.mean(first)
    second = second - np.mean(second)
 
    denominator = (
        np.linalg.norm(first)
        *
        np.linalg.norm(second)
    )
 
    if denominator <= 1e-8:
        return 0.0
 
    correlation = (
        np.dot(first, second)
        /
        denominator
    )
 
    return float(
        max(
            0.0,
            min(
                1.0,
                (correlation + 1.0) / 2.0,
            ),
        )
    )
 # ============================================================
# Drawing Structure Alignment - Part 5-2
# ============================================================
 
def _drawing_structure_align(
    engine: AlignmentEngine,
    before_gray: np.ndarray,
    after_gray: np.ndarray,
) -> AlignmentResult:
 
    before_h, before_w = (
        before_gray.shape[:2]
    )
 
    after_h, after_w = (
        after_gray.shape[:2]
    )
 
    if (
        before_h <= 0
        or before_w <= 0
        or after_h <= 0
        or after_w <= 0
    ):
        return AlignmentResult(
            success=False,
            status="REVIEW",
            confidence=0.0,
            reason="invalid image dimensions",
            scale_x=1.0,
            scale_y=1.0,
            rotation=0.0,
            translation_x=0.0,
            translation_y=0.0,
            before_shape=(
                before_h,
                before_w,
            ),
            after_shape=(
                after_h,
                after_w,
            ),
            feature_method="STRUCTURE",
        )
 
    # --------------------------------------------------------
    # After를 Before 크기로 임시 보정
    # --------------------------------------------------------
 
    scale_x = (
        before_w / after_w
    )
 
    scale_y = (
        before_h / after_h
    )
 
    resized_after = cv2.resize(
        after_gray,
        (
            before_w,
            before_h,
        ),
        interpolation=cv2.INTER_AREA,
    )
 
    # --------------------------------------------------------
    # 구조 profile 계산
    # --------------------------------------------------------
 
    before_horizontal, before_vertical = (
        _drawing_structure_profile(
            before_gray
        )
    )
 
    after_horizontal, after_vertical = (
        _drawing_structure_profile(
            resized_after
        )
    )
 
    horizontal_score = (
        _drawing_profile_similarity(
            before_horizontal,
            after_horizontal,
        )
    )
 
    vertical_score = (
        _drawing_profile_similarity(
            before_vertical,
            after_vertical,
        )
    )
 
    structure_score = (
        horizontal_score * 0.5
        +
        vertical_score * 0.5
    )
 
    # --------------------------------------------------------
    # 실제 픽셀 차이
    # --------------------------------------------------------
 
    difference = cv2.absdiff(
        before_gray,
        resized_after,
    )
 
    mean_difference = float(
        np.mean(difference)
    )
 
    # --------------------------------------------------------
    # 크기 차이 제한
    # --------------------------------------------------------
 
    scale_deviation = max(
        abs(scale_x - 1.0),
        abs(scale_y - 1.0),
    )
 
    if scale_deviation > 0.30:
 
        return AlignmentResult(
            success=False,
            status="REVIEW",
            confidence=0.0,
            reason=(
                "drawing scale difference "
                "is too large"
            ),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            rotation=0.0,
            translation_x=0.0,
            translation_y=0.0,
            before_shape=(
                before_h,
                before_w,
            ),
            after_shape=(
                after_h,
                after_w,
            ),
            feature_method="STRUCTURE",
        )
 
    # --------------------------------------------------------
    # 구조 유사도가 너무 낮으면 사용하지 않는다.
    # --------------------------------------------------------
 
    if structure_score < 0.50:
 
        return AlignmentResult(
            success=False,
            status="REVIEW",
            confidence=float(
                structure_score
            ),
            reason=(
                "drawing structure "
                "similarity is insufficient"
            ),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            rotation=0.0,
            translation_x=0.0,
            translation_y=0.0,
            before_shape=(
                before_h,
                before_w,
            ),
            after_shape=(
                after_h,
                after_w,
            ),
            feature_method="STRUCTURE",
        )
 
    # --------------------------------------------------------
    # 구조 + 픽셀 차이 기반 confidence
    # --------------------------------------------------------
 
    pixel_score = max(
        0.0,
        min(
            1.0,
            1.0 -
            (
                mean_difference /
                100.0
            ),
        ),
    )
 
    confidence = (
        structure_score * 0.75
        +
        pixel_score * 0.25
    )
 
    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )
 
    # --------------------------------------------------------
    # 임시 정렬 이미지
    # --------------------------------------------------------
 
    aligned = resized_after
 
    matrix = np.array(
        [
            [
                scale_x,
                0.0,
                0.0,
            ],
            [
                0.0,
                scale_y,
                0.0,
            ],
        ],
        dtype=np.float32,
    )
 
    # --------------------------------------------------------
    # 상태
    # --------------------------------------------------------
 
    if confidence >= engine.accept_confidence:
 
        status = "ACCEPT"
        success = True
 
    elif confidence >= engine.review_confidence:
 
        status = "REVIEW"
        success = False
 
    else:
 
        status = "REJECT"
        success = False
 
    return AlignmentResult(
        success=success,
        status=status,
        confidence=float(confidence),
        reason=(
            "drawing structure alignment"
        ),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        rotation=0.0,
        translation_x=0.0,
        translation_y=0.0,
        before_shape=(
            before_h,
            before_w,
        ),
        after_shape=(
            after_h,
            after_w,
        ),
        aligned_image=aligned,
        transform_matrix=matrix,
        match_count=0,
        inlier_count=0,
        inlier_ratio=0.0,
        feature_method="STRUCTURE",
    )
    # ============================================================
# Drawing Structure Alignment - Part 5-3
# ============================================================
 
_original_alignment_engine_align = AlignmentEngine.align
 
 
def _extended_alignment(
    self,
    before_image: np.ndarray,
    after_image: np.ndarray,
) -> AlignmentResult:
 
    # --------------------------------------------------------
    # 1. 기존 ORB + ECC 정렬
    # --------------------------------------------------------
 
    original_result = (
        _original_alignment_engine_align(
            self,
            before_image,
            after_image,
        )
    )
 
    # --------------------------------------------------------
    # 기존 방식이 확실하게 성공하면 그대로 사용
    # --------------------------------------------------------
 
    if original_result.success:
 
        return original_result
 
    # --------------------------------------------------------
    # 입력 검증
    # --------------------------------------------------------
 
    if before_image is None:
 
        return original_result
 
    if after_image is None:
 
        return original_result
 
    # --------------------------------------------------------
    # Gray 변환
    # --------------------------------------------------------
 
    try:
 
        before_gray = self._to_gray(
            before_image
        )
 
        after_gray = self._to_gray(
            after_image
        )
 
    except Exception:
 
        return original_result
 
    # --------------------------------------------------------
    # 2. 도면 구조 기반 정렬
    # --------------------------------------------------------
 
    structure_result = (
        _drawing_structure_align(
            self,
            before_gray,
            after_gray,
        )
    )
 
    # --------------------------------------------------------
    # 구조 기반 결과가 더 신뢰할 수 있으면 사용
    # --------------------------------------------------------
 
    if (
        structure_result.confidence
        >
        original_result.confidence
    ):
 
        return structure_result
 
    return original_result
 
 
# ============================================================
# AlignmentEngine에 확장 정렬 적용
# ============================================================
 
AlignmentEngine.align = _extended_alignment
 