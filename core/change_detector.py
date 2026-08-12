from __future__ import annotations
 
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
 
import cv2
import numpy as np
 
from config import CONFIG
 
 
# ============================================================
# Change Region
# ============================================================
 
@dataclass
class ChangeRegion:
    x: int
    y: int
    width: int
    height: int
 
    area: int = 0
    change_ratio: float = 0.0
 
    region_type: str = "unknown"
    confidence: float = 0.0
 
    old_crop: Optional[np.ndarray] = None
    new_crop: Optional[np.ndarray] = None
    difference_crop: Optional[np.ndarray] = None
 
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
 
    @property
    def center(self) -> Tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )
 
 
# ============================================================
# Change Detection Result
# ============================================================
 
@dataclass
class ChangeDetectionResult:
    success: bool
    regions: List[ChangeRegion] = field(default_factory=list)
 
    difference_image: Optional[np.ndarray] = None
    threshold_image: Optional[np.ndarray] = None
 
    change_pixel_ratio: float = 0.0
    reason: str = ""
 
    @property
    def region(self) -> List[ChangeRegion]:
        return self.regions
 
 
# ============================================================
# Change Detector
# ============================================================
 
class ChangeDetector:
 
    def __init__(self, config=None):
        self.config = config or CONFIG
 
        self.minimum_area = self._get_config(
            "change.minimum_area",
            100
        )
 
        self.morph_kernel_size = self._get_config(
            "change.morph_kernel_size",
            3
        )
 
        self.pixel_threshold = self._get_config(
            "change.pixel_threshold",
            30
        )
 
        self.max_region_ratio = self._get_config(
            "change.max_region_ratio",
            0.60
        )
 
        self.merge_distance = self._get_config(
            "change.merge_distance",
            15
        )
 
        self._failed = False
 
    def _get_config(self, path: str, default: Any) -> Any:
        current = self.config
 
        for key in path.split("."):
            try:
                if isinstance(current, dict):
                    current = current[key]
                else:
                    current = getattr(current, key)
            except (AttributeError, KeyError, TypeError):
                return default
 
        return current
 
    def _get_page_image(self, page) -> np.ndarray:
        if isinstance(page, np.ndarray):
            image = page
 
        elif hasattr(page, "image"):
            image = page.image
 
        elif hasattr(page, "array"):
            image = page.array
 
        else:
            raise TypeError(
                "PageImage에서 image 배열을 찾을 수 없습니다."
            )
 
        if image is None:
            raise ValueError("페이지 이미지가 비어 있습니다.")
 
        image = np.asarray(image)
 
        if image.size == 0:
            raise ValueError("페이지 이미지 크기가 0입니다.")
 
        return image
 
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
 
        elif image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2GRAY
                )
            else:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )
 
        else:
            raise ValueError(
                f"지원하지 않는 이미지 차원입니다: {image.shape}"
            )
 
        return gray
 
    def _normalize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
 
        gray = self._to_gray(image)
 
        if gray.dtype != np.uint8:
            gray = cv2.normalize(
                gray,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)
 
        return gray
 
    def _resize_to_same_size(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
 
        h1, w1 = before.shape[:2]
        h2, w2 = after.shape[:2]
 
        if h1 == h2 and w1 == w2:
            return before, after
 
        target_w = max(w1, w2)
        target_h = max(h1, h2)
 
        before_resized = cv2.resize(
            before,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
 
        after_resized = cv2.resize(
            after,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
 
        return before_resized, after_resized
    # ========================================================
    # Automatic Alignment
    # ========================================================
 
    def _align_images(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
 
        before, after = self._resize_to_same_size(
            before,
            after
        )
 
        h, w = before.shape[:2]
 
        if h < 20 or w < 20:
            return before, after
 
        before_gray = self._normalize_image(before)
        after_gray = self._normalize_image(after)
 
        try:
            orb = cv2.ORB_create(
                nfeatures=3000
            )
 
            kp1, des1 = orb.detectAndCompute(
                before_gray,
                None
            )
 
            kp2, des2 = orb.detectAndCompute(
                after_gray,
                None
            )
 
            if (
                des1 is not None
                and des2 is not None
                and len(kp1) >= 4
                and len(kp2) >= 4
            ):
 
                matcher = cv2.BFMatcher(
                    cv2.NORM_HAMMING,
                    crossCheck=True
                )
 
                matches = matcher.match(
                    des1,
                    des2
                )
 
                matches = sorted(
                    matches,
                    key=lambda m: m.distance
                )
 
                if len(matches) >= 4:
 
                    good = matches[
                        :max(
                            4,
                            min(
                                100,
                                len(matches)
                            )
                        )
                    ]
 
                    src = np.float32(
                        [
                            kp1[m.queryIdx].pt
                            for m in good
                        ]
                    ).reshape(-1, 1, 2)
 
                    dst = np.float32(
                        [
                            kp2[m.trainIdx].pt
                            for m in good
                        ]
                    ).reshape(-1, 1, 2)
 
                    matrix, mask = cv2.estimateAffinePartial2D(
                        dst,
                        src,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0
                    )
 
                    if matrix is not None:
 
                        aligned = cv2.warpAffine(
                            after,
                            matrix,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255
                        )
 
                        return before, aligned
 
        except Exception:
            pass
 
        return before, after
 
 
    # ========================================================
    # Difference
    # ========================================================
 
    def _calculate_difference(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
 
        before_gray = self._normalize_image(
            before
        )
 
        after_gray = self._normalize_image(
            after
        )
 
        diff = cv2.absdiff(
            before_gray,
            after_gray
        )
 
        _, threshold = cv2.threshold(
            diff,
            int(self.pixel_threshold),
            255,
            cv2.THRESH_BINARY
        )
 
        kernel_size = max(
            1,
            int(self.morph_kernel_size)
        )
 
        if kernel_size % 2 == 0:
            kernel_size += 1
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
 
        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )
 
        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            kernel
        )
 
        changed_pixels = np.count_nonzero(
            threshold
        )
 
        total_pixels = (
            threshold.shape[0] *
            threshold.shape[1]
        )
 
        ratio = (
            changed_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )
 
        return diff, threshold, ratio
 
 
    # ========================================================
    # Region Detection
    # ========================================================
 
    def _find_regions(
        self,
        threshold: np.ndarray,
        before: np.ndarray,
        after: np.ndarray,
        diff: np.ndarray
    ) -> List[ChangeRegion]:
 
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        image_area = (
            threshold.shape[0] *
            threshold.shape[1]
        )
 
        regions: List[ChangeRegion] = []
 
        for contour in contours:
 
            area = cv2.contourArea(
                contour
            )
 
            if area < self.minimum_area:
                continue
 
            x, y, w, h = cv2.boundingRect(
                contour
            )
 
            region_area = w * h
 
            if image_area > 0:
 
                ratio = (
                    region_area /
                    image_area
                )
 
                if ratio > self.max_region_ratio:
                    continue
 
            crop_before = before[
                y:y + h,
                x:x + w
            ].copy()
 
            crop_after = after[
                y:y + h,
                x:x + w
            ].copy()
 
            crop_diff = diff[
                y:y + h,
                x:x + w
            ].copy()
 
            changed = np.count_nonzero(
                crop_diff >
                self.pixel_threshold
            )
 
            crop_pixels = max(
                1,
                crop_diff.shape[0] *
                crop_diff.shape[1]
            )
 
            change_ratio = (
                changed /
                crop_pixels
            )
 
            region = ChangeRegion(
                x=x,
                y=y,
                width=w,
                height=h,
                area=int(area),
                change_ratio=float(
                    change_ratio
                ),
                old_crop=crop_before,
                new_crop=crop_after,
                difference_crop=crop_diff
            )
 
            regions.append(
                region
            )
 
        return regions
 
    # ========================================================
    # Region Merge
    # ========================================================
 
    def _distance_between_regions(
        self,
        a: ChangeRegion,
        b: ChangeRegion
    ) -> float:
 
        horizontal_gap = max(
            b.left - a.right,
            a.left - b.right,
            0
        )
 
        vertical_gap = max(
            b.top - a.bottom,
            a.top - b.bottom,
            0
        )
 
        return math.sqrt(
            horizontal_gap ** 2 +
            vertical_gap ** 2
        )
 
 
    def _merge_regions(
        self,
        regions: List[ChangeRegion]
    ) -> List[ChangeRegion]:
 
        if len(regions) <= 1:
            return regions
 
        regions = sorted(
            regions,
            key=lambda r: r.area,
            reverse=True
        )
 
        merged: List[ChangeRegion] = []
 
        for region in regions:
 
            merged_into_existing = False
 
            for existing in merged:
 
                distance = (
                    self._distance_between_regions(
                        existing,
                        region
                    )
                )
 
                if distance <= self.merge_distance:
 
                    left = min(
                        existing.left,
                        region.left
                    )
 
                    top = min(
                        existing.top,
                        region.top
                    )
 
                    right = max(
                        existing.right,
                        region.right
                    )
 
                    bottom = max(
                        existing.bottom,
                        region.bottom
                    )
 
                    existing.x = left
                    existing.y = top
                    existing.width = (
                        right - left
                    )
                    existing.height = (
                        bottom - top
                    )
 
                    existing.area = (
                        existing.width *
                        existing.height
                    )
 
                    existing.change_ratio = max(
                        existing.change_ratio,
                        region.change_ratio
                    )
 
                    merged_into_existing = True
                    break
 
            if not merged_into_existing:
                merged.append(
                    region
                )
 
        return merged
 
 
    # ========================================================
    # Region Classification
    # ========================================================
 
    def classify_regions(
        self,
        regions: List[ChangeRegion]
    ) -> List[ChangeRegion]:
 
        for region in regions:
 
            ratio = region.change_ratio
 
            aspect = (
                region.width /
                max(1, region.height)
            )
 
            if ratio < 0.05:
 
                region.region_type = "minor"
                region.confidence = 0.60
 
            elif (
                aspect > 8
                or aspect < 0.125
            ):
 
                region.region_type = (
                    "dimension_or_text"
                )
 
                region.confidence = 0.75
 
            elif (
                region.width < 80
                and region.height < 80
            ):
 
                region.region_type = "detail"
                region.confidence = 0.65
 
            else:
 
                region.region_type = (
                    "drawing_change"
                )
 
                region.confidence = 0.70
 
        return regions
 
 
    # ========================================================
    # Detect
    # ========================================================
 
    def detect(
        self,
        before_page,
        after_page
    ) -> ChangeDetectionResult:
 
        try:
 
            self._failed = False
 
            before = self._get_page_image(
                before_page
            )
 
            after = self._get_page_image(
                after_page
            )
 
            before, after = (
                self._align_images(
                    before,
                    after
                )
            )
 
            diff, threshold, ratio = (
                self._calculate_difference(
                    before,
                    after
                )
            )
 
            regions = self._find_regions(
                threshold,
                before,
                after,
                diff
            )
 
            regions = self._merge_regions(
                regions
            )
 
            regions = self.classify_regions(
                regions
            )
 
            return ChangeDetectionResult(
                success=True,
                regions=regions,
                difference_image=diff,
                threshold_image=threshold,
                change_pixel_ratio=ratio,
                reason="analysis completed"
            )
 
        except Exception as exc:
 
            self._failed = True
 
            return ChangeDetectionResult(
                success=False,
                regions=[],
                difference_image=None,
                threshold_image=None,
                change_pixel_ratio=0.0,
                reason=str(exc)
            )
 
 
    # ========================================================
    # Analyze
    # ========================================================
 
    def analyze(
        self,
        before_page,
        after_page
    ) -> ChangeDetectionResult:
 
        return self.detect(
            before_page,
            after_page
        )
 
 
    # ========================================================
    # Analyze Final
    # ========================================================
 
    def analyze_final(
        self,
        before_page,
        after_page
    ) -> ChangeDetectionResult:
 
        result = self.detect(
            before_page,
            after_page
        )
 
        return self.finalize(
            result
        )
 
 
    # ========================================================
    # Finalize
    # ========================================================
 
    def finalize(
        self,
        result: ChangeDetectionResult
    ) -> ChangeDetectionResult:
 
        if not result.success:
            return result
 
        result.regions = (
            self.classify_regions(
                result.regions
            )
        )
 
        result.regions.sort(
            key=lambda r: (
                r.y,
                r.x
            )
        )
 
        if len(result.regions) == 0:
 
            result.reason = (
                "no significant changes detected"
            )
 
        else:
 
            result.reason = (
                f"{len(result.regions)} "
                "change regions detected"
            )
 
        return result
 
    # ========================================================
    # Draw Regions
    # ========================================================
 
    def draw_regions(
        self,
        image: np.ndarray,
        regions: List[ChangeRegion],
        thickness: int = 3
    ) -> np.ndarray:
 
        output = image.copy()
 
        for index, region in enumerate(
            regions,
            start=1
        ):
 
            cv2.rectangle(
                output,
                (
                    region.x,
                    region.y
                ),
                (
                    region.right,
                    region.bottom
                ),
                (0, 0, 255),
                thickness
            )
 
            label = (
                f"{index}: "
                f"{region.region_type}"
            )
 
            cv2.putText(
                output,
                label,
                (
                    region.x,
                    max(
                        20,
                        region.y - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
 
        return output
 
 
    # ========================================================
    # Save Image
    # ========================================================
 
    def save_image(
        self,
        image: np.ndarray,
        output_path: str | Path
    ) -> Path:
 
        output_path = Path(
            output_path
        )
 
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
 
        cv2.imwrite(
            str(output_path),
            image
        )
 
        return output_path
 
 
    # ========================================================
    # Save Region Crops
    # ========================================================
 
    def save_region_crops(
        self,
        regions: List[ChangeRegion],
        output_folder: str | Path,
        prefix: str = "region"
    ) -> List[Path]:
 
        output_folder = Path(
            output_folder
        )
 
        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )
 
        saved: List[Path] = []
 
        for index, region in enumerate(
            regions,
            start=1
        ):
 
            if region.old_crop is not None:
 
                old_path = (
                    output_folder /
                    f"{prefix}_{index:03d}_old.png"
                )
 
                cv2.imwrite(
                    str(old_path),
                    region.old_crop
                )
 
                saved.append(
                    old_path
                )
 
            if region.new_crop is not None:
 
                new_path = (
                    output_folder /
                    f"{prefix}_{index:03d}_new.png"
                )
 
                cv2.imwrite(
                    str(new_path),
                    region.new_crop
                )
 
                saved.append(
                    new_path
                )
 
            if region.difference_crop is not None:
 
                diff_path = (
                    output_folder /
                    f"{prefix}_{index:03d}_diff.png"
                )
 
                cv2.imwrite(
                    str(diff_path),
                    region.difference_crop
                )
 
                saved.append(
                    diff_path
                )
 
        return saved
 
 
    # ========================================================
    # Region Dict
    # ========================================================
 
    def region_to_dict(
        self,
        region: ChangeRegion
    ) -> Dict[str, Any]:
 
        return {
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height,
            "area": region.area,
            "change_ratio": region.change_ratio,
            "region_type": region.region_type,
            "confidence": region.confidence,
        }
 
 
    # ========================================================
    # Result Dict
    # ========================================================
 
    def result_to_dict(
        self,
        result: ChangeDetectionResult
    ) -> Dict[str, Any]:
 
        return {
            "success": result.success,
            "regions": [
                self.region_to_dict(r)
                for r in result.regions
            ],
            "change_pixel_ratio": (
                result.change_pixel_ratio
            ),
            "reason": result.reason,
            "region_count": len(
                result.regions
            ),
        }
 
 
    # ========================================================
    # Summary
    # ========================================================
 
    def summary(
        self,
        result: ChangeDetectionResult
    ) -> Dict[str, Any]:
 
        types: Dict[str, int] = {}
 
        for region in result.regions:
 
            key = region.region_type
 
            types[key] = (
                types.get(key, 0) + 1
            )
 
        return {
            "success": result.success,
            "change_region_count": len(
                result.regions
            ),
            "change_pixel_ratio": (
                result.change_pixel_ratio
            ),
            "region_types": types,
            "reason": result.reason,
        }
 
 
# ============================================================
# Default detector
# ============================================================
 
_default_detector = ChangeDetector()
 