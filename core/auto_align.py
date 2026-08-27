from __future__ import annotations
import cv2
import numpy as np

class AutoAlign:
    """Conservative affine alignment for engineering drawings."""
    def __init__(self, max_rotation_deg: float = 12.0):
        self.max_rotation_deg = float(max_rotation_deg)

    @staticmethod
    def _gray(image):
        if image.ndim == 2: return image
        if image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _mask(gray):
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        m = max(8, int(min(h, w) * 0.015))
        mask[:m] = 0; mask[-m:] = 0; mask[:, :m] = 0; mask[:, -m:] = 0
        return mask

    def _estimate(self, before_gray, after_gray):
        orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        kp1, des1 = orb.detectAndCompute(before_gray, self._mask(before_gray))
        kp2, des2 = orb.detectAndCompute(after_gray, self._mask(after_gray))
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8: return None
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des2, des1, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance: good.append(pair[0])
        if len(good) < 8: return None
        good.sort(key=lambda m: m.distance); good = good[:250]
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000, confidence=0.995)
        if M is None or inliers is None: return None
        count = int(inliers.sum()); ratio = count / max(1, len(good))
        if count < 12 or ratio < 0.35: return None
        a,b,tx = M[0]; c,d,ty = M[1]
        det = a*d-b*c; scale = float(np.sqrt(abs(det)))
        rotation = float(np.degrees(np.arctan2(c-b, a+d)))
        if det <= 0 or not 0.45 <= scale <= 2.2 or abs(rotation) > self.max_rotation_deg: return None
        return M, count, ratio, scale, rotation

    def align(self, before_img, after_img):
        before = np.asarray(before_img); after = np.asarray(after_img)
        if before.size == 0 or after.size == 0: raise ValueError("Before/After 이미지가 비어 있습니다.")
        h,w = before.shape[:2]
        if h < 50 or w < 50: return after.copy()
        try:
            result = self._estimate(self._gray(before), self._gray(after))
            if result is None:
                print("자동 정렬 보류 : 신뢰할 수 있는 변환을 찾지 못했습니다.")
                return cv2.resize(after, (w,h), interpolation=cv2.INTER_AREA)
            M,count,ratio,scale,rotation = result
            aligned = cv2.warpAffine(after, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            valid = np.full(after.shape[:2],255,np.uint8)
            valid = cv2.warpAffine(valid,M,(w,h),flags=cv2.INTER_NEAREST)
            valid_ratio = float(np.count_nonzero(valid))/max(1,w*h)
            if valid_ratio < 0.70:
                print("자동 정렬 보류 : 유효 영역이 너무 많이 손실되었습니다.")
                return cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            print(f"자동 정렬 완료 : inlier={count}, ratio={ratio:.2f}, scale={scale:.3f}, rotation={rotation:.2f}°, valid={valid_ratio:.2f}")
            return aligned
        except Exception as exc:
            print(f"자동 정렬 보류 : {exc}")
            return cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
