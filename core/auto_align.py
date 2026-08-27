from __future__ import annotations
import cv2
import numpy as np

class AutoAlign:
    """Engineering-drawing alignment: ORB first, ECC fallback.

    The returned image always has the Before page size. Alignment is used only
    to put corresponding local structures into the same coordinate system.
    """
    def __init__(self, max_rotation_deg: float = 12.0):
        self.max_rotation_deg = float(max_rotation_deg)

    @staticmethod
    def _gray(image):
        image = np.asarray(image)
        if image.ndim == 2: return image.astype(np.uint8)
        if image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _mask(gray):
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        m = max(8, int(min(h, w) * 0.015))
        mask[:m] = 0; mask[-m:] = 0; mask[:, :m] = 0; mask[:, -m:] = 0
        return mask

    def _orb_estimate(self, before_gray, after_gray):
        orb = cv2.ORB_create(nfeatures=7000, scaleFactor=1.2, nlevels=8, fastThreshold=8)
        kp1, des1 = orb.detectAndCompute(before_gray, self._mask(before_gray))
        kp2, des2 = orb.detectAndCompute(after_gray, self._mask(after_gray))
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8: return None
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des2, des1, k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75*p[1].distance]
        if len(good) < 8: return None
        good.sort(key=lambda m: m.distance); good = good[:400]
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=4000, confidence=0.995)
        if M is None or inliers is None: return None
        count = int(inliers.sum()); ratio = count/max(1,len(good))
        if count < 10 or ratio < 0.28: return None
        a,b,tx=M[0]; c,d,ty=M[1]
        det=a*d-b*c; scale=float(np.sqrt(abs(det)))
        rotation=float(np.degrees(np.arctan2(c-b,a+d)))
        if det <= 0 or not 0.50 <= scale <= 2.0 or abs(rotation)>self.max_rotation_deg: return None
        return M,count,ratio,scale,rotation

    def _ecc_estimate(self, before_gray, after_gray):
        """ECC is a useful fallback for CAD drawings with few ORB corners."""
        h,w=before_gray.shape[:2]
        # Downsample large pages to make ECC stable and fast.
        scale=min(1.0, 1400.0/max(h,w))
        size=(max(50,int(w*scale)),max(50,int(h*scale)))
        b=cv2.resize(before_gray,size,interpolation=cv2.INTER_AREA)
        a=cv2.resize(after_gray,size,interpolation=cv2.INTER_AREA)
        b=cv2.GaussianBlur(b,(5,5),0); a=cv2.GaussianBlur(a,(5,5),0)
        warp=np.eye(2,3,dtype=np.float32)
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5)
        try:
            cc,_=cv2.findTransformECC(b,a,warp,cv2.MOTION_AFFINE,criteria,None,5)
        except cv2.error:
            return None
        if scale != 1.0:
            # Coordinates are in downsampled space: translation must be restored.
            cc[0,2]/=scale; cc[1,2]/=scale
        A=cc[:,:2]
        det=float(np.linalg.det(A)); s=float(np.sqrt(abs(det)))
        rot=float(np.degrees(np.arctan2(A[1,0]-A[0,1],A[0,0]+A[1,1])))
        if det<=0 or not 0.75<=s<=1.30 or abs(rot)>self.max_rotation_deg: return None
        return cc,0,0.0,s,rot

    def align(self,before_img,after_img):
        before=np.asarray(before_img); after=np.asarray(after_img)
        if before.size==0 or after.size==0: raise ValueError("Before/After 이미지가 비어 있습니다.")
        h,w=before.shape[:2]
        if h<50 or w<50: return after.copy()
        bg=self._gray(before); ag=self._gray(after)
        if ag.shape != bg.shape: ag=cv2.resize(ag,(w,h),interpolation=cv2.INTER_AREA); after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
        try:
            result=self._orb_estimate(bg,ag)
            method="ORB"
            if result is None:
                result=self._ecc_estimate(bg,ag); method="ECC"
            if result is None:
                print("자동 정렬 보류 : 신뢰할 수 있는 변환을 찾지 못했습니다.")
                return after.copy()
            M,count,ratio,scale,rotation=result
            aligned=cv2.warpAffine(after,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            valid=np.full((h,w),255,np.uint8)
            valid=cv2.warpAffine(valid,M,(w,h),flags=cv2.INTER_NEAREST)
            valid_ratio=float(np.count_nonzero(valid))/max(1,w*h)
            if valid_ratio<0.70:
                print("자동 정렬 보류 : 유효 영역이 너무 많이 손실되었습니다.")
                return after.copy()
            if method=="ORB": print(f"자동 정렬 완료 : ORB inlier={count}, ratio={ratio:.2f}, scale={scale:.3f}, rotation={rotation:.2f}°, valid={valid_ratio:.2f}")
            else: print(f"자동 정렬 완료 : ECC scale={scale:.3f}, rotation={rotation:.2f}°, valid={valid_ratio:.2f}")
            return aligned
        except Exception as exc:
            print(f"자동 정렬 보류 : {exc}")
            return after.copy()
