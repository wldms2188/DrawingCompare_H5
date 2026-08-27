from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class AlignmentResult:
    image: np.ndarray
    matrix: np.ndarray | None
    method: str
    success: bool
    scale: float = 1.0
    rotation: float = 0.0
    valid_ratio: float = 1.0

class AutoAlign:
    """Warp After into Before coordinates. Matrix always maps After -> Before."""
    def __init__(self,max_rotation_deg:float=12.0): self.max_rotation_deg=float(max_rotation_deg)
    @staticmethod
    def _gray(image):
        image=np.asarray(image)
        if image.ndim==2:return image.astype(np.uint8)
        if image.shape[2]==4:return cv2.cvtColor(image,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _mask(gray):
        h,w=gray.shape[:2]; mask=np.full((h,w),255,np.uint8); m=max(8,int(min(h,w)*.015)); mask[:m]=0; mask[-m:]=0; mask[:,:m]=0; mask[:,-m:]=0; return mask
    @staticmethod
    def _metrics(M,inliers,total,limit_deg):
        if M is None or inliers is None:return None
        count=int(np.count_nonzero(inliers)); ratio=count/max(1,total); a,b=M[0,:2]; c,d=M[1,:2]; det=float(a*d-b*c)
        scale=float(np.sqrt(abs(det))); rotation=float(np.degrees(np.arctan2(c-b,a+d)))
        if count<10 or ratio<.20 or det<=0 or not .50<=scale<=2 or abs(rotation)>limit_deg:return None
        return count,ratio,scale,rotation
    def _feature_estimate(self,bg,ag,kind='ORB'):
        if kind=='SIFT' and hasattr(cv2,'SIFT_create'):
            detector=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.015,edgeThreshold=10)
            norm=cv2.NORM_L2
        else:
            detector=cv2.ORB_create(nfeatures=10000,scaleFactor=1.15,nlevels=10,fastThreshold=5)
            norm=cv2.NORM_HAMMING
        kp1,d1=detector.detectAndCompute(bg,self._mask(bg)); kp2,d2=detector.detectAndCompute(ag,self._mask(ag))
        if d1 is None or d2 is None or len(kp1)<8 or len(kp2)<8:return None
        knn=cv2.BFMatcher(norm).knnMatch(d2,d1,k=2)
        good=[m[0] for m in knn if len(m)==2 and m[0].distance < (.78 if kind=='SIFT' else .80)*m[1].distance]
        if len(good)<8:return None
        good.sort(key=lambda m:m.distance); good=good[:800]
        src=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2); dst=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=6,maxIters=6000,confidence=.995)
        met=self._metrics(M,inliers,len(good),self.max_rotation_deg)
        if met is None:return None
        return M,*met,kind
    def _ecc_estimate(self,bg,ag):
        h,w=bg.shape[:2]; down=min(1.,1400./max(h,w)); size=(max(50,int(w*down)),max(50,int(h*down)))
        b=cv2.resize(bg,size,interpolation=cv2.INTER_AREA); a=cv2.resize(ag,size,interpolation=cv2.INTER_AREA)
        b=cv2.GaussianBlur(b,(5,5),0); a=cv2.GaussianBlur(a,(5,5),0); warp=np.eye(2,3,dtype=np.float32); criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,500,1e-6)
        try: score,cc=cv2.findTransformECC(b,a,warp,cv2.MOTION_AFFINE,criteria,None,5)
        except cv2.error:return None
        if down!=1.: cc[0,2]/=down; cc[1,2]/=down
        A=cc[:,:2]; det=float(np.linalg.det(A)); s=float(np.sqrt(abs(det))); rot=float(np.degrees(np.arctan2(A[1,0]-A[0,1],A[0,0]+A[1,1])))
        if det<=0 or not .70<=s<=1.40 or abs(rot)>self.max_rotation_deg or float(score)<.25:return None
        return cc,0,float(score),s,rot,'ECC'
    def align(self,before_img,after_img):
        before=np.asarray(before_img); after=np.asarray(after_img); h,w=before.shape[:2]
        if before.size==0 or after.size==0:raise ValueError('Before/After 이미지가 비어 있습니다.')
        if h<50 or w<50:return AlignmentResult(after.copy(),None,'NONE',False)
        bg=self._gray(before); ag=self._gray(after); resized=False
        if ag.shape!=bg.shape:
            after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA); ag=self._gray(after); resized=True
        try:
            result=self._feature_estimate(bg,ag,'SIFT')
            if result is None: result=self._feature_estimate(bg,ag,'ORB')
            if result is None: result=self._ecc_estimate(bg,ag)
            if result is None:
                print('자동 정렬 보류 : SIFT/ORB/ECC에서 신뢰할 수 있는 공통 구조를 찾지 못했습니다.')
                return AlignmentResult(after.copy(),None,'NONE',False)
            M,count,score,scale,rotation,method=result
            if method=='ECC':
                aligned=cv2.warpAffine(after,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
                valid_src=np.full((h,w),255,np.uint8); valid=cv2.warpAffine(valid_src,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            else:
                aligned=cv2.warpAffine(after,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
                valid_src=np.full((h,w),255,np.uint8); valid=cv2.warpAffine(valid_src,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            valid_ratio=float(np.count_nonzero(valid))/max(1,w*h)
            if valid_ratio<.70:
                print('자동 정렬 보류 : 유효 영역이 너무 많이 손실되었습니다.')
                return AlignmentResult(after.copy(),None,'NONE',False)
            if resized: print('자동 정렬 : After 크기를 Before 렌더링 크기에 맞춘 뒤 변환을 계산했습니다.')
            metric=score if method=='ECC' else float(count)
            print(f'자동 정렬 완료 : {method} metric={metric:.3f}, scale={scale:.3f}, rotation={rotation:.2f}°, valid={valid_ratio:.2f}')
            return AlignmentResult(aligned,M,method,True,scale,rotation,valid_ratio)
        except Exception as exc:
            print(f'자동 정렬 보류 : {type(exc).__name__}: {exc}')
            return AlignmentResult(after.copy(),None,'NONE',False)
