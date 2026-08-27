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
    """Maps After into Before coordinates and exposes the exact transform used."""
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
    def _orb_estimate(self,bg,ag):
        orb=cv2.ORB_create(nfeatures=7000,scaleFactor=1.2,nlevels=8,fastThreshold=8); kp1,d1=orb.detectAndCompute(bg,self._mask(bg)); kp2,d2=orb.detectAndCompute(ag,self._mask(ag))
        if d1 is None or d2 is None or len(kp1)<8 or len(kp2)<8:return None
        knn=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2); good=[p[0] for p in knn if len(p)==2 and p[0].distance<.75*p[1].distance]
        if len(good)<8:return None
        good.sort(key=lambda m:m.distance); good=good[:400]; src=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2); dst=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=4000,confidence=.995)
        if M is None or inliers is None:return None
        count=int(inliers.sum()); ratio=count/max(1,len(good)); a,b,tx=M[0]; c,d,ty=M[1]; det=a*d-b*c; scale=float(np.sqrt(abs(det))); rotation=float(np.degrees(np.arctan2(c-b,a+d)))
        if count<10 or ratio<.28 or det<=0 or not .50<=scale<=2 or abs(rotation)>self.max_rotation_deg:return None
        return M,count,ratio,scale,rotation
    def _ecc_estimate(self,bg,ag):
        h,w=bg.shape[:2]; scale=min(1.,1400./max(h,w)); size=(max(50,int(w*scale)),max(50,int(h*scale))); b=cv2.resize(bg,size,interpolation=cv2.INTER_AREA); a=cv2.resize(ag,size,interpolation=cv2.INTER_AREA); b=cv2.GaussianBlur(b,(5,5),0); a=cv2.GaussianBlur(a,(5,5),0); warp=np.eye(2,3,dtype=np.float32); criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5)
        try: cc,_=cv2.findTransformECC(b,a,warp,cv2.MOTION_AFFINE,criteria,None,5)
        except cv2.error:return None
        if scale!=1.:cc[0,2]/=scale; cc[1,2]/=scale
        A=cc[:,:2]; det=float(np.linalg.det(A)); s=float(np.sqrt(abs(det))); rot=float(np.degrees(np.arctan2(A[1,0]-A[0,1],A[0,0]+A[1,1])))
        if det<=0 or not .75<=s<=1.30 or abs(rot)>self.max_rotation_deg:return None
        return cc,0,0.,s,rot
    def align(self,before_img,after_img):
        before=np.asarray(before_img); after=np.asarray(after_img); h,w=before.shape[:2]
        if before.size==0 or after.size==0:raise ValueError("Before/After 이미지가 비어 있습니다.")
        if h<50 or w<50:return AlignmentResult(after.copy(),None,"NONE",False)
        bg=self._gray(before); ag=self._gray(after)
        if ag.shape!=bg.shape: after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA); ag=self._gray(after)
        try:
            result=self._orb_estimate(bg,ag); method="ORB"
            if result is None:result=self._ecc_estimate(bg,ag); method="ECC"
            if result is None:
                print("자동 정렬 보류 : 신뢰할 수 있는 변환을 찾지 못했습니다.")
                return AlignmentResult(after.copy(),None,"NONE",False)
            M,count,ratio,scale,rotation=result
            flags=cv2.INTER_LINEAR if method=="ORB" else (cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            aligned=cv2.warpAffine(after,M,(w,h),flags=flags,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
            valid=np.full((h,w),255,np.uint8); vflags=cv2.INTER_NEAREST if method=="ORB" else (cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP); valid=cv2.warpAffine(valid,M,(w,h),flags=vflags); valid_ratio=float(np.count_nonzero(valid))/max(1,w*h)
            if valid_ratio<.70:
                print("자동 정렬 보류 : 유효 영역이 너무 많이 손실되었습니다.")
                return AlignmentResult(after.copy(),None,"NONE",False)
            print(f"자동 정렬 완료 : {method} scale={scale:.3f}, rotation={rotation:.2f}°, valid={valid_ratio:.2f}")
            return AlignmentResult(aligned,M,method,True,scale,rotation,valid_ratio)
        except Exception as exc:
            print(f"자동 정렬 보류 : {exc}")
            return AlignmentResult(after.copy(),None,"NONE",False)
