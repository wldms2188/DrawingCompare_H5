from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

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
    before_shape: Tuple[int,int]
    after_shape: Tuple[int,int]
    aligned_image: Optional[np.ndarray]=None
    transform_matrix: Optional[np.ndarray]=None
    match_count: int=0
    inlier_count: int=0
    inlier_ratio: float=0.0
    feature_method: str=""
    def to_dict(self)->Dict[str,Any]:
        return {k:v for k,v in self.__dict__.items() if k not in ("aligned_image","transform_matrix")}

class AlignmentEngine:
    """Conservative registration for engineering drawings.

    The old pipeline could accept a visually plausible ORB transform and then
    rotate/shear the entire drawing. That is especially harmful to tiny
    dimensions. H5 now estimates translation/scale only by default and rejects
    suspicious rotation or anisotropic distortion.
    """
    def __init__(self,min_matches=12,min_inlier_ratio=.35,accept_confidence=.70,review_confidence=.50,max_rotation=2.0,max_scale_change=.15):
        self.min_matches=int(min_matches); self.min_inlier_ratio=float(min_inlier_ratio)
        self.accept_confidence=float(accept_confidence); self.review_confidence=float(review_confidence)
        self.max_rotation=float(max_rotation); self.max_scale_change=float(max_scale_change)

    def align(self,before_image,after_image):
        if not isinstance(before_image,np.ndarray) or not isinstance(after_image,np.ndarray) or before_image.size==0 or after_image.size==0:
            return self._error("invalid image")
        bg=self._gray(before_image); ag=self._gray(after_image)
        # First choice: translation/scale from stable drawing content.
        r=self._align_orb_similarity(bg,ag)
        if r.success:return r
        # If registration is not trustworthy, DO NOT force a transform.
        # Same-page normalized coordinates are safer for text comparison.
        return AlignmentResult(False,"REVIEW",0.0,"alignment not confirmed; preserve native geometry",1,1,0,0,0,bg.shape[:2],ag.shape[:2],None,None,0,0,0,"NONE")

    @staticmethod
    def _gray(im):
        if im.ndim==2:return im.copy()
        if im.shape[2]==4:return cv2.cvtColor(im,cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    def _resize(self,g,max_side=1800):
        h,w=g.shape[:2]; s=min(1.0,max_side/max(h,w));
        if s==1:return g,1.0
        return cv2.resize(g,(max(1,int(w*s)),max(1,int(h*s))),interpolation=cv2.INTER_AREA),s

    def _align_orb_similarity(self,bg,ag):
        b,bs=self._resize(bg); a,as_=self._resize(ag)
        orb=cv2.ORB_create(nfeatures=7000,fastThreshold=10)
        kb,db=orb.detectAndCompute(b,None); ka,da=orb.detectAndCompute(a,None)
        if db is None or da is None or len(kb)<self.min_matches or len(ka)<self.min_matches:
            return self._failed(bg,ag,"insufficient features")
        ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
        good=[m[0] for m in ms if len(m)==2 and m[0].distance < .68*m[1].distance]
        if len(good)<self.min_matches:return self._failed(bg,ag,"insufficient good matches")
        src=np.float32([ka[m.queryIdx].pt for m in good]); dst=np.float32([kb[m.trainIdx].pt for m in good])
        M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
        if M is None or mask is None:return self._failed(bg,ag,"transform unavailable")
        inliers=int(mask.sum()); ratio=inliers/max(1,len(good))
        sx=float(np.hypot(M[0,0],M[0,1])); sy=float(np.hypot(M[1,0],M[1,1])); rot=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
        # Partial affine is allowed to scale/rotate, but H5 deliberately accepts
        # only near-zero rotation and nearly equal x/y scale.
        if ratio<self.min_inlier_ratio or abs(rot)>self.max_rotation or abs(sx-sy)>.025 or abs(sx-1)>self.max_scale_change:
            return self._failed(bg,ag,f"transform rejected: rot={rot:.2f}, sx={sx:.3f}, sy={sy:.3f}, inlier={ratio:.2f}")
        # Convert small-image transform to full-resolution coordinates.
        S_b=np.array([[bs,0,0],[0,bs,0],[0,0,1]],np.float64)
        S_a=np.array([[as_,0,0],[0,as_,0],[0,0,1]],np.float64)
        Mf=np.linalg.inv(S_b) @ np.vstack([M,[0,0,1]]) @ S_a
        Mf=Mf[:2]
        h,w=bg.shape; aligned=cv2.warpAffine(self._to_bgr(ag),Mf,(w,h),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
        conf=min(1.0,.5*ratio+.5*min(1,len(good)/50))
        return AlignmentResult(True,"ACCEPT" if conf>=self.accept_confidence else "REVIEW",conf,"conservative similarity alignment",sx,sy,rot,float(Mf[0,2]),float(Mf[1,2]),bg.shape[:2],ag.shape[:2],aligned,Mf,len(good),inliers,ratio,"ORB-SIMILARITY")

    @staticmethod
    def _to_bgr(g):return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR) if g.ndim==2 else g
    def _failed(self,bg,ag,reason):return AlignmentResult(False,"REVIEW",0,reason,1,1,0,0,0,bg.shape[:2],ag.shape[:2],None,None,0,0,0,"ORB-SIMILARITY")
    def _error(self,reason):return AlignmentResult(False,"ERROR",0,reason,1,1,0,0,0,(0,0),(0,0),None,None,0,0,0,"")
