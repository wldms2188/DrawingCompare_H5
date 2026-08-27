from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str="dimension_or_note"; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None
    @property
    def left(self): return self.x
    @property
    def top(self): return self.y
    @property
    def right(self): return self.x+self.width
    @property
    def bottom(self): return self.y+self.height

@dataclass
class ChangeDetectionResult:
    success:bool; regions:List[ChangeRegion]=field(default_factory=list)
    difference_image:Optional[np.ndarray]=None; threshold_image:Optional[np.ndarray]=None
    change_pixel_ratio:float=0.0; reason:str=""
    @property
    def region(self): return self.regions

class ChangeDetector:
    """H5: report only actual dimension/GD&T/note text changes.

    Important: a changed PDF word is NOT sufficient evidence. A candidate must
    also be located in a dimension/note context. Standalone title blocks,
    revision tables, labels and unrelated text are rejected. Geometry-only
    raster differences are never used as a fallback.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38; self.pytesseract=None
        try:
            import pytesseract; self.pytesseract=pytesseract
        except Exception: pass

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return np.asarray(page)
        if hasattr(page,"image"): return np.asarray(page.image)
        if hasattr(page,"array"): return np.asarray(page.array)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")
    @staticmethod
    def _gray(img):
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _norm(s): return re.sub(r"\s+","",str(s).upper().replace("—","-").replace("–","-").replace("−","-"))
    @staticmethod
    def _is_dimension_token(t):
        t=str(t).upper();
        return bool(re.search(r"(?:^|[^A-Z])(\d+(?:\.\d+)?|\.\d+)(?:\s*(?:MM|IN|°|DEG))?$",t)) or bool(re.search(r"(?:Ø|⌀|%%C|±|\+/-|R\s*\d|M\d|[0-9]+\.[0-9]+)",t))
    @staticmethod
    def _is_gdt_token(t):
        t=str(t).upper(); return bool(re.search(r"(?:Ø|⌀|±|\+/-|\|)|(?:POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC)",t))
    @staticmethod
    def _is_note_token(t):
        t=str(t).upper(); return any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE"))
    @staticmethod
    def _target(text,h=8):
        return ChangeDetector._is_dimension_token(text) or ChangeDetector._is_gdt_token(text) or ChangeDetector._is_note_token(text)
    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3]); inter=max(0,x2-x1)*max(0,y2-y1); union=a[2]*a[3]+b[2]*b[3]-inter; return inter/max(1,union)

    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); rect=p.rect; words=p.get_text("words"); doc.close(); out=[]
            for item in words:
                x0,y0,x1,y1,text,*_=item; text=str(text).strip()
                if text: out.append({"text":text,"x":x0/rect.width,"y":y0/rect.height,"w":(x1-x0)/rect.width,"h":(y1-y0)/rect.height})
            return out
        except Exception:return []

    def _estimate_mapping(self,before,after):
        bg=self._gray(before); ag=self._gray(after); h,w=bg.shape
        if ag.shape!=bg.shape: ag=cv2.resize(ag,(w,h),interpolation=cv2.INTER_AREA)
        try:
            orb=cv2.ORB_create(nfeatures=6000,fastThreshold=8); k1,d1=orb.detectAndCompute(bg,None); k2,d2=orb.detectAndCompute(ag,None)
            if d1 is None or d2 is None:return None
            matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2); good=[m[0] for m in matches if len(m)==2 and m[0].distance<.70*m[1].distance]
            if len(good)<10:return None
            src=np.float32([k2[m.queryIdx].pt for m in good]); dst=np.float32([k1[m.trainIdx].pt for m in good]); M,inl=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.5,maxIters=5000,confidence=.995)
            if M is None or inl is None:return None
            ratio=float(inl.sum())/len(good); a,b,tx=M[0]; c,d,ty=M[1]; scale=float(np.sqrt(abs(a*d-b*c))); rot=float(np.degrees(np.arctan2(c-b,a+d)))
            if int(inl.sum())<10 or ratio<.35 or not .70<=scale<=1.45 or abs(rot)>10:return None
            return M,ratio,scale,rot
        except Exception:return None

    @staticmethod
    def _transform_word(word,M,before_shape):
        h,w=before_shape[:2]; cx=(word["x"]+word["w"]/2)*w; cy=(word["y"]+word["h"]/2)*h; p=M[:,:2]@np.array([cx,cy])+M[:,2]; sx=max(.001,float(np.hypot(M[0,0],M[1,0]))); sy=max(.001,float(np.hypot(M[0,1],M[1,1]))); return {**word,"px":float(p[0]),"py":float(p[1]),"pw":word["w"]*w*sx,"ph":word["h"]*h*sy}

    def _context_score(self,word,all_words,shape):
        h,w=shape[:2]; x=word["px"]; y=word["py"]; radius_x=.045*w; radius_y=.035*h
        near=[]
        for q in all_words:
            if q is word: continue
            if abs(q["px"]-x)<radius_x and abs(q["py"]-y)<radius_y: near.append(q)
        # Dimension/GD&T context is strongest when there is a nearby leader,
        # datum letter, tolerance token, unit, or another numeric token.
        score=0
        for q in near:
            t=q["text"].upper()
            if self._is_dimension_token(t): score+=2
            if self._is_gdt_token(t): score+=2
            if re.search(r"(?:DATUM|A|B|C)$",t): score+=1
            if re.search(r"(?:MM|IN|DEG|°)$",t): score+=2
        # Isolated large text is much more likely to be a title/label.
        if word["pw"]>.06*w or word["ph"]>.035*h: score-=2
        return score

    def _native_text_changes(self,bp,ap,before,after,diff,mapping):
        old=self._native_words(bp); new=self._native_words(ap); h,w=before.shape[:2]
        old_t=[x for x in old if self._target(x["text"],x["h"]*h)]; new_t=[x for x in new if self._target(x["text"],x["h"]*after.shape[0])]
        for x in old_t:x["px"]=(x["x"]+x["w"]/2)*w; x["py"]=(x["y"]+x["h"]/2)*h; x["pw"]=x["w"]*w; x["ph"]=x["h"]*h
        if not mapping:
            return [],len(old),len(new),len(old_t),len(new_t),0
        M=mapping[0]
        for x in new_t:x.update(self._transform_word(x,M,before.shape))
        # Build context over all native words, not only numeric candidates.
        old_all=[]; new_all=[]
        for x in old:
            x={**x,"px":(x["x"]+x["w"]/2)*w,"py":(x["y"]+x["h"]/2)*h,"pw":x["w"]*w,"ph":x["h"]*h}; old_all.append(x)
        for x in new:
            x=self._transform_word(x,M,before.shape); new_all.append(x)
        used=set(); candidates=[]; rejected=0
        for o in old_t:
            best=None; best_score=-999
            for j,n in enumerate(new_t):
                if j in used:continue
                dx=abs(o["px"]-n["px"])/w; dy=abs(o["py"]-n["py"])/h; size=abs(np.log(max(o["pw"],1)/max(n["pw"],1)))
                if dx>.018 or dy>.015 or size>.65:continue
                # Same semantic class is required.
                cls=lambda z: (self._is_gdt_token(z["text"]),self._is_dimension_token(z["text"]),self._is_note_token(z["text"]))
                if cls(o)!=cls(n):continue
                score=1-(dx*3+dy*3+min(size,.65)*.20)
                if score>best_score:best_score,best=score,j
            if best is None:continue
            n=new_t[best]; used.add(best)
            if self._norm(o["text"])==self._norm(n["text"]):continue
            context=max(self._context_score(o,old_all,before.shape),self._context_score(n,new_all,before.shape))
            # Crucial: standalone changed text is not a reportable drawing
            # dimension. Require local engineering context.
            if context<2:
                rejected+=1; continue
            x1=int(max(0,min(o["px"]-o["pw"]/2,n["px"]-n["pw"]/2)-max(10,int(.003*w)))); y1=int(max(0,min(o["py"]-o["ph"]/2,n["py"]-n["ph"]/2)-max(10,int(.003*h)))); x2=int(min(w,max(o["px"]+o["pw"]/2,n["px"]+n["pw"]/2)+max(10,int(.003*w)))); y2=int(min(h,max(o["py"]+o["ph"]/2,n["py"]+n["ph"]/2)+max(10,int(.003*h)))); local=diff[y1:y2,x1:x2]; ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
            if ratio>=.0015:candidates.append((x1,y1,x2-x1,y2-y1,.97))
        return candidates,len(old),len(new),len(old_t),len(new_t),rejected

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); after=self._img(after_page) if aligned_after is None else self._img(aligned_after); h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gb=self._gray(before); ga=self._gray(after); diff=cv2.absdiff(gb,ga); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            mapping=self._estimate_mapping(before,self._img(after_page)); native,nob,noa,notg,natg,rejected=self._native_text_changes(before_page,after_page,before,after,diff,mapping)
            regions=[]
            for x,y,rw,rh,conf in native:
                d=diff[y:y+rh,x:x+rw]; regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),"dimension_or_gdt_or_note",conf,before[y:y+rh,x:x+rw].copy(),after[y:y+rh,x:x+rw].copy(),d.copy()))
            reason=f"diag: native={nob}/{noa}, native_target={notg}/{natg}, text_mapping={'ok' if mapping else 'failed'}, raw_diff={float(np.mean(mask>0)):.5f}, native_candidates={len(native)}, context_rejected={rejected}, image_fallback=0, final={len(regions)}"
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
