from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str="dimension_or_note"; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None
    difference_crop:Optional[np.ndarray]=None
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
    """Conservative detector: dimensions/GD&T/engineering notes first; geometry ignored."""
    def __init__(self, config=None):
        self.pixel_threshold=55; self.merge_distance=18; self.pytesseract=None
        try:
            import pytesseract; self.pytesseract=pytesseract
        except Exception: pass

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return page
        if hasattr(page,"image"): return np.asarray(page.image)
        if hasattr(page,"array"): return np.asarray(page.array)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")

    @staticmethod
    def _gray(img):
        if img.ndim==2: return img.astype(np.uint8)
        if img.shape[2]==4: return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    def _ocr(self,gray):
        if self.pytesseract is None:return []
        try:
            d=self.pytesseract.image_to_data(gray,config="--oem 3 --psm 11",output_type=self.pytesseract.Output.DICT)
            out=[]
            for i,t in enumerate(d.get("text",[])):
                t=(t or "").strip()
                try:c=float(d["conf"][i])
                except Exception:c=-1
                if not t or c<45:continue
                x,y,w,h=[int(d[k][i]) for k in ("left","top","width","height")]
                if w>=4 and h>=4:out.append((x,y,w,h,t,c))
            return out
        except Exception:return []

    @staticmethod
    def _norm(s):
        s=s.upper().replace("—","-").replace("–","-").replace("−","-")
        return re.sub(r"\s+","",s)

    @staticmethod
    def _target(t,h):
        t=t.upper(); numeric=bool(re.search(r"\d",t))
        engineering=bool(re.search(r"[Ø⌀R][0-9]|±|\+/-|\+\-|[0-9].*[A-Z]",t))
        note=any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE"))
        return (numeric and (engineering or h>=8)) or note

    def _find_changed(self,before,after,diff):
        old=[x for x in self._ocr(self._gray(before)) if self._target(x[4],x[3])]
        new=[x for x in self._ocr(self._gray(after)) if self._target(x[4],x[3])]
        out=[]; used=set()
        for o in old:
            ox,oy,ow,oh,ot,oc=o; best=None; bestscore=0
            for j,n in enumerate(new):
                if j in used:continue
                nx,ny,nw,nh,nt,nc=n
                dist=np.hypot((ox+ow/2)-(nx+nw/2),(oy+oh/2)-(ny+nh/2))
                maxdist=max(30,8*max(ow,oh,nw,nh))
                if dist>maxdist:continue
                score=.7*(1-dist/maxdist)+.3*max(0,1-abs(np.log(max(ow,1)/max(nw,1))))
                if score>bestscore:bestscore,best=score,j
            if best is None:continue
            n=new[best]; used.add(best); nx,ny,nw,nh,nt,nc=n
            if self._norm(ot)==self._norm(nt):continue
            x=max(0,min(ox,nx)-8); y=max(0,min(oy,ny)-8)
            x2=min(diff.shape[1],max(ox+ow,nx+nw)+8); y2=min(diff.shape[0],max(oy+oh,ny+nh)+8)
            local=diff[y:y2,x:x2]
            if local.size==0 or float(np.mean(local>self.pixel_threshold))<.04:continue
            out.append((x,y,x2-x,y2-y,"dimension_or_note",min(.99,.7+.25*bestscore)))
        return out

    def detect(self,before_page,after_page):
        try:
            before=self._img(before_page); after=self._img(after_page); h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gb,ga=self._gray(before),self._gray(after); diff=cv2.absdiff(gb,ga)
            _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            candidates=self._find_changed(before,after,diff)
            regions=[]
            for x,y,rw,rh,typ,conf in candidates:
                old=before[y:y+rh,x:x+rw].copy(); new=after[y:y+rh,x:x+rw].copy(); d=diff[y:y+rh,x:x+rw].copy()
                regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),typ,conf,old,new,d))
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),"dimension/GD&T/note priority")
        except Exception as exc:return ChangeDetectionResult(False,[],reason=str(exc))
