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
    """Text-first engineering drawing detector.
    Native PDF text is primary, OCR is secondary, raster heuristics are last.
    Geometry-only changes are intentionally excluded.
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
    def _target(text,h=8):
        t=str(text).upper(); return bool(re.search(r"\d|Ø|⌀|%%C|±|\+/-|\+\-|NOTE|TYP|UNLESS|MATERIAL|FINISH|BURR|INSPECT|SEE",t)) or h>=9
    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3]); inter=max(0,x2-x1)*max(0,y2-y1); union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/max(1,union)

    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); rect=p.rect; words=p.get_text("words"); doc.close(); out=[]
            for item in words:
                x0,y0,x1,y1,text,*_=item; text=str(text).strip()
                if text: out.append({"text":text,"x":x0/rect.width,"y":y0/rect.height,"w":(x1-x0)/rect.width,"h":(y1-y0)/rect.height})
            return out
        except Exception:return []

    def _native_text_changes(self,bp,ap,before,after,diff):
        old=self._native_words(bp); new=self._native_words(ap); old_t=[x for x in old if self._target(x["text"],x["h"]*before.shape[0])]; new_t=[x for x in new if self._target(x["text"],x["h"]*after.shape[0])]; used=set(); candidates=[]; h,w=before.shape[:2]
        for o in old_t:
            best=None; best_score=1e9
            for j,n in enumerate(new_t):
                if j in used: continue
                dx=abs(o["x"]+o["w"]/2-(n["x"]+n["w"]/2)); dy=abs(o["y"]+o["h"]/2-(n["y"]+n["h"]/2)); ds=abs(np.log(max(o["w"],1e-5)/max(n["w"],1e-5))); score=2*dx+2*dy+.25*min(ds,.8)
                if dx<.06 and dy<.05 and score<best_score: best_score=score; best=j
            if best is None: continue
            n=new_t[best]; used.add(best)
            if self._norm(o["text"])==self._norm(n["text"]): continue
            x1=int(max(0,(o["x"]-.014)*w)); y1=int(max(0,(o["y"]-.020)*h)); x2=int(min(w,(o["x"]+o["w"]+.014)*w)); y2=int(min(h,(o["y"]+o["h"]+.020)*h)); local=diff[y1:y2,x1:x2]; ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
            if ratio>=.001: candidates.append((x1,y1,x2-x1,y2-y1,.94))
        for j,n in enumerate(new_t):
            if j in used: continue
            near=any(abs(o["x"]+o["w"]/2-(n["x"]+n["w"]/2))<.035 and abs(o["y"]+o["h"]/2-(n["y"]+n["h"]/2))<.035 for o in old_t)
            if near: continue
            x1=int(max(0,(n["x"]-.014)*w)); y1=int(max(0,(n["y"]-.020)*h)); x2=int(min(w,(n["x"]+n["w"]+.014)*w)); y2=int(min(h,(n["y"]+n["h"]+.020)*h)); local=diff[y1:y2,x1:x2]
            if local.size and float(np.mean(local>self.pixel_threshold))>=.001:candidates.append((x1,y1,x2-x1,y2-y1,.92))
        return candidates,len(old),len(new),len(old_t),len(new_t)

    def _ocr(self,gray):
        if self.pytesseract is None:return []
        try:
            d=self.pytesseract.image_to_data(gray,config="--oem 3 --psm 11",output_type=self.pytesseract.Output.DICT); out=[]
            for i,t in enumerate(d.get("text",[])):
                t=(t or "").strip()
                try:c=float(d["conf"][i])
                except Exception:c=-1
                if t and c>=18:
                    x,y,w,h=[int(d[k][i]) for k in ("left","top","width","height")]
                    if w>=3 and h>=3: out.append((x,y,w,h,t,c))
            return out
        except Exception:return []

    def _image_text_fallback(self,before,after,diff,mask):
        gray=self._gray(before); h,w=gray.shape; small=cv2.threshold(gray,185,255,cv2.THRESH_BINARY_INV)[1]; horiz=cv2.morphologyEx(small,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,w//180),1))); vert=cv2.morphologyEx(small,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(9,h//180)))); textmask=cv2.subtract(small,cv2.bitwise_or(horiz,vert)); n,_,stats,_=cv2.connectedComponentsWithStats(textmask,8); out=[]
        for i in range(1,n):
            x,y,ww,hh,area=map(int,stats[i]);
            if area<5 or ww<2 or hh<2 or ww>.025*w or hh>.025*h:continue
            p=max(18,int(max(ww,hh)*3)); x1=max(0,x-p); y1=max(0,y-p); x2=min(w,x+ww+p); y2=min(h,y+hh+p); local=diff[y1:y2,x1:x2]; ratio=float(np.mean(local>self.pixel_threshold))
            if ratio<.025:continue
            changed=np.count_nonzero(local>self.pixel_threshold); box_area=local.shape[0]*local.shape[1]
            if changed>max(180,box_area*.22):continue
            out.append((x1,y1,x2-x1,y2-y1,.52))
        return out

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); after=self._img(after_page) if aligned_after is None else self._img(aligned_after); h,w=before.shape[:2]
            if after.shape[:2]!=(h,w): after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gb=self._gray(before); ga=self._gray(after); diff=cv2.absdiff(gb,ga); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            native,nob,noa,notg,natg=self._native_text_changes(before_page,after_page,before,after,diff); candidates=list(native); ob=oa=obt=oat=0; fallback=0
            if not candidates and self.pytesseract is not None:
                ob=len(self._ocr(gb)); oa=len(self._ocr(ga))
            if not candidates and notg==0 and natg==0:
                candidates=self._image_text_fallback(before,after,diff,mask); fallback=len(candidates)
            regions=[]
            for x,y,rw,rh,conf in candidates:
                d=diff[y:y+rh,x:x+rw]; regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),"dimension_or_note",conf,before[y:y+rh,x:x+rw].copy(),after[y:y+rh,x:x+rw].copy(),d.copy()))
            merged=[]
            for r in regions:
                for m in merged:
                    if self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>.15:
                        l=min(r.x,m.x); t=min(r.y,m.y); rr=max(r.right,m.right); bb=max(r.bottom,m.bottom); m.x,m.y,m.width,m.height=l,t,rr-l,bb-t; m.old_crop=before[t:bb,l:rr].copy(); m.new_crop=after[t:bb,l:rr].copy(); m.difference_crop=diff[t:bb,l:rr].copy(); m.confidence=max(m.confidence,r.confidence); m.change_ratio=float(np.mean(m.difference_crop>self.pixel_threshold)); break
                else: merged.append(r)
            reason=f"diag: native={nob}/{noa}, native_target={notg}/{natg}, ocr={ob}/{oa}, raw_diff={float(np.mean(mask>0)):.5f}, native_candidates={len(native)}, image_fallback={fallback}, final={len(merged)}"
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
