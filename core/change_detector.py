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
    """Conservative drawing detector. Geometry is not reported.
    OCR is optional: when Tesseract is unavailable, character-like image
    evidence around a local structure is used instead of returning zero.
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

    def _ocr(self,gray):
        if self.pytesseract is None:return []
        try:
            d=self.pytesseract.image_to_data(gray,config="--oem 3 --psm 11",output_type=self.pytesseract.Output.DICT); out=[]
            for i,t in enumerate(d.get("text",[])):
                t=(t or "").strip()
                try:c=float(d["conf"][i])
                except Exception:c=-1
                if not t or c<18:continue
                x,y,w,h=[int(d[k][i]) for k in ("left","top","width","height")]
                if w>=3 and h>=3:out.append((x,y,w,h,t,c))
            return out
        except Exception:return []

    @staticmethod
    def _norm(s): return re.sub(r"\s+","",s.upper().replace("—","-").replace("–","-").replace("−","-"))

    @staticmethod
    def _target(text,h):
        t=text.upper(); numeric=bool(re.search(r"\d",t)); engineering=bool(re.search(r"(?:Ø|⌀|R\s*\d|±|\+/-|\+\-|\d+(?:\.\d+)?[A-Z])",t)); note=any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE"))
        return note or (numeric and (engineering or h>=6))

    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3]); inter=max(0,x2-x1)*max(0,y2-y1); union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/max(1,union)

    def _blocks(self,gray):
        h,w=gray.shape[:2]; _,ink=cv2.threshold(gray,215,255,cv2.THRESH_BINARY_INV); k=max(5,min(31,(min(h,w)//180)*2+1)); joined=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))); n,_,stats,_=cv2.connectedComponentsWithStats(joined,8); blocks=[]
        for i in range(1,n):
            x,y,ww,hh,area=map(int,stats[i])
            if area<max(180,int(h*w*0.000025)) or ww<20 or hh<15 or ww>.85*w or hh>.80*h:continue
            if ww>.65*w and hh<.10*h:continue
            if hh>.65*h and ww<.10*w:continue
            px=max(25,int(w*.02)); py=max(25,int(h*.02)); blocks.append((max(0,x-px),max(0,y-py),min(w,x+ww+2*px)-max(0,x-px),min(h,y+hh+2*py)-max(0,y-py)))
        cols=4 if w/h>1.5 else 3; rows=3 if h/w>1.5 else 2; sx=w/cols; sy=h/rows
        for r in range(rows):
            for c in range(cols):
                x=max(0,int((c-.15)*sx)); y=max(0,int((r-.15)*sy)); x2=min(w,int((c+1.15)*sx)); y2=min(h,int((r+1.15)*sy)); blocks.append((x,y,x2-x,y2-y))
        merged=[]
        for b in blocks:
            for j,m in enumerate(merged):
                if self._iou(b,m)>.25:
                    x=min(b[0],m[0]); y=min(b[1],m[1]); x2=max(b[0]+b[2],m[0]+m[2]); y2=max(b[1]+b[3],m[1]+m[3]); merged[j]=(x,y,x2-x,y2-y); break
            else: merged.append(b)
        return merged

    @staticmethod
    def _block(box,blocks):
        if not blocks:return -1
        cx=box[0]+box[2]/2; cy=box[1]+box[3]/2; inside=[(i,b) for i,b in enumerate(blocks) if b[0]-12<=cx<=b[0]+b[2]+12 and b[1]-12<=cy<=b[1]+b[3]+12]
        if inside:return min(inside,key=lambda z:z[1][2]*z[1][3])[0]
        return min(range(len(blocks)),key=lambda i:(cx-(blocks[i][0]+blocks[i][2]/2))**2+(cy-(blocks[i][1]+blocks[i][3]/2))**2)

    def _ocr_pairs(self,before,after,diff,blocks):
        old_all=self._ocr(self._gray(before)); new_all=self._ocr(self._gray(after)); old=[x for x in old_all if self._target(x[4],x[3])]; new=[x for x in new_all if self._target(x[4],x[3])]; gb={i:[] for i in range(len(blocks))}; ga={i:[] for i in range(len(blocks))}
        for x in old:gb[self._block(x,blocks)].append(x)
        for x in new:ga[self._block(x,blocks)].append(x)
        out=[]
        for bi in range(len(blocks)):
            used=set()
            for o in gb[bi]:
                ox,oy,ow,oh,ot,oc=o; best=None; bs=0
                for j,n in enumerate(ga[bi]):
                    if j in used:continue
                    nx,ny,nw,nh,nt,nc=n; dist=np.hypot(ox+ow/2-(nx+nw/2),oy+oh/2-(ny+nh/2)); lim=max(60,16*max(ow,oh,nw,nh))
                    if dist>lim:continue
                    score=.75*(1-dist/lim)+.25*max(0,1-abs(np.log(max(ow,1)/max(nw,1))))
                    if score>bs:bs,best=score,j
                if best is None:continue
                nx,ny,nw,nh,nt,nc=ga[bi][best]; used.add(best)
                if self._norm(ot)==self._norm(nt):continue
                x=max(0,min(ox,nx)-16); y=max(0,min(oy,ny)-16); x2=min(diff.shape[1],max(ox+ow,nx+nw)+16); y2=min(diff.shape[0],max(oy+oh,ny+nh)+16); local=diff[y:y2,x:x2]
                if local.size and float(np.mean(local>self.pixel_threshold))>=.004:out.append((x,y,x2-x,y2-y,.65+.30*bs))
        return out,len(old_all),len(new_all),len(old),len(new)

    def _image_text_fallback(self,before,after,diff,mask):
        """OCR-free fallback: find small, dense changed components whose local
        neighborhood contains repeated character-like strokes. It is much more
        conservative than accepting every geometric difference."""
        gray=self._gray(before); h,w=gray.shape; small=cv2.threshold(gray,185,255,cv2.THRESH_BINARY_INV)[1]
        # Remove long construction/dimension lines; retain compact text-like blobs.
        horiz=cv2.morphologyEx(small,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,w//180),1)))
        vert=cv2.morphologyEx(small,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(9,h//180))))
        textmask=cv2.subtract(small,cv2.bitwise_or(horiz,vert)); n,_,stats,_=cv2.connectedComponentsWithStats(textmask,8); out=[]
        for i in range(1,n):
            x,y,ww,hh,area=map(int,stats[i]);
            if area<5 or ww<2 or hh<2 or ww>.025*w or hh>.025*h:continue
            p=max(18,int(max(ww,hh)*3)); x1=max(0,x-p); y1=max(0,y-p); x2=min(w,x+ww+p); y2=min(h,y+hh+p); local=diff[y1:y2,x1:x2]; ratio=float(np.mean(local>self.pixel_threshold))
            if ratio<.025:continue
            # Require changed pixels to be compact rather than a large global mismatch.
            changed=np.count_nonzero(local>self.pixel_threshold); box_area=local.shape[0]*local.shape[1]
            if changed>max(180,box_area*.22):continue
            out.append((x1,y1,x2-x1,y2-y1,.52))
        return out

    def detect(self,before_page,after_page):
        try:
            before=self._img(before_page); after=self._img(after_page); h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gb=self._gray(before); ga=self._gray(after); diff=cv2.absdiff(gb,ga); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY); blocks=self._blocks(gb)
            candidates,ob,oa,ot,at=self._ocr_pairs(before,after,diff,blocks); fallback_count=0
            if not candidates:
                fallback=self._image_text_fallback(before,after,diff,mask); candidates=fallback; fallback_count=len(fallback)
            regions=[]
            for x,y,rw,rh,conf in candidates:
                old=before[y:y+rh,x:x+rw].copy(); new=after[y:y+rh,x:x+rw].copy(); d=diff[y:y+rh,x:x+rw].copy(); regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),"dimension_or_note",conf,old,new,d))
            merged=[]
            for r in regions:
                for m in merged:
                    if self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>.15:
                        l=min(r.x,m.x); t=min(r.y,m.y); rr=max(r.right,m.right); bb=max(r.bottom,m.bottom); m.x,m.y,m.width,m.height=l,t,rr-l,bb-t; m.confidence=max(m.confidence,r.confidence); m.old_crop=before[t:bb,l:rr].copy(); m.new_crop=after[t:bb,l:rr].copy(); m.difference_crop=diff[t:bb,l:rr].copy(); m.change_ratio=float(np.mean(m.difference_crop>self.pixel_threshold)); break
                else: merged.append(r)
            ocr_state="available" if self.pytesseract is not None else "unavailable"
            reason=(f"diag: ocr={ob}/{oa}, target={ot}/{at}, ocr_state={ocr_state}, blocks={len(blocks)}, " f"raw_diff={float(np.mean(mask>0)):.5f}, ocr_candidates={len(candidates)-fallback_count}, " f"image_fallback={fallback_count}, final={len(merged)}")
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
