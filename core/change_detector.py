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
    """H5: overlapping-tile comparison; whole-page registration is optional."""
    def __init__(self, config=None):
        self.pixel_threshold=38; self.grid_cols=5; self.grid_rows=4; self.overlap=.15
        try:
            import pytesseract; self.pytesseract=pytesseract
        except Exception:self.pytesseract=None

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray):return np.asarray(page)
        if hasattr(page,"image"):return np.asarray(page.image)
        if hasattr(page,"array"):return np.asarray(page.array)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")
    @staticmethod
    def _gray(img):
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _norm(s):return re.sub(r"\s+","",str(s).upper().replace("—","-").replace("–","-").replace("−","-"))
    @staticmethod
    def _class(t):
        t=str(t).upper()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|\|",t):return "gdt"
        if re.search(r"(?:\d+(?:\.\d+)?|\.\d+)(?:\s*(?:MM|IN|°|DEG))?$",t) or re.search(r"Ø|⌀|R\s*\d|M\d|[0-9]+\.[0-9]+",t):return "dimension"
        if any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE")):return "note"
        return "other"
    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path));p=doc.load_page(int(page.page_index));r=p.rect;out=[]
            for z in p.get_text("words"):
                x0,y0,x1,y1,text,*_=z;text=str(text).strip()
                if text:out.append({"text":text,"x":x0/r.width,"y":y0/r.height,"w":(x1-x0)/r.width,"h":(y1-y0)/r.height,"class":self._class(text)})
            doc.close();return out
        except Exception:return []
    def _tiles(self,img):
        h,w=img.shape[:2];tw=w/self.grid_cols;th=h/self.grid_rows;out=[]
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x0=max(0,int(c*tw-tw*self.overlap/2));x1=min(w,int((c+1)*tw+tw*self.overlap/2));y0=max(0,int(r*th-th*self.overlap/2));y1=min(h,int((r+1)*th+th*self.overlap/2));out.append((x0,y0,x1,y1))
        return out
    @staticmethod
    def _crop(img,b):x0,y0,x1,y1=b;return img[y0:y1,x0:x1]
    def _mapping_hint(self,before,after):
        bg=self._gray(before);ag=self._gray(after);h,w=bg.shape;ag=cv2.resize(ag,(w,h),interpolation=cv2.INTER_AREA)
        try:
            orb=cv2.ORB_create(nfeatures=5000,fastThreshold=10);k1,d1=orb.detectAndCompute(bg,None);k2,d2=orb.detectAndCompute(ag,None)
            if d1 is None or d2 is None:return None
            ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2);good=[m[0] for m in ms if len(m)==2 and m[0].distance<.75*m[1].distance]
            if len(good)<8:return None
            src=np.float32([k2[m.queryIdx].pt for m in good]);dst=np.float32([k1[m.trainIdx].pt for m in good]);M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=5)
            if M is None or mask is None or int(mask.sum())<6:return None
            return M
        except Exception:return None
    @staticmethod
    def _apply(p,M):return tuple((M[:,:2]@np.asarray(p)+M[:,2]).tolist()) if M is not None else p
    def _word(self,z,shape,M=None):
        h,w=shape[:2];p=((z["x"]+z["w"]/2)*w,(z["y"]+z["h"]/2)*h);p=self._apply(p,M);return {**z,"px":p[0],"py":p[1],"pw":z["w"]*w,"ph":z["h"]*h}
    def _words_in_tile(self,words,box,shape,M=None):
        x0,y0,x1,y1=box;out=[]
        for z in words:
            q=self._word(z,shape,M)
            if x0<=q["px"]<=x1 and y0<=q["py"]<=y1:out.append(q)
        return out
    def _tile_similarity(self,a,b):
        ga=cv2.resize(self._gray(a),(160,160),interpolation=cv2.INTER_AREA);gb=cv2.resize(self._gray(b),(160,160),interpolation=cv2.INTER_AREA)
        ea=cv2.Canny(ga,40,130);eb=cv2.Canny(gb,40,130)
        corr=float(np.corrcoef(ga.ravel(),gb.ravel())[0,1]) if ga.std()>1 and gb.std()>1 else 0
        ed=1-float(np.mean(np.abs(ea.astype(np.float32)-eb.astype(np.float32)))/255)
        return .55*max(0,corr)+.45*ed
    def _match_tiles(self,before,after,bt,at,bw,aw,M):
        used=set();pairs=[]
        for bi,bb in enumerate(bt):
            bwords=self._words_in_tile(bw,bb,before.shape);best=(-1,None)
            for ai,ab in enumerate(at):
                if ai in used:continue
                awords=self._words_in_tile(aw,ab,after.shape,M)
                sim=self._tile_similarity(self._crop(before,bb),self._crop(after,ab))
                common=sum(1 for x in bwords for y in awords if x["class"]==y["class"] and self._norm(x["text"])==self._norm(y["text"]))
                classes=sum(1 for x in bwords for y in awords if x["class"]==y["class"])
                score=.72*sim+.18*min(1,common/3)+.10*min(1,classes/8)
                if score>best[0]:best=(score,ai)
            if best[1] is not None and best[0]>=.28:used.add(best[1]);pairs.append((bi,best[1],best[0]))
        return pairs
    def _ocr_tile(self,img,box):
        if self.pytesseract is None:return []
        x0,y0,x1,y1=box;crop=img[y0:y1,x0:x1]
        if crop.size==0:return []
        g=self._gray(crop);scale=max(1,1600//max(1,min(g.shape)));g=cv2.resize(g,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);_,g=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        try:data=self.pytesseract.image_to_data(g,config="--psm 11",output_type=self.pytesseract.Output.DICT)
        except Exception:return []
        out=[]
        for i,t in enumerate(data.get("text",[])):
            t=str(t).strip();conf=float(data.get("conf",[-1])[i])
            if not t or conf<30:continue
            x=int(data["left"][i]/scale+x0);y=int(data["top"][i]/scale+y0);ww=int(data["width"][i]/scale);hh=int(data["height"][i]/scale);out.append({"text":t,"px":x+ww/2,"py":y+hh/2,"pw":ww,"ph":hh,"class":self._class(t),"ocr":True})
        return out
    def _compare_words(self,bw,aw,w,h):
        out=[];used=set()
        for o in bw:
            if o["class"]=="other":continue
            best=(-1,None)
            for j,n in enumerate(aw):
                if j in used or n["class"]!=o["class"]:continue
                dx=abs(o["px"]-n["px"])/w;dy=abs(o["py"]-n["py"])/h
                if dx>.06 or dy>.06:continue
                s=1-4*dx-4*dy
                if s>best[0]:best=(s,j)
            if best[1] is None:out.append((o,None,"deleted"));continue
            n=aw[best[1]];used.add(best[1])
            if self._norm(o["text"])!=self._norm(n["text"]):out.append((o,n,"changed"))
        for j,n in enumerate(aw):
            if j not in used and n["class"]!="other":out.append((None,n,"added"))
        return out
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);raw_after=self._img(after_page);after=self._img(aligned_after) if aligned_after is not None else raw_after;h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after));_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page);aw=self._native_words(after_page);M=self._mapping_hint(before,raw_after)
            # Whole-page mapping is a hint only. Tile matching always runs.
            self._last_before=before;self._last_after=after;bt=self._tiles(before);at=self._tiles(after);pairs=self._match_tiles(before,after,bt,at,bw,aw,M)
            candidates=[];added=deleted=0
            for bi,ai,ps in pairs:
                bb=bt[bi];ab=at[ai];bwt=self._words_in_tile(bw,bb,before.shape);awt=self._words_in_tile(aw,ab,after.shape,M)
                if not bwt:bwt=self._ocr_tile(before,bb)
                if not awt:awt=self._ocr_tile(after,ab)
                for o,n,kind in self._compare_words(bwt,awt,w,h):
                    q=o or n;x=int(max(0,q["px"]-q["pw"]/2-18));y=int(max(0,q["py"]-q["ph"]/2-18));x2=int(min(w,q["px"]+q["pw"]/2+18));y2=int(min(h,q["py"]+q["ph"]/2+18));local=diff[y:y2,x:x2]
                    if local.size and float(np.mean(local>self.pixel_threshold))<.0005:continue
                    candidates.append(ChangeRegion(x,y,x2-x,y2-y,(x2-x)*(y2-y),float(np.mean(local>self.pixel_threshold)) if local.size else 0,"text_"+kind,.65+.25*ps,before[y:y2,x:x2].copy(),after[y:y2,x:x2].copy(),local.copy()));added+=kind=="added";deleted+=kind=="deleted"
            merged=[]
            for r in candidates:
                if any(self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>.30 for m in merged):continue
                merged.append(r)
            reason=f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, whole_mapping={'ok' if M is not None else 'failed'}, tiles=20, tile_pairs={len(pairs)}, candidates={len(candidates)}, added={added}, deleted={deleted}, final={len(merged)}"
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3]);inter=max(0,x2-x1)*max(0,y2-y1);union=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,union)
