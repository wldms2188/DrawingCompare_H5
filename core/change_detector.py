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
    """Tile-first H5 detector.

    The page is divided into overlapping tiles. Each tile is analyzed at high
    resolution independently, then Before/After tiles are matched by their
    visual structure and native PDF text inventory. This prevents unrelated
    words from being paired merely because their global coordinates are close.
    Geometry-only differences are never reported.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38; self.grid_cols=5; self.grid_rows=4; self.overlap=.15
        try:
            import pytesseract; self.pytesseract=pytesseract
        except Exception: self.pytesseract=None

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
    def _class(t):
        t=str(t).upper()
        g=bool(re.search(r"(?:Ø|⌀|±|\+/-|\||POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC)",t))
        d=bool(re.search(r"(?:^|[^A-Z])(\d+(?:\.\d+)?|\.\d+)(?:\s*(?:MM|IN|°|DEG))?$",t)) or bool(re.search(r"(?:Ø|⌀|%%C|±|\+/-|R\s*\d|M\d|[0-9]+\.[0-9]+)",t))
        n=any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE"))
        return "gdt" if g else "dimension" if d else "note" if n else "other"
    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); r=p.rect; words=p.get_text("words"); doc.close(); out=[]
            for z in words:
                x0,y0,x1,y1,text,*_=z; text=str(text).strip()
                if text: out.append({"text":text,"x":x0/r.width,"y":y0/r.height,"w":(x1-x0)/r.width,"h":(y1-y0)/r.height,"class":self._class(text)})
            return out
        except Exception:return []
    def _tiles(self,img):
        h,w=img.shape[:2]; tw=w/self.grid_cols; th=h/self.grid_rows; tiles=[]
        for ry in range(self.grid_rows):
            for cx in range(self.grid_cols):
                x0=max(0,int(cx*tw-tw*self.overlap/2)); x1=min(w,int((cx+1)*tw+tw*self.overlap/2)); y0=max(0,int(ry*th-th*self.overlap/2)); y1=min(h,int((ry+1)*th+th*self.overlap/2)); tiles.append((x0,y0,x1,y1))
        return tiles
    def _tile_signature(self,img):
        g=self._gray(img); small=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA); edges=cv2.Canny(g,50,150); e=cv2.resize(edges,(16,16),interpolation=cv2.INTER_AREA); return np.concatenate([(small.astype(np.float32)-small.mean())/(small.std()+1),e.astype(np.float32)/255]).astype(np.float32)
    def _mapping(self,before,after):
        bg=self._gray(before); ag=self._gray(after); h,w=bg.shape; ag=cv2.resize(ag,(w,h),interpolation=cv2.INTER_AREA)
        try:
            orb=cv2.ORB_create(nfeatures=6000,fastThreshold=8); k1,d1=orb.detectAndCompute(bg,None); k2,d2=orb.detectAndCompute(ag,None)
            if d1 is None or d2 is None:return None
            ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2); good=[m[0] for m in ms if len(m)==2 and m[0].distance<.70*m[1].distance]
            if len(good)<10:return None
            src=np.float32([k2[m.queryIdx].pt for m in good]); dst=np.float32([k1[m.trainIdx].pt for m in good]); M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.5,maxIters=5000,confidence=.995)
            if M is None or mask is None or int(mask.sum())<10 or float(mask.sum())/len(good)<.35:return None
            a,b,_=M[0]; c,d,_=M[1]; scale=float(np.sqrt(abs(a*d-b*c))); rot=float(np.degrees(np.arctan2(c-b,a+d)))
            if not .70<=scale<=1.45 or abs(rot)>10:return None
            return M
        except Exception:return None
    @staticmethod
    def _transform_point(x,y,M): return tuple((M[:,:2]@np.array([x,y])+M[:,2]).tolist())
    def _word_px(self,w,shape,M=None):
        h,ww=shape[:2]; p=np.array([(w["x"]+w["w"]/2)*ww,(w["y"]+w["h"]/2)*h]);
        if M is not None:p=np.array(self._transform_point(*p,M))
        return {**w,"px":float(p[0]),"py":float(p[1]),"pw":w["w"]*ww,"ph":w["h"]*h}
    def _tile_words(self,words,box,shape,M=None):
        h,w=shape[:2]; x0,y0,x1,y1=box; out=[]
        for z in words:
            q=self._word_px(z,shape,M)
            if x0<=q["px"]<=x1 and y0<=q["py"]<=y1: out.append(q)
        return out
    def _best_tile_pairs(self,btiles,atiles,bwords,awords,bshape,ashape,M):
        pairs=[]; used=set()
        for bi,bt in enumerate(btiles):
            bw=self._tile_words(bwords,bt,bshape); sig_b=self._tile_text_signature(bw)
            best=None; bestscore=-1
            for ai,at in enumerate(atiles):
                if ai in used:continue
                aw=self._tile_words(awords,at,ashape,M); sig_a=self._tile_text_signature(aw)
                inter=sum(1 for x in sig_b for y in sig_a if x[0]==y[0] and self._norm(x[1])==self._norm(y[1]))
                cls=sum(1 for x in sig_b for y in sig_a if x[0]==y[0])
                # Visual tile signature is used as the anchor; text overlap is
                # a bonus, not a requirement, so newly added text is supported.
                bimg=self._tile_image(self._last_before,bt); aimg=self._tile_image(self._last_after,at)
                vs=float(np.linalg.norm(self._tile_signature(bimg)-self._tile_signature(aimg))); visual=1/(1+vs/50)
                score=.68*visual+.22*min(1,inter/3)+.10*min(1,cls/6)
                if score>bestscore:bestscore,best=score,ai
            if best is not None and bestscore>=.38:pairs.append((bi,best,bestscore)); used.add(best)
        return pairs
    @staticmethod
    def _tile_text_signature(words): return sorted([(w["class"],w["text"]) for w in words])
    def _tile_image(self,img,box): x0,y0,x1,y1=box; return img[y0:y1,x0:x1]
    def _compare_tile_words(self,bw,aw,w,h,diff,origin):
        # Match within the matched tile in aligned coordinates. Unmatched text
        # becomes add/delete; changed text becomes change. No OCR dependency.
        out=[]; used=set(); ox,oy=origin
        for o in bw:
            if o["class"]=="other":continue
            best=None; score=-1
            for j,n in enumerate(aw):
                if j in used or n["class"]!=o["class"]:continue
                dx=abs(o["px"]-n["px"])/w; dy=abs(o["py"]-n["py"])/h; sz=abs(np.log(max(o["pw"],1)/max(n["pw"],1)))
                if dx>.020 or dy>.018 or sz>.75:continue
                s=1-3*dx-3*dy-.2*min(sz,.75)
                if s>score:score,best=s,j
            if best is None:
                out.append((o,None,"deleted"));continue
            n=aw[best];used.add(best)
            if self._norm(o["text"])!=self._norm(n["text"]):out.append((o,n,"changed"))
        for j,n in enumerate(aw):
            if j not in used and n["class"]!="other":out.append((None,n,"added"))
        return out
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); raw_after=self._img(after_page); after=raw_after if aligned_after is None else self._img(aligned_after); h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after)); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page); aw=self._native_words(after_page); M=self._mapping(before,raw_after)
            if M is None:return ChangeDetectionResult(True,[],diff,mask,float(np.mean(mask>0)),f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, text_mapping=failed, tiles=20, tile_pairs=0, candidates=0, added=0, deleted=0, final=0")
            self._last_before=before; self._last_after=after; bt=self._tiles(before); at=self._tiles(after); pairs=self._best_tile_pairs(bt,at,bw,aw,before.shape,after.shape,M); candidates=[]; added=deleted=0
            for bi,ai,ps in pairs:
                bbox=bt[bi]; abox=at[ai]; bwt=self._tile_words(bw,bbox,before.shape); awt=self._tile_words(aw,abox,after.shape,M)
                for o,n,kind in self._compare_tile_words(bwt,awt,w,h,diff,(bbox[0],bbox[1])):
                    q=o or n; x=int(max(0,q["px"]-q["pw"]/2-14)); y=int(max(0,q["py"]-q["ph"]/2-14)); x2=int(min(w,q["px"]+q["pw"]/2+14)); y2=int(min(h,q["py"]+q["ph"]/2+14)); local=diff[y:y2,x:x2]
                    if local.size and float(np.mean(local>self.pixel_threshold))<.001:continue
                    candidates.append(ChangeRegion(x,y,x2-x,y2-y,(x2-x)*(y2-y),float(np.mean(local>self.pixel_threshold)) if local.size else 0,"dimension_or_gdt_or_note_"+kind,.80+0.15*ps,before[y:y2,x:x2].copy(),after[y:y2,x:x2].copy(),local.copy())); added+=kind=="added"; deleted+=kind=="deleted"
            # Remove duplicates caused by overlapping tiles.
            merged=[]
            for r in candidates:
                if any(self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>.35 for m in merged):continue
                merged.append(r)
            reason=f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, text_mapping=ok, tiles=20, tile_pairs={len(pairs)}, candidates={len(candidates)}, added={added}, deleted={deleted}, final={len(merged)}"
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
