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
    """Text-first engineering-drawing comparison.
    Native PDF text is the primary source. OCR is a local high-resolution
    supplement, rendered directly from the original vector PDF. Geometry itself
    is not treated as a change target.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.grid_cols=5; self.grid_rows=4
        self.overlap=.10
        self.local_radius=.12
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
        t=str(t).upper().strip()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|\|",t):return "gdt"
        if re.search(r"Ø|⌀|R\s*\d|M\d|[0-9]+\.[0-9]+|[0-9]+(?:\s*(?:MM|IN|°|DEG))?$",t):return "dimension"
        if any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE")):return "note"
        return "other"

    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); r=p.rect; out=[]
            for z in p.get_text("words"):
                x0,y0,x1,y1,text,*_=z; text=str(text).strip()
                if text:
                    out.append({"text":text,"x":x0/r.width,"y":y0/r.height,"w":(x1-x0)/r.width,"h":(y1-y0)/r.height,"class":self._class(text)})
            doc.close(); return out
        except Exception:return []

    def _tiles(self,img):
        h,w=img.shape[:2];tw=w/self.grid_cols;th=h/self.grid_rows;out=[]
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x0=max(0,int(c*tw-tw*self.overlap/2));x1=min(w,int((c+1)*tw+tw*self.overlap/2))
                y0=max(0,int(r*th-th*self.overlap/2));y1=min(h,int((r+1)*th+th*self.overlap/2))
                out.append((x0,y0,x1,y1))
        return out

    @staticmethod
    def _apply(p,M): return tuple((M[:,:2]@np.asarray(p)+M[:,2]).tolist()) if M is not None else p

    def _mapping_hint(self,before,after):
        bg=self._gray(before); ag=self._gray(after); h,w=bg.shape; ag=cv2.resize(ag,(w,h),interpolation=cv2.INTER_AREA)
        try:
            orb=cv2.ORB_create(nfeatures=5000,fastThreshold=10); k1,d1=orb.detectAndCompute(bg,None); k2,d2=orb.detectAndCompute(ag,None)
            if d1 is None or d2 is None:return None
            ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2); good=[m[0] for m in ms if len(m)==2 and m[0].distance<.70*m[1].distance]
            if len(good)<12:return None
            src=np.float32([k2[m.queryIdx].pt for m in good]); dst=np.float32([k1[m.trainIdx].pt for m in good])
            M,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4)
            if M is None or mask is None or int(mask.sum())<10:return None
            sx=float(np.hypot(M[0,0],M[0,1])); sy=float(np.hypot(M[1,0],M[1,1])); rot=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
            if not(.75<=sx<=1.35 and .75<=sy<=1.35 and abs(rot)<=12):return None
            return M
        except Exception:return None

    def _word(self,z,shape,M=None):
        h,w=shape[:2]; p=((z["x"]+z["w"]/2)*w,(z["y"]+z["h"]/2)*h); p=self._apply(p,M)
        return {**z,"px":p[0],"py":p[1],"pw":z["w"]*w,"ph":z["h"]*h}

    def _words_in_tile(self,words,box,shape,M=None):
        x0,y0,x1,y1=box; out=[]
        for z in words:
            q=self._word(z,shape,M)
            if x0<=q["px"]<=x1 and y0<=q["py"]<=y1:out.append(q)
        return out

    def _highres_ocr(self,page,box):
        """OCR a candidate directly from vector PDF at 1200 DPI.
        Returned coordinates are converted back to page.image pixels.
        """
        if self.pytesseract is None or not hasattr(page,"pdf_path"):return []
        try:
            from core.image_loader import ImageLoader
            hi=ImageLoader().render_region(page,box,dpi=1200,margin=240)
            if hi is None or hi.size==0:return []
            g=self._gray(hi)
            # Do not overprocess engineering strokes. Two conservative variants.
            variants=[g,cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]]
            x0,y0,x1,y1=box
            # render_region includes margin; approximate its page-pixel footprint
            # from the actual crop aspect ratio and center, then map OCR boxes back.
            # The exact mapping is recovered from the original PDF page rectangle.
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); pr=p.rect
            sx=page.width/pr.width; sy=page.height/pr.height
            mx=(240/page.width)*pr.width; my=(240/page.height)*pr.height
            rx0=max(pr.x0,(x0/page.width)*pr.width-mx); ry0=max(pr.y0,(y0/page.height)*pr.height-my)
            scale=1200/72.0
            out=[]
            for src in variants:
                try:data=self.pytesseract.image_to_data(src,config="--psm 11",output_type=self.pytesseract.Output.DICT)
                except Exception:continue
                for i,t in enumerate(data.get("text",[])):
                    t=str(t).strip()
                    try:conf=float(data.get("conf",[-1])[i])
                    except Exception:conf=-1
                    if not t or conf<20:continue
                    xx=float(data["left"][i])/scale+rx0; yy=float(data["top"][i])/scale+ry0
                    ww=float(data["width"][i])/scale; hh=float(data["height"][i])/scale
                    px=xx*sx; py=yy*sy; pw=ww*sx; ph=hh*sy
                    out.append({"text":t,"px":px,"py":py,"pw":pw,"ph":ph,"class":self._class(t),"ocr":True,"conf":conf})
            doc.close()
            uniq=[]
            for q in out:
                if any(self._norm(q["text"])==self._norm(u["text"]) and abs(q["px"]-u["px"])<max(12,q["pw"]) and abs(q["py"]-u["py"])<max(12,q["ph"]) for u in uniq):continue
                uniq.append(q)
            return uniq
        except Exception:return []

    def _merge_text(self,native,ocr):
        out=list(native)
        for q in ocr:
            if any(abs(q["px"]-n["px"])<max(15,n["pw"]*1.2) and abs(q["py"]-n["py"])<max(15,n["ph"]*1.8) for n in native):continue
            out.append(q)
        return out

    def _expanded_text_window(self,q,w,h):
        pad=max(90,int(max(q.get("pw",20),q.get("ph",20))*7.0))
        return (max(0,int(q["px"]-q.get("pw",20)/2-pad)),max(0,int(q["py"]-q.get("ph",20)/2-pad)),min(w,int(q["px"]+q.get("pw",20)/2+pad)),min(h,int(q["py"]+q.get("ph",20)/2+pad)))

    def _compare_words(self,bw,aw,w,h):
        out=[];used=set()
        for o in bw:
            if o["class"]=="other":continue
            best=(-1,None)
            for j,n in enumerate(aw):
                if j in used or n["class"]!=o["class"]:continue
                dx=abs(o["px"]-n["px"])/max(1,w); dy=abs(o["py"]-n["py"])/max(1,h)
                if dx>self.local_radius or dy>self.local_radius:continue
                sd=abs(o["ph"]-n["ph"])/max(1,max(o["ph"],n["ph"]))
                score=1-2.5*dx-2.5*dy-.25*sd
                if score>best[0]:best=(score,j)
            if best[1] is None:out.append((o,None,"deleted"));continue
            n=aw[best[1]];used.add(best[1])
            if self._norm(o["text"])!=self._norm(n["text"]):out.append((o,n,"changed"))
        for j,n in enumerate(aw):
            if j not in used and n["class"]!="other":out.append((None,n,"added"))
        return out

    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3]);inter=max(0,x2-x1)*max(0,y2-y1);union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/max(1,union)

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); raw_after=self._img(after_page); h,w=before.shape[:2]
            M=self._mapping_hint(before,raw_after)
            after=self._img(aligned_after) if aligned_after is not None else raw_after
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after)); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page); aw=self._native_words(after_page); bt=self._tiles(before)
            pairs=[]
            for i,bb in enumerate(bt):
                b=self._words_in_tile(bw,bb,before.shape); a=self._words_in_tile(aw,bb,after.shape,M)
                if any(z["class"]!="other" for z in b) or any(z["class"]!="other" for z in a):pairs.append((i,i,1.0))
            candidates=[];added=deleted=0;ocr_used=0
            for bi,_,score in pairs:
                box=bt[bi]
                bnative=self._words_in_tile(bw,box,before.shape)
                anative=self._words_in_tile(aw,box,after.shape,M)
                # Always supplement sparse/fragmented native text with a true PDF render.
                brocr=self._highres_ocr(before_page,box)
                aocr=self._highres_ocr(after_page,box)
                ocr_used+=bool(brocr)+bool(aocr)
                bwt=self._merge_text(bnative,brocr); awt=self._merge_text(anative,aocr)
                for o,n,kind in self._compare_words(bwt,awt,w,h):
                    q=o or n; x,y,x2,y2=self._expanded_text_window(q,w,h); local=diff[y:y2,x:x2]
                    ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0.0
                    if kind=="changed" and ratio<0.0003:continue
                    candidates.append(ChangeRegion(x,y,x2-x,y2-y,(x2-x)*(y2-y),ratio,"text_"+kind,.75,before[y:y2,x:x2].copy(),after[y:y2,x:x2].copy(),local.copy()))
                    added+=kind=="added";deleted+=kind=="deleted"
            merged=[]
            for r in candidates:
                if any(self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>.20 for m in merged):continue
                merged.append(r)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"whole_mapping={'ok' if M is not None else 'failed'}, tiles=20, tiles_pairs={len(pairs)}, candidates={len(candidates)}, "
                    f"added={added}, deleted={deleted}, ocr_local={ocr_used}, final={len(merged)}")
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")