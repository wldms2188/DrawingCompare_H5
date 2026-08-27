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
    old_text:str=""; new_text:str=""; change_kind:str=""
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
    """H5 text-first drawing comparison with duplicate suppression."""
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.match_radius=.075
        self.cluster_gap=.018
        try:
            import pytesseract; self.pytesseract=pytesseract
        except Exception:self.pytesseract=None

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray):return np.asarray(page)
        if hasattr(page,"image"):return np.asarray(page.image)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")
    @staticmethod
    def _gray(img):
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _norm(s):return re.sub(r"[^A-Z0-9Ø⌀±+\-.*/X°]","",str(s).upper().replace("—","-").replace("–","-").replace("−","-"))
    @staticmethod
    def _class(t):
        t=str(t).upper().strip()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±",t):return "gdt"
        if re.search(r"(?:^|\s)(?:R|M)\s*\d|[0-9]+\.[0-9]+|[0-9]+(?:\s*(?:MM|IN|°|DEG))?$",t):return "dimension"
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
    def _word(self,z,shape):
        h,w=shape[:2];return {**z,"px":(z["x"]+z["w"]/2)*w,"py":(z["y"]+z["h"]/2)*h,"pw":max(1,z["w"]*w),"ph":max(1,z["h"]*h)}
    def _cluster(self,words,shape):
        q=sorted([self._word(z,shape) for z in words if z["class"]!="other"],key=lambda z:(z["py"],z["px"]))
        groups=[]
        for z in q:
            hit=None
            for g in groups:
                cy=np.mean([a["py"] for a in g]);right=max(a["px"]+a["pw"]/2 for a in g);left=z["px"]-z["pw"]/2
                if abs(z["py"]-cy)<=max(z["ph"],np.mean([a["ph"] for a in g]))*1.35 and left-right<=shape[1]*self.cluster_gap:hit=g;break
            if hit is None:groups.append([z])
            else:hit.append(z)
        out=[]
        for g in groups:
            g.sort(key=lambda a:a["px"]);x0=min(a["px"]-a["pw"]/2 for a in g);y0=min(a["py"]-a["ph"]/2 for a in g);x1=max(a["px"]+a["pw"]/2 for a in g);y1=max(a["py"]+a["ph"]/2 for a in g)
            out.append({"text":" ".join(a["text"] for a in g),"px":(x0+x1)/2,"py":(y0+y1)/2,"pw":x1-x0,"ph":y1-y0,"class":"gdt" if any(a["class"]=="gdt" for a in g) else "dimension","parts":g})
        return out
    def _highres_ocr(self,page,box):
        if self.pytesseract is None or not hasattr(page,"pdf_path"):return []
        try:
            from core.image_loader import ImageLoader
            hi=ImageLoader().render_region(page,box,dpi=1200,margin=240)
            if hi is None or hi.size==0:return []
            g=self._gray(hi);variants=[g,cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]];scale=1200/72.0
            import fitz
            doc=fitz.open(Path(page.pdf_path));pr=doc.load_page(int(page.page_index)).rect
            x0,y0,_,_=box;rx0=(x0/page.width)*pr.width;ry0=(y0/page.height)*pr.height
            sx=page.width/pr.width;sy=page.height/pr.height;out=[]
            for src in variants:
                try:d=self.pytesseract.image_to_data(src,config="--psm 11",output_type=self.pytesseract.Output.DICT)
                except Exception:continue
                for i,t in enumerate(d.get("text",[])):
                    t=str(t).strip()
                    try:c=float(d["conf"][i])
                    except Exception:c=-1
                    if not t or c<20:continue
                    xx=float(d["left"][i])/scale+rx0;yy=float(d["top"][i])/scale+ry0;ww=float(d["width"][i])/scale;hh=float(d["height"][i])/scale
                    out.append({"text":t,"px":xx*sx,"py":yy*sy,"pw":ww*sx,"ph":hh*sy,"class":self._class(t),"ocr":True,"conf":c})
            doc.close();return out
        except Exception:return []
    def _box(self,q,w,h):
        pad=max(100,int(max(q["pw"],q["ph"])*7));return(max(0,int(q["px"]-q["pw"]/2-pad)),max(0,int(q["py"]-q["ph"]/2-pad)),min(w,int(q["px"]+q["pw"]/2+pad)),min(h,int(q["py"]+q["ph"]/2+pad)))
    def _match(self,b,a,w,h):
        used=set();pairs=[]
        for o in b:
            best=(-999,None)
            for j,n in enumerate(a):
                if j in used or n["class"]!=o["class"]:continue
                dx=abs(o["px"]-n["px"])/w;dy=abs(o["py"]-n["py"])/h
                if dx>self.match_radius or dy>self.match_radius:continue
                same=self._norm(o["text"])==self._norm(n["text"]);score=(2.0 if same else 0)-4*(dx+dy)
                if score>best[0]:best=(score,j)
            if best[1] is None:pairs.append((o,None,"deleted"))
            else:used.add(best[1]);pairs.append((o,a[best[1]],"matched"))
        for j,n in enumerate(a):
            if j not in used:pairs.append((None,n,"added"))
        return pairs
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)
    def _deduplicate(self,regions):
        """Keep one report per physical change. Larger crops are preferred so
        the same changed text is not reported several times at different zooms."""
        kept=[]
        for r in sorted(regions,key=lambda z:(z.y,z.x)):
            duplicate=None
            for k in kept:
                iou=self._iou((r.x,r.y,r.width,r.height),(k.x,k.y,k.width,k.height))
                cx=abs((r.x+r.width/2)-(k.x+k.width/2))/max(1,min(r.width,k.width))
                cy=abs((r.y+r.height/2)-(k.y+k.height/2))/max(1,min(r.height,k.height))
                text_same=self._norm(r.old_text)==self._norm(k.old_text) and self._norm(r.new_text)==self._norm(k.new_text)
                if iou>.12 or (cx<2.0 and cy<2.0 and text_same):duplicate=k;break
            if duplicate is None:kept.append(r)
            else:
                # Prefer the tighter crop when both refer to the same change.
                if r.width*r.height<duplicate.width*duplicate.height:
                    kept[kept.index(duplicate)]=r
        return kept
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after=self._img(aligned_after) if aligned_after is not None else self._img(after_page);h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after));_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page);aw=self._native_words(after_page);bc=self._cluster(bw,before.shape);ac=self._cluster(aw,after.shape);pairs=self._match(bc,ac,w,h)
            candidates=[];added=deleted=0;ocr_count=0
            for o,n,kind in pairs:
                q=o or n;box=self._box(q,w,h);bo=self._highres_ocr(before_page,box);ao=self._highres_ocr(after_page,box);ocr_count+=int(bool(bo))+int(bool(ao));old=o["text"] if o else "";new=n["text"] if n else ""
                if kind=="matched" and self._norm(old)==self._norm(new):continue
                local=diff[box[1]:box[3],box[0]:box[2]];ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
                if kind=="matched" and ratio<.00015 and not bo and not ao:continue
                if kind=="added":added+=1
                if kind=="deleted":deleted+=1
                candidates.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,"text_"+kind,.85,before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,kind))
            merged=self._deduplicate(candidates)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"whole_mapping=global_text, tiles=0, tiles_pairs=0, text_clusters={len(bc)}/{len(ac)}, pairs={len(pairs)}, "
                    f"candidates={len(candidates)}, added={added}, deleted={deleted}, ocr_local={ocr_count}, final={len(merged)}")
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
