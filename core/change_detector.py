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
    """H5 text-first drawing comparison. Dimensions/GD&T/notes are individual
    anchors; geometry is used only as local context and not as a change target.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.match_radius=.09
        self.word_merge_gap=.008
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
        if re.search(r"(?:^|\s)(?:R|M)\s*\d|[0-9]+(?:\.[0-9]+)?(?:\s*(?:MM|IN|°|DEG))?$",t):return "dimension"
        if any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE")):return "note"
        # Short numeric/symbolic fragments are very often dimensions split by PDF extraction.
        if re.fullmatch(r"[0-9.]+|[0-9]+/[0-9]+|[A-Z]?[0-9]+",t):return "dimension"
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

    def _anchors(self,words,shape):
        """Keep each dimension/GD&T/note as an independent anchor.
        Only fragments that are genuinely adjacent on the same text baseline are joined.
        This prevents nearby dimensions from becoming one large cluster.
        """
        q=sorted([self._word(z,shape) for z in words if z["class"]!="other"],key=lambda z:(z["py"],z["px"]))
        out=[];used=set()
        for i,z in enumerate(q):
            if i in used:continue
            parts=[z];used.add(i)
            # Join only very close fragments, useful for split dimension tokens such as "25" + ".0".
            for j,n in enumerate(q):
                if j in used:continue
                baseline=abs(n["py"]-z["py"])<=max(z["ph"],n["ph"])*.65
                gap=abs((n["px"]-n["pw"]/2)-(z["px"]+z["pw"]/2))
                if baseline and 0<=gap<=shape[1]*self.word_merge_gap and n["class"]==z["class"]:
                    parts.append(n);used.add(j)
            parts.sort(key=lambda a:a["px"]);x0=min(a["px"]-a["pw"]/2 for a in parts);y0=min(a["py"]-a["ph"]/2 for a in parts);x1=max(a["px"]+a["pw"]/2 for a in parts);y1=max(a["py"]+a["ph"]/2 for a in parts)
            out.append({"text":" ".join(a["text"] for a in parts),"px":(x0+x1)/2,"py":(y0+y1)/2,"pw":x1-x0,"ph":y1-y0,"class":z["class"],"parts":parts})
        return out

    def _box(self,q,w,h):
        # Tight enough to identify one dimension, but wide enough to include its nearby leader/geometry.
        pad=max(55,int(max(q["pw"],q["ph"])*4));return(max(0,int(q["px"]-q["pw"]/2-pad)),max(0,int(q["py"]-q["ph"]/2-pad)),min(w,int(q["px"]+q["pw"]/2+pad)),min(h,int(q["py"]+q["ph"]/2+pad)))

    def _match(self,b,a,w,h):
        """Global one-to-one anchor matching. Same text is preferred, but changed
        dimensions can match by position/type. A small local displacement is allowed
        for drawings with different page scales.
        """
        used=set();pairs=[]
        for o in b:
            best=(-999,None)
            for j,n in enumerate(a):
                if j in used or n["class"]!=o["class"]:continue
                dx=abs(o["px"]-n["px"])/w;dy=abs(o["py"]-n["py"])/h
                if dx>self.match_radius or dy>self.match_radius:continue
                same=self._norm(o["text"])==self._norm(n["text"])
                # Text identity is useful for stable dimensions; position is decisive for changed values.
                score=(1.8 if same else 0)-3.0*(dx+dy)-.15*abs(np.log(max(.01,o["pw"])/max(.01,n["pw"])))
                if score>best[0]:best=(score,j)
            if best[1] is None:pairs.append((o,None,"deleted"))
            else:used.add(best[1]);pairs.append((o,a[best[1]],"matched"))
        for j,n in enumerate(a):
            if j not in used:pairs.append((None,n,"added"))
        return pairs

    def _candidate_score(self,o,n,ratio,bo,ao,shape):
        score=0.0
        if o and n:
            if self._norm(o["text"])!=self._norm(n["text"]):score+=5
            if o["class"] in ("dimension","gdt","note"):score+=2
        if bo or ao:score+=1.5
        if ratio>.0005:score+=min(3,ratio*200)
        # Prefer compact text-centered evidence over broad geometry changes.
        if o:score+=max(0,1-min(1,(o["pw"]+o["ph"])/shape[1]*15))
        return score

    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)

    def _deduplicate(self,regions):
        # Duplicate candidates from the same physical text are merged; unrelated nearby dimensions remain separate.
        kept=[]
        for r in sorted(regions,key=lambda z:(-z.confidence,z.y,z.x)):
            dup=None
            for k in kept:
                same_text=self._norm(r.old_text)==self._norm(k.old_text) and self._norm(r.new_text)==self._norm(k.new_text)
                iou=self._iou((r.x,r.y,r.width,r.height),(k.x,k.y,k.width,k.height))
                cx=abs((r.x+r.width/2)-(k.x+k.width/2));cy=abs((r.y+r.height/2)-(k.y+k.height/2))
                if iou>.25 or (same_text and cx<max(r.width,k.width)*.35 and cy<max(r.height,k.height)*.35):dup=k;break
            if dup is None:kept.append(r)
            elif r.confidence>dup.confidence:kept[kept.index(dup)]=r
        return kept

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after=self._img(aligned_after) if aligned_after is not None else self._img(after_page);h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after));_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page);aw=self._native_words(after_page);bc=self._anchors(bw,before.shape);ac=self._anchors(aw,after.shape);pairs=self._match(bc,ac,w,h)
            candidates=[];added=deleted=0
            for o,n,kind in pairs:
                q=o or n;box=self._box(q,w,h);local=diff[box[1]:box[3],box[0]:box[2]];ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
                old=o["text"] if o else "";new=n["text"] if n else ""
                # Same unchanged text is never a change, regardless of nearby geometry.
                if kind=="matched" and self._norm(old)==self._norm(new):continue
                # Changed/added/deleted text must have either a meaningful pixel difference or be a real add/delete.
                if kind=="matched" and ratio<.00015:continue
                if kind=="added":added+=1
                if kind=="deleted":deleted+=1
                score=self._candidate_score(o,n,ratio,[],[],before.shape)
                candidates.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,"text_"+kind,min(1,score/10),before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,kind))
            merged=self._deduplicate(candidates)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"whole_mapping=anchor_text, tiles=0, tiles_pairs=0, anchors={len(bc)}/{len(ac)}, pairs={len(pairs)}, "
                    f"candidates={len(candidates)}, added={added}, deleted={deleted}, ocr_local=disabled_native_first, final={len(merged)}")
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
