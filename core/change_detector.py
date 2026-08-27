from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int
    area:int=0; change_ratio:float=0.0
    region_type:str="dimension_or_note"; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None
    difference_crop:Optional[np.ndarray]=None
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
    """H5 conservative detector.

    A value is compared only after its surrounding drawing context has been
    independently matched. Text equality alone, page coordinates alone, and
    unmatched-word additions/deletions are never sufficient evidence.
    """
    def __init__(self,config=None):
        self.pixel_threshold=38
        self.context=360
        self.search_scales=(.55,.65,.75,.85,1.0,1.15,1.30,1.45,1.60)
        self.min_context=.70
        self.min_margin=.07
        self.min_change=.00008

    @staticmethod
    def _img(p):
        if isinstance(p,np.ndarray): return np.asarray(p)
        if hasattr(p,"image"): return np.asarray(p.image)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")
    @staticmethod
    def _gray(a):
        if a.ndim==2:return a.astype(np.uint8)
        if a.shape[2]==4:return cv2.cvtColor(a,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _norm(s):
        return re.sub(r"[^A-Z0-9Ø⌀±+\-.*/X°]","",str(s).upper().replace("—","-").replace("–","-").replace("−","-"))
    @staticmethod
    def _class(t):
        t=str(t).strip().upper()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|[⏥⌖⌒∥⊥]",t): return "gdt"
        if re.fullmatch(r"(?:[RMD]?\s*)?[0-9]+(?:\.[0-9]+)?(?:\s*(?:MM|IN|°|DEG))?",t) or re.fullmatch(r"[0-9]+/[0-9]+",t): return "dimension"
        if any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE")): return "note"
        if re.search(r"[A-Z]",t) and len(t)>=2:return "note"
        return "other"
    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); r=p.rect; out=[]
            for z in p.get_text("words"):
                x0,y0,x1,y1,text,*_=z; text=str(text).strip()
                if text: out.append({"text":text,"x":x0/r.width,"y":y0/r.height,"w":(x1-x0)/r.width,"h":(y1-y0)/r.height,"class":self._class(text)})
            doc.close(); return out
        except Exception:return []
    @staticmethod
    def _px(z,shape):
        h,w=shape[:2]; return {**z,"px":(z["x"]+z["w"]/2)*w,"py":(z["y"]+z["h"]/2)*h,"pw":max(1,z["w"]*w),"ph":max(1,z["h"]*h)}
    def _anchors(self,words,shape):
        # Do not merge unrelated words. A note/box/dimension is an anchor in its own right.
        return sorted([self._px(z,shape) for z in words if z["class"]!="other"],key=lambda z:(z["py"],z["px"]))
    @staticmethod
    def _crop(img,cx,cy,size):
        h,w=img.shape[:2]; half=int(size/2); x0=max(0,int(cx-half));y0=max(0,int(cy-half));x1=min(w,int(cx+half));y1=min(h,int(cy+half));return img[y0:y1,x0:x1]
    @staticmethod
    def _struct(img,size=256):
        g=ChangeDetector._gray(img)
        g=cv2.resize(g,(size,size),interpolation=cv2.INTER_CUBIC)
        # Suppress tiny glyph strokes. Long drawing lines, boxes and leaders remain.
        bw=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)); bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,k)
        hor=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(17,1)))
        ver=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,17)))
        lines=cv2.bitwise_or(hor,ver)
        return lines
    def _score(self,a,b):
        if a.size==0 or b.size==0:return 0.0
        sa=self._struct(a); sb=self._struct(b)
        corr=max(0,float(cv2.matchTemplate(sa,sb,cv2.TM_CCOEFF_NORMED)[0,0]))
        den=max(1,int(np.count_nonzero(sa))+int(np.count_nonzero(sb)))
        overlap=2.0*int(np.count_nonzero(cv2.bitwise_and(sa,sb)))/den
        return .65*corr+.35*overlap
    def _candidate_score(self,before,after,o,cx,cy,size):
        a=self._crop(before,o["px"],o["py"],self.context)
        b=self._crop(after,cx,cy,size)
        return self._score(a,b)
    def _visual_candidates(self,before,after,o):
        """Whole-page multi-scale search using structural lines, not text pixels."""
        ag=self._gray(after); h,w=ag.shape
        t=self._crop(before,o["px"],o["py"],self.context)
        if t.size==0:return []
        ts=self._struct(t,256); out=[]
        # Work at a fixed search canvas; scale is represented by resizing template.
        search=cv2.resize(ag,(max(640,min(1400,w)),max(640,min(1400,h))),interpolation=cv2.INTER_AREA)
        sw,sh=search.shape[1],search.shape[0]
        se=self._struct(search,512)
        for s in self.search_scales:
            tw=max(100,int(t.shape[1]*s)); th=max(100,int(t.shape[0]*s))
            if tw>=w or th>=h:continue
            templ=self._struct(cv2.resize(t,(tw,th),interpolation=cv2.INTER_CUBIC),256)
            # resize template to same structural canvas scale as search
            tws=max(32,int(tw*sw/w)); ths=max(32,int(th*sh/h)); templ=cv2.resize(templ,(256,256),interpolation=cv2.INTER_AREA)
            res=cv2.matchTemplate(se,templ,cv2.TM_CCOEFF_NORMED)
            for _ in range(4):
                _,mx,_,loc=cv2.minMaxLoc(res)
                if mx<.35:break
                cx=(loc[0]+128)*w/sw; cy=(loc[1]+128)*h/sh
                # Re-score with the actual page crop, preserving scale information.
                score=self._candidate_score(before,after,o,cx,cy,int(self.context*s))
                out.append((score,float(mx),cx,cy,int(self.context*s)))
                cv2.rectangle(res,(max(0,loc[0]-80),max(0,loc[1]-80)),(min(res.shape[1]-1,loc[0]+80),min(res.shape[0]-1,loc[1]+80)),-1,-1)
        out.sort(reverse=True); return out[:12]
    def _nearest_anchor(self,aa,cx,cy,cls,limit):
        vals=[]
        for j,n in enumerate(aa):
            if n["class"]!=cls:continue
            d=((n["px"]-cx)**2+(n["py"]-cy)**2)**.5
            if d<=limit: vals.append((d,j,n))
        return min(vals) if vals else None
    def _match(self,before,after,ba,aa):
        cand=[]
        for i,o in enumerate(ba):
            cs=self._visual_candidates(before,after,o)
            if not cs:continue
            # A real match must be clearly better than the runner-up.
            best=cs[0]; second=cs[1][0] if len(cs)>1 else 0
            if best[0]<self.min_context or best[0]-second<self.min_margin:continue
            near=self._nearest_anchor(aa,best[2],best[3],o["class"],max(70,best[4]*.45))
            if near is None:continue
            d,j,n=near
            # Re-score centered on the actual After text anchor.
            ctx=self._score(self._crop(before,o["px"],o["py"],self.context*1.4),self._crop(after,n["px"],n["py"],self.context*1.4))
            if ctx<self.min_context:continue
            text=self._norm(o["text"]); nt=self._norm(n["text"])
            text_sim=1.0 if text==nt else (len(set(text)&set(nt))/max(1,len(set(text)|set(nt))))
            final=.55*best[0]+.30*ctx+.15*text_sim
            cand.append((final,i,j,best[0],ctx))
        cand.sort(reverse=True); ub=set();ua=set();pairs=[]
        for score,i,j,v,c in cand:
            if i in ub or j in ua:continue
            ub.add(i);ua.add(j);pairs.append((ba[i],aa[j],"matched",score,v,c))
        # Intentionally no automatic added/deleted generation. A separate region detector
        # can add these later after structural correspondence is proven.
        return pairs
    def _box(self,q,w,h):
        pad=max(65,int(max(q["pw"],q["ph"])*5))
        return max(0,int(q["px"]-q["pw"]/2-pad)),max(0,int(q["py"]-q["ph"]/2-pad)),min(w,int(q["px"]+q["pw"]/2+pad)),min(h,int(q["py"]+q["ph"]/2+pad))
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)
    def _dedup(self,rs):
        out=[]
        for r in sorted(rs,key=lambda z:-z.confidence):
            if any(self._iou((r.x,r.y,r.width,r.height),(q.x,q.y,q.width,q.height))>.18 for q in out):continue
            out.append(r)
        return out
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after=self._img(aligned_after) if aligned_after is not None else self._img(after_page)
            h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after));_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._words(before_page);aw=self._words(after_page);ba=self._anchors(bw,before.shape);aa=self._anchors(aw,after.shape)
            pairs=self._match(before,after,ba,aa);regions=[];changed=0;rejected=0
            for o,n,kind,score,visual,ctx in pairs:
                old=o["text"];new=n["text"]
                if self._norm(old)==self._norm(new):continue
                box=self._box(n,w,h);local=diff[box[1]:box[3],box[0]:box[2]];ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
                if ratio<self.min_change:rejected+=1;continue
                changed+=1
                typ="dimension_change" if o["class"]=="dimension" else ("gdt_change" if o["class"]=="gdt" else "note_change")
                conf=min(1,.50*score+.30*visual+.20*ctx)
                regions.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,typ,conf,before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,"changed_value"))
            regions=self._dedup(regions)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"mapping=structural_consensus, anchors={len(ba)}/{len(aa)}, pairs={len(pairs)}, changed_values={changed}, "
                    f"added=0, deleted=0, rejected={rejected}, final={len(regions)}")
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
