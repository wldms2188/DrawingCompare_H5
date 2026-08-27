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
    """H5: compare dimensions/GD&T/notes locally around drawing structures.

    Geometry itself is not reported. The page is first divided into local
    drawing blocks, then OCR text in each block is matched only inside the
    same/nearby block. This prevents a dimension near structure A from being
    paired with an unrelated dimension near structure B.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.merge_distance=18
        self.pytesseract=None
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
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
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
                if not t or c<30:continue
                x,y,w,h=[int(d[k][i]) for k in ("left","top","width","height")]
                if w>=3 and h>=3:out.append((x,y,w,h,t,c))
            return out
        except Exception:return []

    @staticmethod
    def _norm(s):
        s=s.upper().replace("—","-").replace("–","-").replace("−","-")
        return re.sub(r"\s+","",s)

    @staticmethod
    def _target(text,h):
        t=text.upper()
        numeric=bool(re.search(r"\d",t))
        engineering=bool(re.search(r"(?:Ø|⌀|R\s*\d|±|\+/-|\+\-|[0-9]+(?:\.[0-9]+)?[A-Z])",t))
        note=any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE"))
        return note or (numeric and (engineering or h>=7))

    def _structure_blocks(self,gray):
        """Find coarse connected drawing regions, not exact geometry.

        Blocks are only used as a locality constraint. Long border/title-block
        structures are discarded. The remaining blocks are enlarged so nearby
        dimensions and GD&T frames belong to the same local search area.
        """
        h,w=gray.shape[:2]
        bw=cv2.GaussianBlur(gray,(3,3),0)
        _,ink=cv2.threshold(bw,210,255,cv2.THRESH_BINARY_INV)
        # Close nearby lines into drawing components while avoiding page-wide merge.
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        joined=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,kernel,iterations=1)
        n,_,stats,_=cv2.connectedComponentsWithStats(joined,8)
        blocks=[]
        min_area=max(300,int(h*w*0.00008))
        for i in range(1,n):
            x,y,ww,hh,area=[int(v) for v in stats[i]]
            if area<min_area:continue
            if ww>0.82*w or hh>0.78*h:continue
            if ww<35 or hh<25:continue
            # Ignore narrow title/border-like strips.
            if ww>0.65*w and hh<0.12*h:continue
            if hh>0.65*h and ww<0.12*w:continue
            pad_x=max(25,int(0.035*w)); pad_y=max(25,int(0.035*h))
            x=max(0,x-pad_x); y=max(0,y-pad_y)
            x2=min(w,x+ww+2*pad_x); y2=min(h,y+hh+2*pad_y)
            blocks.append((x,y,x2-x,y2-y))
        # If segmentation is too fragmented, use a regular coarse grid. The
        # grid is a safety net, not a reported drawing region.
        if len(blocks)<2:
            cols=3 if w/h>1.4 else 2; rows=2 if h/w<1.4 else 3
            blocks=[]
            for ry in range(rows):
                for cx in range(cols):
                    x=cx*w//cols; y=ry*h//rows
                    x2=(cx+1)*w//cols; y2=(ry+1)*h//rows
                    blocks.append((x,y,x2-x,y2-y))
        # Merge overlapping blocks.
        merged=[]
        for b in blocks:
            done=False
            for i,m in enumerate(merged):
                if self._iou(b,m)>0.05:
                    l=min(b[0],m[0]); t=min(b[1],m[1]); r=max(b[0]+b[2],m[0]+m[2]); bb=max(b[1]+b[3],m[1]+m[3])
                    merged[i]=(l,t,r-l,bb-t); done=True; break
            if not done:merged.append(b)
        return merged

    @staticmethod
    def _iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/max(1,union)

    @staticmethod
    def _inside(box,block,pad=0):
        x,y,w,h=box; bx,by,bw,bh=block
        cx=x+w/2; cy=y+h/2
        return bx-pad<=cx<=bx+bw+pad and by-pad<=cy<=by+bh+pad

    def _local_pairs(self,before,after,diff,blocks):
        gb,ga=self._gray(before),self._gray(after)
        old=[x for x in self._ocr(gb) if self._target(x[4],x[3])]
        new=[x for x in self._ocr(ga) if self._target(x[4],x[3])]
        out=[]; used=set()
        # Assign each text box to its nearest drawing block. A text box outside
        # all blocks is allowed only when it is near the page center, avoiding
        # title-block/border noise.
        def block_for(box):
            candidates=[(i,b) for i,b in enumerate(blocks) if self._inside(box,b,15)]
            if candidates:return candidates[0][0]
            cx=box[0]+box[2]/2; cy=box[1]+box[3]/2
            best=min(range(len(blocks)),key=lambda i:((cx-(blocks[i][0]+blocks[i][2]/2))**2+(cy-(blocks[i][1]+blocks[i][3]/2))**2)**0.5)
            return best if blocks else -1
        old_groups={i:[] for i in range(len(blocks))}; new_groups={i:[] for i in range(len(blocks))}
        for x in old: old_groups[block_for(x)].append(x)
        for x in new: new_groups[block_for(x)].append(x)
        for bi,b in enumerate(blocks):
            for o in old_groups.get(bi,[]):
                ox,oy,ow,oh,ot,oc=o; best=None; bestscore=0
                for j,n in enumerate(new_groups.get(bi,[])):
                    global_index=new.index(n)
                    if global_index in used:continue
                    nx,ny,nw,nh,nt,nc=n
                    dist=np.hypot((ox+ow/2)-(nx+nw/2),(oy+oh/2)-(ny+nh/2))
                    lim=max(45,12*max(ow,oh,nw,nh))
                    if dist>lim:continue
                    spatial=1-dist/lim
                    size=max(0,1-abs(np.log(max(ow,1)/max(nw,1))))
                    score=.7*spatial+.3*size
                    if score>bestscore:bestscore,best=score,global_index
                if best is None:continue
                n=new[best]; used.add(best); nx,ny,nw,nh,nt,nc=n
                if self._norm(ot)==self._norm(nt):continue
                # Crop locally around the matched dimension/note, not the entire structure.
                x=max(0,min(ox,nx)-12); y=max(0,min(oy,ny)-12)
                x2=min(diff.shape[1],max(ox+ow,nx+nw)+12); y2=min(diff.shape[0],max(oy+oh,ny+nh)+12)
                local=diff[y:y2,x:x2]
                if local.size==0:continue
                ratio=float(np.mean(local>self.pixel_threshold))
                if ratio<0.008:continue
                out.append((x,y,x2-x,y2-y,"dimension_or_note",min(.99,.62+.30*bestscore),bi))
        return out

    def detect(self,before_page,after_page):
        try:
            before=self._img(before_page); after=self._img(after_page)
            h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gb=self._gray(before); diff=cv2.absdiff(gb,self._gray(after))
            _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            blocks=self._structure_blocks(gb)
            candidates=self._local_pairs(before,after,diff,blocks)
            regions=[]
            for x,y,rw,rh,typ,conf,bi in candidates:
                old=before[y:y+rh,x:x+rw].copy(); new=after[y:y+rh,x:x+rw].copy(); d=diff[y:y+rh,x:x+rw].copy()
                regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),typ,conf,old,new,d))
            # Merge only overlapping local text changes. Separate dimensions
            # around the same structure remain separate when spatially distinct.
            merged=[]
            for r in regions:
                done=False
                for m in merged:
                    if self._iou((r.x,r.y,r.width,r.height),(m.x,m.y,m.width,m.height))>0.10:
                        l=min(r.x,m.x); t=min(r.y,m.y); rr=max(r.right,m.right); bb=max(r.bottom,m.bottom)
                        m.x,m.y,m.width,m.height=l,t,rr-l,bb-t; m.confidence=max(m.confidence,r.confidence)
                        m.old_crop=before[t:bb,l:rr].copy(); m.new_crop=after[t:bb,l:rr].copy(); m.difference_crop=diff[t:bb,l:rr].copy(); m.change_ratio=float(np.mean(m.difference_crop>self.pixel_threshold)); done=True; break
                if not done:merged.append(r)
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),"structure-local dimension/GD&T/note priority")
        except Exception as exc:return ChangeDetectionResult(False,[],reason=str(exc))
