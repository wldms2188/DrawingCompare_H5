from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np
from .semantic_region_builder import SemanticRegionBuilder

@dataclass(frozen=True)
class Box:
    x:int; y:int; w:int; h:int
    def xyxy(self): return self.x,self.y,self.x+self.w,self.y+self.h
    def norm(self,W,H): return self.x/W,self.y/H,self.w/W,self.h/H
    @staticmethod
    def from_norm(v,W,H): return Box(round(v[0]*W),round(v[1]*H),round(v[2]*W),round(v[3]*H))
    def pad(self,p,W,H):
        x=max(0,self.x-int(p)); y=max(0,self.y-int(p)); return Box(x,y,min(W,self.x+self.w+int(p))-x,min(H,self.y+self.h+int(p))-y)

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0; region_type:str='general_change'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None; old_text:str=''; new_text:str=''; change_kind:str='changed_value'
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
    success:bool; regions:List[ChangeRegion]=field(default_factory=list); difference_image:Optional[np.ndarray]=None; threshold_image:Optional[np.ndarray]=None; change_pixel_ratio:float=0.0; reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    """Detect drawing changes in a common Before coordinate system.

    Text/Note/dimension/GD&T values are compared semantically, while a local
    image difference pass catches non-text geometry changes. After OCR boxes
    are transformed into the Before coordinate system using the same
    homography that produced the aligned image; this prevents mixed-coordinate
    crops and the excessive/incorrect zoom seen in earlier H5 builds.
    """
    def __init__(self,config=None):
        self.builder=SemanticRegionBuilder()
        self.max_region_center_distance=.28
        self.min_pair_score=.42
        self.value_distance=.30
        self.pixel_threshold=30

    @staticmethod
    def _img(p):
        if isinstance(p,np.ndarray): return np.asarray(p)
        if hasattr(p,'image'): return np.asarray(p.image)
        raise TypeError('페이지 이미지 배열을 찾을 수 없습니다.')
    @staticmethod
    def _gray(a):
        if a.ndim==2:return a.astype(np.uint8)
        if a.shape[2]==4:return cv2.cvtColor(a,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _crop(img,b):
        x,y,xx,yy=b.xyxy();H,W=img.shape[:2]
        return img[max(0,y):min(H,yy),max(0,x):min(W,xx)]
    @staticmethod
    def _norm_text(s):
        return re.sub(r'\s+','',str(s).upper().replace('−','-').replace('–','-').replace('—','-'))
    @staticmethod
    def _class(t):
        u=str(t).strip().upper()
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|⌖|⌯|⏥|⌒|∥|⊥',u):return 'GDT'
        if re.fullmatch(r'(?:[RMD]\s*)?(?:Ø|⌀)?\d+(?:\.\d+)?(?:\s*[A-Z°]+)?',u) or re.fullmatch(r'\d+/\d+',u):return 'DIMENSION'
        if re.search(r'NOTE|NOTES|UNLESS|MATERIAL|FINISH|REMOVE|BURR|INSPECT|SEE|REMARK|COMMENT',u):return 'NOTE'
        return 'TEXT'
    @staticmethod
    def _kind(t):
        t=str(t).upper();return 'gdt_change' if t in ('GDT','GD&T') else ('dimension_change' if t=='DIMENSION' else ('note_change' if t in ('NOTE','COMMENT') else 'text_change'))

    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path);p=doc.load_page(int(page.page_index));r=p.rect;out=[]
            for z in p.get_text('words'):
                x0,y0,x1,y1,text,*_=z;text=str(text).strip()
                if text:
                    out.append({'text':text,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(text)})
            doc.close();return out
        except Exception:return []

    @staticmethod
    def _homography_after_to_before(before_page,after_page,aligned_after):
        """Return H mapping original After pixels to Before pixels.

        The current Aligner already computes H in the Before->After point
        convention used by its warp. We derive the inverse by re-running the
        same robust feature estimation here only when no matrix is exposed.
        """
        b=ChangeDetector._img(before_page); a=ChangeDetector._img(after_page)
        bg=ChangeDetector._gray(b); ag=ChangeDetector._gray(a)
        orb=cv2.ORB_create(nfeatures=6000)
        kb,db=orb.detectAndCompute(bg,None); ka,da=orb.detectAndCompute(ag,None)
        if db is None or da is None or len(kb)<8 or len(ka)<8:return None
        m=cv2.BFMatcher(cv2.NORM_HAMMING)
        raw=m.knnMatch(db,da,k=2); good=[x for x,y in raw if x.distance<.75*y.distance]
        if len(good)<8:return None
        src=np.float32([kb[x.queryIdx].pt for x in good]).reshape(-1,1,2)
        dst=np.float32([ka[x.trainIdx].pt for x in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(dst,src,cv2.RANSAC,4.0)
        return H

    def _transform_words(self,words,H,W,Ht):
        if H is None:return []
        out=[]
        for q in words:
            pts=np.float32([[[q['x']*W,q['y']*Ht],[(q['x']+q['w'])*W,(q['y']+q['h'])*Ht]]])
            # input dimensions are supplied explicitly below; this branch is
            # retained only for compatibility and is not used by detect().
            out.append(q)
        return out

    def _transformed_word(self,q,H,W,H):
        pts=np.float32([[[q['x']*W,q['y']*H],[(q['x']+q['w'])*W,(q['y']+q['h'])*H]]])
        p=cv2.perspectiveTransform(pts,H)[0]
        x=min(p[:,0]);y=min(p[:,1]);xx=max(p[:,0]);yy=max(p[:,1])
        return {**q,'x':float(x),'y':float(y),'w':float(max(1,xx-x)),'h':float(max(1,yy-y)),'normalized':False}

    @staticmethod
    def _words_in_pixels(words,b):
        x,y,w,h=b;return [q for q in words if x<=q['x']+q['w']/2<=x+w and y<=q['y']+q['h']/2<=y+h]

    def _pair_text(self,ow,nw):
        """One-to-one pairing tolerant of OCR tokenization differences."""
        candidates=[]
        for i,o in enumerate(ow):
            for j,n in enumerate(nw):
                oc=self._class(o['text']);nc=self._class(n['text'])
                class_bonus=1 if oc==nc else (0.4 if {oc,nc} <= {'TEXT','NOTE'} else 0)
                d=((o['x']+o['w']/2-(n['x']+n['w']/2))**2+(o['y']+o['h']/2-(n['y']+n['h']/2))**2)**.5
                text_sim=cv2.compareHist(*[np.array([[0]],dtype=np.float32)]*2,cv2.HISTCMP_CORREL) if False else 0
                if d<max(80,0.18*max(o['w'],o['h'],n['w'],n['h'])):
                    candidates.append((d/(class_bonus+0.01),i,j))
        candidates.sort();uo=set();un=set();pairs=[]
        for _,i,j in candidates:
            if i not in uo and j not in un:uo.add(i);un.add(j);pairs.append((ow[i],nw[j]))
        return pairs

    def _word_box(self,w,pad=10):
        x=int(w['x']);y=int(w['y']);xx=int(w['x']+w['w']);yy=int(w['y']+w['h'])
        return Box(max(0,x-pad),max(0,y-pad),max(1,xx-x+2*pad),max(1,yy-y+2*pad))

    def _note_blocks(self,words,W,H):
        notes=[w for w in words if self._class(w['text'])=='NOTE']
        blocks=[];used=set()
        for i,w in enumerate(notes):
            if i in used:continue
            group=[w];used.add(i);changed=True
            while changed:
                changed=False
                for j,q in enumerate(notes):
                    if j in used:continue
                    if abs((q['y']+q['h']/2)-(w['y']+w['h']/2)) < max(q['h'],w['h'])*3 and abs(q['x']-w['x']) < W*.25:
                        group.append(q);used.add(j);changed=True
            x=min(q['x'] for q in group);y=min(q['y'] for q in group);xx=max(q['x']+q['w'] for q in group);yy=max(q['y']+q['h'] for q in group)
            blocks.append((Box(int(x),int(y),int((xx-x)),int((yy-y))),group))
        return blocks

    def _local_pixel_regions(self,before,after):
        if before.shape[:2]!=after.shape[:2]:return []
        a=self._gray(before);b=self._gray(after)
        diff=cv2.absdiff(a,b);_,th=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
        k=np.ones((3,3),np.uint8);th=cv2.morphologyEx(th,cv2.MORPH_OPEN,k);th=cv2.dilate(th,k,iterations=1)
        n,lab,stats,_=cv2.connectedComponentsWithStats(th,8);out=[]
        area_img=a.shape[0]*a.shape[1]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area<100 or area>area_img*.05:continue
            out.append(Box(int(x),int(y),int(w),int(h)))
        return out

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            old=self._img(before_page)
            original_after=self._img(after_page)
            view=self._img(aligned_after) if aligned_after is not None else original_after
            H0,W0=old.shape[:2];HV,WV=view.shape[:2]
            ow=self._words(before_page);nw=self._words(after_page)
            # Map original After word boxes into Before coordinates. The
            # aligned image is already in Before geometry, so its crops use
            # exactly these transformed boxes.
            H_after=self._homography_after_to_before(before_page,after_page,view)
            mapped=[]
            if H_after is not None:
                for q in nw:
                    try:mapped.append(self._transformed_word(q,H_after,original_after.shape[1],original_after.shape[0]))
                    except Exception:pass
            if not mapped:mapped=nw
            # If aligned dimensions differ, scale mapped boxes to the actual
            # output image dimensions.
            sx=WV/max(1,W0);sy=HV/max(1,H0)
            mapped=[{**q,'x':q['x']*sx,'y':q['y']*sy,'w':q['w']*sx,'h':q['h']*sy} for q in mapped]
            old_px=[{**q,'x':q['x']*W0,'y':q['y']*H0,'w':q['w']*W0,'h':q['h']*H0} for q in ow]
            # Direct spatial text pairing catches changed dimensions/notes.
            candidates=[]
            for o in old_px:
                for n in mapped:
                    d=((o['x']+o['w']/2-(n['x']+n['w']/2))**2+(o['y']+o['h']/2-(n['y']+n['h']/2))**2)**.5
                    if d <= max(60,min(WV,HV)*.025):
                        candidates.append((d,o,n))
            candidates.sort(key=lambda z:z[0]);uo=set();un=set();pairs=[]
            for d,o,n in candidates:
                if id(o) in uo or id(n) in un:continue
                uo.add(id(o));un.add(id(n));pairs.append((o,n,d))
            regions=[];changed_values=0
            for o,n,d in pairs:
                if self._norm_text(o['text'])==self._norm_text(n['text']):continue
                ob=self._word_box(o,10);nb=self._word_box(n,10)
                # Use the union box in Before coordinates, with only a small
                # safety margin, rather than the old double-padding crop.
                x=min(ob.x,nb.x);y=min(ob.y,nb.y);xx=max(ob.right,nb.right);yy=max(ob.bottom,nb.bottom)
                box=Box(x,y,xx-x,yy-y).pad(6,WV,HV)
                kind=self._kind(self._class(o['text']))
                regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,kind,max(0.5,1-d/max(1,min(WV,HV))),self._crop(old,ob),self._crop(view,box),None,o['text'],n['text'],kind));changed_values+=1
            # Local image-difference candidates are only retained when they
            # overlap a semantic text change or are clearly small geometry
            # edits. Large global differences are discarded as registration
            # noise.
            pixel_regions=self._local_pixel_regions(old,view)
            for pb in pixel_regions:
                if any(self._iou(pb,Box(r.x,r.y,r.width,r.height))>.15 for r in regions):continue
                crop=self._crop(view,pb.pad(6,WV,HV)); oldcrop=self._crop(old,pb.pad(6,W0,H0))
                if crop.size and oldcrop.size:
                    regions.append(ChangeRegion(pb.x,pb.y,pb.w,pb.h,pb.w*pb.h,0.0,'geometry_change',0.45,oldcrop,crop,None,'','', 'geometry_change'))
            out=[]
            for r in sorted(regions,key=lambda z:(z.confidence,-z.area),reverse=True):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.55 for q in out):out.append(r)
            note_count=sum(1 for q in ow if self._class(q['text'])=='NOTE')
            dim_count=sum(1 for q in ow if self._class(q['text'])=='DIMENSION')
            gdt_count=sum(1 for q in ow if self._class(q['text'])=='GDT')
            reason=(f'diag: native={len(ow)}/{len(nw)}, mapped={len(mapped)}, '
                    f'notes={note_count}, dimensions={dim_count}, gdt={gdt_count}, '
                    f'text_pairs={len(pairs)}, changed_values={changed_values}, '
                    f'pixel_candidates={len(pixel_regions)}, final={len(out)}')
            return ChangeDetectionResult(True,out,None,None,0.0,reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')

    @staticmethod
    def _iou(a,b):
        A=a.xyxy();B=b.xyxy();x=max(A[0],B[0]);y=max(A[1],B[1]);xx=min(A[2],B[2]);yy=min(A[3],B[3]);i=max(0,xx-x)*max(0,yy-y);u=a.w*a.h+b.w*b.h-i;return i/max(1,u)
