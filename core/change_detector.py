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
    def __init__(self,config=None):
        self.builder=SemanticRegionBuilder(); self.max_region_center_distance=.20; self.min_pair_score=.52; self.value_distance=.22; self.pixel_threshold=30
    @staticmethod
    def _img(p):
        if isinstance(p,np.ndarray):return np.asarray(p)
        if hasattr(p,'image'):return np.asarray(p.image)
        raise TypeError('페이지 이미지 배열을 찾을 수 없습니다.')
    @staticmethod
    def _gray(a):
        if a.ndim==2:return a.astype(np.uint8)
        if a.shape[2]==4:return cv2.cvtColor(a,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _crop(img,b):
        x,y,xx,yy=b.xyxy();H,W=img.shape[:2];return img[max(0,y):min(H,yy),max(0,x):min(W,xx)]
    @staticmethod
    def _norm_text(s):return re.sub(r'\s+','',str(s).upper().replace('−','-').replace('–','-').replace('—','-'))
    @staticmethod
    def _kind(t):
        t=str(t).upper();return 'gdt_change' if t in ('GDT','GD&T') else ('dimension_change' if t=='DIMENSION' else ('note_change' if t in ('NOTE','COMMENT') else 'text_change'))
    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path);p=doc.load_page(int(page.page_index));r=p.rect;out=[]
            for z in p.get_text('words'):
                x0,y0,x1,y1,text,*_=z;text=str(text).strip()
                if text:out.append({'text':text,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(text)})
            doc.close();return out
        except Exception:return []
    @staticmethod
    def _class(t):
        u=str(t).upper()
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|⌖|⌯|⏥|⌒|∥|⊥',u):return 'GDT'
        if re.fullmatch(r'(?:[RMD]\s*)?(?:Ø|⌀)?\d+(?:\.\d+)?(?:\s*[A-Z°]+)?',u) or re.fullmatch(r'\d+/\d+',u):return 'DIMENSION'
        if re.search(r'NOTE|NOTES|UNLESS|MATERIAL|FINISH|REMOVE|BURR|INSPECT|SEE|REMARK|COMMENT',u):return 'NOTE'
        return 'TEXT'
    def _regions(self,img,words):return self.builder.build(img,words)
    def _rscore(self,b,a,W0,H0,W1,H1,old,new):
        bn=b.norm(W0,H0);an=a.norm(W1,H1);bc=(bn[0]+bn[2]/2,bn[1]+bn[3]/2);ac=(an[0]+an[2]/2,an[1]+an[3]/2);pos=((bc[0]-ac[0])**2+(bc[1]-ac[1])**2)**.5
        if pos>self.max_region_center_distance:return 0.0
        size=(min(bn[2],an[2])/max(bn[2],an[2],1e-9)+min(bn[3],an[3])/max(bn[3],an[3],1e-9))/2
        typ=1.0 if b.kind.upper()==a.kind.upper() else 0.0
        score=.48*(1-pos/self.max_region_center_distance)+.20*size+.17*self._visual(self._crop(old,Box(b.x,b.y,b.w,b.h)),self._crop(new,Box(a.x,a.y,a.w,a.h)))+.15*typ
        return score
    def _visual(self,a,b):
        if a.size==0 or b.size==0:return 0.0
        aa=cv2.resize(self._gray(a),(256,256),interpolation=cv2.INTER_CUBIC);bb=cv2.resize(self._gray(b),(256,256),interpolation=cv2.INTER_CUBIC)
        ea=cv2.Canny(aa,25,110);eb=cv2.Canny(bb,25,110);return max(0.0,min(1.0,float(cv2.matchTemplate(ea,eb,cv2.TM_CCOEFF_NORMED)[0,0])))
    def _pair_regions(self,br,ar,old,new,W0,H0,W1,H1):
        c=[]
        for i,b in enumerate(br):
            for j,a in enumerate(ar):
                s=self._rscore(b,a,W0,H0,W1,H1,old,new)
                if s>=self.min_pair_score:c.append((s,i,j))
        c.sort(reverse=True);ub=set();ua=set();out=[]
        for s,i,j in c:
            if i not in ub and j not in ua:ub.add(i);ua.add(j);out.append((i,j,s))
        return out
    @staticmethod
    def _inside(words,r,W,H):
        x,y,w,h=r.norm(W,H);return [q for q in words if x<=q['x']+q['w']/2<=x+w and y<=q['y']+q['h']/2<=y+h]
    def _pair_values(self,ow,nw,b,a,W0,H0,W1,H1):
        bx,by,bw,bh=b.norm(W0,H0);ax,ay,aw,ah=a.norm(W1,H1);used=set();out=[]
        def rel(q,x,y,w,h):return ((q['x']+q['w']/2-x)/max(w,1e-9),(q['y']+q['h']/2-y)/max(h,1e-9))
        for o in ow:
            ro=rel(o,bx,by,bw,bh);best=None
            for j,n in enumerate(nw):
                if j in used or n['class']!=o['class']:continue
                rn=rel(n,ax,ay,aw,ah);d=((ro[0]-rn[0])**2+(ro[1]-rn[1])**2)**.5
                if d<=self.value_distance and (best is None or d<best[0]):best=(d,j,n)
            if best:used.add(best[1]);out.append((o,best[2]))
        return out
    def _word_box(self,w,W,H,pad=18):
        x=int(w['x']*W);y=int(w['y']*H);xx=int((w['x']+w['w'])*W);yy=int((w['y']+w['h'])*H);return Box(max(0,x-pad),max(0,y-pad),min(W,xx+pad)-max(0,x-pad),min(H,yy+pad)-max(0,y-pad))
    @staticmethod
    def _iou(a,b):
        A=a.xyxy();B=b.xyxy();x=max(A[0],B[0]);y=max(A[1],B[1]);xx=min(A[2],B[2]);yy=min(A[3],B[3]);i=max(0,xx-x)*max(0,yy-y);u=a.w*a.h+b.w*b.h-i;return i/max(1,u)
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            old=self._img(before_page);new=self._img(after_page);view=self._img(aligned_after) if aligned_after is not None else new
            owds=self._words(before_page);nwds=self._words(after_page);H0,W0=old.shape[:2];H1,W1=new.shape[:2];HV,WV=view.shape[:2]
            br=self._regions(old,owds);ar=self._regions(new,nwds);pairs=self._pair_regions(br,ar,old,new,W0,H0,W1,H1);regions=[];changed=0
            for bi,ai,score in pairs:
                b,a=br[bi],ar[ai];ow=self._inside(owds,b,W0,H0);nw=self._inside(nwds,a,W1,H1)
                for o,n in self._pair_values(ow,nw,b,a,W0,H0,W1,H1):
                    if self._norm_text(o['text'])==self._norm_text(n['text']):continue
                    ob=self._word_box(o,W0,H0);nb=self._word_box(n,W1,H1);vb=Box.from_norm(nb.norm(W1,H1),WV,HV).pad(30,WV,HV);old_crop=self._crop(old,ob);new_crop=self._crop(view,vb);regions.append(ChangeRegion(vb.x,vb.y,vb.w,vb.h,vb.w*vb.h,0.0,self._kind(o['class']),score,old_crop.copy(),new_crop.copy(),None,o['text'],n['text']));changed+=1
            out=[]
            for r in sorted(regions,key=lambda z:-z.confidence):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.55 for q in out):out.append(r)
            reason=f'diag: native={len(owds)}/{len(nwds)}, mapping=semantic_regions, regions={len(br)}/{len(ar)}, region_pairs={len(pairs)}, changed_values={changed}, global_fallback=disabled, final={len(out)}'
            return ChangeDetectionResult(True,out,None,None,0.0,reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')
