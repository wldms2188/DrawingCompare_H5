from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np
from .semantic_region_builder import SemanticRegionBuilder

@dataclass(frozen=True)
class Box:
    x: int; y: int; w: int; h: int
    def xyxy(self): return self.x, self.y, self.x + self.w, self.y + self.h
    def pad(self, p, W, H):
        p=int(p); x=max(0,self.x-p); y=max(0,self.y-p)
        return Box(x,y,max(1,min(W,self.x+self.w+p)-x),max(1,min(H,self.y+self.h+p)-y))

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0; region_type:str='general_change'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None
    old_text:str=''; new_text:str=''; change_kind:str='changed_value'
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
    success:bool; regions:List[ChangeRegion]=field(default_factory=list); difference_image:Optional[np.ndarray]=None
    threshold_image:Optional[np.ndarray]=None; change_pixel_ratio:float=0.0; reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    def __init__(self, config=None): self.builder=SemanticRegionBuilder(); self.pixel_threshold=30
    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return np.asarray(page)
        if hasattr(page,'image'): return np.asarray(page.image)
        raise TypeError('페이지 이미지 배열을 찾을 수 없습니다.')
    @staticmethod
    def _gray(img):
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _crop(img,box):
        x,y,xx,yy=box.xyxy(); H,W=img.shape[:2]; return img[max(0,y):min(H,yy),max(0,x):min(W,xx)]
    @staticmethod
    def _norm_text(text): return re.sub(r'\s+','',str(text).upper().replace('−','-').replace('–','-').replace('—','-'))
    @staticmethod
    def _class(text):
        u=str(text).strip().upper()
        if re.fullmatch(r'(?:R|M|D)?\s*(?:Ø|⌀)\s*\d+(?:\.\d+)?',u):return 'DIMENSION'
        if re.fullmatch(r'(?:R|M|D)?\s*\d+(?:\.\d+)?(?:\s*[A-Z°]+)?',u) or re.fullmatch(r'\d+/\d+',u) or '±' in u:return 'DIMENSION'
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌖|⌯|⏥|⌒|∥|⊥',u):return 'GDT'
        if re.search(r'NOTE|NOTES|UNLESS|MATERIAL|FINISH|REMOVE|BURR|INSPECT|SEE|REMARK|COMMENT',u):return 'NOTE'
        return 'TEXT'
    @staticmethod
    def _kind(cls): return {'GDT':'gdt_change','DIMENSION':'dimension_change','NOTE':'note_change'}.get(cls,'text_change')
    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path); p=doc.load_page(int(page.page_index)); r=p.rect; raw=p.get_text('words'); lines={}
            for z in raw:
                if len(z)<8:continue
                x0,y0,x1,y1,text,block_no,line_no,word_no=z[:8]; lines.setdefault((block_no,line_no),[]).append(str(text).strip())
            out=[]
            for z in raw:
                if len(z)<8:continue
                x0,y0,x1,y1,text,block_no,line_no,word_no=z[:8]; text=str(text).strip()
                if text:
                    context=' '.join(lines.get((block_no,line_no),[text])); out.append({'text':text,'context':context,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(context)})
            doc.close(); return out
        except Exception:return []
    @staticmethod
    def _map_box(q,M,srcW,srcH):
        pts=np.float32([[[q['x']*srcW,q['y']*srcH],[(q['x']+q['w'])*srcW,(q['y']+q['h'])*srcH]]]); p=cv2.transform(pts,M)[0]
        x,y=float(np.min(p[:,0])),float(np.min(p[:,1])); xx,yy=float(np.max(p[:,0])),float(np.max(p[:,1])); return {**q,'x':x,'y':y,'w':max(1,xx-x),'h':max(1,yy-y)}
    @staticmethod
    def _iou(a,b):
        ax,ay,axx,ayy=a.xyxy(); bx,by,bxx,byy=b.xyxy(); x=max(ax,bx);y=max(ay,by);xx=min(axx,bxx);yy=min(ayy,byy); inter=max(0,xx-x)*max(0,yy-y); return inter/max(1,a.w*a.h+b.w*b.h-inter)
    @staticmethod
    def _word_box(w,pad=8):
        x=int(round(w['x'])); y=int(round(w['y'])); xx=int(round(w['x']+w['w'])); yy=int(round(w['y']+w['h'])); return Box(max(0,x-pad),max(0,y-pad),max(1,xx-x+2*pad),max(1,yy-y+2*pad))
    def _pixel_regions(self,before,after):
        if before.shape[:2]!=after.shape[:2]:return []
        a=self._gray(before); b=self._gray(after); diff=cv2.absdiff(a,b); _,th=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY); k=np.ones((3,3),np.uint8); th=cv2.morphologyEx(th,cv2.MORPH_OPEN,k); th=cv2.dilate(th,k,1); n,_,stats,_=cv2.connectedComponentsWithStats(th,8); out=[]; total=a.shape[0]*a.shape[1]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area>=100 and area<=total*.03:out.append(Box(int(x),int(y),int(w),int(h)))
        return out
    def detect(self,before_page,after_page,aligned_after=None,alignment_matrix=None):
        try:
            before=self._img(before_page); after_original=self._img(after_page); view=self._img(aligned_after) if aligned_after is not None else after_original
            H0,W0=before.shape[:2]; HV,WV=view.shape[:2]; old=self._words(before_page); new=self._words(after_page); mapped=[]
            if alignment_matrix is not None:
                M=np.asarray(alignment_matrix,dtype=np.float32).reshape(2,3)
                mapped=[self._map_box(q,M,after_original.shape[1],after_original.shape[0]) for q in new]
            else:
                mapped=new
            # The aligned image is already in Before pixel coordinates. Only normalize if dimensions differ.
            sx=WV/max(1,W0); sy=HV/max(1,H0); mapped=[{**q,'x':q['x']*sx,'y':q['y']*sy,'w':q['w']*sx,'h':q['h']*sy} for q in mapped]
            oldpx=[{**q,'x':q['x']*W0,'y':q['y']*H0,'w':q['w']*W0,'h':q['h']*H0} for q in old]
            candidates=[]; maxd=max(50,min(WV,HV)*.025)
            for oi,o in enumerate(oldpx):
                for ni,n in enumerate(mapped):
                    d=float(np.hypot(o['x']+o['w']/2-n['x']-n['w']/2,o['y']+o['h']/2-n['y']-n['h']/2)); bonus=0 if o['class']==n['class'] else 20
                    if d+bonus<=maxd:candidates.append((d+bonus,oi,ni))
            candidates.sort(); used_o=set(); used_n=set(); pairs=[]
            for score,oi,ni in candidates:
                if oi in used_o or ni in used_n:continue
                used_o.add(oi);used_n.add(ni);pairs.append((oldpx[oi],mapped[ni],score))
            regions=[]
            for o,n,score in pairs:
                if self._norm_text(o['text'])==self._norm_text(n['text']):continue
                ob=self._word_box(o); nb=self._word_box(n); x=min(ob.x,nb.x);y=min(ob.y,nb.y);xx=max(ob.x+ob.w,nb.x+nb.w);yy=max(ob.y+ob.h,nb.y+nb.h); box=Box(x,y,xx-x,yy-y).pad(5,WV,HV); cls=self._class(o.get('context',o['text'])); kind=self._kind(cls)
                regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,kind,max(.5,1-score/max(1,maxd)),self._crop(before,ob),self._crop(view,box),None,o['text'],n['text'],kind))
            pixels=self._pixel_regions(before,view)
            for pb in pixels:
                if any(self._iou(pb,Box(r.x,r.y,r.width,r.height))>.15 for r in regions):continue
                box=pb.pad(5,WV,HV);regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,'geometry_change',.45,self._crop(before,box),self._crop(view,box),None,'','', 'geometry_change'))
            final=[]
            for r in sorted(regions,key=lambda z:(z.confidence,-z.area),reverse=True):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.55 for q in final):final.append(r)
            reason=f'diag: native={len(old)}/{len(new)}, mapping=alignment_matrix, notes={sum(q["class"]=="NOTE" for q in old)}, dimensions={sum(q["class"]=="DIMENSION" for q in old)}, gdt={sum(q["class"]=="GDT" for q in old)}, text_pairs={len(pairs)}, changed_values={sum(r.change_kind!="geometry_change" for r in final)}, pixel_candidates={len(pixels)}, final={len(final)}'
            return ChangeDetectionResult(True,final,None,None,0.0,reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f'diag_error: {type(exc).__name__}: {exc}')
