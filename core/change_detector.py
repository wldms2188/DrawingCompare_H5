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
    def pad(self,p,W,H):
        x=max(0,self.x-int(p)); y=max(0,self.y-int(p)); xx=min(W,self.x+self.w+int(p)); yy=min(H,self.y+self.h+int(p))
        return Box(x,y,max(1,xx-x),max(1,yy-y))
    @property
    def right(self): return self.x+self.w
    @property
    def bottom(self): return self.y+self.h

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0; region_type:str='general_change'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None
    old_text:str=''; new_text:str=''; change_kind:str='changed_value'

@dataclass
class ChangeDetectionResult:
    success:bool; regions:List[ChangeRegion]=field(default_factory=list); difference_image:Optional[np.ndarray]=None
    threshold_image:Optional[np.ndarray]=None; change_pixel_ratio:float=0.0; reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    def __init__(self,config=None): self.builder=SemanticRegionBuilder(); self.pixel_threshold=45
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
    def _crop(img,b):
        H,W=img.shape[:2]; x=max(0,min(W-1,int(round(b.x)))); y=max(0,min(H-1,int(round(b.y)))); xx=max(x+1,min(W,int(round(b.x+b.w)))); yy=max(y+1,min(H,int(round(b.y+b.h)))); return img[y:yy,x:xx]
    @staticmethod
    def _norm_text(t): return re.sub(r'\s+','',str(t).upper().replace('−','-').replace('–','-').replace('—','-'))
    @staticmethod
    def _class(t):
        u=str(t).strip().upper()
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌖|⌯|⏥|⌒|∥|⊥|\|',u): return 'GDT'
        if re.search(r'NOTE|NOTES|UNLESS|MATERIAL|FINISH|REMOVE|BURR|INSPECT|SEE|REMARK|COMMENT',u): return 'NOTE'
        if '±' in u or re.fullmatch(r'(?:R|M|D)?\s*(?:Ø|⌀)?\s*\d+(?:\.\d+)?(?:\s*[A-Z°]+)?',u) or re.fullmatch(r'\d+/\d+',u): return 'DIMENSION'
        return 'TEXT'
    @staticmethod
    def _class_token_context(token,context):
        tc=ChangeDetector._class(token); cc=ChangeDetector._class(context)
        # Semantic line context wins for NOTE/GD&T. This prevents values such as
        # 0.05 inside a GD&T frame or AL6061 inside a NOTE from becoming generic dimensions.
        if cc in ('NOTE','GDT'): return cc
        return tc if tc!='TEXT' else cc
    @staticmethod
    def _kind(cls): return {'GDT':'gdt_change','DIMENSION':'dimension_change','NOTE':'note_change'}.get(cls,'text_change')
    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path); p=doc.load_page(int(page.page_index)); r=p.rect; raw=p.get_text('words'); lines={}
            for z in raw:
                if len(z)>=8: lines.setdefault((z[5],z[6]),[]).append(str(z[4]).strip())
            out=[]
            for z in raw:
                if len(z)<8 or not str(z[4]).strip(): continue
                x0,y0,x1,y1,text,block,line,word=z[:8]; context=' '.join(lines.get((block,line),[str(text)])); cls=self._class_token_context(str(text),context)
                out.append({'text':str(text),'context':context,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':cls,'cx':((x0+x1)/2)/r.width,'cy':((y0+y1)/2)/r.height})
            doc.close(); return out
        except Exception:return []
    @staticmethod
    def _map_box(q,M,srcW,srcH):
        pts=np.float32([[[q['x']*srcW,q['y']*srcH],[(q['x']+q['w'])*srcW,(q['y']+q['h'])*srcH]]]); p=cv2.transform(pts,M)[0]; x,y=float(p[:,0].min()),float(p[:,1].min()); xx,yy=float(p[:,0].max()),float(p[:,1].max()); return {**q,'x':x,'y':y,'w':max(1,xx-x),'h':max(1,yy-y),'cx':(x+xx)/2,'cy':(y+yy)/2}
    @staticmethod
    def _iou(a,b):
        ax,ay,axx,ayy=a.xyxy(); bx,by,bxx,byy=b.xyxy(); inter=max(0,min(axx,bxx)-max(ax,bx))*max(0,min(ayy,byy)-max(ay,by)); return inter/max(1,a.w*a.h+b.w*b.h-inter)
    @staticmethod
    def _word_box(w,pad=14):
        x=int(round(w['x'])); y=int(round(w['y'])); xx=int(round(w['x']+w['w'])); yy=int(round(w['y']+w['h'])); return Box(x,y,max(1,xx-x),max(1,yy-y)).pad(pad,10**9,10**9)
    def _pixel_regions(self,before,after):
        if before.shape[:2]!=after.shape[:2]: return []
        a=self._gray(before); b=self._gray(after); diff=cv2.absdiff(a,b); _,th=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY); th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); n,_,stats,_=cv2.connectedComponentsWithStats(th,8); total=a.size; out=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area>=180 and area<=total*.01 and w>=6 and h>=6: out.append(Box(int(x),int(y),int(w),int(h)))
        return out if len(out)<=40 else []
    def detect(self,before_page,after_page,aligned_after=None,alignment_matrix=None):
        try:
            before=self._img(before_page); after_original=self._img(after_page); view=self._img(aligned_after) if aligned_after is not None else after_original; H0,W0=before.shape[:2]; HV,WV=view.shape[:2]; old=self._words(before_page); new=self._words(after_page)
            mapped=[]
            if alignment_matrix is not None:
                M=np.asarray(alignment_matrix,dtype=np.float32).reshape(2,3); mapped=[self._map_box(q,M,after_original.shape[1],after_original.shape[0]) for q in new]
            else: mapped=new
            sx,sy=WV/max(1,W0),HV/max(1,H0); mapped=[{**q,'x':q['x']*sx,'y':q['y']*sy,'w':q['w']*sx,'h':q['h']*sy,'cx':q['cx']*sx,'cy':q['cy']*sy} for q in mapped]; oldpx=[{**q,'x':q['x']*W0,'y':q['y']*H0,'w':q['w']*W0,'h':q['h']*H0,'cx':q['cx']*W0,'cy':q['cy']*H0} for q in old]
            candidates=[]; diag=min(WV,HV); maxd=max(35.0,diag*.018)
            for oi,o in enumerate(oldpx):
                for ni,n in enumerate(mapped):
                    if o['class']!=n['class']: continue
                    d=float(np.hypot(o['cx']-n['cx'],o['cy']-n['cy'])); size_ratio=max(o['h'],n['h'])/max(1.0,min(o['h'],n['h']))
                    if d<=maxd and size_ratio<=1.8: candidates.append((d+abs(o['h']-n['h'])*2,oi,ni))
            candidates.sort(); used_o=set(); used_n=set(); pairs=[]
            for score,oi,ni in candidates:
                if oi in used_o or ni in used_n: continue
                used_o.add(oi); used_n.add(ni); pairs.append((oldpx[oi],mapped[ni],score))
            regions=[]
            for o,n,score in pairs:
                if self._norm_text(o['text'])==self._norm_text(n['text']): continue
                ob=self._word_box(o); nb=self._word_box(n); x=min(ob.x,nb.x); y=min(ob.y,nb.y); xx=max(ob.x+ob.w,nb.x+nb.w); yy=max(ob.y+ob.h,nb.y+nb.h); box=Box(x,y,xx-x,yy-y).pad(6,WV,HV); cls=o['class']; kind=self._kind(cls)
                regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,kind,max(.5,1-score/max(1,maxd)),self._crop(before,box),self._crop(view,box),None,o['context'] if cls=='NOTE' else o['text'],n['context'] if cls=='NOTE' else n['text'],kind))
            pixels=self._pixel_regions(before,view)
            for pb in pixels:
                if any(self._iou(pb,Box(r.x,r.y,r.width,r.height))>.15 for r in regions): continue
                box=pb.pad(6,WV,HV); regions.append(ChangeRegion(box.x,box.y,box.w,box.h,box.w*box.h,0.0,'geometry_change',.45,self._crop(before,box),self._crop(view,box),None,'','', 'geometry_change'))
            final=[]
            for r in sorted(regions,key=lambda z:(z.confidence,-z.area),reverse=True):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.55 for q in final): final.append(r)
            reason=f'diag: native={len(old)}/{len(new)}, mapping=alignment_matrix, notes={sum(q["class"]=="NOTE" for q in old)}, dimensions={sum(q["class"]=="DIMENSION" for q in old)}, gdt={sum(q["class"]=="GDT" for q in old)}, text_pairs={len(pairs)}, changed_values={sum(r.change_kind!="geometry_change" for r in final)}, pixel_candidates={len(pixels)}, final={len(final)}'
            return ChangeDetectionResult(True,final,None,None,0.0,reason)
        except Exception as exc: return ChangeDetectionResult(False,[],reason=f'diag_error: {type(exc).__name__}: {exc}')
