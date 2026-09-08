from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    def xyxy(self): return self.x, self.y, self.x + self.w, self.y + self.h
    def pad(self, p: int, W: int, H: int):
        x=max(0,self.x-p); y=max(0,self.y-p); xx=min(W,self.x+self.w+p); yy=min(H,self.y+self.h+p)
        return Box(x,y,max(1,xx-x),max(1,yy-y))

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str='general_change'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None
    difference_crop:Optional[np.ndarray]=None; old_text:str=''; new_text:str=''; change_kind:str='changed_value'

@dataclass
class ChangeDetectionResult:
    success:bool
    regions:List[ChangeRegion]=field(default_factory=list)
    difference_image:Optional[np.ndarray]=None
    threshold_image:Optional[np.ndarray]=None
    change_pixel_ratio:float=0.0
    reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    """Native-PDF-word comparison. Coordinates stay in raw-page space until raster crops."""
    def __init__(self, config=None):
        self.pixel_threshold=35
        self.max_word_distance_ratio=.035
        self.min_confidence=.55

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return np.asarray(page)
        if hasattr(page,'image'): return np.asarray(page.image)
        raise TypeError('페이지 이미지 배열을 찾을 수 없습니다.')

    @staticmethod
    def _gray(img):
        if img.ndim==2: return img.astype(np.uint8)
        if img.shape[2]==4: return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop(img,b):
        H,W=img.shape[:2]; x=max(0,min(W-1,b.x)); y=max(0,min(H-1,b.y)); xx=max(x+1,min(W,b.x+b.w)); yy=max(y+1,min(H,b.y+b.h)); return img[y:yy,x:xx]

    @staticmethod
    def _norm(t): return re.sub(r'\s+','',str(t).upper().replace('−','-').replace('–','-').replace('—','-'))

    @staticmethod
    def _class(text,context=''):
        u=str(text).strip().upper(); c=str(context).upper()
        if re.search(r'NOTE|NOTES|MATERIAL|FINISH|DEBURR|BURR|UNLESS|REMARK|COMMENT',c): return 'NOTE'
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|POS',c): return 'GDT'
        if re.search(r'±|Ø|⌀|\bR\s*\d|^\d+(?:\.\d+)?$',u): return 'DIMENSION'
        return 'TEXT'

    @staticmethod
    def _value_like(text):
        """True for engineering values/codes where old->new is a value change, not add/delete."""
        u=str(text).strip().upper()
        if not u: return False
        if re.search(r'±|Ø|⌀|^R\s*[-+]?\d|^[-+]?\d+(?:\.\d+)?$',u): return True
        if re.fullmatch(r'[A-Z]{1,8}[-_/]?[A-Z0-9]{2,12}',u) and re.search(r'\d',u): return True
        return False

    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(page.pdf_path); p=doc.load_page(int(page.page_index)); pw,ph=p.rect.width,p.rect.height
            raw=p.get_text('words'); line_map={}
            for z in raw:
                if len(z)>=8: line_map.setdefault((z[5],z[6]),[]).append(str(z[4]).strip())
            out=[]
            for z in raw:
                if len(z)<8 or not str(z[4]).strip(): continue
                x0,y0,x1,y1,text,block,line,word=z[:8]; context=' '.join(line_map.get((block,line),[text]))
                out.append({'text':str(text),'context':context,'x':x0/pw,'y':y0/ph,'w':(x1-x0)/pw,'h':(y1-y0)/ph,'cx':((x0+x1)/2)/pw,'cy':((y0+y1)/2)/ph,'class':self._class(text,context)})
            doc.close(); return out
        except Exception: return []

    @staticmethod
    def _affine_point(x,y,M):
        p=np.array([x,y,1.],dtype=np.float32); q=np.asarray(M,dtype=np.float32).reshape(2,3)@p; return float(q[0]),float(q[1])

    def _mapped_words(self,words,after_shape,view_shape,M):
        H,W=after_shape[:2]; HV,WV=view_shape[:2]; out=[]
        for q in words:
            pts=[self._affine_point(q['x']*W,q['y']*H,M),self._affine_point((q['x']+q['w'])*W,(q['y']+q['h'])*H,M)]
            x=min(a[0] for a in pts); y=min(a[1] for a in pts); xx=max(a[0] for a in pts); yy=max(a[1] for a in pts)
            out.append({**q,'x':x,'y':y,'w':max(1,xx-x),'h':max(1,yy-y),'cx':(x+xx)/2,'cy':(y+yy)/2})
        return out

    @staticmethod
    def _box(q,pad,W,H):
        x=int(round(q['x'])); y=int(round(q['y'])); xx=int(round(q['x']+q['w'])); yy=int(round(q['y']+q['h']))
        return Box(x,y,max(1,xx-x),max(1,yy-y)).pad(pad,W,H)

    @staticmethod
    def _iou(a,b):
        ax,ay,axx,ayy=a.xyxy(); bx,by,bxx,byy=b.xyxy(); iw=max(0,min(axx,bxx)-max(ax,bx)); ih=max(0,min(ayy,byy)-max(ay,by)); inter=iw*ih; return inter/max(1,a.w*a.h+b.w*b.h-inter)

    def _pair_words(self,old,new,W,H):
        maxd=max(20,min(W,H)*self.max_word_distance_ratio); cand=[]
        for i,o in enumerate(old):
            for j,n in enumerate(new):
                if o['class']!=n['class']: continue
                d=float(np.hypot(o['cx']-n['cx'],o['cy']-n['cy']))
                if d>maxd: continue
                size=max(o['h'],n['h'])/max(1,min(o['h'],n['h']))
                if size>2.2: continue
                text_bonus=0 if self._norm(o['text'])==self._norm(n['text']) else -0.25
                cand.append((d/(maxd)+text_bonus,i,j))
        cand.sort(); used_o=set(); used_n=set(); pairs=[]
        for score,i,j in cand:
            if i in used_o or j in used_n: continue
            used_o.add(i); used_n.add(j); pairs.append((old[i],new[j],score))
        return pairs,used_o,used_n

    def _add_region(self,regions,before,after,b,kind,old_text='',new_text='',confidence=.65):
        blank=np.full((b.h,b.w,3),255,np.uint8)
        regions.append(ChangeRegion(b.x,b.y,b.w,b.h,b.w*b.h,0.0,kind,confidence,
                                    self._crop(before,b) if old_text else blank.copy(),
                                    self._crop(after,b) if new_text else blank.copy(),
                                    None,old_text,new_text,kind))

    def detect(self,before_page,after_page,aligned_after=None,alignment_matrix=None):
        try:
            before=self._img(before_page); after=self._img(aligned_after if aligned_after is not None else after_page); H,W=before.shape[:2]
            old=self._words(before_page); raw_new=self._words(after_page)
            M=alignment_matrix
            if M is not None:
                new=self._mapped_words(raw_new,self._img(after_page).shape,after.shape,M)
            else:
                ah,aw=self._img(after_page).shape[:2]; sx=W/max(1,aw); sy=H/max(1,ah)
                new=[{**q,'x':q['x']*aw*sx,'y':q['y']*ah*sy,'w':q['w']*aw*sx,'h':q['h']*ah*sy,'cx':q['cx']*aw*sx,'cy':q['cy']*ah*sy} for q in raw_new]
            oldpx=[{**q,'x':q['x']*W,'y':q['y']*H,'w':q['w']*W,'h':q['h']*H,'cx':q['cx']*W,'cy':q['cy']*H} for q in old]
            pairs,used_o,used_n=self._pair_words(oldpx,new,W,H)
            regions=[]
            for o,n,score in pairs:
                if self._norm(o['text'])==self._norm(n['text']): continue
                cls=o['class']; ob=self._box(o,10,W,H); nb=self._box(n,10,W,H)
                x=min(ob.x,nb.x); y=min(ob.y,nb.y); xx=max(ob.x+ob.w,nb.x+nb.w); yy=max(ob.y+ob.h,nb.y+nb.h); b=Box(x,y,xx-x,yy-y).pad(8,W,H)
                conf=max(.55,min(.99,1-score))
                if cls in ('NOTE','TEXT') and not (self._value_like(o['text']) and self._value_like(n['text'])):
                    okind='note_deleted' if cls=='NOTE' else 'text_deleted'; nkind='note_added' if cls=='NOTE' else 'text_added'
                    self._add_region(regions,before,after,ob,okind,o['text'],'',conf)
                    self._add_region(regions,before,after,nb,nkind,'',n['text'],conf)
                else:
                    typ={'DIMENSION':'dimension_change','GDT':'gdt_change','NOTE':'note_change','TEXT':'text_change'}[cls]
                    regions.append(ChangeRegion(b.x,b.y,b.w,b.h,b.w*b.h,0,typ,conf,self._crop(before,b),self._crop(after,b),None,o['text'],n['text'],typ))
            matched_boxes=[self._box(o,28,W,H) for o,_,_ in pairs]+[self._box(n,28,W,H) for _,n,_ in pairs]
            for side,words,used in [('deleted',oldpx,used_o),('added',new,used_n)]:
                for i,q in enumerate(words):
                    if i in used: continue
                    qb=self._box(q,10,W,H)
                    if not any(self._iou(qb,m)>0.02 or (abs(q['cx']-(m.x+m.w/2))<max(35,m.w) and abs(q['cy']-(m.y+m.h/2))<max(35,m.h)) for m in matched_boxes): continue
                    cls=q['class']; typ={'DIMENSION':'dimension_added' if side=='added' else 'dimension_deleted','GDT':'gdt_added' if side=='added' else 'gdt_deleted','NOTE':'note_added' if side=='added' else 'note_deleted','TEXT':'text_added' if side=='added' else 'text_deleted'}[cls]
                    b=qb.pad(8,W,H); self._add_region(regions,before,after,b,typ,q['text'] if side=='deleted' else '',q['text'] if side=='added' else '',.65)
            if before.shape[:2]==after.shape[:2]:
                a=self._gray(before); bgray=self._gray(after); d=cv2.absdiff(a,bgray); _,th=cv2.threshold(d,self.pixel_threshold,255,cv2.THRESH_BINARY); th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)); n,_,stats,_=cv2.connectedComponentsWithStats(th,8)
                for k in range(1,n):
                    x,y,w,h,area=stats[k]
                    if area<250 or area>before.size*.01: continue
                    b=Box(int(x),int(y),int(w),int(h)).pad(8,W,H)
                    if any(self._iou(b,Box(r.x,r.y,r.width,r.height))>.15 for r in regions): continue
                    regions.append(ChangeRegion(b.x,b.y,b.w,b.h,b.w*b.h,area/max(1,b.w*b.h),'geometry_change',.60,self._crop(before,b),self._crop(after,b),self._crop(d,b),'','', 'geometry_change'))
            final=[]
            for r in sorted(regions,key=lambda z:(z.confidence,z.area),reverse=True):
                rb=Box(r.x,r.y,r.width,r.height)
                if not any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.60 for q in final): final.append(r)
            reason=f'v2 native={len(old)}/{len(raw_new)}, pairs={len(pairs)}, changed_values={sum(1 for r in final if r.change_kind.endswith("change") and r.change_kind!="geometry_change")}, unmatched={len(old)-len(used_o)}/{len(raw_new)-len(used_n)}, final={len(final)}'
            return ChangeDetectionResult(True,final,None,None,0.0,reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f'v2_error: {type(exc).__name__}: {exc}')
