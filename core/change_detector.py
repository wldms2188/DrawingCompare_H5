from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re, cv2, numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str='dimension_or_note'; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None
    old_text:str=''; new_text:str=''; change_kind:str=''
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
    change_pixel_ratio:float=0.0; reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    """H5: correspondence-first drawing comparison.

    Pages are normalized into a 5x4 grid. Each Before tile is matched only
    against the same/adjacent After tiles, then native text/GD&T/dimension
    items are paired inside that established region. This prevents a value
    in an unrelated drawing area from becoming a false 'added' change.
    """
    def __init__(self,config=None):
        self.pixel_threshold=38
        self.rows=4; self.cols=5
        self.tile_margin=1
        self.tile_min=.30
        self.anchor_tolerance=.22
        self.change_min=.00004

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
    def _norm(s):
        return re.sub(r'[^A-Z0-9Ø⌀±+\-.*/X°⌖⌯⏥⌒∥⊥]','',str(s).upper().replace('—','-').replace('–','-').replace('−','-'))
    @staticmethod
    def _class(t):
        t=str(t).strip().upper()
        if re.search(r'POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|[⌖⌯⏥⌒∥⊥]',t): return 'gdt'
        if re.fullmatch(r'(?:[RMD]?\s*)?[0-9]+(?:\.[0-9]+)?(?:\s*(?:MM|IN|°|DEG))?',t) or re.fullmatch(r'[0-9]+/[0-9]+',t): return 'dimension'
        if any(k in t for k in ('NOTE','TYP','UNLESS','MATERIAL','FINISH','REMOVE','BURR','INSPECT','SEE')): return 'note'
        if re.search(r'[A-Z]',t) and len(t)>=2:return 'note'
        return 'other'
    def _words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path)); p=doc.load_page(int(page.page_index)); r=p.rect; out=[]
            for z in p.get_text('words'):
                x0,y0,x1,y1,text,*_=z; text=str(text).strip()
                if text: out.append({'text':text,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(text)})
            doc.close(); return out
        except Exception:return []
    @staticmethod
    def _px(z,shape):
        h,w=shape[:2]; return {**z,'px':(z['x']+z['w']/2)*w,'py':(z['y']+z['h']/2)*h,'pw':max(1,z['w']*w),'ph':max(1,z['h']*h)}
    @staticmethod
    def _crop(img,x0,y0,x1,y1):
        h,w=img.shape[:2]; return img[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
    @staticmethod
    def _structure(img,size=256):
        if img.size==0:return np.zeros((size,size),np.uint8)
        g=cv2.resize(ChangeDetector._gray(img),(size,size),interpolation=cv2.INTER_CUBIC)
        b=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,8)
        h=cv2.morphologyEx(b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(11,1)))
        v=cv2.morphologyEx(b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,11)))
        e=cv2.Canny(g,40,120)
        return cv2.bitwise_or(cv2.bitwise_or(h,v),e)
    @classmethod
    def _score(cls,a,b):
        if a.size==0 or b.size==0:return 0.0
        aa=cls._structure(a);bb=cls._structure(b)
        corr=max(0,float(cv2.matchTemplate(aa,bb,cv2.TM_CCOEFF_NORMED)[0,0]))
        da=float(np.mean(aa>0));db=float(np.mean(bb>0))
        density=1-min(1,abs(da-db)*5)
        return .72*corr+.28*density
    def _tile_box(self,shape,r,c,pad=0):
        h,w=shape[:2]; tw=w/self.cols;th=h/self.rows
        x0=max(0,int((c-pad)*tw));y0=max(0,int((r-pad)*th));x1=min(w,int((c+1+pad)*tw));y1=min(h,int((r+1+pad)*th))
        return x0,y0,x1,y1
    def _tile_maps(self,before,after):
        maps=[]; scores=[]
        for r in range(self.rows):
            for c in range(self.cols):
                b=self._crop(before,*self._tile_box(before.shape,r,c))
                cand=[]
                for rr in range(max(0,r-self.tile_margin),min(self.rows,r+self.tile_margin+1)):
                    for cc in range(max(0,c-self.tile_margin),min(self.cols,c+self.tile_margin+1)):
                        a=self._crop(after,*self._tile_box(after.shape,rr,cc))
                        cand.append((self._score(b,a),rr,cc))
                cand.sort(reverse=True); best=cand[0] if cand else (0,r,c)
                maps.append((r,c,best[1],best[2]));scores.append(best[0])
        return maps,scores
    def _anchors(self,words,shape):
        return [self._px(x,shape) for x in words if x['class']!='other']
    def _pair_in_tile(self,ba,aa,r,c,ar,ac,bs,as_):
        out=[]; used=set()
        for i,o in enumerate(ba):
            if not (c/ self.cols <= o['x'] < (c+1)/self.cols and r/self.rows <= o['y'] < (r+1)/self.rows):continue
            best=None
            for j,n in enumerate(aa):
                if j in used or n['class']!=o['class']:continue
                if not (ac/self.cols <= n['x'] < (ac+1)/self.cols and ar/self.rows <= n['y'] < (ar+1)/self.rows):continue
                dx=(o['x']-(c+.5)/self.cols) - (n['x']-(ac+.5)/self.cols)
                dy=(o['y']-(r+.5)/self.rows) - (n['y']-(ar+.5)/self.rows)
                d=(dx*dx+dy*dy)**.5
                # normalized position inside mapped tiles is the primary correspondence key
                ox=(o['x']*self.cols-c); oy=(o['y']*self.rows-r)
                nx=(n['x']*self.cols-ac); ny=(n['y']*self.rows-ar)
                pd=((ox-nx)**2+(oy-ny)**2)**.5
                if pd>self.anchor_tolerance:continue
                text_sim=1.0 if self._norm(o['text'])==self._norm(n['text']) else .0
                score=pd+.08*(1-text_sim)+.15*d
                if best is None or score<best[0]:best=(score,j,n)
            if best:
                used.add(best[1]);out.append((o,best[2],best[0]))
        return out
    def _box_for(self,q,w,h):
        # Keep the complete nearby dimension/leader context, not only the glyph.
        pad=max(90,int(max(q['pw'],q['ph'])*6))
        return max(0,int(q['px']-q['pw']/2-pad)),max(0,int(q['py']-q['ph']/2-pad)),min(w,int(q['px']+q['pw']/2+pad)),min(h,int(q['py']+q['ph']/2+pad))
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)
    def _dedup(self,rs):
        out=[]
        for r in sorted(rs,key=lambda z:-z.confidence):
            if any(self._iou((r.x,r.y,r.width,r.height),(q.x,q.y,q.width,q.height))>.20 for q in out):continue
            out.append(r)
        return out
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after=self._img(aligned_after) if aligned_after is not None else self._img(after_page)
            h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            g0=self._gray(before);g1=self._gray(after);diff=cv2.absdiff(g0,g1);_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._words(before_page);aw=self._words(after_page);ba=self._anchors(bw,before.shape);aa=self._anchors(aw,after.shape)
            maps,ts=self._tile_maps(before,after); pairs=[]
            for r,c,ar,ac in maps:
                if ts[r*self.cols+c]<self.tile_min:continue
                pairs.extend(self._pair_in_tile(ba,aa,r,c,ar,ac,before.shape,after.shape))
            regions=[];changed=0;rejected=0
            for o,n,pair_cost in pairs:
                if self._norm(o['text'])==self._norm(n['text']):continue
                box=self._box_for(n,w,h);local=diff[box[1]:box[3],box[0]:box[2]];ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
                if ratio<self.change_min:rejected+=1;continue
                typ={'dimension':'dimension_change','gdt':'gdt_change','note':'note_change'}.get(o['class'],'text_change')
                conf=max(.5,min(1,1-pair_cost));changed+=1
                regions.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,typ,conf,before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),o['text'],n['text'],'changed_value'))
            regions=self._dedup(regions)
            good=sum(s>=self.tile_min for s in ts)
            reason=(f'diag: native={len(bw)}/{len(aw)}, native_target={len(ba)}/{len(aa)}, mapping=normalized_tiles, '
                    f'tiles={self.rows*self.cols}, tiles_pairs={good}, anchors={len(ba)}/{len(aa)}, pairs={len(pairs)}, '
                    f'changed_values={changed}, added=0, deleted=0, rejected={rejected}, final={len(regions)}')
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')
