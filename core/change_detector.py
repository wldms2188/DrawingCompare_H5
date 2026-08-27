from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re, cv2, numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str='general_change'; confidence:float=0.0
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
    success:bool; regions:List[ChangeRegion]=field(default_factory=list)
    difference_image:Optional[np.ndarray]=None; threshold_image:Optional[np.ndarray]=None
    change_pixel_ratio:float=0.0; reason:str=''
    @property
    def region(self): return self.regions

class ChangeDetector:
    """Region-first drawing comparison.

    No fixed grid. The detector builds semantic regions from visible frames,
    NOTE blocks and connected drawing structure. A region is matched globally
    by normalized geometry/edge structure. Only after a region correspondence
    is established are native PDF text items compared inside that region.
    Small dimensions and GD&T therefore inherit the context of their drawing
    instead of being matched as isolated numbers.
    """
    def __init__(self,config=None):
        self.pixel_threshold=34
        self.min_region_area=900
        self.region_gap=28
        self.match_threshold=.40
        self.change_threshold=.00002
        self.text_position_tol=.075

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
            doc=fitz.open(page.pdf_path); p=doc.load_page(int(page.page_index)); r=p.rect; out=[]
            for z in p.get_text('words'):
                x0,y0,x1,y1,text,*_=z; text=str(text).strip()
                if text: out.append({'text':text,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(text)})
            doc.close(); return out
        except Exception:return []
    @staticmethod
    def _crop(img,box):
        x0,y0,x1,y1=map(int,box); h,w=img.shape[:2]
        return img[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
    @staticmethod
    def _expand(box,pad,w,h):
        x,y,ww,hh=box
        return max(0,int(x-pad)),max(0,int(y-pad)),min(w,int(x+ww+pad)),min(h,int(y+hh+pad))
    def _visual_regions(self,img):
        """Extract actual connected drawing regions, not artificial page tiles."""
        g=self._gray(img); h,w=g.shape
        bw=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,9)
        # Join close strokes so a frame + its contents + attached leaders form one component.
        k=max(3,int(min(h,w)/1000)); k += k%2==0
        closed=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k*3,k*3)))
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        raw=[]
        for c in contours:
            x,y,ww,hh=cv2.boundingRect(c); area=ww*hh
            if area<self.min_region_area or ww<18 or hh<18: continue
            if ww>.92*w and hh>.92*h: continue
            raw.append([x,y,ww,hh])
        # Merge boxes that are spatially connected/near. This deliberately keeps dimensions
        # attached to their nearby structure rather than making a separate number region.
        merged=True
        while merged:
            merged=False; out=[]; used=[False]*len(raw)
            for i,a in enumerate(raw):
                if used[i]: continue
                x,y,ww,hh=a; used[i]=True
                again=True
                while again:
                    again=False
                    ax0,ay0,ax1,ay1=x-self.region_gap,y-self.region_gap,x+ww+self.region_gap,y+hh+self.region_gap
                    for j,b in enumerate(raw):
                        if used[j]: continue
                        bx,by,bw,bh=b
                        if bx<ax1 and bx+bw>ax0 and by<ay1 and by+bh>ay0:
                            nx,ny=min(x,bx),min(y,by); xx,yy=max(x+ww,bx+bw),max(y+hh,by+bh)
                            x,y,ww,hh=nx,ny,xx-nx,yy-ny; used[j]=True; again=True; merged=True
                out.append([x,y,ww,hh])
            raw=out
        return raw
    @staticmethod
    def _edge_score(a,b):
        if a.size==0 or b.size==0:return 0.0
        aa=cv2.resize(ChangeDetector._gray(a),(384,384),interpolation=cv2.INTER_CUBIC)
        bb=cv2.resize(ChangeDetector._gray(b),(384,384),interpolation=cv2.INTER_CUBIC)
        ae=cv2.Canny(aa,30,105); be=cv2.Canny(bb,30,105)
        corr=float(cv2.matchTemplate(ae,be,cv2.TM_CCOEFF_NORMED)[0,0])
        da=float(np.mean(ae>0)); db=float(np.mean(be>0))
        density=1-min(1,abs(da-db)*6)
        return max(0,.80*corr+.20*density)
    def _region_kind(self,box,words,shape):
        x,y,w,h=box; vals=[]
        for q in words:
            px=(q['x']+q['w']/2)*shape[1]; py=(q['y']+q['h']/2)*shape[0]
            if x<=px<=x+w and y<=py<=y+h: vals.append(q)
        text=' '.join(q['text'] for q in vals).upper()
        if 'NOTE' in text or any(q['class']=='note' for q in vals): return 'note'
        if any(q['class']=='gdt' for q in vals): return 'gdt_region'
        if any(q['class']=='dimension' for q in vals): return 'dimension_region'
        return 'structural_region'
    def _match_regions(self,before,after,br,ar):
        bh,bw=before.shape[:2]; ah,aw=after.shape[:2]; candidates=[]
        for i,(x,y,w,h) in enumerate(br):
            bc=self._crop(before,(x,y,x+w,y+h)); bcx=(x+w/2)/bw; bcy=(y+h/2)/bh
            for j,(xx,yy,ww,hh) in enumerate(ar):
                ac=self._crop(after,(xx,yy,xx+ww,yy+hh)); acx=(xx+ww/2)/aw; acy=(yy+hh/2)/ah
                pos=((bcx-acx)**2+(bcy-acy)**2)**.5
                size=abs(np.log(max(1,w)/max(1,ww)))+abs(np.log(max(1,h)/max(1,hh)))
                vs=self._edge_score(bc,ac)
                # visual structure is strongest; page-relative position/scale break ties.
                score=.68*vs+.20*max(0,1-pos*2.5)+.12*max(0,1-size/2.2)
                candidates.append((score,vs,i,j,pos,size))
        candidates.sort(reverse=True)
        used_b=set(); used_a=set(); pairs=[]
        for score,vs,i,j,pos,size in candidates:
            if i in used_b or j in used_a or score<self.match_threshold: continue
            used_b.add(i);used_a.add(j);pairs.append((i,j,score,vs))
        return pairs
    def _inside_words(self,words,box,shape):
        x,y,w,h=box;out=[]
        for q in words:
            px=(q['x']+q['w']/2)*shape[1];py=(q['y']+q['h']/2)*shape[0]
            if x<=px<=x+w and y<=py<=y+h:out.append(q)
        return out
    def _word_pairs(self,old,new,old_shape,new_shape):
        pairs=[]; used=set()
        # First prefer identical normalized text: it gives stable local anchors without
        # turning an unmatched value into an 'added' change.
        for i,o in enumerate(old):
            best=None
            for j,n in enumerate(new):
                if j in used or n['class']!=o['class']: continue
                d=((o['x']+o['w']/2-(n['x']+n['w']/2))**2+(o['y']+o['h']/2-(n['y']+n['h']/2))**2)**.5
                if d>self.text_position_tol: continue
                same=self._norm(o['text'])==self._norm(n['text'])
                score=d-(.045 if same else 0)
                if best is None or score<best[0]: best=(score,j,n,same)
            if best:
                used.add(best[1]);pairs.append((o,best[2],best[3]))
        return pairs
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); after=self._img(aligned_after) if aligned_after is not None else self._img(after_page)
            old_words=self._words(before_page); new_words=self._words(after_page)
            br=self._visual_regions(before); ar=self._visual_regions(after)
            region_pairs=self._match_regions(before,after,br,ar)
            gray0=self._gray(before); gray1=self._gray(after)
            if gray1.shape!=gray0.shape: gray1=cv2.resize(gray1,(gray0.shape[1],gray0.shape[0]),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(gray0,gray1); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            regions=[]; changed=0
            for bi,ai,score,visual in region_pairs:
                bx,by,bw,bh=br[bi]; ax,ay,aw,ah=ar[ai]
                pad=max(16,int(max(aw,ah)*.18))
                # For reporting use the matched After region with generous context for attached dimensions/leaders.
                x,y,x2,y2=self._expand((ax,ay,aw,ah),pad,after.shape[1],after.shape[0]); box=(x,y,x2-x,y2-y)
                old_box=self._expand((bx,by,bw,bh),pad,before.shape[1],before.shape[0])
                old_local=self._crop(before,old_box); new_local=self._crop(after,box); local_diff=self._crop(diff,box)
                ratio=float(np.mean(local_diff>self.pixel_threshold)) if local_diff.size else 0
                ow=self._inside_words(old_words,old_box,before.shape); nw=self._inside_words(new_words,box,after.shape)
                for o,n,same in self._word_pairs(ow,nw,before.shape,after.shape):
                    if same: continue
                    changed+=1; kind=self._region_kind(box,old_words,before.shape)
                    typ={'note':'note_change','gdt_region':'gdt_change','dimension_region':'dimension_change'}.get(kind,'structural_text_change')
                    regions.append(ChangeRegion(x,y,x2-x,y2-y,(x2-x)*(y2-y),ratio,typ,min(1,score),old_local.copy(),new_local.copy(),local_diff.copy(),o['text'],n['text'],'changed_value'))
                # If native PDF text is absent, retain a high-confidence visual change only when
                # the matched region itself has meaningful changed pixels. This handles vector/raster text.
                if not ow or not nw:
                    if ratio>=self.change_threshold and visual>=.55:
                        regions.append(ChangeRegion(x,y,x2-x,y2-y,(x2-x)*(y2-y),ratio,'visual_change',min(1,score),old_local.copy(),new_local.copy(),local_diff.copy(),'','', 'visual_change'))
            # Merge duplicate reports generated by multiple changed words in the same semantic region.
            out=[]
            for r in sorted(regions,key=lambda q:-q.confidence):
                if any(self._iou((r.x,r.y,r.width,r.height),(q.x,q.y,q.width,q.height))>.55 for q in out): continue
                out.append(r)
            reason=(f'diag: native={len(old_words)}/{len(new_words)}, mapping=semantic_regions, '
                    f'regions={len(br)}/{len(ar)}, region_pairs={len(region_pairs)}, changed_values={changed}, '
                    f'added=0, deleted=0, final={len(out)}')
            return ChangeDetectionResult(True,out,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')

    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)
