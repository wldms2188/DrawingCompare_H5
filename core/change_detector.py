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
    """H5 robust detector.

    Pipeline: PDF-native text anchors -> connected/line structure blocks ->
    multi-scale correspondence -> local high-resolution comparison.
    Native text is a cue, never the sole matching criterion.  Unmatched text
    is not reported as added/deleted until a structural region has first been
    established.  This deliberately favors precision over false positives.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.context=420
        self.scales=(.50,.60,.70,.80,.90,1.0,1.10,1.25,1.40,1.60,1.80)
        self.block_min=180
        self.match_min=.46
        self.match_margin=.025
        self.change_min=.00006

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
    def _anchors(self,words,shape): return sorted([self._px(z,shape) for z in words if z['class']!='other'],key=lambda z:(z['py'],z['px']))
    @staticmethod
    def _crop(img,cx,cy,w,h=None):
        if h is None:h=w
        ih,iw=img.shape[:2]; x0=max(0,int(cx-w/2)); y0=max(0,int(cy-h/2)); x1=min(iw,int(cx+w/2)); y1=min(ih,int(cy+h/2)); return img[y0:y1,x0:x1]
    @staticmethod
    def _line_structure(img,size=256):
        g=ChangeDetector._gray(img); g=cv2.resize(g,(size,size),interpolation=cv2.INTER_CUBIC)
        b=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,8)
        hor=cv2.morphologyEx(b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(13,1)))
        ver=cv2.morphologyEx(b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,13)))
        diag=cv2.Canny(g,45,130)
        return cv2.bitwise_or(cv2.bitwise_or(hor,ver),diag)
    @staticmethod
    def _structure_score(a,b):
        if a.size==0 or b.size==0:return 0.0
        aa=ChangeDetector._line_structure(a); bb=ChangeDetector._line_structure(b)
        corr=max(0,float(cv2.matchTemplate(aa,bb,cv2.TM_CCOEFF_NORMED)[0,0]))
        da=float(np.mean(aa>0)); db=float(np.mean(bb>0)); density=1-min(1,abs(da-db)*5)
        return .72*corr+.28*density
    def _block_candidates(self,before,after,o):
        t=self._crop(before,o['px'],o['py'],self.context)
        if t.size==0:return []
        ag=self._gray(after); h,w=ag.shape
        # coarse search, followed by exact-resolution re-scoring
        sh=max(900,min(1800,w)); sv=max(700,min(1800,h)); search=cv2.resize(ag,(sh,sv),interpolation=cv2.INTER_AREA)
        se=self._line_structure(search,512); out=[]
        for s in self.scales:
            tw=int(self.context*s); th=int(self.context*s)
            if tw>=w or th>=h:continue
            templ=self._line_structure(cv2.resize(t,(max(128,tw),max(128,th)),interpolation=cv2.INTER_CUBIC),256)
            # Always normalize template size for the coarse map; exact crop is rescored below.
            res=cv2.matchTemplate(se,templ,cv2.TM_CCOEFF_NORMED)
            for k in range(5):
                _,mx,_,loc=cv2.minMaxLoc(res)
                if mx<.15:break
                cx=(loc[0]+templ.shape[1]/2)*w/sh; cy=(loc[1]+templ.shape[0]/2)*h/sv
                exact=self._structure_score(t,self._crop(after,cx,cy,int(self.context*s)))
                out.append((exact,float(mx),cx,cy,int(self.context*s)))
                cv2.rectangle(res,(max(0,loc[0]-templ.shape[1]//2),max(0,loc[1]-templ.shape[0]//2)),(min(res.shape[1]-1,loc[0]+templ.shape[1]//2),min(res.shape[0]-1,loc[1]+templ.shape[0]//2)),-1,-1)
        out.sort(reverse=True); return out[:20]
    def _near(self,aa,cx,cy,cls,limit):
        v=[]
        for j,n in enumerate(aa):
            if n['class']!=cls:continue
            d=((n['px']-cx)**2+(n['py']-cy)**2)**.5
            if d<=limit:v.append((d,j,n))
        return min(v) if v else None
    def _match(self,before,after,ba,aa):
        cand=[]
        for i,o in enumerate(ba):
            cs=self._block_candidates(before,after,o)
            if not cs:continue
            best=cs[0]; second=cs[1][0] if len(cs)>1 else 0
            if best[0]<self.match_min or best[0]-second<self.match_margin:continue
            near=self._near(aa,best[2],best[3],o['class'],max(90,best[4]*.55))
            if not near:continue
            d,j,n=near
            # Context around the actual After anchor. This catches wrong nearby blocks.
            ctx=self._structure_score(self._crop(before,o['px'],o['py'],self.context*1.5),self._crop(after,n['px'],n['py'],self.context*1.5))
            if ctx<self.match_min:continue
            ts=self._norm(o['text']); ns=self._norm(n['text'])
            text_sim=1 if ts==ns else (len(set(ts)&set(ns))/max(1,len(set(ts)|set(ns))))
            score=.50*best[0]+.35*ctx+.15*text_sim
            cand.append((score,i,j,best[0],ctx))
        cand.sort(reverse=True); used_b=set(); used_a=set(); pairs=[]
        for z in cand:
            score,i,j,v,c=z
            if i in used_b or j in used_a:continue
            used_b.add(i);used_a.add(j);pairs.append((ba[i],aa[j],score,v,c))
        return pairs
    def _box(self,q,w,h):
        # Generous context box: dimensions are frequently outside the glyph bbox.
        pad=max(100,int(max(q['pw'],q['ph'])*7))
        return max(0,int(q['px']-q['pw']/2-pad)),max(0,int(q['py']-q['ph']/2-pad)),min(w,int(q['px']+q['pw']/2+pad)),min(h,int(q['py']+q['ph']/2+pad))
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)
    def _dedup(self,rs):
        out=[]
        for r in sorted(rs,key=lambda z:-z.confidence):
            if any(self._iou((r.x,r.y,r.width,r.height),(q.x,q.y,q.width,q.height))>.22 for q in out):continue
            out.append(r)
        return out
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page); after=self._img(aligned_after) if aligned_after is not None else self._img(after_page)
            h,w=before.shape[:2]
            if after.shape[:2]!=(h,w): after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            gray0=self._gray(before); gray1=self._gray(after); diff=cv2.absdiff(gray0,gray1); _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._words(before_page); aw=self._words(after_page); ba=self._anchors(bw,before.shape); aa=self._anchors(aw,after.shape)
            pairs=self._match(before,after,ba,aa); regions=[]; changed=0; rejected=0
            for o,n,score,visual,ctx in pairs:
                old=o['text']; new=n['text']
                if self._norm(old)==self._norm(new):continue
                box=self._box(n,w,h); local=diff[box[1]:box[3],box[0]:box[2]]; ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0
                if ratio<self.change_min:rejected+=1;continue
                typ={'dimension':'dimension_change','gdt':'gdt_change','note':'note_change'}.get(o['class'],'text_change')
                conf=min(1,.45*score+.30*visual+.25*ctx); changed+=1
                regions.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,typ,conf,before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,'changed_value'))
            regions=self._dedup(regions)
            reason=(f'diag: native={len(bw)}/{len(aw)}, native_target={sum(x["class"]!="other" for x in bw)}/{sum(x["class"]!="other" for x in aw)}, '
                    f'mapping=region_multiscale, anchors={len(ba)}/{len(aa)}, pairs={len(pairs)}, changed_values={changed}, added=0, deleted=0, rejected={rejected}, final={len(regions)}')
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')
