from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass(frozen=True)
class Box:
    x:int; y:int; w:int; h:int
    def xyxy(self): return (self.x,self.y,self.x+self.w,self.y+self.h)
    def pad(self,p,W,H):
        x=max(0,self.x-int(p)); y=max(0,self.y-int(p))
        return Box(x,y,min(W,self.x+self.w+int(p))-x,min(H,self.y+self.h+int(p))-y)
    def norm(self,W,H): return (self.x/W,self.y/H,self.w/W,self.h/H)
    @staticmethod
    def from_norm(v,W,H): return Box(round(v[0]*W),round(v[1]*H),round(v[2]*W),round(v[3]*H))

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
    """H5 region-first detector.

    Coordinate contract: every internal rectangle is Box(x,y,w,h). Conversion
    to xyxy occurs only inside _crop(). Native PDF text is stored in normalized
    PAGE coordinates. Raw After geometry is matched first; an aligned raster is
    used only for display/difference crops. No global anchor fallback is used,
    because it can compare unrelated drawing locations.
    """
    def __init__(self,config=None):
        self.pixel_threshold=34
        self.min_region_area=650
        self.region_gap=28
        self.match_threshold=.50
        self.max_pos=.16
        self.max_size_log=1.8
        self.text_tol=.11

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
    def _crop(img,b:Box):
        x0,y0,x1,y1=b.xyxy(); H,W=img.shape[:2]
        return img[max(0,y0):min(H,y1),max(0,x0):min(W,x1)]
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
            doc=fitz.open(page.pdf_path);p=doc.load_page(int(page.page_index));r=p.rect;out=[]
            for z in p.get_text('words'):
                x0,y0,x1,y1,text,*_=z;text=str(text).strip()
                if text:out.append({'text':text,'x':x0/r.width,'y':y0/r.height,'w':(x1-x0)/r.width,'h':(y1-y0)/r.height,'class':self._class(text)})
            doc.close();return out
        except Exception:return []
    def _visual_regions(self,img):
        g=self._gray(img);H,W=g.shape
        bw=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,8)
        k=max(3,int(min(H,W)/1100)); k += k%2==0
        closed=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k*2+1,k*2+1)))
        contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        raw=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w*h<self.min_region_area or w<16 or h<12 or (w>.94*W and h>.94*H):continue
            raw.append(Box(x,y,w,h))
        merged=True
        while merged:
            merged=False;out=[];used=[False]*len(raw)
            for i,a in enumerate(raw):
                if used[i]:continue
                cur=a;used[i]=True;again=True
                while again:
                    again=False;A=cur.pad(self.region_gap,W,H).xyxy()
                    for j,b in enumerate(raw):
                        if used[j]:continue
                        B=b.xyxy()
                        if B[0]<A[2] and B[2]>A[0] and B[1]<A[3] and B[3]>A[1]:
                            x=min(cur.x,b.x);y=min(cur.y,b.y);xx=max(cur.x+cur.w,b.x+b.w);yy=max(cur.y+cur.h,b.y+b.h)
                            cur=Box(x,y,xx-x,yy-y);used[j]=True;again=True;merged=True
                out.append(cur)
            raw=out
        return raw
    @staticmethod
    def _edge_score(a,b):
        if a.size==0 or b.size==0:return 0.0
        aa=cv2.resize(ChangeDetector._gray(a),(384,384),interpolation=cv2.INTER_CUBIC)
        bb=cv2.resize(ChangeDetector._gray(b),(384,384),interpolation=cv2.INTER_CUBIC)
        ae=cv2.Canny(aa,25,110);be=cv2.Canny(bb,25,110)
        corr=max(0.0,float(cv2.matchTemplate(ae,be,cv2.TM_CCOEFF_NORMED)[0,0]))
        da=float(np.mean(ae>0));db=float(np.mean(be>0));dens=max(0.0,1-min(1,abs(da-db)*6))
        # Edge correlation is intentionally not dominant because changed text
        # can lower it even when the parent drawing structure is identical.
        return .55*corr+.45*dens
    def _match_regions(self,before,after,br,ar):
        H0,W0=before.shape[:2];H1,W1=after.shape[:2];cand=[]
        for i,b in enumerate(br):
            bn=b.norm(W0,H0);bcx=bn[0]+bn[2]/2;bcy=bn[1]+bn[3]/2;bc=self._crop(before,b)
            for j,a in enumerate(ar):
                an=a.norm(W1,H1);acx=an[0]+an[2]/2;acy=an[1]+an[3]/2
                pos=((bcx-acx)**2+(bcy-acy)**2)**.5
                size=abs(np.log(max(bn[2],1e-6)/max(an[2],1e-6)))+abs(np.log(max(bn[3],1e-6)/max(an[3],1e-6)))
                if pos>self.max_pos or size>self.max_size_log:continue
                vs=self._edge_score(bc,self._crop(after,a))
                score=.50*vs+.35*(1-pos/self.max_pos)+.15*(1-size/self.max_size_log)
                cand.append((score,i,j))
        cand.sort(reverse=True);ub=set();ua=set();pairs=[]
        for score,i,j in cand:
            if score<self.match_threshold or i in ub or j in ua:continue
            ub.add(i);ua.add(j);pairs.append((i,j,score))
        return pairs
    def _inside_words(self,words,b:Box,W,H):
        bx,by,bw,bh=b.norm(W,H);return [q for q in words if bx<=q['x']+q['w']/2<=bx+bw and by<=q['y']+q['h']/2<=by+bh]
    def _pair_words(self,old,new,ob:Box,nb:Box,W0,H0,W1,H1):
        ox,oy,ow,oh=ob.norm(W0,H0);nx,ny,nw,nh=nb.norm(W1,H1)
        def rel(q,x,y,w,h):return ((q['x']+q['w']/2-x)/max(w,1e-9),(q['y']+q['h']/2-y)/max(h,1e-9))
        out=[];used=set()
        for o in old:
            a=rel(o,ox,oy,ow,oh);best=None
            for j,n in enumerate(new):
                if j in used:continue
                b=rel(n,nx,ny,nw,nh);d=((a[0]-b[0])**2+(a[1]-b[1])**2)**.5
                if d>self.text_tol:continue
                s=d+(0 if o['class']==n['class'] else .025)
                if best is None or s<best[0]:best=(s,j,n)
            if best:used.add(best[1]);out.append((o,best[2],self._norm(o['text'])==self._norm(best[2]['text'])))
        return out
    def _kind(self,words,b,W,H):
        vals=self._inside_words(words,b,W,H);text=' '.join(v['text'] for v in vals).upper()
        if 'NOTE' in text or any(v['class']=='note' for v in vals):return 'note_change'
        if any(v['class']=='gdt' for v in vals):return 'gdt_change'
        if any(v['class']=='dimension' for v in vals):return 'dimension_change'
        return 'text_change'
    @staticmethod
    def _iou(a:Box,b:Box):
        A=a.xyxy();B=b.xyxy();x=max(A[0],B[0]);y=max(A[1],B[1]);xx=min(A[2],B[2]);yy=min(A[3],B[3]);inter=max(0,xx-x)*max(0,yy-y);u=a.w*a.h+b.w*b.h-inter
        return inter/max(1,u)
    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after_raw=self._img(after_page)
            # NEVER mix raw After coordinates with aligned After coordinates.
            after_view=self._img(aligned_after) if aligned_after is not None else after_raw
            old_words=self._words(before_page);new_words=self._words(after_page)
            br=self._visual_regions(before);ar=self._visual_regions(after_raw)
            pairs=self._match_regions(before,after_raw,br,ar)
            H0,W0=before.shape[:2];H1,W1=after_raw.shape[:2];HV,WV=after_view.shape[:2]
            g0=self._gray(before);g1=self._gray(after_view)
            if g1.shape!=g0.shape:g1=cv2.resize(g1,(W0,H0),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(g0,g1);_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            regions=[];changed=0
            for bi,ai,score in pairs:
                ob=br[bi];nb=ar[ai]
                ob_pad=ob.pad(max(16,int(max(ob.w,ob.h)*.18)),W0,H0)
                nb_pad=nb.pad(max(16,int(max(nb.w,nb.h)*.18)),W1,H1)
                # Explicit normalized conversion is the ONLY bridge between raw
                # After and aligned After coordinate spaces.
                vb=Box.from_norm(nb_pad.norm(W1,H1),WV,HV)
                old_local=self._crop(before,ob_pad);new_local=self._crop(after_view,vb);local_diff=self._crop(diff,vb)
                ow=self._inside_words(old_words,ob_pad,W0,H0);nw=self._inside_words(new_words,nb_pad,W1,H1)
                for o,n,same in self._pair_words(ow,nw,ob_pad,nb_pad,W0,H0,W1,H1):
                    if same:continue
                    changed+=1;ratio=float(np.mean(local_diff>self.pixel_threshold)) if local_diff.size else 0
                    typ=self._kind(new_words,nb_pad,W1,H1)
                    regions.append(ChangeRegion(vb.x,vb.y,vb.w,vb.h,vb.w*vb.h,ratio,typ,min(1,score),old_local.copy(),new_local.copy(),local_diff.copy(),o['text'],n['text'],'changed_value'))
            out=[]
            for r in sorted(regions,key=lambda z:-z.confidence):
                rb=Box(r.x,r.y,r.width,r.height)
                if any(self._iou(rb,Box(q.x,q.y,q.width,q.height))>.55 for q in out):continue
                out.append(r)
            reason=f'diag: native={len(old_words)}/{len(new_words)}, mapping=semantic_regions, regions={len(br)}/{len(ar)}, region_pairs={len(pairs)}, changed_values={changed}, added=0, deleted=0, final={len(out)}'
            return ChangeDetectionResult(True,out,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f'diag_error: {exc}')
