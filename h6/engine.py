from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import cv2
import fitz
import numpy as np
from .config import CONFIG

@dataclass
class Page:
    pdf: Path
    index: int
    image: np.ndarray

@dataclass
class Match:
    before: Page
    after: Page
    score: float
    status: str

@dataclass
class Change:
    page_before: int
    page_after: int
    kind: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    before_crop: np.ndarray
    after_crop: np.ndarray
    old_text: str = ""
    new_text: str = ""

class H6Engine:
    def __init__(self, config=CONFIG): self.cfg = config

    def load_pdf(self, path: str | Path):
        path = Path(path); doc = fitz.open(path); pages=[]
        scale=self.cfg.dpi/72.0; mat=fitz.Matrix(scale,scale)
        for i,p in enumerate(doc):
            pix=p.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            arr=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
            pages.append(Page(path,i,cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)))
        doc.close(); return pages

    @staticmethod
    def _gray(img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img

    @staticmethod
    def _thumb(img):
        g=H6Engine._gray(img); h,w=g.shape; s=600/max(h,w); s=min(1.0,s)
        g=cv2.resize(g,(max(40,int(w*s)),max(40,int(h*s))),interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g,(3,3),0)

    def _page_score(self,a,b):
        ga,gb=self._thumb(a.image),self._thumb(b.image)
        # compare layout at low resolution; this is deliberately not used for change detection
        ga=cv2.resize(ga,(512,512)); gb=cv2.resize(gb,(512,512))
        ea=cv2.Canny(ga,50,150); eb=cv2.Canny(gb,50,150)
        # small translation search prevents page matching from failing because of print margins
        best=0.0
        for dy in (-12,0,12):
            for dx in (-12,0,12):
                M=np.float32([[1,0,dx],[0,1,dy]])
                wa=cv2.warpAffine(ea,M,(512,512),borderValue=0)
                inter=np.logical_and(wa>0,eb>0).sum(); union=np.logical_or(wa>0,eb>0).sum()
                best=max(best,float(inter/max(1,union)))
        return best

    def match_pages(self,before_pages,after_pages):
        if not before_pages or not after_pages:return []
        pairs=[]; used=set()
        for bp in before_pages:
            scored=sorted(((self._page_score(bp,ap),j,ap) for j,ap in enumerate(after_pages) if j not in used),reverse=True,key=lambda z:z[0])
            if not scored: continue
            score,j,ap=scored[0]; status='MATCH' if score>=.18 else ('REVIEW' if score>=.10 else 'NO_MATCH')
            if status!='NO_MATCH': used.add(j); pairs.append(Match(bp,ap,score,status))
        return pairs

    def _align(self,bp,ap):
        b=self._gray(bp.image); a=self._gray(ap.image); h,w=b.shape
        a=cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA)
        # ECC estimates a transform that maps template->input; invert it for After->Before.
        ds=min(1.0,1000/max(h,w)); sz=(max(80,int(w*ds)),max(80,int(h*ds)))
        bs=cv2.resize(b,sz,interpolation=cv2.INTER_AREA); ass=cv2.resize(a,sz,interpolation=cv2.INTER_AREA)
        warp=np.eye(2,3,dtype=np.float32)
        try:
            score,cc=cv2.findTransformECC(bs,ass,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,1e-6),None,5)
            if ds!=1: cc[:,2]/=ds
            M=np.linalg.inv(np.vstack([cc,[0,0,1]]))[:2].astype(np.float32)
            A=M[:,:2]; scale=float(np.sqrt(abs(np.linalg.det(A)))); rot=float(np.degrees(np.arctan2(A[1,0]-A[0,1],A[0,0]+A[1,1])))
            if score<self.cfg.alignment_min_score or abs(rot)>self.cfg.max_rotation_deg or abs(scale-1)>self.cfg.max_scale_delta:return None
            aligned=cv2.warpAffine(ap.image,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=255)
            valid=cv2.warpAffine(np.ones((h,w),np.uint8)*255,M,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
            if np.mean(valid>0)<.82:return None
            return aligned,M,float(score)
        except cv2.error:return None

    @staticmethod
    def _mask_border(th):
        h,w=th.shape; m=max(10,int(min(h,w)*.02)); out=th.copy(); out[:m]=0; out[-m:]=0; out[:,:m]=0; out[:,-m:]=0; return out

    def _pixel_changes(self,before,after):
        a=self._gray(before); b=self._gray(after)
        # Distance-tolerant difference suppresses double edges caused by tiny registration errors.
        aa=cv2.threshold(a,235,255,cv2.THRESH_BINARY_INV)[1]; bb=cv2.threshold(b,235,255,cv2.THRESH_BINARY_INV)[1]
        da=cv2.dilate(aa,np.ones((5,5),np.uint8)); db=cv2.dilate(bb,np.ones((5,5),np.uint8))
        one=((aa>0)&(db==0)).astype(np.uint8)*255
        two=((bb>0)&(da==0)).astype(np.uint8)*255
        th=cv2.bitwise_or(one,two)
        th=self._mask_border(th); th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)); th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,_,stats,_=cv2.connectedComponentsWithStats(th,8); H,W=th.shape; out=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area>=self.cfg.min_component_area and area<=H*W*self.cfg.max_component_fraction and w>=8 and h>=6: out.append((x,y,w,h,area))
        return out

    @staticmethod
    def _words(path,index):
        try:
            doc=fitz.open(path); p=doc.load_page(index); words=p.get_text('words'); doc.close(); return words
        except Exception:return []

    def compare_match(self,m:Match):
        aligned=self._align(m.before,m.after)
        if aligned is None:return []
        after,M,score=aligned; changes=[]
        # Vector text comparison is supplementary: geometry remains the primary detector.
        old=self._words(m.before.pdf,m.before.index); new=self._words(m.after.pdf,m.after.index)
        for ow in old:
            text=str(ow[4]).strip()
            if not text:continue
            best=None
            ox=(ow[0]+ow[2])/2; oy=(ow[1]+ow[3])/2
            p=np.float32([[[ox,oy]]]); q=cv2.transform(p,M)[0,0]
            for nw in new:
                nx=(nw[0]+nw[2])/2; ny=(nw[1]+nw[3])/2; d=float(np.hypot(q[0]-nx,q[1]-ny))
                if d<20 and (best is None or d<best[0]):best=(d,nw)
            if best and re.sub(r'\s+','',text).upper()!=re.sub(r'\s+','',str(best[1][4])).upper():
                x=int(max(0,min(m.before.image.shape[1]-1,q[0]-35))); y=int(max(0,min(m.before.image.shape[0]-1,q[1]-20)))
                changes.append(Change(m.before.index+1,m.after.index+1,'TEXT',min(0.98,0.70+score*.25),x,y,70,45,m.before.image[max(0,y):y+45,max(0,x):x+70],after[max(0,y):y+45,max(0,x):x+70],text,str(best[1][4])))
        for x,y,w,h,area in self._pixel_changes(m.before.image,after):
            if any(abs(x-c.x)<max(w,c.w)*.7 and abs(y-c.y)<max(h,c.h)*.7 for c in changes):continue
            pad=12; x0=max(0,x-pad); y0=max(0,y-pad); x1=min(after.shape[1],x+w+pad); y1=min(after.shape[0],y+h+pad)
            changes.append(Change(m.before.index+1,m.after.index+1,'GEOMETRY',min(0.95,0.45+score*.4),x0,y0,x1-x0,y1-y0,m.before.image[y0:y1,x0:x1],after[y0:y1,x0:x1]))
        return changes
