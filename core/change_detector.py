from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
import re,cv2,numpy as np
@dataclass
class ChangeRegion:
 x:int;y:int;width:int;height:int;area:int=0;change_ratio:float=0.;region_type:str="dimension_or_note";confidence:float=0.;old_crop:object=None;new_crop:object=None;difference_crop:object=None
 @property
 def right(self):return self.x+self.width
 @property
 def bottom(self):return self.y+self.height
@dataclass
class ChangeDetectionResult:
 success:bool;regions:list=field(default_factory=list);difference_image:object=None;threshold_image:object=None;change_pixel_ratio:float=0.;reason:str=""
 @property
 def region(self):return self.regions
class ChangeDetector:
 def __init__(self,config=None):self.pixel_threshold=38
 @staticmethod
 def _img(p):return np.asarray(p if isinstance(p,np.ndarray) else p.image)
 @staticmethod
 def _gray(a):
  if a.ndim==2:return a.astype(np.uint8)
  if a.shape[2]==4:return cv2.cvtColor(a,cv2.COLOR_RGBA2GRAY)
  return cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
 @staticmethod
 def _norm(s):return re.sub(r'\s+','',str(s).upper().replace('—','-').replace('–','-').replace('−','-'))
 @staticmethod
 def _target(s):return bool(re.search(r'\d|Ø|⌀|%%C|±|\+/-|\+\-|NOTE|TYP|UNLESS|MATERIAL|FINISH|BURR|INSPECT|SEE',str(s),re.I))
 def _words(self,page):
  try:
   import fitz; d=fitz.open(Path(page.pdf_path));p=d.load_page(page.page_index);r=p.rect;w=p.get_text('words');d.close()
   return [{'text':str(a[4]).strip(),'x':a[0]/r.width,'y':a[1]/r.height,'w':(a[2]-a[0])/r.width,'h':(a[3]-a[1])/r.height} for a in w if str(a[4]).strip()]
  except Exception:return []
 def _map(self,before,after):
  a=self._gray(before);b=self._gray(after);h,w=a.shape
  try:
   orb=cv2.ORB_create(nfeatures=5000,fastThreshold=10);k1,d1=orb.detectAndCompute(a,None);k2,d2=orb.detectAndCompute(b,None)
   if d1 is None or d2 is None:return None
   ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2);good=[m[0] for m in ms if len(m)==2 and m[0].distance<.72*m[1].distance]
   if len(good)<8:return None
   src=np.float32([k2[m.queryIdx].pt for m in good]);dst=np.float32([k1[m.trainIdx].pt for m in good]);M,ins=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=3000,confidence=.995)
   if M is None or ins is None or ins.sum()<8 or ins.sum()/len(good)<.25:return None
   det=np.linalg.det(M[:,:2]);s=np.sqrt(abs(det));rot=np.degrees(np.arctan2(M[1,0]-M[0,1],M[0,0]+M[1,1]))
   if det<=0 or not .6<=s<=1.7 or abs(rot)>15:return None
   return M,float(ins.sum()/len(good)),float(s),float(rot)
  except Exception:return None
 @staticmethod
 def _xy(x,M,w,h):
  p=M[:,:2]@np.array([(x['x']+x['w']/2)*w,(x['y']+x['h']/2)*h])+M[:,2];sx=np.linalg.norm(M[:,0]);sy=np.linalg.norm(M[:,1]);return {**x,'px':p[0],'py':p[1],'pw':x['w']*w*sx,'ph':x['h']*h*sy}
 def _native(self,bp,ap,before,after,diff):
  old=[x for x in self._words(bp) if self._target(x['text'])];new=[x for x in self._words(ap) if self._target(x['text'])];h,w=before.shape[:2];M=self._map(before,self._img(ap));
  for x in old:x.update(px=(x['x']+x['w']/2)*w,py=(x['y']+x['h']/2)*h,pw=x['w']*w,ph=x['h']*h)
  if M:new=[self._xy(x,M[0],w,h) for x in new]
  else:
   for x in new:x.update(px=(x['x']+x['w']/2)*w,py=(x['y']+x['h']/2)*h,pw=x['w']*w,ph=x['h']*h)
  used=set();out=[]
  for o in old:
   choices=[]
   for j,n in enumerate(new):
    if j in used:continue
    dx=abs(o['px']-n['px'])/w;dy=abs(o['py']-n['py'])/h;sz=abs(np.log(max(o['pw'],1)/max(n['pw'],1)))
    if dx>.012 or dy>.010 or sz>.55:continue
    # Require the same local drawing neighborhood, not merely similar page position.
    r=max(25,int(2.5*max(o['pw'],o['ph'],n['pw'],n['ph'])));x1=max(0,int(o['px']-r));y1=max(0,int(o['py']-r));x2=min(w,int(o['px']+r));y2=min(h,int(o['py']+r));
    r2=max(25,int(2.5*max(n['pw'],n['ph'])));u1=max(0,int(n['px']-r2));v1=max(0,int(n['py']-r2));u2=min(w,int(n['px']+r2));v2=min(h,int(n['py']+r2));
    a=cv2.resize(self._gray(before[y1:y2,x1:x2]),(80,80));b=cv2.resize(self._gray(after[v1:v2,u1:u2]),(80,80));sim=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0])
    if sim<.20:continue
    score=dx+dy+.12*sz-.008*sim;choices.append((score,j))
   if not choices:continue
   _,j=min(choices);n=new[j];used.add(j)
   if self._norm(o['text'])==self._norm(n['text']):continue
   x1=max(0,int(min(o['px']-o['pw']/2,n['px']-n['pw']/2)-12));y1=max(0,int(min(o['py']-o['ph']/2,n['py']-n['ph']/2)-12));x2=min(w,int(max(o['px']+o['pw']/2,n['px']+n['pw']/2)+12));y2=min(h,int(max(o['py']+o['ph']/2,n['py']+n['ph']/2)+12));local=diff[y1:y2,x1:x2]
   if local.size and np.mean(local>self.pixel_threshold)>=.001:out.append((x1,y1,x2-x1,y2-y1,.97))
  return out,len(self._words(bp)),len(self._words(ap)),len(old),len(new),M
 def detect(self,before_page,after_page,aligned_after=None):
  try:
   before=self._img(before_page);after=self._img(after_page) if aligned_after is None else self._img(aligned_after);h,w=before.shape[:2]
   if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
   gb=self._gray(before);ga=self._gray(after);diff=cv2.absdiff(gb,ga);_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
   cand,nb,na,nt,at,M=self._native(before_page,after_page,before,after,diff);regions=[]
   for x,y,rw,rh,c in cand:
    d=diff[y:y+rh,x:x+rw];regions.append(ChangeRegion(x,y,rw,rh,rw*rh,float(np.mean(d>self.pixel_threshold)),"dimension_or_note",c,before[y:y+rh,x:x+rw].copy(),after[y:y+rh,x:x+rw].copy(),d.copy()))
   # Do not merge nearby but distinct dimension changes.
   reason=f"diag: native={nb}/{na}, native_target={nt}/{at}, ocr=0/0, text_mapping={'ok' if M else 'none'}, raw_diff={np.mean(mask>0):.5f}, native_candidates={len(cand)}, image_fallback=0, final={len(regions)}"
   return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
  except Exception as e:return ChangeDetectionResult(False,[],reason=f'diag_error: {e}')
