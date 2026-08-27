from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List
import re
import cv2
import numpy as np

@dataclass
class SemanticRegion:
    x:int; y:int; w:int; h:int; kind:str; words:List[dict]=field(default_factory=list); score:float=0.0
    @property
    def right(self): return self.x+self.w
    @property
    def bottom(self): return self.y+self.h
    def norm(self,W,H): return (self.x/W,self.y/H,self.w/W,self.h/H)

class SemanticRegionBuilder:
    """Build meaningful regions without fixed page tiling.

    Priority: explicit rectangular/NOTE containers -> GD&T/dimension groups ->
    remaining drawing components. All boxes are xywh in the source image.
    """
    NOTE_RE=re.compile(r'\b(?:NOTE|NOTES|UNLESS|MATERIAL|FINISH|REMOVE|BURR|INSPECT|TYP|SEE)\b',re.I)
    GD_RE=re.compile(r'(?:POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|⌖|⌯|⏥|⌒|∥|⊥)',re.I)
    DIM_RE=re.compile(r'^(?:[RMD]\s*)?(?:Ø|⌀)?\d+(?:\.\d+)?(?:\s*[A-Z°]+)?$')
    def build(self,image:np.ndarray,words:List[dict]) -> List[SemanticRegion]:
        H,W=image.shape[:2]; out=[]
        boxes=self._containers(image)
        used=set()
        for b in boxes:
            inside=self._words_in(words,b,W,H)
            txt=' '.join(w['text'] for w in inside)
            kind='NOTE' if self.NOTE_RE.search(txt) else ('GD&T' if any(self.GD_RE.search(w['text']) for w in inside) else 'BOX')
            out.append(SemanticRegion(*b,kind,inside,.95)); used.update(id(w) for w in inside)
        special=[w for w in words if self._special(w)]
        out.extend(self._group_special(special,W,H,out))
        return self._merge(out,W,H)
    def _containers(self,image):
        g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        e=cv2.Canny(g,40,140); lines=cv2.HoughLinesP(e,1,np.pi/180,threshold=max(25,min(g.shape)//80),minLineLength=max(35,min(g.shape)//30),maxLineGap=8)
        rect=[]
        if lines is None:return rect
        hs=[];vs=[]
        for l in lines[:,0]:
            x1,y1,x2,y2=map(int,l)
            if abs(y2-y1)<=3 and abs(x2-x1)>25: hs.append((min(x1,x2),max(x1,x2),y1))
            if abs(x2-x1)<=3 and abs(y2-y1)>25: vs.append((min(y1,y2),max(y1,y2),x1))
        for x1,x2,y in hs:
            for yy1,yy2,x in vs:
                if x<=x1+8 or x>=x2-8:continue
                if yy1<=y<=yy2:
                    for xx1,xx2,yy in hs:
                        if yy>y+15 and xx1<=x<=xx2 and xx1<=x2 and xx2>=x1:
                            w=x2-x1;h=yy-y
                            if w>40 and h>25 and w*h<.45*g.shape[0]*g.shape[1]:rect.append((x1,y,w,h))
                            break
                    break
        uniq=[]
        for b in sorted(rect,key=lambda z:z[2]*z[3],reverse=True):
            if not any(self._iou(b,q)>.8 for q in uniq):uniq.append(b)
        return uniq[:80]
    def _special(self,w):
        t=str(w.get('text','')).strip(); return bool(self.NOTE_RE.search(t) or self.GD_RE.search(t) or self.DIM_RE.fullmatch(t))
    def _words_in(self,words,b,W,H):
        x,y,w,h=b; return [q for q in words if x<=q['x']*W+q['w']*W/2<=x+w and y<=q['y']*H+q['h']*H/2<=y+h]
    def _group_special(self,words,W,H,existing):
        out=[]; seen=set()
        for w in words:
            if id(w) in seen:continue
            cx=(w['x']+.5*w['w'])*W; cy=(w['y']+.5*w['h'])*H
            group=[w]; seen.add(id(w))
            for q in words:
                if id(q) in seen:continue
                qx=(q['x']+.5*q['w'])*W; qy=(q['y']+.5*q['h'])*H
                if abs(qx-cx)<max(140,.18*W) and abs(qy-cy)<max(90,.08*H): group.append(q);seen.add(id(q))
            x=min(int(q['x']*W) for q in group); y=min(int(q['y']*H) for q in group); xx=max(int((q['x']+q['w'])*W) for q in group); yy=max(int((q['y']+q['h'])*H) for q in group)
            pad=max(10,int(min(xx-x,yy-y)*.5)); b=(max(0,x-pad),max(0,y-pad),min(W,x+pad+xx)-max(0,x-pad),min(H,y+pad+yy)-max(0,y-pad))
            kind='NOTE' if any(self.NOTE_RE.search(q['text']) for q in group) else ('GD&T' if any(self.GD_RE.search(q['text']) for q in group) else 'DIMENSION')
            if not any(self._iou(b,(r.x,r.y,r.w,r.h))>.65 for r in existing+out):out.append(SemanticRegion(*b,kind,group,.85))
        return out
    def _merge(self,regions,W,H):
        out=[]
        for r in sorted(regions,key=lambda z:z.score,reverse=True):
            if r.w<12 or r.h<8:continue
            if not any(self._iou((r.x,r.y,r.w,r.h),(q.x,q.y,q.w,q.h))>.72 and r.kind==q.kind for q in out):out.append(r)
        return out
    @staticmethod
    def _iou(a,b):
        ax,ay,aw,ah=a;bx,by,bw,bh=b; x=max(ax,bx);y=max(ay,by);xx=min(ax+aw,bx+bw);yy=min(ay+ah,by+bh);i=max(0,xx-x)*max(0,yy-y);u=aw*ah+bw*bh-i;return i/max(1,u)
