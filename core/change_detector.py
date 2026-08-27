from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re
import cv2
import numpy as np

@dataclass
class ChangeRegion:
    x:int; y:int; width:int; height:int; area:int=0; change_ratio:float=0.0
    region_type:str="dimension_or_note"; confidence:float=0.0
    old_crop:Optional[np.ndarray]=None; new_crop:Optional[np.ndarray]=None; difference_crop:Optional[np.ndarray]=None
    old_text:str=""; new_text:str=""; change_kind:str=""
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
    change_pixel_ratio:float=0.0; reason:str=""
    @property
    def region(self): return self.regions

class ChangeDetector:
    """Text-first drawing comparison.

    Matching order is deliberate:
      1) boxed/table/note blocks -> match as one local region
      2) ordinary notes -> text + local image context
      3) GD&T / dimensions -> type + relative position + multi-scale image context
      4) only then consider add/delete

    Geometry itself is never reported as a change target.
    """
    def __init__(self, config=None):
        self.pixel_threshold=38
        self.match_radius=.12
        self.word_merge_gap=.006
        self.context_size=320

    @staticmethod
    def _img(page):
        if isinstance(page,np.ndarray): return np.asarray(page)
        if hasattr(page,"image"): return np.asarray(page.image)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")
    @staticmethod
    def _gray(img):
        if img.ndim==2:return img.astype(np.uint8)
        if img.shape[2]==4:return cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def _norm(s):
        return re.sub(r"[^A-Z0-9Ø⌀±+\-.*/X°]","",str(s).upper().replace("—","-").replace("–","-").replace("−","-"))
    @staticmethod
    def _class(t):
        t=str(t).strip().upper()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|[⏥⌖⌒∥⊥]",t): return "gdt"
        if re.search(r"(?:^|\s)(?:R|M)\s*\d|[0-9]+(?:\.[0-9]+)?(?:\s*(?:MM|IN|°|DEG))?$",t): return "dimension"
        if re.fullmatch(r"[0-9.]+|[0-9]+/[0-9]+|[A-Z]?[0-9]+",t): return "dimension"
        if any(k in t for k in ("NOTE","TYP","UNLESS","MATERIAL","FINISH","REMOVE","BURR","INSPECT","SEE")): return "note"
        # Normal words/sentences are also notes or callouts; previously these were lost.
        if re.search(r"[A-Z]",t) and len(t)>=2: return "note"
        return "other"

    def _native_words(self,page):
        try:
            import fitz
            doc=fitz.open(Path(page.pdf_path));p=doc.load_page(int(page.page_index));r=p.rect;out=[]
            for z in p.get_text("words"):
                x0,y0,x1,y1,text,*_=z;text=str(text).strip()
                if text: out.append({"text":text,"x":x0/r.width,"y":y0/r.height,"w":(x1-x0)/r.width,"h":(y1-y0)/r.height,"class":self._class(text)})
            doc.close();return out
        except Exception:return []

    def _word(self,z,shape):
        h,w=shape[:2];return {**z,"px":(z["x"]+z["w"]/2)*w,"py":(z["y"]+z["h"]/2)*h,"pw":max(1,z["w"]*w),"ph":max(1,z["h"]*h)}

    def _anchors(self,words,shape):
        """Create independent text anchors. Very close same-baseline fragments
        are joined, but separate dimensions remain separate."""
        q=sorted([self._word(z,shape) for z in words if z["class"]!="other"],key=lambda z:(z["py"],z["px"]))
        out=[];used=set()
        for i,z in enumerate(q):
            if i in used:continue
            parts=[z];used.add(i)
            for j,n in enumerate(q):
                if j in used:continue
                baseline=abs(n["py"]-z["py"])<=max(z["ph"],n["ph"])*.55
                gap=(n["px"]-n["pw"]/2)-(z["px"]+z["pw"]/2)
                if baseline and 0<=gap<=shape[1]*self.word_merge_gap and n["class"]==z["class"]:
                    parts.append(n);used.add(j)
            parts.sort(key=lambda a:a["px"])
            x0=min(a["px"]-a["pw"]/2 for a in parts);y0=min(a["py"]-a["ph"]/2 for a in parts);x1=max(a["px"]+a["pw"]/2 for a in parts);y1=max(a["py"]+a["ph"]/2 for a in parts)
            out.append({"text":" ".join(a["text"] for a in parts),"px":(x0+x1)/2,"py":(y0+y1)/2,"pw":x1-x0,"ph":y1-y0,"class":z["class"],"parts":parts})
        return out

    def _blocks(self,img,anchors):
        """Find rectangular/table-like blocks and assign text anchors to them.
        This gives boxed GD&T/title/note areas a single structural identity."""
        g=self._gray(img);edges=cv2.Canny(g,60,160);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);h,w=g.shape[:2];blocks=[]
        for c in contours:
            x,y,bw,bh=cv2.boundingRect(c);area=bw*bh
            if bw<max(30,w*.015) or bh<max(20,h*.01):continue
            if area>w*h*.30:continue
            peri=cv2.arcLength(c,True);approx=cv2.approxPolyDP(c,.03*peri,True)
            if 4<=len(approx)<=6 and bw/bh<30 and bh/bw<30:
                blocks.append((x,y,bw,bh))
        # Remove nested duplicates; keep the smallest useful enclosing box.
        blocks=sorted(blocks,key=lambda b:b[2]*b[3])
        kept=[]
        for b in blocks:
            if any(b[0]>=k[0] and b[1]>=k[1] and b[0]+b[2]<=k[0]+k[2] and b[1]+b[3]<=k[1]+k[3] for k in kept):continue
            kept.append(b)
        result=[];claimed=set()
        for bi,b in enumerate(kept[:80]):
            x,y,bw,bh=b;inside=[]
            for i,a in enumerate(anchors):
                if x<=a["px"]<=x+bw and y<=a["py"]<=y+bh:inside.append(i);claimed.add(i)
            if inside:
                result.append({"id":bi,"box":b,"indices":inside,"class":"block","text":" ".join(anchors[i]["text"] for i in sorted(inside,key=lambda j:(anchors[j]["py"],anchors[j]["px"]))),"px":x+bw/2,"py":y+bh/2,"pw":bw,"ph":bh})
        return result,claimed

    @staticmethod
    def _crop_norm(img,center,box_size=320):
        h,w=img.shape[:2];cx,cy=center;half=box_size//2;x0=max(0,int(cx-half));y0=max(0,int(cy-half));x1=min(w,int(cx+half));y1=min(h,int(cy+half));crop=img[y0:y1,x0:x1]
        if crop.size==0:return np.zeros((128,128),np.uint8)
        g=ChangeDetector._gray(crop);g=cv2.resize(g,(128,128),interpolation=cv2.INTER_AREA);g=cv2.GaussianBlur(g,(3,3),0);return g

    def _context_similarity(self,before,after,o,n):
        """Scale-normalized local image similarity. The same boxed/leader/arrow
        context remains similar even when page rendering scale differs."""
        c1=self._crop_norm(before,(o["px"],o["py"]),self.context_size)
        best=0.0
        # Multiple crop scales emulate zoom in/out without rotating/warping the drawing.
        for sz in (220,320,450,600):
            a=self._crop_norm(after,(n["px"],n["py"]),sz)
            aa=cv2.normalize(c1,None,0,255,cv2.NORM_MINMAX);bb=cv2.normalize(a,None,0,255,cv2.NORM_MINMAX)
            corr=float(cv2.matchTemplate(aa,bb,cv2.TM_CCOEFF_NORMED)[0,0]) if aa.shape==bb.shape else 0.0
            # Also compare binary edge layout, which is less sensitive to line weight.
            ea=cv2.Canny(aa,60,150);eb=cv2.Canny(bb,60,150);edge=1.0-float(np.mean(cv2.absdiff(ea,eb)))/255.0
            best=max(best,.65*max(0,corr)+.35*edge)
        return best

    def _match_blocks(self,bb,ab):
        used=set();out=[]
        for b in bb:
            best=(-9,None)
            for j,a in enumerate(ab):
                if j in used:continue
                dx=abs(b["px"]/max(1,1)-a["px"])/max(1,1000000) # placeholder, replaced below
                # normalized page coordinates are stored separately by caller via px/pw ratios
                score=0
                same=self._norm(b["text"])==self._norm(a["text"])
                score += 4 if same and b["text"] else 0
                score -= abs(b["rx"]-a["rx"])*4+abs(b["ry"]-a["ry"])*4
                score -= abs(b["rw"]-a["rw"])+abs(b["rh"]-a["rh"])
                if score>best[0]:best=(score,j)
            if best[1] is not None and best[0]>-2:
                used.add(best[1]);out.append((b,ab[best[1]],"block"))
        return out,used

    def _match_anchors(self,b,a,w,h,before,after):
        """One-to-one matching with text identity, type, normalized position and
        scale-normalized local image context. No rotation or geometric distortion."""
        used=set();pairs=[]
        for o in b:
            best=(-99,None)
            for j,n in enumerate(a):
                if j in used:continue
                if n["class"]!=o["class"]:continue
                dx=abs(o["px"]/w-n["px"]/w);dy=abs(o["py"]/h-n["py"]/h)
                if dx>self.match_radius or dy>self.match_radius:continue
                same=self._norm(o["text"])==self._norm(n["text"])
                ctx=self._context_similarity(before,after,o,n)
                score=(4.5 if same else 0)+3.0*ctx-2.0*(dx+dy)-.10*abs(np.log(max(.01,o["pw"])/max(.01,n["pw"])))
                if score>best[0]:best=(score,j)
            if best[1] is None:pairs.append((o,None,"deleted"))
            else:used.add(best[1]);pairs.append((o,a[best[1]],"matched"))
        for j,n in enumerate(a):
            if j not in used:pairs.append((None,n,"added"))
        return pairs

    def _pair_add_delete(self,pairs,w,h):
        dels=[i for i,p in enumerate(pairs) if p[2]=="deleted"];adds=[i for i,p in enumerate(pairs) if p[2]=="added"];ud=set();ua=set();result=[]
        for di in dels:
            o=pairs[di][0];best=None
            for ai in adds:
                if ai in ua:continue
                n=pairs[ai][1]
                if n["class"]!=o["class"]:continue
                dx=abs(o["px"]/w-n["px"]/w);dy=abs(o["py"]/h-n["py"]/h)
                if dx<=.06 and dy<=.06:
                    s=dx+dy
                    if best is None or s<best[0]:best=(s,ai)
            if best:ud.add(di);ua.add(best[1]);result.append((o,pairs[best[1]][1],"changed_value"))
        for i,p in enumerate(pairs):
            if i not in ud and i not in ua:result.append(p)
        return result

    def _box(self,q,w,h):
        pad=max(45,int(max(q["pw"],q["ph"])*3.0));return(max(0,int(q["px"]-q["pw"]/2-pad)),max(0,int(q["py"]-q["ph"]/2-pad)),min(w,int(q["px"]+q["pw"]/2+pad)),min(h,int(q["py"]+q["ph"]/2+pad)))
    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]);y=max(a[1],b[1]);xx=min(a[0]+a[2],b[0]+b[2]);yy=min(a[1]+a[3],b[1]+b[3]);inter=max(0,xx-x)*max(0,yy-y);u=a[2]*a[3]+b[2]*b[3]-inter;return inter/max(1,u)

    def _deduplicate(self,regions):
        kept=[]
        for r in sorted(regions,key=lambda z:(-z.confidence,z.y,z.x)):
            dup=None
            for k in kept:
                same=self._norm(r.old_text)==self._norm(k.old_text) and self._norm(r.new_text)==self._norm(k.new_text)
                iou=self._iou((r.x,r.y,r.width,r.height),(k.x,k.y,k.width,k.height))
                cx=abs((r.x+r.width/2)-(k.x+k.width/2));cy=abs((r.y+r.height/2)-(k.y+k.height/2))
                if iou>.22 or (same and cx<max(r.width,k.width)*.45 and cy<max(r.height,k.height)*.45):dup=k;break
            if dup is None:kept.append(r)
            elif r.confidence>dup.confidence:kept[kept.index(dup)]=r
        return kept

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page);after=self._img(aligned_after) if aligned_after is not None else self._img(after_page);h,w=before.shape[:2]
            if after.shape[:2]!=(h,w):after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after));_,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)
            bw=self._native_words(before_page);aw=self._native_words(after_page);bc=self._anchors(bw,before.shape);ac=self._anchors(aw,after.shape)
            bb,bc_claimed=self._blocks(before,bc);ab,ac_claimed=self._blocks(after,ac)
            # Add normalized coordinates to block descriptors for cross-page comparison.
            for b in bb:b.update(rx=b["px"]/w,ry=b["py"]/h,rw=b["pw"]/w,rh=b["ph"]/h)
            for a in ab:a.update(rx=a["px"]/w,ry=a["py"]/h,rw=a["pw"]/w,rh=a["ph"]/h)
            block_pairs,_=self._match_blocks(bb,ab)
            paired_b=set();paired_a=set();pairs=[]
            for b,a,k in block_pairs:
                paired_b.update(b["indices"]);paired_a.update(a["indices"])
                # A block is one change candidate, not many independent text candidates.
                old=b["text"];new=a["text"]
                if self._norm(old)==self._norm(new):continue
                pairs.append((b,a,"block_changed"))
            # Remaining individual anchors: dimensions/GD&T and unboxed notes.
            rem_b=[x for i,x in enumerate(bc) if i not in paired_b];rem_a=[x for i,x in enumerate(ac) if i not in paired_a]
            pairs.extend(self._match_anchors(rem_b,rem_a,w,h,before,after))
            pairs=self._pair_add_delete(pairs,w,h)
            candidates=[];added=deleted=changed=blocks_changed=0
            for o,n,kind in pairs:
                q=o or n;box=self._box(q,w,h);local=diff[box[1]:box[3],box[0]:box[2]];ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0;old=o["text"] if o else "";new=n["text"] if n else ""
                if kind=="matched" and self._norm(old)==self._norm(new):continue
                if kind=="matched" and ratio<.00015:continue
                if kind=="added":added+=1
                elif kind=="deleted":deleted+=1
                elif kind=="changed_value":changed+=1
                elif kind=="block_changed":blocks_changed+=1
                # Require actual text/value change for text targets; block candidates also carry text change.
                if self._norm(old)==self._norm(new):continue
                score=5.0 if kind in ("changed_value","block_changed") else 3.0
                if o and n:score+=3.0
                if ratio>.0005:score+=min(2.5,ratio*160)
                rtype="boxed_region" if kind=="block_changed" else ("dimension_change" if kind=="changed_value" or (o and o["class"]=="dimension") else ("gdt_change" if o and o["class"]=="gdt" else "note_change"))
                candidates.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,rtype,min(1,score/12),before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,kind))
            merged=self._deduplicate(candidates)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"mapping=blocks_first, blocks={len(bb)}/{len(ab)}, block_pairs={len(block_pairs)}, anchors={len(bc)}/{len(ac)}, "
                    f"pairs={len(pairs)}, candidates={len(candidates)}, block_changed={blocks_changed}, changed_values={changed}, added={added}, deleted={deleted}, final={len(merged)}")
            return ChangeDetectionResult(True,merged,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
