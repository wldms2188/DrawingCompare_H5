from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import re
import cv2
import numpy as np


@dataclass
class ChangeRegion:
    x: int; y: int; width: int; height: int
    area: int = 0; change_ratio: float = 0.0
    region_type: str = "dimension_or_note"; confidence: float = 0.0
    old_crop: Optional[np.ndarray] = None; new_crop: Optional[np.ndarray] = None
    difference_crop: Optional[np.ndarray] = None
    old_text: str = ""; new_text: str = ""; change_kind: str = ""
    @property
    def left(self): return self.x
    @property
    def top(self): return self.y
    @property
    def right(self): return self.x + self.width
    @property
    def bottom(self): return self.y + self.height


@dataclass
class ChangeDetectionResult:
    success: bool
    regions: List[ChangeRegion] = field(default_factory=list)
    difference_image: Optional[np.ndarray] = None
    threshold_image: Optional[np.ndarray] = None
    change_pixel_ratio: float = 0.0
    reason: str = ""
    @property
    def region(self): return self.regions


class ChangeDetector:
    """H5 text/measurement comparison.

    IMPORTANT: page coordinates are NOT assumed to correspond.
    Each useful Before anchor is searched against the whole After page at
    several scales. Matching is based on local visual context first, then
    text/type/size. Geometry-only differences are not reported.
    """
    def __init__(self, config=None):
        self.pixel_threshold = 38
        self.min_similarity = 0.48
        self.anchor_search_size = 260
        self.scales = (0.65, 0.75, 0.85, 1.0, 1.15, 1.30, 1.45)

    @staticmethod
    def _img(page):
        if isinstance(page, np.ndarray): return np.asarray(page)
        if hasattr(page, "image"): return np.asarray(page.image)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")

    @staticmethod
    def _gray(img):
        if img.ndim == 2: return img.astype(np.uint8)
        if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _norm(s):
        return re.sub(r"[^A-Z0-9Ø⌀±+\-.*/X°]", "", str(s).upper().replace("—", "-").replace("–", "-").replace("−", "-"))

    @staticmethod
    def _class(t):
        t = str(t).strip().upper()
        if re.search(r"POSITION|PROFILE|FLATNESS|PARALLEL|PERPENDICULAR|CONCENTRIC|RUNOUT|DATUM|MMC|LMC|⌀|Ø|±|[⏥⌖⌒∥⊥]", t): return "gdt"
        if re.search(r"(?:^|\s)(?:R|M)\s*\d|[0-9]+(?:\.[0-9]+)?(?:\s*(?:MM|IN|°|DEG))?$", t): return "dimension"
        if re.fullmatch(r"[0-9.]+|[0-9]+/[0-9]+|[A-Z]?[0-9]+", t): return "dimension"
        if any(k in t for k in ("NOTE", "TYP", "UNLESS", "MATERIAL", "FINISH", "REMOVE", "BURR", "INSPECT", "SEE")): return "note"
        if re.search(r"[A-Z]", t) and len(t) >= 2: return "note"
        return "other"

    def _native_words(self, page):
        try:
            import fitz
            doc = fitz.open(Path(page.pdf_path))
            p = doc.load_page(int(page.page_index)); r = p.rect
            out = []
            for z in p.get_text("words"):
                x0, y0, x1, y1, text, *_ = z
                text = str(text).strip()
                if text:
                    out.append({"text": text, "x": x0/r.width, "y": y0/r.height,
                                "w": (x1-x0)/r.width, "h": (y1-y0)/r.height,
                                "class": self._class(text)})
            doc.close(); return out
        except Exception:
            return []

    @staticmethod
    def _to_px(z, shape):
        h, w = shape[:2]
        return {**z, "px": (z["x"]+z["w"]/2)*w, "py": (z["y"]+z["h"]/2)*h,
                "pw": max(1, z["w"]*w), "ph": max(1, z["h"]*h)}

    def _anchors(self, words, shape):
        # Keep every dimension/GD&T/note independently. Only split PDF fragments
        # that touch on the same baseline are joined.
        q = sorted([self._to_px(z, shape) for z in words if z["class"] != "other"], key=lambda z:(z["py"], z["px"]))
        out=[]; used=set()
        for i,z in enumerate(q):
            if i in used: continue
            parts=[z]; used.add(i)
            for j,n in enumerate(q):
                if j in used or n["class"] != z["class"]: continue
                baseline=abs(n["py"]-z["py"]) <= max(z["ph"], n["ph"])*.55
                gap=(n["px"]-n["pw"]/2)-(z["px"]+z["pw"]/2)
                if baseline and 0 <= gap <= shape[1]*.006:
                    parts.append(n); used.add(j)
            parts.sort(key=lambda a:a["px"])
            x0=min(a["px"]-a["pw"]/2 for a in parts); y0=min(a["py"]-a["ph"]/2 for a in parts)
            x1=max(a["px"]+a["pw"]/2 for a in parts); y1=max(a["py"]+a["ph"]/2 for a in parts)
            out.append({"text":" ".join(a["text"] for a in parts), "px":(x0+x1)/2, "py":(y0+y1)/2,
                        "pw":x1-x0, "ph":y1-y0, "class":z["class"], "parts":parts})
        return out

    @staticmethod
    def _rect_from_anchor(a, shape, scale=1.0):
        h,w=shape[:2]; size=int(max(140, min(520, 260*scale)))
        half=size//2; x=int(a["px"]); y=int(a["py"])
        return max(0,x-half), max(0,y-half), min(w,x+half), min(h,y+half)

    @staticmethod
    def _prepare(crop, size=128):
        if crop.size == 0: return np.zeros((size,size), np.uint8)
        g=ChangeDetector._gray(crop)
        g=cv2.resize(g,(size,size),interpolation=cv2.INTER_AREA)
        # Edge image makes line/leader/box context robust to DPI and line weight.
        return cv2.Canny(g,45,140)

    def _template_score(self, template, image, x, y, tw, th):
        patch=image[y:y+th, x:x+tw]
        if patch.shape != template.shape: return 0.0
        a=self._prepare(template); b=self._prepare(patch)
        corr=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0])
        edge=1.0-float(np.mean(cv2.absdiff(a,b)))/255.0
        return .65*max(0,corr)+.35*edge

    def _global_visual_match(self, before, after, anchor):
        """Search the ENTIRE After page, not the same x/y location.
        The returned position is the best local structural match."""
        bg=self._gray(before); ag=self._gray(after); h,w=bg.shape
        x0,y0,x1,y1=self._rect_from_anchor(anchor,bg.shape,1.0)
        template=bg[y0:y1,x0:x1]
        best=(0.0,None)
        for scale in self.scales:
            tw=max(80,int(template.shape[1]*scale)); th=max(80,int(template.shape[0]*scale))
            if tw>=ag.shape[1] or th>=ag.shape[0]: continue
            t=cv2.resize(template,(tw,th),interpolation=cv2.INTER_AREA)
            te=self._prepare(t)
            ae=self._prepare(ag)
            # Downsample search for speed, then use the exact best location.
            result=cv2.matchTemplate(ae,te,cv2.TM_CCOEFF_NORMED)
            _,mx,_,loc=cv2.minMaxLoc(result)
            if mx>best[0]: best=(float(mx),(loc[0]+tw//2,loc[1]+th//2,tw,th,scale))
        return best

    def _text_similarity(self,o,n):
        a=self._norm(o["text"]); b=self._norm(n["text"])
        if not a or not b:return 0.0
        if a==b:return 1.0
        # Character overlap helps 25.0 -> 26.0 and similar GD&T values.
        sa=set(a); sb=set(b); return len(sa&sb)/max(1,len(sa|sb))

    def _match_anchors_global(self, before, after, ba, aa):
        """Global visual candidates + one-to-one assignment.
        A candidate must have either strong visual context or strong text identity.
        This prevents unrelated regions from being paired merely because their
        normalized page coordinates happen to be close."""
        candidates=[]
        for i,o in enumerate(ba):
            score,loc=self._global_visual_match(before,after,o)
            if loc is None or score < self.min_similarity: continue
            cx,cy,tw,th,scale=loc
            # Select the actual After text anchor nearest the visually matched point.
            best=None
            for j,n in enumerate(aa):
                d=((n["px"]-cx)**2+(n["py"]-cy)**2)**0.5
                lim=max(100,0.75*max(tw,th))
                if d>lim or n["class"]!=o["class"]: continue
                ts=self._text_similarity(o,n)
                total=.70*score+.20*ts+.10*max(0,1-d/lim)
                if best is None or total>best[0]: best=(total,j,score)
            if best is not None:
                candidates.append((best[0],i,best[1],best[2]))
        candidates.sort(reverse=True)
        used_b=set(); used_a=set(); pairs=[]
        for total,i,j,visual in candidates:
            if i in used_b or j in used_a: continue
            used_b.add(i); used_a.add(j); pairs.append((ba[i],aa[j],"matched",total,visual))
        for i,o in enumerate(ba):
            if i not in used_b:pairs.append((o,None,"deleted",0,0))
        for j,n in enumerate(aa):
            if j not in used_a:pairs.append((None,n,"added",0,0))
        return pairs

    def _pair_add_delete(self,pairs,w,h):
        # Convert nearby same-type add/delete pairs into one value change.
        dels=[i for i,p in enumerate(pairs) if p[2]=="deleted"]
        adds=[i for i,p in enumerate(pairs) if p[2]=="added"]
        ud=set();ua=set();out=[]
        for di in dels:
            o=pairs[di][0]; best=None
            for ai in adds:
                if ai in ua: continue
                n=pairs[ai][1]
                if n["class"]!=o["class"]: continue
                dx=abs(o["px"]/w-n["px"]/w); dy=abs(o["py"]/h-n["py"]/h)
                if dx<=.08 and dy<=.08:
                    s=dx+dy
                    if best is None or s<best[0]: best=(s,ai)
            if best:
                ud.add(di); ua.add(best[1]); out.append((o,pairs[best[1]][1],"changed_value",0,0))
        for i,p in enumerate(pairs):
            if i not in ud and i not in ua: out.append(p)
        return out

    def _box(self,q,w,h):
        # Crop centered on the matched text/region. Keep enough surrounding
        # geometry to make the Excel result useful, but avoid huge unrelated areas.
        pad=max(55,int(max(q["pw"],q["ph"])*4))
        return (max(0,int(q["px"]-q["pw"]/2-pad)), max(0,int(q["py"]-q["ph"]/2-pad)),
                min(w,int(q["px"]+q["pw"]/2+pad)), min(h,int(q["py"]+q["ph"]/2+pad)))

    @staticmethod
    def _iou(a,b):
        x=max(a[0],b[0]); y=max(a[1],b[1]); xx=min(a[0]+a[2],b[0]+b[2]); yy=min(a[1]+a[3],b[1]+b[3])
        inter=max(0,xx-x)*max(0,yy-y); u=a[2]*a[3]+b[2]*b[3]-inter
        return inter/max(1,u)

    def _deduplicate(self,regions):
        kept=[]
        for r in sorted(regions,key=lambda z:(-z.confidence,z.y,z.x)):
            dup=None
            for k in kept:
                same=self._norm(r.old_text)==self._norm(k.old_text) and self._norm(r.new_text)==self._norm(k.new_text)
                iou=self._iou((r.x,r.y,r.width,r.height),(k.x,k.y,k.width,k.height))
                if iou>.22 or (same and abs(r.x-k.x)<max(r.width,k.width)*.5 and abs(r.y-k.y)<max(r.height,k.height)*.5):
                    dup=k; break
            if dup is None: kept.append(r)
            elif r.confidence>dup.confidence: kept[kept.index(dup)]=r
        return kept

    def detect(self,before_page,after_page,aligned_after=None):
        try:
            before=self._img(before_page)
            after=self._img(aligned_after) if aligned_after is not None else self._img(after_page)
            h,w=before.shape[:2]
            if after.shape[:2]!=(h,w): after=cv2.resize(after,(w,h),interpolation=cv2.INTER_AREA)
            diff=cv2.absdiff(self._gray(before),self._gray(after))
            _,mask=cv2.threshold(diff,self.pixel_threshold,255,cv2.THRESH_BINARY)

            bw=self._native_words(before_page); aw=self._native_words(after_page)
            ba=self._anchors(bw,before.shape); aa=self._anchors(aw,after.shape)
            pairs=self._match_anchors_global(before,after,ba,aa)
            pairs=self._pair_add_delete(pairs,w,h)

            regions=[]; changed=added=deleted=0; rejected=0
            for o,n,kind,match_score,visual_score in pairs:
                q=o or n
                if q is None: continue
                old=o["text"] if o else ""; new=n["text"] if n else ""
                if kind=="matched" and self._norm(old)==self._norm(new): continue
                if kind=="matched" and visual_score<self.min_similarity and self._norm(old)!=self._norm(new):
                    rejected+=1; continue
                box=self._box(q,w,h)
                local=diff[box[1]:box[3],box[0]:box[2]]
                ratio=float(np.mean(local>self.pixel_threshold)) if local.size else 0.0
                if kind=="changed_value": changed+=1
                elif kind=="added": added+=1
                elif kind=="deleted": deleted+=1
                # Do not report additions/deletions unless they are text targets.
                if q["class"]=="other": continue
                score=0.0
                if kind=="changed_value": score+=5
                if kind=="matched": score+=3
                if o and n: score+=3
                score+=2*min(1.0,visual_score)
                score+=min(2.0,ratio*160)
                if kind in ("added","deleted") and match_score==0 and ratio<.0005:
                    # Weak isolated native-text candidates are noise.
                    rejected+=1; continue
                if kind=="changed_value": rtype="dimension_change" if q["class"]=="dimension" else ("gdt_change" if q["class"]=="gdt" else "note_change")
                elif q["class"]=="dimension": rtype="dimension_change"
                elif q["class"]=="gdt": rtype="gdt_change"
                else: rtype="note_change"
                regions.append(ChangeRegion(box[0],box[1],box[2]-box[0],box[3]-box[1],(box[2]-box[0])*(box[3]-box[1]),ratio,rtype,min(1,score/12),before[box[1]:box[3],box[0]:box[2]].copy(),after[box[1]:box[3],box[0]:box[2]].copy(),local.copy(),old,new,kind))

            regions=self._deduplicate(regions)
            reason=(f"diag: native={len(bw)}/{len(aw)}, native_target={sum(x['class']!='other' for x in bw)}/{sum(x['class']!='other' for x in aw)}, "
                    f"mapping=global_visual_search, anchors={len(ba)}/{len(aa)}, pairs={len(pairs)}, "
                    f"changed_values={changed}, added={added}, deleted={deleted}, rejected={rejected}, final={len(regions)}")
            return ChangeDetectionResult(True,regions,diff,mask,float(np.mean(mask>0)),reason)
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=f"diag_error: {exc}")
