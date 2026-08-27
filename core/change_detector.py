from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import re
import cv2
import numpy as np
from config import CONFIG


@dataclass
class ChangeRegion:
    x: int
    y: int
    width: int
    height: int
    area: int = 0
    change_ratio: float = 0.0
    region_type: str = "unknown"
    confidence: float = 0.0
    old_crop: Optional[np.ndarray] = None
    new_crop: Optional[np.ndarray] = None
    difference_crop: Optional[np.ndarray] = None

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
    """Drawing-change detector optimized for dimensions and note text.

    Shape/geometry changes are deliberately conservative in H5. Text-like
    regions are detected first, because dimension values, tolerances and
    GD&T/engineering notes are the current priority.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.pixel_threshold = self._cfg("change.pixel_threshold", 45)
        self.minimum_area = self._cfg("change.minimum_area", 80)
        self.merge_distance = self._cfg("change.merge_distance", 25)
        self.max_region_ratio = self._cfg("change.max_region_ratio", 0.15)
        self.ocr_enabled = True
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except Exception:
            self.pytesseract = None
            self.ocr_enabled = False

    def _cfg(self, path, default):
        cur = self.config
        for key in path.split("."):
            try: cur = cur[key] if isinstance(cur, dict) else getattr(cur, key)
            except Exception: return default
        return cur

    def _image(self, page):
        if isinstance(page, np.ndarray): return page
        if hasattr(page, "image"): return np.asarray(page.image)
        if hasattr(page, "array"): return np.asarray(page.array)
        raise TypeError("페이지 이미지 배열을 찾을 수 없습니다.")

    def _gray(self, image):
        if image.ndim == 2: return image.astype(np.uint8)
        if image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _same_size(self, a, b):
        h, w = a.shape[:2]
        if b.shape[:2] == (h, w): return a, b
        return a, cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)

    def _diff_mask(self, a, b):
        ga, gb = self._gray(a), self._gray(b)
        diff = cv2.absdiff(ga, gb)
        # Remove tiny rasterization noise while retaining thin text strokes.
        _, mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        k = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return diff, mask

    def _ocr(self, gray):
        if not self.ocr_enabled: return []
        try:
            data = self.pytesseract.image_to_data(
                gray,
                config="--psm 11",
                output_type=self.pytesseract.Output.DICT,
            )
            out = []
            n = len(data.get("text", []))
            for i in range(n):
                text = (data["text"][i] or "").strip()
                try: conf = float(data["conf"][i])
                except Exception: conf = -1
                if not text or conf < 35: continue
                x,y,w,h = [int(data[k][i]) for k in ("left","top","width","height")]
                if w < 3 or h < 3: continue
                out.append((x,y,w,h,text,conf))
            return out
        except Exception:
            return []

    @staticmethod
    def _norm_text(s):
        s = s.upper().replace("O", "0") if re.fullmatch(r"[A-Z0-9 .+\-=/()Ø°±×X*%µΜ≤≥]+", s.upper()) else s.upper()
        return re.sub(r"\s+", "", s)

    def _text_changes(self, before, after, diff):
        gb, ga = self._gray(before), self._gray(after)
        old = self._ocr(gb)
        new = self._ocr(ga)
        regions = []
        used = set()
        for ox,oy,ow,oh,ot,oc in old:
            best = None
            best_score = 0
            ocx, ocy = ox+ow/2, oy+oh/2
            for j,(nx,ny,nw,nh,nt,nc) in enumerate(new):
                if j in used: continue
                ncx,ncy = nx+nw/2, ny+nh/2
                dist = ((ocx-ncx)**2+(ocy-ncy)**2)**0.5
                scale = max(ow,oh,nw,nh,20)
                spatial = max(0.0, 1.0-dist/(scale*4.0))
                if spatial <= 0: continue
                text_score = 1.0 if self._norm_text(ot)==self._norm_text(nt) else 0.0
                score = 0.65*spatial + 0.35*text_score
                if score > best_score:
                    best_score,best = score,j
            if best is not None:
                nx,ny,nw,nh,nt,nc = new[best]
                used.add(best)
                changed_text = self._norm_text(ot) != self._norm_text(nt)
                # Require visible pixel difference around the OCR boxes when
                # OCR text is unchanged, preventing OCR jitter from becoming a change.
                x1=max(0,min(ox,nx)-8); y1=max(0,min(oy,ny)-8)
                x2=min(diff.shape[1],max(ox+ow,nx+nw)+8); y2=min(diff.shape[0],max(oy+oh,ny+nh)+8)
                local = diff[y1:y2,x1:x2]
                visible = float(np.mean(local > self.pixel_threshold)) if local.size else 0
                if not changed_text and visible < 0.08: continue
                if not changed_text and best_score < 0.88: continue
                rx,ry,rw,rh=x1,y1,x2-x1,y2-y1
                regions.append((rx,ry,rw,rh,"dimension_or_note",min(0.99,0.70+0.25*best_score)))
            else:
                regions.append((max(0,ox-8),max(0,oy-8),ow+16,oh+16,"dimension_or_note",0.78))
        # New OCR items without a nearby old item are additions.
        for j,(nx,ny,nw,nh,nt,nc) in enumerate(new):
            if j in used: continue
            regions.append((max(0,nx-8),max(0,ny-8),nw+16,nh+16,"dimension_or_note",0.80))
        return regions

    def _pixel_regions(self, mask, before, after, diff):
        # Conservative fallback: only keep compact text-like components.
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        out=[]
        image_area=mask.shape[0]*mask.shape[1]
        for c in contours:
            area=cv2.contourArea(c)
            if area < self.minimum_area: continue
            x,y,w,h=cv2.boundingRect(c)
            ra=w*h
            if ra/image_area > self.max_region_ratio: continue
            # Very large filled/structural regions are intentionally ignored.
            if w>0.35*mask.shape[1] or h>0.20*mask.shape[0]: continue
            aspect=w/max(h,1)
            if aspect < 0.15 or aspect > 20: continue
            out.append((x,y,w,h,"text_or_detail",0.55))
        return out

    def _merge(self, items, before, after, diff):
        regs=[]
        for x,y,w,h,typ,conf in items:
            merged=False
            for r in regs:
                gap=max(r.left-x, x+w-r.right, r.top-y, y+h-r.bottom, 0)
                if gap <= self.merge_distance:
                    l=min(r.x,x); t=min(r.y,y); rr=max(r.right,x+w); bb=max(r.bottom,y+h)
                    r.x,r.y,r.width,r.height=l,t,rr-l,bb-t
                    r.confidence=max(r.confidence,conf)
                    merged=True; break
            if not merged:
                regs.append(ChangeRegion(x,y,w,h,area=w*h,region_type=typ,confidence=conf))
        result=[]
        H,W=before.shape[:2]
        for r in regs:
            r.x=max(0,min(r.x,W-1)); r.y=max(0,min(r.y,H-1))
            r.width=min(r.width,W-r.x); r.height=min(r.height,H-r.y)
            if r.width<6 or r.height<6: continue
            r.old_crop=before[r.y:r.bottom,r.x:r.right].copy()
            r.new_crop=after[r.y:r.bottom,r.x:r.right].copy()
            r.difference_crop=diff[r.y:r.bottom,r.x:r.right].copy()
            changed=float(np.mean(r.difference_crop > self.pixel_threshold)) if r.difference_crop.size else 0
            r.change_ratio=changed
            if changed < 0.025: continue
            result.append(r)
        return result

    def detect(self, before_page, after_page):
        try:
            before=self._image(before_page)
            after=self._image(after_page)
            before,after=self._same_size(before,after)
            diff,mask=self._diff_mask(before,after)
            items=[]
            # OCR is the primary detector. This is deliberate: H5 currently
            # prioritizes dimensions, tolerances and engineering/GD&T notes.
            if self.ocr_enabled:
                items.extend(self._text_changes(before,after,diff))
            # Only use pixel fallback when OCR found nothing, and keep it very conservative.
            if not items:
                items=self._pixel_regions(mask,before,after,diff)
            regions=self._merge(items,before,after,diff)
            ratio=float(np.mean(mask>0)) if mask.size else 0.0
            return ChangeDetectionResult(True,regions,diff,mask,ratio,"text-priority detection")
        except Exception as exc:
            return ChangeDetectionResult(False,[],reason=str(exc))
