from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .change_detector import ChangeDetector, ChangeDetectionResult, ChangeRegion


@dataclass(frozen=True)
class PageMatch:
    before_index: int
    after_index: int
    score: float
    method: str = "semantic_region"


@dataclass
class CompareEngineResult:
    success: bool
    page_results: List[ChangeDetectionResult] = field(default_factory=list)
    page_matches: List[PageMatch] = field(default_factory=list)
    reason: str = ""

    @property
    def regions(self) -> List[ChangeRegion]:
        out: List[ChangeRegion] = []
        for r in self.page_results:
            out.extend(r.regions)
        return out


class CompareEngine:
    """Orchestrates H5 comparison without inventing global text matches.

    Page/region correspondence is established before value comparison.  The
    ChangeDetector owns coordinate conversion; this layer never manipulates
    raw x2/y2 coordinates, preventing coordinate-space leakage.
    """

    def __init__(self, detector: Optional[ChangeDetector] = None):
        self.detector = detector or ChangeDetector()

    @staticmethod
    def _image(page: Any) -> np.ndarray:
        if isinstance(page, np.ndarray):
            return page
        if hasattr(page, "image"):
            return np.asarray(page.image)
        raise TypeError("page.image is required")

    @staticmethod
    def _page_signature(page: Any) -> np.ndarray:
        img = CompareEngine._image(page)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return cv2.Canny(img, 30, 100)

    @classmethod
    def _page_score(cls, a: Any, b: Any) -> float:
        aa = cls._page_signature(a)
        bb = cls._page_signature(b)
        corr = float(cv2.matchTemplate(aa, bb, cv2.TM_CCOEFF_NORMED)[0, 0])
        return max(0.0, min(1.0, corr))

    def match_pages(self, before_pages: List[Any], after_pages: List[Any]) -> List[PageMatch]:
        """One-to-one page matching using visual structure, not page number only."""
        candidates = []
        for i, bp in enumerate(before_pages):
            for j, ap in enumerate(after_pages):
                s = self._page_score(bp, ap)
                candidates.append((s, i, j))
        candidates.sort(reverse=True)
        used_b, used_a = set(), set()
        out = []
        for s, i, j in candidates:
            if i in used_b or j in used_a:
                continue
            if s < 0.35:
                continue
            used_b.add(i); used_a.add(j)
            out.append(PageMatch(i, j, s))
        return sorted(out, key=lambda x: x.before_index)

    def compare(self, before_pages: List[Any], after_pages: List[Any]) -> CompareEngineResult:
        try:
            matches = self.match_pages(before_pages, after_pages)
            results: List[ChangeDetectionResult] = []
            for m in matches:
                results.append(self.detector.detect(before_pages[m.before_index], after_pages[m.after_index]))
            ok = bool(matches) and all(r.success for r in results)
            reason = f"pages={len(before_pages)}/{len(after_pages)}, page_matches={len(matches)}, final={sum(len(r.regions) for r in results)}"
            return CompareEngineResult(ok, results, matches, reason)
        except Exception as exc:
            return CompareEngineResult(False, [], [], f"engine_error: {exc}")

    def run(self, before_pages: List[Any], after_pages: List[Any]) -> CompareEngineResult:
        return self.compare(before_pages, after_pages)
