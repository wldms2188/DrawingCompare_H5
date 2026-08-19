from __future__ import annotations
 
from typing import List
 
 
class ChangeRegionMerger:
 
    def __init__(
        self,
        gap: int = 35,
        min_width: int = 5,
        min_height: int = 5,
    ):
 
        self.gap = gap
        self.min_width = min_width
        self.min_height = min_height
 
    # ========================================================
    # PUBLIC
    # ========================================================
 
    def merge(
        self,
        regions: List,
    ) -> List:
 
        if not regions:
            return []
 
        boxes = []
 
        for region in regions:
 
            box = self._get_box(
                region
            )
 
            if box is None:
                continue
 
            x, y, w, h = box
 
            if (
                w < self.min_width
                or
                h < self.min_height
            ):
                continue
 
            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "regions": [region],
                }
            )
 
        changed = True
 
        while changed:
 
            changed = False
 
            result = []
 
            while boxes:
 
                current = boxes.pop()
 
                merged = False
 
                for i, other in enumerate(
                    boxes
                ):
 
                    if self._should_merge(
                        current,
                        other
                    ):
 
                        combined = (
                            self._combine(
                                current,
                                other
                            )
                        )
 
                        boxes.pop(i)
 
                        boxes.append(
                            combined
                        )
 
                        merged = True
                        changed = True
 
                        break
 
                if not merged:
 
                    result.append(
                        current
                    )
 
            boxes = result
 
        return [
            self._restore_region(
                item
            )
            for item in boxes
        ]
 
    # ========================================================
    # BOX
    # ========================================================
 
    def _get_box(
        self,
        region
    ):
 
        # x / y / width / height
        if all(
            hasattr(
                region,
                name
            )
            for name in (
                "x",
                "y",
                "width",
                "height",
            )
        ):
 
            return (
                int(region.x),
                int(region.y),
                int(region.width),
                int(region.height),
            )
 
        # bbox
        bbox = getattr(
            region,
            "bbox",
            None
        )
 
        if bbox is not None:
 
            try:
 
                x1, y1, x2, y2 = bbox
 
                return (
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                )
 
            except Exception:
                pass
 
        return None
 
    # ========================================================
    # MERGE CONDITION
    # ========================================================
 
    def _should_merge(
        self,
        a,
        b
    ) -> bool:
 
        ax1 = a["x"]
        ay1 = a["y"]
 
        ax2 = (
            a["x"]
            +
            a["w"]
        )
 
        ay2 = (
            a["y"]
            +
            a["h"]
        )
 
        bx1 = b["x"]
        by1 = b["y"]
 
        bx2 = (
            b["x"]
            +
            b["w"]
        )
 
        by2 = (
            b["y"]
            +
            b["h"]
        )
 
        # gap을 포함한 확장 영역
        ax1 -= self.gap
        ay1 -= self.gap
        ax2 += self.gap
        ay2 += self.gap
 
        return not (
            ax2 < bx1
            or
            bx2 < ax1
            or
            ay2 < by1
            or
            by2 < ay1
        )
 
    # ========================================================
    # COMBINE
    # ========================================================
 
    def _combine(
        self,
        a,
        b
    ):
 
        x1 = min(
            a["x"],
            b["x"]
        )
 
        y1 = min(
            a["y"],
            b["y"]
        )
 
        x2 = max(
            a["x"] + a["w"],
            b["x"] + b["w"]
        )
 
        y2 = max(
            a["y"] + a["h"],
            b["y"] + b["h"]
        )
 
        return {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "regions": (
                a["regions"]
                +
                b["regions"]
            ),
        }
 
    # ========================================================
    # RESTORE
    # ========================================================
 
    def _restore_region(
        self,
        item
    ):
 
        regions = item[
            "regions"
        ]
 
        # 하나뿐이면 기존 객체 그대로 반환
        if len(regions) == 1:
 
            return regions[0]
 
        # 여러 영역이 합쳐진 경우
        # 첫 번째 객체를 기준으로
        # bounding box만 확장한다.
        base = regions[0]
 
        try:
 
            base.x = item["x"]
            base.y = item["y"]
            base.width = item["w"]
            base.height = item["h"]
 
        except Exception:
 
            pass
 
        return base