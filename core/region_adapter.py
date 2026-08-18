from __future__ import annotations
 
from typing import List, Any
 
from core.region_matcher import DrawingRegion
 
 
class RegionAdapter:
    """
    기존 OCR/TextRegion 결과를
    RegionMatcher가 사용할 수 있는 DrawingRegion으로 변환한다.
 
    기존 OCR 구조를 직접 수정하지 않고 중간에서 연결하기 위한
    Adapter 역할을 한다.
    """
 
    def __init__(self):
        pass
 
    # ========================================================
    # OCR 결과 -> DrawingRegion
    # ========================================================
 
    def convert(
        self,
        regions: List[Any],
    ) -> List[DrawingRegion]:
 
        result = []
 
        for index, region in enumerate(regions):
 
            converted = self._convert_one(
                index,
                region,
            )
 
            if converted is not None:
 
                result.append(
                    converted
                )
 
        return result
 
    # ========================================================
    # Single region
    # ========================================================
 
    def _convert_one(
        self,
        index: int,
        region: Any,
    ):
 
        bbox = self._get_bbox(
            region
        )
 
        if bbox is None:
 
            return None
 
        x, y, width, height = bbox
 
        text = self._get_text(
            region
        )
 
        region_type = self._get_type(
            region
        )
 
        converted = DrawingRegion(
            region_id=index,
            x=x,
            y=y,
            width=width,
            height=height,
            region_type=region_type,
            confidence=self._get_confidence(
                region
            ),
        )
 
        if text:
 
            converted.text_regions = [
                region
            ]
 
        return converted
 
    # ========================================================
    # Bounding box
    # ========================================================
 
    def _get_bbox(
        self,
        region: Any,
    ):
 
        # ------------------------------------
        # x / y / width / height
        # ------------------------------------
 
        if all(
            hasattr(
                region,
                name,
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
 
        # ------------------------------------
        # bbox
        # ------------------------------------
 
        bbox = getattr(
            region,
            "bbox",
            None,
        )
 
        if bbox is not None:
 
            try:
 
                if len(bbox) == 4:
 
                    x1, y1, x2, y2 = bbox
 
                    return (
                        int(x1),
                        int(y1),
                        max(
                            1,
                            int(x2 - x1),
                        ),
                        max(
                            1,
                            int(y2 - y1),
                        ),
                    )
 
            except Exception:
 
                pass
 
        # ------------------------------------
        # left / top / right / bottom
        # ------------------------------------
 
        names = (
            "left",
            "top",
            "right",
            "bottom",
        )
 
        if all(
            hasattr(
                region,
                name,
            )
            for name in names
        ):
 
            return (
                int(region.left),
                int(region.top),
                max(
                    1,
                    int(
                        region.right
                        -
                        region.left
                    ),
                ),
                max(
                    1,
                    int(
                        region.bottom
                        -
                        region.top
                    ),
                ),
            )
 
        return None
 
    # ========================================================
    # Text
    # ========================================================
 
    def _get_text(
        self,
        region: Any,
    ) -> str:
 
        text = getattr(
            region,
            "text",
            "",
        )
 
        if text is None:
 
            return ""
 
        return str(
            text
        ).strip()
 
    # ========================================================
    # Type
    # ========================================================
 
    def _get_type(
        self,
        region: Any,
    ) -> str:
 
        region_type = getattr(
            region,
            "region_type",
            None,
        )
 
        if region_type is None:
 
            region_type = getattr(
                region,
                "type",
                None,
            )
 
        if region_type is None:
 
            region_type = "SECTION"
 
        region_type = str(
            region_type
        ).upper()
 
        # 기존 명칭을 통일
        mapping = {
 
            "TITLE": "TITLE_BLOCK",
 
            "TITLEBLOCK": "TITLE_BLOCK",
 
            "TITLE BLOCK":
                "TITLE_BLOCK",
 
            "DIM":
                "DIMENSION",
 
            "GDT":
                "GDT",
 
            "GD&T":
                "GDT",
 
            "NOTE":
                "COMMENT",
 
            "NOTES":
                "COMMENT",
 
        }
 
        return mapping.get(
            region_type,
            region_type,
        )
 
    # ========================================================
    # Confidence
    # ========================================================
 
    def _get_confidence(
        self,
        region: Any,
    ) -> float:
 
        value = getattr(
            region,
            "confidence",
            1.0,
        )
 
        try:
 
            return float(
                value
            )
 
        except Exception:
 
            return 1.0
 
 
# ============================================================
# Convenience
# ============================================================
 
def convert_regions(
    regions: List[Any],
) -> List[DrawingRegion]:
 
    adapter = RegionAdapter()
 
    return adapter.convert(
        regions
    )