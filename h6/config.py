from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    dpi: int = 180
    diff_threshold: int = 45
    min_component_area: int = 120
    max_component_fraction: float = 0.015
    max_regions: int = 40
    alignment_min_score: float = 0.30
    max_rotation_deg: float = 5.0
    max_scale_delta: float = 0.12

CONFIG = Config()
