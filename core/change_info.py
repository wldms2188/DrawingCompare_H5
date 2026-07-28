from dataclasses import dataclass
from typing import Optional
 
import numpy as np
 
 
@dataclass
class ChangeInfo:
    """변경 영역 하나를 저장하는 객체"""
 
    id: int
    page: int
 
    x: int
    y: int
    w: int
    h: int
 
    before_image: Optional[np.ndarray] = None
    after_image: Optional[np.ndarray] = None
 
    before_image_path: str = ""
    after_image_path: str = ""
 
    before_text: str = ""
    after_text: str = ""
 
    change_type: str = ""