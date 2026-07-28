from dataclasses import dataclass
 
 
@dataclass
class ChangeInfo:
 
    id: int
 
    x: int
    y: int
    w: int
    h: int
 
    before_image = None
    after_image = None
 
    before_text: str = ""
    after_text: str = ""
 
    change_type: str = ""
    page: int = 1