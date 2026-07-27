import cv2
import numpy as np
 
from core.image_loader import PDFImageLoader
from core.auto_align import AutoAlign
from core.change_detector import ChangeDetector
 
loader = PDFImageLoader(dpi=300)
aligner = AutoAlign()
detector = ChangeDetector()
 
before_pages = loader.load("input/before/sample_before.pdf")
after_pages = loader.load("input/after/sample_after.pdf")
 
before = cv2.cvtColor(np.array(before_pages[0]), cv2.COLOR_RGB2BGR)
after = cv2.cvtColor(np.array(after_pages[0]), cv2.COLOR_RGB2BGR)
 
aligned = aligner.align(before, after)
 
result, boxes = detector.detect(before, aligned)
 
cv2.imwrite("output/change_result.png", result)
 
print(f"변경 영역 개수 : {len(boxes)}")
print("변경 검출 완료")