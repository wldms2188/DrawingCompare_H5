import cv2
import numpy as np
 
from core.image_loader import PDFImageLoader
from core.auto_align import AutoAlign
 
loader = PDFImageLoader(dpi=300)
aligner = AutoAlign()
 
before_pages = loader.load("input/before/sample_before.pdf")
after_pages = loader.load("input/after/sample_after.pdf")
 
before = cv2.cvtColor(np.array(before_pages[0]), cv2.COLOR_RGB2BGR)
after = cv2.cvtColor(np.array(after_pages[0]), cv2.COLOR_RGB2BGR)
 
aligned = aligner.align(before, after)
 
cv2.imwrite("output/before.png", before)
cv2.imwrite("output/after.png", after)
cv2.imwrite("output/aligned.png", aligned)
 
print("자동 정렬 완료")