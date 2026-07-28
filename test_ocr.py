import cv2
import numpy as np
 
from core.image_loader import PDFImageLoader
from core.ocr_detector import OCRDetector
 
loader = PDFImageLoader(dpi=300)
ocr = OCRDetector()
 
pages = loader.load("input/before/sample_before.pdf")
 
image = cv2.cvtColor(
    np.array(pages[0]),
    cv2.COLOR_RGB2BGR
)
 
text = ocr.read_text(image)
 
print("========== OCR 결과 ==========")
print(text)