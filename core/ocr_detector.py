import cv2
import pytesseract
 
 
class OCRDetector:
 
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
 
    def _preprocess(self, image):
        """OCR 성능 향상을 위한 전처리"""
 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
 
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )
 
        return binary
 
    def _read(self, image):
 
        processed = self._preprocess(image)
 
        text = pytesseract.image_to_string(
            processed,
            lang="eng",
            config="--oem 3 --psm 6"
        )
 
        return text.strip()
 
    def run(self, changes):
 
        for change in changes:
 
            if change.before_image is not None:
                change.before_text = self._read(change.before_image)
 
            if change.after_image is not None:
                change.after_text = self._read(change.after_image)
 
        return changes