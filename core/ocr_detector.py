import cv2
import pytesseract
 
 
class OCRDetector:
 
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
 
    def _read(self, image):
 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        gray = cv2.medianBlur(gray, 3)
 
        _, binary = cv2.threshold(
            gray,
            180,
            255,
            cv2.THRESH_BINARY
        )
 
        text = pytesseract.image_to_string(
            binary,
            lang="eng",
            config="--psm 6"
        )
 
        return text.strip()
 
    def run(self, changes):
 
        for change in changes:
 
            if change.before_image is not None:
                change.before_text = self._read(change.before_image)
 
            if change.after_image is not None:
                change.after_text = self._read(change.after_image)
 
        return changes