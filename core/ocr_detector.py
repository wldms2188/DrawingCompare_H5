import cv2
import pytesseract
 
 
class OCRDetector:
 
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
 
    def read_text(self, image):
 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
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