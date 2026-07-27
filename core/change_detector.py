import cv2
 
 
class ChangeDetector:
 
    def __init__(self, threshold=30, min_area=100):
        self.threshold = threshold
        self.min_area = min_area
 
    def detect(self, before_img, after_img):
 
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
 
        diff = cv2.absdiff(before_gray, after_gray)
 
        _, binary = cv2.threshold(
            diff,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )
 
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )
 
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        boxes = []
 
        result = after_img.copy()
 
        for contour in contours:
 
            if cv2.contourArea(contour) < self.min_area:
                continue
 
            x, y, w, h = cv2.boundingRect(contour)
 
            boxes.append((x, y, w, h))
 
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )
 
        return result, boxes