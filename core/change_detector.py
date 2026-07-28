import cv2
 
from core.change_info import ChangeInfo
 
 
class ChangeDetector:
 
    def detect(self, before_img, after_img):
 
        gray1 = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
 
        diff = cv2.absdiff(gray1, gray2)
 
        _, thresh = cv2.threshold(
            diff,
            30,
            255,
            cv2.THRESH_BINARY
        )
 
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )
 
        thresh = cv2.dilate(
            thresh,
            kernel,
            iterations=2
        )
 
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
 
        result = after_img.copy()
        changes = []
 
        change_id = 1
 
        for contour in contours:
 
            area = cv2.contourArea(contour)
 
            if area < 100:
                continue
 
            x, y, w, h = cv2.boundingRect(contour)
 
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )
 
            changes.append(
                ChangeInfo(
                    id=change_id,
                    page=1,
                    x=x,
                    y=y,
                    w=w,
                    h=h
                )
            )
 
            change_id += 1
 
        return result, changes