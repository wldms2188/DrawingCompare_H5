import cv2
import numpy as np
 
 
class AutoAlign:
 
    def __init__(self, max_features=5000, keep_percent=0.2):
        self.max_features = max_features
        self.keep_percent = keep_percent
 
    def align(self, before_img, after_img):
 
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
 
        orb = cv2.ORB_create(self.max_features)
 
        kp1, des1 = orb.detectAndCompute(before_gray, None)
        kp2, des2 = orb.detectAndCompute(after_gray, None)
 
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
 
        matches = sorted(matches, key=lambda x: x.distance)
 
        keep = int(len(matches) * self.keep_percent)
        matches = matches[:keep]
 
        pts_before = np.zeros((len(matches), 2), dtype=np.float32)
        pts_after = np.zeros((len(matches), 2), dtype=np.float32)
 
        for i, m in enumerate(matches):
            pts_before[i] = kp1[m.queryIdx].pt
            pts_after[i] = kp2[m.trainIdx].pt
 
        H, _ = cv2.findHomography(
            pts_after,
            pts_before,
            cv2.RANSAC
        )
 
        height, width = before_img.shape[:2]
 
        aligned = cv2.warpPerspective(
            after_img,
            H,
            (width, height)
        )
 
        return aligned