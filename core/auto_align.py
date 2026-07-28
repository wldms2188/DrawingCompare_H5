import cv2
import numpy as np
 
 
class AutoAlign:
 
    def align(self, before_img, after_img):
 
        gray1 = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
 
        orb = cv2.ORB_create(5000)
 
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
 
        if des1 is None or des2 is None:
            print("정렬 실패 : 특징점을 찾을 수 없습니다.")
            return after_img
 
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
        matches = matcher.match(des1, des2)
 
        if len(matches) < 10:
            print("정렬 실패 : 매칭 개수가 부족합니다.")
            return after_img
 
        matches = sorted(matches, key=lambda x: x.distance)
 
        src_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
 
        dst_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
 
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )
 
        if H is None:
            print("정렬 실패 : Homography 계산 실패")
            return after_img
 
        aligned = cv2.warpPerspective(
            after_img,
            H,
            (before_img.shape[1], before_img.shape[0])
        )
 
        print("자동 정렬 완료")
 
        return aligned