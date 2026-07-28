import os
 
import cv2
 
 
class Cropper:
 
    def crop_changes(self, before_img, after_img, boxes, save_dir="output/crops"):
 
        os.makedirs(save_dir, exist_ok=True)
 
        results = []
 
        for idx, (x, y, w, h) in enumerate(boxes):
 
            margin = 20
 
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
 
            x2 = min(before_img.shape[1], x + w + margin)
            y2 = min(before_img.shape[0], y + h + margin)
 
            before_crop = before_img[y1:y2, x1:x2]
            after_crop = after_img[y1:y2, x1:x2]
 
            before_path = os.path.join(
                save_dir,
                f"{idx+1}_before.png"
            )
 
            after_path = os.path.join(
                save_dir,
                f"{idx+1}_after.png"
            )
 
            cv2.imwrite(before_path, before_crop)
            cv2.imwrite(after_path, after_crop)
 
            results.append(
                (
                    before_crop,
                    after_crop,
                    before_path,
                    after_path
                )
            )
 
        return results